use super::{
    AmgCoarsening, AmgSmoother, CooMatrix, CsrMatrix, IterativeMethod, IterativeParams, LinSolParams, LinSolTrait, Preconditioner, StatsLinSol, Sym,
};
use crate::StrError;
use algebra_linear::sparse::amgcl::{
    AggregationParams, Amg, AmgParams, ChebyshevParams, Coarsening, CoarseningParams, Error, Ilu0Params, KrylovMethod, Parallelism,
    SmoothedAggregationParams, Smoother, SolveReport, SolverParams, amg_setup, krylov_solve,
};
use russell_lab::{Stopwatch, Vector};

/// Maps an `algebra_linear` AMGCL error to the crate's string error
///
/// The [`Error::SolverBreakdown`] message depends on the Krylov method, which is
/// why the method is an argument.
pub(crate) fn iterative_error(method: KrylovMethod, error: Error) -> StrError {
    match error {
        Error::InvalidMatrix(_) => "the iterative solver rejected the matrix",
        Error::InvalidParameter(_) => "the iterative solver rejected a parameter",
        Error::DimensionMismatch { .. } => "the iterative solver received incompatible dimensions",
        Error::SingularMatrix { .. } => "the AMG coarse-level matrix is singular",
        Error::SolverBreakdown(_) => match method {
            KrylovMethod::Cg => "CG failed: the matrix may not be symmetric positive definite",
            KrylovMethod::BiCgStab | KrylovMethod::Gmres => "the iterative solver broke down",
        },
        Error::NonFinite => "the iterative solver encountered a non-finite value",
    }
}

/// Solves sparse linear systems with the pure-Rust Krylov and AMG solvers from `algebra_linear`
///
/// This is the backend for [`crate::Genie::Iterative`]. It needs no external
/// library and supports the [`crate::IterativeMethod::Cg`],
/// [`crate::IterativeMethod::BiCgStab`], and [`crate::IterativeMethod::Gmres`]
/// methods with either algebraic multigrid or identity preconditioning.
///
/// # Rebuilding on every preprocess
///
/// Unlike the direct solvers, this solver accepts a new matrix structure on
/// every call to `preprocess`: it rebuilds the AMG hierarchy from scratch. The
/// "same structure" rule documented on [LinSolTrait::preprocess] is a
/// direct-solver rule and does not apply here.
///
/// # Supported parameters
///
/// * [`IterativeParams`] controls the method parameters, the preconditioner, and
///   the AMG coarsening, smoother, and hierarchy size.
/// * `compute_determinant` is not available; `preprocess` returns an error.
/// * The ordering, scaling, and other direct-solver options are ignored.
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_approx_eq, Vector};
/// use russell_sparse::prelude::*;
/// use russell_sparse::StrError;
///
/// fn main() -> Result<(), StrError> {
///     // ┌             ┐
///     // │  4 -1    0  │
///     // │ -1  4   -1  │
///     // │  0 -1    4  │
///     // └             ┘
///     let ndim = 3;
///     let mut coo = CooMatrix::new(ndim, ndim, 7, Sym::YesFull)?;
///     coo.put(0, 0, 4.0)?;
///     coo.put(0, 1, -1.0)?;
///     coo.put(1, 0, -1.0)?;
///     coo.put(1, 1, 4.0)?;
///     coo.put(1, 2, -1.0)?;
///     coo.put(2, 1, -1.0)?;
///     coo.put(2, 2, 4.0)?;
///
///     let mut solver = SolverIterative::new(IterativeMethod::Cg)?;
///     solver.preprocess(&coo, None)?;
///
///     let rhs = Vector::from(&[3.0, 2.0, 3.0]);
///     let mut x = Vector::new(ndim);
///     solver.solve(&mut x, &rhs, false)?;
///
///     let correct = vec![1.0, 1.0, 1.0];
///     vec_approx_eq(&x, &correct, 1e-10);
///     Ok(())
/// }
/// ```
pub struct SolverIterative {
    /// Holds the AMG state (None before the first preprocess)
    amg: Option<Amg>,

    /// Holds the Krylov method selected at construction
    method: KrylovMethod,

    /// Holds the SolverParams of the last preprocess
    params: SolverParams,

    /// Indicates whether the sparse matrix has been preprocessed or not
    preprocessed: bool,

    /// Holds the report of the last solve (internal copy; the public statistics block is Phase 3)
    #[allow(dead_code)]
    last_report: Option<SolveReport>,

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on initialize in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on preprocess in nanoseconds
    time_preprocess_ns: u128,

    /// Time spent on solve in nanoseconds
    time_solve_ns: u128,
}

impl SolverIterative {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `method` -- the Krylov method
    pub fn new(method: IterativeMethod) -> Result<Self, StrError> {
        let method = match method {
            IterativeMethod::Cg => KrylovMethod::Cg,
            IterativeMethod::BiCgStab => KrylovMethod::BiCgStab,
            IterativeMethod::Gmres => KrylovMethod::Gmres,
        };
        Ok(SolverIterative {
            amg: None,
            method,
            params: SolverParams::default(),
            preprocessed: false,
            last_report: None,
            stopwatch: Stopwatch::new(),
            time_initialize_ns: 0,
            time_preprocess_ns: 0,
            time_solve_ns: 0,
        })
    }

    /// Returns the number of rows of the preprocessed matrix
    fn ndim(&self) -> usize {
        self.amg
            .as_ref()
            .unwrap()
            .levels()
            .next()
            .expect("the AMG hierarchy has at least one level")
            .unknowns
    }
}

/// Validates the iterative parameters
///
/// The restart is used by GMRES alone, so it is checked only for that method.
fn validate_params(method: KrylovMethod, params: &IterativeParams) -> Result<(), StrError> {
    if !params.tolerance.is_finite() || params.tolerance < 0.0 {
        return Err("the iterative tolerance must be finite and nonnegative");
    }
    if !params.absolute_tolerance.is_finite() || params.absolute_tolerance < 0.0 {
        return Err("the iterative absolute tolerance must be finite and nonnegative");
    }
    if params.max_iterations == 0 {
        return Err("the maximum number of iterations must be positive");
    }
    if method == KrylovMethod::Gmres && params.gmres_restart == 0 {
        return Err("the GMRES restart must be positive");
    }
    if params.num_threads == 0 {
        return Err("the number of threads must be positive");
    }
    if params.min_parallel_len == 0 {
        return Err("the parallel length threshold must be positive");
    }
    if params.amg_coarse_enough == 0 {
        return Err("the AMG coarse-enough size must be positive");
    }
    if params.amg_max_levels == 0 {
        return Err("the AMG maximum number of levels must be positive");
    }
    Ok(())
}

/// Maps the crate's coarsening selection to the AMGCL one
fn coarsening(params: AmgCoarsening) -> Coarsening {
    match params {
        AmgCoarsening::RugeStuben => Coarsening::RugeStuben(CoarseningParams::default()),
        AmgCoarsening::Aggregation => Coarsening::Aggregation(AggregationParams::default()),
        AmgCoarsening::SmoothedAggregation => Coarsening::SmoothedAggregation(SmoothedAggregationParams::default()),
    }
}

/// Maps the crate's smoother selection to the AMGCL one
fn smoother(params: AmgSmoother) -> Smoother {
    match params {
        AmgSmoother::GaussSeidel => Smoother::GaussSeidel,
        AmgSmoother::DampedJacobi => Smoother::DampedJacobi(0.72),
        AmgSmoother::Ilu0 => Smoother::Ilu0(Ilu0Params::default()),
        AmgSmoother::Chebyshev => Smoother::Chebyshev(ChebyshevParams::default()),
        AmgSmoother::Spai0 => Smoother::Spai0,
        AmgSmoother::Spai1 => Smoother::Spai1,
    }
}

impl LinSolTrait for SolverIterative {
    /// Performs the preprocessing (builds the AMG hierarchy)
    ///
    /// # Input
    ///
    /// * `mat` -- the sparse matrix (COO)
    /// * `params` -- configuration parameters; None => use default
    ///
    /// # Notes
    ///
    /// 1. Every call rebuilds the AMG state, so the matrix structure may change
    ///    between calls. The "same structure" rule of the direct solvers does
    ///    not apply here.
    /// 2. For symmetric matrices, this solver requires [Sym::YesFull].
    /// 3. The determinant is not available for this solver.
    fn preprocess(&mut self, mat: &CooMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        if mat.nrow != mat.ncol {
            return Err("the matrix must be square");
        }
        if mat.nnz < 1 {
            return Err("the COO matrix must have at least one non-zero value");
        }
        if mat.symmetric == Sym::YesLower || mat.symmetric == Sym::YesUpper {
            return Err("Iterative requires Sym::YesFull for symmetric matrices");
        }

        // Read the parameters of this call (a new preprocess may change them).
        let params = match params {
            Some(params) => params,
            None => LinSolParams::new(),
        };
        if params.compute_determinant {
            return Err("determinant evaluation is not available for this Genie");
        }
        validate_params(self.method, &params.iterative)?;

        let iterative = params.iterative;
        let solver_params = SolverParams {
            max_iterations: iterative.max_iterations,
            tolerance: iterative.tolerance,
            absolute_tolerance: iterative.absolute_tolerance,
            gmres_restart: iterative.gmres_restart,
            parallelism: Parallelism {
                threads: iterative.num_threads,
                min_parallel_len: iterative.min_parallel_len,
            },
        };
        let setup_params = match iterative.preconditioner {
            Preconditioner::Identity => None,
            Preconditioner::Amg => Some(AmgParams {
                coarsening: coarsening(iterative.amg_coarsening),
                smoother: smoother(iterative.amg_smoother),
                coarse_enough: iterative.amg_coarse_enough,
                max_levels: iterative.amg_max_levels,
                parallelism: Parallelism {
                    threads: iterative.num_threads,
                    min_parallel_len: iterative.min_parallel_len,
                },
                ..AmgParams::default()
            }),
        };

        // Convert to the CSR arrays required by the AMG setup.
        self.stopwatch.reset();
        let csr = CsrMatrix::from_coo(mat)?;
        let row_ptr: Vec<usize> = csr.get_row_pointers().iter().map(|&p| p as usize).collect();
        let col_idx: Vec<usize> = csr.get_col_indices().iter().map(|&i| i as usize).collect();
        self.time_initialize_ns = self.stopwatch.stop();

        // Build the AMG hierarchy (or the identity preconditioner).
        self.stopwatch.reset();
        let amg = amg_setup(&row_ptr, &col_idx, csr.get_values(), setup_params).map_err(|e| iterative_error(self.method, e))?;
        self.time_preprocess_ns = self.stopwatch.stop();

        self.amg = Some(amg);
        self.params = solver_params;
        self.preprocessed = true;
        self.last_report = None;
        Ok(())
    }

    fn solve(&mut self, x: &mut Vector, rhs: &Vector, _verbose: bool) -> Result<(), StrError> {
        if !self.preprocessed {
            return Err("the function preprocess must be called before solve");
        }
        if x.dim() != self.ndim() {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.ndim() {
            return Err("the dimension of the right-hand side vector is incorrect");
        }
        x.as_mut_data().fill(0.0);
        self.stopwatch.reset();
        let report = krylov_solve(self.method, self.amg.as_mut().unwrap(), rhs.as_data(), x.as_mut_data(), self.params)
            .map_err(|e| iterative_error(self.method, e))?;
        self.time_solve_ns = self.stopwatch.stop();
        self.last_report = Some(report);
        if !report.converged {
            return Err("the iterative solver did not converge");
        }
        Ok(())
    }

    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = match self.method {
            KrylovMethod::Cg => "CG".to_string(),
            KrylovMethod::BiCgStab => "BiCGStab".to_string(),
            KrylovMethod::Gmres => "GMRES".to_string(),
        };
        stats.output.effective_ordering = "None".to_string();
        stats.output.effective_scaling = "None".to_string();
        stats.time_nanoseconds.initialize_array.push(self.time_initialize_ns);
        stats.time_nanoseconds.preprocess_array.push(self.time_preprocess_ns);
        stats.time_nanoseconds.solve_array.push(self.time_solve_ns);
    }

    fn get_ns_init(&self) -> u128 {
        self.time_initialize_ns
    }

    fn get_ns_preprocess(&self) -> u128 {
        self.time_preprocess_ns
    }

    fn get_ns_solve(&self) -> u128 {
        self.time_solve_ns
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComplexLinSolver, Genie, LinSolver, Samples};
    use russell_lab::vec_approx_eq;

    /// Builds the 1-D Laplacian A = tridiag(-1, 2, -1) with n unknowns
    fn laplacian_1d(n: usize) -> CooMatrix {
        let mut coo = CooMatrix::new(n, n, 3 * n - 2, Sym::YesFull).unwrap();
        for i in 0..n {
            coo.put(i, i, 2.0).unwrap();
            if i > 0 {
                coo.put(i, i - 1, -1.0).unwrap();
            }
            if i + 1 < n {
                coo.put(i, i + 1, -1.0).unwrap();
            }
        }
        coo
    }

    /// Builds a nonsymmetric, diagonally dominant 5x5 system
    fn nonsymmetric_5x5() -> CooMatrix {
        let mut coo = CooMatrix::new(5, 5, 14, Sym::No).unwrap();
        let a = [
            [5.0, -1.0, 0.0, 0.0, 0.0],
            [-1.0, 6.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 7.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 8.0, -1.0],
            [2.0, 0.0, 0.0, -1.0, 9.0],
        ];
        for i in 0..5 {
            for j in 0..5 {
                if a[i][j] != 0.0 {
                    coo.put(i, j, a[i][j]).unwrap();
                }
            }
        }
        coo
    }

    /// Solves the system with the Direct solver
    fn direct_solution(coo: &CooMatrix, rhs: &Vector) -> Vector {
        let mut x = Vector::new(rhs.dim());
        LinSolver::compute(Genie::Direct, &mut x, coo, rhs, None).unwrap();
        x
    }

    #[test]
    fn new_and_preprocess_reject_invalid_inputs() {
        let mut solver = SolverIterative::new(IterativeMethod::Cg).unwrap();
        assert!(!solver.preprocessed);
        assert!(solver.last_report.is_none());

        // empty COO
        let coo = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        assert_eq!(
            solver.preprocess(&coo, None).err(),
            Some("the COO matrix must have at least one non-zero value")
        );

        // rectangular
        let (coo, _, _, _) = Samples::rectangular_1x7();
        assert_eq!(solver.preprocess(&coo, None).err(), Some("the matrix must be square"));

        // symmetric storage must be full
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_lower(false, false);
        assert_eq!(
            solver.preprocess(&coo, None).err(),
            Some("Iterative requires Sym::YesFull for symmetric matrices")
        );

        // determinant is not available
        let coo = laplacian_1d(3);
        let mut params = LinSolParams::new();
        params.compute_determinant = true;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("determinant evaluation is not available for this Genie")
        );
    }

    #[test]
    fn preprocess_checks_every_parameter() {
        let coo = laplacian_1d(3);
        let mut solver = SolverIterative::new(IterativeMethod::Cg).unwrap();

        let mut params = LinSolParams::new();
        params.iterative.tolerance = -1.0;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("the iterative tolerance must be finite and nonnegative")
        );
        params.iterative.tolerance = f64::NAN;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("the iterative tolerance must be finite and nonnegative")
        );

        let mut params = LinSolParams::new();
        params.iterative.absolute_tolerance = -1.0;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("the iterative absolute tolerance must be finite and nonnegative")
        );

        let mut params = LinSolParams::new();
        params.iterative.max_iterations = 0;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("the maximum number of iterations must be positive")
        );

        let mut params = LinSolParams::new();
        params.iterative.gmres_restart = 0;
        let mut gmres = SolverIterative::new(IterativeMethod::Gmres).unwrap();
        assert_eq!(gmres.preprocess(&coo, Some(params)).err(), Some("the GMRES restart must be positive"));

        // CG does not use the GMRES restart, so zero is accepted
        let mut params = LinSolParams::new();
        params.iterative.gmres_restart = 0;
        solver.preprocess(&coo, Some(params)).unwrap();

        let mut params = LinSolParams::new();
        params.iterative.num_threads = 0;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("the number of threads must be positive")
        );

        let mut params = LinSolParams::new();
        params.iterative.min_parallel_len = 0;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("the parallel length threshold must be positive")
        );

        let mut params = LinSolParams::new();
        params.iterative.amg_coarse_enough = 0;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("the AMG coarse-enough size must be positive")
        );

        let mut params = LinSolParams::new();
        params.iterative.amg_max_levels = 0;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("the AMG maximum number of levels must be positive")
        );
    }

    #[test]
    fn solve_checks_the_state_and_dimensions() {
        let coo = laplacian_1d(2);
        let mut solver = SolverIterative::new(IterativeMethod::Cg).unwrap();
        let mut x = Vector::new(2);
        let rhs = Vector::new(2);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the function preprocess must be called before solve")
        );

        solver.preprocess(&coo, None).unwrap();
        let mut x = Vector::new(1);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        let mut x = Vector::new(2);
        let rhs = Vector::new(1);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );
    }

    #[test]
    fn cg_with_amg_matches_the_direct_solution() {
        let n = 100;
        let coo = laplacian_1d(n);
        let rhs = Vector::filled(n, 1.0);
        let x_direct = direct_solution(&coo, &rhs);

        let mut solver = SolverIterative::new(IterativeMethod::Cg).unwrap();
        solver.preprocess(&coo, None).unwrap();
        let mut x = Vector::new(n);
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_direct.as_data(), 1e-8);

        let report = solver.last_report.unwrap();
        assert!(report.converged);
        assert!(report.iterations > 0);
        assert!(report.relative_residual <= 1e-10);

        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.main.solver, "CG");
        assert_eq!(stats.output.effective_ordering, "None");
        assert_eq!(stats.output.effective_scaling, "None");
        assert_eq!(stats.time_nanoseconds.initialize_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.preprocess_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.solve_array.len(), 1);
        assert!(solver.get_ns_init() > 0);
        assert!(solver.get_ns_preprocess() > 0);
        assert!(solver.get_ns_solve() > 0);
    }

    #[test]
    fn bicgstab_and_gmres_match_the_direct_solution() {
        let coo = nonsymmetric_5x5();
        let rhs = Vector::from(&[4.0, 4.0, 5.0, 6.0, 10.0]);
        let x_direct = direct_solution(&coo, &rhs);

        for method in [IterativeMethod::BiCgStab, IterativeMethod::Gmres] {
            let mut solver = SolverIterative::new(method).unwrap();
            solver.preprocess(&coo, None).unwrap();
            let mut x = Vector::new(5);
            solver.solve(&mut x, &rhs, false).unwrap();
            vec_approx_eq(&x, x_direct.as_data(), 1e-8);
            assert!(solver.last_report.unwrap().converged);

            let mut stats = StatsLinSol::new();
            solver.update_stats(&mut stats);
            let label = match method {
                IterativeMethod::Cg => "CG",
                IterativeMethod::BiCgStab => "BiCGStab",
                IterativeMethod::Gmres => "GMRES",
            };
            assert_eq!(stats.main.solver, label);
        }
    }

    #[test]
    fn identity_preconditioner_needs_more_iterations_than_amg() {
        let n = 100;
        let coo = laplacian_1d(n);
        let rhs = Vector::filled(n, 1.0);
        let x_direct = direct_solution(&coo, &rhs);

        let mut solver_amg = SolverIterative::new(IterativeMethod::Cg).unwrap();
        solver_amg.preprocess(&coo, None).unwrap();
        let mut x = Vector::new(n);
        solver_amg.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_direct.as_data(), 1e-8);
        let iterations_amg = solver_amg.last_report.unwrap().iterations;

        let mut params = LinSolParams::new();
        params.iterative.preconditioner = Preconditioner::Identity;
        let mut solver_identity = SolverIterative::new(IterativeMethod::Cg).unwrap();
        solver_identity.preprocess(&coo, Some(params)).unwrap();
        let mut x = Vector::new(n);
        solver_identity.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_direct.as_data(), 1e-8);
        let iterations_identity = solver_identity.last_report.unwrap().iterations;

        assert!(iterations_identity > iterations_amg);
    }

    #[test]
    fn a_low_iteration_limit_reports_nonconvergence() {
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        let mut params = LinSolParams::new();
        params.iterative.preconditioner = Preconditioner::Identity;
        params.iterative.max_iterations = 1;
        let mut solver = SolverIterative::new(IterativeMethod::BiCgStab).unwrap();
        solver.preprocess(&coo, Some(params)).unwrap();
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let mut x = Vector::new(5);
        assert_eq!(solver.solve(&mut x, &rhs, false), Err("the iterative solver did not converge"));
        assert!(!solver.last_report.unwrap().converged);
    }

    #[test]
    fn preprocess_rebuilds_for_new_values_and_structure() {
        let mut solver = SolverIterative::new(IterativeMethod::Cg).unwrap();

        // first matrix
        let coo = laplacian_1d(6);
        let rhs = Vector::filled(6, 1.0);
        let x_direct = direct_solution(&coo, &rhs);
        solver.preprocess(&coo, None).unwrap();
        let mut x = Vector::new(6);
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_direct.as_data(), 1e-8);

        // same structure, new values
        let mut other = CooMatrix::new(6, 6, coo.nnz, Sym::YesFull).unwrap();
        for k in 0..coo.nnz {
            let i = coo.indices_i[k] as usize;
            let j = coo.indices_j[k] as usize;
            let value = if i == j { 3.0 } else { coo.values[k] };
            other.put(i, j, value).unwrap();
        }
        let x_direct = direct_solution(&other, &rhs);
        solver.preprocess(&other, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_direct.as_data(), 1e-8);

        // different structure (dimension and pattern)
        let coo = laplacian_1d(9);
        let rhs = Vector::filled(9, 1.0);
        let x_direct = direct_solution(&coo, &rhs);
        solver.preprocess(&coo, None).unwrap();
        let mut x = Vector::new(9);
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_direct.as_data(), 1e-8);
    }

    #[test]
    fn iterative_genie_dispatches_through_linsolver() {
        let coo = laplacian_1d(10);
        let rhs = Vector::filled(10, 1.0);
        let x_direct = direct_solution(&coo, &rhs);
        for method in [IterativeMethod::Cg, IterativeMethod::BiCgStab, IterativeMethod::Gmres] {
            let mut x = Vector::new(10);
            LinSolver::compute(Genie::Iterative(method), &mut x, &coo, &rhs, None).unwrap();
            vec_approx_eq(&x, x_direct.as_data(), 1e-8);
        }
    }

    #[test]
    fn complex_solvers_reject_iterative_methods() {
        assert_eq!(
            ComplexLinSolver::new(Genie::Iterative(IterativeMethod::Cg)).err(),
            Some("the iterative solvers do not support complex matrices")
        );
    }

    #[test]
    fn iterative_error_maps_every_variant() {
        let message = "the iterative solver rejected the matrix";
        assert_eq!(iterative_error(KrylovMethod::Cg, Error::InvalidMatrix("bad".to_string())), message);
        assert_eq!(
            iterative_error(KrylovMethod::Cg, Error::InvalidParameter("bad".to_string())),
            "the iterative solver rejected a parameter"
        );
        assert_eq!(
            iterative_error(
                KrylovMethod::Cg,
                Error::DimensionMismatch {
                    expected: 2,
                    actual: 1,
                    context: "test"
                }
            ),
            "the iterative solver received incompatible dimensions"
        );
        assert_eq!(
            iterative_error(KrylovMethod::Cg, Error::SingularMatrix { row: 0 }),
            "the AMG coarse-level matrix is singular"
        );
        assert_eq!(
            iterative_error(KrylovMethod::Cg, Error::SolverBreakdown("breakdown")),
            "CG failed: the matrix may not be symmetric positive definite"
        );
        assert_eq!(
            iterative_error(KrylovMethod::BiCgStab, Error::SolverBreakdown("breakdown")),
            "the iterative solver broke down"
        );
        assert_eq!(
            iterative_error(KrylovMethod::Gmres, Error::SolverBreakdown("breakdown")),
            "the iterative solver broke down"
        );
        assert_eq!(
            iterative_error(KrylovMethod::Cg, Error::NonFinite),
            "the iterative solver encountered a non-finite value"
        );
    }

    #[test]
    fn every_coarsening_and_smoother_mapping_preprocesses() {
        let coo = laplacian_1d(20);
        let coarsenings = [AmgCoarsening::RugeStuben, AmgCoarsening::Aggregation, AmgCoarsening::SmoothedAggregation];
        let smoothers = [
            AmgSmoother::GaussSeidel,
            AmgSmoother::DampedJacobi,
            AmgSmoother::Ilu0,
            AmgSmoother::Chebyshev,
            AmgSmoother::Spai0,
            AmgSmoother::Spai1,
        ];
        for amg_coarsening in coarsenings {
            for amg_smoother in smoothers {
                let mut params = LinSolParams::new();
                params.iterative.amg_coarsening = amg_coarsening;
                params.iterative.amg_smoother = amg_smoother;
                let mut solver = SolverIterative::new(IterativeMethod::Cg).unwrap();
                solver.preprocess(&coo, Some(params)).unwrap();
            }
        }
    }

    #[test]
    fn timings_start_at_zero() {
        let mut solver = SolverIterative::new(IterativeMethod::Cg).unwrap();
        assert_eq!(solver.get_ns_init(), 0);
        assert_eq!(solver.get_ns_preprocess(), 0);
        assert_eq!(solver.get_ns_solve(), 0);
        let coo = laplacian_1d(3);
        solver.preprocess(&coo, None).unwrap();
        assert!(solver.get_ns_init() > 0);
        assert!(solver.get_ns_preprocess() > 0);
    }
}
