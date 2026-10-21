use super::{CooMatrix, CscMatrix, LinSolParams, LinSolTrait, StatsLinSol, Sym};
use crate::StrError;
use algebra_linear::sparse::{CscPattern, CscView, LuMethod, LuSolver, LuSymbolic, SparseError, SparseLuStats};
use russell_lab::{Stopwatch, Vector};
use std::time::Duration;

/// Maps an `algebra_linear` sparse error to the crate's string error.
pub(crate) fn native_error(error: SparseError) -> StrError {
    match error {
        SparseError::Singular => "the matrix is singular",
        SparseError::StructureMismatch => "the structure of the matrix changed; allocate a new solver",
        SparseError::NotFinite => "the matrix has a NaN or infinite entry",
        SparseError::BadShape => "the matrix shape is inconsistent",
        SparseError::DuplicateEntry => "the matrix has a duplicate entry",
        SparseError::UnsortedRows => "the matrix rows are not sorted",
        SparseError::InvalidStructure => "the matrix has an invalid CSC structure",
        SparseError::IndexOverflow => "the matrix is too large for the native solver",
        _ => "the native LU factorization failed",
    }
}

/// Builds the statistics of a native factorization.
pub(crate) fn native_stats<T>(factors: &LuSolver<T>, n: usize, nnz_a: usize, initialize_ns: u128, factorize_ns: u128) -> SparseLuStats {
    let (l_col_ptr, u_col_ptr) = match factors.as_simplicial() {
        Some(f) => (f.l_col_ptr(), f.u_col_ptr()),
        None => {
            let f = factors.as_supernodal().expect("one of the two factor types");
            (f.l_col_ptr(), f.u_col_ptr())
        }
    };
    let nnz_l = l_col_ptr[n];
    let nnz_u = u_col_ptr[n];
    let mut flops_estimate = 0usize;
    for j in 0..n {
        let nl = l_col_ptr[j + 1] - l_col_ptr[j];
        let nu = u_col_ptr[j + 1] - u_col_ptr[j];
        flops_estimate = flops_estimate.saturating_add(nl.saturating_mul(nu));
    }
    SparseLuStats {
        n,
        nnz_a,
        nnz_l,
        nnz_u,
        fill_ratio: if nnz_a == 0 { 0.0 } else { (nnz_l + nnz_u) as f64 / nnz_a as f64 },
        flops_estimate,
        ordering_time: Duration::from_nanos(initialize_ns as u64),
        factorization_time: Duration::from_nanos(factorize_ns as u64),
    }
}

/// Solves sparse linear systems with the pure-Rust LU from `algebra_linear`
///
/// This is the `Genie::Native` backend. It is a direct LU factorization with
/// partial pivoting and a COLAMD column ordering; a flop proxy chooses between
/// the simplicial and supernodal variants. It needs no external library.
///
/// # Supported parameters
///
/// * `ordering`: `Auto` and `Colamd` both use COLAMD. Other values are treated
///   as `Auto`; this backend has no natural-ordering option.
/// * `scaling`: no scaling is applied, for either `Auto` or `No`. Other values
///   are treated as `Auto`.
/// * `compute_determinant`: not available; `factorize` returns an error.
/// * `compute_error_estimates` and `compute_condition_numbers`: not available;
///   the corresponding statistics stay zero.
///
/// The fill, flop and timing figures are available from [`Self::stats`].
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_approx_eq, Vector};
/// use russell_sparse::prelude::*;
/// use russell_sparse::StrError;
///
/// fn main() -> Result<(), StrError> {
///     let ndim = 3;
///     let nnz = 5;
///
///     let mut solver = SolverNative::new()?;
///
///     // ┌                   ┐
///     // │   0.2   0.2     0 │
///     // │   0.5 -0.25     0 │
///     // │     0     0  0.25 │
///     // └                   ┘
///     let mut coo = CooMatrix::new(ndim, ndim, nnz, Sym::No)?;
///     coo.put(0, 0, 0.2)?;
///     coo.put(0, 1, 0.2)?;
///     coo.put(1, 0, 0.5)?;
///     coo.put(1, 1, -0.25)?;
///     coo.put(2, 2, 0.25)?;
///
///     solver.factorize(&coo, None)?;
///
///     let rhs = Vector::from(&[1.0, 1.0, 1.0]);
///     let mut x = Vector::new(ndim);
///     solver.solve(&mut x, &rhs, false)?;
///
///     let correct = vec![3.0, 2.0, 4.0];
///     vec_approx_eq(&x, &correct, 1e-14);
///     Ok(())
/// }
/// ```
pub struct SolverNative {
    /// Holds the CSC matrix used in the last factorization
    csc: Option<CscMatrix>,

    /// The symbolic analysis, reused while the structure does not change
    symbolic: Option<LuSymbolic>,

    /// The numeric factors
    factors: Option<LuSolver<f64>>,

    /// Statistics of the last factorization
    stats: Option<SparseLuStats>,

    /// Indicates whether the solver has been initialized or not
    initialized: bool,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Holds the symmetric flag saved at the first factorization
    initialized_sym: Sym,

    /// Holds the matrix dimension saved at the first factorization
    initialized_ndim: usize,

    /// Holds the number of non-zeros saved at the first factorization
    initialized_nnz: usize,

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on the symbolic analysis in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on the numeric factorization in nanoseconds
    time_factorize_ns: u128,

    /// Time spent on the solve in nanoseconds
    time_solve_ns: u128,
}

impl SolverNative {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        Ok(SolverNative {
            csc: None,
            symbolic: None,
            factors: None,
            stats: None,
            initialized: false,
            factorized: false,
            initialized_sym: Sym::No,
            initialized_ndim: 0,
            initialized_nnz: 0,
            stopwatch: Stopwatch::new(),
            time_initialize_ns: 0,
            time_factorize_ns: 0,
            time_solve_ns: 0,
        })
    }

    /// Returns the statistics of the last factorization, if any
    pub fn stats(&self) -> Option<&SparseLuStats> {
        self.stats.as_ref()
    }
}

impl LinSolTrait for SolverNative {
    fn factorize(&mut self, mat: &CooMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        if mat.nrow != mat.ncol {
            return Err("the matrix must be square");
        }
        if mat.nnz < 1 {
            return Err("the COO matrix must have at least one non-zero value");
        }
        if mat.symmetric == Sym::YesLower || mat.symmetric == Sym::YesUpper {
            return Err("Native requires Sym::YesFull for symmetric matrices");
        }
        if let Some(par) = params {
            if par.compute_determinant {
                return Err("determinant evaluation is not available for this Genie");
            }
        }

        // The structure is fixed by the first call; later calls may only change
        // the values.
        if !self.initialized {
            self.initialized_sym = mat.symmetric;
            self.initialized_ndim = mat.nrow;
            self.initialized_nnz = mat.nnz;
            self.stopwatch.reset();
            let csc = CscMatrix::from_coo(mat)?;
            let nnz = csc.get_info().2;
            let col_pointers: Vec<usize> = csc.get_col_pointers()[..=mat.ncol].iter().map(|&p| p as usize).collect();
            let row_indices: Vec<usize> = csc.get_row_indices()[..nnz].iter().map(|&i| i as usize).collect();
            let pattern = CscPattern::new(mat.nrow, mat.ncol, &col_pointers, &row_indices).map_err(native_error)?;
            let symbolic = LuSymbolic::analyze(pattern, LuMethod::auto()).map_err(native_error)?;
            self.time_initialize_ns = self.stopwatch.stop();
            self.csc = Some(csc);
            self.symbolic = Some(symbolic);
            self.initialized = true;
        } else {
            if mat.symmetric != self.initialized_sym {
                return Err("subsequent factorizations must use the same matrix (symmetric differs)");
            }
            if mat.nrow != self.initialized_ndim {
                return Err("subsequent factorizations must use the same matrix (ndim differs)");
            }
            if mat.nnz != self.initialized_nnz {
                return Err("subsequent factorizations must use the same matrix (nnz differs)");
            }
            let csc = CscMatrix::from_coo(mat)?;
            self.csc = Some(csc);
        }

        // Refresh the values (and re-widen the indices if the structure moved;
        // the fingerprint in factorize catches that case).
        let csc = self.csc.as_ref().unwrap();
        let nnz = csc.get_info().2;
        let col_pointers: Vec<usize> = csc.get_col_pointers()[..=mat.ncol].iter().map(|&p| p as usize).collect();
        let row_indices: Vec<usize> = csc.get_row_indices()[..nnz].iter().map(|&i| i as usize).collect();
        let view = CscView::new(mat.nrow, mat.ncol, &col_pointers, &row_indices, &csc.get_values()[..nnz]).map_err(native_error)?;

        self.stopwatch.reset();
        let factors = self.symbolic.as_ref().unwrap().factorize(view).map_err(native_error)?;
        self.time_factorize_ns = self.stopwatch.stop();
        self.stats = Some(native_stats(&factors, mat.nrow, mat.nnz, self.time_initialize_ns, self.time_factorize_ns));
        self.factors = Some(factors);
        self.factorized = true;
        Ok(())
    }

    fn solve(&mut self, x: &mut Vector, rhs: &Vector, _verbose: bool) -> Result<(), StrError> {
        if !self.factorized {
            return Err("the function factorize must be called before solve");
        }
        if x.dim() != self.initialized_ndim {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.initialized_ndim {
            return Err("the dimension of the right-hand side vector is incorrect");
        }
        x.as_mut_data().copy_from_slice(rhs.as_data());
        self.stopwatch.reset();
        self.factors.as_ref().unwrap().solve(x.as_mut_data()).map_err(native_error)?;
        self.time_solve_ns = self.stopwatch.stop();
        Ok(())
    }

    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = "Native".to_string();
        stats.output.effective_ordering = "Colamd".to_string();
        stats.output.effective_scaling = "No".to_string();
        stats.time_nanoseconds.initialize_array.push(self.time_initialize_ns);
        stats.time_nanoseconds.factorize_array.push(self.time_factorize_ns);
        stats.time_nanoseconds.solve_array.push(self.time_solve_ns);
    }

    fn get_ns_init(&self) -> u128 {
        self.time_initialize_ns
    }

    fn get_ns_fact(&self) -> u128 {
        self.time_factorize_ns
    }

    fn get_ns_solve(&self) -> u128 {
        self.time_solve_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, Samples};
    use russell_lab::{approx_eq, vec_approx_eq};

    #[test]
    fn new_and_factorize_errors_work() {
        let mut solver = SolverNative::new().unwrap();
        assert!(!solver.factorized);
        assert!(solver.stats().is_none());

        // empty COO
        let coo = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("the COO matrix must have at least one non-zero value")
        );

        // rectangular
        let (coo, _, _, _) = Samples::rectangular_1x7();
        assert_eq!(solver.factorize(&coo, None).err(), Some("the matrix must be square"));

        // symmetric storage must be full
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_lower(false, false);
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("Native requires Sym::YesFull for symmetric matrices")
        );

        // determinant is not available
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        let mut params = LinSolParams::new();
        params.compute_determinant = true;
        assert_eq!(
            solver.factorize(&coo, Some(params)).err(),
            Some("determinant evaluation is not available for this Genie")
        );

        // structure changes are refused
        solver.factorize(&coo, None).unwrap();
        let mut other = CooMatrix::new(2, 2, 2, Sym::YesFull).unwrap();
        other.put(0, 0, 1.0).unwrap();
        other.put(1, 1, 2.0).unwrap();
        assert_eq!(
            solver.factorize(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetric differs)")
        );
        let mut other = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        other.put(0, 0, 1.0).unwrap();
        assert_eq!(
            solver.factorize(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );
        let mut other = CooMatrix::new(2, 2, 1, Sym::No).unwrap();
        other.put(0, 0, 1.0).unwrap();
        assert_eq!(
            solver.factorize(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn factorize_works_and_reuses_the_analysis() {
        let mut solver = SolverNative::new().unwrap();
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        solver.factorize(&coo, None).unwrap();
        assert!(solver.factorized);
        let stats = solver.stats().unwrap();
        assert_eq!(stats.n, 5);
        assert!(stats.nnz_l >= 5);
        assert!(stats.flops_estimate > 0);

        // Second call with the same values works and keeps the analysis.
        let init = solver.get_ns_init();
        solver.factorize(&coo, None).unwrap();
        assert_eq!(solver.get_ns_init(), init);

        // Changing only the values is allowed.
        let mut other = CooMatrix::new(5, 5, coo.nnz, Sym::No).unwrap();
        for k in 0..coo.nnz {
            let i = coo.indices_i[k] as usize;
            let j = coo.indices_j[k] as usize;
            let value = if i == j { 2.0 } else { coo.values[k] };
            other.put(i, j, value).unwrap();
        }
        solver.factorize(&other, None).unwrap();
    }

    #[test]
    fn factorize_fails_on_singular_matrix() {
        let mut solver = SolverNative::new().unwrap();
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 0.0).unwrap();
        assert_eq!(solver.factorize(&coo, None), Err("the matrix is singular"));
    }

    #[test]
    fn solve_handles_errors() {
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        coo.put(1, 1, 456.0).unwrap();
        let mut solver = SolverNative::new().unwrap();
        let mut x = Vector::new(2);
        let rhs = Vector::new(2);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the function factorize must be called before solve")
        );
        let mut x = Vector::new(1);
        solver.factorize(&coo, None).unwrap();
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
    fn solve_works_unsymmetric_and_symmetric() {
        let mut solver = SolverNative::new().unwrap();
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = Vector::new(5);
        solver.factorize(&coo, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-13);

        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.main.solver, "Native");
        assert_eq!(stats.output.effective_ordering, "Colamd");
        assert_eq!(stats.output.effective_scaling, "No");
        assert_eq!(stats.time_nanoseconds.initialize_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.factorize_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.solve_array.len(), 1);
        assert!(solver.get_ns_init() > 0);
        assert!(solver.get_ns_fact() > 0);
        assert!(solver.get_ns_solve() > 0);

        // Symmetric full matrix.
        let mut solver = SolverNative::new().unwrap();
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_full();
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let x_correct = &[-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        let mut x = Vector::new(5);
        solver.factorize(&coo, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-10);

        // stats carry the fill numbers
        let stats = solver.stats().unwrap();
        approx_eq(stats.fill_ratio, (stats.nnz_l + stats.nnz_u) as f64 / coo.nnz as f64, 1e-14);
    }
}
