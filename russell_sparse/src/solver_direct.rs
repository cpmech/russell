use super::{CooMatrix, CscMatrix, DirectMethod, LinSolParams, LinSolTrait, StatsLinSol, Sym};
use crate::StrError;
use algebra_linear::sparse::{LuMethod, SparseError, SparseLu, SparseLuStats};
use russell_lab::{Stopwatch, Vector};

/// Maps an `algebra_linear` sparse error to the crate's string error.
pub(crate) fn direct_error(error: SparseError) -> StrError {
    match error {
        SparseError::BadShape => "the matrix shape is inconsistent",
        SparseError::InvalidStructure => "the matrix has an invalid CSC structure",
        SparseError::UnsortedRows => "the matrix rows are not sorted",
        SparseError::DuplicateEntry => "the matrix has a duplicate entry",
        SparseError::Singular => "the matrix is singular",
        SparseError::IndexOverflow => "the matrix is too large for the direct solver",
        SparseError::StructureMismatch => "the structure of the matrix changed; allocate a new solver",
        SparseError::NotFinite => "the matrix has a NaN or infinite entry",
        SparseError::NotFactorized => "the matrix has not been factorized",
    }
}

/// Solves sparse linear systems with the pure-Rust LU from `algebra_linear`
///
/// This is the `Genie::Direct` backend. It is a direct LU factorization with
/// partial pivoting and a COLAMD column ordering; a flop proxy chooses between
/// the simplicial and supernodal variants. It needs no external library.
///
/// # Supported parameters
///
/// * `ordering`: `Auto` and `Colamd` both use COLAMD. Other values are treated
///   as `Auto`; this backend has no natural-ordering option.
/// * `scaling`: no scaling is applied, for either `Auto` or `No`. Other values
///   are treated as `Auto`.
/// * `direct_method`: `Auto` (the default) picks simplicial or supernodal from the fill's flop
///   proxy, `Simplicial` and `Supernodal` force one of them;
///   the simplicial path is the default.
/// * `compute_determinant`: not available; `preprocess` returns an error.
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
///     let mut solver = SolverDirect::new()?;
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
///     solver.preprocess(&coo, None)?;
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
pub struct SolverDirect {
    /// Holds the CSC matrix used in the last preprocess
    csc: Option<CscMatrix>,

    /// The opaque LU handle
    lu: Option<SparseLu<f64>>,

    /// Indicates whether the solver has been initialized or not
    initialized: bool,

    /// Indicates whether the sparse matrix has been preprocessed or not
    preprocessed: bool,

    /// Holds the symmetric flag saved at the first preprocess
    initialized_sym: Sym,

    /// Holds the matrix dimension saved at the first preprocess
    initialized_ndim: usize,

    /// Holds the number of non-zeros saved at the first preprocess
    initialized_nnz: usize,

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on the symbolic analysis in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on the numeric factorization in nanoseconds
    time_preprocess_ns: u128,

    /// Time spent on the solve in nanoseconds
    time_solve_ns: u128,
}

impl SolverDirect {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        Ok(SolverDirect {
            csc: None,
            lu: None,
            initialized: false,
            preprocessed: false,
            initialized_sym: Sym::No,
            initialized_ndim: 0,
            initialized_nnz: 0,
            stopwatch: Stopwatch::new(),
            time_initialize_ns: 0,
            time_preprocess_ns: 0,
            time_solve_ns: 0,
        })
    }

    /// Returns the statistics of the last preprocess, if any
    pub fn stats(&self) -> Option<&SparseLuStats> {
        self.lu.as_ref().and_then(|lu| lu.stats())
    }
}

impl LinSolTrait for SolverDirect {
    fn preprocess(&mut self, mat: &CooMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        if mat.nrow != mat.ncol {
            return Err("the matrix must be square");
        }
        if mat.nnz < 1 {
            return Err("the COO matrix must have at least one non-zero value");
        }
        if mat.symmetric == Sym::YesLower || mat.symmetric == Sym::YesUpper {
            return Err("Direct requires Sym::YesFull for symmetric matrices");
        }
        if let Some(par) = params {
            if par.compute_determinant {
                return Err("determinant evaluation is not available for this Genie");
            }
        }

        // Pick the factorization before the first analysis; later calls reuse it.
        let method = match params {
            Some(par) => match par.direct_method {
                DirectMethod::Auto { flop_ratio_threshold } => LuMethod::Auto { flop_ratio_threshold },
                DirectMethod::Simplicial => LuMethod::Simplicial,
                DirectMethod::Supernodal => LuMethod::Supernodal,
            },
            None => LuMethod::Auto {
                flop_ratio_threshold: DirectMethod::DEFAULT_FLOP_RATIO_THRESHOLD,
            },
        };

        if !self.initialized {
            self.initialized_sym = mat.symmetric;
            self.initialized_ndim = mat.nrow;
            self.initialized_nnz = mat.nnz;
            self.stopwatch.reset();
            let csc = CscMatrix::from_coo(mat)?;
            let nnz = csc.get_info().2;
            let col_pointers: Vec<usize> = csc.get_col_pointers()[..=mat.ncol].iter().map(|&p| p as usize).collect();
            let row_indices: Vec<usize> = csc.get_row_indices()[..nnz].iter().map(|&i| i as usize).collect();
            let mut lu = SparseLu::analyze(&col_pointers, &row_indices, method).map_err(direct_error)?;
            self.time_initialize_ns = self.stopwatch.stop();

            self.stopwatch.reset();
            lu.factorize(&col_pointers, &row_indices, &csc.get_values()[..nnz])
                .map_err(direct_error)?;
            self.time_preprocess_ns = self.stopwatch.stop();

            self.csc = Some(csc);
            self.lu = Some(lu);
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
            self.stopwatch.reset();
            let csc = CscMatrix::from_coo(mat)?;
            let nnz = csc.get_info().2;
            let col_pointers: Vec<usize> = csc.get_col_pointers()[..=mat.ncol].iter().map(|&p| p as usize).collect();
            let row_indices: Vec<usize> = csc.get_row_indices()[..nnz].iter().map(|&i| i as usize).collect();
            self.lu
                .as_mut()
                .unwrap()
                .factorize(&col_pointers, &row_indices, &csc.get_values()[..nnz])
                .map_err(direct_error)?;
            self.time_preprocess_ns = self.stopwatch.stop();
            self.csc = Some(csc);
        }

        self.preprocessed = true;
        Ok(())
    }

    fn solve(&mut self, x: &mut Vector, rhs: &Vector, _verbose: bool) -> Result<(), StrError> {
        if !self.preprocessed {
            return Err("the function preprocess must be called before solve");
        }
        if x.dim() != self.initialized_ndim {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.initialized_ndim {
            return Err("the dimension of the right-hand side vector is incorrect");
        }
        x.as_mut_data().copy_from_slice(rhs.as_data());
        self.stopwatch.reset();
        self.lu.as_ref().unwrap().solve(x.as_mut_data()).map_err(direct_error)?;
        self.time_solve_ns = self.stopwatch.stop();
        Ok(())
    }

    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = "Direct".to_string();
        stats.output.effective_ordering = "Colamd".to_string();
        stats.output.effective_scaling = "No".to_string();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, Samples};
    use russell_lab::{approx_eq, vec_approx_eq};

    #[test]
    fn new_and_preprocess_errors_work() {
        let mut solver = SolverDirect::new().unwrap();
        assert!(!solver.preprocessed);
        assert!(solver.stats().is_none());

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
            Some("Direct requires Sym::YesFull for symmetric matrices")
        );

        // determinant is not available
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        let mut params = LinSolParams::new();
        params.compute_determinant = true;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("determinant evaluation is not available for this Genie")
        );

        // structure changes are refused
        solver.preprocess(&coo, None).unwrap();
        let mut other = CooMatrix::new(2, 2, 2, Sym::YesFull).unwrap();
        other.put(0, 0, 1.0).unwrap();
        other.put(1, 1, 2.0).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetric differs)")
        );
        let mut other = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        other.put(0, 0, 1.0).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );
        let mut other = CooMatrix::new(2, 2, 1, Sym::No).unwrap();
        other.put(0, 0, 1.0).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn preprocess_works_and_reuses_the_analysis() {
        let mut solver = SolverDirect::new().unwrap();
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        solver.preprocess(&coo, None).unwrap();
        assert!(solver.preprocessed);
        let stats = solver.stats().unwrap();
        assert_eq!(stats.n, 5);
        assert!(stats.nnz_l >= 5);
        assert!(stats.flops_estimate > 0);

        // Second call with the same values works and keeps the analysis.
        let init = solver.get_ns_init();
        solver.preprocess(&coo, None).unwrap();
        assert_eq!(solver.get_ns_init(), init);

        // Changing only the values is allowed.
        let mut other = CooMatrix::new(5, 5, coo.nnz, Sym::No).unwrap();
        for k in 0..coo.nnz {
            let i = coo.indices_i[k] as usize;
            let j = coo.indices_j[k] as usize;
            let value = if i == j { 2.0 } else { coo.values[k] };
            other.put(i, j, value).unwrap();
        }
        solver.preprocess(&other, None).unwrap();
    }

    #[test]
    fn preprocess_and_solve_work_supernodal() {
        let mut coo = CooMatrix::new(2, 2, 4, Sym::No).unwrap();
        coo.put(0, 0, 4.0).unwrap();
        coo.put(0, 1, 1.0).unwrap();
        coo.put(1, 0, 1.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();

        let params = LinSolParams {
            direct_method: DirectMethod::Supernodal,
            ..LinSolParams::new()
        };
        let mut solver = SolverDirect::new().unwrap();
        solver.preprocess(&coo, Some(params)).unwrap();

        // [4 1; 1 3] x = [1; 2]  =>  x = [1/11; 7/11]
        let rhs = Vector::from(&[1.0, 2.0]);
        let x_correct = &[1.0 / 11.0, 7.0 / 11.0];
        let mut x = Vector::new(2);
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-14);
    }

    #[test]
    fn preprocess_detects_a_reordered_pattern() {
        // Both matrices have the same sym, ndim and nnz, but the first stores
        // (0,0) (0,1) (1,1) and the second (0,1) (1,0) (1,1).
        let mut a = CooMatrix::new(2, 2, 3, Sym::No).unwrap();
        a.put(0, 0, 1.0).unwrap();
        a.put(0, 1, 2.0).unwrap();
        a.put(1, 1, 3.0).unwrap();
        let mut b = CooMatrix::new(2, 2, 3, Sym::No).unwrap();
        b.put(0, 1, 2.0).unwrap();
        b.put(1, 0, 1.0).unwrap();
        b.put(1, 1, 3.0).unwrap();

        let mut solver = SolverDirect::new().unwrap();
        solver.preprocess(&a, None).unwrap();
        assert_eq!(
            solver.preprocess(&b, None),
            Err("the structure of the matrix changed; allocate a new solver")
        );
    }

    #[test]
    fn preprocess_fails_on_singular_matrix() {
        let mut solver = SolverDirect::new().unwrap();
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 0.0).unwrap();
        assert_eq!(solver.preprocess(&coo, None), Err("the matrix is singular"));
    }

    #[test]
    fn solve_handles_errors() {
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        coo.put(1, 1, 456.0).unwrap();
        let mut solver = SolverDirect::new().unwrap();
        let mut x = Vector::new(2);
        let rhs = Vector::new(2);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the function preprocess must be called before solve")
        );
        let mut x = Vector::new(1);
        solver.preprocess(&coo, None).unwrap();
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
        let mut solver = SolverDirect::new().unwrap();
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = Vector::new(5);
        solver.preprocess(&coo, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-13);

        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.main.solver, "Direct");
        assert_eq!(stats.output.effective_ordering, "Colamd");
        assert_eq!(stats.output.effective_scaling, "No");
        assert_eq!(stats.time_nanoseconds.initialize_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.preprocess_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.solve_array.len(), 1);
        assert!(solver.get_ns_init() > 0);
        assert!(solver.get_ns_preprocess() > 0);
        assert!(solver.get_ns_solve() > 0);

        // Symmetric full matrix.
        let mut solver = SolverDirect::new().unwrap();
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_full();
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let x_correct = &[-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        let mut x = Vector::new(5);
        solver.preprocess(&coo, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-10);

        // stats carry the fill numbers
        let stats = solver.stats().unwrap();
        approx_eq(stats.fill_ratio, (stats.nnz_l + stats.nnz_u) as f64 / coo.nnz as f64, 1e-14);
    }

    #[test]
    fn direct_error_maps_every_variant() {
        assert_eq!(direct_error(SparseError::BadShape), "the matrix shape is inconsistent");
        assert_eq!(direct_error(SparseError::InvalidStructure), "the matrix has an invalid CSC structure");
        assert_eq!(direct_error(SparseError::UnsortedRows), "the matrix rows are not sorted");
        assert_eq!(direct_error(SparseError::DuplicateEntry), "the matrix has a duplicate entry");
        assert_eq!(direct_error(SparseError::Singular), "the matrix is singular");
        assert_eq!(direct_error(SparseError::IndexOverflow), "the matrix is too large for the direct solver");
        assert_eq!(
            direct_error(SparseError::StructureMismatch),
            "the structure of the matrix changed; allocate a new solver"
        );
        assert_eq!(direct_error(SparseError::NotFinite), "the matrix has a NaN or infinite entry");
        assert_eq!(direct_error(SparseError::NotFactorized), "the matrix has not been factorized");
    }
}
