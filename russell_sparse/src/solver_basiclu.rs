use super::{CooMatrix, CscMatrix, LinSolParams, LinSolTrait, StatsLinSol, Sym};
use crate::StrError;
use algebra_linear::sparse::basiclu::{BasicLu, BasicLuStats, Error};
use russell_lab::{Stopwatch, Vector};

/// Maps an `algebra_linear` BasicLU error to the crate's string error.
pub(crate) fn basiclu_error(error: Error) -> StrError {
    match error {
        Error::BadShape => "the matrix shape is inconsistent",
        Error::InvalidStructure => "the matrix has an invalid CSC structure",
        Error::UnsortedRows => "the matrix rows are not sorted",
        Error::DuplicateEntry => "the matrix has a duplicate entry",
        Error::NotFinite => "the matrix has a NaN or infinite entry",
        Error::Singular { .. } => "the matrix is singular",
        Error::SingularUpdate => "the column update produced a singular pivot",
        Error::MaxUpdatesReached => "the maximum number of updates was reached; refactorize",
        Error::InvalidColumn { .. } => "the column index in the update is out of bounds",
        Error::NotFactorized => "the matrix has not been factorized",
    }
}

/// Solves sparse linear systems with the pure-Rust BasicLU port from `algebra_linear`
///
/// This is the `Genie::BasicLu` backend. It is a square LU factorization with
/// column-oriented Markowitz threshold pivoting, written for LP bases: column and
/// row singletons are peeled from the CSC matrix before the remaining bump is
/// factorized. It needs no external library.
///
/// # Supported parameters
///
/// * `basiclu`: forwarded to `algebra_linear`'s `BasicLuParams`, which carries the
///   pivot tolerances, the drop tolerance, and `allow_singular`.
/// * `ordering`: BasicLU searches pivots with its own Markowitz rule, so values
///   other than the reported `Markowitz` are ignored.
/// * `scaling`: no scaling is applied, so values other than the reported `No` are
///   ignored.
/// * `compute_determinant`: not available; `preprocess` returns an error.
/// * `compute_error_estimates` and `compute_condition_numbers`: not available;
///   the corresponding statistics stay zero.
///
/// # Input class
///
/// Square general systems, stored as `Sym::No` or `Sym::YesFull`. This is an
/// LP-basis LU, not a general-purpose replacement for [`SolverFaer`](crate::SolverFaer):
/// a rank-deficient matrix fails with the singular error unless
/// `BasicLuParams::allow_singular` is set.
///
/// BasicLU has no separate symbolic phase. The first `preprocess` converts the COO
/// matrix to CSC and factorizes it, so all of that time is preprocessing and
/// [`Self::get_ns_init`] stays zero. A later `preprocess` re-factorizes the new
/// values from scratch instead of verifying a pattern fingerprint; the stored
/// structure checks (symmetric flag, dimension, and non-zero count) are what protect
/// the refactorization contract.
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
///     let mut solver = SolverBasicLu::new()?;
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
pub struct SolverBasicLu {
    /// Holds the CSC matrix used in the last preprocess
    csc: Option<CscMatrix>,

    /// The opaque LU handle
    lu: Option<BasicLu>,

    /// Indicates whether the solver has been initialized or not
    initialized: bool,

    /// Parameters used by the first `preprocess`, which a refactorization may repeat
    initialized_params: LinSolParams,

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

    /// Time spent on the symbolic analysis in nanoseconds (always zero for BasicLU)
    time_initialize_ns: u128,

    /// Time spent on the numeric factorization in nanoseconds
    time_preprocess_ns: u128,

    /// Time spent on the solve in nanoseconds
    time_solve_ns: u128,
}

impl SolverBasicLu {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        Ok(SolverBasicLu {
            csc: None,
            lu: None,
            initialized: false,
            initialized_params: LinSolParams::new(),
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
    pub fn stats(&self) -> Option<&BasicLuStats> {
        self.lu.as_ref().and_then(|lu| lu.stats())
    }
}

impl LinSolTrait for SolverBasicLu {
    fn preprocess(&mut self, mat: &CooMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        if self.initialized {
            if mat.symmetric != self.initialized_sym {
                return Err("subsequent factorizations must use the same matrix (symmetric differs)");
            }
            if mat.nrow != self.initialized_ndim {
                return Err("subsequent factorizations must use the same matrix (ndim differs)");
            }
            if mat.nnz != self.initialized_nnz {
                return Err("subsequent factorizations must use the same matrix (nnz differs)");
            }
            if let Some(p) = params {
                if p != self.initialized_params {
                    return Err("subsequent factorizations must not change LinSolParams");
                }
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
                .map_err(basiclu_error)?;
            self.time_preprocess_ns = self.stopwatch.stop();
            self.csc = Some(csc);
        } else {
            if mat.nrow != mat.ncol {
                return Err("the matrix must be square");
            }
            if mat.nnz < 1 {
                return Err("the COO matrix must have at least one non-zero value");
            }
            if mat.symmetric == Sym::YesLower || mat.symmetric == Sym::YesUpper {
                return Err("BasicLU requires Sym::YesFull for symmetric matrices");
            }
            let effective_params = params.unwrap_or_else(LinSolParams::new);
            if effective_params.compute_determinant {
                return Err("determinant evaluation is not available for this Genie");
            }
            self.initialized_params = effective_params;
            self.initialized_sym = mat.symmetric;
            self.initialized_ndim = mat.nrow;
            self.initialized_nnz = mat.nnz;
            self.stopwatch.reset();
            let csc = CscMatrix::from_coo(mat)?;
            let nnz = csc.get_info().2;
            let col_pointers: Vec<usize> = csc.get_col_pointers()[..=mat.ncol].iter().map(|&p| p as usize).collect();
            let row_indices: Vec<usize> = csc.get_row_indices()[..nnz].iter().map(|&i| i as usize).collect();
            let mut lu = BasicLu::new(effective_params.basiclu);
            lu.factorize(&col_pointers, &row_indices, &csc.get_values()[..nnz])
                .map_err(basiclu_error)?;
            self.time_preprocess_ns = self.stopwatch.stop();
            self.csc = Some(csc);
            self.lu = Some(lu);
            self.initialized = true;
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
        self.lu.as_mut().unwrap().solve(x.as_mut_data()).map_err(basiclu_error)?;
        self.time_solve_ns = self.stopwatch.stop();
        Ok(())
    }

    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = "BasicLU".to_string();
        stats.output.effective_ordering = "Markowitz".to_string();
        stats.output.effective_scaling = "No".to_string();
        stats.output.basiclu_rank = self.stats().map(|s| s.rank).unwrap_or(0);
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
    use crate::{CooMatrix, Samples};
    use russell_lab::vec_approx_eq;

    #[test]
    fn solve_works_on_known_4x4() {
        // ┌                ┐
        // │  4  1  0  0    │
        // │  1  5  2  0    │
        // │  0  2  6  3    │
        // │  7  0  3  8    │
        // └                ┘
        let mut coo = CooMatrix::new(4, 4, 11, Sym::No).unwrap();
        coo.put(0, 0, 4.0).unwrap();
        coo.put(0, 1, 1.0).unwrap();
        coo.put(1, 0, 1.0).unwrap();
        coo.put(1, 1, 5.0).unwrap();
        coo.put(1, 2, 2.0).unwrap();
        coo.put(2, 1, 2.0).unwrap();
        coo.put(2, 2, 6.0).unwrap();
        coo.put(2, 3, 3.0).unwrap();
        coo.put(3, 0, 7.0).unwrap();
        coo.put(3, 2, 3.0).unwrap();
        coo.put(3, 3, 8.0).unwrap();

        let x_correct = &[1.0, 2.0, 3.0, 4.0];
        let rhs = Vector::from(&[6.0, 17.0, 34.0, 48.0]);

        let mut solver = SolverBasicLu::new().unwrap();
        solver.preprocess(&coo, None).unwrap();
        assert_eq!(solver.stats().unwrap().rank, 4);

        let mut x = Vector::new(4);
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-13);

        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.main.solver, "BasicLU");
        assert_eq!(stats.output.effective_ordering, "Markowitz");
        assert_eq!(stats.output.effective_scaling, "No");
        assert_eq!(stats.output.basiclu_rank, 4);
        assert_eq!(stats.time_nanoseconds.initialize_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.preprocess_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.solve_array.len(), 1);
        assert_eq!(solver.get_ns_init(), 0);
        assert!(solver.get_ns_preprocess() > 0);
        assert!(solver.get_ns_solve() > 0);
    }

    #[test]
    fn preprocess_errors_work() {
        let mut solver = SolverBasicLu::new().unwrap();

        // rectangular
        let (coo, _, _, _) = Samples::rectangular_1x7();
        assert_eq!(solver.preprocess(&coo, None).err(), Some("the matrix must be square"));

        // triangular storage is rejected
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_lower(false, false);
        assert_eq!(
            solver.preprocess(&coo, None).err(),
            Some("BasicLU requires Sym::YesFull for symmetric matrices")
        );
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_upper(false, false);
        assert_eq!(
            solver.preprocess(&coo, None).err(),
            Some("BasicLU requires Sym::YesFull for symmetric matrices")
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

        // singular matrix
        let mut singular = CooMatrix::new(2, 2, 4, Sym::No).unwrap();
        singular.put(0, 0, 1.0).unwrap();
        singular.put(0, 1, 1.0).unwrap();
        singular.put(1, 0, 1.0).unwrap();
        singular.put(1, 1, 1.0).unwrap();
        assert_eq!(solver.preprocess(&singular, None).err(), Some("the matrix is singular"));

        // non-finite value
        let mut nan = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        nan.put(0, 0, 1.0).unwrap();
        nan.put(1, 1, f64::NAN).unwrap();
        assert_eq!(solver.preprocess(&nan, None).err(), Some("the matrix has a NaN or infinite entry"));

        // allow_singular lets the same matrix through with the achieved rank
        let mut params = LinSolParams::new();
        params.basiclu.allow_singular = true;
        solver.preprocess(&singular, Some(params)).unwrap();
        assert_eq!(solver.stats().unwrap().rank, 1);
    }

    #[test]
    fn refactorization_contract_works() {
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        let first_params = LinSolParams::new();
        let mut solver = SolverBasicLu::new().unwrap();
        solver.preprocess(&coo, Some(first_params)).unwrap();

        // None and identical params are both accepted
        solver.preprocess(&coo, None).unwrap();
        solver.preprocess(&coo, Some(first_params)).unwrap();
        assert_eq!(solver.get_ns_init(), 0);

        // a changed BasicLU parameter is rejected
        let mut changed = first_params;
        changed.basiclu.rel_pivot_tol = 0.5;
        assert_eq!(
            solver.preprocess(&coo, Some(changed)),
            Err("subsequent factorizations must not change LinSolParams")
        );

        // symmetric mismatch
        let (other, _, _, _) = Samples::mkl_symmetric_5x5_full();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetric differs)")
        );

        // dimension mismatch
        let mut other = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        other.put(0, 0, 1.0).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );

        // non-zero count mismatch
        let mut other = CooMatrix::new(5, 5, 5, Sym::No).unwrap();
        for i in 0..5 {
            other.put(i, i, 1.0).unwrap();
        }
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn solve_handles_errors() {
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        coo.put(1, 1, 456.0).unwrap();
        let mut solver = SolverBasicLu::new().unwrap();
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
    fn basiclu_error_maps_every_variant() {
        assert_eq!(basiclu_error(Error::BadShape), "the matrix shape is inconsistent");
        assert_eq!(basiclu_error(Error::InvalidStructure), "the matrix has an invalid CSC structure");
        assert_eq!(basiclu_error(Error::UnsortedRows), "the matrix rows are not sorted");
        assert_eq!(basiclu_error(Error::DuplicateEntry), "the matrix has a duplicate entry");
        assert_eq!(basiclu_error(Error::NotFinite), "the matrix has a NaN or infinite entry");
        assert_eq!(basiclu_error(Error::Singular { rank: 1, n: 2 }), "the matrix is singular");
        assert_eq!(basiclu_error(Error::SingularUpdate), "the column update produced a singular pivot");
        assert_eq!(
            basiclu_error(Error::MaxUpdatesReached),
            "the maximum number of updates was reached; refactorize"
        );
        assert_eq!(
            basiclu_error(Error::InvalidColumn { col: 7, n: 2 }),
            "the column index in the update is out of bounds"
        );
        assert_eq!(basiclu_error(Error::NotFactorized), "the matrix has not been factorized");
    }
}
