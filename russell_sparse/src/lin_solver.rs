use super::{Genie, Ordering, Scaling, SolverIntelDSS, SolverMUMPS, SolverUMFPACK, SparseMatrix, StatsLinSol};
use crate::StrError;
use russell_lab::Vector;

/// Defines the configuration parameters for the linear system solver
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinSolParams {
    /// Defines the symmetric permutation (ordering)
    pub ordering: Ordering,

    /// Defines the scaling strategy
    pub scaling: Scaling,

    /// Requests that the determinant be computed
    ///
    /// **Note:** The determinant will be available after `factorize`
    pub compute_determinant: bool,

    /// Requests that the error estimates be computed
    ///
    /// **Note:** Will need to use the `actual` solver to access the results.
    pub compute_error_estimates: bool,

    /// Requests that condition numbers be computed
    ///
    /// **Note:** Will need to use the `actual` solver to access the results.
    pub compute_condition_numbers: bool,

    /// Sets the % increase in the estimated working space (MUMPS only)
    ///
    /// **Note:** The default (recommended) value is 100 (%)
    pub mumps_pct_inc_workspace: usize,

    /// Sets the max size of the working memory in mega bytes (MUMPS only)
    ///
    /// **Note:** Set this value to 0 for an automatic configuration
    pub mumps_max_work_memory: usize,

    /// Defines the number of OpenMP threads (MUMPS only)
    ///
    /// **Note:** Set this value to 0 to allow an automatic detection
    pub mumps_openmp_num_threads: usize,

    /// Enforces the unsymmetric strategy, even for symmetric matrices (not recommended; UMFPACK only)
    pub umfpack_enforce_unsymmetric_strategy: bool,

    /// Show additional messages
    pub verbose: bool,
}

impl LinSolParams {
    /// Allocates a new instance with default values
    pub fn new() -> Self {
        LinSolParams {
            ordering: Ordering::Auto,
            scaling: Scaling::Auto,
            compute_determinant: false,
            compute_error_estimates: false,
            compute_condition_numbers: false,
            mumps_pct_inc_workspace: 100,
            mumps_max_work_memory: 0,
            mumps_openmp_num_threads: 0,
            umfpack_enforce_unsymmetric_strategy: false,
            verbose: false,
        }
    }
}

/// Defines a unified interface for linear system solvers
pub trait LinSolTrait {
    /// Performs the factorization (and analysis/initialization if needed)
    ///
    /// # Input
    ///
    /// * `mat` -- The sparse matrix (COO, CSC, or CSR).
    /// * `params` -- configuration parameters; None => use default
    ///
    /// # Notes
    ///
    /// 1. The structure of the matrix (nrow, ncol, nnz, symmetry) must be
    ///    exactly the same among multiple calls to `factorize`. The values may differ
    ///    from call to call, nonetheless.
    /// 2. The first call to `factorize` will define the structure which must be
    ///    kept the same for the next calls.
    /// 3. If the structure of the matrix needs to be changed, the solver must
    ///    be "dropped" and a new solver allocated.
    fn factorize(&mut self, mat: &mut SparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError>;

    /// Computes the solution of the linear system
    ///
    /// Solves the linear system:
    ///
    /// ```text
    ///   A   · x = rhs
    /// (m,n)  (n)  (m)
    /// ```
    ///
    /// # Output
    ///
    /// * `x` -- the vector of unknown values with dimension equal to mat.ncol
    ///
    /// # Input
    ///
    /// * `mat` -- the coefficient matrix A.
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to mat.ncol
    /// * `verbose` -- shows messages
    ///
    /// **Warning:** the matrix must be same one used in `factorize`.
    fn solve(&mut self, x: &mut Vector, mat: &SparseMatrix, rhs: &Vector, verbose: bool) -> Result<(), StrError>;

    /// Updates the stats structure (should be called after solve)
    fn update_stats(&self, stats: &mut StatsLinSol);
}

/// Unifies the access to linear system solvers
pub struct LinSolver<'a> {
    /// Holds the actual implementation
    pub actual: Box<dyn LinSolTrait + 'a>,
}

impl<'a> LinSolver<'a> {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `genie` -- the actual implementation that does all the magic
    pub fn new(genie: Genie) -> Result<Self, StrError> {
        let actual: Box<dyn LinSolTrait> = match genie {
            Genie::Mumps => Box::new(SolverMUMPS::new()?),
            Genie::Umfpack => Box::new(SolverUMFPACK::new()?),
            Genie::IntelDss => Box::new(SolverIntelDSS::new()?),
        };
        Ok(LinSolver { actual })
    }

    /// Computes the solution of a linear system
    ///
    /// Solves the linear system:
    ///
    /// ```text
    ///   A   · x = rhs
    /// (m,n)  (n)  (m)
    /// ```
    ///
    /// # Output
    ///
    /// * `x` -- the vector of unknown values with dimension equal to mat.ncol
    ///
    /// # Input
    ///
    /// * `genie` -- the actual implementation that does all the magic
    /// * `mat` -- the matrix representing the sparse coefficient matrix A (see Notes below)
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to coo.nrow
    /// * `verbose` -- shows messages
    ///
    /// # Notes
    ///
    /// 1. For symmetric matrices, `MUMPS` requires that the symmetry/storage be Lower or Full.
    /// 2. For symmetric matrices, `UMFPACK` requires that the symmetry/storage be Full.
    /// 3. For symmetric matrices, `IntelDSS` requires that the symmetry/storage be Upper.
    /// 4. This function calls the actual implementation (genie) via the functions `factorize`, and `solve`.
    /// 5. This function is best for a **single-use**, whereas the actual
    ///    solver should be considered for a recurrent use (e.g., inside a loop).
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{vec_approx_eq, Vector};
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // constants
    ///     let ndim = 3; // number of rows = number of columns
    ///     let nnz = 5; // number of non-zero values
    ///
    ///     // allocate the coefficient matrix
    ///     let mut mat = SparseMatrix::new_coo(ndim, ndim, nnz, None, false)?;
    ///     mat.put(0, 0, 0.2)?;
    ///     mat.put(0, 1, 0.2)?;
    ///     mat.put(1, 0, 0.5)?;
    ///     mat.put(1, 1, -0.25)?;
    ///     mat.put(2, 2, 0.25)?;
    ///
    ///     // print matrix
    ///     let mut a = mat.as_dense();
    ///     let correct = "┌                   ┐\n\
    ///                    │   0.2   0.2     0 │\n\
    ///                    │   0.5 -0.25     0 │\n\
    ///                    │     0     0  0.25 │\n\
    ///                    └                   ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///
    ///     // allocate the right-hand side vector
    ///     let rhs = Vector::from(&[1.0, 1.0, 1.0]);
    ///
    ///     // calculate the solution
    ///     let mut x = Vector::new(ndim);
    ///     LinSolver::compute(Genie::Umfpack, &mut x, &mut mat, &rhs, None)?;
    ///     let correct = vec![3.0, 2.0, 4.0];
    ///     vec_approx_eq(x.as_data(), &correct, 1e-14);
    ///     Ok(())
    /// }
    /// ```
    pub fn compute(
        genie: Genie,
        x: &mut Vector,
        mat: &mut SparseMatrix,
        rhs: &Vector,
        params: Option<LinSolParams>,
    ) -> Result<Self, StrError> {
        let mut solver = LinSolver::new(genie)?;
        solver.actual.factorize(mat, params)?;
        let verbose = if let Some(p) = params { p.verbose } else { false };
        solver.actual.solve(x, mat, rhs, verbose)?;
        Ok(solver)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{LinSolParams, LinSolver};
    use crate::{Genie, Ordering, Samples, Scaling, SparseMatrix};
    use russell_lab::{vec_approx_eq, Vector};

    #[test]
    fn clone_copy_and_debug_work() {
        let params = LinSolParams::new();
        let copy = params;
        let clone = params.clone();
        assert!(format!("{:?}", params).len() > 0);
        assert_eq!(copy, params);
        assert_eq!(clone, params);
    }

    #[test]
    fn lin_sol_params_new_works() {
        let params = LinSolParams::new();
        assert_eq!(params.ordering, Ordering::Auto);
        assert_eq!(params.scaling, Scaling::Auto);
        assert_eq!(params.compute_determinant, false);
        assert_eq!(params.mumps_pct_inc_workspace, 100);
        assert_eq!(params.mumps_max_work_memory, 0);
        assert_eq!(params.mumps_openmp_num_threads, 0);
        assert!(!params.umfpack_enforce_unsymmetric_strategy);
    }

    #[test]
    fn lin_solver_new_works() {
        LinSolver::new(Genie::Mumps).unwrap();
        LinSolver::new(Genie::Umfpack).unwrap();
        if cfg!(with_intel_dss) {
            LinSolver::new(Genie::IntelDss).unwrap();
        }
    }

    #[test]
    fn lin_solver_compute_works() {
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_full(false);
        let mut mat = SparseMatrix::from_coo(coo);
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        LinSolver::compute(Genie::Umfpack, &mut x, &mut mat, &rhs, None).unwrap();
        let x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        vec_approx_eq(x.as_data(), &x_correct, 1e-10);
    }
}
