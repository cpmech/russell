use super::{Genie, Ordering, Scaling, SolverIntelDSS, SolverMUMPS, SolverUMFPACK, SparseMatrix};
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
    /// MUMPS: computes the backward errors omega1 and omega2 (page 14):
    ///
    /// ```text
    ///                                       |b - A · x_bar|ᵢ
    /// omega1 = largest_scaled_residual_of ————————————————————
    ///                                     (|b| + |A| |x_bar|)ᵢ
    ///
    ///                                            |b - A · x_bar|ᵢ
    /// omega2 = largest_scaled_residual_of ——————————————————————————————————
    ///                                     (|A| |x_approx|)ᵢ + ‖Aᵢ‖∞ ‖x_bar‖∞
    ///
    /// where x_bar is the actual (approximate) solution returned by the linear solver
    /// ```
    pub compute_error_estimates: bool,

    /// Estimates the reciprocal condition number (rcond)
    ///
    /// * `cond` -- is the condition number
    /// * `rcond` -- is the reciprocal condition number (estimate), `rcond ~= 1/cond`
    ///
    /// ```text
    /// cond = norm(A) · norm(inverse(A))
    /// ```
    ///
    /// ```text
    ///                      1
    /// rcond ~= ——————————————————————————
    ///          norm(A) · norm(inverse(A))
    ///
    /// Reference:
    /// Arioli M, Demmel JW, and Duff IS (1989) Solving sparse linear systems with
    /// sparse backward error, SIAM J. Matrix Analysis Applied, 10(2):165-190
    /// ```
    ///
    /// UMFPACK computes a rough estimate of the reciprocal condition number:
    ///
    /// ```text
    /// rcond = min (abs (diag (U))) / max (abs (diag (U)))
    /// ```
    ///
    /// Note: the reciprocal condition number will be zero if the diagonal of U is all zero (UMFPACK).
    ///
    /// Matlab: The reciprocal condition number is a scale-invariant measure
    /// of how close a given matrix is to the set of singular matrices.
    ///
    /// * If rcond ~ 0.0, the matrix is nearly singular and badly conditioned.
    /// * If rcond ~ 1.0, the matrix is well conditioned.
    pub compute_condition_number_estimate: bool,

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
            compute_condition_number_estimate: false,
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

    /// Returns the error estimates (if requested)
    ///
    /// MUMPS: computes the backward errors omega1 and omega2 (page 14):
    ///
    /// ```text
    ///                                       |b - A · x_bar|ᵢ
    /// omega1 = largest_scaled_residual_of ————————————————————
    ///                                     (|b| + |A| |x_bar|)ᵢ
    ///
    ///                                            |b - A · x_bar|ᵢ
    /// omega2 = largest_scaled_residual_of ——————————————————————————————————
    ///                                     (|A| |x_approx|)ᵢ + ‖Aᵢ‖∞ ‖x_bar‖∞
    ///
    /// where x_bar is the actual (approximate) solution returned by the linear solver
    /// ```
    fn get_error_estimates(&self) -> (f64, f64);

    /// Returns the reciprocal condition number estimate (if requested)
    ///
    /// * `cond` -- is the condition number
    /// * `rcond` -- is the reciprocal condition number (estimate), `rcond ~= 1/cond`
    ///
    /// ```text
    /// cond = norm(A) · norm(inverse(A))
    /// ```
    ///
    /// ```text
    ///                      1
    /// rcond ~= ——————————————————————————
    ///          norm(A) · norm(inverse(A))
    ///
    /// Reference:
    /// Arioli M, Demmel JW, and Duff IS (1989) Solving sparse linear systems with
    /// sparse backward error, SIAM J. Matrix Analysis Applied, 10(2):165-190
    /// ```
    ///
    /// UMFPACK computes a rough estimate of the reciprocal condition number:
    ///
    /// ```text
    /// rcond = min (abs (diag (U))) / max (abs (diag (U)))
    /// ```
    ///
    /// Note: the reciprocal condition number will be zero if the diagonal of U is all zero (UMFPACK).
    ///
    /// Matlab: The reciprocal condition number is a scale-invariant measure
    /// of how close a given matrix is to the set of singular matrices.
    ///
    /// * If rcond ~ 0.0, the matrix is nearly singular and badly conditioned.
    /// * If rcond ~ 1.0, the matrix is well conditioned.
    fn get_reciprocal_condition_number_estimate(&self) -> f64;

    /// Returns the determinant
    ///
    /// Returns the three values `(mantissa, base, exponent)`, such that the determinant is calculated by:
    ///
    /// ```text
    /// determinant = mantissa · pow(base, exponent)
    /// ```
    ///
    /// **Note:** This is only available if compute_determinant was requested.
    fn get_determinant(&self) -> (f64, f64, f64);

    /// Returns the ordering effectively used by the solver
    fn get_effective_ordering(&self) -> String;

    /// Returns the scaling effectively used by the solver
    fn get_effective_scaling(&self) -> String;

    /// Returns the strategy (concerning symmetry) effectively used by the solver
    ///
    /// For example, returns whether the `symmetric strategy` was used or not (UMFPACK only)
    fn get_effective_strategy(&self) -> String;

    /// Returns the name of the underlying solver (Genie)
    fn get_name(&self) -> String;
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
    use russell_chk::vec_approx_eq;
    use russell_lab::Vector;

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
