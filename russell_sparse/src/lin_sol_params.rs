use super::{Ordering, Scaling};

/// Defines the configuration parameters for the linear system solver
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinSolParams {
    /// Defines the symmetric permutation (ordering)
    pub ordering: Ordering,

    /// Defines the scaling strategy
    pub scaling: Scaling,

    /// Indicates that the coefficient matrix is positive-definite (only considered if the matrix is symmetric)
    pub positive_definite: bool,

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

    /// Defines the number of threads for MUMPS
    ///
    /// **Note:** Set this value to 0 to allow an automatic detection
    pub mumps_num_threads: usize,

    /// Overrides the prevention of number-of-threads issue with OpenBLAS (not recommended)
    pub mumps_override_prevent_nt_issue_with_openblas: bool,

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
            positive_definite: false,
            compute_determinant: false,
            compute_error_estimates: false,
            compute_condition_numbers: false,
            mumps_pct_inc_workspace: 100,
            mumps_max_work_memory: 0,
            mumps_num_threads: 0,
            mumps_override_prevent_nt_issue_with_openblas: false,
            umfpack_enforce_unsymmetric_strategy: false,
            verbose: false,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::LinSolParams;
    use crate::{Ordering, Scaling};

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
        assert_eq!(params.mumps_num_threads, 0);
        assert!(!params.umfpack_enforce_unsymmetric_strategy);
    }
}
