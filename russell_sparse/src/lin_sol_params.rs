use super::{Ordering, Scaling};

/// Chooses the LU factorization used by the Faer solver
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FaerMethod {
    /// Picks from the fill's flop proxy: supernodal above `flop_ratio_threshold`
    ///
    /// The proxy is `Σ col_counts² / nnz(A)` over the supernode column counts of the
    /// analysis, so it costs nothing beyond the symbolic analysis the solver runs anyway.
    Auto {
        /// Threshold on the proxy: supernodal is chosen above it, simplicial at or below
        flop_ratio_threshold: f64,
    },

    /// The simplicial factorization
    Simplicial,

    /// The supernodal factorization
    Supernodal,
}

impl FaerMethod {
    /// Default threshold used when `--faer-method auto` is requested without `--faer-flop-ratio`
    ///
    /// Note that `FaerMethod::default()` is [`FaerMethod::Simplicial`], not `Auto`; its
    /// documentation carries the measured comparison that led to that default.
    pub const DEFAULT_FLOP_RATIO_THRESHOLD: f64 = 1.0;
}

impl Default for FaerMethod {
    /// Defaults to [`FaerMethod::Simplicial`] based on the release sweep (12 matrices).
    ///
    /// On the SuiteSparse matrices of `benchmark-russell` with at most 1.2M pattern entries, the
    /// supernodal path (which omits contribution blocks) was slower on every matrix that completed —
    /// **1.06x** on `bwm2000` to **54x** on `Goodwin_040` — level only on the 157x157
    /// `pmsim-hex8-27`, and it passed the 180s probe on `xenon1`, where simplicial takes 55s. The
    /// synthetic 3D lattices of the sweep showed no win either, and no `flop_ratio` threshold
    /// separated wins from regressions. `Auto { flop_ratio_threshold }` remains available for
    /// callers whose fronts are small and dense.
    fn default() -> Self {
        Self::Simplicial
    }
}

/// Defines the configuration parameters for the linear system solver
///
/// The solvers compare these parameters with `==` to decide whether a refactorization may be
/// given parameters again: the first `preprocess` stores them, and a later call must pass
/// `None` or the same `Some(params)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinSolParams {
    /// Defines the symmetric permutation (ordering)
    pub ordering: Ordering,

    /// Defines the scaling strategy
    pub scaling: Scaling,

    /// Indicates that the coefficient matrix is positive-definite (only considered if the matrix is symmetric)
    pub positive_definite: bool,

    /// Requests that the determinant be computed
    ///
    /// **Note:** The determinant will be available after `preprocess`
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

    /// Defines the number of threads for MUMPS (ICNTL(16))
    ///
    /// **Note:** 0 lets MUMPS take the OpenMP default (`OMP_NUM_THREADS`). That holds for every
    /// BLAS: the MUMPS build is thread-safe by itself (see `architecture-mumps.md`).
    pub mumps_num_threads: usize,

    /// Enforces the unsymmetric strategy, even for symmetric matrices (not recommended; UMFPACK only)
    pub umfpack_enforce_unsymmetric_strategy: bool,

    /// Chooses the LU factorization in the Faer solver (Faer only)
    ///
    /// `Auto` picks from the fill's flop proxy (`Σ col_counts² / nnz(A)`),
    /// while `Simplicial` and `Supernodal` force a single path.
    pub faer_method: FaerMethod,

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
            umfpack_enforce_unsymmetric_strategy: false,
            faer_method: FaerMethod::default(),
            verbose: false,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{FaerMethod, LinSolParams};
    use crate::{Ordering, Scaling};

    #[test]
    #[allow(clippy::clone_on_copy)] // test the derived Clone implementation
    fn clone_copy_and_debug_work() {
        let params = LinSolParams::new();
        let copy = params;
        let clone = params.clone();
        assert!(format!("{:?}", params) == format!("{:?}", copy));
        assert!(format!("{:?}", params) == format!("{:?}", clone));
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
        assert_eq!(params.faer_method, FaerMethod::Simplicial);
        assert_eq!(params.faer_method, FaerMethod::default());
    }
}
