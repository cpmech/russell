use super::{Ordering, Scaling};

/// Selects the preconditioner used by the iterative solvers
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Preconditioner {
    /// Algebraic multigrid
    Amg,

    /// Identity (no preconditioning)
    Identity,
}

/// Selects the AMG coarsening strategy
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AmgCoarsening {
    /// Classical Ruge-Stüben C/F splitting with direct interpolation
    RugeStuben,

    /// Piecewise-constant tentative prolongation with scaled Galerkin products
    Aggregation,

    /// Strongly filtered, damped-Jacobi-smoothed tentative prolongation
    SmoothedAggregation,
}

/// Selects the AMG smoother
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AmgSmoother {
    /// Forward pre-sweeps and backward post-sweeps of Gauss-Seidel
    GaussSeidel,

    /// Damped Jacobi with the AMGCL default damping factor
    ///
    /// This maps to `Smoother::DampedJacobi(0.72)`.
    DampedJacobi,

    /// Incomplete LU with zero fill-in
    Ilu0,

    /// Chebyshev polynomial relaxation
    Chebyshev,

    /// Diagonal sparse approximate inverse
    Spai0,

    /// Sparse approximate inverse with the sparsity pattern of the system matrix
    Spai1,
}

/// Defines the configuration parameters for the iterative solvers
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IterativeParams {
    /// Selects the preconditioner
    pub preconditioner: Preconditioner,

    /// Relative residual target: `‖b-Ax‖₂ ≤ tol * ‖b‖₂`
    pub tolerance: f64,

    /// Absolute target; the solver stops at the larger of this and the relative target
    pub absolute_tolerance: f64,

    /// Maximum number of Krylov iterations
    pub max_iterations: usize,

    /// Maximum Krylov dimension between restarts (GMRES only)
    pub gmres_restart: usize,

    /// Number of worker threads, including the calling thread
    pub num_threads: usize,

    /// Minimum vector length or matrix row count before threads are created
    pub min_parallel_len: usize,

    /// AMG coarsening strategy
    pub amg_coarsening: AmgCoarsening,

    /// AMG smoother
    pub amg_smoother: AmgSmoother,

    /// Directly solve AMG levels with at most this many rows
    pub amg_coarse_enough: usize,

    /// Maximum number of AMG levels, including the finest and coarsest
    pub amg_max_levels: usize,
}

impl Default for IterativeParams {
    fn default() -> Self {
        IterativeParams {
            preconditioner: Preconditioner::Amg,
            tolerance: 1e-10,
            absolute_tolerance: 0.0,
            max_iterations: 1000,
            gmres_restart: 30,
            num_threads: 1,
            min_parallel_len: 1024,
            amg_coarsening: AmgCoarsening::RugeStuben,
            amg_smoother: AmgSmoother::GaussSeidel,
            amg_coarse_enough: 64,
            amg_max_levels: 32,
        }
    }
}

/// Defines the configuration parameters for the linear system solver
#[derive(Clone, Copy, Debug)]
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

    /// Defines the number of threads for MUMPS
    ///
    /// **Note:** Set this value to 0 to allow an automatic detection
    pub mumps_num_threads: usize,

    /// Overrides the prevention of number-of-threads issue with OpenBLAS (not recommended)
    pub mumps_override_prevent_nt_issue_with_openblas: bool,

    /// Enforces the unsymmetric strategy, even for symmetric matrices (not recommended; UMFPACK only)
    pub umfpack_enforce_unsymmetric_strategy: bool,

    /// Selects the supernodal LU in the Direct solver (Direct only)
    ///
    /// The simplicial path is the default. The supernodal path is still
    /// experimental: it uses the relaxed partition and compressed fronts, but
    /// it has no BLAS-3 kernels and has been slower than simplicial on the
    /// benchmark matrices so far.
    pub direct_supernodal: bool,

    /// Show additional messages
    pub verbose: bool,

    /// Configuration parameters for the iterative solvers
    pub iterative: IterativeParams,
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
            direct_supernodal: false,
            verbose: false,
            iterative: IterativeParams::default(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{AmgCoarsening, AmgSmoother, IterativeParams, LinSolParams, Preconditioner};
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
        assert!(!params.direct_supernodal);
        assert_eq!(params.iterative, IterativeParams::default());
    }

    #[test]
    fn iterative_params_default_works() {
        let params = IterativeParams::default();
        assert_eq!(params.preconditioner, Preconditioner::Amg);
        assert_eq!(params.tolerance, 1e-10);
        assert_eq!(params.absolute_tolerance, 0.0);
        assert_eq!(params.max_iterations, 1000);
        assert_eq!(params.gmres_restart, 30);
        assert_eq!(params.num_threads, 1);
        assert_eq!(params.min_parallel_len, 1024);
        assert_eq!(params.amg_coarsening, AmgCoarsening::RugeStuben);
        assert_eq!(params.amg_smoother, AmgSmoother::GaussSeidel);
        assert_eq!(params.amg_coarse_enough, 64);
        assert_eq!(params.amg_max_levels, 32);
    }
}
