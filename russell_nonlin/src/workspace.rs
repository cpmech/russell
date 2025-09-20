#![allow(unused)]

use super::{Config, IterationError, Logger, Method, Stats, System};
use russell_lab::Vector;
use russell_sparse::{CooMatrix, LinSolver};

/// Holds workspace data shared among the ODE solver and actual implementations
pub(crate) struct Workspace<'a> {
    // control variables and structures ------------------------------------------
    //
    /// Indicates that the Gu matrix has been allocated
    ///
    /// Gu is always allocated for the Natural method. Nonetheless, for the
    /// Arclength method, Gu is allocated only if either the bordering algorithm
    /// is activated or the Gu matrix is symmetric (triangular storage).
    pub(crate) with_ggu: bool,

    /// Linear solver
    pub(crate) ls: LinSolver<'a>,

    /// Iteration error
    pub(crate) err: IterationError,

    /// Logger
    pub(crate) log: Logger,

    // stats and flags -----------------------------------------------------------
    //
    /// Holds statistics and benchmarking data
    pub(crate) stats: Stats,

    /// Indicates automatic stepsize adjustment
    pub(crate) auto: bool,

    /// Current number of iterations
    pub(crate) n_iteration: usize,

    /// Records the number of times that the iterations failed
    ///
    /// Failure here means that the iterations failed to converge
    ///
    /// Three problems may be detected:
    ///
    /// 1. ‖(δu,δλ)‖∞ is too large
    /// 2. continued divergence detected
    /// 3. max number of iterations reached
    pub(crate) n_continued_failure: usize,

    /// Records the number of times that a step update was rejected
    ///
    /// The rejection may occur due to the following criteria:
    ///
    /// 1. The distance between the predictor and the converged result is too large
    /// 2. A measure of curvature from the previous to the converged result is too large
    /// 3. The number of iterations exceeded a desired number of iterations
    pub(crate) n_continued_rejection: usize,

    /// Indicates that this step follows a previously failed step
    pub(crate) follows_failure: bool,

    /// Indicates that this step follows a previously rejected step
    pub(crate) follows_rejection: bool,

    /// Flags that the solver stopped due to a secondary update error
    pub(crate) stopped_due_to_secondary_update_fail_predictor: bool,

    /// Flags that the solver stopped due to continued failure
    pub(crate) stopped_due_to_continued_failure: bool,

    /// Flags that the solver stopped due to continued rejection
    pub(crate) stopped_due_to_continued_rejection: bool,

    /// Flags that the solver stopped because the stepsize became too small
    pub(crate) stopped_due_to_small_stepsize: bool,

    // state variables -----------------------------------------------------------
    //
    /// Holds G(u, λ)
    ///
    /// (ndim)
    pub(crate) gg: Vector,

    /// Holds the Gu = ∂G/∂u matrix
    ///
    /// (ndim x ndim)
    pub(crate) ggu: CooMatrix,

    /// Next stepsize estimate
    pub(crate) h_estimate: f64,

    /// Stepsize: either σ (pseudo-arclength) or Δλ (natural parameter)
    pub(crate) h: f64,

    /// Holds current u
    pub(crate) u: Vector,

    /// Holds current λ
    pub(crate) l: f64,

    /// Part of the tangent vector (duds,dλds) for the pseudo-arclength method
    ///
    /// **Note**: this vector is only allocated for the pseudo-arclength method
    ///
    /// (ndim)
    pub(crate) duds: Vector,

    /// Part of the tangent vector (duds,dλds) for the pseudo-arclength method
    pub(crate) dlds: f64,

    /// Holds -δu (negative of iteration increment)
    pub(crate) mdu: Vector,

    /// Auxiliary u vector #1 (e.g., for numerical Jacobian)
    pub(crate) u_aux1: Vector,

    /// Auxiliary u vector #2 (e.g., for numerical Jacobian)
    pub(crate) u_aux2: Vector,

    /// Indicates whether this step results were acceptable or not
    pub(crate) acceptable: bool,

    /// Holds the predictor values for debugging
    ///
    /// Holds λ and the first two components of u (if available), calculated by the predictor step.
    pub(crate) predictor_values_debug: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
}

impl<'a> Workspace<'a> {
    /// Allocates a new instance
    pub(crate) fn new<'b, A>(config: &Config, system: &System<'b, A>) -> Self {
        // allocate Gu matrix
        let (ggu, with_ggu, with_tangent) = match config.method {
            Method::Arclength => {
                if config.bordering || system.sym_ggu.triangular() {
                    (
                        CooMatrix::new(system.ndim, system.ndim, system.nnz_ggu, system.sym_ggu).unwrap(),
                        true,
                        true,
                    )
                } else {
                    (CooMatrix::new(1, 1, 1, system.sym_ggu).unwrap(), false, true)
                }
            }
            Method::Natural => (
                CooMatrix::new(system.ndim, system.ndim, system.nnz_ggu, system.sym_ggu).unwrap(),
                true,
                false,
            ),
        };

        // determine Jacobian size
        let ndim_num_jac = if config.use_numerical_jacobian || system.calc_ggu.is_none() {
            system.ndim
        } else {
            0
        };

        // auxiliary variable
        let ndim_tangent = if with_tangent { system.ndim } else { 0 };

        // allocate the workspace
        Workspace {
            // control variables and structures
            with_ggu,
            ls: LinSolver::new(config.genie).unwrap(),
            err: IterationError::new(config, system.ndim),
            log: Logger::new(config),

            // stats and flags
            stats: Stats::new(config),
            auto: false,
            n_iteration: 0,
            n_continued_failure: 0,
            n_continued_rejection: 0,
            follows_failure: false,
            follows_rejection: false,
            stopped_due_to_secondary_update_fail_predictor: false,
            stopped_due_to_continued_failure: false,
            stopped_due_to_continued_rejection: false,
            stopped_due_to_small_stepsize: false,

            // state variables
            gg: Vector::new(system.ndim),
            ggu,
            h_estimate: 0.0,
            h: config.h_ini,
            u: Vector::new(system.ndim),
            l: 0.0,
            duds: Vector::new(ndim_tangent),
            dlds: 0.0,
            mdu: Vector::new(system.ndim),
            u_aux1: Vector::new(ndim_num_jac),
            u_aux2: Vector::new(ndim_num_jac),
            acceptable: true,

            // debugging
            predictor_values_debug: None,
        }
    }

    /// Resets stats and flags
    pub(crate) fn reset_stats_and_flags(&mut self, auto: bool) {
        self.stats.reset(auto);
        self.auto = auto;
        self.n_iteration = 0;
        self.n_continued_failure = 0;
        self.n_continued_rejection = 0;
        self.follows_failure = false;
        self.follows_rejection = false;
        self.stopped_due_to_secondary_update_fail_predictor = false;
        self.stopped_due_to_continued_failure = false;
        self.stopped_due_to_continued_rejection = false;
        self.stopped_due_to_small_stepsize = false;
    }

    /// Returns error messages
    pub(crate) fn errors(&self) -> Vec<String> {
        let mut msg = self.err.messages();
        if self.stopped_due_to_secondary_update_fail_predictor {
            msg.push("secondary update failed in the predictor phase".to_string());
        }
        if self.stopped_due_to_continued_failure {
            msg.push("too many continued (iteration) failures".to_string());
        }
        if self.stopped_due_to_continued_rejection {
            msg.push("too many continued (error behavior) rejections".to_string());
        }
        if self.stopped_due_to_small_stepsize {
            msg.push("the stepsize becomes too small".to_string());
        }
        msg
    }
}
