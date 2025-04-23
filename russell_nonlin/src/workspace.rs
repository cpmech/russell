#![allow(unused)]

use super::{Config, IterationError, Logger, Method, Stats, System};
use russell_lab::Vector;
use russell_sparse::{CooMatrix, LinSolver};

/// Holds workspace data shared among the ODE solver and actual implementations
pub(crate) struct Workspace<'a> {
    /// Indicates automatic stepsize adjustment
    pub(crate) auto: bool,

    /// Holds statistics and benchmarking data
    pub(crate) stats: Stats,

    /// Number of continued rejections
    pub(crate) n_continued_rejection: usize,

    /// Indicates that the step follows a reject
    pub(crate) follows_reject_step: bool,

    /// Indicates that the iterations have failed
    pub(crate) iterations_failed: bool,

    /// Need to stop due to continued rejection
    pub(crate) stop_continued_rejection: bool,

    /// Need to stop due to small stepsize
    pub(crate) stop_small_stepsize: bool,

    /// Multiplier to the stepsize when the iterations are failing
    ///
    /// Note: this value is currently constant, but it could be made variable
    ///       by analyzing Newton's convergence rate as in the ODE crate.
    pub(crate) h_multiplier_failure: f64,

    /// Previous stepsize
    pub(crate) h_prev: f64,

    /// Next stepsize estimate
    pub(crate) h_new: f64,

    /// Previous relative error
    pub(crate) rel_error_prev: f64,

    /// Current relative error
    pub(crate) rel_error: f64,

    /// Iteration error
    pub(crate) err: IterationError,

    /// Logger
    pub(crate) log: Logger,

    /// Holds G(u, λ)
    ///
    /// (ndim)
    pub(crate) gg: Vector,

    /// Holds the Gu = ∂G/∂u matrix
    ///
    /// (ndim x ndim)
    pub(crate) ggu: CooMatrix,

    /// Indicates that the Gu matrix has been allocated
    ///
    /// Gu is always allocated for the Natural method. Nonetheless, for the
    /// Arclength method, Gu is allocated only if either the bordering algorithm
    /// is activated or the Gu matrix is symmetric (triangular storage).
    pub(crate) with_ggu: bool,

    /// Linear solver
    pub(crate) ls: LinSolver<'a>,

    /// Holds current u
    pub(crate) u: Vector,

    /// Holds current λ
    pub(crate) l: f64,

    /// Holds -δu
    pub(crate) mdu: Vector,

    /// Auxiliary u vector #1 (e.g., for numerical Jacobian)
    pub(crate) u_aux1: Vector,

    /// Auxiliary u vector #2 (e.g., for numerical Jacobian)
    pub(crate) u_aux2: Vector,

    /// Initial multiplier to the stepsize when the iterations are diverging
    h_multiplier_failure_initial: f64,
}

impl<'a> Workspace<'a> {
    /// Allocates a new instance
    pub(crate) fn new<'b, A>(config: &Config, system: &System<'b, A>) -> Self {
        // allocate Gu matrix
        let (ggu, with_ggu) = match config.method {
            Method::Arclength => {
                if config.bordering || system.sym_ggu.triangular() {
                    (
                        CooMatrix::new(system.ndim, system.ndim, system.nnz_ggu, system.sym_ggu).unwrap(),
                        true,
                    )
                } else {
                    (CooMatrix::new(1, 1, 1, system.sym_ggu).unwrap(), false)
                }
            }
            Method::Natural => (
                CooMatrix::new(system.ndim, system.ndim, system.nnz_ggu, system.sym_ggu).unwrap(),
                true,
            ),
        };
        // determine Jacobian size
        let ndim_num_jac = if config.use_numerical_jacobian || system.calc_ggu.is_none() {
            system.ndim
        } else {
            0
        };
        // allocate the workspace
        Workspace {
            auto: false,
            stats: Stats::new(config.method),
            n_continued_rejection: 0,
            follows_reject_step: false,
            iterations_failed: false,
            stop_continued_rejection: false,
            stop_small_stepsize: false,
            h_multiplier_failure: config.m_failure,
            h_prev: 0.0,
            h_new: 0.0,
            rel_error_prev: 0.0,
            rel_error: 0.0,
            err: IterationError::new(config, system.ndim),
            log: Logger::new(config),
            gg: Vector::new(system.ndim),
            ggu,
            with_ggu,
            ls: LinSolver::new(config.genie).unwrap(),
            u: Vector::new(system.ndim),
            l: 0.0,
            mdu: Vector::new(system.ndim),
            u_aux1: Vector::new(ndim_num_jac),
            u_aux2: Vector::new(ndim_num_jac),
            h_multiplier_failure_initial: config.m_failure,
        }
    }

    /// Resets all values
    pub(crate) fn reset(&mut self, h: f64, rel_error_prev_min: f64, auto: bool) {
        self.stats.reset(h);
        self.auto = auto;
        self.n_continued_rejection = 0;
        self.follows_reject_step = false;
        self.iterations_failed = false;
        self.stop_continued_rejection = false;
        self.stop_small_stepsize = false;
        self.h_multiplier_failure = self.h_multiplier_failure_initial;
        self.h_prev = h;
        self.h_new = h;
        self.rel_error_prev = rel_error_prev_min;
        self.rel_error = 0.0;
    }

    /// Returns error messages
    pub(crate) fn errors(&self) -> Vec<String> {
        let mut msg = self.err.messages();
        if self.stop_continued_rejection {
            msg.push("too many continued rejections".to_string());
        }
        if self.stop_small_stepsize {
            msg.push("the stepsize becomes too small".to_string());
        }
        msg
    }
}
