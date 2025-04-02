#![allow(unused)]

use super::{Logger, NlConfig, NlSystem, NumError, Stats};
use russell_lab::Vector;
use russell_sparse::{CooMatrix, LinSolver};

/// Holds workspace data shared among the ODE solver and actual implementations
pub(crate) struct Workspace<'a> {
    /// Holds statistics and benchmarking data
    pub(crate) stats: Stats,

    /// Indicates that the step follows a reject
    pub(crate) follows_reject_step: bool,

    /// Indicates that the iterations (in an implicit method) are diverging
    pub(crate) iterations_diverging: bool,

    /// Multiplier to the stepsize when the iterations are diverging
    pub(crate) h_multiplier_diverging: f64,

    /// Previous stepsize
    pub(crate) h_prev: f64,

    /// Next stepsize estimate
    pub(crate) h_new: f64,

    /// Previous relative error
    pub(crate) rel_error_prev: f64,

    /// Current relative error
    pub(crate) rel_error: f64,

    /// Numerical error
    pub(crate) err: NumError,

    /// Logger
    pub(crate) log: Logger,

    /// Holds G(u, λ)
    pub(crate) gg: Vector,

    /// Holds the Gu = ∂G/∂u matrix
    pub(crate) ggu: CooMatrix,

    /// Linear solver
    pub(crate) ls: LinSolver<'a>,

    /// Holds current u
    pub(crate) u: Vector,

    /// Holds current λ
    pub(crate) l: f64,

    /// Holds -δu
    pub(crate) mdu: Vector,

    /// Holds -δλ
    pub(crate) mdl: f64,

    /// Auxiliary u vector #1 (e.g., for numerical Jacobian)
    pub(crate) u_aux1: Vector,

    /// Auxiliary u vector #2 (e.g., for numerical Jacobian)
    pub(crate) u_aux2: Vector,
}

impl<'a> Workspace<'a> {
    /// Allocates a new instance
    pub(crate) fn new<'b, A>(config: &NlConfig, system: &NlSystem<'b, A>) -> Self {
        let n_num_j = if config.use_numerical_jacobian || system.calc_ggu.is_none() {
            system.ndim
        } else {
            0
        };
        Workspace {
            stats: Stats::new(config.method),
            follows_reject_step: false,
            iterations_diverging: false,
            h_multiplier_diverging: 1.0,
            h_prev: 0.0,
            h_new: 0.0,
            rel_error_prev: 0.0,
            rel_error: 0.0,
            err: NumError::new(config),
            log: Logger::new(config),
            gg: Vector::new(system.ndim),
            ggu: CooMatrix::new(system.ndim, system.ndim, system.nnz_ggu, system.sym_ggu).unwrap(),
            ls: LinSolver::new(config.genie).unwrap(),
            u: Vector::new(system.ndim),
            l: 0.0,
            mdu: Vector::new(system.ndim),
            mdl: 0.0,
            u_aux1: Vector::new(n_num_j),
            u_aux2: Vector::new(n_num_j),
        }
    }

    /// Resets all values
    pub(crate) fn reset(&mut self, h: f64, rel_error_prev_min: f64) {
        self.stats.reset(h);
        self.follows_reject_step = false;
        self.iterations_diverging = false;
        self.h_multiplier_diverging = 1.0;
        self.h_prev = h;
        self.h_new = h;
        self.rel_error_prev = rel_error_prev_min;
        self.rel_error = 0.0;
    }
}
