use super::{Config, IterationError, Logger, Method, Stats, System};
use russell_lab::Vector;

/// Holds workspace data shared among the nonlinear solver and its method-specific implementations
pub(crate) struct Workspace {
    // control variables and structures ------------------------------------------
    //
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

    /// Indicates the target λ or uᵢ has been reached
    pub(crate) target_reached: bool,

    /// Indicates that the solver should stop gracefully (e.g., requested by the secondary update function)
    pub(crate) stop_gracefully: bool,

    /// Indicates that a fresh step must copy the external state variables before it starts
    ///
    /// Set by `reset_stats_and_flags` and by every accepted step; cleared when the copy is made.
    /// A step retried after a rejection keeps it false because the restore already put the external
    /// state back at the step-start snapshot.
    pub(crate) backup_needed: bool,

    /// Indicates that a step has started and has not been accepted or rejected yet
    ///
    /// Set at the beginning of a step attempt (before any fallible operation) and cleared by
    /// `accept` and `reject`. `Solver::solve` uses it to restore the external state variables when
    /// the step is still in flight at a terminal failure.
    pub(crate) step_in_flight: bool,

    // state variables -----------------------------------------------------------
    //
    /// Stepsize: either σ (pseudo-arclength) or Δλ (natural parameter)
    pub(crate) h: f64,

    /// Holds current u
    pub(crate) u: Vector,

    /// Holds current λ
    pub(crate) l: f64,

    /// Holds G(u, λ)
    pub(crate) gg: Vector,

    /// Part of the tangent vector (duds,dλds) (pseudo-arclength only)
    pub(crate) duds: Vector,

    /// Part of the tangent vector (duds,dλds) (pseudo-arclength only)
    pub(crate) dlds: f64,

    /// Holds the predictor values for debugging
    ///
    /// Holds λ and the first two components of u (if available), calculated by the predictor step.
    pub(crate) predictor_values_debug: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
}

impl Workspace {
    /// Allocates a new instance
    pub(crate) fn new<'b, A>(config: &Config, system: &System<'b, A>) -> Self {
        // allocate duds vector
        let duds = match config.method {
            Method::Arclength => Vector::new(system.ndim),
            Method::Natural => Vector::new(0),
        };

        // allocate the workspace
        Workspace {
            // control variables and structures
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
            target_reached: false,
            stop_gracefully: false,
            backup_needed: true,
            step_in_flight: false,

            // state variables
            h: 0.0,
            u: Vector::new(system.ndim),
            l: 0.0,
            gg: Vector::new(system.ndim),
            duds,
            dlds: 0.0,

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
        self.target_reached = false;
        self.stop_gracefully = false;
        self.backup_needed = true;
        self.step_in_flight = false;
    }
}
