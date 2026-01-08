use super::{Config, Status};
use russell_lab::{vec_norm, Norm, Vector};

/// Calculates the iteration error
pub(crate) struct IterationError {
    /// Holds max(‖G‖∞,|N|)
    pub(crate) residual_max: f64,

    /// Holds max(‖δu‖∞,|δλ|)
    pub(crate) delta_max: f64,

    /// RMS error on (δu,δλ)
    pub(crate) delta_rms: f64,

    /// Converged on (‖G‖∞,|N|)
    pub(crate) residual_converged: bool,

    /// Diverging on (‖G‖∞,|N|)
    pub(crate) residual_diverging: bool,

    /// Converged on RMS(δu,δλ)
    pub(crate) delta_converged: bool,

    /// Diverging on RMS(δu,δλ)
    pub(crate) delta_diverging: bool,

    /// Previous `residual_max`
    residual_max_prev: f64,

    /// Previous `delta_max`
    delta_max_prev: f64,

    /// Tolerance on max(‖G‖∞,|N|)
    tol_abs_residual: f64,

    /// Absolute tolerance on RMS(δu,δλ)
    tol_abs_delta: f64,

    /// Relative tolerance on RMS(δu,δλ)
    tol_rel_delta: f64,

    /// Allowed `delta_max`
    allowed_delta_max: f64,

    /// Previous `residual_diverging`
    prev_residual_diverging: bool,

    /// Previous `delta_diverging`
    prev_delta_diverging: bool,

    /// Number of large max(‖δu‖∞,|δλ|)
    n_large_delta: usize,

    /// Number of continued `residual_diverging`
    n_continued_residual_divergence: usize,

    /// Number of continued `delta_diverging`
    n_continued_delta_divergence: usize,

    /// Maximum allowed number of continued divergence on ‖G,N‖∞
    n_cont_residual_divergence_max: usize,

    /// Maximum allowed number of continued divergence on ‖δu,δλ‖∞
    n_cont_delta_divergence_max: usize,

    /// Maximum allowed number of iterations
    n_iteration_max: usize,

    /// Scaling vector for RMS(δu,δλ)
    scaling: Vector,
}

impl IterationError {
    /// Creates a new instance
    pub fn new(config: &Config, ndim: usize) -> Self {
        Self {
            residual_max: 0.0,
            delta_max: 0.0,
            delta_rms: 0.0,
            residual_converged: false,
            residual_diverging: false,
            delta_converged: false,
            delta_diverging: false,
            residual_max_prev: 0.0,
            delta_max_prev: 0.0,
            tol_abs_residual: config.tol_abs_residual,
            tol_abs_delta: config.tol_abs_delta,
            tol_rel_delta: config.tol_rel_delta,
            allowed_delta_max: config.delta_max_allowed,
            prev_residual_diverging: false,
            prev_delta_diverging: false,
            n_large_delta: 0,
            n_continued_residual_divergence: 0,
            n_continued_delta_divergence: 0,
            n_cont_residual_divergence_max: config.n_cont_residual_divergence_max,
            n_cont_delta_divergence_max: config.n_cont_delta_divergence_max,
            n_iteration_max: config.n_iteration_max,
            scaling: Vector::new(ndim + 1), // +1 for λ
        }
    }

    /// Resets the convergence flags and divergence counter
    pub fn reset(&mut self, u: &Vector, l: f64) {
        self.residual_converged = false;
        self.residual_diverging = false;
        self.delta_converged = false;
        self.delta_diverging = false;
        self.prev_residual_diverging = false;
        self.prev_delta_diverging = false;
        self.n_large_delta = 0;
        self.n_continued_residual_divergence = 0;
        self.n_continued_delta_divergence = 0;
        let ndim = self.scaling.dim() - 1; // -1 due to λ
        for i in 0..ndim {
            self.scaling[i] = self.tol_abs_delta + self.tol_rel_delta * f64::abs(u[i]);
        }
        self.scaling[ndim] = self.tol_abs_delta + self.tol_rel_delta * f64::abs(l);
    }

    /// Checks if the solution has converged based on any criterion
    pub fn converged(&self) -> bool {
        self.residual_converged || self.delta_converged
    }

    /// Analyzes convergence on max(‖G‖∞,|N|)
    ///
    /// Returns true if NaN or Inf is found, false otherwise (success)
    pub fn analyze_residual(&mut self, iteration: usize, gg: &Vector, nn: f64) -> bool {
        // compute max norm
        self.residual_max = f64::max(vec_norm(gg, Norm::Max), f64::abs(nn));

        // check for NaN or Inf
        if !self.residual_max.is_finite() {
            return true;
        }

        // check convergence
        self.residual_converged = self.residual_max < self.tol_abs_residual;

        // check if diverging
        self.prev_residual_diverging = self.residual_diverging;
        self.residual_diverging = if iteration == 0 {
            false
        } else {
            self.residual_max > self.residual_max_prev
        };

        // increment continued divergence counter
        if self.prev_residual_diverging && self.residual_diverging {
            self.n_continued_residual_divergence += 1;
        }

        // for subsequent iterations
        self.residual_max_prev = self.residual_max;

        // success, no NaN or Inf found
        false
    }

    /// Analyzes convergence on RMS(‖δu‖∞,|δλ|)
    ///
    /// `x` may be simply `δu` or the augmented vector `(δu, δλ)`
    ///
    /// Returns true if NaN or Inf is found, false otherwise (success)
    pub fn analyze_delta(&mut self, iteration: usize, x: &Vector) -> bool {
        // compute max norm
        self.delta_max = vec_norm(x, Norm::Max);

        // check for NaN or Inf
        if !self.delta_max.is_finite() {
            return true;
        }

        // compute RMS(δu,δλ)
        let mut sum = 0.0;
        for i in 0..x.dim() {
            sum += (x[i] / self.scaling[i]) * (x[i] / self.scaling[i]);
        }
        self.delta_rms = f64::sqrt(sum / (x.dim() as f64));

        // check convergence
        self.delta_converged = self.delta_rms <= 1.0;

        // check if diverging
        self.prev_delta_diverging = self.delta_diverging;
        self.delta_diverging = if iteration == 0 {
            false
        } else {
            self.delta_max > self.delta_max_prev
        };

        // increment continued divergence counter
        if self.prev_delta_diverging && self.delta_diverging {
            self.n_continued_delta_divergence += 1;
        }

        // for subsequent iterations
        self.delta_max_prev = self.delta_max;

        // success, no NaN or Inf found
        false
    }

    /// Captures eventual failures
    pub fn capture_failures(&mut self, iteration: usize) -> Status {
        // large (‖δu‖∞,|δλ|)
        if self.delta_max > self.allowed_delta_max {
            self.n_large_delta += 1;
            return Status::LargeDelta;
        }

        // continued divergence on residual
        if self.n_continued_residual_divergence >= self.n_cont_residual_divergence_max {
            return Status::ContinuedResidualDivergence;
        }

        // continued divergence on delta
        if self.n_continued_delta_divergence >= self.n_cont_delta_divergence_max {
            return Status::ContinuedDeltaDivergence;
        }

        // max number of iterations reached
        if iteration == self.n_iteration_max - 1 {
            return Status::ReachedMaxIterations;
        }

        // no failure
        Status::Success
    }
}
