use super::{Config, State, Stats};
use crate::StrError;
use russell_lab::{vec_norm, Norm, Vector};

/// Calculates the iteration error
pub(crate) struct IterationError {
    /// Holds max(‖G‖∞,|Nₒ|)
    pub(crate) residual_max: f64,

    /// Holds max(‖δu‖∞,|δλ|)
    pub(crate) delta_max: f64,

    /// RMS error on (δu,δλ)
    pub(crate) delta_rms: f64,

    /// Converged on (‖G‖∞,|Nₒ|)
    pub(crate) residual_converged: bool,

    /// Diverging on (‖G‖∞,|Nₒ|)
    pub(crate) residual_diverging: bool,

    /// Converged on RMS(δu,δλ)
    pub(crate) delta_converged: bool,

    /// Diverging on RMS(δu,δλ)
    pub(crate) delta_diverging: bool,

    /// Previous `residual_max`
    residual_max_prev: f64,

    /// Previous `delta_max`
    delta_max_prev: f64,

    /// Tolerance on max(‖G‖∞,|Nₒ|)
    tol_abs_residual: f64,

    /// Absolute tolerance on RMS(δu,δλ)
    tol_abs_delta: f64,

    /// Relative tolerance on RMS(δu,δλ)
    tol_rel_delta: f64,

    /// Allowed `delta_max`
    allowed_delta_max: f64,

    /// Previous `delta_diverging`
    delta_diverging_prev: bool,

    /// Number of continued `delta_diverging`
    n_continued_delta_diverging: usize,

    /// Allowed `n_cont_delta_diverging`
    allowed_continued_divergence: usize,

    /// Allowed number of iterations
    allowed_iterations: usize,

    /// Failed on large (‖δu‖∞,|δλ|)
    fail_large_delta: bool,

    /// Failed on continued divergence
    fail_continued_divergence: bool,

    /// Failed on max number of iterations
    fail_max_iterations: bool,

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
            allowed_delta_max: config.allowed_delta_max,
            delta_diverging_prev: false,
            n_continued_delta_diverging: 0,
            allowed_continued_divergence: config.allowed_continued_divergence,
            allowed_iterations: config.allowed_iterations,
            fail_large_delta: false,
            fail_continued_divergence: false,
            fail_max_iterations: false,
            scaling: Vector::new(ndim + 1), // +1 for λ
        }
    }

    /// Resets the convergence flags and divergence counter
    pub fn reset(&mut self, state: &State) {
        self.residual_converged = false;
        self.residual_diverging = false;
        self.delta_converged = false;
        self.delta_diverging = false;
        self.delta_diverging_prev = false;
        self.n_continued_delta_diverging = 0;
        self.fail_continued_divergence = false;
        self.fail_large_delta = false;
        self.fail_max_iterations = false;
        let ndim = self.scaling.dim() - 1; // -1 due to λ
        for i in 0..ndim {
            self.scaling[i] = self.tol_abs_delta + self.tol_rel_delta * f64::abs(state.u[i]);
        }
        self.scaling[ndim] = self.tol_abs_delta + self.tol_rel_delta * f64::abs(state.l);
    }

    /// Marks the problem as converged for linear analysis
    pub fn set_converged_linear_problem(&mut self) {
        self.residual_converged = true;
    }

    /// Returns whether max(‖δu‖∞,|δλ|) is too large
    pub fn is_delta_large(&self) -> bool {
        self.delta_max > self.allowed_delta_max
    }

    /// Checks if the solution has converged based on any criterion
    pub fn converged(&self) -> bool {
        self.residual_converged || self.delta_converged
    }

    /// Analyzes convergence on max(‖G‖∞,|Nₒ|)
    pub fn analyze_residual(&mut self, iteration: usize, gg: &Vector, nno: f64) -> Result<(), StrError> {
        // compute max norm
        self.residual_max = f64::max(vec_norm(gg, Norm::Max), f64::abs(nno));

        // check for NaN or Inf
        if !self.residual_max.is_finite() {
            return Err("Found NaN or Inf in max(‖G‖∞,|Nₒ|)");
        }

        // check convergence
        self.residual_converged = self.residual_max < self.tol_abs_residual;

        // check if diverging
        self.residual_diverging = if iteration == 0 {
            false
        } else {
            self.residual_max > self.residual_max_prev
        };

        // for subsequent iterations
        self.residual_max_prev = self.residual_max;
        Ok(())
    }

    /// Analyzes convergence on RMS(‖δu‖∞,|δλ|)
    ///
    /// `x` may be simply `δu` or the augmented vector `(δu, δλ)`
    pub fn analyze_delta(&mut self, iteration: usize, x: &Vector) -> Result<(), StrError> {
        // compute max norm
        self.delta_max = vec_norm(x, Norm::Max);

        // check for NaN or Inf
        if !self.delta_max.is_finite() {
            return Err("Found NaN or Inf in max(‖δu‖∞,|δλ|)");
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
        self.delta_diverging_prev = self.delta_diverging;
        self.delta_diverging = if iteration == 0 {
            false
        } else {
            self.delta_max > self.delta_max_prev
        };

        // increment continued divergence counter
        if self.delta_diverging_prev && self.delta_diverging {
            self.n_continued_delta_diverging += 1;
        }

        // for subsequent iterations
        self.delta_max_prev = self.delta_max;
        Ok(())
    }

    /// Checks for failures and returns a flag to break the iteration loop
    ///
    /// Returns `break` flag
    pub fn failures(&mut self, iteration: usize, stats: &mut Stats) -> bool {
        // large (δu,δλ)
        if self.delta_max > self.allowed_delta_max {
            stats.n_large_delta += 1;
            self.fail_large_delta = true;
        }

        // continued divergence
        if self.n_continued_delta_diverging >= self.allowed_continued_divergence {
            stats.n_continued_divergence += 1;
            self.fail_continued_divergence = true;
        }

        // max number of iterations reached
        if iteration == self.allowed_iterations - 1 {
            stats.n_iterations_max += 1;
            self.fail_max_iterations = true;
        }

        // flag to break the iteration loop
        self.fail_large_delta || self.fail_continued_divergence || self.fail_max_iterations
    }

    /// Returns whether the simulation has failed or not
    pub fn failed(&self) -> bool {
        self.fail_large_delta || self.fail_continued_divergence || self.fail_max_iterations
    }

    /// Returns error messages
    pub fn messages(&self) -> Vec<String> {
        let mut messages = Vec::with_capacity(3);
        if self.fail_large_delta {
            messages.push("max(‖δu‖∞,|δλ|) is too large".to_string());
        }
        if self.fail_continued_divergence {
            messages.push("continued divergence detected".to_string());
        }
        if self.fail_max_iterations {
            messages.push("max number of iterations reached".to_string());
        }
        messages
    }
}
