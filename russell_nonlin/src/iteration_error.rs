use super::{Config, StateRef, Stats};
use crate::StrError;
use russell_lab::{vec_norm, Norm, Vector};

/// Calculates the iteration error
pub(crate) struct IterationError {
    /// Holds max(‖G‖∞,|H|)
    pub(crate) max_gh: f64,

    /// Holds max(‖δu‖∞,|δλ|)
    pub(crate) max_ul: f64,

    /// RMS error on (δu,δλ)
    pub(crate) rms_ul: f64,

    /// Converged on
    pub(crate) converged_on_gh: bool,

    /// Diverging on
    pub(crate) diverging_on_gh: bool,

    /// Converged on RMS(δu,δλ)
    pub(crate) converged_on_ul: bool,

    /// Diverging on RMS(δu,δλ)
    pub(crate) diverging_on_ul: bool,

    /// Previous max(‖G‖∞,|H|)
    max_gh_prev: f64,

    /// Previous max(‖δu‖∞,|δλ|)
    max_ul_prev: f64,

    /// Tolerance on max(‖G‖∞,|H|)
    tol_gh_abs: f64,

    /// Absolute tolerance on RMS(δu,δλ)
    tol_ul_abs: f64,

    /// Relative tolerance on RMS(δu,δλ)
    tol_ul_rel: f64,

    /// Maximum max(‖δu‖∞,|δλ|) allowed
    max_ul_allowed: f64,

    /// Previous divergence flag (on max(‖δu‖∞,|δλ|))
    prev_div_ul: bool,

    /// Number of continued divergence on max(‖δu‖∞,|δλ|)
    n_cont_div_ul: usize,

    /// Max number of continuing divergence on max(‖δu‖∞,|δλ|) allowed
    n_cont_div_ul_allowed: usize,

    /// Maximum number of iterations allowed
    n_iteration_max: usize,

    /// Failed on max(‖G‖∞,|H|)
    fail_large_du_dl: bool,

    /// Failed on continued divergence
    fail_cont_div_ul: bool,

    /// Failed on max number of iterations
    fail_max_iter: bool,

    /// Scaling vector for RMS(δu,δλ)
    scaling_ul: Vector,
}

impl IterationError {
    /// Creates a new instance
    pub fn new(config: &Config, ndim: usize) -> Self {
        Self {
            max_gh: 0.0,
            max_ul: 0.0,
            rms_ul: 0.0,
            converged_on_gh: false,
            diverging_on_gh: false,
            converged_on_ul: false,
            diverging_on_ul: false,
            max_gh_prev: 0.0,
            max_ul_prev: 0.0,
            tol_gh_abs: config.tol_gh_abs,
            tol_ul_abs: config.tol_ul_abs,
            tol_ul_rel: config.tol_ul_rel,
            max_ul_allowed: config.max_ul_allowed,
            prev_div_ul: false,
            n_cont_div_ul: 0,
            n_cont_div_ul_allowed: config.n_cont_div_ul_allowed,
            n_iteration_max: config.n_iteration_max,
            fail_large_du_dl: false,
            fail_cont_div_ul: false,
            fail_max_iter: false,
            scaling_ul: Vector::new(ndim + 1),
        }
    }

    /// Resets the convergence flags and divergence counter
    pub fn reset(&mut self, state: &StateRef) {
        self.converged_on_gh = false;
        self.diverging_on_gh = false;
        self.converged_on_ul = false;
        self.diverging_on_ul = false;
        self.prev_div_ul = false;
        self.n_cont_div_ul = 0;
        self.fail_cont_div_ul = false;
        self.fail_large_du_dl = false;
        self.fail_max_iter = false;
        let ndim = self.scaling_ul.dim() - 1;
        for i in 0..ndim {
            self.scaling_ul[i] = self.tol_ul_abs + self.tol_ul_rel * f64::abs(state.u[i]);
        }
        self.scaling_ul[ndim] = self.tol_ul_abs + self.tol_ul_rel * f64::abs(*state.l);
    }

    /// Marks the problem as converged for linear analysis
    pub fn set_converged_linear_problem(&mut self) {
        self.converged_on_gh = true;
    }

    /// Returns whether max(‖δu‖∞,|δλ|) is too large
    pub fn large_du_dl(&self) -> bool {
        if self.max_ul > self.max_ul_allowed {
            true
        } else {
            false
        }
    }

    /// Checks if the solution has converged based on any criterion
    pub fn converged(&self) -> bool {
        self.converged_on_gh || self.converged_on_ul
    }

    /// Analyzes convergence on max(‖G‖∞,|H|)
    pub fn analyze_gh(&mut self, iteration: usize, gg: &Vector, hh: f64) -> Result<(), StrError> {
        // compute max(‖G‖∞,|H|)
        self.max_gh = f64::max(vec_norm(gg, Norm::Max), f64::abs(hh));

        // check for NaN or Inf
        if !self.max_gh.is_finite() {
            return Err("Found NaN or Inf in max(‖G‖∞,|H|)");
        }

        // check convergence
        self.converged_on_gh = self.max_gh < self.tol_gh_abs;

        // check if diverging
        self.diverging_on_gh = if iteration == 0 {
            false
        } else {
            self.max_gh > self.max_gh_prev
        };

        // for subsequent iterations
        self.max_gh_prev = self.max_gh;
        Ok(())
    }

    /// Analyzes convergence on max(‖δu‖∞,|δλ|)
    pub fn analyze_ul(&mut self, iteration: usize, du: &Vector, dl: f64) -> Result<(), StrError> {
        // compute max(‖δu‖∞,|δλ|)
        self.max_ul = f64::max(vec_norm(du, Norm::Max), f64::abs(dl));

        // check for NaN or Inf
        if !self.max_ul.is_finite() {
            return Err("Found NaN or Inf in max(‖δu‖∞,|δλ|)");
        }

        // compute RMS(δu,δλ)
        let ndim = self.scaling_ul.dim() - 1;
        let mut sum = 0.0;
        for i in 0..ndim {
            sum += du[i] / self.scaling_ul[i] * du[i] / self.scaling_ul[i];
        }
        sum += dl / self.scaling_ul[ndim] * dl / self.scaling_ul[ndim];
        self.rms_ul = f64::sqrt(sum / ((ndim + 1) as f64));

        // check convergence
        self.converged_on_ul = self.rms_ul <= 1.0;

        // check if diverging
        self.prev_div_ul = self.diverging_on_ul;
        self.diverging_on_ul = if iteration == 0 {
            false
        } else {
            self.max_ul > self.max_ul_prev
        };

        // increment divergence counter
        if self.prev_div_ul && self.diverging_on_ul {
            self.n_cont_div_ul += 1;
        }

        // for subsequent iterations
        self.max_ul_prev = self.max_ul;
        Ok(())
    }

    /// Checks for failures and returns a flag to break the iteration loop
    ///
    /// Returns `break` flag
    pub fn failures(&mut self, iteration: usize, stats: &mut Stats) -> bool {
        // large (δu,δλ)
        if self.max_ul > self.max_ul_allowed {
            stats.n_large_du_dl += 1;
            self.fail_large_du_dl = true;
        }

        // continued divergence
        if self.n_cont_div_ul >= self.n_cont_div_ul_allowed {
            stats.n_continued_divergence += 1;
            self.fail_cont_div_ul = true;
        }

        // max number of iterations reached
        if iteration == self.n_iteration_max - 1 {
            stats.n_iterations_max += 1;
            self.fail_max_iter = true;
        }

        // flag to break the iteration loop
        self.fail_large_du_dl || self.fail_cont_div_ul || self.fail_max_iter
    }

    /// Returns whether the simulation has failed or not
    pub fn failed(&self) -> bool {
        self.fail_large_du_dl || self.fail_cont_div_ul || self.fail_max_iter
    }

    /// Returns error messages
    pub fn messages(&self) -> Vec<String> {
        let mut messages = Vec::with_capacity(3);
        if self.fail_large_du_dl {
            messages.push("max(‖δu‖∞,|δλ|) is too large".to_string());
        }
        if self.fail_cont_div_ul {
            messages.push("continued divergence detected".to_string());
        }
        if self.fail_max_iter {
            messages.push("max number of iterations reached".to_string());
        }
        messages
    }
}
