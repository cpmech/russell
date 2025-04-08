use super::{Config, Stats};
use crate::StrError;
use russell_lab::{vec_norm, Norm, Vector};

/// Calculates the numerical error
pub(crate) struct NumError {
    /// Holds max(‖G‖∞,|H|)
    pub(crate) max_gh: f64,

    /// Holds max(‖δu‖∞,|δλ|)
    pub(crate) max_ul: f64,

    /// Converged on
    pub(crate) converged_on_gh: bool,

    /// Diverging on
    pub(crate) diverging_on_gh: bool,

    /// Converged on max(‖δu‖∞,|δλ|)
    pub(crate) converged_on_ul: bool,

    /// Diverging on max(‖δu‖∞,|δλ|)
    pub(crate) diverging_on_ul: bool,

    /// Previous max(‖G‖∞,|H|)
    max_gh_prev: f64,

    /// Previous max(‖δu‖∞,|δλ|)
    max_ul_prev: f64,

    /// Tolerance on max(‖G‖∞,|H|)
    tol_gh: f64,

    /// Tolerance on max(‖δu‖∞,|δλ|)
    tol_ul: f64,

    /// Maximum max(‖δu‖∞,|δλ|) allowed
    max_ul_allowed: f64,

    /// Previous divergence flag (on max(‖δu‖∞,|δλ|))
    prev_div_ul: bool,

    /// Number of continued divergence on max(‖δu‖∞,|δλ|)
    n_cont_div_ul: usize,

    /// Number of allowed continuing divergence on max(‖δu‖∞,|δλ|)
    n_allowed_cont_div_ul: usize,

    /// Maximum number of iterations allowed
    n_iteration_max: usize,
}

impl NumError {
    /// Creates a new instance
    pub fn new(config: &Config) -> Self {
        Self {
            max_gh: 0.0,
            max_ul: 0.0,
            converged_on_gh: false,
            diverging_on_gh: false,
            converged_on_ul: false,
            diverging_on_ul: false,
            max_gh_prev: 0.0,
            max_ul_prev: 0.0,
            tol_gh: config.tol_gh,
            tol_ul: config.tol_ul,
            max_ul_allowed: config.max_ul_allowed,
            prev_div_ul: false,
            n_cont_div_ul: 0,
            n_allowed_cont_div_ul: config.n_allowed_cont_div_ul,
            n_iteration_max: config.n_iteration_max,
        }
    }

    /// Resets the convergence flags and divergence counter
    pub fn reset(&mut self) {
        self.converged_on_gh = false;
        self.diverging_on_gh = false;
        self.converged_on_ul = false;
        self.diverging_on_ul = false;
        self.prev_div_ul = false;
        self.n_cont_div_ul = 0;
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
        self.converged_on_gh = self.max_gh < self.tol_gh;

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

        // check convergence
        self.converged_on_ul = if iteration == 0 {
            false
        } else {
            self.max_ul < self.tol_ul
        };

        // check if diverging
        self.prev_div_ul = self.diverging_on_ul;
        self.diverging_on_ul = if iteration < 2 {
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

    /// Checks for errors and returns break flag
    ///
    /// Returns `break flag`
    pub fn check(&self, iteration: usize, stats: &mut Stats, auto: bool) -> Result<bool, StrError> {
        if self.max_ul > self.max_ul_allowed {
            // large (δu,δλ)
            stats.n_large_du_dl += 1;
            if auto {
                Ok(true)
            } else {
                Err("max(‖δu‖∞,|δλ|) is too large")
            }
        } else if self.n_cont_div_ul >= self.n_allowed_cont_div_ul {
            // continued divergence
            stats.n_continued_divergence += 1;
            if auto {
                Ok(true)
            } else {
                Err("Continued divergence detected")
            }
        } else if iteration == self.n_iteration_max - 1 {
            // failed to converge
            stats.n_failed_converge += 1;
            if auto {
                Ok(true)
            } else {
                Err("Newton-Raphson did not converge")
            }
        } else {
            // keep going
            Ok(false)
        }
    }
}
