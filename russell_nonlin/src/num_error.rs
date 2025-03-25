use super::NlParams;
use crate::StrError;
use russell_lab::{vec_norm, Norm, Vector};

/// Calculates the numerical error
pub(crate) struct NumError<'a> {
    /// Configuration parameters
    params: &'a NlParams,

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
}

impl<'a> NumError<'a> {
    /// Creates a new instance
    pub fn new(params: &'a NlParams) -> Self {
        Self {
            params,
            max_gh: 0.0,
            max_ul: 0.0,
            converged_on_gh: false,
            diverging_on_gh: false,
            converged_on_ul: false,
            diverging_on_ul: false,
            max_gh_prev: 0.0,
            max_ul_prev: 0.0,
        }
    }

    /// Resets convergence flags for a new step
    pub fn reset(&mut self) {
        self.converged_on_gh = false;
        self.diverging_on_gh = false;
        self.converged_on_ul = false;
        self.diverging_on_ul = false;
    }

    /// Marks the problem as converged for linear analysis
    pub fn set_converged_linear_problem(&mut self) {
        self.converged_on_gh = true;
    }

    /// Returns whether max(‖δu‖∞,|δλ|) is too large
    pub fn large_norm_ul(&self) -> bool {
        if self.max_ul > self.params.max_ul {
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
        self.converged_on_gh = self.max_gh < self.params.tol_gh;

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
            self.max_ul < self.params.tol_ul
        };

        // check if diverging
        self.diverging_on_ul = if iteration < 2 {
            false
        } else {
            self.max_ul > self.max_ul_prev
        };

        // for subsequent iterations
        self.max_ul_prev = self.max_ul;
        Ok(())
    }
}
