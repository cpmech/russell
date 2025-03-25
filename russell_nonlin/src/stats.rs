#![allow(unused)]

use super::NlMethod;
use russell_lab::{format_nanoseconds, Stopwatch};
use std::fmt::{self, Write};

/// Holds statistics and benchmarking data
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Holds the method
    method: NlMethod,

    /// Number of calls to G(u(s), λ(s)) function
    pub n_function: usize,

    /// Number of Jacobian matrix (Gu) evaluations
    pub n_jacobian: usize,

    /// Number of factorizations
    pub n_factor: usize,

    /// Number of linear system solutions
    pub n_lin_sol: usize,

    /// Collects the number of steps, successful or not
    pub n_steps: usize,

    /// Collects the number of accepted steps
    pub n_accepted: usize,

    /// Collects the number of rejected steps
    pub n_rejected: usize,

    /// Number of large max(‖δu‖∞,|δλ|)
    pub n_large_du_dl: usize,

    /// Last number of iterations
    pub n_iterations: usize,

    /// Max number of iterations among all steps
    pub n_iterations_max: usize,

    /// Last accepted/suggested step size h_new
    pub h_accepted: f64,

    /// Max nanoseconds spent on steps
    pub nanos_step_max: u128,

    /// Max nanoseconds spent on Jacobian evaluation
    pub nanos_jacobian_max: u128,

    /// Max nanoseconds spent on the coefficient matrix factorization
    pub nanos_factor_max: u128,

    /// Max nanoseconds spent on the solution of the linear system
    pub nanos_lin_sol_max: u128,

    /// Total nanoseconds spent on the solution
    pub nanos_total: u128,

    /// Holds a stopwatch for measuring the elapsed time during a step
    pub(crate) sw_step: Stopwatch,

    /// Holds a stopwatch for measuring the elapsed time during the Jacobian computation
    pub(crate) sw_jacobian: Stopwatch,

    /// Holds a stopwatch for measuring the elapsed time during the coefficient matrix factorization
    pub(crate) sw_factor: Stopwatch,

    /// Holds a stopwatch for measuring the elapsed time during the solution of the linear system
    pub(crate) sw_lin_sol: Stopwatch,

    /// Holds a stopwatch for measuring the total elapsed time
    pub(crate) sw_total: Stopwatch,
}

impl Stats {
    /// Allocates a new instance
    pub fn new(method: NlMethod) -> Self {
        Stats {
            method,
            n_function: 0,
            n_jacobian: 0,
            n_factor: 0,
            n_lin_sol: 0,
            n_steps: 0,
            n_accepted: 0,
            n_rejected: 0,
            n_large_du_dl: 0,
            n_iterations: 0,
            n_iterations_max: 0,
            h_accepted: 0.0,
            nanos_step_max: 0,
            nanos_jacobian_max: 0,
            nanos_factor_max: 0,
            nanos_lin_sol_max: 0,
            nanos_total: 0,
            sw_step: Stopwatch::new(),
            sw_jacobian: Stopwatch::new(),
            sw_factor: Stopwatch::new(),
            sw_lin_sol: Stopwatch::new(),
            sw_total: Stopwatch::new(),
        }
    }

    /// Resets all values
    pub(crate) fn reset(&mut self, h: f64) {
        self.n_function = 0;
        self.n_jacobian = 0;
        self.n_factor = 0;
        self.n_lin_sol = 0;
        self.n_steps = 0;
        self.n_accepted = 0;
        self.n_rejected = 0;
        self.n_large_du_dl = 0;
        self.n_iterations = 0;
        self.n_iterations_max = 0;
        self.h_accepted = h;
        self.nanos_step_max = 0;
        self.nanos_jacobian_max = 0;
        self.nanos_factor_max = 0;
        self.nanos_lin_sol_max = 0;
        self.nanos_total = 0;
    }

    /// Stops the stopwatch and updates step nanoseconds
    pub(crate) fn stop_sw_step(&mut self) {
        let nanos = self.sw_step.stop();
        if nanos > self.nanos_step_max {
            self.nanos_step_max = nanos;
        }
    }

    /// Stops the stopwatch and updates Jacobian nanoseconds
    pub(crate) fn stop_sw_jacobian(&mut self) {
        let nanos = self.sw_jacobian.stop();
        if nanos > self.nanos_jacobian_max {
            self.nanos_jacobian_max = nanos;
        }
    }

    /// Stops the stopwatch and updates factor nanoseconds
    pub(crate) fn stop_sw_factor(&mut self) {
        let nanos = self.sw_factor.stop();
        if nanos > self.nanos_factor_max {
            self.nanos_factor_max = nanos;
        }
    }

    /// Stops the stopwatch and updates lin_sol nanoseconds
    pub(crate) fn stop_sw_lin_sol(&mut self) {
        let nanos = self.sw_lin_sol.stop();
        if nanos > self.nanos_lin_sol_max {
            self.nanos_lin_sol_max = nanos;
        }
    }

    /// Stops the stopwatch and updates total nanoseconds
    pub(crate) fn stop_sw_total(&mut self) {
        let nanos = self.sw_total.stop();
        self.nanos_total = nanos;
    }

    /// Stops the stopwatch and updates n_iterations_max nanoseconds
    pub(crate) fn update_n_iterations_max(&mut self) {
        if self.n_iterations > self.n_iterations_max {
            self.n_iterations_max = self.n_iterations;
        }
    }

    /// Returns a pretty formatted string with the stats
    pub fn summary(&self) -> String {
        let mut buffer = String::new();
        write!(
            &mut buffer,
            "{:?}: {}\n\
             Number of function evaluations   = {}\n\
             Number of Jacobian evaluations   = {}\n\
             Number of factorizations         = {}\n\
             Number of lin sys solutions      = {}\n\
             Number of performed steps        = {}\n\
             Number of accepted steps         = {}\n\
             Number of rejected steps         = {}\n\
             Number of iterations (maximum)   = {}",
            self.method,
            self.method.description(),
            self.n_function,
            self.n_jacobian,
            self.n_factor,
            self.n_lin_sol,
            self.n_steps,
            self.n_accepted,
            self.n_rejected,
            self.n_iterations_max,
        )
        .unwrap();
        buffer
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\n\
             Number of iterations (last step) = {}\n\
             Last accepted/suggested stepsize = {}\n\
             Max time spent on a step         = {}\n\
             Max time spent on the Jacobian   = {}\n\
             Max time spent on factorization  = {}\n\
             Max time spent on lin solution   = {}\n\
             Total time                       = {}",
            self.summary(),
            self.n_iterations,
            self.h_accepted,
            format_nanoseconds(self.nanos_step_max),
            format_nanoseconds(self.nanos_jacobian_max),
            format_nanoseconds(self.nanos_factor_max),
            format_nanoseconds(self.nanos_lin_sol_max),
            format_nanoseconds(self.nanos_total),
        )
        .unwrap();
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Stats;
    use crate::NlMethod;

    #[test]
    fn clone_copy_and_debug_work() {
        let mut stats = Stats::new(NlMethod::Arclength);
        stats.n_accepted += 1;
        let copy = stats;
        let clone = stats.clone();
        assert_eq!(copy.n_accepted, stats.n_accepted);
        assert_eq!(clone.n_accepted, stats.n_accepted);
        assert!(format!("{:?}", stats).len() > 0);
    }

    #[test]
    fn summary_works() {
        let stats = Stats::new(NlMethod::Arclength);
        println!("{}", stats.summary());
        assert_eq!(
            format!("{}", stats.summary()),
            "Arclength: Pseudo-arclength continuation; solves G(u) = 0\n\
             Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of factorizations         = 0\n\
             Number of lin sys solutions      = 0\n\
             Number of performed steps        = 0\n\
             Number of accepted steps         = 0\n\
             Number of rejected steps         = 0\n\
             Number of iterations (maximum)   = 0"
        );
    }

    #[test]
    fn display_works() {
        let stats = Stats::new(NlMethod::Arclength);
        assert_eq!(
            format!("{}", stats),
            "Arclength: Pseudo-arclength continuation; solves G(u) = 0\n\
             Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of factorizations         = 0\n\
             Number of lin sys solutions      = 0\n\
             Number of performed steps        = 0\n\
             Number of accepted steps         = 0\n\
             Number of rejected steps         = 0\n\
             Number of iterations (maximum)   = 0\n\
             Number of iterations (last step) = 0\n\
             Last accepted/suggested stepsize = 0\n\
             Max time spent on a step         = 0ns\n\
             Max time spent on the Jacobian   = 0ns\n\
             Max time spent on factorization  = 0ns\n\
             Max time spent on lin solution   = 0ns\n\
             Total time                       = 0ns"
        );
    }
}
