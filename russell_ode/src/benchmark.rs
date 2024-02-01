use russell_lab::{format_nanoseconds, Stopwatch};
use std::fmt;

/// Holds benchmark information
#[derive(Clone, Copy, Debug)]
pub struct Benchmark {
    /// Number of calls to ODE system function
    pub n_function_eval: usize,

    /// Number of Jacobian matrix evaluations
    pub n_jacobian_eval: usize,

    /// Collects the number of steps, successful or not
    pub n_performed_steps: usize,

    /// Collects the number of accepted steps
    pub n_accepted_steps: usize,

    /// Collects the number of rejected steps
    pub n_rejected_steps: usize,

    /// Last number of iterations
    pub n_iterations_last: usize,

    /// Max number of iterations among all steps
    pub n_iterations_max: usize,

    /// Optimal step size at the end
    pub h_optimal: f64,

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

impl Benchmark {
    /// Allocates a new instance
    pub fn new() -> Self {
        Benchmark {
            n_function_eval: 0,
            n_jacobian_eval: 0,
            n_performed_steps: 0,
            n_accepted_steps: 0,
            n_rejected_steps: 0,
            n_iterations_last: 0,
            n_iterations_max: 0,
            h_optimal: 0.0,
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
    pub(crate) fn reset(&mut self) {
        self.n_function_eval = 0;
        self.n_jacobian_eval = 0;
        self.n_performed_steps = 0;
        self.n_accepted_steps = 0;
        self.n_rejected_steps = 0;
        self.n_iterations_last = 0;
        self.n_iterations_max = 0;
        self.h_optimal = 0.0;
        self.nanos_step_max = 0;
        self.nanos_jacobian_max = 0;
        self.nanos_factor_max = 0;
        self.nanos_lin_sol_max = 0;
        self.nanos_total = 0;
    }

    pub(crate) fn stop_sw_step(&mut self) {
        let nanos = self.sw_step.stop();
        if nanos > self.nanos_step_max {
            self.nanos_step_max = nanos;
        }
    }

    pub(crate) fn stop_sw_jacobian(&mut self) {
        let nanos = self.sw_jacobian.stop();
        if nanos > self.nanos_jacobian_max {
            self.nanos_jacobian_max = nanos;
        }
    }

    pub(crate) fn stop_sw_factor(&mut self) {
        let nanos = self.sw_factor.stop();
        if nanos > self.nanos_factor_max {
            self.nanos_factor_max = nanos;
        }
    }

    pub(crate) fn stop_sw_lin_sol(&mut self) {
        let nanos = self.sw_lin_sol.stop();
        if nanos > self.nanos_lin_sol_max {
            self.nanos_lin_sol_max = nanos;
        }
    }

    pub(crate) fn stop_sw_total(&mut self) {
        let nanos = self.sw_total.stop();
        self.nanos_total = nanos;
    }

    pub(crate) fn update_n_iterations_max(&mut self) {
        if self.n_iterations_last > self.n_iterations_max {
            self.n_iterations_max = self.n_iterations_last;
        }
    }
}

impl fmt::Display for Benchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Number of function evaluations   = {}\n\
             Number of Jacobian evaluations   = {}\n\
             Number of performed steps        = {}\n\
             Number of accepted steps         = {}\n\
             Number of rejected steps         = {}\n\
             Number of iterations (last step) = {}\n\
             Number of iterations (maximum)   = {}\n\
             Optimal stepsize (h)             = {}\n\
             Max time spent on a step         = {}\n\
             Max time spent on the Jacobian   = {}\n\
             Max time spent on factorization  = {}\n\
             Max time spent on lin solution   = {}\n\
             Total time                       = {}\n",
            self.n_function_eval,
            self.n_jacobian_eval,
            self.n_performed_steps,
            self.n_accepted_steps,
            self.n_rejected_steps,
            self.n_iterations_last,
            self.n_iterations_max,
            self.h_optimal,
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
    use super::Benchmark;

    #[test]
    fn clone_copy_and_debug_work() {
        let mut bench = Benchmark::new();
        bench.n_accepted_steps += 1;
        let copy = bench;
        let clone = bench.clone();
        assert_eq!(copy.n_accepted_steps, bench.n_accepted_steps);
        assert_eq!(clone.n_accepted_steps, bench.n_accepted_steps);
        assert!(format!("{:?}", bench).len() > 0);
    }

    #[test]
    fn display_works() {
        let bench = Benchmark::new();
        assert_eq!(
            format!("{}", bench),
            "Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of performed steps        = 0\n\
             Number of accepted steps         = 0\n\
             Number of rejected steps         = 0\n\
             Number of iterations (last step) = 0\n\
             Number of iterations (maximum)   = 0\n\
             Optimal stepsize (h)             = 0\n\
             Max time spent on a step         = 0ns\n\
             Max time spent on the Jacobian   = 0ns\n\
             Max time spent on factorization  = 0ns\n\
             Max time spent on lin solution   = 0ns\n\
             Total time                       = 0ns\n"
        );
    }
}
