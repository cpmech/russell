use super::Method;
use russell_lab::{format_nanoseconds, Stopwatch};
use russell_stat::{statistics, Histogram};
use std::fmt::{self, Write};

/// Holds statistics and benchmarking data
#[derive(Clone, Debug)]
pub struct Stats {
    /// Holds the method
    method: Method,

    /// Hide timings when displaying statistics
    hide_timings: bool,

    /// Indicates whether to record the iterations residuals or not
    ///
    /// Will populate `iterations_errors` with the iteration residuals
    record_iterations_residuals: bool,

    /// Indicates automatic stepsize adjustment
    auto: bool,

    /// Holds the iterations residuals for the current step, temporarily
    temporary_iterations_residuals: Vec<f64>,

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
    pub n_large_delta: usize,

    /// Number of times that maximum iterations have been reached
    pub n_max_iterations_reached: usize,

    /// Number of iterations with continued divergence
    pub n_continued_divergence: usize,

    /// Total number of iterations
    pub n_iteration_total: usize,

    /// Max number of iterations among all steps
    pub n_iteration_max: usize,

    /// Holds the iteration residuals
    ///
    /// Each element is a vector with the iteration residuals for each step.
    ///
    /// Only used if `record_iterations_residuals` is true.
    ///
    /// `(nstep, n_iteration_in_step)`
    pub iterations_residuals: Option<Vec<Vec<f64>>>,

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
    pub fn new(method: Method, hide_timings: bool, record_iterations_residuals: bool) -> Self {
        Stats {
            method,
            hide_timings,
            record_iterations_residuals,
            auto: false,
            temporary_iterations_residuals: Vec::new(),
            n_function: 0,
            n_jacobian: 0,
            n_factor: 0,
            n_lin_sol: 0,
            n_steps: 0,
            n_accepted: 0,
            n_rejected: 0,
            n_large_delta: 0,
            n_max_iterations_reached: 0,
            n_continued_divergence: 0,
            n_iteration_total: 0,
            n_iteration_max: 0,
            iterations_residuals: None,
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
    pub(crate) fn reset(&mut self, auto: bool) {
        self.auto = auto;
        self.n_function = 0;
        self.n_jacobian = 0;
        self.n_factor = 0;
        self.n_lin_sol = 0;
        self.n_steps = 0;
        self.n_accepted = 0;
        self.n_rejected = 0;
        self.n_large_delta = 0;
        self.n_max_iterations_reached = 0;
        self.n_continued_divergence = 0;
        self.n_iteration_total = 0;
        self.n_iteration_max = 0;
        self.iterations_residuals = None;
        self.h_accepted = 0.0;
        self.nanos_step_max = 0;
        self.nanos_jacobian_max = 0;
        self.nanos_factor_max = 0;
        self.nanos_lin_sol_max = 0;
        self.nanos_total = 0;
    }

    /// Starts the recording of iterations residuals
    pub(crate) fn record_iterations_residuals_start(&mut self) {
        if self.record_iterations_residuals {
            self.temporary_iterations_residuals.clear();
            if self.iterations_residuals.is_none() {
                self.iterations_residuals = Some(Vec::new());
            }
        }
    }

    /// Appends the current iteration residual to the current step
    pub(crate) fn record_iterations_residuals_append(&mut self, residual_max: f64) {
        if self.record_iterations_residuals {
            self.temporary_iterations_residuals.push(residual_max);
        }
    }

    /// Stops the recording of iterations residuals
    pub(crate) fn record_iterations_residuals_stop(&mut self, converged: bool) {
        if self.record_iterations_residuals {
            if converged {
                if let Some(residuals) = &mut self.iterations_residuals {
                    residuals.push(self.temporary_iterations_residuals.clone());
                }
            }
        }
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

    /// Returns a pretty formatted string with the stats
    pub fn summary(&self) -> String {
        let mut buffer = String::new();
        write!(
            &mut buffer,
            "{} ({})\n\
             Number of function evaluations   = {}\n\
             Number of Jacobian evaluations   = {}\n\
             Number of factorizations         = {}\n\
             Number of lin sys solutions      = {}\n\
             Number of performed steps        = {}\n\
             Number of accepted steps         = {}\n\
             Number of rejected steps         = {}\n\
             Number of large max(‖δu‖∞,|δλ|)  = {}\n\
             Number of max iterations reached = {}\n\
             Number of continued divergence   = {}\n\
             Number of iterations (maximum)   = {}\n\
             Number of iterations (total)     = {}\n\
             Last accepted/suggested stepsize = {}",
            self.method.description(),
            if self.auto { "auto steps" } else { "fixed steps" },
            self.n_function,
            self.n_jacobian,
            self.n_factor,
            self.n_lin_sol,
            self.n_steps,
            self.n_accepted,
            self.n_rejected,
            self.n_large_delta,
            self.n_max_iterations_reached,
            self.n_continued_divergence,
            self.n_iteration_max,
            self.n_iteration_total,
            self.h_accepted,
        )
        .unwrap();
        if self.record_iterations_residuals {
            if let Some(residuals) = &self.iterations_residuals {
                let n_iter_data = residuals.iter().map(|res| res.len()).collect::<Vec<usize>>();
                let stations = (0..11).collect::<Vec<usize>>();
                let mut histogram = Histogram::new(&stations).unwrap();
                histogram.set_bar_char('*').set_bar_max_len(40);
                histogram.count(&n_iter_data);
                write!(
                    &mut buffer,
                    "\nDistribution of the number of converged iterations across all steps:\n",
                )
                .unwrap();
                write!(&mut buffer, "{}", histogram).unwrap();
                let mut conv_ratio_values = Vec::new();
                for err in residuals {
                    if err.len() >= 3 {
                        let l = err.len();
                        let rate = f64::ln(err[l - 1] / err[l - 2]) / f64::ln(err[l - 2] / err[l - 3]);
                        if rate.is_finite() && rate >= 0.9 && rate <= 2.1 {
                            conv_ratio_values.push(rate);
                        }
                    }
                }
                let res = statistics(&conv_ratio_values);
                write!(
                    &mut buffer,
                    "Convergence: mean = {:.3}, std_dev = {:.3}, min_max = ({:.3}, {:.3})",
                    res.mean, res.std_dev, res.min, res.max
                )
                .unwrap();
            }
        }
        buffer
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.hide_timings {
            write!(f, "{}", self.summary()).unwrap();
        } else {
            write!(
                f,
                "{}\n\
                 Max time spent on a step         = {}\n\
                 Max time spent on the Jacobian   = {}\n\
                 Max time spent on factorization  = {}\n\
                 Max time spent on lin solution   = {}\n\
                 Total time                       = {}",
                self.summary(),
                format_nanoseconds(self.nanos_step_max),
                format_nanoseconds(self.nanos_jacobian_max),
                format_nanoseconds(self.nanos_factor_max),
                format_nanoseconds(self.nanos_lin_sol_max),
                format_nanoseconds(self.nanos_total),
            )
            .unwrap();
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Stats;
    use crate::Method;

    #[test]
    fn clone_copy_and_debug_work() {
        let mut stats = Stats::new(Method::Arclength, false, false);
        stats.n_accepted += 1;
        let clone = stats.clone();
        assert_eq!(clone.n_accepted, stats.n_accepted);
        assert!(format!("{:?}", stats).len() > 0);
    }

    #[test]
    fn summary_works() {
        let stats = Stats::new(Method::Arclength, false, false);
        println!("{}", stats.summary());
        assert_eq!(
            format!("{}", stats.summary()),
            "Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0 (fixed steps)\n\
             Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of factorizations         = 0\n\
             Number of lin sys solutions      = 0\n\
             Number of performed steps        = 0\n\
             Number of accepted steps         = 0\n\
             Number of rejected steps         = 0\n\
             Number of large max(‖δu‖∞,|δλ|)  = 0\n\
             Number of max iterations reached = 0\n\
             Number of continued divergence   = 0\n\
             Number of iterations (maximum)   = 0\n\
             Number of iterations (total)     = 0\n\
             Last accepted/suggested stepsize = 0"
        );
    }

    #[test]
    fn display_works() {
        let stats = Stats::new(Method::Natural, false, false);
        assert_eq!(
            format!("{}", stats),
            "Natural parameter continuation; solves G(u, λ) = 0 (fixed steps)\n\
             Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of factorizations         = 0\n\
             Number of lin sys solutions      = 0\n\
             Number of performed steps        = 0\n\
             Number of accepted steps         = 0\n\
             Number of rejected steps         = 0\n\
             Number of large max(‖δu‖∞,|δλ|)  = 0\n\
             Number of max iterations reached = 0\n\
             Number of continued divergence   = 0\n\
             Number of iterations (maximum)   = 0\n\
             Number of iterations (total)     = 0\n\
             Last accepted/suggested stepsize = 0\n\
             Max time spent on a step         = 0ns\n\
             Max time spent on the Jacobian   = 0ns\n\
             Max time spent on factorization  = 0ns\n\
             Max time spent on lin solution   = 0ns\n\
             Total time                       = 0ns"
        );

        let stats = Stats::new(Method::Natural, true, false);
        assert_eq!(
            format!("{}", stats),
            "Natural parameter continuation; solves G(u, λ) = 0 (fixed steps)\n\
             Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of factorizations         = 0\n\
             Number of lin sys solutions      = 0\n\
             Number of performed steps        = 0\n\
             Number of accepted steps         = 0\n\
             Number of rejected steps         = 0\n\
             Number of large max(‖δu‖∞,|δλ|)  = 0\n\
             Number of max iterations reached = 0\n\
             Number of continued divergence   = 0\n\
             Number of iterations (maximum)   = 0\n\
             Number of iterations (total)     = 0\n\
             Last accepted/suggested stepsize = 0"
        );
    }
}
