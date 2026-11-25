use super::{Config, Method, Status};
use russell_lab::{Stopwatch, format_nanoseconds};
use russell_stat::{Histogram, Statistics};
use std::fmt::{self, Write};

/// Records summary metrics for a single step attempt
#[derive(Clone, Debug)]
pub struct StepSummary {
    /// 1-based step attempt index (`Stats::n_steps`)
    pub step_index: usize,

    /// Whether this attempt was accepted
    pub accepted: bool,

    /// Outcome status of this attempt
    pub status: Status,

    /// Stepsize used on this attempt
    pub h_used: f64,

    /// Next stepsize after adaptation or cutback
    pub h_next: f64,

    /// Tangent or secant relative change `rdiff` (`0.0` on failed attempts)
    pub rdiff: f64,

    /// Number of Newton-Raphson iterations evaluated in this attempt
    pub n_iteration: usize,

    /// Maximum residual norm at iteration 0 (`NaN` if the predictor failed before iteration 0)
    pub residual_0: f64,

    /// Maximum residual norm at the last evaluated iteration (`NaN` if none)
    pub residual_last: f64,

    /// Continuation parameter `λ` after the attempt
    pub lambda: f64,
}

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

    /// Indicates whether to record per-attempt step summaries or not
    record_step_summary: bool,

    /// Indicates automatic stepsize adjustment
    auto: bool,

    /// Indicates the use of numerical Jacobian (set externally by the user)
    pub num_jacobian: bool,

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

    /// Total number of iterations
    pub n_iteration_total: usize,

    /// Holds the iteration residuals
    ///
    /// Each element is a vector with the iteration residuals for each step.
    ///
    /// Only used if `record_iterations_residuals` is true.
    ///
    /// `(nstep, n_iteration_in_step)`
    pub iterations_residuals: Option<Vec<Vec<f64>>>,

    /// Last accepted stepsize h_used
    pub h_accepted: f64,

    /// Next suggested stepsize h_next
    pub h_next: f64,

    /// Number of step attempts failing with `Status::LargeDelta`
    pub n_reject_large_delta: usize,

    /// Number of step attempts failing with `Status::ContinuedResidualDivergence`
    pub n_reject_res_div: usize,

    /// Number of step attempts failing with `Status::ContinuedDeltaDivergence`
    pub n_reject_delta_div: usize,

    /// Number of step attempts failing with `Status::ReachedMaxIterations`
    pub n_reject_max_iter: usize,

    /// Number of step attempts failing with `Status::BorderingSmallDenominator`
    pub n_reject_small_denom: usize,

    /// Number of step attempts failing with `Status::SecondaryUpdateError`
    pub n_reject_secondary: usize,

    /// Number of step attempts failing with `Status::UnmetStopCriterion`
    pub n_reject_unmet_stop: usize,

    /// Number of step attempts failing with any other failure status
    pub n_reject_other: usize,

    /// Most recent rejection or failure status
    pub last_rejection_status: Option<Status>,

    /// Stepsize used on the most recent step attempt
    pub step_h_used: f64,

    /// Next stepsize computed after the most recent step attempt
    pub step_h_next: f64,

    /// Relative tangent/secant difference on the most recent step attempt
    pub step_rdiff: f64,

    /// Number of iterations on the most recent step attempt
    pub step_n_iteration: usize,

    /// Initial residual norm on the most recent step attempt
    pub step_residual_0: f64,

    /// Final residual norm on the most recent step attempt
    pub step_residual_last: f64,

    /// Optional per-attempt summary history (populated when `Config::set_record_step_summary(true)` is set)
    pub step_summaries: Option<Vec<StepSummary>>,

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
    pub fn new(config: &Config) -> Self {
        Stats {
            method: config.method,
            hide_timings: config.hide_timings,
            record_iterations_residuals: config.record_iterations_residuals,
            record_step_summary: config.record_step_summary,
            auto: false,
            num_jacobian: false,
            temporary_iterations_residuals: Vec::new(),
            n_function: 0,
            n_jacobian: 0,
            n_factor: 0,
            n_lin_sol: 0,
            n_steps: 0,
            n_accepted: 0,
            n_rejected: 0,
            n_iteration_total: 0,
            iterations_residuals: None,
            h_accepted: 0.0,
            h_next: 0.0,
            n_reject_large_delta: 0,
            n_reject_res_div: 0,
            n_reject_delta_div: 0,
            n_reject_max_iter: 0,
            n_reject_small_denom: 0,
            n_reject_secondary: 0,
            n_reject_unmet_stop: 0,
            n_reject_other: 0,
            last_rejection_status: None,
            step_h_used: 0.0,
            step_h_next: 0.0,
            step_rdiff: 0.0,
            step_n_iteration: 0,
            step_residual_0: 0.0,
            step_residual_last: 0.0,
            step_summaries: None,
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

    /// Returns the convergence rates across all steps
    ///
    /// The convergence rate is calculated for the last three results and returned only
    /// if the rate is within the lower and upper limits, inclusive.
    ///
    /// Note: This is only available if `record_iterations_residuals` is enabled
    ///
    /// # Arguments
    ///
    /// * `lower_limit` - The lower limit for the convergence rates; e.g. 0.9
    /// * `upper_limit` - The upper limit for the convergence rates; e.g. 2.1
    pub fn get_convergence_rates(&self, lower_limit: f64, upper_limit: f64) -> Vec<f64> {
        let mut conv_ratio_values = Vec::new();
        if self.record_iterations_residuals {
            if let Some(residuals) = &self.iterations_residuals {
                for err in residuals {
                    if err.len() >= 3 {
                        let l = err.len();
                        let rate = f64::ln(err[l - 1] / err[l - 2]) / f64::ln(err[l - 2] / err[l - 3]);
                        if rate.is_finite() && rate >= lower_limit && rate <= upper_limit {
                            conv_ratio_values.push(rate);
                        }
                    }
                }
            }
        }
        conv_ratio_values
    }

    /// Returns a histogram of the number of iterations across all steps
    ///
    /// Note: This is only available if `record_iterations_residuals` is enabled
    ///
    /// # Arguments
    ///
    /// * `niter_upper` - Upper bound on the number of iterations (in the stations array); e.g., 13
    /// * `character` - The character to use for the histogram bars
    /// * `bar_len` - The maximum length of the histogram bars; e.g., 40
    pub fn get_histogram_of_iterations(&self, niter_upper: usize, character: char, bar_len: usize) -> Histogram<usize> {
        if self.record_iterations_residuals {
            if let Some(residuals) = &self.iterations_residuals {
                let n_iter_data: Vec<usize> = residuals
                    .iter()
                    .map(|res| {
                        let n = res.len();
                        if n >= 2 {
                            n - 1 // exclude the last residual because it should be near zero for converged steps
                        } else {
                            n
                        }
                    })
                    .collect();
                let stations = (0..niter_upper).collect::<Vec<usize>>();
                let mut histogram = Histogram::new(&stations).unwrap();
                histogram.set_bar_char(character).set_bar_max_len(bar_len);
                histogram.count(&n_iter_data);
                return histogram;
            }
        }
        Histogram::new(&[0, 1]).unwrap() // empty histogram
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
        self.n_iteration_total = 0;
        self.iterations_residuals = None;
        self.h_accepted = 0.0;
        self.h_next = 0.0;
        self.n_reject_large_delta = 0;
        self.n_reject_res_div = 0;
        self.n_reject_delta_div = 0;
        self.n_reject_max_iter = 0;
        self.n_reject_small_denom = 0;
        self.n_reject_secondary = 0;
        self.n_reject_unmet_stop = 0;
        self.n_reject_other = 0;
        self.last_rejection_status = None;
        self.step_h_used = 0.0;
        self.step_h_next = 0.0;
        self.step_rdiff = 0.0;
        self.step_n_iteration = 0;
        self.step_residual_0 = 0.0;
        self.step_residual_last = 0.0;
        self.step_summaries = None;
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

    /// Records a failed or rejected step status into the per-status counters
    pub(crate) fn record_rejection_status(&mut self, status: &Status) {
        match status {
            Status::LargeDelta => self.n_reject_large_delta += 1,
            Status::ContinuedResidualDivergence => self.n_reject_res_div += 1,
            Status::ContinuedDeltaDivergence => self.n_reject_delta_div += 1,
            Status::ReachedMaxIterations => self.n_reject_max_iter += 1,
            Status::BorderingSmallDenominator => self.n_reject_small_denom += 1,
            Status::SecondaryUpdateError(_) => self.n_reject_secondary += 1,
            Status::UnmetStopCriterion => self.n_reject_unmet_stop += 1,
            _ => self.n_reject_other += 1,
        }
        self.last_rejection_status = Some(status.clone());
    }

    /// Records the last-step telemetry scalars and optionally appends a [`StepSummary`]
    pub(crate) fn record_step_attempt(&mut self, summary: StepSummary) {
        self.step_h_used = summary.h_used;
        self.step_h_next = summary.h_next;
        self.step_rdiff = summary.rdiff;
        self.step_n_iteration = summary.n_iteration;
        self.step_residual_0 = summary.residual_0;
        self.step_residual_last = summary.residual_last;
        if self.record_step_summary {
            self.step_summaries.get_or_insert_with(Vec::new).push(summary);
        }
    }

    /// Returns a compact summary of step rejections by status
    pub fn rejection_summary(&self) -> String {
        let mut parts = Vec::new();
        if self.n_reject_large_delta > 0 {
            parts.push(format!("LargeDelta={}", self.n_reject_large_delta));
        }
        if self.n_reject_res_div > 0 {
            parts.push(format!("ResDiv={}", self.n_reject_res_div));
        }
        if self.n_reject_delta_div > 0 {
            parts.push(format!("DeltaDiv={}", self.n_reject_delta_div));
        }
        if self.n_reject_max_iter > 0 {
            parts.push(format!("MaxIter={}", self.n_reject_max_iter));
        }
        if self.n_reject_small_denom > 0 {
            parts.push(format!("SmallDenom={}", self.n_reject_small_denom));
        }
        if self.n_reject_secondary > 0 {
            parts.push(format!("Secondary={}", self.n_reject_secondary));
        }
        if self.n_reject_unmet_stop > 0 {
            parts.push(format!("UnmetStop={}", self.n_reject_unmet_stop));
        }
        if self.n_reject_other > 0 {
            parts.push(format!("Other={}", self.n_reject_other));
        }
        let breakdown = if parts.is_empty() { "none".to_string() } else { parts.join(", ") };
        match &self.last_rejection_status {
            Some(status) => format!("n_rejected={}, [{}], last={:?}", self.n_rejected, breakdown, status),
            None => format!("n_rejected={}, [{}]", self.n_rejected, breakdown),
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
    ///
    /// The returned string includes the continuation method, numerical Jacobian flag,
    /// counts of function/Jacobian evaluations, factorizations, linear solves, accepted
    /// and rejected steps, total iterations, and the last accepted stepsize. If iteration
    /// residuals are recorded, it also includes convergence rate statistics and a histogram
    /// of iteration counts.
    pub fn summary(&self) -> String {
        // Length of the histogram bars
        const BAR_LEN: usize = 53;

        // Convergence rates statistics
        let rates_stats = if !self.record_iterations_residuals {
            String::new()
        } else {
            let rates = self.get_convergence_rates(0.9, 2.1);
            if rates.is_empty() {
                String::new()
            } else {
                let res = Statistics::new(&rates);
                format!(
                    "\n\nConvergence rates: (min, max) = ({:.3}, {:.3})\n(0.9 ≤ cr ≤ 2.1)       (μ, σ) = ({:.3}, {:.3})",
                    res.min, res.max, res.mean, res.std_dev
                )
            }
        };

        // Histogram of the number of iterations
        let niter_hist = if !self.record_iterations_residuals {
            String::new()
        } else {
            let hist = self.get_histogram_of_iterations(13, '■', BAR_LEN);
            format!("\n\nDistribution of the number of converged iterations across all steps:\n{}", hist)
        };

        // Write summary to buffer
        let mut buffer = String::new();
        write!(
            &mut buffer,
            "{} ({})\n\
             Using numerical Jacobian         = {}\n\
             Number of function evaluations   = {}\n\
             Number of Jacobian evaluations   = {}\n\
             Number of factorizations         = {}\n\
             Number of lin sys solutions      = {}\n\
             Number of accepted steps         = {}\n\
             Number of rejected steps         = {}\n\
             Number of performed steps        = {}\n\
             Number of iterations (total)     = {}\n\
             Last accepted/suggested stepsize = {}{}{}",
            self.method.description(),
            if self.auto { "auto" } else { "fixed" },
            self.num_jacobian,
            self.n_function,
            self.n_jacobian,
            self.n_factor,
            self.n_lin_sol,
            self.n_accepted,
            self.n_rejected,
            self.n_steps,
            self.n_iteration_total,
            self.h_accepted,
            rates_stats,
            niter_hist
        )
        .unwrap();
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
    use super::{Stats, StepSummary};
    use crate::{Config, Method, Status};
    use russell_lab::array_approx_eq;

    #[test]
    fn clone_copy_and_debug_work() {
        let mut config = Config::new();
        config.set_method(Method::Arclength);
        let mut stats = Stats::new(&config);
        stats.n_accepted += 1;
        let clone = stats.clone();
        assert_eq!(clone.n_accepted, stats.n_accepted);
        assert!(!format!("{:?}", stats).is_empty());
    }

    #[test]
    fn convergence_rate_works() {
        // check empty vector due to false flag
        let mut config = Config::new();
        config.set_method(Method::Arclength);
        let stats = Stats::new(&config);
        assert!(stats.get_convergence_rates(0.0, 10.0).is_empty());
        // check empty vector due to no values added
        config.record_iterations_residuals = true;
        let mut stats = Stats::new(&config);
        assert!(stats.get_convergence_rates(0.0, 10.0).is_empty());
        // first step
        stats.record_iterations_residuals_start();
        stats.record_iterations_residuals_append(1.0 / 2.0);
        stats.record_iterations_residuals_append(1.0 / 4.0);
        stats.record_iterations_residuals_append(1.0 / 16.0);
        stats.record_iterations_residuals_stop(true);
        // second step
        stats.record_iterations_residuals_start();
        stats.record_iterations_residuals_append(1.0e-2);
        stats.record_iterations_residuals_append(1.0e-4);
        stats.record_iterations_residuals_append(1.0e-8);
        stats.record_iterations_residuals_stop(true);
        let rates = stats.get_convergence_rates(0.0, 10.0);
        array_approx_eq(&rates, &[2.0, 2.0], 1e-15);
    }

    #[test]
    fn histogram_works() {
        let mut config = Config::new();
        config.set_method(Method::Arclength);
        // check empty string due to false flag
        let stats = Stats::new(&config);
        assert_eq!(stats.get_histogram_of_iterations(13, '*', 40).get_counts(), &[0]);
        // check empty string due to no values added
        config.record_iterations_residuals = true;
        let mut stats = Stats::new(&config);
        assert_eq!(stats.get_histogram_of_iterations(13, '*', 40).get_counts(), &[0]);
        // first step
        stats.record_iterations_residuals_start();
        stats.record_iterations_residuals_append(1.0 / 2.0);
        stats.record_iterations_residuals_append(1.0 / 4.0);
        stats.record_iterations_residuals_append(1.0 / 16.0);
        stats.record_iterations_residuals_stop(true);
        // second step
        stats.record_iterations_residuals_start();
        stats.record_iterations_residuals_append(1.0e-2);
        stats.record_iterations_residuals_append(1.0e-4);
        stats.record_iterations_residuals_append(1.0e-8);
        stats.record_iterations_residuals_stop(true);
        let hist = stats.get_histogram_of_iterations(13, '■', 40);
        assert_eq!(
            format!("{}", hist),
            "[ 0,  1) | 0 \n\
             [ 1,  2) | 0 \n\
             [ 2,  3) | 2 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n\
             [ 3,  4) | 0 \n\
             [ 4,  5) | 0 \n\
             [ 5,  6) | 0 \n\
             [ 6,  7) | 0 \n\
             [ 7,  8) | 0 \n\
             [ 8,  9) | 0 \n\
             [ 9, 10) | 0 \n\
             [10, 11) | 0 \n\
             [11, 12) | 0 \n\
             \x20\x20\x20\x20\x20sum = 2\n"
        );
    }

    #[test]
    fn summary_works_basic() {
        let mut config = Config::new();
        config.set_method(Method::Arclength);
        let mut stats = Stats::new(&config);
        stats.n_accepted = 8;
        stats.n_rejected = 2;
        stats.n_steps = 10;
        assert_eq!(
            format!("{}", stats.summary()),
            "Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0 (fixed)\n\
             Using numerical Jacobian         = false\n\
             Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of factorizations         = 0\n\
             Number of lin sys solutions      = 0\n\
             Number of accepted steps         = 8\n\
             Number of rejected steps         = 2\n\
             Number of performed steps        = 10\n\
             Number of iterations (total)     = 0\n\
             Last accepted/suggested stepsize = 0"
        );
    }

    #[test]
    fn summary_works_with_convergence_rate_info() {
        let mut config = Config::new();
        config.set_method(Method::Arclength);
        config.record_iterations_residuals = true;
        let mut stats = Stats::new(&config);
        stats.record_iterations_residuals_start();
        stats.record_iterations_residuals_append(1.0e-2);
        stats.record_iterations_residuals_append(1.0e-4);
        stats.record_iterations_residuals_append(1.0e-8);
        stats.record_iterations_residuals_stop(true);
        println!("{}", stats.summary());
        assert_eq!(
            format!("{}", stats.summary()),
            "Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0 (fixed)\n\
             Using numerical Jacobian         = false\n\
             Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of factorizations         = 0\n\
             Number of lin sys solutions      = 0\n\
             Number of accepted steps         = 0\n\
             Number of rejected steps         = 0\n\
             Number of performed steps        = 0\n\
             Number of iterations (total)     = 0\n\
             Last accepted/suggested stepsize = 0\n\
             \n\
             Convergence rates: (min, max) = (2.000, 2.000)\n\
             (0.9 ≤ cr ≤ 2.1)       (μ, σ) = (2.000, 0.000)\n\
             \n\
             Distribution of the number of converged iterations across all steps:\n\
             [ 0,  1) | 0 \n\
             [ 1,  2) | 0 \n\
             [ 2,  3) | 1 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n\
             [ 3,  4) | 0 \n\
             [ 4,  5) | 0 \n\
             [ 5,  6) | 0 \n\
             [ 6,  7) | 0 \n\
             [ 7,  8) | 0 \n\
             [ 8,  9) | 0 \n\
             [ 9, 10) | 0 \n\
             [10, 11) | 0 \n\
             [11, 12) | 0 \n\
             \x20\x20\x20\x20\x20sum = 1\n"
        );
    }

    #[test]
    fn display_works() {
        let mut config = Config::new();
        let stats = Stats::new(&config);
        assert_eq!(
            format!("{}", stats),
            "Natural parameter continuation; solves G(u, λ) = 0 (fixed)\n\
             Using numerical Jacobian         = false\n\
             Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of factorizations         = 0\n\
             Number of lin sys solutions      = 0\n\
             Number of accepted steps         = 0\n\
             Number of rejected steps         = 0\n\
             Number of performed steps        = 0\n\
             Number of iterations (total)     = 0\n\
             Last accepted/suggested stepsize = 0\n\
             Max time spent on a step         = 0ns\n\
             Max time spent on the Jacobian   = 0ns\n\
             Max time spent on factorization  = 0ns\n\
             Max time spent on lin solution   = 0ns\n\
             Total time                       = 0ns"
        );

        config.hide_timings = true;
        let stats = Stats::new(&config);
        assert_eq!(
            format!("{}", stats),
            "Natural parameter continuation; solves G(u, λ) = 0 (fixed)\n\
             Using numerical Jacobian         = false\n\
             Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of factorizations         = 0\n\
             Number of lin sys solutions      = 0\n\
             Number of accepted steps         = 0\n\
             Number of rejected steps         = 0\n\
             Number of performed steps        = 0\n\
             Number of iterations (total)     = 0\n\
             Last accepted/suggested stepsize = 0"
        );
    }

    #[test]
    fn rejection_telemetry_and_step_summary_work() {
        let mut config = Config::new();
        config.set_record_step_summary(true);
        let mut stats = Stats::new(&config);
        assert_eq!(stats.rejection_summary(), "n_rejected=0, [none]");

        stats.n_rejected = 8;
        stats.record_rejection_status(&Status::LargeDelta);
        stats.record_rejection_status(&Status::ContinuedResidualDivergence);
        stats.record_rejection_status(&Status::ContinuedDeltaDivergence);
        stats.record_rejection_status(&Status::ReachedMaxIterations);
        stats.record_rejection_status(&Status::BorderingSmallDenominator);
        stats.record_rejection_status(&Status::SecondaryUpdateError("sup"));
        stats.record_rejection_status(&Status::UnmetStopCriterion);
        stats.record_rejection_status(&Status::NanOrInfResidual);

        assert_eq!(stats.n_reject_large_delta, 1);
        assert_eq!(stats.n_reject_res_div, 1);
        assert_eq!(stats.n_reject_delta_div, 1);
        assert_eq!(stats.n_reject_max_iter, 1);
        assert_eq!(stats.n_reject_small_denom, 1);
        assert_eq!(stats.n_reject_secondary, 1);
        assert_eq!(stats.n_reject_unmet_stop, 1);
        assert_eq!(stats.n_reject_other, 1);
        assert_eq!(stats.last_rejection_status, Some(Status::NanOrInfResidual));
        assert_eq!(
            stats.rejection_summary(),
            "n_rejected=8, [LargeDelta=1, ResDiv=1, DeltaDiv=1, MaxIter=1, SmallDenom=1, Secondary=1, UnmetStop=1, Other=1], last=NanOrInfResidual"
        );

        stats.record_step_attempt(StepSummary {
            step_index: 1,
            accepted: true,
            status: Status::Success,
            h_used: 0.25,
            h_next: 0.3,
            rdiff: 0.1,
            n_iteration: 3,
            residual_0: 1e-2,
            residual_last: 1e-12,
            lambda: 0.5,
        });
        assert_eq!(stats.step_h_used, 0.25);
        assert_eq!(stats.step_h_next, 0.3);
        assert_eq!(stats.step_rdiff, 0.1);
        assert_eq!(stats.step_n_iteration, 3);
        assert_eq!(stats.step_residual_0, 1e-2);
        assert_eq!(stats.step_residual_last, 1e-12);
        let summaries = stats.step_summaries.as_ref().unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].step_index, 1);
        assert_eq!(summaries[0].lambda, 0.5);

        stats.reset(true);
        assert_eq!(stats.n_reject_large_delta, 0);
        assert_eq!(stats.last_rejection_status, None);
        assert!(stats.step_summaries.is_none());
        assert_eq!(stats.h_next, 0.0);
    }
}
