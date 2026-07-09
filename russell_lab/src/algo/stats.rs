use super::UNINITIALIZED;
use crate::{Stopwatch, format_nanoseconds};
use std::fmt::{self, Write};

/// Holds generic statistics for the algorithms
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Indicates whether statistics collection is enabled or disabled
    enabled: bool,

    /// Number of calls to f(x) (function evaluations)
    n_function: usize,

    /// Number of Jacobian matrix evaluations
    n_jacobian: usize,

    /// Number of iterations
    n_iterations: usize,

    /// Holds an estimate of the absolute or relative error (depending on the algorithm)
    error_estimate: f64,

    /// Holds the total nanoseconds during a computation
    nanos_total: u128,

    /// Holds a stopwatch for measuring the elapsed time during a computation
    sw_total: Stopwatch,
}

impl Stats {
    /// Allocates a new instance
    pub fn new() -> Stats {
        Stats {
            enabled: false,
            n_function: 0,
            n_jacobian: 0,
            n_iterations: 0,
            error_estimate: UNINITIALIZED,
            nanos_total: 0,
            sw_total: Stopwatch::new(),
        }
    }

    /// Enables statistics collection
    #[inline]
    pub(crate) fn enable(&mut self, value: bool) {
        self.enabled = value;
    }

    /// Indicates whether statistics collection is enabled
    #[inline]
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Resets the statistics to their initial state
    #[inline]
    pub(crate) fn reset(&mut self) {
        if self.enabled {
            self.n_function = 0;
            self.n_jacobian = 0;
            self.n_iterations = 0;
            self.error_estimate = UNINITIALIZED;
            self.nanos_total = 0;
            self.sw_total.reset();
        }
    }

    /// Increments the number of function evaluations
    #[inline]
    pub(crate) fn inc_n_function(&mut self, count: usize) {
        if self.enabled {
            self.n_function += count;
        }
    }

    /// Increments the number of Jacobian evaluations
    #[inline]
    pub(crate) fn inc_n_jacobian(&mut self, count: usize) {
        if self.enabled {
            self.n_jacobian += count;
        }
    }

    /// Increments the number of iterations
    #[inline]
    pub(crate) fn inc_n_iterations(&mut self, count: usize) {
        if self.enabled {
            self.n_iterations += count;
        }
    }

    /// Sets the error estimate
    #[inline]
    pub(crate) fn set_error_estimate(&mut self, value: f64) {
        if self.enabled {
            self.error_estimate = value;
        }
    }

    /// Stops the stopwatch and updates total nanoseconds
    #[inline]
    pub(crate) fn stop_sw_total(&mut self) {
        if self.enabled {
            self.nanos_total = self.sw_total.stop();
        }
    }

    /// Returns the number of function evaluations
    pub fn get_n_function(&self) -> usize {
        self.n_function
    }

    /// Returns the number of Jacobian evaluations
    pub fn get_n_jacobian(&self) -> usize {
        self.n_jacobian
    }

    /// Returns the number of iterations
    pub fn get_n_iterations(&self) -> usize {
        self.n_iterations
    }

    /// Returns the error estimate
    pub fn get_error_estimate(&self) -> f64 {
        self.error_estimate
    }

    /// Returns the elapsed time in a pretty formatted string
    pub fn get_elapsed_time(&self) -> String {
        format_nanoseconds(self.nanos_total)
    }

    /// Returns a pretty formatted string with the stats
    pub fn summary(&self) -> String {
        let mut buffer = String::new();
        if self.enabled {
            let est_err = if self.error_estimate == UNINITIALIZED {
                "unavailable".to_string()
            } else {
                format!("{:.2e}", self.error_estimate)
            };
            write!(
                &mut buffer,
                "Number of function evaluations   = {}\n\
                 Number of Jacobian evaluations   = {}\n\
                 Number of iterations             = {}\n\
                 Error estimate                   = {}",
                self.n_function, self.n_jacobian, self.n_iterations, est_err
            )
            .unwrap();
        }
        buffer
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.enabled {
            write!(
                f,
                "{}\n\
                Total computation time           = {}",
                self.summary(),
                format_nanoseconds(self.nanos_total),
            )
            .unwrap();
        } else {
            write!(f, "Statistics tracking is disabled").unwrap();
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Stats;

    #[test]
    fn stats_summary_and_display_work() {
        let mut stats = Stats::new();
        stats.enable(true);
        assert_eq!(
            format!("{}", stats),
            "Number of function evaluations   = 0\n\
             Number of Jacobian evaluations   = 0\n\
             Number of iterations             = 0\n\
             Error estimate                   = unavailable\n\
             Total computation time           = 0ns"
        );
    }

    #[test]
    fn stats_reset_works() {
        let mut stats = Stats::new();
        stats.enable(true);
        stats.inc_n_function(5);
        stats.inc_n_jacobian(3);
        stats.inc_n_iterations(2);
        stats.stop_sw_total();
        stats.reset();
        assert_eq!(stats.get_n_function(), 0);
        assert_eq!(stats.get_n_jacobian(), 0);
        assert_eq!(stats.get_n_iterations(), 0);
    }
}
