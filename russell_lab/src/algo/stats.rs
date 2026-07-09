use super::UNINITIALIZED;
use crate::{Stopwatch, format_nanoseconds};
use std::fmt::{self, Write};

/// Holds generic statistics for the algorithms
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Number of calls to f(x) (function evaluations)
    pub n_function: usize,

    /// Number of Jacobian matrix evaluations
    pub n_jacobian: usize,

    /// Number of iterations
    pub n_iterations: usize,

    /// Holds an estimate of the absolute or relative error (depending on the algorithm)
    pub error_estimate: f64,

    /// Holds the total nanoseconds during a computation
    pub nanos_total: u128,

    /// Holds a stopwatch for measuring the elapsed time during a computation
    pub(crate) sw_total: Stopwatch,
}

impl Stats {
    /// Allocates a new instance
    pub fn new() -> Stats {
        Stats {
            n_function: 0,
            n_jacobian: 0,
            n_iterations: 0,
            error_estimate: UNINITIALIZED,
            nanos_total: 0,
            sw_total: Stopwatch::new(),
        }
    }

    /// Resets the statistics to their initial state
    pub fn reset(&mut self) {
        self.n_function = 0;
        self.n_jacobian = 0;
        self.n_iterations = 0;
        self.error_estimate = UNINITIALIZED;
        self.nanos_total = 0;
        self.sw_total.reset();
    }

    /// Stops the stopwatch and updates total nanoseconds
    pub(crate) fn stop_sw_total(&mut self) {
        self.nanos_total = self.sw_total.stop();
    }

    /// Returns a pretty formatted string with the stats
    pub fn summary(&self) -> String {
        let mut buffer = String::new();
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
        buffer
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\n\
             Total computation time           = {}",
            self.summary(),
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

    #[test]
    fn stats_summary_and_display_work() {
        let stats = Stats::new();
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
        stats.n_function = 5;
        stats.n_jacobian = 3;
        stats.n_iterations = 2;
        stats.nanos_total = 1000;
        stats.reset();
        assert_eq!(stats.n_function, 0);
        assert_eq!(stats.n_jacobian, 0);
        assert_eq!(stats.n_iterations, 0);
        assert_eq!(stats.nanos_total, 0);
    }
}
