use crate::{format_nanoseconds, Stopwatch};
use std::fmt::{self, Write};

/// Constant to indicate an uninitialized value
pub(crate) const UNINITIALIZED: f64 = f64::INFINITY;

/// Indicates that no extra arguments for f(x) are needed
pub type NoArgs = u8;

/// Holds statistics for generic algorithms
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Number of calls to f(x) (function evaluations)
    pub n_function: usize,

    /// Number of calls to the dy/dx function
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

/// Holds the results of a root finding or minimum bracketing algorithm
///
/// The root yields `f(xo) = 0.0`. The root is bracketed by a pair of points,
/// `a` and `b`, such that the function has opposite sign at those two points,
/// i.e., `f(a) × f(b) < 0`.
///
/// The (local) minimum yields `f(xo) = min{f(x)} in [a, b]`. The (local) minimum is
/// bracketed  by a triple of points `a`, `xo`, and `c`, such that `f(xo) < f(a)`
/// and `f(xo) < f(b)`, with `a < xo < b`.
#[derive(Clone, Copy, Debug)]
pub struct Bracket {
    /// Holds the lower bound
    pub a: f64,

    /// Holds the upper bound
    pub b: f64,

    /// Holds the function evaluated at the lower bound
    pub fa: f64,

    /// Holds the function evaluated at the upper bound
    pub fb: f64,

    /// Holds the r**o**ot or **o**ptimal coordinate
    pub xo: f64,

    /// Holds the function evaluated at the root or optimal coordinate
    pub fxo: f64,
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

impl fmt::Display for Bracket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "lower bound:                   a = {}\n\
             root/optimum:                 xo = {}\n\
             upper bound:                   b = {}\n\
             function @ a:               f(a) = {}\n\
             function @ root/optimum:   f(xo) = {}\n\
             function @ b:               f(b) = {}",
            self.a, self.xo, self.b, self.fa, self.fxo, self.fb
        )
        .unwrap();
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Bracket, Stats};

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
    fn bracket_display_works() {
        let bracket = Bracket {
            a: 1.0,
            xo: 2.0,
            b: 3.0,
            fa: 4.0,
            fxo: 5.0,
            fb: 6.0,
        };
        assert_eq!(
            format!("{}", bracket),
            "lower bound:                   a = 1\n\
             root/optimum:                 xo = 2\n\
             upper bound:                   b = 3\n\
             function @ a:               f(a) = 4\n\
             function @ root/optimum:   f(xo) = 5\n\
             function @ b:               f(b) = 6",
        );
    }
}
