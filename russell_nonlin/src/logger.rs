use russell_lab::format_scientific;

use super::{Config, IterationError, Method, State, Stats, Status};
use crate::StrError;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::Path;

/// Prints information during time stepping
pub(crate) struct Logger {
    /// Method
    method: Method,

    /// Enables the logging of steps and iterations
    enabled: bool,

    /// Enables detailed output of iterations
    with_iterations: bool,

    /// Enables the output of the legend
    with_legend: bool,

    /// Enables the generation of a statistics report
    with_statistics: bool,

    /// Number of characters for the horizontal line
    nchar: usize,

    /// Output buffer
    buffer: String,

    /// Full path to the log file
    ///
    /// Note: the file is written by the footer method
    full_path: Option<String>,
}

impl Logger {
    /// Creates a new instance
    pub fn new(config: &Config) -> Self {
        let nchar = match config.method {
            Method::Arclength => 64,
            Method::Natural => 62,
        };
        Logger {
            method: config.method,
            enabled: config.verbose || config.log_file.is_some(),
            with_iterations: config.verbose_iterations,
            with_legend: config.verbose_legend,
            with_statistics: config.verbose_stats,
            nchar,
            buffer: String::new(),
            full_path: config.log_file.clone(),
        }
    }

    /// Prints the header before time stepping and convergence statistics
    pub fn header(&mut self) {
        if !self.enabled {
            return;
        }
        if self.with_legend {
            self.legend();
        }
        writeln!(&mut self.buffer, "{}", "─".repeat(self.nchar)).unwrap();
        match self.method {
            Method::Arclength => {
                writeln!(
                    &mut self.buffer,
                    "{:>8} {:>8} {:>5} {:>10} ➖ {:>10} {:>12} ➖",
                    "λ", "Δs", "iter", "‖(G,N)‖∞", "‖(δu,δλ)‖∞", "Rel((δu,δλ))"
                )
                .unwrap();
            }
            Method::Natural => {
                writeln!(
                    &mut self.buffer,
                    "{:>8} {:>8} {:>5} {:>10} ➖ {:>10} {:>10} ➖",
                    "λ", "Δλ", "iter", "‖G‖∞", "‖δu‖∞", "Rel(δu)"
                )
                .unwrap();
            }
        }
        writeln!(&mut self.buffer, "{}", "─".repeat(self.nchar)).unwrap();
    }

    /// Prints step information
    pub fn step(&mut self, h: f64, state: &State) {
        if !self.enabled {
            return;
        }
        let str_l = format_scientific(state.l, 8, 3);
        let str_h = format_scientific(h, 8, 3);
        writeln!(&mut self.buffer, "{} {}", str_l, str_h).unwrap();
    }

    /// Prints iteration information
    pub fn iteration(&mut self, iter: usize, err: &IterationError) {
        if !(self.enabled && self.with_iterations) {
            return;
        }
        let icon_gh = self.icon(iter, err.residual_converged, err.residual_diverging);
        let icon_ul = self.icon(iter, err.delta_converged, err.delta_diverging);
        let k = iter + 1;
        match self.method {
            Method::Arclength => {
                let res_max = format_scientific(err.residual_max, 10, 2);
                let del_max = format_scientific(err.delta_max, 10, 2);
                let del_rms = format_scientific(err.delta_rms, 12, 2);
                writeln!(
                    &mut self.buffer,
                    "{:>8} {:>8} {:>5} {} {} {} {} {}",
                    "·", "·", k, res_max, icon_gh, del_max, del_rms, icon_ul
                )
                .unwrap();
            }
            Method::Natural => {
                if err.residual_converged {
                    let res_max = format_scientific(err.residual_max, 10, 2);
                    writeln!(
                        &mut self.buffer,
                        "{:>8} {:>8} {:>5} {} {} {:>10} {:>10}",
                        "·", "·", k, res_max, icon_gh, "·", "·"
                    )
                    .unwrap();
                } else {
                    let res_max = format_scientific(err.residual_max, 10, 2);
                    let del_max = format_scientific(err.delta_max, 10, 2);
                    let del_rms = format_scientific(err.delta_rms, 12, 2);
                    writeln!(
                        &mut self.buffer,
                        "{:>8} {:>8} {:>5} {} {} {} {} {}",
                        "·", "·", k, res_max, icon_gh, del_max, del_rms, icon_ul
                    )
                    .unwrap();
                }
            }
        }
    }

    /// Returns the icon
    #[inline]
    fn icon(&self, iter: usize, converged: bool, diverging: bool) -> &'static str {
        if converged {
            "✅"
        } else if diverging {
            "🎈"
        } else if iter == 0 {
            "  "
        } else {
            "🔹"
        }
    }

    /// Prints the legend
    #[inline]
    fn legend(&mut self) {
        writeln!(
            &mut self.buffer,
            "Legend:\n  ✅  Converged\n  🎈  Diverging\n  🔹  Not converged"
        )
        .unwrap();
    }

    /// Prints a message when the step is rejected because it did not converge
    pub fn did_not_converge(&mut self) {
        if self.enabled {
            writeln!(
                &mut self.buffer,
                "{:^w$}",
                "(rejected: did not converge)",
                w = self.nchar
            )
            .unwrap();
        }
    }

    /// Prints a message when the step is rejected because alpha is not acceptable
    pub fn alpha_is_not_acceptable(&mut self) {
        if self.enabled {
            writeln!(
                &mut self.buffer,
                "{:^w$}",
                "(rejected: alpha is not acceptable)",
                w = self.nchar
            )
            .unwrap();
        }
    }

    /// Prints statistics and eventual errors
    pub fn footer(&mut self, stats: &Stats, status: Status) -> Result<(), StrError> {
        if self.enabled {
            writeln!(&mut self.buffer, "{}\n", "─".repeat(self.nchar)).unwrap();
            if self.with_statistics {
                writeln!(&mut self.buffer, "{}", stats).unwrap();
            }
            if status.failure() {
                writeln!(&mut self.buffer, "\n{:═^1$}", " FAILURE ", 60).unwrap();
                writeln!(&mut self.buffer, "❌ {:?} ❌", status).unwrap();
                writeln!(&mut self.buffer, "{}\n", "═".repeat(60)).unwrap();
            }
        }
        if let Some(fp) = self.full_path.as_ref() {
            let path = Path::new(fp).to_path_buf();
            if let Some(p) = path.parent() {
                fs::create_dir_all(p).map_err(|_| "cannot create directory")?;
            }
            let mut file = File::create(&path).map_err(|_| "cannot create file")?;
            file.write_all(self.buffer.as_bytes())
                .map_err(|_| "cannot write file")?;
            file.sync_all().map_err(|_| "cannot sync file")?;
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Logger;
    use crate::{Config, IterationError, Method, State, Stats, Status};

    #[test]
    fn logger_new_works() {
        // Test with Arclength method
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_iterations = true;
        config.verbose_legend = true;
        config.verbose_stats = true;

        let logger = Logger::new(&config);
        assert_eq!(logger.method, Method::Arclength);
        assert!(logger.enabled);
        assert!(logger.with_iterations);
        assert!(logger.with_legend);
        assert!(logger.with_statistics);
        assert_eq!(logger.nchar, 64);

        // Test with Natural method
        let config = Config::new(Method::Natural);
        let logger = Logger::new(&config);
        assert_eq!(logger.method, Method::Natural);
        assert!(!logger.enabled); // default is false
        assert!(!logger.with_iterations);
        assert!(!logger.with_legend);
        assert!(!logger.with_statistics);
        assert_eq!(logger.nchar, 62);
    }

    #[test]
    fn logger_header_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let mut logger = Logger::new(&config);

        logger.header();
        assert!(logger.buffer.is_empty());
    }

    #[test]
    fn logger_header_arclength_with_legend() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_legend = true;
        let mut logger = Logger::new(&config);

        logger.header();

        let expected = r#"Legend:
  ✅  Converged
  🎈  Diverging
  🔹  Not converged
────────────────────────────────────────────────────────────────
       λ       Δs  iter   ‖(G,N)‖∞ ➖ ‖(δu,δλ)‖∞ Rel((δu,δλ)) ➖
────────────────────────────────────────────────────────────────
"#;
        assert_eq!(logger.buffer, expected);
    }

    #[test]
    fn logger_header_arclength_without_legend() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_legend = false;
        let mut logger = Logger::new(&config);

        logger.header();

        let expected = r#"
────────────────────────────────────────────────────────────────
       λ       Δs  iter   ‖(G,N)‖∞ ➖ ‖(δu,δλ)‖∞ Rel((δu,δλ)) ➖
────────────────────────────────────────────────────────────────
"#;
        assert_eq!(logger.buffer, &expected[1..]);
    }

    #[test]
    fn logger_header_natural_without_legend() {
        let mut config = Config::new(Method::Natural);
        config.verbose = true;
        config.verbose_legend = false;
        let mut logger = Logger::new(&config);

        logger.header();

        let expected = r#"
──────────────────────────────────────────────────────────────
       λ       Δλ  iter       ‖G‖∞ ➖      ‖δu‖∞    Rel(δu) ➖
──────────────────────────────────────────────────────────────
"#;
        assert_eq!(logger.buffer, &expected[1..]);
    }

    #[test]
    fn logger_step_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let mut logger = Logger::new(&config);
        let state = State::new(1);

        logger.step(0.1, &state);
        assert!(logger.buffer.is_empty());
    }

    #[test]
    fn logger_step_works() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        let mut logger = Logger::new(&config);

        let mut state = State::new(2);
        state.l = 1.5;
        let h = 0.25;

        logger.step(h, &state);

        let expected = "1.500E+00 2.500E-01\n";
        assert_eq!(logger.buffer, expected);
    }

    #[test]
    fn logger_iteration_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let mut logger = Logger::new(&config);
        let err = IterationError::new(&config, 1);

        logger.iteration(0, &err);
        assert!(logger.buffer.is_empty());
    }

    #[test]
    fn logger_iteration_verbose_iterations_false_does_nothing() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_iterations = false;
        let mut logger = Logger::new(&config);
        let err = IterationError::new(&config, 1);

        logger.iteration(0, &err);
        assert!(logger.buffer.is_empty());
    }

    #[test]
    fn logger_iteration_arclength_converged() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_iterations = true;
        let mut logger = Logger::new(&config);

        let mut err = IterationError::new(&config, 1);
        err.residual_max = 1e-10;
        err.delta_max = 1e-11;
        err.delta_rms = 1e-12;
        err.residual_converged = true;
        err.delta_converged = true;

        logger.iteration(0, &err);

        let expected = "       ·        ·     1   1.00E-10 ✅   1.00E-11     1.00E-12 ✅\n";
        assert_eq!(logger.buffer, expected);
    }

    #[test]
    fn logger_iteration_arclength_diverging() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_iterations = true;
        let mut logger = Logger::new(&config);

        let mut err = IterationError::new(&config, 1);
        err.residual_max = 1e10;
        err.delta_max = 1e11;
        err.delta_rms = 1e12;
        err.residual_diverging = true;
        err.delta_diverging = true;

        logger.iteration(1, &err);

        let expected = "       ·        ·     2   1.00E+10 🎈   1.00E+11     1.00E+12 🎈\n";
        assert_eq!(logger.buffer, expected);
    }

    #[test]
    fn logger_iteration_natural_converged() {
        let mut config = Config::new(Method::Natural);
        config.verbose = true;
        config.verbose_iterations = true;
        let mut logger = Logger::new(&config);

        let mut err = IterationError::new(&config, 1);
        err.residual_max = 1e-10;
        err.residual_converged = true;

        logger.iteration(0, &err);

        let expected = "       ·        ·     1   1.00E-10 ✅          ·          ·\n";
        assert_eq!(logger.buffer, expected);
    }

    #[test]
    fn logger_iteration_natural_not_converged() {
        let mut config = Config::new(Method::Natural);
        config.verbose = true;
        config.verbose_iterations = true;
        let mut logger = Logger::new(&config);

        let mut err = IterationError::new(&config, 1);
        err.residual_max = 1e-3;
        err.delta_max = 1e-4;
        err.delta_rms = 1e-5;
        err.residual_converged = false;
        err.delta_converged = false;

        logger.iteration(2, &err);

        let expected = "       ·        ·     3   1.00E-03 🔹   1.00E-04     1.00E-05 🔹\n";
        assert_eq!(logger.buffer, expected);
    }

    #[test]
    fn logger_icon_works() {
        let config = Config::new(Method::Arclength);
        let logger = Logger::new(&config);

        // Test converged case
        assert_eq!(logger.icon(1, true, false), "✅");
        assert_eq!(logger.icon(0, true, true), "✅"); // converged takes precedence

        // Test diverging case
        assert_eq!(logger.icon(1, false, true), "🎈");
        assert_eq!(logger.icon(5, false, true), "🎈");

        // Test first iteration, not converged, not diverging
        assert_eq!(logger.icon(0, false, false), "  ");

        // Test other iterations, not converged, not diverging
        assert_eq!(logger.icon(1, false, false), "🔹");
        assert_eq!(logger.icon(10, false, false), "🔹");
    }

    #[test]
    fn logger_did_not_converge_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let mut logger = Logger::new(&config);

        logger.did_not_converge();
        assert!(logger.buffer.is_empty());
    }

    #[test]
    fn logger_did_not_converge_works() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        let mut logger = Logger::new(&config);

        logger.did_not_converge();

        let expected = "                  (rejected: did not converge)                  \n";
        assert_eq!(logger.buffer, expected);
    }

    #[test]
    fn logger_alpha_is_not_acceptable_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let mut logger = Logger::new(&config);

        logger.alpha_is_not_acceptable();
        assert!(logger.buffer.is_empty());
    }

    #[test]
    fn logger_alpha_is_not_acceptable_works() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        let mut logger = Logger::new(&config);

        logger.alpha_is_not_acceptable();

        let expected = "              (rejected: alpha is not acceptable)               \n";
        assert_eq!(logger.buffer, expected);
    }

    #[test]
    fn logger_footer_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let mut logger = Logger::new(&config);
        let stats = Stats::new(&config);

        logger.footer(&stats, Status::Success).unwrap();
        assert!(logger.buffer.is_empty());
    }

    #[test]
    fn logger_footer_success_with_stats() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_stats = true;
        let mut logger = Logger::new(&config);
        let stats = Stats::new(&config);

        logger.footer(&stats, Status::Success).unwrap();

        // Should contain the line separator and stats, but no failure message
        let expected = r#"
────────────────────────────────────────────────────────────────

Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0 (fixed)
Number of function evaluations   = 0
Number of Jacobian evaluations   = 0
Number of factorizations         = 0
Number of lin sys solutions      = 0
Number of accepted steps         = 0
Number of rejected steps         = 0
Number of performed steps        = 0
Number of iterations (total)     = 0
Last accepted/suggested stepsize = 0
Max time spent on a step         = 0ns
Max time spent on the Jacobian   = 0ns
Max time spent on factorization  = 0ns
Max time spent on lin solution   = 0ns
Total time                       = 0ns
"#;
        assert_eq!(logger.buffer, &expected[1..]);
    }

    #[test]
    fn logger_footer_failure_with_stats() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_stats = true;
        let mut logger = Logger::new(&config);
        let stats = Stats::new(&config);

        logger.footer(&stats, Status::LargeDelta).unwrap();

        // Should contain the line separator, stats, and failure message
        // (notice the double new line at the end)
        let expected = r#"
────────────────────────────────────────────────────────────────

Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0 (fixed)
Number of function evaluations   = 0
Number of Jacobian evaluations   = 0
Number of factorizations         = 0
Number of lin sys solutions      = 0
Number of accepted steps         = 0
Number of rejected steps         = 0
Number of performed steps        = 0
Number of iterations (total)     = 0
Last accepted/suggested stepsize = 0
Max time spent on a step         = 0ns
Max time spent on the Jacobian   = 0ns
Max time spent on factorization  = 0ns
Max time spent on lin solution   = 0ns
Total time                       = 0ns

═════════════════════════ FAILURE ══════════════════════════
❌ LargeDelta ❌
════════════════════════════════════════════════════════════

"#;
        assert_eq!(logger.buffer, &expected[1..]);
    }

    #[test]
    fn logger_footer_success_without_stats() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_stats = false;
        let mut logger = Logger::new(&config);
        let stats = Stats::new(&config);

        logger.footer(&stats, Status::Success).unwrap();

        // Should contain only the line separator, no stats or failure message
        // (notice the double new line at the end)
        let expected = r#"
────────────────────────────────────────────────────────────────

"#;
        assert_eq!(logger.buffer, &expected[1..]);
    }

    #[test]
    fn logger_complete_workflow() {
        let mut config = Config::new(Method::Natural);
        config.verbose = true;
        config.verbose_iterations = true;
        config.verbose_legend = true;
        config.verbose_stats = true;
        let mut logger = Logger::new(&config);

        // Simulate a complete workflow
        logger.header();

        let mut state = State::new(1);
        state.l = 0.5;
        logger.step(0.1, &state);

        let mut err = IterationError::new(&config, 1);
        err.residual_max = 1e-8;
        err.residual_converged = true;
        logger.iteration(0, &err);

        let stats = Stats::new(&config);
        logger.footer(&stats, Status::Success).unwrap();

        let expected = r#"
Legend:
  ✅  Converged
  🎈  Diverging
  🔹  Not converged
──────────────────────────────────────────────────────────────
       λ       Δλ  iter       ‖G‖∞ ➖      ‖δu‖∞    Rel(δu) ➖
──────────────────────────────────────────────────────────────
5.000E-01 1.000E-01
       ·        ·     1   1.00E-08 ✅          ·          ·
──────────────────────────────────────────────────────────────

Natural parameter continuation; solves G(u, λ) = 0 (fixed)
Number of function evaluations   = 0
Number of Jacobian evaluations   = 0
Number of factorizations         = 0
Number of lin sys solutions      = 0
Number of accepted steps         = 0
Number of rejected steps         = 0
Number of performed steps        = 0
Number of iterations (total)     = 0
Last accepted/suggested stepsize = 0
Max time spent on a step         = 0ns
Max time spent on the Jacobian   = 0ns
Max time spent on factorization  = 0ns
Max time spent on lin solution   = 0ns
Total time                       = 0ns
"#;
        assert_eq!(logger.buffer, &expected[1..]);
    }
}
