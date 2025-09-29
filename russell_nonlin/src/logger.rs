use super::{Config, IterationError, Method, State, Stats, Status};
use crate::StrError;
use russell_lab::format_scientific;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::Path;

/// Number of characters for the horizontal line
const NCHAR: usize = 68;

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
        Logger {
            method: config.method,
            enabled: config.verbose || config.log_file.is_some(),
            with_iterations: config.verbose_iterations,
            with_legend: config.verbose_legend,
            with_statistics: config.verbose_stats,
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
            let legend_text = "Legend:\n  ✅  Converged\n  🎈  Diverging\n  🔹  Not converged";
            if self.full_path.is_none() {
                println!("{}", legend_text);
            } else {
                writeln!(&mut self.buffer, "{}", legend_text).unwrap();
            }
        }

        let header_line = "─".repeat(NCHAR);
        let column_headers = match self.method {
            Method::Arclength => format!(
                "{:>10} {:>10} {:>5} {:>10} ➖ {:>10} {:>12} ➖",
                "λ", "Δs", "iter", "‖(G,N)‖∞", "‖(δu,δλ)‖∞", "Rel((δu,δλ))"
            ),
            Method::Natural => format!(
                "{:>10} {:>10} {:>5} {:>10} ➖ {:>10} {:>12} ➖",
                "λ", "Δλ", "iter", "‖G‖∞", "‖δu‖∞", "Rel(δu)"
            ),
        };

        if self.full_path.is_none() {
            println!("{}", header_line);
            println!("{}", column_headers);
            println!("{}", header_line);
        } else {
            writeln!(&mut self.buffer, "{}", header_line).unwrap();
            writeln!(&mut self.buffer, "{}", column_headers).unwrap();
            writeln!(&mut self.buffer, "{}", header_line).unwrap();
        }
    }

    /// Prints step information
    pub fn step(&mut self, h: f64, state: &State) {
        if !self.enabled {
            return;
        }

        let str_l = format_scientific(state.l, 10, 3);
        let str_h = format_scientific(h, 10, 3);
        let output = format!("{} {}", str_l, str_h);

        if self.full_path.is_none() {
            println!("{}", output);
        } else {
            writeln!(&mut self.buffer, "{}", output).unwrap();
        }
    }

    /// Prints iteration information
    pub fn iteration(&mut self, iter: usize, err: &IterationError) {
        if !(self.enabled && self.with_iterations) {
            return;
        }

        let icon_gh = self.icon(iter, err.residual_converged, err.residual_diverging);
        let icon_ul = self.icon(iter, err.delta_converged, err.delta_diverging);
        let k = iter + 1;

        let output = match self.method {
            Method::Arclength => {
                let res_max = format_scientific(err.residual_max, 10, 2);
                let del_max = format_scientific(err.delta_max, 10, 2);
                let del_rms = format_scientific(err.delta_rms, 12, 2);
                format!(
                    "{:>10} {:>10} {:>5} {} {} {} {} {}",
                    "·", "·", k, res_max, icon_gh, del_max, del_rms, icon_ul
                )
            }
            Method::Natural => {
                if err.residual_converged {
                    let res_max = format_scientific(err.residual_max, 10, 2);
                    format!(
                        "{:>10} {:>10} {:>5} {} {} {:>10} {:>12}",
                        "·", "·", k, res_max, icon_gh, "·", "·"
                    )
                } else {
                    let res_max = format_scientific(err.residual_max, 10, 2);
                    let del_max = format_scientific(err.delta_max, 10, 2);
                    let del_rms = format_scientific(err.delta_rms, 12, 2);
                    format!(
                        "{:>10} {:>10} {:>5} {} {} {} {} {}",
                        "·", "·", k, res_max, icon_gh, del_max, del_rms, icon_ul
                    )
                }
            }
        };

        if self.full_path.is_none() {
            println!("{}", output);
        } else {
            writeln!(&mut self.buffer, "{}", output).unwrap();
        }
    }

    /// Prints a message when the step is rejected because it did not converge
    pub fn did_not_converge(&mut self) {
        if !self.enabled {
            return;
        }

        let message = format!("{:^w$}", "(rejected: did not converge)", w = NCHAR);

        if self.full_path.is_none() {
            println!("{}", message);
        } else {
            writeln!(&mut self.buffer, "{}", message).unwrap();
        }
    }

    /// Prints a message when the step is rejected because alpha is not acceptable
    pub fn alpha_is_not_acceptable(&mut self) {
        if !self.enabled {
            return;
        }

        let message = format!("{:^w$}", "(rejected: alpha is not acceptable)", w = NCHAR);

        if self.full_path.is_none() {
            println!("{}", message);
        } else {
            writeln!(&mut self.buffer, "{}", message).unwrap();
        }
    }

    /// Prints statistics and eventual errors
    pub fn footer(&mut self, stats: &Stats, status: Status) -> Result<(), StrError> {
        if !self.enabled {
            return Ok(());
        }

        let footer_line = format!("{}\n", "─".repeat(NCHAR));
        let mut output = String::new();

        writeln!(&mut output, "{}", footer_line.trim_end()).unwrap();
        if self.with_statistics {
            writeln!(&mut output, "\n{}", stats).unwrap();
        }
        if status.failure() {
            writeln!(&mut output, "\n{:═^1$}", " FAILURE ", NCHAR).unwrap();
            writeln!(&mut output, "❌ {:?} ❌", status).unwrap();
            writeln!(&mut output, "{}\n", "═".repeat(NCHAR)).unwrap();
        }

        if self.full_path.is_none() {
            // Write directly to stdout
            print!("{}", output);
            std::io::stdout().flush().map_err(|_| "cannot flush stdout")?;
        } else {
            // Add to buffer for file output
            write!(&mut self.buffer, "{}", output).unwrap();

            // Write buffer to file
            let path = Path::new(self.full_path.as_ref().unwrap()).to_path_buf();
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

    /// Logs extra text (for debugging purposes)
    pub(crate) fn _extra(&mut self, text: &str) {
        if !self.enabled {
            return;
        }
        if self.full_path.is_none() {
            println!("{}", text);
        } else {
            writeln!(&mut self.buffer, "{}", text).unwrap();
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Logger;
    use crate::{Config, IterationError, Method, State, Stats, Status};
    use std::fs;

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

        // Test with Natural method
        let config = Config::new(Method::Natural);
        let logger = Logger::new(&config);
        assert_eq!(logger.method, Method::Natural);
        assert!(!logger.enabled); // default is false
        assert!(!logger.with_iterations);
        assert!(!logger.with_legend);
        assert!(!logger.with_statistics);
    }

    #[test]
    fn logger_complete_workflow_natural() {
        let mut config = Config::new(Method::Natural);
        config.verbose = true;
        config.verbose_iterations = true;
        config.verbose_legend = true;
        config.verbose_stats = true;

        let full_path = "/tmp/russell_nonlin/logger_complete_workflow_natural.txt";
        config.set_log_file(full_path);

        // Simulate a complete workflow
        let mut logger = Logger::new(&config);
        let mut state = State::new(1);
        let mut err = IterationError::new(&config, 1);

        logger.header();
        state.l = 0.5;
        logger.step(0.1, &state);
        err.residual_max = 1e-8;
        err.residual_converged = true;
        logger.iteration(0, &err);
        logger.did_not_converge();
        logger.alpha_is_not_acceptable();
        let stats = Stats::new(&config);
        logger.footer(&stats, Status::SmallStepsize).unwrap();

        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();

        let expected = r#"
Legend:
  ✅  Converged
  🎈  Diverging
  🔹  Not converged
────────────────────────────────────────────────────────────────────
         λ         Δλ  iter       ‖G‖∞ ➖      ‖δu‖∞      Rel(δu) ➖
────────────────────────────────────────────────────────────────────
 5.000E-01  1.000E-01
         ·          ·     1   1.00E-08 ✅          ·            ·
                    (rejected: did not converge)                    
                (rejected: alpha is not acceptable)                 
────────────────────────────────────────────────────────────────────

Natural parameter continuation; solves G(u, λ) = 0 (fixed)
Using numerical Jacobian         = false
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

═════════════════════════════ FAILURE ══════════════════════════════
❌ SmallStepsize ❌
════════════════════════════════════════════════════════════════════

"#;
        assert_eq!(&contents, &expected[1..]);
    }

    #[test]
    fn logger_complete_workflow_arclength() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_iterations = true;
        config.verbose_legend = true;
        config.verbose_stats = true;

        let full_path = "/tmp/russell_nonlin/logger_complete_workflow_arclength.txt";
        config.set_log_file(full_path);

        // Simulate a complete workflow
        let mut logger = Logger::new(&config);
        let mut state = State::new(1);
        let mut err = IterationError::new(&config, 1);

        logger.header();
        state.l = 0.5;
        logger.step(0.1, &state);
        err.residual_max = 1e-8;
        err.residual_converged = true;
        logger.iteration(0, &err);
        logger.did_not_converge();
        logger.alpha_is_not_acceptable();
        let stats = Stats::new(&config);
        logger.footer(&stats, Status::SmallStepsize).unwrap();

        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();

        let expected = r#"
Legend:
  ✅  Converged
  🎈  Diverging
  🔹  Not converged
────────────────────────────────────────────────────────────────────
         λ         Δs  iter   ‖(G,N)‖∞ ➖ ‖(δu,δλ)‖∞ Rel((δu,δλ)) ➖
────────────────────────────────────────────────────────────────────
 5.000E-01  1.000E-01
         ·          ·     1   1.00E-08 ✅   0.00E+00     0.00E+00   
                    (rejected: did not converge)                    
                (rejected: alpha is not acceptable)                 
────────────────────────────────────────────────────────────────────

Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0 (fixed)
Using numerical Jacobian         = false
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

═════════════════════════════ FAILURE ══════════════════════════════
❌ SmallStepsize ❌
════════════════════════════════════════════════════════════════════

"#;
        assert_eq!(&contents, &expected[1..]);
    }
}
