use super::{Config, IterationError, Method, State, Status, Workspace};

/// Prints information during time stepping
pub(crate) struct Logger {
    /// Method
    method: Method,

    /// Enables verbose output
    verbose: bool,

    /// Enables verbose output for iterations
    verbose_iterations: bool,

    /// Enables verbose output for the legend
    verbose_legend: bool,

    /// Show statistics
    verbose_stats: bool,

    /// Number of characters for the horizontal line
    nchar: usize,
}

impl Logger {
    /// Creates a new instance
    pub fn new(config: &Config) -> Self {
        let nchar = match config.method {
            Method::Arclength => 64,
            Method::Natural => 59,
        };
        Self {
            method: config.method,
            verbose: config.verbose,
            verbose_iterations: config.verbose_iterations,
            verbose_legend: config.verbose_legend,
            verbose_stats: config.verbose_stats,
            nchar,
        }
    }

    /// Prints the header before time stepping and convergence statistics
    pub fn header(&self) {
        if !self.verbose {
            return;
        }
        if self.verbose_legend {
            self.legend();
        }
        println!("{}", "─".repeat(self.nchar));
        match self.method {
            Method::Arclength => {
                println!(
                    "{:>8} {:>8} {:>5} {:>10} ➖ {:>10} {:>12} ➖",
                    "λ", "Δs", "iter", "‖(G,N)‖∞", "‖(δu,δλ)‖∞", "Rel((δu,δλ))"
                );
            }
            Method::Natural => {
                println!(
                    "{:>8} {:>8} {:>5} {:>9} ➖ {:>9} {:>9} ➖",
                    "λ", "Δλ", "iter", "‖G‖∞", "‖δu‖∞", "Rel(δu)"
                );
            }
        }
        println!("{}", "─".repeat(self.nchar));
    }

    /// Prints step information
    pub fn step(&self, h: f64, state: &State) {
        if !self.verbose {
            return;
        }
        println!("{:>8.3e} {:>8.3e}", state.l, h);
    }

    /// Prints iteration information
    pub fn iteration(&self, iter: usize, err: &IterationError) {
        if !(self.verbose && self.verbose_iterations) {
            return;
        }
        let icon_gh = self.icon(iter, err.residual_converged, err.residual_diverging);
        let icon_ul = self.icon(iter, err.delta_converged, err.delta_diverging);
        let k = iter + 1;
        match self.method {
            Method::Arclength => {
                println!(
                    "{:>8} {:>8} {:>5} {:>10.2e} {} {:>10.2e} {:>12.2e} {}",
                    "·", "·", k, err.residual_max, icon_gh, err.delta_max, err.delta_rms, icon_ul
                );
            }
            Method::Natural => {
                if err.residual_converged {
                    println!(
                        "{:>8} {:>8} {:>5} {:>9.2e} {} {:>9} {:>9}",
                        "·", "·", k, err.residual_max, icon_gh, "·", "·"
                    );
                } else {
                    println!(
                        "{:>8} {:>8} {:>5} {:>9.2e} {} {:>9.2e} {:>9.2e} {}",
                        "·", "·", k, err.residual_max, icon_gh, err.delta_max, err.delta_rms, icon_ul
                    );
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
    fn legend(&self) {
        println!("Legend:");
        println!("  ✅  Converged");
        println!("  🎈  Diverging");
        println!("  🔹  Not converged");
    }

    /// Prints a message when the step is rejected because it did not converge
    pub fn did_not_converge(&self) {
        if self.verbose {
            println!("{:^w$}", "(rejected: did not converge)", w = self.nchar);
        }
    }

    /// Prints a message when the step is rejected because alpha is not acceptable
    pub fn alpha_is_not_acceptable(&self) {
        if self.verbose {
            println!("{:^w$}", "(rejected: alpha is not acceptable)", w = self.nchar);
        }
    }

    /// Prints statistics and eventual errors
    pub fn footer(&self, work: &Workspace, status: Status) {
        if self.verbose {
            println!("{}\n", "─".repeat(self.nchar));
            if self.verbose_stats {
                println!("{}", work.stats);
            }
            if status.failure() {
                println!("\n{:═^1$}", " FAILURE ", 60);
                println!("❌ {:?} ❌", status);
                println!("{}\n", "═".repeat(60));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Logger;
    use crate::{Config, IterationError, Method, Samples, State, Status, Workspace};

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
        assert!(logger.verbose);
        assert!(logger.verbose_iterations);
        assert!(logger.verbose_legend);
        assert!(logger.verbose_stats);
        assert_eq!(logger.nchar, 64);

        // Test with Natural method
        let config = Config::new(Method::Natural);
        let logger = Logger::new(&config);
        assert_eq!(logger.method, Method::Natural);
        assert!(!logger.verbose); // default is false
        assert!(!logger.verbose_iterations);
        assert!(!logger.verbose_legend);
        assert!(!logger.verbose_stats);
        assert_eq!(logger.nchar, 59);
    }

    #[test]
    fn logger_header_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let logger = Logger::new(&config);

        // Should not print anything when verbose is false
        logger.header();
        // No assertion needed - just ensuring no panic
    }

    #[test]
    fn logger_step_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let logger = Logger::new(&config);
        let state = State::new(1);

        // Should not print anything when verbose is false
        logger.step(0.1, &state);
        // No assertion needed - just ensuring no panic
    }

    #[test]
    fn logger_iteration_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let logger = Logger::new(&config);
        let err = IterationError::new(&config, 1);

        // Should not print anything when verbose is false
        logger.iteration(0, &err);
        // No assertion needed - just ensuring no panic
    }

    #[test]
    fn logger_iteration_verbose_iterations_false_does_nothing() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_iterations = false;
        let logger = Logger::new(&config);
        let err = IterationError::new(&config, 1);

        // Should not print anything when verbose_iterations is false
        logger.iteration(0, &err);
        // No assertion needed - just ensuring no panic
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
        let logger = Logger::new(&config);

        // Should not print anything when verbose is false
        logger.did_not_converge();
        // No assertion needed - just ensuring no panic
    }

    #[test]
    fn logger_alpha_is_not_acceptable_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let logger = Logger::new(&config);

        // Should not print anything when verbose is false
        logger.alpha_is_not_acceptable();
        // No assertion needed - just ensuring no panic
    }

    #[test]
    fn logger_footer_verbose_false_does_nothing() {
        let config = Config::new(Method::Arclength);
        let logger = Logger::new(&config);
        let (system, _, _) = Samples::simple_linear_problem(false, false);
        let work = Workspace::new(&config, &system);

        // Should not print anything when verbose is false
        logger.footer(&work, Status::Success);
        // No assertion needed - just ensuring no panic
    }

    #[test]
    fn logger_footer_with_success_status() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_stats = true;
        let logger = Logger::new(&config);
        let (system, _, _) = Samples::simple_linear_problem(false, false);
        let work = Workspace::new(&config, &system);

        // Should print stats but no failure message for success
        logger.footer(&work, Status::Success);
        // No assertion needed - just ensuring no panic and proper formatting
    }

    #[test]
    fn logger_footer_with_failure_status() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_stats = true;
        let logger = Logger::new(&config);
        let (system, _, _) = Samples::simple_linear_problem(false, false);
        let work = Workspace::new(&config, &system);

        // Should print stats and failure message
        logger.footer(&work, Status::LargeDelta);
        // No assertion needed - just ensuring no panic and proper formatting
    }

    #[test]
    fn logger_iteration_error_states_work() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        config.verbose_iterations = true;
        let logger = Logger::new(&config);

        // Test various error states for Arclength method
        let mut err_converged = IterationError::new(&config, 1);
        err_converged.delta_converged = true;
        logger.iteration(0, &err_converged);

        println!("\n");

        let mut err_diverging = IterationError::new(&config, 1);
        err_diverging.residual_diverging = true;
        logger.iteration(1, &err_diverging);

        println!("\n");

        let mut err_not_converged = IterationError::new(&config, 1);
        err_not_converged.residual_max = 1e-3;
        logger.iteration(2, &err_not_converged);
    }

    #[test]
    fn logger_iteration_natural_method_works() {
        let mut config = Config::new(Method::Natural);
        config.verbose = true;
        config.verbose_iterations = true;
        let logger = Logger::new(&config);

        // Test converged case for Natural method
        let mut err_converged = IterationError::new(&config, 1);
        err_converged.delta_converged = true;
        logger.iteration(0, &err_converged);

        // Test not converged case for Natural method
        let mut err_not_converged = IterationError::new(&config, 1);
        err_not_converged.delta_diverging = true;
        logger.iteration(1, &err_not_converged);
    }

    #[test]
    fn logger_step_works() {
        let mut config = Config::new(Method::Arclength);
        config.verbose = true;
        let logger = Logger::new(&config);

        let mut state = State::new(2);
        state.l = 1.5;
        let h = 0.25;

        // Should print step information
        logger.step(h, &state);
        // No assertion needed - just ensuring no panic and proper formatting
    }
}
