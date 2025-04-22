use super::{Config, IterationError, Method, State, Workspace};

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
            Method::Arclength => 56,
            Method::Natural => 61,
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
                    "{:>8} {:>8} {:>8} {:>5} {:>9} ➖ {:>9} ➖",
                    "λ", "s", "Δs", "iter", "(δu,δλ)", "(G,Nₒ)"
                );
            }
            Method::Natural => {
                println!(
                    "{:>8} {:>8} {:>5} {:>9} ➖ {:>9} {:>9} ➖",
                    "λ", "Δλ", "iter", "RMS(δu)", "‖δu‖∞", "‖G‖∞"
                );
            }
        }
        println!("{}", "─".repeat(self.nchar));
    }

    /// Prints step information
    pub fn step(&self, state: &State) {
        if !self.verbose {
            return;
        }
        match self.method {
            Method::Arclength => {
                println!("{:>8.3e} {:>8.3e} {:>8.3e}", state.l, state.s, state.h);
            }
            Method::Natural => {
                println!("{:>8.3e} {:>8.3e}", state.l, state.h);
            }
        }
    }

    /// Prints iteration information
    pub fn iteration(&self, iter: usize, err: &IterationError) {
        if !(self.verbose && self.verbose_iterations) {
            return;
        }
        let (icon_gh, icon_ul) = if iter == 0 {
            ("  ", "  ")
        } else {
            if iter == 1 && err.residual_converged {
                (self.icon(err.residual_converged, err.residual_diverging), "  ")
            } else {
                (
                    self.icon(err.residual_converged, err.residual_diverging),
                    self.icon(err.delta_converged, err.delta_diverging),
                )
            }
        };
        let k = iter + 1;
        match self.method {
            Method::Arclength => {
                println!(
                    "{:>8} {:>8} {:>8} {:>5} {:>9.2e} {} {:>9.2e} {}",
                    "·", "·", "·", k, err.delta_max, icon_ul, err.residual_max, icon_gh
                );
            }
            Method::Natural => {
                println!(
                    "{:>8} {:>8} {:>5} {:>9.2e} {} {:>9.2e} {:>9.2e} {}",
                    "·", "·", k, err.delta_rms, icon_ul, err.delta_max, err.residual_max, icon_gh
                );
            }
        }
    }

    /// Returns the icon
    #[inline]
    fn icon(&self, converged: bool, diverging: bool) -> &'static str {
        if converged {
            "✅"
        } else if diverging {
            "🎈"
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

    /// Prints statistics and eventual errors
    pub fn footer(&self, work: &Workspace) {
        if self.verbose {
            println!("{}\n", "─".repeat(self.nchar));
            if self.verbose_stats {
                println!("{}", work.stats);
                println!("Automatic stepsize adjustment    = {:?}", work.auto);
            }
            let messages = work.errors();
            if messages.len() > 0 {
                println!("\n{:═^1$}", " ERRORS ", 60);
                for message in &messages {
                    println!("{}", message);
                }
                println!("{}\n", "═".repeat(60));
            }
        }
    }
}
