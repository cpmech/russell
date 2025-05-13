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
            Method::Arclength => 73,
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
                    "{:>8} {:>8} {:>8} {:>5} {:>10} ➖ {:>10} {:>12} ➖",
                    "λ", "s", "Δs", "iter", "‖(G,N)‖∞", "‖(δu,δλ)‖∞", "Rel((δu,δλ))"
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
        let icon_gh = self.icon(iter, err.residual_converged, err.residual_diverging);
        let icon_ul = self.icon(iter, err.delta_converged, err.delta_diverging);
        let k = iter + 1;
        match self.method {
            Method::Arclength => {
                println!(
                    "{:>8} {:>8} {:>8} {:>5} {:>10.2e} {} {:>10.2e} {:>12.2e} {}",
                    "·", "·", "·", k, err.residual_max, icon_gh, err.delta_max, err.delta_rms, icon_ul
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
                    println!("❌ {} ❌", message);
                }
                println!("{}\n", "═".repeat(60));
            }
        }
    }
}
