use super::{NlMethod, NlParams, NumError};

/// Prints information during time stepping
pub(crate) struct Logger {
    /// Method
    method: NlMethod,

    /// Enables verbose output
    verbose: bool,

    /// Enables verbose output for iterations
    verbose_iterations: bool,

    /// Enables verbose output for the legend
    verbose_legend: bool,

    /// List of error messages
    errors: Vec<String>,

    /// Number of characters for the horizontal line
    nchar: usize,
}

impl Logger {
    /// Creates a new instance
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters including convergence tolerances
    pub fn new(params: &NlParams) -> Self {
        let nchar = match params.method {
            NlMethod::Arclength => 56,
            NlMethod::Parametric => 39,
            NlMethod::Simple => 23,
        };
        Self {
            method: params.method,
            verbose: params.verbose || params.verbose_iterations,
            verbose_iterations: params.verbose_iterations,
            verbose_legend: params.verbose_legend,
            errors: Vec::new(),
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
            NlMethod::Arclength => {
                println!(
                    "{:>8} {:>8} {:>8} {:>5} {:>9} ➖ {:>9} ➖",
                    "λ", "s", "Δs", "iter", "(δu,δλ)", "(G,H)"
                );
            }
            NlMethod::Parametric => {
                println!(
                    "{:>8} {:>8} {:>5} {:>9} ➖ {:>9} ➖",
                    "λ", "Δλ", "iter", "‖δu‖∞", "‖G‖∞"
                );
            }
            NlMethod::Simple => {
                println!("{:>5} {:>9} ➖ {:>9} ➖", "iter", "‖δu‖∞", "‖G‖∞");
            }
        }
        println!("{}", "─".repeat(self.nchar));
    }

    /// Prints step information
    pub fn step(&self, increment: usize, l: f64, s: f64, h: f64) {
        if !self.verbose {
            return;
        }
        if increment == 0 {
            match self.method {
                NlMethod::Arclength => {
                    println!("{:>8.3e} {:>8.3e} {:>8.3e}", l, s, h);
                }
                NlMethod::Parametric => {
                    println!("{:>8.3e} {:>8.3e}", l, h);
                }
                NlMethod::Simple => (),
            }
        } else {
            match self.method {
                NlMethod::Arclength => {
                    println!("{:>8} {:>8} {:>8}", "·", "·", "·");
                }
                NlMethod::Parametric => {
                    println!("{:>8} {:>8}", "·", "·");
                }
                NlMethod::Simple => (),
            }
        }
    }

    /// Prints iteration information
    pub fn iteration(&self, iter: usize, err: &NumError) {
        if !self.verbose_iterations {
            return;
        }
        let (icon_gh, icon_ul) = if iter == 0 {
            ("", "")
        } else {
            if iter == 1 && err.converged_on_gh {
                ("", self.icon(err.converged_on_gh, err.diverging_on_gh))
            } else {
                (
                    self.icon(err.converged_on_gh, err.diverging_on_gh),
                    self.icon(err.converged_on_ul, err.diverging_on_ul),
                )
            }
        };
        match self.method {
            NlMethod::Arclength => {
                println!(
                    "{:>8} {:>8} {:>8} {:>5} {:>9.2e} {} {:>9.2e} {}",
                    "·", "·", "·", iter, err.max_ul, icon_ul, err.max_gh, icon_gh
                );
            }
            NlMethod::Parametric => {
                println!(
                    "{:>8} {:>8} {:>5} {:>9.2e} {} {:>9.2e} {}",
                    "·", "·", iter, err.max_ul, icon_ul, err.max_gh, icon_gh
                );
            }
            NlMethod::Simple => {
                println!(
                    "{:>5} {:>9.2e} {} {:>9.2e} {}",
                    iter, err.max_ul, icon_ul, err.max_gh, icon_gh
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
}
