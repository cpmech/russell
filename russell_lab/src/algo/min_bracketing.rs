use super::{Bracket, Stats, UNINITIALIZED};
use crate::StrError;

/// Holds parameters for a bracket algorithm
#[derive(Clone, Copy, Debug)]
pub struct BracketMin {
    /// Max number of iterations
    ///
    /// ```text
    /// n_iteration_max ≥ 2
    /// ```
    pub n_iteration_max: usize,

    /// Initial step
    ///
    /// e.g., 1e-2
    pub initial_step: f64,

    /// Step expansion factor
    ///
    /// e.g., 2.0
    pub expansion_factor: f64,

    /// Uses a nonlinear step
    pub nonlinear_step: bool,
}

impl BracketMin {
    /// Allocates a new instance with default parameters
    pub fn new() -> Self {
        BracketMin {
            n_iteration_max: 20,
            initial_step: 1e-2,
            expansion_factor: 2.0,
            nonlinear_step: true,
        }
    }

    /// Validates the parameters
    pub fn validate(&self) -> Result<(), StrError> {
        if self.n_iteration_max < 2 {
            return Err("n_iteration_max must be ≥ 2");
        }
        Ok(())
    }

    /// Tries to bracket the minimum of f(x)
    ///
    /// **Note:** This function is suitable for *unimodal functions*---it may fail otherwise.
    /// The code is based on the one presented in Chapter 3 (page 36) of the Reference.
    ///
    /// Searches (iteratively) for `a`, `b` and `xo` such that:
    ///
    /// ```text
    /// f(xo) < f(a)  and  f(xo) < f(b)
    ///
    /// with a < xo < b
    /// ```
    ///
    /// Thus, `f(xo)` is the minimum of `f(x)` in the `[a, b]` interval.
    ///
    /// # Input
    ///
    /// * `x_guess` -- a starting guess
    /// * `args` -- extra arguments for the callback function
    /// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
    ///
    /// # Output
    ///
    /// Returns `(bracket, stats)` where:
    ///
    /// * `bracket` -- holds the results
    /// * `stats` -- holds statistics about the computations
    ///
    /// # Reference
    ///
    /// * Kochenderfer MJ and Wheeler TA (2019) Algorithms for Optimization, The MIT Press, 512p
    pub fn try_bracket_min<F, A>(&self, x_guess: f64, args: &mut A, mut f: F) -> Result<(Bracket, Stats), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        // validate parameters
        self.validate()?;

        // allocate stats struct
        let mut stats = Stats::new();

        // initialization
        let mut step = self.initial_step;
        let (mut a, mut xo) = (x_guess, x_guess + step);
        let (mut fa, mut fxo) = (f(a, args)?, f(xo, args)?);
        stats.n_function += 2;

        // swap values (make sure to go "downhill")
        if fxo > fa {
            swap(&mut a, &mut xo);
            swap(&mut fa, &mut fxo);
            step = -step;
        }

        // iterations
        let mut converged = false;
        let mut b = UNINITIALIZED;
        let mut fb = UNINITIALIZED;
        for k in 0..self.n_iteration_max {
            stats.n_iterations += 1;
            stats.n_function += 1;
            b = xo + step;
            fb = f(b, args)?;
            if fb > fxo {
                converged = true;
                break;
            }
            a = xo;
            fa = fxo;
            xo = b;
            fxo = fb;
            if self.nonlinear_step {
                step *= self.expansion_factor * f64::powf(2.0, k as f64);
            } else {
                step *= self.expansion_factor;
            }
        }

        // check
        if !converged {
            return Err("try_bracket_min failed to converge");
        }

        // done
        if a > b {
            swap(&mut a, &mut b);
            swap(&mut fa, &mut fb);
        }
        stats.stop_sw_total();
        Ok((Bracket { a, b, fa, fb, xo, fxo }, stats))
    }
}

/// Swaps two numbers
#[inline]
pub(super) fn swap(a: &mut f64, b: &mut f64) {
    let a_copy = a.clone();
    *a = *b;
    *b = a_copy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{swap, Bracket, BracketMin};
    use crate::algo::testing::get_test_functions;
    use crate::algo::NoArgs;
    use crate::approx_eq;

    #[test]
    fn swap_works() {
        let mut a = 12.34;
        let mut b = 56.78;
        swap(&mut a, &mut b);
        assert_eq!(a, 56.78);
        assert_eq!(b, 12.34);
    }

    #[test]
    fn validate_captures_errors() {
        let mut params = BracketMin::new();
        params.n_iteration_max = 0;
        assert_eq!(params.validate().err(), Some("n_iteration_max must be ≥ 2"));
    }

    #[test]
    fn try_bracket_min_captures_errors_1() {
        let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
        let args = &mut 0;
        assert_eq!(f(1.0, args).unwrap(), 0.0);
        let mut params = BracketMin::new();
        params.n_iteration_max = 0;
        assert_eq!(
            params.try_bracket_min(0.0, args, f).err(),
            Some("n_iteration_max must be ≥ 2")
        );
    }

    #[test]
    fn try_bracket_min_captures_errors_2() {
        struct Args {
            count: usize,
            target: usize,
        }
        let f = |x, args: &mut Args| {
            let res = if args.count == args.target {
                Err("stop")
            } else {
                Ok(x * x - 1.0)
            };
            args.count += 1;
            res
        };
        let args = &mut Args { count: 0, target: 0 };
        let params = BracketMin::new();
        // first function call
        assert_eq!(params.try_bracket_min(0.0, args, f).err(), Some("stop"));
        // second function call
        args.count = 0;
        args.target = 1;
        assert_eq!(params.try_bracket_min(0.0, args, f).err(), Some("stop"));
        // third function call
        args.count = 0;
        args.target = 2;
        assert_eq!(params.try_bracket_min(0.0, args, f).err(), Some("stop"));
    }

    fn check_consistency(bracket: &Bracket) {
        assert!(bracket.a < bracket.xo);
        assert!(bracket.xo < bracket.b);
        assert!(bracket.fa > bracket.fxo);
        assert!(bracket.fb > bracket.fxo);
    }

    #[test]
    fn try_bracket_min_works_1() {
        let args = &mut 0;
        let params = BracketMin::new();
        for (i, test) in get_test_functions().iter().enumerate() {
            if test.min1.is_none() {
                continue;
            }
            println!("\n===================================================================");
            println!("\n{}", test.name);
            let x_guess = if i == 4 {
                0.15
            } else {
                if i % 2 == 0 {
                    -0.1
                } else {
                    0.1
                }
            };
            let (bracket, stats) = params.try_bracket_min(x_guess, args, test.f).unwrap();
            println!("\n{}", bracket);
            println!("\n{}", stats);
            check_consistency(&bracket);
            approx_eq((test.f)(bracket.a, args).unwrap(), bracket.fa, 1e-15);
            approx_eq((test.f)(bracket.b, args).unwrap(), bracket.fb, 1e-15);
            approx_eq((test.f)(bracket.xo, args).unwrap(), bracket.fxo, 1e-15);
        }
        println!("\n===================================================================\n");
    }

    #[test]
    fn try_bracket_min_fails_on_non_converged() {
        let f = |x, _: &mut NoArgs| Ok(f64::powi(x - 1.0, 2) + 5.0 * f64::sin(x));
        let args = &mut 0;
        assert!(f(1.0, args).unwrap() > 0.0);
        let mut params = BracketMin::new();
        params.n_iteration_max = 2;
        params.nonlinear_step = false;
        assert_eq!(
            params.try_bracket_min(0.0, args, f).err(),
            Some("try_bracket_min failed to converge")
        );
    }
}
