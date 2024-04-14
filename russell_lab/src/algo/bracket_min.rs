#![allow(unused)]

use crate::StrError;

/// Indicates that no arguments are needed
pub type NoArgs = u8;

/// Holds parameters for a bracket algorithm
#[derive(Clone, Copy, Debug)]
pub struct BracketMinParams {
    /// Max number of iterations
    ///
    /// ```text
    /// n_iteration_max ≥ 2
    /// ```
    pub n_iteration_max: usize,
}

impl BracketMinParams {
    /// Allocates a new instance
    pub fn new() -> Self {
        BracketMinParams { n_iteration_max: 200 }
    }

    /// Validates the parameters
    pub fn validate(&self) -> Result<(), StrError> {
        if self.n_iteration_max < 2 {
            return Err("n_iteration_max must be ≥ 2");
        }
        Ok(())
    }
}

/// Holds statistics for a bracket algorithm
#[derive(Clone, Copy, Debug)]
pub struct BracketMinStats {
    /// Number of calls to f(x) (function evaluations)
    pub n_function: usize,

    /// Number of iterations
    pub n_iterations: usize,
}

impl BracketMinStats {
    /// Allocates a new instance
    pub fn new() -> BracketMinStats {
        BracketMinStats {
            n_function: 0,
            n_iterations: 0,
        }
    }
}

/// Holds the results of a root or minimum bracketing algorithm
///
/// The goal is to bracket a root or minimum of a function.
///
///	A root of a function is bracketed by a pair of points, `a` and `b`, such that
/// the function has opposite sign at those two points, i.e., `f(a) * f(b) < 0.0`
///
///	A minimum is bracketed by a triple of points `a`, `b`, and `c`, such that
/// `f(b) < f(a)` and `f(b) < f(c)`, with `a < b < c`.
#[derive(Clone, Copy, Debug)]
pub struct BracketMin {
    /// Holds the point `a`
    pub a: f64,

    /// Holds the point `b`
    pub b: f64,

    /// Holds the point `c`
    pub c: f64,

    /// Holds the function evaluation `f(x=a)`
    pub fa: f64,

    /// Holds the function evaluation `f(x=b)`
    pub fb: f64,

    /// Holds the function evaluation `f(x=c)`
    pub fc: f64,
}

/// Tolerance to compare nearly identical points
const NEAR_IDENTICAL: f64 = 10.0 * f64::EPSILON;

/// Tries to bracket the minimum of f(x)
///
/// Searches (iteratively) for `a`, `b` and `c` such that:
///
/// ```text
/// f(b) < f(a)  and  f(b) < f(c)
///
/// with a < b < c
/// ```
///
/// I.e., `f(b)` is the minimum in the `[a, b]` interval.
///
/// # Input
///
/// `a0` -- The initial point (must be different than `b0`)
/// `b0` -- The initial point (must be different than `a0`)
/// `params` -- Optional parameters
///
/// Note that `a0 < b0` and `b0 > a0` are accepted (they will be swapped internally).
///
/// # Output
///
/// * `maybe_optimal` -- holds the `a`, `b`, and `c` points and the functions evaluated at those points.
/// * `stats` -- holds some statistics
pub fn try_bracket_min<F, A>(
    a0: f64,
    b0: f64,
    params: Option<BracketMinParams>,
    args: &mut A,
    mut f: F,
) -> Result<(BracketMin, BracketMinStats), StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    // check
    if f64::abs(a0 - b0) < NEAR_IDENTICAL {
        return Err("a0 must be different than b0");
    }

    // parameters
    let par = match params {
        Some(p) => p,
        None => BracketMinParams::new(),
    };
    par.validate()?;

    // allocate stats struct
    let stats = BracketMinStats::new();

    panic!("TODO")
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
    use super::{swap, try_bracket_min, BracketMin, BracketMinParams, NoArgs};
    use crate::algo::testing::get_functions;
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
    fn params_validate_captures_errors() {
        let mut params = BracketMinParams::new();
        params.n_iteration_max = 0;
        assert_eq!(params.validate().err(), Some("n_iteration_max must be ≥ 2"));
    }

    #[test]
    fn try_bracket_min_captures_errors_1() {
        let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
        let args = &mut 0;
        assert_eq!(
            try_bracket_min(0.0, f64::EPSILON, None, args, f).err(),
            Some("a0 must be different than b0")
        );
        let mut params = BracketMinParams::new();
        params.n_iteration_max = 0;
        assert_eq!(
            try_bracket_min(0.0, 1.0, Some(params), args, f).err(),
            Some("n_iteration_max must be ≥ 2")
        );
    }

    #[test]
    #[should_panic]
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
        // first function call
        assert_eq!(try_bracket_min(0.0, 1.0, None, args, f).err(), Some("stop"));
        // second function call
        args.count = 0;
        args.target = 1;
        assert_eq!(try_bracket_min(0.0, 1.0, None, args, f).err(), Some("stop"));
        // third function call
        args.count = 0;
        args.target = 2;
        assert_eq!(try_bracket_min(0.0, 1.0, None, args, f).err(), Some("stop"));
    }

    fn check_consistency(bracket: &BracketMin) {
        if bracket.a >= bracket.b {
            panic!("a should be smaller than b");
        }
        if bracket.c <= bracket.b {
            panic!("c should be greater than b");
        }
        if bracket.fa < bracket.fb {
            panic!("fa should be greater than or equal to fb");
        }
        if bracket.fc < bracket.fb {
            panic!("fc should be greater than or equal to fb");
        }
    }

    #[test]
    #[should_panic]
    fn try_bracket_min_works_1() {
        let args = &mut 0;
        for test in &get_functions() {
            let loc_1 = &test.local_min_1;
            let (bracket, _stats) = try_bracket_min(loc_1.a, loc_1.b, None, args, test.f).unwrap();
            // println!("\n\n========================================================================================");
            println!("\n{}", test.name);
            println!("\n{:?}", _stats);
            println!("\n{:?}", bracket);
            println!("\n{:?}", loc_1);
            check_consistency(&bracket);
            approx_eq((test.f)(bracket.a, args).unwrap(), bracket.fa, 1e-15);
            approx_eq((test.f)(bracket.b, args).unwrap(), bracket.fb, 1e-15);
            approx_eq((test.f)(bracket.c, args).unwrap(), bracket.fc, 1e-15);
        }
    }
}
