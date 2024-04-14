use crate::StrError;

/// Indicates that no arguments are needed
pub type NoArgs = u8;

/// Holds parameters for a bracket algorithm
#[derive(Clone, Copy, Debug)]
pub struct BracketParams {
    /// Max number of iterations
    ///
    /// ```text
    /// n_iteration_max ≥ 2
    /// ```
    pub n_iteration_max: usize,

    /// Maximum magnification allowed for a step with the parabolic fit
    ///
    /// ```text
    /// ratio_limit ≥ 10
    /// ```
    pub ratio_limit: f64,

    /// Ratio by which successive intervals are magnified
    golden_ratio: f64,
}

impl BracketParams {
    /// Allocates a new instance
    pub fn new() -> Self {
        BracketParams {
            n_iteration_max: 200,
            ratio_limit: 100.0,
            golden_ratio: (1.0 + f64::sqrt(5.0)) / 2.0,
        }
    }

    /// Validates the parameters
    pub fn validate(&self) -> Result<(), StrError> {
        if self.n_iteration_max < 2 {
            return Err("n_iteration_max must be ≥ 2");
        }
        if self.ratio_limit < 10.0 {
            return Err("n_iteration_max must be ≥ 10.0");
        }
        Ok(())
    }
}

/// Holds statistics for a bracket algorithm
#[derive(Clone, Copy, Debug)]
pub struct BracketStats {
    /// Number of calls to f(x) (function evaluations)
    pub n_function: usize,

    /// Number of iterations
    pub n_iterations: usize,
}

impl BracketStats {
    /// Allocates a new instance
    pub fn new() -> BracketStats {
        BracketStats {
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
pub struct Bracket {
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
/// The algorithm is based on Reference #1, where the search is performed in the
/// "downhill" direction (defined by the function evaluated at the initial points).
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
///
///	# Reference
///
///	* Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes:
///	  The Art of Scientific Computing. Third Edition. Cambridge University Press. 1235p.
pub fn try_bracket_min<F, A>(
    a0: f64,
    b0: f64,
    params: Option<BracketParams>,
    args: &mut A,
    mut f: F,
) -> Result<(Bracket, BracketStats), StrError>
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
        None => BracketParams::new(),
    };
    par.validate()?;

    // allocate stats struct
    let mut stats = BracketStats::new();

    // select a and b such that the search goes "downhill"
    stats.n_function += 2;
    let (fa0, fb0) = (f(a0, args)?, f(b0, args)?);
    let (mut a, mut b, mut fa, mut fb) = if fb0 > fa0 {
        (b0, a0, fb0, fa0)
    } else {
        (a0, b0, fa0, fb0)
    };

    // guess c
    let mut c = b + par.golden_ratio * (b - a);
    let mut fc = f(c, args)?;
    stats.n_function += 1;

    // perform iterative search
    let mut converged = false;
    for it in 0..par.n_iteration_max {
        // check convergence
        if fb <= fc {
            stats.n_iterations = it + 1;
            converged = true;
            break;
        }

        // parabolic extrapolation
        let r = (b - a) * (fb - fc);
        let q = (b - c) * (fb - fa);
        let den = 2.0 * f64::copysign(f64::max(f64::abs(q - r), NEAR_IDENTICAL), q - r);
        let mut u = b - ((b - c) * q - (b - a) * r) / den;
        let mut fu = f(u, args)?;
        let u_lim = b + par.ratio_limit * (c - b);
        stats.n_function += 1;

        // search minimum
        if (b - u) * (u - c) > 0.0 {
            if fu < fc {
                // minimum between u and c
                a = b;
                b = u;
                fa = fb;
                fb = fu;
                stats.n_iterations = it + 1;
                converged = true;
                break;
            } else if fu > fb {
                // minimum between a and u
                c = u;
                fc = fu;
                stats.n_iterations = it + 1;
                converged = true;
                break;
            }
            // parabolic fit failed; use default magnification
            u = c + par.golden_ratio * (c - b);
            fu = f(u, args)?;
            stats.n_function += 1;
        } else if (c - u) * (u - u_lim) > 0.0 {
            // parabolic fit is between c and the limit
            if fu < fc {
                let aux = u + par.golden_ratio * (u - c);
                let fu_copy = fu;
                shift3(&mut b, &mut c, &mut u, aux);
                shift3(&mut fb, &mut fc, &mut fu, fu_copy);
            }
        } else if (u - u_lim) * (u_lim - c) >= 0.0 {
            // limit parabolic u to maximum allowed value
            u = u_lim;
            fu = f(u, args)?;
            stats.n_function += 1;
        } else {
            // reject parabolic u; use default magnification
            u = c + par.golden_ratio * (c - b);
            fu = f(u, args)?;
            stats.n_function += 1;
        }

        // eliminate oldest point and continue
        shift3(&mut a, &mut b, &mut c, u);
        shift3(&mut fa, &mut fb, &mut fc, fu);
    }

    // check iterations
    if !converged {
        return Err("try_min failed to converge");
    }

    // sort the results
    if c < a {
        swap(&mut a, &mut c);
        swap(&mut fa, &mut fc);
    }
    Ok((Bracket { a, b, c, fa, fb, fc }, stats))
}

/// Swaps two numbers
#[inline]
pub(super) fn swap(a: &mut f64, b: &mut f64) {
    let a_copy = a.clone();
    *a = *b;
    *b = a_copy;
}

/// Shifts three numbers
#[inline]
pub(super) fn shift3(a: &mut f64, b: &mut f64, c: &mut f64, d: f64) {
    *a = *b;
    *b = *c;
    *c = d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{shift3, swap, try_bracket_min, Bracket};
    use crate::{algo::testing::get_functions, approx_eq};

    #[test]
    fn swap_works() {
        let mut a = 12.34;
        let mut b = 56.78;
        swap(&mut a, &mut b);
        assert_eq!(a, 56.78);
        assert_eq!(b, 12.34);
    }

    #[test]
    fn shift3_works() {
        let mut a = 1.0;
        let mut b = 2.0;
        let mut c = 3.0;
        let d = 4.0;
        shift3(&mut a, &mut b, &mut c, d);
        assert_eq!(a, 2.0);
        assert_eq!(b, 3.0);
        assert_eq!(c, 4.0);
    }

    fn check_consistency(bracket: &Bracket) {
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
    fn try_bracket_min_works_1() {
        let args = &mut 0;
        for test in &get_functions() {
            let loc_1 = &test.local_min_1;
            let (bracket, _stats) = try_bracket_min(loc_1.a, loc_1.b, None, args, test.f).unwrap();
            // println!("\n\n========================================================================================");
            // println!("\n{}", test.name);
            // println!("\n{:?}", _stats);
            // println!("\n{:?}", bracket);
            // println!("\n{:?}", loc_1);
            check_consistency(&bracket);
            approx_eq((test.f)(bracket.a, args).unwrap(), bracket.fa, 1e-15);
            approx_eq((test.f)(bracket.b, args).unwrap(), bracket.fb, 1e-15);
            approx_eq((test.f)(bracket.c, args).unwrap(), bracket.fc, 1e-15);
        }
    }
}
