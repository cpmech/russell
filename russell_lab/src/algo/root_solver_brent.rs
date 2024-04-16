use super::{AlgoParams, AlgoStats};
use crate::StrError;

/// Employs Brent's method to find the roots of an equation
///
/// See: <https://mathworld.wolfram.com/BrentsMethod.html>
///
/// See also: <https://en.wikipedia.org/wiki/Brent%27s_method>
///
/// # Input
///
/// * `xa` -- initial "bracket" coordinate such that `f(xa) × f(xb) < 0`
/// * `xb` -- initial "bracket" coordinate such that `f(xa) × f(xb) < 0`
/// * `params` -- optional control parameters
/// * `args` -- extra arguments for the callback function
/// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
///
/// **Note:** `xa` must be different from `xb`
///
/// # Output
///
/// Returns `(xo, stats)` where:
///
/// * `xo` -- is the root such that `f(xo) = 0`
/// * `stats` -- some statistics regarding the computations
///
/// # Details
///
/// Based on ZEROIN C math library: <http://www.netlib.org/c/>
/// By: Oleg Keselyov <oleg@ponder.csci.unt.edu, oleg@unt.edu> May 23, 1991
///
/// G.Forsythe, M.Malcolm, C.Moler, Computer methods for mathematical
/// computations. M., Mir, 1980, p.180 of the Russian edition
///
/// The function makes use of the bisection procedure combined with
/// the linear or quadric inverse interpolation.
/// At every step program operates on three abscissae - a, b, and c.
///
/// * b - the last and the best approximation to the root
/// * a - the last but one approximation
/// * c - the last but one or even earlier approximation than a that
///     1. |f(b)| <= |f(c)|
///     2. f(b) and f(c) have opposite signs, i.e. b and c confine the root
///
/// At every step Zeroin selects one of the two new approximations, the
/// former being obtained by the bisection procedure and the latter
/// resulting in the interpolation (if a,b, and c are all different
/// the quadric interpolation is utilized, otherwise the linear one).
/// If the latter (i.e. obtained by the interpolation) point is
/// reasonable (i.e. lies within the current interval `[b,c]` not being
/// too close to the boundaries) it is accepted. The bisection result
/// is used in the other case. Therefore, the range of uncertainty is
/// ensured to be reduced at least by the factor 1.6
pub fn root_solver_brent<F, A>(
    xa: f64,
    xb: f64,
    params: Option<AlgoParams>,
    args: &mut A,
    mut f: F,
) -> Result<(f64, AlgoStats), StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    // check
    if f64::abs(xa - xb) < 10.0 * f64::EPSILON {
        return Err("xa must be different from xb");
    }

    // parameters
    let par = match params {
        Some(p) => p,
        None => AlgoParams::new(),
    };
    par.validate()?;

    // allocate stats struct
    let mut stats = AlgoStats::new();

    // initialization
    let (mut a, mut b) = (xa, xb);
    let (mut fa, mut fb) = (f(a, args)?, f(b, args)?);
    let mut c = a;
    let mut fc = fa;
    stats.n_function += 2;

    // check
    if fa * fb >= -f64::EPSILON {
        return Err("xa and xb must bracket the root and f(xa) × f(xb) < 0");
    }

    // solve
    let mut converged = false;
    for _ in 0..par.n_iteration_max {
        stats.n_iterations += 1;

        // old step
        let step_old = b - a;

        // swap data
        if f64::abs(fc) < f64::abs(fb) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        // tolerance
        let tol = 2.0 * f64::EPSILON * f64::abs(b) + par.tolerance / 2.0;

        // new step
        let mut step_new = (c - b) / 2.0;

        // converged?
        if f64::abs(step_new) <= tol || fb == 0.0 {
            converged = true;
            break;
        }

        // perform interpolation
        if f64::abs(step_old) >= tol && f64::abs(fa) > f64::abs(fb) {
            // delta
            let del = c - b;

            // linear interpolation
            let (mut p, mut q) = if a == c {
                let t0 = fb / fa;
                (del * t0, 1.0 - t0)

            // quadratic inverse interpolation
            } else {
                let t0 = fa / fc;
                let t1 = fb / fc;
                let t2 = fb / fa;
                (
                    t2 * (del * t0 * (t0 - t1) - (b - a) * (t1 - 1.0)),
                    (t0 - 1.0) * (t1 - 1.0) * (t2 - 1.0),
                )
            };

            // fix the sign of p and q
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }

            // update step
            if p < (0.75 * del * q - f64::abs(tol * q) / 2.0) && p < f64::abs(step_old * q / 2.0) {
                step_new = p / q;
            }
        }

        // limit the step
        if f64::abs(step_new) < tol {
            if step_new > 0.0 {
                step_new = tol;
            } else {
                step_new = -tol;
            }
        }

        // update a
        a = b;
        fa = fb;

        // update b
        b += step_new;
        fb = f(b, args)?;
        stats.n_function += 1;

        // update c
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
        }
    }

    // check
    if !converged {
        return Err("root_solver_brent failed to converge");
    }

    // done
    stats.stop_sw_total();
    Ok((b, stats))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::root_solver_brent;
    use crate::algo::testing::get_functions;
    use crate::algo::{AlgoParams, NoArgs};
    use crate::approx_eq;

    #[test]
    fn root_solver_brent_captures_errors_1() {
        let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
        let args = &mut 0;
        assert_eq!(f(1.0, args).unwrap(), 0.0);
        assert_eq!(
            root_solver_brent(-0.5, -0.5, None, args, f).err(),
            Some("xa must be different from xb")
        );
        assert_eq!(
            root_solver_brent(-0.5, -0.5 - 10.0 * f64::EPSILON, None, args, f).err(),
            Some("xa and xb must bracket the root and f(xa) × f(xb) < 0")
        );
        let mut params = AlgoParams::new();
        params.n_iteration_max = 0;
        assert_eq!(
            root_solver_brent(-0.5, 2.0, Some(params), args, f).err(),
            Some("n_iteration_max must be ≥ 2")
        );
    }

    #[test]
    fn root_solver_brent_captures_errors_2() {
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
        assert_eq!(root_solver_brent(-0.5, 2.0, None, args, f).err(), Some("stop"));
        // second function call
        args.count = 0;
        args.target = 1;
        assert_eq!(root_solver_brent(-0.5, 2.0, None, args, f).err(), Some("stop"));
        // third function call
        args.count = 0;
        args.target = 2;
        assert_eq!(root_solver_brent(-0.5, 2.0, None, args, f).err(), Some("stop"));
    }

    #[test]
    fn root_solver_brent_works_1() {
        let args = &mut 0;
        for test in &get_functions() {
            println!("\n\n===========================================================");
            println!("\n{}", test.name);
            if let Some(bracket) = test.root1 {
                let (xo, stats) = root_solver_brent(bracket.a, bracket.b, None, args, test.f).unwrap();
                println!("\nxo = {:?}", xo);
                println!("\n{}", stats);
                approx_eq(xo, bracket.xo, 1e-11);
                approx_eq((test.f)(xo, args).unwrap(), 0.0, test.tol_root);
            }
            if let Some(bracket) = test.root2 {
                let (xo, stats) = root_solver_brent(bracket.a, bracket.b, None, args, test.f).unwrap();
                println!("\nxo = {:?}", xo);
                println!("\n{}", stats);
                approx_eq(xo, bracket.xo, 1e-11);
                approx_eq((test.f)(xo, args).unwrap(), 0.0, test.tol_root);
            }
            if let Some(bracket) = test.root3 {
                let (xo, stats) = root_solver_brent(bracket.a, bracket.b, None, args, test.f).unwrap();
                println!("\nxo = {:?}", xo);
                println!("\n{}", stats);
                approx_eq(xo, bracket.xo, 1e-13);
                approx_eq((test.f)(xo, args).unwrap(), 0.0, test.tol_root);
            }
        }
    }

    #[test]
    fn root_solver_brent_fails_on_non_converged() {
        let f = |x, _: &mut NoArgs| Ok(f64::powi(x - 1.0, 2) + 5.0 * f64::sin(x));
        let args = &mut 0;
        assert!(f(1.0, args).unwrap() > 0.0);
        let mut params = AlgoParams::new();
        params.n_iteration_max = 2;
        assert_eq!(
            root_solver_brent(-2.0, -0.7, Some(params), args, f).err(),
            Some("root_solver_brent failed to converge")
        );
    }
}
