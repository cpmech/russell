use super::{AlgoParams, AlgoStats};
use crate::math::SQRT_EPSILON;
use crate::StrError;

/// Golden section ratio: (3 - sqrt(5)) / 2
const GSR: f64 = 0.38196601125010515179541316563436188227969082019424;

/// Employs Brent's method to find the minimum of a function
///
/// See: <https://mathworld.wolfram.com/BrentsMethod.html>
///
/// See also: <https://en.wikipedia.org/wiki/Brent%27s_method>
///
/// # Input
///
/// * `xa` -- first coordinate of the "bracket" containing the local minimum
/// * `xb` -- second coordinate of the "bracket" containing the local minimum
///
/// **Note:** `xa < xb` or `xa > xb` are accepted. However, `xa` must be different from `xb`.
///
/// # Output
///
/// Returns `(xo, stats)` where:
///
/// * `xo` -- is the coordinate of the minimum
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
/// The function makes use of the "gold section" procedure combined with
/// the parabolic interpolation.
/// At every step program operates three abscissae - x,v, and w.
/// * x - the last and the best approximation to the minimum location,
///       i.e. f(x) <= f(a) or/and f(x) <= f(b)
///       (if the function f has a local minimum in (a,b), then the both
///       conditions are fulfilled after one or two steps).
///
/// v,w are previous approximations to the minimum location. They may
/// coincide with a, b, or x (although the algorithm tries to make all
/// u, v, and w distinct). Points x, v, and w are used to construct
/// interpolating parabola whose minimum will be treated as a new
/// approximation to the minimum location if the former falls within
/// `[a,b]` and reduces the range enveloping minimum more efficient than
/// the gold section procedure.
///
/// When f(x) has a second derivative positive at the minimum location
/// (not coinciding with a or b) the procedure converges super-linearly
/// at a rate order about 1.324
///
/// The function always obtains a local minimum which coincides with
/// the global one only if a function under investigation being
/// uni-modular. If a function being examined possesses no local minimum
/// within the given range, The code returns 'a' (if f(a) < f(b)), otherwise
/// it returns the right range boundary value b.
pub fn min_solver_brent<F, A>(
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
    let (mut a, mut b) = if xa < xb { (xa, xb) } else { (xb, xa) };
    let mut v = a + GSR * (b - a);
    let mut fv = f(v, args)?;
    stats.n_function += 1;

    // auxiliary
    let mut x = v;
    let mut w = v;
    let mut fx = fv;
    let mut fw = fv;

    // solve
    let mut converged = false;
    for _ in 0..par.n_iteration_max {
        stats.n_iterations += 1;

        // auxiliary variables
        let del = b - a;
        let mid = (a + b) / 2.0;
        let tol = SQRT_EPSILON * f64::abs(x) + par.tolerance / 3.0;

        // converged?
        if f64::abs(x - mid) + del / 2.0 <= 2.0 * tol {
            converged = true;
            break;
        }

        // gold section step
        let mut tmp = a - x;
        if x < mid {
            tmp = b - x;
        }
        let mut step_new = GSR * tmp;

        // try interpolation
        if f64::abs(x - w) >= tol {
            let t = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * t;
            let mut q = 2.0 * (q - t);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            if f64::abs(p) < f64::abs(step_new * q) && p > q * (a - x + 2.0 * tol) && p < q * (b - x - 2.0 * tol) {
                step_new = p / q;
            }
        }

        // adjust the step
        if f64::abs(step_new) < tol {
            if step_new > 0.0 {
                step_new = tol;
            } else {
                step_new = -tol;
            }
        }

        // next approximation
        let t = x + step_new;
        let ft = f(t, args)?;
        stats.n_function += 1;

        // t is a better approximation
        if ft <= fx {
            if t < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            w = x;
            x = t;
            fv = fw;
            fw = fx;
            fx = ft;

        // x remains the better approx
        } else {
            if t < x {
                a = t;
            } else {
                b = t;
            }
            if ft <= fw || w == x {
                v = w;
                w = t;
                fv = fw;
                fw = ft;
            } else if ft <= fv || v == x || v == w {
                v = t;
                fv = ft;
            }
        }
    }

    // check
    if !converged {
        return Err("min_solver_brent failed to converge");
    }

    // done
    stats.stop_sw_total();
    Ok((x, stats))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::min_solver_brent;
    use crate::algo::testing::get_functions;
    use crate::algo::{AlgoParams, NoArgs};
    use crate::approx_eq;

    #[test]
    fn min_solver_brent_captures_errors_1() {
        let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
        let args = &mut 0;
        assert_eq!(f(1.0, args).unwrap(), 0.0);
        assert_eq!(
            min_solver_brent(-0.5, -0.5, None, args, f).err(),
            Some("xa must be different from xb")
        );
        let mut params = AlgoParams::new();
        params.n_iteration_max = 0;
        assert_eq!(
            min_solver_brent(-0.5, 2.0, Some(params), args, f).err(),
            Some("n_iteration_max must be ≥ 2")
        );
    }

    #[test]
    fn min_solver_brent_captures_errors_2() {
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
        assert_eq!(min_solver_brent(-0.5, 2.0, None, args, f).err(), Some("stop"));
        // second function call
        args.count = 0;
        args.target = 1;
        assert_eq!(min_solver_brent(-0.5, 2.0, None, args, f).err(), Some("stop"));
        /*
        // third function call
        args.count = 0;
        args.target = 2;
        assert_eq!(min_solver_brent(-0.5, 2.0, None, args, f).err(), Some("stop"));
        */
    }

    #[test]
    fn min_solver_brent_works_1() {
        // Solving the first problem with Python/SciPy
        // (too many iterations for such simple problem!)
        //
        // ```python
        // import scipy.optimize as opt
        // def f(x): return (x**2.0-1.0)
        // opt.minimize_scalar(f,bracket=(-5.0,5.0),method='brent')
        // ```
        //
        // Output
        //
        // ```text
        //  message:
        //           Optimization terminated successfully;
        //           The returned value satisfies the termination criteria
        //           (using xtol = 1.48e-08 )
        //  success: True
        //      fun: -1.0
        //        x: 3.5919470973405176e-11
        //      nit: 38
        //     nfev: 41
        // ```
        let args = &mut 0;
        for (i, test) in get_functions().iter().enumerate() {
            println!("\n\n===========================================================");
            println!("\n{}", test.name);
            if let Some(bracket) = test.min1 {
                let (a, b) = if i % 2 == 0 {
                    (bracket.a, bracket.b)
                } else {
                    (bracket.b, bracket.a)
                };
                let (xo, stats) = min_solver_brent(a, b, None, args, test.f).unwrap();
                println!("\nxo = {:?}", xo);
                println!("\n{}", stats);
                approx_eq(xo, bracket.xo, test.tol_min);
                approx_eq((test.f)(xo, args).unwrap(), bracket.fxo, 1e-15);
            }
            if let Some(bracket) = test.min2 {
                let (a, b) = if i % 2 == 0 {
                    (bracket.a, bracket.b)
                } else {
                    (bracket.b, bracket.a)
                };
                let (xo, stats) = min_solver_brent(a, b, None, args, test.f).unwrap();
                println!("\nxo = {:?}", xo);
                println!("\n{}", stats);
                approx_eq(xo, bracket.xo, test.tol_min);
                approx_eq((test.f)(xo, args).unwrap(), bracket.fxo, 1e-15);
            }
            if let Some(bracket) = test.min3 {
                let (a, b) = (bracket.a, bracket.b);
                let (xo, stats) = min_solver_brent(a, b, None, args, test.f).unwrap();
                println!("\nxo = {:?}", xo);
                println!("\n{}", stats);
                approx_eq(xo, bracket.xo, test.tol_min);
                approx_eq((test.f)(xo, args).unwrap(), bracket.fxo, 1e-15);
            }
        }
    }

    #[test]
    fn min_solver_brent_fails_on_non_converged() {
        let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
        let args = &mut 0;
        assert_eq!(f(0.0, args).unwrap(), -1.0);
        let mut params = AlgoParams::new();
        params.n_iteration_max = 2;
        assert_eq!(
            min_solver_brent(-5.0, 5.0, Some(params), args, f).err(),
            Some("min_solver_brent failed to converge")
        );
    }
}
