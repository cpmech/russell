use super::{Quadrature, Stats};
use crate::math::{PI, SQRT_EPSILON};
use crate::StrError;

/// Defines the min distance between roots
///
/// This constant is critical to counting the number of roots with
/// the Kronecker-Picard formula---it must be `≥ 1e-7`
pub const ROOTS_MIN_DISTANCE: f64 = 1e-7;

/// Defines the γ constant for the Kronecker-Picard formula
///
/// This constant must be calibrated alongside [KP_CUTOFF] and it affects the
/// convergence of the quadrature method. Note that the Kronecker-Picard (KP)
/// algorithm is quite sensitive near the boundaries coinciding with two roots,
/// since the KP formula is only defined for the open interval (a, b). For example,
/// the function `f(x) = sin(x) in [0, π]` poses challenge to the algorithm because
/// `f(a) ≈ 0` and `f(b) ≈ 0`.
///
/// # Requirements
///
/// 1. `γ` must be a positive number close to 1.0, but small
/// 2. `γ` must be small enough to yield the number of roots close to an integer
/// 3. `γ` must not be too small otherwise the quadrature fails to converge
// const KP_GAMMA: f64 = 0.05; // ok
// const KP_GAMMA: f64 = 0.05; // ok
const KP_GAMMA: f64 = 0.05; // ok

/// Implements algorithms for finding the roots of an equation
#[derive(Clone, Debug)]
pub struct RootSolver {
    /// Max number of iterations
    ///
    /// ```text
    /// n_iteration_max ≥ 2
    /// ```
    pub n_iteration_max: usize,

    /// Tolerance
    ///
    /// e.g., 1e-10
    pub tolerance: f64,

    /// Integrator for the Kronecker-Picard formula (if count is called)
    quadrature: Option<Quadrature>,
}

impl RootSolver {
    /// Allocates a new instance
    pub fn new() -> Self {
        RootSolver {
            n_iteration_max: 100,
            tolerance: 1e-10,
            quadrature: None,
        }
    }

    /// Validates the parameters
    fn validate_params(&self) -> Result<(), StrError> {
        if self.n_iteration_max < 2 {
            return Err("n_iteration_max must be ≥ 2");
        }
        if self.tolerance < 10.0 * f64::EPSILON {
            return Err("the tolerance must be ≥ 10.0 * f64::EPSILON");
        }
        Ok(())
    }

    /// Counts the number of roots using Kronecker-Picard formula
    ///
    /// Counts the number of roots in the open interval `(a, b)`;
    /// i.e., the roots must not on the boundaries.
    ///
    /// # Requirements
    ///
    /// 1. `b > a + SQRT_EPSILON`
    ///
    /// where `SQRT_EPSILON = f64::sqrt(EPSILON) ≈ 1.5e-8`
    ///
    /// # Input
    ///
    /// * `a` -- lower bound
    /// * `b` -- upper bound
    /// * `args` -- extra arguments for the callback functions
    /// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
    /// * `g` -- is the callback function implementing `df/dx(x)` as `g(x, args)`; it returns `g @ x` or it may return an error.
    /// * `h` -- is the callback function implementing `d²f/dx²(x)` as `h(x, args)`; it returns `h @ x` or it may return an error.
    ///
    /// # Output
    ///
    /// Returns `(number_of_roots, stats)` where
    ///
    /// * `number_of_roots` -- is the number of roots in the interval
    /// * `stats` -- some statistics about the computations
    pub fn count<F, G, H, A>(
        &mut self,
        xa: f64,
        xb: f64,
        args: &mut A,
        mut f: F,
        mut g: G,
        mut h: H,
    ) -> Result<(usize, Stats), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
        G: FnMut(f64, &mut A) -> Result<f64, StrError>,
        H: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        // check
        let (mut a, mut b) = (xa, xb);
        if b <= a + SQRT_EPSILON {
            return Err("b must satisfy: b > a + SQRT_EPSILON");
        }

        // shrink the interval so `[a, b]` becomes `(a, b)`
        let mut number_of_roots = 0;
        let mut fa = f(a, args)?;
        if f64::abs(fa) <= SQRT_EPSILON {
            number_of_roots += 1;
            a += 1e-3;
            fa = f(a, args)?;
        }
        let mut fb = f(b, args)?;
        if f64::abs(fb) <= SQRT_EPSILON {
            number_of_roots += 1;
            b -= 1e-3;
            fb = f(b, args)?;
        }

        // mutable reference to Quadrature (allocates it first if needed)
        if self.quadrature.is_none() {
            self.quadrature = Some(Quadrature::new());
            let q = self.quadrature.as_mut().unwrap();
            q.n_iteration_max = 500;
            q.tolerance = 1e-10;
            q.n_gauss = 10;
        };
        let quad = self.quadrature.as_mut().unwrap();

        // apply the Kronecker-Picard formula
        let gg = KP_GAMMA * KP_GAMMA;
        let (kp, mut stats) = quad.integrate(a, b, args, |x, ar| {
            let fx = f(x, ar)?;
            let gx = g(x, ar)?;
            let hx = h(x, ar)?;
            let num = fx * hx - gx * gx;
            let den = fx * fx + gg * gx * gx;
            Ok(num / den)
        })?;
        stats.n_function *= 3; // multiply by 3 because of f, g, and h in the integral

        // compute the number of roots
        let ga = g(a, args)?;
        let gb = g(b, args)?;
        let num = KP_GAMMA * (fa * gb - fb * ga);
        let den = fa * fb + gg * ga * gb;
        // println!("den = {}", den);
        let act = f64::atan(num / den);
        // let act = f64::atan2(num, den);
        println!("act/PI: {}, {}", act / PI, f64::atan2(num, den) / PI);
        let nr = (f64::atan(num / den) - KP_GAMMA * kp) / PI;
        let nr2 = (f64::atan2(num, den) - KP_GAMMA * kp) / PI;

        // round nr
        let diff = f64::abs(nr - 1.0);
        println!("kkp = {}", -KP_GAMMA * kp / PI);
        println!("nr = {}, nr2 = {}", nr, nr2);
        if nr != nr2 {
            println!("😨");
        }
        let nr_round = if diff < 2.0 * f64::EPSILON {
            1.0
        } else if nr < 1.0 {
            0.0
        } else {
            f64::round(nr)
        };

        // truncate nr
        number_of_roots += nr_round as usize;

        // done
        Ok((number_of_roots, stats))
    }

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
    pub fn brent<F, A>(&self, xa: f64, xb: f64, args: &mut A, mut f: F) -> Result<(f64, Stats), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        // check
        if f64::abs(xa - xb) < 10.0 * f64::EPSILON {
            return Err("xa must be different from xb");
        }

        // validate the parameters
        self.validate_params()?;

        // allocate stats struct
        let mut stats = Stats::new();

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
        for _ in 0..self.n_iteration_max {
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
            let tol = 2.0 * f64::EPSILON * f64::abs(b) + self.tolerance / 2.0;

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
            return Err("brent solver failed to converge");
        }

        // done
        stats.stop_sw_total();
        Ok((b, stats))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::RootSolver;
    use crate::algo::testing::get_test_functions;
    use crate::algo::NoArgs;
    use crate::approx_eq;

    #[test]
    fn validate_params_captures_errors() {
        let mut solver = RootSolver::new();
        solver.n_iteration_max = 0;
        assert_eq!(solver.validate_params().err(), Some("n_iteration_max must be ≥ 2"));
        solver.n_iteration_max = 2;
        solver.tolerance = 0.0;
        assert_eq!(
            solver.validate_params().err(),
            Some("the tolerance must be ≥ 10.0 * f64::EPSILON")
        );
    }

    #[test]
    fn count_works_1() {
        let f = |x, _: &mut NoArgs| Ok(x);
        let g = |_, _: &mut NoArgs| Ok(1.0);
        let h = |_, _: &mut NoArgs| Ok(0.0);
        let args = &mut 0;
        let mut solver = RootSolver::new();
        let (nr, stats) = solver.count(0.0, 1.0, args, f, g, h).unwrap();
        println!("\nnr = {:?}", nr);
        println!("\n{}", stats);
        assert_eq!(nr, 1);
    }

    #[test]
    fn count_works_2() {
        let args = &mut 0;
        let mut solver = RootSolver::new();
        for (i, test) in get_test_functions().iter().enumerate() {
            if i == 0 {
                // if i != 3 {
                // if i != 5 {
                // if i != 6 {
                // if i != 7 {
                // if i != 8 {
                // if i != 9 {
                // if i != 13 {
                // if i != 14 {
                continue;
            }
            println!("\n===================================================================");
            println!("\n{}", test.name);
            let (nr, _stats) = solver
                .count(test.range.0, test.range.1, args, test.f, test.g, test.h)
                .unwrap();
            println!("\nnr = {:?} ({})", nr, test.n_root);
            // println!("\n{}", stats);
            // assert_eq!(nr, test.n_root);
        }
        println!("\n===================================================================\n");
    }

    #[test]
    fn brent_captures_errors_1() {
        let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
        let args = &mut 0;
        let mut solver = RootSolver::new();
        assert_eq!(f(1.0, args).unwrap(), 0.0);
        assert_eq!(
            solver.brent(-0.5, -0.5, args, f).err(),
            Some("xa must be different from xb")
        );
        assert_eq!(
            solver.brent(-0.5, -0.5 - 10.0 * f64::EPSILON, args, f).err(),
            Some("xa and xb must bracket the root and f(xa) × f(xb) < 0")
        );
        solver.n_iteration_max = 0;
        assert_eq!(
            solver.brent(-0.5, 2.0, args, f).err(),
            Some("n_iteration_max must be ≥ 2")
        );
    }

    #[test]
    fn brent_captures_errors_2() {
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
        let solver = RootSolver::new();
        // first function call
        assert_eq!(solver.brent(-0.5, 2.0, args, f).err(), Some("stop"));
        // second function call
        args.count = 0;
        args.target = 1;
        assert_eq!(solver.brent(-0.5, 2.0, args, f).err(), Some("stop"));
        // third function call
        args.count = 0;
        args.target = 2;
        assert_eq!(solver.brent(-0.5, 2.0, args, f).err(), Some("stop"));
    }

    #[test]
    fn brent_works_1() {
        let args = &mut 0;
        let solver = RootSolver::new();
        for test in &get_test_functions() {
            println!("\n===================================================================");
            println!("\n{}", test.name);
            if let Some(bracket) = test.root1 {
                let (xo, stats) = solver.brent(bracket.a, bracket.b, args, test.f).unwrap();
                println!("\nxo = {:?}", xo);
                println!("\n{}", stats);
                approx_eq(xo, bracket.xo, 1e-11);
                approx_eq((test.f)(xo, args).unwrap(), 0.0, test.tol_root);
            }
            if let Some(bracket) = test.root2 {
                let (xo, stats) = solver.brent(bracket.a, bracket.b, args, test.f).unwrap();
                println!("\nxo = {:?}", xo);
                println!("\n{}", stats);
                approx_eq(xo, bracket.xo, 1e-11);
                approx_eq((test.f)(xo, args).unwrap(), 0.0, test.tol_root);
            }
            if let Some(bracket) = test.root3 {
                let (xo, stats) = solver.brent(bracket.a, bracket.b, args, test.f).unwrap();
                println!("\nxo = {:?}", xo);
                println!("\n{}", stats);
                approx_eq(xo, bracket.xo, 1e-13);
                approx_eq((test.f)(xo, args).unwrap(), 0.0, test.tol_root);
            }
        }
        println!("\n===================================================================\n");
    }

    #[test]
    fn brent_fails_on_non_converged() {
        let f = |x, _: &mut NoArgs| Ok(f64::powi(x - 1.0, 2) + 5.0 * f64::sin(x));
        let args = &mut 0;
        assert!(f(1.0, args).unwrap() > 0.0);
        let mut solver = RootSolver::new();
        solver.n_iteration_max = 2;
        assert_eq!(
            solver.brent(-2.0, -0.7, args, f).err(),
            Some("brent solver failed to converge")
        );
    }
}
