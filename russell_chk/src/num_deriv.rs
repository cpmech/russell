/// Initial stepsize h for deriv_central5
pub const STEPSIZE_CENTRAL5: f64 = 1e-3;

/// Computes the numerical derivative and errors using central differences with 5 points
///
/// # Input
///
/// * `at_x` -- location for the derivative of f(x, {arguments}) w.r.t x
/// * `f` -- function f(x, {arguments})
/// * `args` -- extra arguments for f(x, {arguments})
/// * `h` -- stepsize (1e-3 recommended)
///
/// **IMPORTANT:** The function is evaluated in [at_x-h, at_x+h].
///
/// # Output
///
/// Returns the triple (dfdx, abs_trunc_err, abs_round_err) where:
///
/// * `dfdx` -- numerical derivative of f(x) w.r.t x @ x
/// * `abs_trunc_err` -- estimated truncation error O(h²)
/// * `abs_round_err` -- rounding error due to cancellations
///
/// # Notes
///
/// * Computes the derivative using the 5-point rule (at_x-h, at_x-h/2, at_x, at_x+h/2, at_x+h)
///
/// # Example
///
/// ```
/// use russell_chk::deriv_and_errors_central5;
/// struct Arguments {}
/// let f = |x: f64, _: &mut Arguments| f64::exp(-2.0 * x);
/// let args = &mut Arguments {};
/// let at_x = 1.0;
/// let h = 1e-3;
/// let (d, err, rerr) = deriv_and_errors_central5(at_x, args, h, f);
/// let d_correct = -2.0 * f64::exp(-2.0 * at_x);
/// assert!(f64::abs(d - d_correct) < 1e-13);
/// assert!(err < 1e-6);
/// assert!(rerr < 1e-12);
/// ```
pub fn deriv_and_errors_central5<F, A>(at_x: f64, args: &mut A, h: f64, mut f: F) -> (f64, f64, f64)
where
    F: FnMut(f64, &mut A) -> f64,
{
    // numerical derivative
    let fm1 = f(at_x - h, args);
    let fp1 = f(at_x + h, args);
    let fmh = f(at_x - h / 2.0, args);
    let fph = f(at_x + h / 2.0, args);
    let r3 = 0.5 * (fp1 - fm1);
    let r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;
    let dfdx = r5 / h;

    // error estimation
    let e3 = (f64::abs(fp1) + f64::abs(fm1)) * f64::EPSILON;
    let e5 = 2.0 * (f64::abs(fph) + f64::abs(fmh)) * f64::EPSILON + e3;
    let dy = f64::max(f64::abs(r3 / h), f64::abs(r5 / h)) * (f64::abs(at_x) / h) * f64::EPSILON;
    let abs_trunc_err = f64::abs((r5 - r3) / h);
    let abs_round_err = f64::abs(e5 / h) + dy;

    // results
    (dfdx, abs_trunc_err, abs_round_err)
}

/// Computes the numerical derivative using central differences with 5 points
///
/// # Input
///
/// * `at_x` -- location for the derivative of f(x, {arguments}) w.r.t x
/// * `f` -- function f(x, {arguments})
/// * `args` -- extra arguments for f(x, {arguments})
///
/// **IMPORTANT:** The function is evaluated around at_x (with a small tolerance).
///
/// # Output
///
/// * `dfdx` -- numerical derivative of f(x) w.r.t x @ x
///
/// # Notes
///
/// * Computes the derivative using the 5-point rule (at_x-h, at_x-h/2, at_x, at_x+h/2, at_x+h)
/// * A pre-selected stepsize is scaled based on error estimates
///
/// # Example
///
/// ```
/// use russell_chk::deriv_central5;
/// struct Arguments {}
/// let f = |x: f64, _: &mut Arguments| f64::exp(-2.0 * x);
/// let args = &mut Arguments {};
/// let at_x = 1.0;
/// let d = deriv_central5(at_x, args, f);
/// let d_correct = -2.0 * f64::exp(-2.0 * at_x);
/// assert!(f64::abs(d - d_correct) < 1e-11);
/// ```
pub fn deriv_central5<F, A>(at_x: f64, args: &mut A, mut f: F) -> f64
where
    F: FnMut(f64, &mut A) -> f64,
{
    // trial derivative
    let h = STEPSIZE_CENTRAL5;
    let (dfdx, err, rerr) = deriv_and_errors_central5(at_x, args, h, &mut f);
    let err_total = err + rerr;

    // done with zero-error
    if err == 0.0 || rerr == 0.0 {
        return dfdx;
    }

    // done with very small truncation error
    if err < rerr {
        return dfdx;
    }

    // improved derivative
    let h_improv = h * f64::powf(rerr / (2.0 * err), 1.0 / 3.0);
    let (dfdx_improv, err_improv, rerr_improv) = deriv_and_errors_central5(at_x, args, h_improv, &mut f);
    let err_total_improv = err_improv + rerr_improv;

    // ignore improved estimate because of larger error
    if err_total_improv > err_total {
        return dfdx;
    }

    // ignore improved estimate because of out-of-bounds value
    if f64::abs(dfdx_improv - dfdx) > 4.0 * err_total {
        return dfdx;
    }

    // return improved derivative
    dfdx_improv
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{deriv_and_errors_central5, deriv_central5};
    use std::f64::consts::PI;

    struct Arguments {}

    struct TestFunction {
        pub name: &'static str,                // name
        pub f: fn(f64, &mut Arguments) -> f64, // f(x)
        pub g: fn(f64, &mut Arguments) -> f64, // g=df/dx
        pub at_x: f64,                         // @x value
        pub tol_diff: f64,                     // tolerance for |num - ana|
        pub tol_err: f64,                      // tolerance for truncation error
        pub tol_rerr: f64,                     // tolerance for rounding error
        pub improv_tol_diff: f64,              // tolerance for |num - ana|
    }

    fn gen_functions() -> Vec<TestFunction> {
        vec![
            TestFunction {
                name: "x",
                f: |x, _| x,
                g: |_, _| 1.0,
                at_x: 0.0,
                tol_diff: 1e-15,
                tol_err: 1e-15,
                tol_rerr: 1e-15,
                improv_tol_diff: 1e-15,
            },
            TestFunction {
                name: "x²",
                f: |x, _| x * x,
                g: |x, _| 2.0 * x,
                at_x: 1.0,
                tol_diff: 1e-12,
                tol_err: 1e-13,
                tol_rerr: 1e-11,
                improv_tol_diff: 1e-12,
            },
            TestFunction {
                name: "exp(x)",
                f: |x, _| f64::exp(x),
                g: |x, _| f64::exp(x),
                at_x: 2.0,
                tol_diff: 1e-11,
                tol_err: 1e-5,
                tol_rerr: 1e-10,
                improv_tol_diff: 1e-10, // worse
            },
            TestFunction {
                name: "exp(-x²)",
                f: |x, _| f64::exp(-x * x),
                g: |x, _| -2.0 * x * f64::exp(-x * x),
                at_x: 2.0,
                tol_diff: 1e-13,
                tol_err: 1e-6,
                tol_rerr: 1e-13,
                improv_tol_diff: 1e-11, // worse
            },
            TestFunction {
                name: "1/x",
                f: |x, _| 1.0 / x,
                g: |x, _| -1.0 / (x * x),
                at_x: 0.2,
                tol_diff: 1e-8,
                tol_err: 1e-3,
                tol_rerr: 1e-11,
                improv_tol_diff: 1e-9, // better
            },
            TestFunction {
                name: "x⋅√x",
                f: |x, _| x * f64::sqrt(x),
                g: |x, _| 1.5 * f64::sqrt(x),
                at_x: 25.0,
                tol_diff: 1e-10,
                tol_err: 1e-9,
                tol_rerr: 1e-9,
                improv_tol_diff: 1e-10,
            },
            TestFunction {
                name: "sin(1/x)",
                f: |x, _| f64::sin(1.0 / x),
                g: |x, _| -f64::cos(1.0 / x) / (x * x),
                at_x: 0.5,
                tol_diff: 1e-10,
                tol_err: 1e-4,
                tol_rerr: 1e-11,
                improv_tol_diff: 1e-10,
            },
            TestFunction {
                name: "cos(π⋅x/2)",
                f: |x, _| f64::cos(PI * x / 2.0),
                g: |x, _| -f64::sin(PI * x / 2.0) * PI / 2.0,
                at_x: 1.0,
                tol_diff: 1e-12,
                tol_err: 1e-6,
                tol_rerr: 1e-12,
                improv_tol_diff: 1e-10, // worse
            },
        ]
    }

    #[test]
    fn deriv_and_errors_central5_works() {
        let tests = gen_functions();
        println!(
            "{:>10}{:>15}{:>22}{:>11}{:>10}{:>10}",
            "function", "numerical", "analytical", "|num-ana|", "err", "rerr"
        );
        for test in &tests {
            let args = &mut Arguments {};
            let (d, err, rerr) = deriv_and_errors_central5(test.at_x, args, 1e-3, test.f);
            let d_correct = (test.g)(test.at_x, args);
            println!(
                "{:>10}{:15.9}{:22}{:11.2e}{:10.2e}{:10.2e}",
                test.name,
                d,
                d_correct,
                f64::abs(d - d_correct),
                err,
                rerr,
            );
            assert!(f64::abs(d - d_correct) < test.tol_diff);
            assert!(err < test.tol_err);
            assert!(rerr < test.tol_rerr);
        }
    }

    #[test]
    fn deriv_central5_works() {
        let tests = gen_functions();
        println!(
            "{:>10}{:>15}{:>22}{:>11}",
            "function", "numerical", "analytical", "|num-ana|"
        );
        // for test in &[&tests[2]] {
        for test in &tests {
            let args = &mut Arguments {};
            let d = deriv_central5(test.at_x, args, test.f);
            let d_correct = (test.g)(test.at_x, args);
            println!(
                "{:>10}{:15.9}{:22}{:11.2e}",
                test.name,
                d,
                d_correct,
                f64::abs(d - d_correct),
            );
            assert!(f64::abs(d - d_correct) < test.improv_tol_diff);
        }
    }
}
