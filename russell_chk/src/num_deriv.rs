/// Estimates the derivative of f with respect to x and associated errors
///
/// ```text
///     df │
/// g = —— │
///     dx │x
/// ```
///
/// # Input
///
/// * `f` -- function f(x)
/// * `x` -- location for the derivative f(x)
/// * `h` -- stepsize
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
/// * The function computes the derivative using the 5-point rule
///   (x-h, x-h/2, x, x+h/2, x+h); but the central point is not used
/// * The truncation error in the r5 approximation is O(h⁴); however,
///   for safety, we estimate the error from r5-r3, which is O(h²).
///   This allows the automatic scaling of h.
///
/// # Example
///
/// ```
/// use russell_chk::num_deriv_and_errors;
/// let f = |x: f64| f64::exp(-2.0 * x);
/// let g = |x: f64| -2.0 * f64::exp(-2.0 * x);
/// let h = 1e-3;
/// let x = 1.0;
/// let (d, err, rerr) = num_deriv_and_errors(f, x, h);
/// assert!(f64::abs(d - g(x)) < 1e-13);
/// assert!(err < 1e-6);
/// assert!(rerr < 1e-11);
/// ```
pub fn num_deriv_and_errors<F>(f: F, x: f64, h: f64) -> (f64, f64, f64)
where
    F: Fn(f64) -> f64,
{
    // numerical derivative
    let fm1 = f(x - h);
    let fp1 = f(x + h);
    let fmh = f(x - h / 2.0);
    let fph = f(x + h / 2.0);
    let r3 = 0.5 * (fp1 - fm1);
    let r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;
    let dfdx = r5 / h;

    // smallest number satisfying 1.0 + EPS > 1.0
    const EPS: f64 = 1.0e-15;

    // error estimation
    let e3 = (f64::abs(fp1) + f64::abs(fm1)) * EPS;
    let e5 = 2.0 * (f64::abs(fph) + f64::abs(fmh)) * EPS + e3;
    let dy = f64::max(f64::abs(r3 / h), f64::abs(r5 / h)) * (f64::abs(x) / h) * EPS;
    let abs_trunc_err = f64::abs((r5 - r3) / h);
    let abs_round_err = f64::abs(e5 / h) + dy;

    // results
    (dfdx, abs_trunc_err, abs_round_err)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::num_deriv_and_errors;
    use std::f64::consts::PI;

    #[test]
    fn num_deriv_and_errors_works() {
        let f = |x: f64| f64::cos(PI * x / 2.0);
        let g = |x: f64| -f64::sin(PI * x / 2.0) * PI / 2.0;
        let h = 1e-3;
        let x = 1.0;
        let (d, err, rerr) = num_deriv_and_errors(f, x, h);
        assert!(f64::abs(d - g(x)) < 1e-12);
        assert!(err < 1e-6);
        assert!(rerr < 1e-11);
    }
}
