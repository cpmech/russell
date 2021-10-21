/// Estimates the derivative of f with respect to x
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
/// * `dfdx` -- numerical derivative of f(x) w.r.t x @ x
///
/// # Notes
///
/// * Thus function computes the derivative using the 5-point rule
///   (x-h, x-h/2, x, x+h/2, x+h); but the central point is not used
///
/// # Example
///
/// ```
/// use russell_chk::central_deriv;
/// use std::f64::consts::PI;
/// let f = |x: f64| f64::sin(PI * x / 2.0);
/// let g = |x: f64| f64::cos(PI * x / 2.0) * PI / 2.0;
/// let d = central_deriv(f, 1.0, 1e-3);
/// assert!(f64::abs(d - g(1.0)) < 1e-12);
/// ```
pub fn central_deriv<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let fm1 = f(x - h);
    let fp1 = f(x + h);
    let fmh = f(x - h / 2.0);
    let fph = f(x + h / 2.0);
    let r3 = 0.5 * (fp1 - fm1);
    let r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;
    let dfdx = r5 / h;
    dfdx
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::central_deriv;
    use std::f64::consts::PI;

    #[test]
    fn central_deriv_works() {
        let f = |x: f64| f64::cos(PI * x / 2.0);
        let g = |x: f64| -f64::sin(PI * x / 2.0) * PI / 2.0;
        let h = 1e-3;
        let x1 = 1.0;
        let d1 = central_deriv(f, x1, h);
        assert!(f64::abs(d1 - g(x1)) < 1e-12);
    }
}
