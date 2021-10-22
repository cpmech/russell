/// Asserts that derivatives have approximately equal values
///
/// # Input
///
/// `dfdx: f64` -- Derivative of f with respect to x at a given x
/// `f: Fn(f64) -> f64` -- (analytical) function f(x) to compute a numerical derivative
/// `x: f64` -- Given x where the derivative is evaluated
/// `tol: f64` -- Error tolerance such that `|dfdx - dfdx_num| < tol`
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// # #[macro_use] extern crate russell_chk;
/// # fn main() {
/// let f = |x: f64| -x;
/// let at_x = 8.0;
/// let dfdx_at_x = -1.01;
/// assert_deriv_approx_eq!(dfdx_at_x, f, at_x, 1e-2);
/// # }
/// ```
///
/// ## Panics due to tighter tolerance
///
/// ```should_panic
/// # #[macro_use] extern crate russell_chk;
/// # fn main() {
/// let f = |x: f64| -x;
/// let at_x = 8.0;
/// let dfdx_at_x = -1.01;
/// assert_deriv_approx_eq!(dfdx_at_x, f, at_x, 1e-3);
/// # }
/// ```
#[macro_export]
macro_rules! assert_deriv_approx_eq {
    ($dfdx:expr, $f:expr, $x:expr, $tol:expr) => {{
        let (dfdx, f, x) = (&$dfdx, &$f, &$x);
        let tol = $tol as f64;
        let dfdx_num = $crate::deriv_central5(*f, *x);
        assert!(
            ((*dfdx - dfdx_num) as f64).abs() < tol,
            "assert derivative failed: `(dfdx != dfdx_num)` \
             (dfdx: `{:?}`, dfdx_num: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *dfdx,
            dfdx_num,
            tol,
            ((*dfdx - dfdx_num) as f64).abs()
        );
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    #[should_panic(expected = "assert derivative failed: `(dfdx != dfdx_num)`")]
    fn panics_on_different_deriv() {
        let f = |x: f64| x * x / 2.0;
        let at_x = 1.5;
        let dfdx_at_x = 1.51;
        assert_deriv_approx_eq!(dfdx_at_x, f, at_x, 1e-2);
    }

    #[test]
    fn accepts_approx_equal_deriv() {
        let f = |x: f64| x * x / 2.0;
        let at_x = 1.5;
        let dfdx_at_x = 1.501;
        assert_deriv_approx_eq!(dfdx_at_x, f, at_x, 1e-2);
    }
}
