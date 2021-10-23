/// Asserts that derivatives have approximately equal values
///
/// # Input
///
/// * `dfdx: f64` -- derivative of f with respect to x at a given x
/// * `at_x: f64` -- location for the derivative of f(x, {arguments}) w.r.t x
/// * `f: fn(f64, &A) -> f64` -- function `f(x: f64, args: &A)`
/// * `tol: f64` -- Error tolerance such that `|dfdx - dfdx_num| < tol`
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// # #[macro_use] extern crate russell_chk;
/// # fn main() {
/// struct Arguments {}
/// let f = |x: f64, _: &mut Arguments| -x;
/// let args = &mut Arguments {};
/// let at_x = 8.0;
/// let dfdx = -1.01;
/// assert_deriv_approx_eq!(dfdx, at_x, f, args, 1e-2);
/// # }
/// ```
///
/// ## Panics due to tighter tolerance
///
/// ```should_panic
/// # #[macro_use] extern crate russell_chk;
/// # fn main() {
/// struct Arguments {}
/// let f = |x: f64, _: &mut Arguments| -x;
/// let args = &mut Arguments {};
/// let at_x = 8.0;
/// let dfdx = -1.01;
/// assert_deriv_approx_eq!(dfdx, at_x, f, args, 1e-3);
/// # }
/// ```
#[macro_export]
macro_rules! assert_deriv_approx_eq {
    ($dfdx:expr, $at_x:expr, $f:expr, $args:expr, $tol:expr) => {{
        let dfdx_num = $crate::deriv_central5($at_x, $f, $args);
        assert!(
            (($dfdx - dfdx_num) as f64).abs() < $tol,
            "assert derivative failed: `(dfdx != dfdx_num)` \
             (dfdx: `{:?}`, dfdx_num: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            $dfdx,
            dfdx_num,
            $tol,
            (($dfdx - dfdx_num) as f64).abs()
        );
    }};
}

#[cfg(test)]
mod tests {
    struct Arguments {}

    #[test]
    #[should_panic(expected = "assert derivative failed: `(dfdx != dfdx_num)`")]
    fn panics_on_different_deriv() {
        let f = |x: f64, _: &mut Arguments| x * x / 2.0;
        let args = &mut Arguments {};
        let at_x = 1.5;
        let dfdx = 1.51;
        assert_deriv_approx_eq!(dfdx, at_x, f, args, 1e-2);
    }

    #[test]
    fn accepts_approx_equal_deriv() {
        let f = |x: f64, _: &mut Arguments| x * x / 2.0;
        let args = &mut Arguments {};
        let at_x = 1.5;
        let dfdx = 1.501;
        assert_deriv_approx_eq!(dfdx, at_x, f, args, 1e-2);
        assert_deriv_approx_eq!(dfdx, at_x, f, args, 1e-2);
    }
}
