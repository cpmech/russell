/// Asserts that two numbers are approximately equal to each other
///
/// # Input
///
/// `a` -- Left value
/// `b` -- Right value
/// `tol: f64` -- Error tolerance such that `|a - b| < tol`
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// use russell_chk::assert_approx_eq;
///
/// fn main() {
///     let a = 3.0000001;
///     let b = 3.0;
///     assert_approx_eq!(a, b, 1e-6);
/// }
/// ```
///
/// ## Panics on different value
///
/// ```should_panic
/// use russell_chk::assert_approx_eq;
///
/// fn main() {
///     let a = 1.0;
///     let b = 2.0;
///     assert_approx_eq!(a, b, 1e-6);
/// }
/// ```
#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $tol:expr) => {{
        assert!(
            (($a - $b) as f64).abs() < $tol,
            "assertion failed: `(left != right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            $a,
            $b,
            $tol,
            (($a - $b) as f64).abs()
        );
    }};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    #[test]
    #[should_panic(expected = "assertion failed: `(left != right)` \
                               (left: `2.0`, right: `2.5`, \
                               expect diff: `0.1`, real diff: `0.5`)")]
    fn panics_on_different_values() {
        assert_approx_eq!(2.0, 2.5, 1e-1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left != right)` \
                               (left: `2.0`, right: `2.5`, \
                               expect diff: `0.1`, real diff: `0.5`)")]
    fn panics_on_different_values_f32() {
        assert_approx_eq!(2f32, 2.5f32, 1e-1);
    }

    #[test]
    fn accepts_approx_equal_values() {
        let a = 2.0;
        let b = 2.02;
        let tol = 0.03;
        assert_approx_eq!(a, b, tol);
        assert_approx_eq!(a, &b, tol);
        assert_approx_eq!(&a, b, tol);
        assert_approx_eq!(&a, &b, tol);
    }

    #[test]
    fn accepts_approx_equal_values_f32() {
        assert_approx_eq!(2f32, 2.02f32, 0.03);
    }
}
