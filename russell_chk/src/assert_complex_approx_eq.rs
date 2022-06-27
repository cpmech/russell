/// Asserts that two complex numbers are approximately equal to each other
///
/// # Input
///
/// `a` -- Left value
/// `b` -- Right value
/// `tol: f64` -- Error tolerance such that `|a.re - b.re| < tol` and `|a.im - b.im| < tol`
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// #[macro_use] extern crate russell_chk;
///
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = Complex64::new(3.0000001, 2.0000001);
///     let b = Complex64::new(3.0, 2.0);
///     assert_complex_approx_eq!(a, b, 1e-6);
/// }
/// ```
///
/// ## Panics on different values
///
/// ```should_panic
/// #[macro_use] extern crate russell_chk;
///
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = Complex64::new(1.0, 3.0);
///     let b = Complex64::new(2.0, 3.0);
///     assert_complex_approx_eq!(a, b, 1e-6);
/// }
/// ```
///
/// ```should_panic
/// #[macro_use] extern crate russell_chk;
///
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = Complex64::new(1.0, 3.0);
///     let b = Complex64::new(1.0, 4.0);
///     assert_complex_approx_eq!(a, b, 1e-6);
/// }
/// ```
#[macro_export]
macro_rules! assert_complex_approx_eq {
    ($a:expr, $b:expr, $tol:expr) => {{
        assert!(
            ((($a.re - $b.re) as f64).abs() < $tol) && ((($a.im - $b.im) as f64).abs() < $tol),
            "assertion failed: `(left != right)` \
             (left: `({:?},{:?})`, right: `({:?},{:?})`, expect diff: `({:?},{:?})`, real diff: `({:?},{:?})`)",
            $a.re,
            $a.im,
            $b.re,
            $b.im,
            $tol,
            $tol,
            (($a.re - $b.re) as f64).abs(),
            (($a.im - $b.im) as f64).abs(),
        );
    }};
}

#[cfg(test)]
mod tests {
    use num_complex::{Complex32, Complex64};

    #[test]
    #[should_panic(expected = "assertion failed: `(left != right)` \
                               (left: `(2.0,3.0)`, right: `(2.5,3.0)`, \
                               expect diff: `(0.1,0.1)`, real diff: `(0.5,0.0)`)")]
    fn panics_on_different_values_re() {
        assert_complex_approx_eq!(Complex64::new(2.0, 3.0), Complex64::new(2.5, 3.0), 1e-1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left != right)` \
                               (left: `(2.0,3.0)`, right: `(2.0,3.5)`, \
                               expect diff: `(0.1,0.1)`, real diff: `(0.0,0.5)`)")]
    fn panics_on_different_values_im() {
        assert_complex_approx_eq!(Complex64::new(2.0, 3.0), Complex64::new(2.0, 3.5), 1e-1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left != right)` \
                               (left: `(2.0,3.0)`, right: `(2.5,3.0)`, \
                               expect diff: `(0.1,0.1)`, real diff: `(0.5,0.0)`)")]
    fn panics_on_different_values_f32_re() {
        assert_complex_approx_eq!(Complex32::new(2f32, 3f32), Complex32::new(2.5f32, 3f32), 1e-1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left != right)` \
                               (left: `(2.0,3.0)`, right: `(2.0,3.5)`, \
                               expect diff: `(0.1,0.1)`, real diff: `(0.0,0.5)`)")]
    fn panics_on_different_values_f32_im() {
        assert_complex_approx_eq!(Complex32::new(2f32, 3f32), Complex32::new(2f32, 3.5f32), 1e-1);
    }

    #[test]
    fn accepts_approx_equal_values() {
        let tol = 0.03;

        let a = Complex64::new(2.0, 3.0);
        let b = Complex64::new(2.02, 3.0);
        assert_complex_approx_eq!(a, b, tol);
        assert_complex_approx_eq!(a, &b, tol);
        assert_complex_approx_eq!(&a, b, tol);
        assert_complex_approx_eq!(&a, &b, tol);

        let a = Complex64::new(2.0, 3.0);
        let b = Complex64::new(2.0, 3.02);
        assert_complex_approx_eq!(a, b, tol);
    }

    #[test]
    fn accepts_approx_equal_values_f32() {
        let tol = 0.03;

        let a = Complex32::new(2.0, 3.0);
        let b = Complex32::new(2.0, 3.02);
        assert_complex_approx_eq!(a, b, tol);

        let a = Complex32::new(2.0, 3.0);
        let b = Complex32::new(2.02, 3.0);
        assert_complex_approx_eq!(a, b, tol);
    }
}
