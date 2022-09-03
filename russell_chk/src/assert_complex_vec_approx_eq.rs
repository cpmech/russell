/// Asserts that two complex vectors have the same length and approximately equal values
///
/// # Input
///
/// * `a` -- Left vector
/// * `b` -- Right vector
/// * `tol: f64` -- Error tolerance such that `|a[i].re - b[i].re| < tol`
///   and `|a[i].im - b[i].im| < tol` for all `i in [0,a.len()]`
///
/// # Note
///
/// This macro also checks that a.len() == b.len()
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
///     let a = [Complex64::new(1.0, 4.0), Complex64::new(2.0, 5.0), Complex64::new(3.0000001, 6.0)];
///     let b = [Complex64::new(1.0, 4.0), Complex64::new(2.0, 5.0), Complex64::new(3.0, 6.0)];
///     assert_complex_vec_approx_eq!(a, b, 1e-6);
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
///     let a = [Complex64::new(1.0, 4.0), Complex64::new(2.0, 5.0), Complex64::new(3.0, 6.0)];
///     let b = [Complex64::new(1.0, 4.0), Complex64::new(2.0, 5.0), Complex64::new(4.0, 6.0)];
///     assert_complex_vec_approx_eq!(a, b, 1e-6);
/// }
/// ```
///
/// ## Panics on different lengths
///
/// ```should_panic
/// #[macro_use] extern crate russell_chk;
///
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = [Complex64::new(1.0, 4.0), Complex64::new(2.0, 5.0), Complex64::new(3.0, 6.0)];
///     let b = [Complex64::new(1.0, 4.0), Complex64::new(2.0, 5.0)];
///     assert_complex_vec_approx_eq!(a, b, 1e-6);
/// }
/// ```
#[macro_export]
macro_rules! assert_complex_vec_approx_eq {
    ($a:expr, $b:expr, $tol:expr) => {{
        assert!(
            $a.len() == $b.len(),
            "assertion failed: `(left.len() != right.len())` \
             (left: `{:?}`, right: `{:?}`)",
            $a,
            $b,
        );
        for i in 0..$a.len() {
            assert!(
                ((($a[i].re - $b[i].re) as f64).abs() < $tol) && ((($a[i].im - $b[i].im) as f64).abs() < $tol),
                "assertion failed: `(left[{:?}] != right[{:?}])` \
                 (left[{:?}]: `{}`, right[{:?}]: `{}`, \
                 expect diff: `({:?},{:?})`, real diff: `({:?},{:?})`)",
                i,
                i,
                i,
                $a[i],
                i,
                $b[i],
                $tol,
                $tol,
                (($a[i].re - $b[i].re) as f64).abs(),
                (($a[i].im - $b[i].im) as f64).abs(),
            );
        }
    }};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use num_complex::{Complex32, Complex64};

    #[test]
    #[should_panic(expected = "assertion failed: `(left.len() != right.len())` \
                               (left: `[Complex { re: 1.0, im: 5.0 }, Complex { re: 2.0, im: 5.0 }]`, right: `[Complex { re: 1.0, im: 5.0 }]`)")]
    fn panics_on_different_lengths() {
        assert_complex_vec_approx_eq!(
            &[Complex64::new(1.0, 5.0), Complex64::new(2.0, 5.0)],
            &[Complex64::new(1.0, 5.0)],
            1e-1
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left[1] != right[1])` \
                               (left[1]: `2+5i`, right[1]: `2.5+5i`, \
                               expect diff: `(0.1,0.1)`, real diff: `(0.5,0.0)`)")]
    fn panics_on_different_values_re() {
        assert_complex_vec_approx_eq!(
            &[Complex64::new(1.0, 5.0), Complex64::new(2.0, 5.0)],
            &[Complex64::new(1.0, 5.0), Complex64::new(2.5, 5.0)],
            1e-1
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left[1] != right[1])` \
                               (left[1]: `2+5i`, right[1]: `2+5.5i`, \
                               expect diff: `(0.1,0.1)`, real diff: `(0.0,0.5)`)")]
    fn panics_on_different_values_im() {
        assert_complex_vec_approx_eq!(
            &[Complex64::new(1.0, 5.0), Complex64::new(2.0, 5.0)],
            &[Complex64::new(1.0, 5.0), Complex64::new(2.0, 5.5)],
            1e-1
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left.len() != right.len())` \
                               (left: `[Complex { re: 1.0, im: 5.0 }, Complex { re: 2.0, im: 5.0 }]`, right: `[Complex { re: 1.0, im: 5.0 }]`)")]
    fn panics_on_different_lengths_f32() {
        assert_complex_vec_approx_eq!(
            &[Complex32::new(1f32, 5f32), Complex32::new(2f32, 5f32)],
            &[Complex32::new(1f32, 5f32)],
            1e-1
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left[1] != right[1])` \
                               (left[1]: `2+5i`, right[1]: `2.5+5i`, \
                               expect diff: `(0.1,0.1)`, real diff: `(0.5,0.0)`)")]
    fn panics_on_different_values_f32_re() {
        assert_complex_vec_approx_eq!(
            &[Complex32::new(1f32, 5f32), Complex32::new(2f32, 5f32)],
            &[Complex32::new(1f32, 5f32), Complex32::new(2.5f32, 5f32)],
            1e-1
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left[1] != right[1])` \
                               (left[1]: `2+5i`, right[1]: `2+5.5i`, \
                               expect diff: `(0.1,0.1)`, real diff: `(0.0,0.5)`)")]
    fn panics_on_different_values_f32_im() {
        assert_complex_vec_approx_eq!(
            &[Complex32::new(1f32, 5f32), Complex32::new(2f32, 5f32)],
            &[Complex32::new(1f32, 5f32), Complex32::new(2f32, 5.5f32)],
            1e-1
        );
    }

    #[test]
    fn accepts_approx_equal_values() {
        let tol = 0.03;

        let u = [Complex64::new(1.0, 5.0), Complex64::new(2.0, 5.0)];
        let v = [Complex64::new(1.0, 5.0), Complex64::new(2.02, 5.0)];
        assert_complex_vec_approx_eq!(u, v, tol);
        assert_complex_vec_approx_eq!(u, &v, tol);
        assert_complex_vec_approx_eq!(&u, v, tol);
        assert_complex_vec_approx_eq!(&u, &v, tol);

        let u = [Complex64::new(1.0, 5.0), Complex64::new(2.0, 5.0)];
        let v = [Complex64::new(1.0, 5.0), Complex64::new(2.0, 5.02)];
        assert_complex_vec_approx_eq!(u, v, tol);
    }

    #[test]
    fn accepts_approx_equal_values_f32() {
        let tol = 0.03;

        let u = [Complex32::new(1.0, 5.0), Complex32::new(2.0, 5.0)];
        let v = [Complex32::new(1.0, 5.0), Complex32::new(2.02, 5.0)];
        assert_complex_vec_approx_eq!(u, v, tol);
        assert_complex_vec_approx_eq!(u, &v, tol);
        assert_complex_vec_approx_eq!(&u, v, tol);
        assert_complex_vec_approx_eq!(&u, &v, tol);

        let u = [Complex32::new(1.0, 5.0), Complex32::new(2.0, 5.0)];
        let v = [Complex32::new(1.0, 5.0), Complex32::new(2.0, 5.02)];
        assert_complex_vec_approx_eq!(u, v, tol);
    }
}
