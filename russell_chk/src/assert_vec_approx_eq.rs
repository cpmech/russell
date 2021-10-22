/// Asserts that two vectors have the same length and approximately equal values
///
/// # Input
///
/// `a` -- Left vector
/// `b` -- Right vector
/// `tol: f64` -- Error tolerance such that `|a[i] - b[i]| < tol` for all `i in [0,a.len()]`
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
/// # #[macro_use] extern crate russell_chk;
/// # fn main() {
/// let a = [1.0, 2.0, 3.0000001];
/// let b = [1.0, 2.0, 3.0];
/// assert_vec_approx_eq!(a, b, 1e-6);
/// # }
/// ```
///
/// ## Panics on different values
///
/// ```should_panic
/// # #[macro_use] extern crate russell_chk;
/// # fn main() {
/// let a = [1.0, 2.0, 3.0];
/// let b = [1.0, 2.0, 4.0];
/// assert_vec_approx_eq!(a, b, 1e-6);
/// # }
/// ```
///
/// ## Panics on different lengths
///
/// ```should_panic
/// # #[macro_use] extern crate russell_chk;
/// # fn main() {
/// let a = [1.0, 2.0, 3.0];
/// let b = [1.0, 2.0];
/// assert_vec_approx_eq!(a, b, 1e-6);
/// # }
/// ```
#[macro_export]
macro_rules! assert_vec_approx_eq {
    ($a:expr, $b:expr, $tol:expr) => {{
        let (a, b) = (&$a, &$b);
        assert!(
            a.len() == b.len(),
            "assertion failed: `(left.len() != right.len())` \
             (left: `{:?}`, right: `{:?}`)",
            *a,
            *b,
        );
        let tol = $tol as f64;
        for i in 0..a.len() {
            assert!(
                ((a[i] - b[i]) as f64).abs() < tol,
                "assertion failed: `(left[{:?}] != right[{:?}])` \
                 (left[{:?}]: `{:?}`, right[{:?}]: `{:?}`, \
                 expect diff: `{:?}`, real diff: `{:?}`)",
                i,
                i,
                i,
                a[i],
                i,
                b[i],
                tol,
                ((a[i] - b[i]) as f64).abs()
            );
        }
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    #[should_panic(expected = "assertion failed: `(left.len() != right.len())` \
                               (left: `[1.0, 2.0]`, right: `[1.0]`)")]
    fn panics_on_different_lengths() {
        assert_vec_approx_eq!(&[1.0, 2.0], &[1.0], 1e-1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left[1] != right[1])` \
                               (left[1]: `2.0`, right[1]: `2.5`, \
                               expect diff: `0.1`, real diff: `0.5`)")]
    fn panics_on_different_values() {
        assert_vec_approx_eq!(&[1.0, 2.0], &[1.0, 2.5], 1e-1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left.len() != right.len())` \
                               (left: `[1.0, 2.0]`, right: `[1.0]`)")]
    fn panics_on_different_lengths_f32() {
        assert_vec_approx_eq!(&[1f32, 2f32], &[1f32], 1e-1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left[1] != right[1])` \
                               (left[1]: `2.0`, right[1]: `2.5`, \
                               expect diff: `0.1`, real diff: `0.5`)")]
    fn panics_on_different_values_f32() {
        assert_vec_approx_eq!(&[1f32, 2f32], &[1f32, 2.5f32], 1e-1);
    }

    #[test]
    fn accepts_approx_equal_values() {
        assert_vec_approx_eq!(&[1.0, 2.0], &[1.0, 2.02], 0.03);
    }

    #[test]
    fn accepts_approx_equal_values_f32() {
        assert_vec_approx_eq!(&[1f32, 2f32], &[1f32, 2.02f32], 0.03);
    }
}
