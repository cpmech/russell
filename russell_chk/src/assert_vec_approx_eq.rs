/// Asserts that two vectors have the same length and approximately equal values.
///
/// On panic, this macro will print the values of the expressions with their
/// debug representations.
///
/// A tolerance must be given for the absolute comparison of float-point numbers.
///
/// Code inspired by [assert_approx_eq](https://github.com/ashleygwilliams/assert_approx_eq)
///
/// # Examples
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
    fn it_should_panic_on_different_lengths() {
        assert_vec_approx_eq!(&[1.0, 2.0], &[1.0], 1e-1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left[1] != right[1])` \
                               (left[1]: `2.0`, right[1]: `2.5`, \
                               expect diff: `0.1`, real diff: `0.5`)")]
    fn it_should_panic_on_different_values() {
        assert_vec_approx_eq!(&[1.0, 2.0], &[1.0, 2.5], 1e-1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left.len() != right.len())` \
                               (left: `[1.0, 2.0]`, right: `[1.0]`)")]
    fn it_should_panic_on_different_lengths_f32() {
        assert_vec_approx_eq!(&[1f32, 2f32], &[1f32], 1e-1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left[1] != right[1])` \
                               (left[1]: `2.0`, right[1]: `2.5`, \
                               expect diff: `0.1`, real diff: `0.5`)")]
    fn it_should_panic_on_different_values_f32() {
        assert_vec_approx_eq!(&[1f32, 2f32], &[1f32, 2.5f32], 1e-1);
    }

    #[test]
    fn it_should_accept_approx_equal_values() {
        assert_vec_approx_eq!(&[1.0, 2.0], &[1.0, 2.01], 1e-2);
    }

    #[test]
    fn it_should_accept_approx_equal_values_f32() {
        assert_vec_approx_eq!(&[1f32, 2f32], &[1f32, 2.01f32], 1e-2);
    }
}
