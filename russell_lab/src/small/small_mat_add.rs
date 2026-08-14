use super::SmallMatrix;
use num_traits::Num;

/// Performs the addition of two small matrices
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// Note: Only the top-left `n×n` block of the `N×N` matrices is considered,
/// where `n` is the *active* dimension (`n ≤ N`).
///
/// See also: [`crate::mat_add`] (the heap-allocated counterpart).
///
/// # Input
///
/// * `c` -- (N,N) matrix that will hold the result; only the top-left `n×n`
///   block is overwritten.
/// * `alpha` -- scaling factor for `a`.
/// * `a` -- (N,N) matrix.
/// * `beta` -- scaling factor for `b`.
/// * `b` -- (N,N) matrix.
/// * `n` -- dimension of the (active) square matrices to operate on; must
///   satisfy `n ≤ N`.
///
/// # Panics
///
/// A panic will occur if `n > N`.
///
/// # Examples
///
/// ```
/// use russell_lab::small_mat_add;
///
/// let a = [
///     [10.0, 20.0, 0.0, 0.0],
///     [30.0, 40.0, 0.0, 0.0],
///     [ 0.0,  0.0, 9.0, 9.0],
///     [ 0.0,  0.0, 9.0, 9.0],
/// ];
/// let b = [
///     [1.0, 2.0, 0.0, 0.0],
///     [3.0, 4.0, 0.0, 0.0],
///     [0.0, 0.0, 9.0, 9.0],
///     [0.0, 0.0, 9.0, 9.0],
/// ];
/// let mut c = [[0.0; 4]; 4];
/// small_mat_add(&mut c, 1.0, &a, 2.0, &b, 2); // n = 2
/// let correct = [
///     [12.0, 24.0, 0.0, 0.0],
///     [36.0, 48.0, 0.0, 0.0],
///     [ 0.0,  0.0, 0.0, 0.0],
///     [ 0.0,  0.0, 0.0, 0.0],
/// ];
/// assert_eq!(c, correct);
/// ```
#[inline]
pub fn small_mat_add<T, const N: usize>(
    c: &mut SmallMatrix<T, N>,
    alpha: T,
    a: &SmallMatrix<T, N>,
    beta: T,
    b: &SmallMatrix<T, N>,
    n: usize,
) where
    T: Num + Copy,
{
    assert!(n <= N, "n must be <= N");
    for i in 0..n {
        for j in 0..n {
            c[i][j] = alpha * a[i][j] + beta * b[i][j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::small_mat_add;

    #[test]
    #[should_panic(expected = "n must be <= N")]
    fn small_mat_add_panics_on_wrong_dim() {
        let a = [[0.0; 2]; 2];
        let b = [[0.0; 2]; 2];
        let mut c = [[0.0; 2]; 2];
        small_mat_add(&mut c, 1.0, &a, 1.0, &b, 3); // n = 3 > N = 2
    }

    #[test]
    fn small_mat_add_works() {
        const NOISE: f64 = 1234.567;
        #[rustfmt::skip]
        let a = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ];
        #[rustfmt::skip]
        let b = [
            [0.5, 1.0, 1.5],
            [0.5, 1.0, 1.5],
            [0.5, 1.0, 1.5],
        ];
        let mut c = [[NOISE; 3]; 3];
        small_mat_add(&mut c, 1.0, &a, -4.0, &b, 3);
        #[rustfmt::skip]
        let correct = [
            [-1.0, -2.0, -3.0],
            [-1.0, -2.0, -3.0],
            [-1.0, -2.0, -3.0],
        ];
        assert_eq!(c, correct);
    }

    #[test]
    fn small_mat_add_5x5_works() {
        const NOISE: f64 = 1234.567;
        let a = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        let b = [
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
        ];
        let mut c = [[NOISE; 5]; 5];
        small_mat_add(&mut c, 1.0, &a, -4.0, &b, 5);
        #[rustfmt::skip]
        let correct = [
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
        ];
        assert_eq!(c, correct);
    }

    #[test]
    fn small_mat_add_sub_block_works() {
        // only the top-left 2x2 block of the 4x4 matrices is operated on
        const NOISE: f64 = 1234.567;
        let a = [[NOISE; 4]; 4];
        let b = [[NOISE; 4]; 4];
        let mut c = [[NOISE; 4]; 4];
        small_mat_add(&mut c, 0.0, &a, 0.0, &b, 2); // n = 2
        // the top-left 2x2 block is overwritten with 0.0, the rest is untouched
        for i in 0..4 {
            for j in 0..4 {
                let correct = if i < 2 && j < 2 { 0.0 } else { NOISE };
                assert_eq!(c[i][j], correct);
            }
        }
    }
}
