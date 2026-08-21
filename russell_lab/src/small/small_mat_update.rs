use super::SmallMatrix;
use num_traits::Num;

/// Updates a small matrix based on another small matrix
///
/// ```text
/// b += α⋅a
/// ```
///
/// Note: Only the top-left `n×n` block of the `N×N` matrices is considered,
/// where `n` is the *active* dimension (`n ≤ N`).
///
/// See also: [`crate::mat_update`] (the heap-allocated counterpart).
///
/// # Input
///
/// * `b` -- (N,N) matrix that will be updated; only the top-left `n×n` block
///   is modified.
/// * `alpha` -- scaling factor for `a`.
/// * `a` -- (N,N) matrix.
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
/// use russell_lab::small_mat_update;
///
/// let a = [
///     [10.0, 20.0, 30.0],
///     [40.0, 50.0, 60.0],
///     [ 0.0,  0.0,  0.0],
/// ];
/// let mut b = [
///     [100.0, 200.0, 300.0],
///     [400.0, 500.0, 600.0],
///     [  0.0,   0.0,   0.0],
/// ];
/// small_mat_update(&mut b, 2.0, &a, 3); // n = 3
/// let correct = [
///     [120.0, 240.0, 360.0],
///     [480.0, 600.0, 720.0],
///     [  0.0,   0.0,   0.0],
/// ];
/// assert_eq!(b, correct);
/// ```
#[inline]
pub fn small_mat_update<T, const N: usize>(b: &mut SmallMatrix<T, N>, alpha: T, a: &SmallMatrix<T, N>, n: usize)
where
    T: Num + Copy,
{
    assert!(n <= N, "n must be <= N");
    for i in 0..n {
        for j in 0..n {
            b[i][j] = b[i][j] + alpha * a[i][j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::small_mat_update;

    #[test]
    #[should_panic(expected = "n must be <= N")]
    fn small_mat_update_panics_on_wrong_dim() {
        let a = [[0.0; 2]; 2];
        let mut b = [[0.0; 2]; 2];
        small_mat_update(&mut b, 1.0, &a, 3); // n = 3 > N = 2
    }

    #[test]
    fn small_mat_update_works() {
        #[rustfmt::skip]
        let a = [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [70.0, 80.0, 90.0],
        ];
        #[rustfmt::skip]
        let mut b = [
            [100.0, 200.0, 300.0],
            [400.0, 500.0, 600.0],
            [700.0, 800.0, 900.0],
        ];
        small_mat_update(&mut b, 2.0, &a, 3);
        #[rustfmt::skip]
        let correct = [
            [120.0, 240.0, 360.0],
            [480.0, 600.0, 720.0],
            [840.0, 960.0, 1080.0],
        ];
        assert_eq!(b, correct);
    }

    #[test]
    fn small_mat_update_sub_block_works() {
        // only the top-left 2x2 block of the 3x3 matrices is updated
        const NOISE: f64 = 1234.567;
        let a = [[1.0; 3]; 3];
        let mut b = [[NOISE; 3]; 3];
        small_mat_update(&mut b, 1.0, &a, 2); // n = 2
        for i in 0..3 {
            for j in 0..3 {
                let correct = if i < 2 && j < 2 { NOISE + 1.0 } else { NOISE };
                assert_eq!(b[i][j], correct);
            }
        }
    }
}
