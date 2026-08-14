use super::SmallMatrix;
use num_traits::Num;

/// Performs the multiplication of two small matrices
///
/// ```text
///   c  :=  α  a   ⋅   b   +  β  c
/// (n,n)     (n,n)   (n,n)     (n,n)
/// ```
///
/// Note: Only the top-left `n×n` block of the `N×N` matrices is considered,
/// where `n` is the *active* dimension (`n ≤ N`).
///
/// See also: [`crate::mat_mat_mul`] (the heap-allocated counterpart).
///
/// # Input
///
/// * `c` -- (N,N) matrix that will hold the result; only the top-left `n×n`
///   block is overwritten.
/// * `alpha` -- scaling factor for `a ⋅ b`.
/// * `a` -- (N,N) matrix.
/// * `b` -- (N,N) matrix.
/// * `beta` -- scaling factor for `c`.
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
/// use russell_lab::small_mat_mat_mul;
///
/// let a = [
///     [ 1.0,  2.0, 0.0],
///     [ 3.0,  4.0, 0.0],
///     [ 0.0,  0.0, 0.0],
/// ];
/// let b = [
///     [-1.0, -2.0, 0.0],
///     [-4.0, -5.0, 0.0],
///     [ 0.0,  0.0, 0.0],
/// ];
/// let mut c = [[0.0; 3]; 3];
/// small_mat_mat_mul(&mut c, 1.0, &a, &b, 0.0, 2); // n = 2
/// let correct = [
///     [ -9.0, -12.0, 0.0],
///     [-19.0, -26.0, 0.0],
///     [  0.0,   0.0, 0.0],
/// ];
/// assert_eq!(c, correct);
/// ```
#[inline]
pub fn small_mat_mat_mul<T, const N: usize>(
    c: &mut SmallMatrix<T, N>,
    alpha: T,
    a: &SmallMatrix<T, N>,
    b: &SmallMatrix<T, N>,
    beta: T,
    n: usize,
) where
    T: Num + Copy,
{
    assert!(n <= N, "n must be <= N");
    for i in 0..n {
        for j in 0..n {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + a[i][k] * b[k][j];
            }
            c[i][j] = alpha * sum + beta * c[i][j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::small_mat_mat_mul;

    #[test]
    #[should_panic(expected = "n must be <= N")]
    fn small_mat_mat_mul_panics_on_wrong_dim() {
        let a = [[0.0; 2]; 2];
        let b = [[0.0; 2]; 2];
        let mut c = [[0.0; 2]; 2];
        small_mat_mat_mul(&mut c, 1.0, &a, &b, 0.0, 3); // n = 3 > N = 2
    }

    #[test]
    fn small_mat_mat_mul_works_1() {
        // c := 1⋅a⋅b
        let a = [
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let b = [
            [-1.0, -2.0, 0.0, 0.0],
            [-4.0, -5.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ];
        let mut c = [[0.0; 4]; 4];
        small_mat_mat_mul(&mut c, 1.0, &a, &b, 0.0, 4);
        #[rustfmt::skip]
        let correct = [
            [ -9.0, -12.0, 0.0, 0.0],
            [-19.0, -26.0, 0.0, 0.0],
            [  0.0,   0.0, 2.0, 0.0],
            [  0.0,   0.0, 0.0, 3.0],
        ];
        assert_eq!(c, correct);
    }

    #[test]
    fn small_mat_mat_mul_works_2() {
        // c := 2⋅a⋅b + 10⋅c
        let a = [[1.0, 2.0], [3.0, 4.0]];
        let b = [[-1.0, -2.0], [-4.0, -5.0]];
        let mut c = [[100.0; 2]; 2];
        small_mat_mat_mul(&mut c, 2.0, &a, &b, 10.0, 2);
        #[rustfmt::skip]
        let correct = [
            [ 982.0, 976.0],
            [ 962.0, 948.0],
        ];
        assert_eq!(c, correct);
    }

    #[test]
    fn small_mat_mat_mul_sub_block_works() {
        // only the top-left 2x2 block of the 4x4 matrices is operated on
        const NOISE: f64 = 1234.567;
        let a = [[NOISE; 4]; 4];
        let b = [[NOISE; 4]; 4];
        let mut c = [[NOISE; 4]; 4];
        small_mat_mat_mul(&mut c, 0.0, &a, &b, 1.0, 0); // n = 0 (no-op)
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(c[i][j], NOISE);
            }
        }
    }
}
