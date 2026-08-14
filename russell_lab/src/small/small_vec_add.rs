use super::SmallVector;
use num_traits::Num;

/// Performs the addition of two small vectors
///
/// ```text
/// w := α⋅u + β⋅v
/// ```
///
/// Note: Only the first `n` components are considered, where `n` is the
/// *active* dimension (`n ≤ N`).
///
/// See also: [`crate::vec_add`] (the heap-allocated counterpart).
///
/// # Input
///
/// * `w` -- the vector that will hold the result; only the first `n`
///   components are overwritten.
/// * `alpha` -- scaling factor for `u`.
/// * `u` -- the first vector.
/// * `beta` -- scaling factor for `v`.
/// * `v` -- the second vector.
/// * `n` -- dimension of the (active) vectors to operate on; must satisfy
///   `n ≤ N`.
///
/// # Panics
///
/// A panic will occur if `n > N`.
///
/// # Examples
///
/// ```
/// use russell_lab::small_vec_add;
///
/// let u = [10.0, 20.0, 30.0, 40.0];
/// let v = [2.0, 1.5, 1.0, 0.5];
/// let mut w = [0.0; 4];
/// small_vec_add(&mut w, 1.0, &u, 2.0, &v, 4);
/// assert_eq!(w, [14.0, 23.0, 32.0, 41.0]);
/// ```
#[inline]
pub fn small_vec_add<T, const N: usize>(
    w: &mut SmallVector<T, N>,
    alpha: T,
    u: &SmallVector<T, N>,
    beta: T,
    v: &SmallVector<T, N>,
    n: usize,
) where
    T: Num + Copy,
{
    assert!(n <= N, "n must be <= N");
    for i in 0..n {
        w[i] = alpha * u[i] + beta * v[i];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::small_vec_add;

    #[test]
    #[should_panic(expected = "n must be <= N")]
    fn small_vec_add_panics_on_wrong_dim() {
        let u = [0.0; 2];
        let v = [0.0; 2];
        let mut w = [0.0; 2];
        small_vec_add(&mut w, 1.0, &u, 1.0, &v, 3); // n = 3 > N = 2
    }

    #[test]
    fn small_vec_add_works() {
        const NOISE: f64 = 1234.567;
        #[rustfmt::skip]
        let u = [
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
        ];
        #[rustfmt::skip]
        let v = [
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
        ];
        let mut w = [NOISE; 8];
        small_vec_add(&mut w, 1.0, &u, -4.0, &v, 8);
        #[rustfmt::skip]
        let correct = [
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
        ];
        assert_eq!(w, correct);
    }

    #[test]
    fn small_vec_add_sub_block_works() {
        // only the first 3 components of the vectors are operated on
        const NOISE: f64 = 1234.567;
        let u = [NOISE; 5];
        let v = [NOISE; 5];
        let mut w = [NOISE; 5];
        small_vec_add(&mut w, 0.0, &u, 0.0, &v, 3); // n = 3
        for i in 0..5 {
            let correct = if i < 3 { 0.0 } else { NOISE };
            assert_eq!(w[i], correct);
        }
    }
}
