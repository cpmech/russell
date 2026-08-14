use super::SmallVector;
use num_traits::Num;

/// Updates a small vector based on another small vector
///
/// ```text
/// v += α⋅u
/// ```
///
/// Note: Only the first `n` components are considered, where `n` is the
/// *active* dimension (`n ≤ N`).
///
/// See also: [`crate::vec_update`] (the heap-allocated counterpart).
///
/// # Input
///
/// * `v` -- the vector that will be updated; only the first `n` components are
///   modified.
/// * `alpha` -- scaling factor for `u`.
/// * `u` -- the other vector.
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
/// use russell_lab::small_vec_update;
///
/// let u = [10.0, 20.0, 30.0];
/// let mut v = [100.0, 200.0, 300.0];
/// small_vec_update(&mut v, 2.0, &u, 3);
/// assert_eq!(v, [120.0, 240.0, 360.0]);
/// ```
#[inline]
pub fn small_vec_update<T, const N: usize>(v: &mut SmallVector<T, N>, alpha: T, u: &SmallVector<T, N>, n: usize)
where
    T: Num + Copy,
{
    assert!(n <= N, "n must be <= N");
    for i in 0..n {
        v[i] = v[i] + alpha * u[i];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::small_vec_update;

    #[test]
    #[should_panic(expected = "n must be <= N")]
    fn small_vec_update_panics_on_wrong_dim() {
        let u = [0.0; 2];
        let mut v = [0.0; 2];
        small_vec_update(&mut v, 1.0, &u, 3); // n = 3 > N = 2
    }

    #[test]
    fn small_vec_update_works() {
        let u = [10.0, 20.0, 30.0];
        let mut v = [100.0, 200.0, 300.0];
        small_vec_update(&mut v, 2.0, &u, 3);
        let correct = [120.0, 240.0, 360.0];
        assert_eq!(v, correct);
    }

    #[test]
    fn small_vec_update_sub_block_works() {
        // only the first 2 components of the vectors are updated
        const NOISE: f64 = 1234.567;
        let u = [1.0; 3];
        let mut v = [NOISE; 3];
        small_vec_update(&mut v, 1.0, &u, 2); // n = 2
        for i in 0..3 {
            let correct = if i < 2 { NOISE + 1.0 } else { NOISE };
            assert_eq!(v[i], correct);
        }
    }
}
