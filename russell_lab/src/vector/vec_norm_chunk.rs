use super::Vector;
use crate::Norm;

/// Returns the vector norm of a chunk of the vector
///
/// Effectively, this function uses a slice such as:
///
/// ```text
/// norm(v, start, stop) = ‖&v[start..stop]‖
/// ```
///
/// Note that `stop` is exclusive, i.e., the slice goes up to `stop - 1`.
///
/// Requirements: `start` must be < `stop` and `stop` must be ≤ `v.dim()`.
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_norm_chunk, Norm, Vector};
/// use russell_lab::approx_eq;
/// use russell_lab::math::SQRT_3;
///
/// fn main() {
///     let u = Vector::from(&[8.0, -2.0, 2.0, -2.0, -3.0]);
///     // Note that &u[1..4] = [-2.0, 2.0, -2.0]
///     approx_eq(vec_norm_chunk(&u, Norm::Euc, 1, 4), 2.0 * SQRT_3, 1e-15);
///     assert_eq!(vec_norm_chunk(&u, Norm::One, 1, 4), 6.0);
///     assert_eq!(vec_norm_chunk(&u, Norm::Max, 1, 4), 2.0);
/// }
/// ```
pub fn vec_norm_chunk(v: &Vector, kind: Norm, start: usize, stop: usize) -> f64 {
    if v.dim() == 0 {
        return 0.0;
    }
    assert!(start < stop, "start must be < stop");
    assert!(stop <= v.dim(), "stop must be ≤ v.dim");
    let slice = &v.as_data()[start..stop];
    match kind {
        Norm::Euc | Norm::Fro => crate::base::blas_dispatch::dnrm2(slice),
        // `idamax` returns a 0-based index here, so this is `abs` of the largest
        // entry, which is the infinity norm.
        Norm::Inf | Norm::Max => f64::abs(slice[crate::base::blas_dispatch::idamax(slice)]),
        Norm::One => crate::base::blas_dispatch::dasum(slice),
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Vector, vec_norm_chunk};
    use crate::{Norm, approx_eq, math::SQRT_6};

    #[test]
    fn vec_norm_chunk_works_full() {
        let u0 = Vector::new(0);
        assert_eq!(vec_norm_chunk(&u0, Norm::Euc, 0, u0.dim()), 0.0);
        assert_eq!(vec_norm_chunk(&u0, Norm::Fro, 0, u0.dim()), 0.0);
        assert_eq!(vec_norm_chunk(&u0, Norm::Inf, 0, u0.dim()), 0.0);
        assert_eq!(vec_norm_chunk(&u0, Norm::Max, 0, u0.dim()), 0.0);
        assert_eq!(vec_norm_chunk(&u0, Norm::One, 0, u0.dim()), 0.0);

        let u = Vector::from(&[-3.0, 2.0, 1.0, 1.0, 1.0]);
        assert_eq!(vec_norm_chunk(&u, Norm::Euc, 0, u.dim()), 4.0);
        assert_eq!(vec_norm_chunk(&u, Norm::Fro, 0, u.dim()), 4.0);
        assert_eq!(vec_norm_chunk(&u, Norm::Inf, 0, u.dim()), 3.0);
        assert_eq!(vec_norm_chunk(&u, Norm::Max, 0, u.dim()), 3.0);
        assert_eq!(vec_norm_chunk(&u, Norm::One, 0, u.dim()), 8.0);

        // example from https://netlib.org/lapack/lug/node75.html
        let diff = Vector::from(&[-0.1, 1.0, -2.0]);
        approx_eq(vec_norm_chunk(&diff, Norm::Euc, 0, diff.dim()), 2.238, 0.001);
        assert_eq!(vec_norm_chunk(&diff, Norm::Inf, 0, diff.dim()), 2.0);
        assert_eq!(vec_norm_chunk(&diff, Norm::One, 0, diff.dim()), 3.1);
    }

    #[test]
    fn vec_norm_chunk_works_full_neg() {
        let u = Vector::from(&[-3.0, -2.0, -1.0, -1.0, -1.0]);
        assert_eq!(vec_norm_chunk(&u, Norm::Euc, 0, u.dim()), 4.0);
        assert_eq!(vec_norm_chunk(&u, Norm::Fro, 0, u.dim()), 4.0);
        assert_eq!(vec_norm_chunk(&u, Norm::Inf, 0, u.dim()), 3.0);
        assert_eq!(vec_norm_chunk(&u, Norm::Max, 0, u.dim()), 3.0);
        assert_eq!(vec_norm_chunk(&u, Norm::One, 0, u.dim()), 8.0);
    }

    #[test]
    fn vec_norm_chunk_works_full_large() {
        let u = Vector::from(&[
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 3.0, 2.0, 1.0, 1.0, 1.0,
        ]);
        approx_eq(vec_norm_chunk(&u, Norm::Euc, 0, u.dim()), 17.1755640373177, 1e-13);
        approx_eq(vec_norm_chunk(&u, Norm::Fro, 0, u.dim()), 17.1755640373177, 1e-13);
        assert_eq!(vec_norm_chunk(&u, Norm::Inf, 0, u.dim()), 3.0);
        assert_eq!(vec_norm_chunk(&u, Norm::Max, 0, u.dim()), 3.0);
        assert_eq!(vec_norm_chunk(&u, Norm::One, 0, u.dim()), 101.0);
    }

    #[test]
    fn vec_norm_chunk_partial() {
        // Note that &u[1..4] = [2.0, 1.0, 1.0]
        let u = Vector::from(&[-3.0, 2.0, 1.0, 1.0, 1.0]);
        approx_eq(vec_norm_chunk(&u, Norm::Euc, 1, 4), SQRT_6, 1e-15);
        approx_eq(vec_norm_chunk(&u, Norm::Fro, 1, 4), SQRT_6, 1e-15);
        assert_eq!(vec_norm_chunk(&u, Norm::Inf, 1, 4), 2.0);
        assert_eq!(vec_norm_chunk(&u, Norm::Max, 1, 4), 2.0);
        assert_eq!(vec_norm_chunk(&u, Norm::One, 1, 4), 4.0);
    }
}
