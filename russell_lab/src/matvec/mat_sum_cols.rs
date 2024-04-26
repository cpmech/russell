use crate::matrix::Matrix;
use crate::vector::Vector;
use crate::StrError;

/// Sums the columns of a matrix
///
/// ```text
/// vᵢ = Σ_j aᵢⱼ
/// ```
///
/// ```text
///  _                                             _     _      _
/// |      a00      a01      a02  ...      a0(n-1)  |   |   v0   |
/// |      a10      a11      a12  ...      a1(n-1)  |   |   v1   |
/// |                             ...               | = |  ...   |
/// |_ a(m-1)0  a(m-1)1  a(m-1)2  ...  a(m-1)(n-1) _|   |_v(m-1)_|
/// ```
///
/// # Input
///
/// * `v` -- vector with dim = m
/// * `a` -- (m, n) matrix
///
/// # Note
///
/// This function is not as optimized (e.g., multi-threaded) as it could be.
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let a = Matrix::from(&[
///         [1.0, 2.0, 3.0],
///         [2.0, 1.0, 2.0],
///     ]);
///     let mut v = Vector::new(2); // 2 rows
///     mat_sum_cols(&mut v, &a)?;
///     assert_eq!(v.as_data(), &[6.0, 5.0]);
///     Ok(())
/// }
/// ```
pub fn mat_sum_cols(v: &mut Vector, a: &Matrix) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if v.dim() != m {
        return Err("vector is incompatible");
    }
    for i in 0..m {
        v[i] = 0.0;
        for j in 0..n {
            v[i] += a.get(i, j);
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_sum_cols, Matrix, Vector};
    use crate::vec_approx_eq;

    #[test]
    fn mat_sum_cols_fails_on_wrong_dims() {
        let a_1x2 = Matrix::new(1, 2);
        let mut v = Vector::new(3);
        assert_eq!(mat_sum_cols(&mut v, &a_1x2), Err("vector is incompatible"));
    }

    #[test]
    fn mat_sum_cols_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [ 5.0, -2.0, 0.0, 1.0],
            [10.0, -4.0, 0.0, 2.0],
            [15.0, -6.0, 0.0, 3.0],
        ]);
        let mut v = Vector::new(a.nrow());
        mat_sum_cols(&mut v, &a).unwrap();
        let correct = &[4.0, 8.0, 12.0];
        vec_approx_eq(&v, correct, 1e-15);
    }
}
