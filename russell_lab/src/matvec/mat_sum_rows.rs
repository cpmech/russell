use crate::matrix::Matrix;
use crate::vector::Vector;
use crate::StrError;

/// Sums the rows of a matrix
///
/// ```text
/// vⱼ = Σ_i aᵢⱼ
/// ```
///
/// ```text
///  _                                             _
/// |      a00      a01      a02  ...      a0(n-1)  |
/// |      a10      a11      a12  ...      a1(n-1)  |
/// |                             ...               |
/// |_ a(m-1)0  a(m-1)1  a(m-1)2  ...  a(m-1)(n-1) _|
///                           =
/// [       v0       v1       v2  ...       v(n-1)  ]
/// ```
///
/// # Input
///
/// * `v` -- vector with dim = n
/// * `a` -- (m, n) matrix
///
/// # Note
///
/// This function is not as optimized (e.g., multi-threaded) as it could be.
pub fn mat_sum_rows(v: &mut Vector, a: &Matrix) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if v.dim() != n {
        return Err("vector is incompatible");
    }
    for j in 0..n {
        v[j] = 0.0;
        for i in 0..m {
            v[j] += a[i][j];
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_sum_rows, Matrix, Vector};
    use crate::vec_approx_eq;

    #[test]
    fn mat_sum_rows_fails_on_wrong_dims() {
        let a_1x2 = Matrix::new(1, 2);
        let mut v = Vector::new(3);
        assert_eq!(mat_sum_rows(&mut v, &a_1x2), Err("vector is incompatible"));
    }

    #[test]
    fn mat_sum_rows_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [ 5.0, -2.0, 0.0, 1.0],
            [10.0, -4.0, 0.0, 2.0],
            [15.0, -6.0, 0.0, 3.0],
        ]);
        let mut v = Vector::new(a.ncol());
        mat_sum_rows(&mut v, &a).unwrap();
        let correct = &[30.0, -12.0, 0.0, 6.0];
        vec_approx_eq(&v, correct, 1e-15);
    }
}
