use crate::matrix::Matrix;
use crate::vector::Vector;
use crate::StrError;

/// Sums the rows of a matrix
///
/// ```text
///      m-1
/// vⱼ =  Σ  aᵢⱼ
///      i=0
///
/// 0 ≤ j ≤ n-1
/// ```
///
/// Sum downwards:
///
/// ```text
/// ┌                                       ┐
/// │    a₀₀     a₀₁     a₀₂  ···    a₀·ₙ₋₁ │
/// │    a₁₀     a₁₁     a₁₂  ···    a₁·ₙ₋₁ │
/// │                         ···           │
/// │ aₘ₋₁·₀  aₘ₋₁·₁  aₘ₋₁·₂  ···  aₘ₋₁·ₙ₋₁ │
/// └                                       ┘
///                     =
/// [     v₀      v₁      v₂  ···      vₙ₋₁ ]
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
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let a = Matrix::from(&[
///         [1.0, 2.0, 3.0], //
///         [2.1, 1.2, 0.3], //
///     ]);
///     let mut v = Vector::new(3); // 3 columns
///     mat_sum_rows(&mut v, &a)?;
///     assert_eq!(v.as_data(), &[3.1, 3.2, 3.3]);
///     Ok(())
/// }
/// ```
pub fn mat_sum_rows(v: &mut Vector, a: &Matrix) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if v.dim() != n {
        return Err("vector is incompatible");
    }
    for j in 0..n {
        v[j] = 0.0;
        for i in 0..m {
            v[j] += a.get(i, j);
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
