use super::Matrix;
use crate::NormMat;
use russell_openblas::{dlange, to_i32};

/// Computes the matrix norm
///
/// Computes one of:
///
/// ```text
/// One:  1-norm
///
///       ‖a‖_1 = max_j ( Σ_i |aᵢⱼ| )
/// ```
///
/// ```text
/// Inf:  inf-norm
///
///       ‖a‖_∞ = max_i ( Σ_j |aᵢⱼ| )
/// ```
///
/// ```text
/// Fro:  Frobenius-norm (2-norm)
///
///       ‖a‖_F = sqrt(Σ_i Σ_j aᵢⱼ⋅aᵢⱼ) == ‖a‖_2
/// ```
///
/// ```text
/// Max: max-norm
///
///      ‖a‖_max = max_ij ( |aᵢⱼ| )
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{matrix_norm, Matrix, NormMat};
///
/// fn main() {
///     let a = Matrix::from(&[
///         [-2.0,  2.0],
///         [ 1.0, -4.0],
///     ]);
///     assert_eq!(matrix_norm(&a, NormMat::One), 6.0);
///     assert_eq!(matrix_norm(&a, NormMat::Inf), 5.0);
///     assert_eq!(matrix_norm(&a, NormMat::Fro), 5.0);
///     assert_eq!(matrix_norm(&a, NormMat::Max), 4.0);
/// }
/// ```
pub fn matrix_norm(a: &Matrix, kind: NormMat) -> f64 {
    let (m, n) = a.dims();
    if m == 0 || n == 0 {
        return 0.0;
    }
    let norm = match kind {
        NormMat::One => b'1',
        NormMat::Inf => b'I',
        NormMat::Fro => b'F',
        NormMat::Max => b'M',
    };
    let (m_i32, n_i32) = (to_i32(m), to_i32(n));
    dlange(norm, m_i32, n_i32, &a.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{matrix_norm, Matrix};
    use crate::NormMat;

    #[test]
    fn matrix_norm_works() {
        let a_0x0 = Matrix::new(0, 0);
        let a_0x1 = Matrix::new(0, 1);
        let a_1x0 = Matrix::new(1, 0);
        assert_eq!(matrix_norm(&a_0x0, NormMat::One), 0.0);
        assert_eq!(matrix_norm(&a_0x0, NormMat::Inf), 0.0);
        assert_eq!(matrix_norm(&a_0x0, NormMat::Fro), 0.0);
        assert_eq!(matrix_norm(&a_0x0, NormMat::Max), 0.0);
        assert_eq!(matrix_norm(&a_0x1, NormMat::One), 0.0);
        assert_eq!(matrix_norm(&a_0x1, NormMat::Inf), 0.0);
        assert_eq!(matrix_norm(&a_0x1, NormMat::Fro), 0.0);
        assert_eq!(matrix_norm(&a_0x1, NormMat::Max), 0.0);
        assert_eq!(matrix_norm(&a_1x0, NormMat::One), 0.0);
        assert_eq!(matrix_norm(&a_1x0, NormMat::Inf), 0.0);
        assert_eq!(matrix_norm(&a_1x0, NormMat::Fro), 0.0);
        assert_eq!(matrix_norm(&a_1x0, NormMat::Max), 0.0);
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [ 5.0, -4.0, 2.0],
            [-1.0,  2.0, 3.0],
            [-2.0,  1.0, 0.0],
        ]);
        assert_eq!(matrix_norm(&a, NormMat::One), 8.0);
        assert_eq!(matrix_norm(&a, NormMat::Inf), 11.0);
        assert_eq!(matrix_norm(&a, NormMat::Fro), 8.0);
        assert_eq!(matrix_norm(&a, NormMat::Max), 5.0);
    }
}
