use super::ComplexMatrix;
use crate::NormMat;
use russell_openblas::{to_i32, zlange};

/// Computes the matrix norm (complex version)
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
///       ‖a‖_F = sqrt(Σ_i Σ_j |aij|²) == ‖a‖_2
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
/// use russell_lab::{complex_mat_norm, ComplexMatrix, NormMat};
/// use russell_chk::approx_eq;
///
/// fn main() {
///     let a = ComplexMatrix::from(&[
///         [-2.0,  2.0],
///         [ 1.0, -4.0],
///     ]);
///     approx_eq(complex_mat_norm(&a, NormMat::One), 6.0, 1e-15);
///     approx_eq(complex_mat_norm(&a, NormMat::Inf), 5.0, 1e-15);
///     approx_eq(complex_mat_norm(&a, NormMat::Fro), 5.0, 1e-15);
///     approx_eq(complex_mat_norm(&a, NormMat::Max), 4.0, 1e-15);
/// }
/// ```
pub fn complex_mat_norm(a: &ComplexMatrix, kind: NormMat) -> f64 {
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
    zlange(norm, m_i32, n_i32, &a.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_norm, ComplexMatrix};
    use crate::NormMat;
    use num_complex::{Complex64, ComplexFloat};
    use russell_chk::approx_eq;

    #[test]
    fn complex_mat_norm_works() {
        let a_0x0 = ComplexMatrix::new(0, 0);
        let a_0x1 = ComplexMatrix::new(0, 1);
        let a_1x0 = ComplexMatrix::new(1, 0);
        assert_eq!(complex_mat_norm(&a_0x0, NormMat::One), 0.0);
        assert_eq!(complex_mat_norm(&a_0x0, NormMat::Inf), 0.0);
        assert_eq!(complex_mat_norm(&a_0x0, NormMat::Fro), 0.0);
        assert_eq!(complex_mat_norm(&a_0x0, NormMat::Max), 0.0);
        assert_eq!(complex_mat_norm(&a_0x1, NormMat::One), 0.0);
        assert_eq!(complex_mat_norm(&a_0x1, NormMat::Inf), 0.0);
        assert_eq!(complex_mat_norm(&a_0x1, NormMat::Fro), 0.0);
        assert_eq!(complex_mat_norm(&a_0x1, NormMat::Max), 0.0);
        assert_eq!(complex_mat_norm(&a_1x0, NormMat::One), 0.0);
        assert_eq!(complex_mat_norm(&a_1x0, NormMat::Inf), 0.0);
        assert_eq!(complex_mat_norm(&a_1x0, NormMat::Fro), 0.0);
        assert_eq!(complex_mat_norm(&a_1x0, NormMat::Max), 0.0);
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [Complex64::new( 5.0, 1.0), Complex64::new(-4.0, 2.0), Complex64::new(2.0, 3.0)],
            [Complex64::new(-1.0, 1.0), Complex64::new( 2.0, 2.0), Complex64::new(3.0, 3.0)],
            [Complex64::new(-2.0, 1.0), Complex64::new( 1.0, 2.0), Complex64::new(0.0, 3.0)],
        ]);
        approx_eq(
            complex_mat_norm(&a, NormMat::One),
            a[0][2].abs() + a[1][2].abs() + a[2][2].abs(),
            1e-15,
        );
        approx_eq(
            complex_mat_norm(&a, NormMat::Inf),
            a[0][0].abs() + a[0][1].abs() + a[0][2].abs(),
            1e-15,
        );
        let mut fro = 0.0;
        for v in a.as_data() {
            fro += v.abs() * v.abs();
        }
        fro = f64::sqrt(fro);
        approx_eq(complex_mat_norm(&a, NormMat::Fro), fro, 1e-15);
        approx_eq(
            complex_mat_norm(&a, NormMat::Max),
            Complex64::new(5.0, 1.0).abs(),
            1e-15,
        );
    }
}
