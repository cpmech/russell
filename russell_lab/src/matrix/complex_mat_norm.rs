use super::ComplexMatrix;
use crate::Norm;
use russell_openblas::{to_i32, zlange};

/// Computes the matrix norm (complex version)
///
/// # Example
///
/// ```
/// use russell_lab::{complex_mat_norm, ComplexMatrix, Norm};
/// use russell_chk::approx_eq;
///
/// fn main() {
///     let a = ComplexMatrix::from(&[
///         [-2.0,  2.0],
///         [ 1.0, -4.0],
///     ]);
///     approx_eq(complex_mat_norm(&a, Norm::One), 6.0, 1e-15);
///     approx_eq(complex_mat_norm(&a, Norm::Inf), 5.0, 1e-15);
///     approx_eq(complex_mat_norm(&a, Norm::Fro), 5.0, 1e-15);
///     approx_eq(complex_mat_norm(&a, Norm::Max), 4.0, 1e-15);
/// }
/// ```
pub fn complex_mat_norm(a: &ComplexMatrix, kind: Norm) -> f64 {
    let (m, n) = a.dims();
    if m == 0 || n == 0 {
        return 0.0;
    }
    let norm = match kind {
        Norm::Euc | Norm::Fro => b'F',
        Norm::Inf => b'I',
        Norm::Max => b'M',
        Norm::One => b'1',
    };
    let (m_i32, n_i32) = (to_i32(m), to_i32(n));
    zlange(norm, m_i32, n_i32, &a.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_norm, ComplexMatrix};
    use crate::Norm;
    use num_complex::{Complex64, ComplexFloat};
    use russell_chk::approx_eq;

    #[test]
    fn complex_mat_norm_works() {
        let a_0x0 = ComplexMatrix::new(0, 0);
        let a_0x1 = ComplexMatrix::new(0, 1);
        let a_1x0 = ComplexMatrix::new(1, 0);
        assert_eq!(complex_mat_norm(&a_0x0, Norm::One), 0.0);
        assert_eq!(complex_mat_norm(&a_0x0, Norm::Inf), 0.0);
        assert_eq!(complex_mat_norm(&a_0x0, Norm::Fro), 0.0);
        assert_eq!(complex_mat_norm(&a_0x0, Norm::Max), 0.0);
        assert_eq!(complex_mat_norm(&a_0x1, Norm::One), 0.0);
        assert_eq!(complex_mat_norm(&a_0x1, Norm::Inf), 0.0);
        assert_eq!(complex_mat_norm(&a_0x1, Norm::Fro), 0.0);
        assert_eq!(complex_mat_norm(&a_0x1, Norm::Max), 0.0);
        assert_eq!(complex_mat_norm(&a_1x0, Norm::One), 0.0);
        assert_eq!(complex_mat_norm(&a_1x0, Norm::Inf), 0.0);
        assert_eq!(complex_mat_norm(&a_1x0, Norm::Fro), 0.0);
        assert_eq!(complex_mat_norm(&a_1x0, Norm::Max), 0.0);
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [Complex64::new( 5.0, 1.0), Complex64::new(-4.0, 2.0), Complex64::new(2.0, 3.0)],
            [Complex64::new(-1.0, 1.0), Complex64::new( 2.0, 2.0), Complex64::new(3.0, 3.0)],
            [Complex64::new(-2.0, 1.0), Complex64::new( 1.0, 2.0), Complex64::new(0.0, 3.0)],
        ]);
        approx_eq(
            complex_mat_norm(&a, Norm::One),
            a.get(0, 2).abs() + a.get(1, 2).abs() + a.get(2, 2).abs(),
            1e-15,
        );
        approx_eq(
            complex_mat_norm(&a, Norm::Inf),
            a.get(0, 0).abs() + a.get(0, 1).abs() + a.get(0, 2).abs(),
            1e-15,
        );
        let mut fro = 0.0;
        for v in a.as_data() {
            fro += v.abs() * v.abs();
        }
        fro = f64::sqrt(fro);
        approx_eq(complex_mat_norm(&a, Norm::Fro), fro, 1e-15);
        approx_eq(complex_mat_norm(&a, Norm::Max), Complex64::new(5.0, 1.0).abs(), 1e-15);
    }
}
