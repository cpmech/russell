use super::ComplexMatrix;
use crate::{to_i32, Norm};
use num_complex::Complex64;

extern "C" {
    fn c_zlange(
        norm_code: i32,
        m: *const i32,
        n: *const i32,
        a: *const Complex64,
        lda: *const i32,
        work: *mut f64,
    ) -> f64;
}

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

    let m_i32 = to_i32(m);
    let n_i32 = to_i32(n);
    let lda = m_i32;
    let mut work = if kind == Norm::Inf { vec![0.0; m] } else { Vec::new() };
    let norm_code = kind as i32;
    unsafe { c_zlange(norm_code, &m_i32, &n_i32, a.as_data().as_ptr(), &lda, work.as_mut_ptr()) }
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
            1e-14,
        );
        approx_eq(
            complex_mat_norm(&a, Norm::Inf),
            a.get(0, 0).abs() + a.get(0, 1).abs() + a.get(0, 2).abs(),
            1e-14,
        );
        let mut fro = 0.0;
        for v in a.as_data() {
            fro += v.abs() * v.abs();
        }
        fro = f64::sqrt(fro);
        approx_eq(complex_mat_norm(&a, Norm::Fro), fro, 1e-14);
        approx_eq(complex_mat_norm(&a, Norm::Max), Complex64::new(5.0, 1.0).abs(), 1e-14);
    }
}
