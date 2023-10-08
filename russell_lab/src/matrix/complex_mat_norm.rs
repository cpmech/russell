use super::ComplexMatrix;
use crate::{to_i32, Norm};
use num_complex::Complex64;

extern "C" {
    // Computes the matrix norm (complex version)
    // <http://www.netlib.org/lapack/explore-html/d5/d8f/zlange_8f.html>
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
/// Computes one of:
///
/// ```text
/// ‖a‖_1 = max_j ( Σ_i |aij| )
///
/// ‖a‖_∞ = max_i ( Σ_j |aij| )
///
/// ‖a‖_F = sqrt(Σ_i Σ_j |aij|²) == ‖a‖_2
///
/// ‖a‖_max = max_ij ( |aij| )
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::*;
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
    use crate::{approx_eq, cpx, Norm};
    use num_complex::{Complex64, ComplexFloat};

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
            [cpx!( 5.0, 1.0), cpx!(-4.0, 2.0), cpx!(2.0, 3.0)],
            [cpx!(-1.0, 1.0), cpx!( 2.0, 2.0), cpx!(3.0, 3.0)],
            [cpx!(-2.0, 1.0), cpx!( 1.0, 2.0), cpx!(0.0, 3.0)],
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
        approx_eq(complex_mat_norm(&a, Norm::Max), cpx!(5.0, 1.0).abs(), 1e-14);
    }
}
