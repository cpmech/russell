use super::ComplexMatrix;
use crate::Norm;

/// (zlange) Computes the matrix norm
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
/// See also: <https://www.netlib.org/lapack/explore-html/d5/d8f/zlange_8f.html>
///
/// # Examples
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
    // `norm_char` is the one mapping from `Norm` to the BLAS character, and the
    // dispatch maps it back to the discriminant the shim expects; going through
    // the character keeps a single source for the translation.
    let norm = crate::native::lapack::norm_char(kind as i32);
    crate::base::blas_dispatch::zlange(norm, a.as_data(), m, n)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{ComplexMatrix, complex_mat_norm};
    use crate::{ComplexFloat, Norm, approx_eq, cpx};

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
