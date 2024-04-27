use super::ComplexVector;
use crate::{to_i32, Complex64, Norm};

extern "C" {
    // Computes the Euclidean norm
    // <https://www.netlib.org/lapack/explore-html/d1/d5a/dznrm2_8f90.html>
    fn cblas_dznrm2(n: i32, x: *const Complex64, incx: i32) -> f64;
}

/// Returns the vector norm
///
/// Here:
///
/// ```text
/// abs(z) = |z| = z.norm() = sqrt(z.real² + z.imag²)
/// ```
///
/// Euclidean-norm (Euc or Fro):
///
/// ```text
/// ‖u‖_2 = sqrt(Σ_i uᵢ ⋅ conjugate(uᵢ))
/// ```
///
/// Max-norm (Max or Inf):
///
/// ```text
/// ‖u‖_max = max_i ( |uᵢ| ) == ‖u‖_∞
/// ```
///
/// 1-norm (One, taxicab or sum of abs values):
///
/// ```text
/// ‖u‖_1 := sum_i |uᵢ|
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() {
///     let u = ComplexVector::from(&[cpx!(1.0, 1.0), cpx!(3.0, 1.0), cpx!(5.0, -1.0)]);
///     approx_eq(complex_vec_norm(&u, Norm::One), 9.67551073613426, 1e-15);
///     approx_eq(complex_vec_norm(&u, Norm::Euc), 6.164414002968976, 1e-15);
///     approx_eq(complex_vec_norm(&u, Norm::Max), 5.099019513592784, 1e-15);
/// }
/// ```
pub fn complex_vec_norm(v: &ComplexVector, kind: Norm) -> f64 {
    let n = to_i32(v.dim());
    if n == 0 {
        return 0.0;
    }
    unsafe {
        match kind {
            Norm::Euc | Norm::Fro => cblas_dznrm2(n, v.as_data().as_ptr(), 1),
            Norm::Inf | Norm::Max => {
                let mut max = 0.0;
                for i in 0..v.dim() {
                    let abs = v[i].norm();
                    if abs > max {
                        max = abs;
                    }
                }
                max
            }
            Norm::One => v.as_data().iter().fold(0.0, |acc, z| acc + z.norm()),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_vec_norm;
    use crate::{approx_eq, cpx, Complex64, ComplexVector, Norm};

    #[test]
    fn complex_vec_norm_works() {
        let u0 = ComplexVector::new(0);
        assert_eq!(complex_vec_norm(&u0, Norm::Euc), 0.0);
        assert_eq!(complex_vec_norm(&u0, Norm::Fro), 0.0);
        assert_eq!(complex_vec_norm(&u0, Norm::Inf), 0.0);
        assert_eq!(complex_vec_norm(&u0, Norm::Max), 0.0);
        assert_eq!(complex_vec_norm(&u0, Norm::One), 0.0);

        let u = ComplexVector::from(&[
            cpx!(-3.0, 0.0),
            cpx!(2.0, 0.0),
            cpx!(1.0, 0.0),
            cpx!(1.0, 0.0),
            cpx!(1.0, 0.0),
        ]);
        assert_eq!(complex_vec_norm(&u, Norm::Euc), 4.0);
        assert_eq!(complex_vec_norm(&u, Norm::Fro), 4.0);
        assert_eq!(complex_vec_norm(&u, Norm::Inf), 3.0);
        assert_eq!(complex_vec_norm(&u, Norm::Max), 3.0);
        assert_eq!(complex_vec_norm(&u, Norm::One), 8.0);

        let u = ComplexVector::from(&[
            cpx!(-3.0, 1.0),
            cpx!(2.0, -1.0),
            cpx!(1.0, 1.0),
            cpx!(1.0, -1.0),
            cpx!(1.0, 1.0),
        ]);
        approx_eq(complex_vec_norm(&u, Norm::Euc), 4.58257569495584, 1e-15);
        approx_eq(complex_vec_norm(&u, Norm::Fro), 4.58257569495584, 1e-15);
        approx_eq(complex_vec_norm(&u, Norm::Inf), 3.1622776601683795, 1e-15);
        approx_eq(complex_vec_norm(&u, Norm::Max), 3.1622776601683795, 1e-15);
        approx_eq(complex_vec_norm(&u, Norm::One), 9.640986324787455, 1e-15);

        // example from https://netlib.org/lapack/lug/node75.html
        let diff = ComplexVector::from(&[cpx!(-0.1, 0.0), cpx!(1.0, 0.0), cpx!(-2.0, 0.0)]);
        approx_eq(complex_vec_norm(&diff, Norm::Euc), 2.238, 0.001);
        assert_eq!(complex_vec_norm(&diff, Norm::Inf), 2.0);
        assert_eq!(complex_vec_norm(&diff, Norm::One), 3.1);
    }
}
