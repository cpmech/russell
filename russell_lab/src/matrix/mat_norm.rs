use super::Matrix;
use crate::Norm;

/// (dlange) Computes the matrix norm
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
/// See also: <https://www.netlib.org/lapack/explore-html/dc/d09/dlange_8f.html>
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() {
///     let a = Matrix::from(&[
///         [-2.0,  2.0],
///         [ 1.0, -4.0],
///     ]);
///     approx_eq(mat_norm(&a, Norm::One), 6.0, 1e-15);
///     approx_eq(mat_norm(&a, Norm::Inf), 5.0, 1e-15);
///     approx_eq(mat_norm(&a, Norm::Fro), 5.0, 1e-15);
///     approx_eq(mat_norm(&a, Norm::Max), 4.0, 1e-15);
/// }
/// ```
pub fn mat_norm(a: &Matrix, kind: Norm) -> f64 {
    let (m, n) = a.dims();
    if m == 0 || n == 0 {
        return 0.0;
    }
    // These are the characters LAPACK's `dlange` takes; `'O'` is its other
    // spelling of the one-norm, and `Euc` and `Fro` are both the Frobenius norm
    // for a matrix (see `Norm`). The mapping is stated here rather than in the
    // C shim so the native and accelerated paths resolve the same norm.
    let norm = match kind {
        Norm::Euc | Norm::Fro => 'F',
        Norm::Inf => 'I',
        Norm::Max => 'M',
        Norm::One => 'O',
    };
    crate::base::blas_dispatch::dlange(norm, a.as_data(), m, n)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Matrix, mat_norm};
    use crate::{Norm, approx_eq};

    #[test]
    fn mat_norm_works() {
        let a_0x0 = Matrix::new(0, 0);
        let a_0x1 = Matrix::new(0, 1);
        let a_1x0 = Matrix::new(1, 0);
        assert_eq!(mat_norm(&a_0x0, Norm::One), 0.0);
        assert_eq!(mat_norm(&a_0x0, Norm::Inf), 0.0);
        assert_eq!(mat_norm(&a_0x0, Norm::Euc), 0.0);
        assert_eq!(mat_norm(&a_0x0, Norm::Fro), 0.0);
        assert_eq!(mat_norm(&a_0x0, Norm::Max), 0.0);
        assert_eq!(mat_norm(&a_0x1, Norm::One), 0.0);
        assert_eq!(mat_norm(&a_0x1, Norm::Inf), 0.0);
        assert_eq!(mat_norm(&a_0x1, Norm::Fro), 0.0);
        assert_eq!(mat_norm(&a_0x1, Norm::Max), 0.0);
        assert_eq!(mat_norm(&a_1x0, Norm::One), 0.0);
        assert_eq!(mat_norm(&a_1x0, Norm::Inf), 0.0);
        assert_eq!(mat_norm(&a_1x0, Norm::Fro), 0.0);
        assert_eq!(mat_norm(&a_1x0, Norm::Max), 0.0);
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [ 5.0, -4.0, 2.0],
            [-1.0,  2.0, 3.0],
            [-2.0,  1.0, 0.0],
        ]);
        approx_eq(mat_norm(&a, Norm::One), 8.0, 1e-15);
        approx_eq(mat_norm(&a, Norm::Inf), 11.0, 1e-15);
        approx_eq(mat_norm(&a, Norm::Euc), 8.0, 1e-15);
        approx_eq(mat_norm(&a, Norm::Fro), 8.0, 1e-15);
        approx_eq(mat_norm(&a, Norm::Max), 5.0, 1e-15);

        // example from https://netlib.org/lapack/lug/node75.html
        #[rustfmt::skip]
        let diff = Matrix::from(&[
            [ 0.56, -0.36, -0.04],
            [ 0.91, -0.87, -0.66],
            [-0.36,  0.23,  0.93],
        ]);
        assert_eq!(mat_norm(&diff, Norm::Inf), 2.44);
        assert_eq!(mat_norm(&diff, Norm::One), 1.83);
        approx_eq(mat_norm(&diff, Norm::Fro), 1.87, 0.01);
    }
}
