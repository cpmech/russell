use super::Matrix;
use crate::{to_i32, Norm};

extern "C" {
    // Computes the matrix norm
    // <https://www.netlib.org/lapack/explore-html/dc/d09/dlange_8f.html>
    fn c_dlange(norm_code: i32, m: *const i32, n: *const i32, a: *const f64, lda: *const i32, work: *mut f64) -> f64;
}

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
    let m_i32 = to_i32(m);
    let n_i32 = to_i32(n);
    let lda = m_i32;
    let mut work = if kind == Norm::Inf { vec![0.0; m] } else { Vec::new() };
    let norm_code = kind as i32;
    unsafe { c_dlange(norm_code, &m_i32, &n_i32, a.as_data().as_ptr(), &lda, work.as_mut_ptr()) }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_norm, Matrix};
    use crate::{approx_eq, Norm};

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
