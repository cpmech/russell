use super::Vector;
use crate::{to_i32, Norm};

extern "C" {
    // Computes the sum of the absolute values (1-norm or taxicab norm)
    // <https://www.netlib.org/lapack/explore-html/de/d05/dasum_8f.html>
    fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64;

    // Computes the Euclidean norm
    // <https://www.netlib.org/lapack/explore-html/d6/de0/dnrm2_8f90.html>
    fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64;

    // Finds the index of the maximum absolute value
    // <https://www.netlib.org/lapack/explore-html/dd/de0/idamax_8f.html>
    fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32;
}

/// Returns the vector norm
///
/// # Example
///
/// ```
/// use russell_lab::{vec_norm, Norm, Vector};
///
/// fn main() {
///     let u = Vector::from(&[2.0, -2.0, 2.0, -2.0, -3.0]);
///     assert_eq!(vec_norm(&u, Norm::One), 11.0);
///     assert_eq!(vec_norm(&u, Norm::Euc), 5.0);
///     assert_eq!(vec_norm(&u, Norm::Max), 3.0);
/// }
/// ```
pub fn vec_norm(v: &Vector, kind: Norm) -> f64 {
    let n = to_i32(v.dim());
    if n == 0 {
        return 0.0;
    }
    unsafe {
        match kind {
            Norm::Euc | Norm::Fro => cblas_dnrm2(n, v.as_data().as_ptr(), 1),
            Norm::Inf | Norm::Max => {
                let idx = cblas_idamax(n, v.as_data().as_ptr(), 1);
                f64::abs(v.get(idx as usize))
            }
            Norm::One => cblas_dasum(n, v.as_data().as_ptr(), 1),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_norm, Vector};
    use crate::{approx_eq, Norm};

    #[test]
    fn vec_norm_works() {
        let u0 = Vector::new(0);
        assert_eq!(vec_norm(&u0, Norm::Euc), 0.0);
        assert_eq!(vec_norm(&u0, Norm::Fro), 0.0);
        assert_eq!(vec_norm(&u0, Norm::Inf), 0.0);
        assert_eq!(vec_norm(&u0, Norm::Max), 0.0);
        assert_eq!(vec_norm(&u0, Norm::One), 0.0);

        let u = Vector::from(&[-3.0, 2.0, 1.0, 1.0, 1.0]);
        assert_eq!(vec_norm(&u, Norm::Euc), 4.0);
        assert_eq!(vec_norm(&u, Norm::Fro), 4.0);
        assert_eq!(vec_norm(&u, Norm::Inf), 3.0);
        assert_eq!(vec_norm(&u, Norm::Max), 3.0);
        assert_eq!(vec_norm(&u, Norm::One), 8.0);

        // example from https://netlib.org/lapack/lug/node75.html
        let diff = Vector::from(&[-0.1, 1.0, -2.0]);
        approx_eq(vec_norm(&diff, Norm::Euc), 2.238, 0.001);
        assert_eq!(vec_norm(&diff, Norm::Inf), 2.0);
        assert_eq!(vec_norm(&diff, Norm::One), 3.1);
    }
}
