use super::{vec_minus, vec_norm, Vector};
use crate::Norm;

/// Returns the norm of the difference between two vectors
///
/// ```text
/// norm_diff(u,v) = ‖u - v‖
/// ```
///
/// Note: This function creates a temporary vector to store the difference `u - v`.
///
/// # Panics
///
/// This function will panic of u.dim() != v.dim()
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_norm_diff, Norm, Vector};
///
/// fn main() {
///     let u = Vector::from(&[4.0, -4.0, 0.0,  0.0, -6.0]);
///     let v = Vector::from(&[2.0, -2.0, 2.0, -2.0, -3.0]);
///     assert_eq!(vec_norm_diff(&u, &v, Norm::One), 11.0);
///     assert_eq!(vec_norm_diff(&u, &v, Norm::Euc), 5.0);
///     assert_eq!(vec_norm_diff(&u, &v, Norm::Max), 3.0);
/// }
/// ```
pub fn vec_norm_diff(u: &Vector, v: &Vector, kind: Norm) -> f64 {
    let n = v.dim();
    assert!(u.dim() == n);
    if n == 0 {
        return 0.0;
    }
    let mut diff = Vector::new(n);
    vec_minus(&mut diff, u, v).unwrap();
    vec_norm(&diff, kind)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_norm_diff, Vector};
    use crate::{approx_eq, Norm};

    #[test]
    fn vec_norm_diff_works() {
        let u0 = Vector::new(0);
        let v0 = Vector::new(0);
        assert_eq!(vec_norm_diff(&u0, &v0, Norm::Euc), 0.0);
        assert_eq!(vec_norm_diff(&u0, &v0, Norm::Fro), 0.0);
        assert_eq!(vec_norm_diff(&u0, &v0, Norm::Inf), 0.0);
        assert_eq!(vec_norm_diff(&u0, &v0, Norm::Max), 0.0);
        assert_eq!(vec_norm_diff(&u0, &v0, Norm::One), 0.0);

        let u = Vector::from(&[-3.0, 2.0, 1.0, 1.0, 1.0]);
        let v = Vector::from(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(vec_norm_diff(&u, &v, Norm::Euc), 4.0);
        assert_eq!(vec_norm_diff(&u, &v, Norm::Fro), 4.0);
        assert_eq!(vec_norm_diff(&u, &v, Norm::Inf), 3.0);
        assert_eq!(vec_norm_diff(&u, &v, Norm::Max), 3.0);
        assert_eq!(vec_norm_diff(&u, &v, Norm::One), 8.0);

        let u = Vector::from(&[3.0, 1.0, 2.0]);
        let v = Vector::from(&[-2.0, 0.0, 3.0]);
        approx_eq(vec_norm_diff(&u, &v, Norm::Euc), 5.196152422706632, 1e-15);
        approx_eq(vec_norm_diff(&u, &v, Norm::Fro), 5.196152422706632, 1e-15);
        assert_eq!(vec_norm_diff(&u, &v, Norm::Inf), 5.0);
        assert_eq!(vec_norm_diff(&u, &v, Norm::Max), 5.0);
        assert_eq!(vec_norm_diff(&u, &v, Norm::One), 7.0);
    }
}
