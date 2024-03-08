use super::Vector;

/// Returns the scaled root-mean-square of a vector with components normalized by a scaling factor
///
/// The scaling factor is calculated from a reference vector and an absolute and a relative tolerance.
///
/// ```text
///              ____________________________________
///             /     ————                         2 `
///       \    /  1   \    /          vᵢ           \
/// rms =  \  /  ———  /    | ————————————————————— |
///         \/    N   ———— \ ϵ_abs + ϵ_rel ⋅ |v0ᵢ| /
///                     i
/// N = v.dim()
/// ```
///
/// # Notes
///
/// * The absolute tolerance and relative tolerance should be > 0
/// * This equation is inspired by Eq. (8.21) on page 124 of Hairer and Wanner (2002)
///
/// # Panics
///
/// This function will panic of v.dim() != v0.dim()
///
/// # Reference
///
/// Hairer E and Wanner G (2002) Solving Ordinary Differential Equations II
/// Stiff and Differential-Algebraic Problems, 2nd Revision, Springer, 627p
pub fn vec_rms_scaled(v: &Vector, v0: &Vector, abs_tol: f64, rel_tol: f64) -> f64 {
    let m = v.dim();
    assert!(v0.dim() == m);
    if m == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..m {
        let den = abs_tol + rel_tol * f64::abs(v0[i]);
        sum += v[i] * v[i] / (den * den);
    }
    f64::sqrt(sum / (m as f64))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_rms_scaled, Vector};
    use crate::approx_eq;
    use crate::math::SQRT_2_BY_3;

    #[test]
    fn vec_rms_scaled_works() {
        let empty = Vector::new(0);
        assert_eq!(vec_rms_scaled(&empty, &empty, 1.0, 1.0), 0.0);

        let v = Vector::from(&[-2.0, 0.0, 2.0]);
        let v0 = Vector::from(&[-1.0, -1.0, -1.0]);
        let rms = vec_rms_scaled(&v, &v0, 1.0, 1.0);
        approx_eq(rms, SQRT_2_BY_3, 1e-15);
    }
}
