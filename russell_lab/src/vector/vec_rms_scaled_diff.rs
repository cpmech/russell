use super::Vector;

/// Returns the scaled root-mean-square of a vector difference with components normalized by a scaling factor
///
/// The scaling factor is calculated by (see Eq. (4.10) on page 167 of Ref #1):
///
/// ```text
/// scᵢ = ϵ_abs + ϵ_rel ⋅ max(|uᵢ|, |vᵢ|)
/// ```
///
/// The RMS error is then calculated by (see Eq. (4.11) on page 168 of Ref #1):
///
/// ```text
///              _______________________
///             /     ————            2 `
///       \    /  1   \    ⎛ uᵢ - vᵢ ⎞
/// rms =  \  /  ———  /    ⎜ ——————— ⎟
///         \/    N   ———— ⎝   scᵢ   ⎠
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
/// This function will panic of u.dim() != v.dim()
///
/// # Reference
///
/// 1. Hairer E and Wanner G (2002) Solving Ordinary Differential Equations II
///    Stiff and Differential-Algebraic Problems, 2nd Revision, Springer, 627p
pub fn vec_rms_scaled_diff(u: &Vector, v: &Vector, abs_tol: f64, rel_tol: f64) -> f64 {
    let n = v.dim();
    assert!(u.dim() == n);
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let sci = abs_tol + rel_tol * f64::max(f64::abs(u[i]), f64::abs(v[i]));
        let diff = u[i] - v[i];
        sum += diff * diff / (sci * sci);
    }
    f64::sqrt(sum / (n as f64))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_rms_scaled_diff, Vector};
    use crate::approx_eq;

    #[test]
    fn vec_rms_scaled_diff_works() {
        let empty = Vector::new(0);
        assert_eq!(vec_rms_scaled_diff(&empty, &empty, 1.0, 1.0), 0.0);

        let abs_tol = 1.0;
        let rel_tol = 1.0;
        let u = Vector::from(&[3.0, 1.0, 2.0]);
        let v = Vector::from(&[-2.0, 0.0, 3.0]);
        // d = [5, 1, -1]
        // s = 1 + 1 * [3, 1, 3] = [4, 2, 4]
        // d/s = [5/4, 1/2, -1/4]
        // (d/s)^2 = [25/16, 1/4, 1/16]
        // sum((d/s)^2) = 25/16 + 1/4 + 1/16 = 30/16
        // rms = sqrt(sum / 3) = sqrt(30/48) = sqrt(5/8)
        let rms = vec_rms_scaled_diff(&u, &v, abs_tol, rel_tol);
        approx_eq(rms, f64::sqrt(5.0 / 8.0), 1e-15);

        let abs_tol = 0.1;
        let rel_tol = 0.25;
        let rms = vec_rms_scaled_diff(&u, &v, abs_tol, rel_tol);
        approx_eq(rms, 3.8362057850464972, 1e-15);
    }
}
