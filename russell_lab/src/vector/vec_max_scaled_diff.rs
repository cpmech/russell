use super::Vector;

/// Returns the maximum component of the difference between two vectors scaled by the components of a reference vector
///
/// ```text
///             ⎛ |uᵢ - vᵢ| ⎞
/// res = max_i ⎜ ————————— ⎟
///             ⎝  1 + |sᵢ| ⎠
/// ```
///
/// # Panics
///
/// This function will panic of u.dim() != v.dim() or s.dim() != v.dim()
pub fn vec_max_scaled_diff(u: &Vector, v: &Vector, s: &Vector) -> f64 {
    let n = v.dim();
    assert!(u.dim() == n, "u.dim() != v.dim()");
    assert!(s.dim() == n, "d.dim() != v.dim()");
    if n == 0 {
        return 0.0;
    }
    let mut res = f64::MIN;
    for i in 0..n {
        res = f64::max(res, f64::abs(u[i] - v[i]) / (1.0 + f64::abs(s[i])));
    }
    res
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_max_scaled_diff, Vector};

    #[test]
    fn vec_max_scaled_diff_works() {
        let empty = Vector::new(0);
        assert_eq!(vec_max_scaled_diff(&empty, &empty, &empty), 0.0);

        let u = Vector::from(&[1.0, 0.0, 4.0]);
        let v = Vector::from(&[-2.0, 0.0, 2.0]);
        let d = Vector::from(&[-1.0, -1.0, -1.0]);
        let res = vec_max_scaled_diff(&u, &v, &d);
        assert_eq!(res, 1.5);

        let u = Vector::from(&[18.0, 0.0, 2.0]);
        let v = Vector::from(&[9.0, 0.0, 2.0]);
        let d = Vector::from(&[-2.0, -1.0, -1.0]);
        let res = vec_max_scaled_diff(&u, &v, &d);
        assert_eq!(res, 3.0);

        let u = Vector::from(&[-1.0, 0.0, 24.0]);
        let v = Vector::from(&[-1.0, 0.0, 12.0]);
        let d = Vector::from(&[-2.0, -1.0, 0.0]);
        let res = vec_max_scaled_diff(&u, &v, &d);
        assert_eq!(res, 12.0);

        let u = Vector::from(&[0.02, -0.01]);
        let v = Vector::from(&[0.01, -0.01]);
        let d = Vector::from(&[0.0, 0.0]);
        let res = vec_max_scaled_diff(&u, &v, &d);
        assert_eq!(res, 0.01);
    }
}
