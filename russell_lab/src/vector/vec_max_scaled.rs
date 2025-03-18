use super::Vector;

/// Returns the maximum component of a vector scaled by the components of a reference vector
///
/// ```text
///             ⎛     |vᵢ|    ⎞
/// res = max_i ⎜ ——————————— ⎟
///             ⎝ one + |v0ᵢ| ⎠
/// ```
///
/// # Panics
///
/// This function will panic of v.dim() != v0.dim()
pub fn vec_max_scaled(v: &Vector, v0: &Vector, one: f64) -> f64 {
    let m = v.dim();
    assert!(v0.dim() == m);
    if m == 0 {
        return 0.0;
    }
    let mut res = f64::MIN;
    for i in 0..m {
        res = f64::max(res, f64::abs(v[i]) / (one + f64::abs(v0[i])));
    }
    res
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_max_scaled, Vector};

    #[test]
    fn vec_max_scaled_works() {
        let empty = Vector::new(0);
        assert_eq!(vec_max_scaled(&empty, &empty, 1.0), 0.0);

        let v = Vector::from(&[-2.0, 0.0, 2.0]);
        let v0 = Vector::from(&[-1.0, -1.0, -1.0]);
        let res = vec_max_scaled(&v, &v0, 1.0);
        assert_eq!(res, 1.0);

        let v = Vector::from(&[-9.0, 0.0, 2.0]);
        let v0 = Vector::from(&[-2.0, -1.0, -1.0]);
        let res = vec_max_scaled(&v, &v0, 1.0);
        assert_eq!(res, 3.0);

        let v = Vector::from(&[-1.0, 0.0, 12.0]);
        let v0 = Vector::from(&[-2.0, -1.0, 0.0]);
        let res = vec_max_scaled(&v, &v0, 1.0);
        assert_eq!(res, 12.0);

        let v = Vector::from(&[0.01, -0.01]);
        let v0 = Vector::from(&[0.0, 0.0]);
        let res = vec_max_scaled(&v, &v0, 1.0);
        assert_eq!(res, 0.01);
    }
}
