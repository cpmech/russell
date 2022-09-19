use super::Vector;
use crate::Norm;
use russell_openblas::{dasum, dnrm2, idamax, to_i32};

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
    match kind {
        Norm::Euc | Norm::Fro => dnrm2(n, &v.as_data(), 1),
        Norm::Inf | Norm::Max => {
            let idx = idamax(n, &v.as_data(), 1);
            f64::abs(v.get(idx as usize))
        }
        Norm::One => dasum(n, &v.as_data(), 1),
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_norm, Vector};
    use crate::Norm;

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
    }
}
