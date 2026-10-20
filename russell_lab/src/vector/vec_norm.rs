use super::Vector;
use crate::Norm;

/// Returns the vector norm
///
/// ```text
/// norm(v) = ‖v‖
/// ```
///
/// # Examples
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
    if v.dim() == 0 {
        return 0.0;
    }
    match kind {
        Norm::Euc | Norm::Fro => crate::base::blas_dispatch::dnrm2(v.as_data()),
        Norm::Inf | Norm::Max => f64::abs(v.get(crate::base::blas_dispatch::idamax(v.as_data()))),
        Norm::One => crate::base::blas_dispatch::dasum(v.as_data()),
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Vector, vec_norm};
    use crate::{Norm, approx_eq};

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

    #[test]
    fn vec_norm_works_neg() {
        let u = Vector::from(&[-3.0, -2.0, -1.0, -1.0, -1.0]);
        assert_eq!(vec_norm(&u, Norm::Euc), 4.0);
        assert_eq!(vec_norm(&u, Norm::Fro), 4.0);
        assert_eq!(vec_norm(&u, Norm::Inf), 3.0);
        assert_eq!(vec_norm(&u, Norm::Max), 3.0);
        assert_eq!(vec_norm(&u, Norm::One), 8.0);
    }

    #[test]
    fn vec_norm_works_large() {
        let u = Vector::from(&[
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 3.0, 2.0, 1.0, 1.0, 1.0,
        ]);
        approx_eq(vec_norm(&u, Norm::Euc), 17.1755640373177, 1e-13);
        approx_eq(vec_norm(&u, Norm::Fro), 17.1755640373177, 1e-13);
        assert_eq!(vec_norm(&u, Norm::Inf), 3.0);
        assert_eq!(vec_norm(&u, Norm::Max), 3.0);
        assert_eq!(vec_norm(&u, Norm::One), 101.0);
    }
}
