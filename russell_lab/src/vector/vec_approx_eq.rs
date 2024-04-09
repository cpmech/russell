use super::Vector;
use crate::{array_approx_eq, AsArray1D};

/// Panics if two vectors are not approximately equal to each other
///
/// # Panics
///
/// 1. Will panic if the dimensions are different
/// 2. Will panic if NaN or Inf is found
/// 3. Will panic if the absolute difference of components is greater than the tolerance
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// use russell_lab::{vec_approx_eq, Vector};
///
/// fn main() {
///     let u = Vector::from(&[3.0000001, 2.0]);
///     let v = Vector::from(&[3.0,       2.0]);
///     vec_approx_eq(&u, &v, 1e-6);
/// }
/// ```
///
/// ## Panics on different value
///
/// ```should_panic
/// use russell_lab::{vec_approx_eq, Vector};
///
/// fn main() {
///     let u = Vector::from(&[3.0000001, 2.0]);
///     let v = Vector::from(&[4.0,       2.0]);
///     vec_approx_eq(&u, &v, 1e-6);
/// }
/// ```
pub fn vec_approx_eq<'a, T>(u: &Vector, v: &'a T, tol: f64)
where
    T: AsArray1D<'a, f64>,
{
    array_approx_eq(u.as_data(), v.as_slice(), tol);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::vec_approx_eq;
    use crate::Vector;

    #[test]
    fn vec_approx_eq_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let v = Vector::from(&[1.0, 2.0, 3.01]);
        vec_approx_eq(&u, &v, 0.011);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. diff[2] =")]
    fn vec_approx_eq_panics() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let v = Vector::from(&[1.0, 2.0, 3.01]);
        vec_approx_eq(&u, &v, 0.009);
    }
}
