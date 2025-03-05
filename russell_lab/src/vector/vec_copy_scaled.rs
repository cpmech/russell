use super::Vector;
use crate::StrError;

/// Copies a scaled vector into another
///
/// Performs the operation:
/// ```text
/// v = α⋅u
/// ```
///
/// # Input
///
/// * `v` -- destination vector that will receive the scaled values
/// * `alpha` -- scaling factor
/// * `u` -- source vector to be scaled
///
/// # Returns
///
/// Returns an error if the vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_copy_scaled, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     const NOISE: f64 = 123.456;
///     let u = Vector::from(&[10.0, 20.0, 30.0]);
///     let mut v = Vector::from(&[NOISE, NOISE, NOISE]);
///     vec_copy_scaled(&mut v, 0.1, &u)?;
///     let correct = "┌   ┐\n\
///                    │ 1 │\n\
///                    │ 2 │\n\
///                    │ 3 │\n\
///                    └   ┘";
///     assert_eq!(format!("{}", v), correct);
///     Ok(())
/// }
/// ```
///
/// # Note
///
/// The function optimizes performance by processing vectors in chunks of 4 elements
/// when possible, with special handling for the remainder elements.
pub fn vec_copy_scaled(v: &mut Vector, alpha: f64, u: &Vector) -> Result<(), StrError> {
    let n = v.dim();
    if u.dim() != n {
        return Err("vectors are incompatible");
    }
    let m = n % 4;
    for i in 0..m {
        v[i] = alpha * u[i];
    }
    for i in (m..n).step_by(4) {
        v[i] = alpha * u[i];
        v[i + 1] = alpha * u[i + 1];
        v[i + 2] = alpha * u[i + 2];
        v[i + 3] = alpha * u[i + 3];
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_copy_scaled, Vector};
    use crate::vec_approx_eq;

    #[test]
    fn vec_copy_scaled_fails_on_wrong_dims() {
        let u = Vector::new(4);
        let mut v = Vector::new(3);
        assert_eq!(vec_copy_scaled(&mut v, 1.0, &u), Err("vectors are incompatible"));
    }

    #[test]
    fn vec_copy_scaled_works() {
        const NOISE: f64 = 123.456;
        let u = Vector::from(&[100.0, 200.0, 300.0]);
        let mut v = Vector::from(&[NOISE, NOISE, NOISE]);
        vec_copy_scaled(&mut v, 0.01, &u).unwrap();
        let correct = &[1.0, 2.0, 3.0];
        vec_approx_eq(&v, correct, 1e-15);
    }

    #[test]
    fn vec_copy_scaled_works_with_zero_alpha() {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let mut v = Vector::from(&[10.0, 20.0, 30.0, 40.0]);
        vec_copy_scaled(&mut v, 0.0, &u).unwrap();
        let correct = &[0.0, 0.0, 0.0, 0.0];
        vec_approx_eq(&v, correct, 1e-15);
    }

    #[test]
    fn vec_copy_scaled_works_with_negative_alpha() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let mut v = Vector::new(3);
        vec_copy_scaled(&mut v, -2.0, &u).unwrap();
        let correct = &[-2.0, -4.0, -6.0];
        vec_approx_eq(&v, correct, 1e-15);
    }

    #[test]
    fn vec_copy_scaled_works_with_small_vector() {
        let u = Vector::from(&[1.0, 2.0]);
        let mut v = Vector::new(2);
        vec_copy_scaled(&mut v, 3.0, &u).unwrap();
        let correct = &[3.0, 6.0];
        vec_approx_eq(&v, correct, 1e-15);
    }

    #[test]
    fn vec_copy_scaled_works_with_large_vector() {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let mut v = Vector::new(7);
        vec_copy_scaled(&mut v, 2.0, &u).unwrap();
        let correct = &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
        vec_approx_eq(&v, correct, 1e-15);
    }

    #[test]
    fn vec_copy_scaled_works_with_exactly_four_elements() {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let mut v = Vector::new(4);
        vec_copy_scaled(&mut v, 0.5, &u).unwrap();
        let correct = &[0.5, 1.0, 1.5, 2.0];
        vec_approx_eq(&v, correct, 1e-15);
    }
}
