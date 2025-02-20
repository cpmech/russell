use super::Vector;
use crate::StrError;

/// Copies a scaled vector into another
///
/// ```text
/// v = α⋅u
/// ```
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
pub fn vec_copy_scaled(v: &mut Vector, alpha: f64, u: &Vector) -> Result<(), StrError> {
    // Check dimensions
    let n = v.dim();
    if u.dim() != n {
        return Err("vectors are incompatible");
    }

    let m = n % 4;

    // Handle the remainder when n is not divisible by 4
    if m != 0 {
        for i in 0..m {
            v[i] = alpha * u[i];
        }
    }

    // If n is less than 4, return early
    if n < 4 {
        return Ok(());
    }

    // Process the rest in chunks of 4
    let mp1 = m;
    for i in (mp1..n).step_by(4) {
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
}
