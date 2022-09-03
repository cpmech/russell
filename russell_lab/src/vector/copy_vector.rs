use super::Vector;
use crate::StrError;
use russell_openblas::{dcopy, to_i32};

/// Copies vector
///
/// ```text
/// v := u
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{copy_vector, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[1.0, 2.0, 3.0]);
///     let mut v = Vector::from(&[-1.0, -2.0, -3.0]);
///     copy_vector(&mut v, &u)?;
///     let correct = "┌   ┐\n\
///                    │ 1 │\n\
///                    │ 2 │\n\
///                    │ 3 │\n\
///                    └   ┘";
///     assert_eq!(format!("{}", v), correct);
///     Ok(())
/// }
/// ```
pub fn copy_vector(v: &mut Vector, u: &Vector) -> Result<(), StrError> {
    let n = v.dim();
    if u.dim() != n {
        return Err("vectors are incompatible");
    }
    let n_i32: i32 = to_i32(n);
    dcopy(n_i32, u.as_data(), 1, v.as_mut_data(), 1);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{copy_vector, Vector};
    use crate::vec_approx_eq;

    #[test]
    fn copy_vector_fails_on_wrong_dims() {
        let u = Vector::new(4);
        let mut v = Vector::new(3);
        assert_eq!(copy_vector(&mut v, &u), Err("vectors are incompatible"));
    }

    #[test]
    fn copy_vector_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let mut v = Vector::from(&[100.0, 200.0, 300.0]);
        copy_vector(&mut v, &u).unwrap();
        let correct = &[1.0, 2.0, 3.0];
        vec_approx_eq(&v, correct, 1e-15);
    }
}
