use super::Vector;
use crate::StrError;
use russell_openblas::{daxpy, to_i32};

/// Updates vector based on another vector (axpy)
///
/// ```text
/// v += α⋅u
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{update_vector, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[10.0, 20.0, 30.0]);
///     let mut v = Vector::from(&[10.0, 20.0, 30.0]);
///     update_vector(&mut v, 0.1, &u)?;
///     let correct = "┌    ┐\n\
///                    │ 11 │\n\
///                    │ 22 │\n\
///                    │ 33 │\n\
///                    └    ┘";
///     assert_eq!(format!("{}", v), correct);
///     Ok(())
/// }
/// ```
pub fn update_vector(v: &mut Vector, alpha: f64, u: &Vector) -> Result<(), StrError> {
    let n = v.dim();
    if u.dim() != n {
        return Err("vectors are incompatible");
    }
    let n_i32: i32 = to_i32(n);
    daxpy(n_i32, alpha, u.as_data(), 1, v.as_mut_data(), 1);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{update_vector, Vector};
    use crate::StrError;
    use russell_chk::assert_vec_approx_eq;

    #[test]
    fn update_vector_fails_on_wrong_dims() {
        let u = Vector::new(4);
        let mut v = Vector::new(3);
        assert_eq!(update_vector(&mut v, 1.0, &u), Err("vectors are incompatible"));
    }

    #[test]
    fn update_vector_works() -> Result<(), StrError> {
        let u = Vector::from(&[10.0, 20.0, 30.0]);
        let mut v = Vector::from(&[100.0, 200.0, 300.0]);
        update_vector(&mut v, 2.0, &u)?;
        let correct = &[120.0, 240.0, 360.0];
        assert_vec_approx_eq!(v.as_data(), correct, 1e-15);
        Ok(())
    }
}
