use super::Vector;
use crate::{to_i32, StrError};

extern "C" {
    fn cblas_dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy: i32);
}

/// Copies vector
///
/// ```text
/// v := u
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{vec_copy, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[1.0, 2.0, 3.0]);
///     let mut v = Vector::from(&[-1.0, -2.0, -3.0]);
///     vec_copy(&mut v, &u)?;
///     let correct = "┌   ┐\n\
///                    │ 1 │\n\
///                    │ 2 │\n\
///                    │ 3 │\n\
///                    └   ┘";
///     assert_eq!(format!("{}", v), correct);
///     Ok(())
/// }
/// ```
pub fn vec_copy(v: &mut Vector, u: &Vector) -> Result<(), StrError> {
    let n = v.dim();
    if u.dim() != n {
        return Err("vectors are incompatible");
    }
    let n_i32 = to_i32(n);
    unsafe {
        cblas_dcopy(n_i32, u.as_data().as_ptr(), 1, v.as_mut_data().as_mut_ptr(), 1);
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_copy, Vector};
    use russell_chk::vec_approx_eq;

    #[test]
    fn vec_copy_fails_on_wrong_dims() {
        let u = Vector::new(4);
        let mut v = Vector::new(3);
        assert_eq!(vec_copy(&mut v, &u), Err("vectors are incompatible"));
    }

    #[test]
    fn vec_copy_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let mut v = Vector::from(&[100.0, 200.0, 300.0]);
        vec_copy(&mut v, &u).unwrap();
        let correct = &[1.0, 2.0, 3.0];
        vec_approx_eq(v.as_data(), correct, 1e-15);
    }
}
