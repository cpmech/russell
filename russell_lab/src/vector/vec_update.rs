use super::Vector;
use crate::{to_i32, StrError};

extern "C" {
    fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32);
}

/// Updates vector based on another vector
///
/// ```text
/// v += α⋅u
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{vec_update, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[10.0, 20.0, 30.0]);
///     let mut v = Vector::from(&[10.0, 20.0, 30.0]);
///     vec_update(&mut v, 0.1, &u)?;
///     let correct = "┌    ┐\n\
///                    │ 11 │\n\
///                    │ 22 │\n\
///                    │ 33 │\n\
///                    └    ┘";
///     assert_eq!(format!("{}", v), correct);
///     Ok(())
/// }
/// ```
pub fn vec_update(v: &mut Vector, alpha: f64, u: &Vector) -> Result<(), StrError> {
    let n = v.dim();
    if u.dim() != n {
        return Err("vectors are incompatible");
    }
    let n_i32 = to_i32(n);
    unsafe {
        cblas_daxpy(n_i32, alpha, u.as_data().as_ptr(), 1, v.as_mut_data().as_mut_ptr(), 1);
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_update, Vector};
    use russell_chk::vec_approx_eq;

    #[test]
    fn vec_update_fails_on_wrong_dims() {
        let u = Vector::new(4);
        let mut v = Vector::new(3);
        assert_eq!(vec_update(&mut v, 1.0, &u), Err("vectors are incompatible"));
    }

    #[test]
    fn vec_update_works() {
        let u = Vector::from(&[10.0, 20.0, 30.0]);
        let mut v = Vector::from(&[100.0, 200.0, 300.0]);
        vec_update(&mut v, 2.0, &u).unwrap();
        let correct = &[120.0, 240.0, 360.0];
        vec_approx_eq(v.as_data(), correct, 1e-15);
    }
}
