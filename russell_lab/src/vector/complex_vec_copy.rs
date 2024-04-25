use super::ComplexVector;
use crate::{to_i32, StrError};
use num_complex::Complex64;

extern "C" {
    // Copies a vector into another
    // <https://www.netlib.org/lapack/explore-html/d6/d53/zcopy_8f.html>
    fn cblas_zcopy(n: i32, x: *const Complex64, incx: i32, y: *mut Complex64, incy: i32);
}

/// (zcopy) Copies a vector into another
///
/// ```text
/// v := u
/// ```
///
/// See also: <https://www.netlib.org/lapack/explore-html/d6/d53/zcopy_8f.html>
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let u = ComplexVector::from(&[1.0, 2.0, 3.0]);
///     let mut v = ComplexVector::from(&[-1.0, -2.0, -3.0]);
///     complex_vec_copy(&mut v, &u)?;
///     let correct = "┌      ┐\n\
///                    │ 1+0i │\n\
///                    │ 2+0i │\n\
///                    │ 3+0i │\n\
///                    └      ┘";
///     assert_eq!(format!("{}", v), correct);
///     Ok(())
/// }
/// ```
pub fn complex_vec_copy(v: &mut ComplexVector, u: &ComplexVector) -> Result<(), StrError> {
    let n = v.dim();
    if u.dim() != n {
        return Err("vectors are incompatible");
    }
    let n_i32 = to_i32(n);
    unsafe {
        cblas_zcopy(n_i32, u.as_data().as_ptr(), 1, v.as_mut_data().as_mut_ptr(), 1);
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_vec_copy, ComplexVector};
    use crate::{complex_vec_approx_eq, cpx};
    use num_complex::Complex64;

    #[test]
    fn complex_vec_copy_fails_on_wrong_dims() {
        let u = ComplexVector::new(4);
        let mut v = ComplexVector::new(3);
        assert_eq!(complex_vec_copy(&mut v, &u), Err("vectors are incompatible"));
    }

    #[test]
    fn complex_vec_copy_works() {
        let u = ComplexVector::from(&[1.0, 2.0, 3.0]);
        let mut v = ComplexVector::from(&[100.0, 200.0, 300.0]);
        complex_vec_copy(&mut v, &u).unwrap();
        let correct = &[cpx!(1.0, 0.0), cpx!(2.0, 0.0), cpx!(3.0, 0.0)];
        complex_vec_approx_eq(&v, correct, 1e-15);
    }
}
