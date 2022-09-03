use super::ComplexVector;
use crate::StrError;
use russell_openblas::{to_i32, zcopy};

/// Copies vector (complex version)
///
/// ```text
/// v := u
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{complex_copy_vector, ComplexVector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = ComplexVector::from(&[1.0, 2.0, 3.0]);
///     let mut v = ComplexVector::from(&[-1.0, -2.0, -3.0]);
///     complex_copy_vector(&mut v, &u)?;
///     let correct = "┌      ┐\n\
///                    │ 1+0i │\n\
///                    │ 2+0i │\n\
///                    │ 3+0i │\n\
///                    └      ┘";
///     assert_eq!(format!("{}", v), correct);
///     Ok(())
/// }
/// ```
pub fn complex_copy_vector(v: &mut ComplexVector, u: &ComplexVector) -> Result<(), StrError> {
    let n = v.dim();
    if u.dim() != n {
        return Err("vectors are incompatible");
    }
    let n_i32: i32 = to_i32(n);
    zcopy(n_i32, u.as_data(), 1, v.as_mut_data(), 1);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_copy_vector, ComplexVector};
    use crate::complex_vec_approx_eq;
    use crate::StrError;
    use num_complex::Complex64;

    #[test]
    fn complex_copy_vector_fails_on_wrong_dims() {
        let u = ComplexVector::new(4);
        let mut v = ComplexVector::new(3);
        assert_eq!(complex_copy_vector(&mut v, &u), Err("vectors are incompatible"));
    }

    #[test]
    fn complex_copy_vector_works() -> Result<(), StrError> {
        let u = ComplexVector::from(&[1.0, 2.0, 3.0]);
        let mut v = ComplexVector::from(&[100.0, 200.0, 300.0]);
        complex_copy_vector(&mut v, &u)?;
        let correct = &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        complex_vec_approx_eq(&v, correct, 1e-15);
        Ok(())
    }
}
