use crate::ComplexVector;
use crate::StrError;
use crate::Vector;

/// Zips two arrays (real and imag) to make a new ComplexVector
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let mut v = ComplexVector::new(3);
///     let real = Vector::from(&[1.0, 2.0, 3.0]);
///     let imag = Vector::from(&[0.1, 0.2, 0.3]);
///     complex_vec_zip(&mut v, &real, &imag)?;
///     assert_eq!(
///         format!("{}", v),
///         "┌        ┐\n\
///          │ 1+0.1i │\n\
///          │ 2+0.2i │\n\
///          │ 3+0.3i │\n\
///          └        ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn complex_vec_zip(v: &mut ComplexVector, real: &Vector, imag: &Vector) -> Result<(), StrError> {
    let dim = v.dim();
    if real.dim() != dim || imag.dim() != dim {
        return Err("vectors are incompatible");
    }
    for i in 0..dim {
        v[i].re = real[i];
        v[i].im = imag[i];
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_vec_zip;
    use crate::{ComplexVector, Vector};

    #[test]
    fn complex_vec_zip_handles_errors() {
        let mut v = ComplexVector::new(2);
        let wrong = Vector::new(1);
        let ok = Vector::new(2);
        assert_eq!(
            complex_vec_zip(&mut v, &wrong, &ok).err(),
            Some("vectors are incompatible")
        );
        assert_eq!(
            complex_vec_zip(&mut v, &ok, &wrong).err(),
            Some("vectors are incompatible")
        );
    }

    #[test]
    fn complex_vec_zip_works() {
        let real = Vector::from(&[1.0, 2.0, 3.0]);
        let imag = Vector::from(&[4.0, 5.0, 6.0]);
        let mut v = ComplexVector::new(3);
        complex_vec_zip(&mut v, &real, &imag).unwrap();
        assert_eq!(
            format!("{}", v),
            "┌      ┐\n\
             │ 1+4i │\n\
             │ 2+5i │\n\
             │ 3+6i │\n\
             └      ┘"
        );
    }
}
