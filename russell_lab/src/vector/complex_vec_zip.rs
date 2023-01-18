use crate::ComplexVector;
use crate::StrError;
use crate::Vector;

/// Zips two arrays (real and imag) to make a new ComplexVector
///
/// # Example
///
/// ```
/// use russell_lab::{complex_vec_zip, StrError, Vector};
///
/// fn main() -> Result<(), StrError> {
///     let real = Vector::from(&[1.0, 2.0, 3.0]);
///     let imag = Vector::from(&[0.1, 0.2, 0.3]);
///     let v = complex_vec_zip(&real, &imag)?;
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
pub fn complex_vec_zip(real: &Vector, imag: &Vector) -> Result<ComplexVector, StrError> {
    let n = real.dim();
    if imag.dim() != n {
        return Err("arrays are incompatible");
    }
    let mut v = ComplexVector::new(n);
    for i in 0..n {
        v[i].re = real[i];
        v[i].im = imag[i];
    }
    Ok(v)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_vec_zip;
    use crate::Vector;

    #[test]
    fn complex_vec_zip_handles_errors() {
        assert_eq!(
            complex_vec_zip(&Vector::from(&[1.0]), &Vector::new(0)).err(),
            Some("arrays are incompatible")
        );
    }

    #[test]
    fn complex_vec_zip_works() {
        let real = Vector::from(&[1.0, 2.0, 3.0]);
        let imag = Vector::from(&[4.0, 5.0, 6.0]);
        let v = complex_vec_zip(&real, &imag).unwrap();
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
