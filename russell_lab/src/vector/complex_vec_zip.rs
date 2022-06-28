use crate::ComplexVector;
use crate::StrError;

/// Zips two arrays (real and imag) to make a new ComplexVector
///
/// # Example
///
/// ```
/// use russell_lab::{complex_vec_zip, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let v = complex_vec_zip(&[1.0, 2.0, 3.0], &[0.1, 0.2, 0.3])?;
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
pub fn complex_vec_zip(real: &[f64], imag: &[f64]) -> Result<ComplexVector, StrError> {
    let n = real.len();
    if imag.len() != n {
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
    use crate::StrError;

    #[test]
    fn complex_vec_zip_handles_errors() {
        assert_eq!(complex_vec_zip(&[1.0], &[]).err(), Some("arrays are incompatible"));
    }

    #[test]
    fn complex_vec_zip_works() -> Result<(), StrError> {
        let v = complex_vec_zip(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])?;
        assert_eq!(
            format!("{}", v),
            "┌      ┐\n\
             │ 1+4i │\n\
             │ 2+5i │\n\
             │ 3+6i │\n\
             └      ┘"
        );
        Ok(())
    }
}
