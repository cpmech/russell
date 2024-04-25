use crate::ComplexVector;
use crate::StrError;
use crate::Vector;

/// Zips two arrays (real and imag) to make a new ComplexVector
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// use num_complex::Complex64;
///
/// fn main() -> Result<(), StrError> {
///     let v = ComplexVector::from(&[cpx!(1.0, 0.1), cpx!(2.0, 0.2), cpx!(3.0, 0.3)]);
///     let mut real = Vector::new(3);
///     let mut imag = Vector::new(3);
///     complex_vec_unzip(&mut real, &mut imag, &v)?;
///     assert_eq!(
///         format!("{}", real),
///         "┌   ┐\n\
///          │ 1 │\n\
///          │ 2 │\n\
///          │ 3 │\n\
///          └   ┘"
///     );
///     assert_eq!(
///         format!("{}", imag),
///         "┌     ┐\n\
///          │ 0.1 │\n\
///          │ 0.2 │\n\
///          │ 0.3 │\n\
///          └     ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn complex_vec_unzip(real: &mut Vector, imag: &mut Vector, v: &ComplexVector) -> Result<(), StrError> {
    let dim = v.dim();
    if real.dim() != dim || imag.dim() != dim {
        return Err("vectors are incompatible");
    }
    for i in 0..dim {
        real[i] = v[i].re;
        imag[i] = v[i].im;
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_vec_unzip;
    use crate::{cpx, vec_approx_eq, ComplexVector, Vector};
    use num_complex::Complex64;

    #[test]
    fn complex_vec_unzip_handles_errors() {
        let v = ComplexVector::new(2);
        let mut wrong = Vector::new(1);
        let mut ok = Vector::new(2);
        assert_eq!(
            complex_vec_unzip(&mut wrong, &mut ok, &v).err(),
            Some("vectors are incompatible")
        );
        assert_eq!(
            complex_vec_unzip(&mut ok, &mut wrong, &v).err(),
            Some("vectors are incompatible")
        );
    }

    #[test]
    fn complex_vec_unzip_works() {
        let v = ComplexVector::from(&[cpx!(1.0, 4.0), cpx!(2.0, 5.0), cpx!(3.0, 6.0)]);
        let mut real = Vector::new(3);
        let mut imag = Vector::new(3);
        complex_vec_unzip(&mut real, &mut imag, &v).unwrap();
        vec_approx_eq(&real, &[1.0, 2.0, 3.0], 1e-15);
        vec_approx_eq(&imag, &[4.0, 5.0, 6.0], 1e-15);
    }
}
