use crate::AsArray2D;
use crate::ComplexMatrix;
use crate::StrError;

/// Zips two arrays (real and imag) to make a new ComplexMatrix
///
/// # Example
///
/// ```
/// use russell_lab::{complex_mat_zip, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///     let b = &[[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6]];
///     let c = complex_mat_zip(a, b)?;
///     assert_eq!(
///         format!("{}", c),
///         "┌                      ┐\n\
///          │ 1+0.1i 2+0.2i 3+0.3i │\n\
///          │ 4-0.4i 5-0.5i 6-0.6i │\n\
///          └                      ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn complex_mat_zip<'a, T>(real: &'a T, imag: &'a T) -> Result<ComplexMatrix, StrError>
where
    T: AsArray2D<'a, f64>,
{
    let (m, n) = real.size();
    let (mm, nn) = imag.size();
    if mm != m || nn != n {
        return Err("2d arrays are incompatible");
    }
    let mut a = ComplexMatrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            a[i][j].re = real.at(i, j);
            a[i][j].im = imag.at(i, j)
        }
    }
    Ok(a)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_mat_zip;
    use crate::StrError;

    #[test]
    fn complex_mat_zip_handles_errors() {
        let a: &[&[f64]] = &[&[1.0, 2.0]];
        let b: &[&[f64]] = &[&[1.0]];
        assert_eq!(complex_mat_zip(&a, &b).err(), Some("2d arrays are incompatible"));
    }

    #[test]
    fn complex_mat_zip_works() -> Result<(), StrError> {
        let a = &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = &[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let c = complex_mat_zip(a, b)?;
        assert_eq!(
            format!("{}", c),
            "┌                      ┐\n\
             │ 1+0.1i 2+0.2i 3+0.3i │\n\
             │ 4+0.4i 5+0.5i 6+0.6i │\n\
             └                      ┘"
        );
        Ok(())
    }
}
