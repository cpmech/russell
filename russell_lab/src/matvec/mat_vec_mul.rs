use crate::matrix::*;
use crate::vector::*;
use russell_openblas::*;

/// Performs the matrix-vector multiplication resulting in a vector
///
/// ```text
///  v  :=  α ⋅  a   ⋅  u
/// (m)        (m,n)   (n)
/// ```
///
/// # Note
///
/// The length of vector u must equal the rows of matrix a and
/// the length of vector v must equal the columns of matrix a
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// let a = Matrix::from(&[
///     &[ 5.0, -2.0, 1.0],
///     &[-4.0,  0.0, 2.0],
///     &[15.0, -6.0, 0.0],
///     &[ 3.0,  5.0, 1.0],
/// ])?;
/// let u = Vector::from(&[1.0, 2.0, 3.0]);
/// let mut v = Vector::new(a.nrow());
/// mat_vec_mul(&mut v, 0.5, &a, &u)?;
/// let correct = "┌     ┐\n\
///                │   2 │\n\
///                │   1 │\n\
///                │ 1.5 │\n\
///                │   8 │\n\
///                └     ┘";
/// assert_eq!(format!("{}", v), correct);
/// # Ok(())
/// # }
/// ```
pub fn mat_vec_mul(v: &mut Vector, alpha: f64, a: &Matrix, u: &Vector) -> Result<(), &'static str> {
    let m = v.data.len();
    let n = u.data.len();
    if m != a.nrow || n != a.ncol {
        return Err("matrix and vectors have incompatible dimensions");
    }
    let m_i32: i32 = to_i32(m);
    let n_i32: i32 = to_i32(n);
    let lda_i32 = m_i32;
    dgemv(
        false,
        m_i32,
        n_i32,
        alpha,
        &a.data,
        lda_i32,
        &u.data,
        1,
        0.0,
        &mut v.data,
        1,
    );
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn mat_vec_mul_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[ 5.0, -2.0, 0.0, 1.0],
            &[10.0, -4.0, 0.0, 2.0],
            &[15.0, -6.0, 0.0, 3.0],
        ])?;
        let u = Vector::from(&[1.0, 3.0, 8.0, 5.0]);
        let mut v = Vector::new(a.nrow());
        mat_vec_mul(&mut v, 1.0, &a, &u)?;
        let correct = &[4.0, 8.0, 12.0];
        assert_vec_approx_eq!(v.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn mat_vec_mul_fails_on_wrong_dimensions() {
        let u = Vector::new(2);
        let a_1x2 = Matrix::new(1, 2);
        let a_3x1 = Matrix::new(3, 1);
        let mut v = Vector::new(3);
        assert_eq!(
            mat_vec_mul(&mut v, 1.0, &a_1x2, &u),
            Err("matrix and vectors have incompatible dimensions")
        );
        assert_eq!(
            mat_vec_mul(&mut v, 1.0, &a_3x1, &u),
            Err("matrix and vectors have incompatible dimensions")
        );
    }
}
