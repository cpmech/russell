use crate::matrix::*;
use crate::vector::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Performs the outer (tensor) product between two vectors resulting in a matrix
///
/// ```text
///   a  :=  alpha * u  outer  v
/// (m,n)           (m)       (n)
/// ```
///
/// # Note
///
/// The rows of matrix a must equal the length of vector u and
/// the columns of matrix a must equal the length of vector v
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// let u = Vector::from(&[1.0, 2.0, 3.0]);
/// let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
/// let mut a = Matrix::new(u.dim(), v.dim());
/// outer(&mut a, 1.0, &u, &v)?;
/// let correct = "┌             ┐\n\
///                │  5 -2  0  1 │\n\
///                │ 10 -4  0  2 │\n\
///                │ 15 -6  0  3 │\n\
///                └             ┘";
/// assert_eq!(format!("{}", a), correct);
/// # Ok(())
/// # }
/// ```
pub fn outer(a: &mut Matrix, alpha: f64, u: &Vector, v: &Vector) -> Result<(), &'static str> {
    let m = u.data.len();
    let n = v.data.len();
    if a.nrow != m || a.ncol != n {
        return Err("matrix and vectors have incompatible dimensions");
    }
    let m_i32: i32 = m.try_into().unwrap();
    let n_i32: i32 = n.try_into().unwrap();
    let lda_i32 = m_i32;
    dger(
        m_i32,
        n_i32,
        alpha,
        &u.data,
        1,
        &v.data,
        1,
        &mut a.data,
        lda_i32,
    );
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn outer_works() -> Result<(), &'static str> {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
        let mut a = Matrix::new(u.data.len(), v.data.len());
        outer(&mut a, 3.0, &u, &v)?;
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[15.0,  -6.0, 0.0, 3.0],
            &[30.0, -12.0, 0.0, 6.0],
            &[45.0, -18.0, 0.0, 9.0],
        ])?;
        assert_vec_approx_eq!(a.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn outer_works_1() -> Result<(), &'static str> {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let v = Vector::from(&[1.0, 1.0, -2.0]);
        let mut a = Matrix::new(u.data.len(), v.data.len());
        outer(&mut a, 1.0, &u, &v)?;
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[1.0, 1.0, -2.0],
            &[2.0, 2.0, -4.0],
            &[3.0, 3.0, -6.0],
            &[4.0, 4.0, -8.0],
        ])?;
        assert_vec_approx_eq!(a.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn mat_vec_mul_fail_on_wrong_dimensions() {
        let u = Vector::new(2);
        let v = Vector::new(3);
        let mut a_1x3 = Matrix::new(1, 3);
        let mut a_2x1 = Matrix::new(2, 1);
        assert_eq!(
            outer(&mut a_1x3, 1.0, &u, &v),
            Err("matrix and vectors have incompatible dimensions")
        );
        assert_eq!(
            outer(&mut a_2x1, 1.0, &u, &v),
            Err("matrix and vectors have incompatible dimensions")
        );
    }
}
