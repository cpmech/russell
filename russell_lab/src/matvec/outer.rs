use crate::matrix::Matrix;
use crate::vector::Vector;
use russell_openblas::{dger, to_i32};

/// Performs the outer (tensor) product between two vectors resulting in a matrix
///
/// ```text
///   a  :=   α ⋅ u  outer  v
/// (m,n)        (m)       (n)
/// ```
///
/// # Note
///
/// The rows of matrix a must equal the length of vector u and
/// the columns of matrix a must equal the length of vector v
///
/// # Example
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
    let m = u.dim();
    let n = v.dim();
    if a.nrow() != m || a.ncol() != n {
        return Err("matrix and vectors are incompatible");
    }
    let m_i32: i32 = to_i32(m);
    let n_i32: i32 = to_i32(n);
    dger(m_i32, n_i32, alpha, u.as_data(), 1, v.as_data(), 1, a.as_mut_data());
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{outer, Matrix, Vector};
    use russell_chk::*;

    #[test]
    fn mat_vec_mul_fail_on_wrong_dims() {
        let u = Vector::new(2);
        let v = Vector::new(3);
        let mut a_1x3 = Matrix::new(1, 3);
        let mut a_2x1 = Matrix::new(2, 1);
        assert_eq!(
            outer(&mut a_1x3, 1.0, &u, &v),
            Err("matrix and vectors are incompatible")
        );
        assert_eq!(
            outer(&mut a_2x1, 1.0, &u, &v),
            Err("matrix and vectors are incompatible")
        );
    }

    #[test]
    fn outer_works() -> Result<(), &'static str> {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
        let (m, n) = (u.dim(), v.dim());
        let mut a = Matrix::new(m, n);
        outer(&mut a, 3.0, &u, &v)?;
        #[rustfmt::skip]
        let correct = [
            15.0,  -6.0, 0.0, 3.0,
            30.0, -12.0, 0.0, 6.0,
            45.0, -18.0, 0.0, 9.0,
        ];
        assert_vec_approx_eq!(a.as_data(), correct, 1e-15);
        Ok(())
    }

    #[test]
    fn outer_works_1() -> Result<(), &'static str> {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let v = Vector::from(&[1.0, 1.0, -2.0]);
        let (m, n) = (u.dim(), v.dim());
        let mut a = Matrix::new(m, n);
        outer(&mut a, 1.0, &u, &v)?;
        #[rustfmt::skip]
        let correct = [
            1.0, 1.0, -2.0,
            2.0, 2.0, -4.0,
            3.0, 3.0, -6.0,
            4.0, 4.0, -8.0,
        ];
        assert_vec_approx_eq!(a.as_data(), correct, 1e-15);
        Ok(())
    }
}
