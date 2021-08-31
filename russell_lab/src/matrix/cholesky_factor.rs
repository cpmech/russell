use crate::matrix::*;
use russell_openblas::*;

/// Performs the Cholesky factorization of a symmetric positive-definite matrix
///
/// Finds `l` such that:
///
/// ```text
/// a = l⋅lᵀ
/// ```
///
/// where `l` is a lower-triangular matrix
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// // import
/// use russell_lab::*;
///
/// // set matrix
/// let a = Matrix::from(&[
///     &[  4.0,  12.0, -16.0],
///     &[ 12.0,  37.0, -43.0],
///     &[-16.0, -43.0,  98.0],
/// ])?;
///
/// // perform factorization
/// let m = a.nrow();
/// let mut l = Matrix::new(m, m);
/// cholesky_factor(&mut l, &a)?;
///
/// // compare with solution
/// let l_correct = "┌          ┐\n\
///                  │  2  0  0 │\n\
///                  │  6  1  0 │\n\
///                  │ -8  5  3 │\n\
///                  └          ┘";
/// assert_eq!(format!("{}", l), l_correct);
///
/// // check if l⋅lᵀ == a
/// let mut l_lt = Matrix::new(m, m);
/// for i in 0..m {
///     for j in 0..m {
///         for k in 0..m {
///             l_lt.plus_equal(i, j, l.get(i, k)? * l.get(j, k)?);
///         }
///     }
/// }
/// let l_lt_correct = "┌             ┐\n\
///                     │   4  12 -16 │\n\
///                     │  12  37 -43 │\n\
///                     │ -16 -43  98 │\n\
///                     └             ┘";
/// assert_eq!(format!("{}", l), l_correct);
/// # Ok(())
/// # }
/// ```
pub fn cholesky_factor(l: &mut Matrix, a: &Matrix) -> Result<(), &'static str> {
    // check
    let (m, n) = (a.nrow, a.ncol);
    if m != n {
        return Err("matrix must be square");
    }

    // copy lower+diagonal part and set upper part to zero
    for i in 0..m {
        for j in 0..n {
            if i >= j {
                l.data[i + j * m] = a.data[i + j * m];
            } else {
                l.data[i + j * m] = 0.0;
            }
        }
    }

    // perform factorization
    let m_i32 = to_i32(m);
    let lda_i32 = m_i32;
    dpotrf(false, m_i32, &mut l.data, lda_i32)?;

    // done
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn cholesky_factor_fails_on_non_square() {
        let a = Matrix::new(3, 4);
        let m = a.nrow;
        let mut l = Matrix::new(m, m);
        assert_eq!(cholesky_factor(&mut l, &a), Err("matrix must be square"));
    }

    #[test]
    fn cholesky_factor_3x3_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[25.0, 15.0, -5.0],
            &[15.0, 18.0,  0.0],
            &[-5.0,  0.0, 11.0],
        ])?;
        let m = a.nrow;
        let mut l = Matrix::new(m, m);
        cholesky_factor(&mut l, &a)?;
        #[rustfmt::skip]
        let l_correct = Matrix::from(&[
            &[ 5.0, 0.0, 0.0],
            &[ 3.0, 3.0, 0.0],
            &[-1.0, 1.0, 3.0],
        ])?;
        assert_vec_approx_eq!(l.data, l_correct.data, 1e-15);
        let mut l_lt = Matrix::new(m, m);
        for i in 0..m {
            for j in 0..m {
                for k in 0..m {
                    l_lt.plus_equal(i, j, l.data[i + k * m] * l.data[j + k * m]);
                }
            }
        }
        assert_vec_approx_eq!(l_lt.data, a.data, 1e-15);
        Ok(())
    }

    #[test]
    fn cholesky_factor_5x5_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[2.0, 1.0, 1.0, 3.0, 2.0],
            &[1.0, 2.0, 2.0, 1.0, 1.0],
            &[1.0, 2.0, 9.0, 1.0, 5.0],
            &[3.0, 1.0, 1.0, 7.0, 1.0],
            &[2.0, 1.0, 5.0, 1.0, 8.0],
        ])?;
        let m = a.nrow;
        let mut l = Matrix::new(m, m);
        cholesky_factor(&mut l, &a)?;
        let sqrt2 = std::f64::consts::SQRT_2;
        #[rustfmt::skip]
        let l_correct = Matrix::from(&[
            &[    sqrt2,                 0.0,                0.0,                     0.0,   0.0],
            &[1.0/sqrt2,  f64::sqrt(3.0/2.0),                0.0,                     0.0,   0.0],
            &[1.0/sqrt2,  f64::sqrt(3.0/2.0),     f64::sqrt(7.0),                     0.0,   0.0],
            &[3.0/sqrt2, -1.0/f64::sqrt(6.0),                0.0,      f64::sqrt(7.0/3.0),   0.0],
            &[    sqrt2,                 0.0, 4.0/f64::sqrt(7.0), -2.0*f64::sqrt(3.0/7.0), sqrt2],
        ])?;
        assert_vec_approx_eq!(l.data, l_correct.data, 1e-15);
        let mut l_lt = Matrix::new(m, m);
        for i in 0..m {
            for j in 0..m {
                for k in 0..m {
                    l_lt.plus_equal(i, j, l.data[i + k * m] * l.data[j + k * m]);
                }
            }
        }
        assert_vec_approx_eq!(l_lt.data, a.data, 1e-15);
        Ok(())
    }
}
