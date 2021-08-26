use crate::matrix::*;
use russell_openblas::*;

// constants
const ZERO_DETERMINANT: f64 = 1e-15;
const SINGLE_VALUE_TOLERANCE: f64 = 1e-14;

/// Computes the inverse or pseudo-inverse matrix and returns the determinant (if square matrix)
///
/// ```text
///   ai  :=  inverse(a)
/// (n,m)       (m,n)
/// ```
///
/// # Output
///
/// * `ai` -- (n,m) inverse matrix
/// * the matrix determinant if m == n
///
/// # Input
///
/// * `a` -- (m,n) matrix, symmetric or not
///
/// # Examples
///
/// # First
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// let mut a = Matrix::from(&[
///     &[-1.0,  1.5],
///     &[ 1.0, -1.0],
/// ])?;
/// let mut ai = Matrix::new(2, 2);
/// inverse(&mut ai, &mut a)?;
/// let ai_correct = "┌     ┐\n\
///                   │ 2 3 │\n\
///                   │ 2 2 │\n\
///                   └     ┘";
/// assert_eq!(format!("{}", ai), ai_correct);
/// # Ok(())
/// # }
/// ```
///
/// # Second
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// let mut a = Matrix::from(&[
///     &[1.0, 2.0, 3.0],
///     &[0.0, 1.0, 4.0],
///     &[5.0, 6.0, 0.0],
/// ])?;
/// let mut ai = Matrix::new(3, 3);
/// inverse(&mut ai, &mut a)?;
/// let ai_correct = "┌             ┐\n\
///                   │ -24  18   5 │\n\
///                   │  20 -15  -4 │\n\
///                   │  -5   4   1 │\n\
///                   └             ┘";
/// assert_eq!(format!("{}", ai), ai_correct);
/// # Ok(())
/// # }
/// ```
pub fn inverse(ai: &mut Matrix, a: &Matrix) -> Result<f64, &'static str> {
    // check
    let (m, n) = (a.nrow, a.ncol);
    if ai.nrow != n || ai.ncol != m {
        return Err("[ai] matrix has wrong dimensions");
    }

    // handle small square matrix
    if m == 1 && n == 1 {
        let det = a.get(0, 0);
        if f64::abs(det) <= ZERO_DETERMINANT {
            return Err("cannot compute inverse due to zero determinant");
        }
        ai.set(0, 0, 1.0 / det);
        return Ok(det);
    }

    if m == 2 && n == 2 {
        let det = a.get(0, 0) * a.get(1, 1) - a.get(0, 1) * a.get(1, 0);
        if f64::abs(det) <= ZERO_DETERMINANT {
            return Err("cannot compute inverse due to zero determinant");
        }
        ai.set(0, 0, a.get(1, 1) / det);
        ai.set(0, 1, -a.get(0, 1) / det);
        ai.set(1, 0, -a.get(1, 0) / det);
        ai.set(1, 1, a.get(0, 0) / det);
        return Ok(det);
    }

    if m == 3 && n == 3 {
        let det = a.get(0, 0) * (a.get(1, 1) * a.get(2, 2) - a.get(1, 2) * a.get(2, 1))
            - a.get(0, 1) * (a.get(1, 0) * a.get(2, 2) - a.get(1, 2) * a.get(2, 0))
            + a.get(0, 2) * (a.get(1, 0) * a.get(2, 1) - a.get(1, 1) * a.get(2, 0));
        if f64::abs(det) <= ZERO_DETERMINANT {
            return Err("cannot compute inverse due to zero determinant");
        }

        #[rustfmt::skip]
		ai.set(0, 0, (a.get(1, 1)*a.get(2, 2)-a.get(1, 2)*a.get(2, 1))/det);
        #[rustfmt::skip]
		ai.set(0, 1, (a.get(0, 2)*a.get(2, 1)-a.get(0, 1)*a.get(2, 2))/det);
        #[rustfmt::skip]
		ai.set(0, 2, (a.get(0, 1)*a.get(1, 2)-a.get(0, 2)*a.get(1, 1))/det);

        #[rustfmt::skip]
		ai.set(1, 0, (a.get(1, 2)*a.get(2, 0)-a.get(1, 0)*a.get(2, 2))/det);
        #[rustfmt::skip]
		ai.set(1, 1, (a.get(0, 0)*a.get(2, 2)-a.get(0, 2)*a.get(2, 0))/det);
        #[rustfmt::skip]
		ai.set(1, 2, (a.get(0, 2)*a.get(1, 0)-a.get(0, 0)*a.get(1, 2))/det);

        #[rustfmt::skip]
		ai.set(2, 0, (a.get(1, 0)*a.get(2, 1)-a.get(1, 1)*a.get(2, 0))/det);
        #[rustfmt::skip]
		ai.set(2, 1, (a.get(0, 1)*a.get(2, 0)-a.get(0, 0)*a.get(2, 1))/det);
        #[rustfmt::skip]
		ai.set(2, 2, (a.get(0, 0)*a.get(1, 1)-a.get(0, 1)*a.get(1, 0))/det);
        return Ok(det);
    }

    // copy a into ai
    let min_mn = if m < n { m } else { n };
    let m_i32 = to_i32(m);
    let n_i32 = to_i32(n);
    dcopy(m_i32 * n_i32, &a.data, 1, &mut ai.data, 1);

    // handle square matrix
    if m == n {
        let lda_i32 = m_i32;
        let mut ipiv = vec![0_i32; min_mn];
        dgetrf(m_i32, n_i32, &mut ai.data, lda_i32, &mut ipiv)?;
        dgetri(n_i32, &mut ai.data, lda_i32, &ipiv)?;
        let mut det = 1.0;
        for i in 0..m_i32 {
            let iu = i as usize;
            // NOTE: ipiv are 1-based indices
            if ipiv[iu] - 1 == i {
                det = det * ai.get(iu, iu);
            } else {
                det = -det * ai.get(iu, iu);
            }
        }
        return Ok(det);
    }

    // singular value decomposition
    let mut s = vec![0.0; min_mn];
    let mut u = vec![0.0; m * m];
    let mut vt = vec![0.0; n * n];
    let mut superb = vec![0.0; min_mn];
    dgesvd(
        b'A',
        b'A',
        m_i32,
        n_i32,
        &mut ai.data,
        m_i32,
        &mut s,
        &mut u,
        m_i32,
        &mut vt,
        n_i32,
        &mut superb,
    )?;

    // handle rectangular matrix => pseudo-inverse
    for i in 0..n {
        for j in 0..m {
            ai.data[i + j * n] = 0.0;
            for k in 0..min_mn {
                if s[k] > SINGLE_VALUE_TOLERANCE {
                    ai.data[i + j * n] += vt[k + i * n] * u[j + k * m] / s[k];
                }
            }
        }
    }

    // done
    Ok(0.0)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn inverse_fails_on_wrong_dimensions() {
        let mut a_2x3 = Matrix::new(2, 3);
        let mut ai_1x2 = Matrix::new(1, 2);
        let mut ai_2x1 = Matrix::new(2, 1);
        assert_eq!(
            inverse(&mut ai_1x2, &mut a_2x3),
            Err("[ai] matrix has wrong dimensions")
        );
        assert_eq!(
            inverse(&mut ai_2x1, &mut a_2x3),
            Err("[ai] matrix has wrong dimensions")
        );
    }

    #[test]
    fn inverse_1x1_works() -> Result<(), &'static str> {
        let mut a_1x1 = Matrix::from(&[&[2.0]])?;
        let mut ai_1x1 = Matrix::new(1, 1);
        let det = inverse(&mut ai_1x1, &mut a_1x1)?;
        assert_eq!(det, 2.0);
        assert_vec_approx_eq!(ai_1x1.data, &[0.5], 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_1x1_fails_on_zero_det() -> Result<(), &'static str> {
        let mut a_1x1 = Matrix::from(&[&[1e-15]])?;
        let mut ai_1x1 = Matrix::new(1, 1);
        let res = inverse(&mut ai_1x1, &mut a_1x1);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
        Ok(())
    }

    #[test]
    fn inverse_2x2_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a_2x2 = Matrix::from(&[
            &[1.0, 2.0],
            &[3.0, 2.0],
        ])?;
        let mut ai_2x2 = Matrix::new(2, 2);
        let det = inverse(&mut ai_2x2, &mut a_2x2)?;
        assert_eq!(det, -4.0);
        assert_vec_approx_eq!(ai_2x2.data, &[-0.5, 0.75, 0.5, -0.25], 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_2x2_fails_on_zero_det() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a_2x2 = Matrix::from(&[
            &[   -1.0, 3.0/2.0],
            &[2.0/3.0,    -1.0],
        ])?;
        let mut ai_2x2 = Matrix::new(2, 2);
        let res = inverse(&mut ai_2x2, &mut a_2x2);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
        Ok(())
    }

    #[test]
    fn inverse_3x3_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a_3x3 = Matrix::from(&[
            &[1.0, 2.0, 3.0],
            &[0.0, 4.0, 5.0],
            &[1.0, 0.0, 6.0],
        ])?;
        let mut ai_3x3 = Matrix::new(3, 3);
        let det = inverse(&mut ai_3x3, &mut a_3x3)?;
        assert_eq!(det, 22.0);
        #[rustfmt::skip]
        let ai_correct = Matrix::from(&[
            &[ 12.0/11.0, -6.0/11.0, -1.0/11.0],
            &[  2.5/11.0,  1.5/11.0, -2.5/11.0],
            &[ -2.0/11.0,  1.0/11.0,  2.0/11.0],
        ])?;
        assert_vec_approx_eq!(ai_3x3.data, ai_correct.data, 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_3x3_fails_on_zero_det() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a_3x3 = Matrix::from(&[
            &[1.0, 0.0, 3.0],
            &[0.0, 0.0, 5.0],
            &[1.0, 0.0, 6.0],
        ])?;
        let mut ai_3x3 = Matrix::new(3, 3);
        let res = inverse(&mut ai_3x3, &mut a_3x3);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
        Ok(())
    }

    #[test]
    fn pseudo_inverse_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[-5.773502691896260e-01, -5.773502691896260e-01, 1.000000000000000e+00],
            &[ 5.773502691896260e-01, -5.773502691896260e-01, 1.000000000000000e+00],
            &[-5.773502691896260e-01,  5.773502691896260e-01, 1.000000000000000e+00],
            &[ 5.773502691896260e-01,  5.773502691896260e-01, 1.000000000000000e+00],
        ])?;
        let (m, n) = a.dims();
        let mut ai = Matrix::new(n, m);
        inverse(&mut ai, &a)?;
        #[rustfmt::skip]
        let ai_correct = Matrix::from(&[
            &[-4.330127018922192e-01,  4.330127018922192e-01, -4.330127018922192e-01, 4.330127018922192e-01],
            &[-4.330127018922192e-01, -4.330127018922192e-01,  4.330127018922192e-01, 4.330127018922192e-01],
            &[ 2.500000000000000e-01,  2.500000000000000e-01,  2.500000000000000e-01, 2.500000000000000e-01],
        ])?;
        assert_vec_approx_eq!(ai.data, ai_correct.data, 1e-15);
        Ok(())
    }
}
