use crate::matrix::*;
use russell_openblas::*;

// constants
const ZERO_DETERMINANT: f64 = 1e-15;
const SINGLE_VALUE_RCOND: f64 = 1e-15;

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
/// * Returns the matrix determinant if m == n
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

    // handle zero-sized matrix
    if m == 0 && n == 0 {
        return Ok(0.0);
    }

    // handle small square matrix
    if m == 1 && n == 1 {
        let det = a.data[0];
        if f64::abs(det) <= ZERO_DETERMINANT {
            return Err("cannot compute inverse due to zero determinant");
        }
        ai.data[0] = 1.0 / det;
        return Ok(det);
    }

    if m == 2 && n == 2 {
        let det = a.data[0 + 0 * m] * a.data[1 + 1 * m] - a.data[0 + 1 * m] * a.data[1 + 0 * m];
        if f64::abs(det) <= ZERO_DETERMINANT {
            return Err("cannot compute inverse due to zero determinant");
        }
        ai.data[0 + 0 * n] = a.data[1 + 1 * m] / det;
        ai.data[0 + 1 * n] = -a.data[0 + 1 * m] / det;
        ai.data[1 + 0 * n] = -a.data[1 + 0 * m] / det;
        ai.data[1 + 1 * n] = a.data[0 + 0 * m] / det;
        return Ok(det);
    }

    if m == 3 && n == 3 {
        #[rustfmt::skip]
        let det =
              a.data[0+0*m] * (a.data[1+1*m] * a.data[2+ 2*m] - a.data[1+ 2*m] * a.data[2+ 1*m])
            - a.data[0+1*m] * (a.data[1+0*m] * a.data[2+ 2*m] - a.data[1+ 2*m] * a.data[2+ 0*m])
            + a.data[0+2*m] * (a.data[1+0*m] * a.data[2+ 1*m] - a.data[1+ 1*m] * a.data[2+ 0*m]);

        if f64::abs(det) <= ZERO_DETERMINANT {
            return Err("cannot compute inverse due to zero determinant");
        }

        ai.data[0 + 0 * n] =
            (a.data[1 + 1 * m] * a.data[2 + 2 * m] - a.data[1 + 2 * m] * a.data[2 + 1 * m]) / det;
        ai.data[0 + 1 * n] =
            (a.data[0 + 2 * m] * a.data[2 + 1 * m] - a.data[0 + 1 * m] * a.data[2 + 2 * m]) / det;
        ai.data[0 + 2 * n] =
            (a.data[0 + 1 * m] * a.data[1 + 2 * m] - a.data[0 + 2 * m] * a.data[1 + 1 * m]) / det;

        ai.data[1 + 0 * n] =
            (a.data[1 + 2 * m] * a.data[2 + 0 * m] - a.data[1 + 0 * m] * a.data[2 + 2 * m]) / det;
        ai.data[1 + 1 * n] =
            (a.data[0 + 0 * m] * a.data[2 + 2 * m] - a.data[0 + 2 * m] * a.data[2 + 0 * m]) / det;
        ai.data[1 + 2 * n] =
            (a.data[0 + 2 * m] * a.data[1 + 0 * m] - a.data[0 + 0 * m] * a.data[1 + 2 * m]) / det;

        ai.data[2 + 0 * n] =
            (a.data[1 + 0 * m] * a.data[2 + 1 * m] - a.data[1 + 1 * m] * a.data[2 + 0 * m]) / det;
        ai.data[2 + 1 * n] =
            (a.data[0 + 1 * m] * a.data[2 + 0 * m] - a.data[0 + 0 * m] * a.data[2 + 1 * m]) / det;
        ai.data[2 + 2 * n] =
            (a.data[0 + 0 * m] * a.data[1 + 1 * m] - a.data[0 + 1 * m] * a.data[1 + 0 * m]) / det;

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
        // first, compute the determinant ai.data from dgetrf
        let mut det = 1.0;
        for i in 0..m_i32 {
            let iu = i as usize;
            // NOTE: ipiv are 1-based indices
            if ipiv[iu] - 1 == i {
                det = det * ai.data[iu + iu * n];
            } else {
                det = -det * ai.data[iu + iu * n];
            }
        }
        // second, perform the inversion
        dgetri(n_i32, &mut ai.data, lda_i32, &ipiv)?;
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

    // singular value tolerance (note that singular values are positive or zero)
    let idx_largest = idamax(to_i32(min_mn), &s, 1) as usize;
    let sv_largest = s[idx_largest];
    let sv_tolerance = SINGLE_VALUE_RCOND * sv_largest;

    // handle rectangular matrix => pseudo-inverse
    for i in 0..n {
        for j in 0..m {
            ai.data[i + j * n] = 0.0;
            for k in 0..min_mn {
                if s[k] > sv_tolerance {
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

    /// Computes a⋅ai that should equal I for a square matrix
    fn get_a_times_ai(a: &Matrix, ai: &Matrix) -> Matrix {
        let (m, n) = (a.nrow, a.ncol);
        let mut a_ai = Matrix::new(m, m);
        for i in 0..m {
            for j in 0..m {
                for k in 0..n {
                    a_ai.data[i + j * m] += a.data[i + k * m] * ai.data[k + j * n];
                }
            }
        }
        a_ai
    }

    /// Computes a⋅ai⋅a that should equal a
    fn get_a_times_ai_times_a(a: &Matrix, ai: &Matrix) -> Matrix {
        // compute a⋅ai
        let a_ai = get_a_times_ai(&a, &ai);
        // compute a⋅ai⋅a == a
        let (m, n) = (a.nrow, a.ncol);
        let mut a_ai_a = Matrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                for k in 0..m {
                    a_ai_a.data[i + j * m] += a_ai.data[i + k * m] * a.data[k + j * m];
                }
            }
        }
        a_ai_a
    }

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
    fn inverse_0x0_works() -> Result<(), &'static str> {
        let mut a = Matrix::new(0, 0);
        let mut ai = Matrix::new(0, 0);
        let det = inverse(&mut ai, &mut a)?;
        assert_eq!(det, 0.0);
        assert_eq!(ai.data, []);
        Ok(())
    }

    #[test]
    fn inverse_1x1_works() -> Result<(), &'static str> {
        let mut a = Matrix::from(&[&[2.0]])?;
        let mut ai = Matrix::new(1, 1);
        let det = inverse(&mut ai, &mut a)?;
        assert_eq!(det, 2.0);
        assert_vec_approx_eq!(ai.data, &[0.5], 1e-15);
        let a_ai = get_a_times_ai(&a, &ai);
        assert_vec_approx_eq!(a_ai.data, &[1.0], 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_1x1_fails_on_zero_det() -> Result<(), &'static str> {
        let mut a = Matrix::from(&[&[ZERO_DETERMINANT / 10.0]])?;
        let mut ai = Matrix::new(1, 1);
        let res = inverse(&mut ai, &mut a);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
        Ok(())
    }

    #[test]
    fn inverse_2x2_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[1.0, 2.0],
            &[3.0, 2.0],
        ])?;
        let mut ai = Matrix::new(2, 2);
        let det = inverse(&mut ai, &mut a)?;
        assert_eq!(det, -4.0);
        assert_vec_approx_eq!(ai.data, &[-0.5, 0.75, 0.5, -0.25], 1e-15);
        let a_ai = get_a_times_ai(&a, &ai);
        assert_vec_approx_eq!(a_ai.data, &[1.0, 0.0, 0.0, 1.0], 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_2x2_fails_on_zero_det() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[   -1.0, 3.0/2.0],
            &[2.0/3.0,    -1.0],
        ])?;
        let mut ai = Matrix::new(2, 2);
        let res = inverse(&mut ai, &mut a);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
        Ok(())
    }

    #[test]
    fn inverse_3x3_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[1.0, 2.0, 3.0],
            &[0.0, 4.0, 5.0],
            &[1.0, 0.0, 6.0],
        ])?;
        let mut ai = Matrix::new(3, 3);
        let det = inverse(&mut ai, &mut a)?;
        assert_eq!(det, 22.0);
        #[rustfmt::skip]
        let ai_correct = Matrix::from(&[
            &[ 12.0/11.0, -6.0/11.0, -1.0/11.0],
            &[  2.5/11.0,  1.5/11.0, -2.5/11.0],
            &[ -2.0/11.0,  1.0/11.0,  2.0/11.0],
        ])?;
        assert_vec_approx_eq!(ai.data, ai_correct.data, 1e-15);
        let identity = Matrix::identity(3);
        let a_ai = get_a_times_ai(&a, &ai);
        assert_vec_approx_eq!(a_ai.data, identity.data, 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_3x3_fails_on_zero_det() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[1.0, 0.0, 3.0],
            &[0.0, 0.0, 5.0],
            &[1.0, 0.0, 6.0],
        ])?;
        let mut ai = Matrix::new(3, 3);
        let res = inverse(&mut ai, &mut a);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
        Ok(())
    }

    #[test]
    fn inverse_4x4_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[ 3.0,  0.0,  2.0, -1.0],
            &[ 1.0,  2.0,  0.0, -2.0],
            &[ 4.0,  0.0,  6.0, -3.0],
            &[ 5.0,  0.0,  2.0,  0.0],
        ])?;
        let mut ai = Matrix::new(4, 4);
        let det = inverse(&mut ai, &mut a)?;
        assert_approx_eq!(det, 20.0, 1e-14);
        #[rustfmt::skip]
        let ai_correct = Matrix::from(&[
            &[ 0.6,  0.0, -0.2,  0.0],
            &[-2.5,  0.5,  0.5,  1.0],
            &[-1.5,  0.0,  0.5,  0.5],
            &[-2.2,  0.0,  0.4,  1.0],
        ])?;
        assert_vec_approx_eq!(ai.data, ai_correct.data, 1e-15);
        let identity = Matrix::identity(4);
        let a_ai = get_a_times_ai(&a, &ai);
        assert_vec_approx_eq!(a_ai.data, identity.data, 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_5x5_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[12.0, 28.0, 22.0, 20.0,  8.0],
            &[ 0.0,  3.0,  5.0, 17.0, 28.0],
            &[56.0,  0.0, 23.0,  1.0,  0.0],
            &[12.0, 29.0, 27.0, 10.0,  1.0],
            &[ 9.0,  4.0, 13.0,  8.0, 22.0],
        ])?;
        let mut ai = Matrix::new(5, 5);
        let det = inverse(&mut ai, &mut a)?;
        assert_approx_eq!(det, -167402.0, 1e-9);
        #[rustfmt::skip]
        let ai_correct = Matrix::from(&[
            &[ 6.9128803717996279e-01, -7.4226114383340802e-01, -9.8756287260606410e-02, -6.9062496266472417e-01,  7.2471057693456553e-01],
            &[ 1.5936129795342968e+00, -1.7482347881148397e+00, -2.8304321334273236e-01, -1.5600769405383470e+00,  1.7164430532490673e+00],
            &[-1.6345384165063759e+00,  1.7495848317224429e+00,  2.7469205863729274e-01,  1.6325730875377857e+00, -1.7065745928961444e+00],
            &[-1.1177465024312745e+00,  1.3261729250546601e+00,  2.1243473793622566e-01,  1.1258168958554866e+00, -1.3325766717243535e+00],
            &[ 7.9976941733073770e-01, -8.9457712572131853e-01, -1.4770432850264653e-01, -8.0791149448632715e-01,  9.2990525800169743e-01],
        ])?;
        assert_vec_approx_eq!(ai.data, ai_correct.data, 1e-14);
        let identity = Matrix::identity(5);
        let a_ai = get_a_times_ai(&a, &ai);
        assert_vec_approx_eq!(a_ai.data, identity.data, 1e-13);
        Ok(())
    }

    #[test]
    fn inverse_6x6_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[ 3.46540497998689445e-05, -1.39368151175265866e-05, -1.39368151175265866e-05,  0.00000000000000000e+00, 7.15957288480514429e-23, -2.93617909908697186e+02],
            &[-1.39368151175265866e-05,  3.46540497998689445e-05, -1.39368151175265866e-05,  0.00000000000000000e+00, 7.15957288480514429e-23, -2.93617909908697186e+02],
            &[-1.39368151175265866e-05, -1.39368151175265866e-05,  3.46540497998689445e-05,  0.00000000000000000e+00, 7.15957288480514429e-23, -2.93617909908697186e+02],
            &[ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  4.85908649173955311e-05, 0.00000000000000000e+00,  0.00000000000000000e+00],
            &[ 3.13760264822604860e-18,  3.13760264822604860e-18,  3.13760264822604860e-18,  0.00000000000000000e+00, 1.00000000000000000e+00, -1.93012141894243434e+07],
            &[ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, -0.00000000000000000e+00, 0.00000000000000000e+00,  1.00000000000000000e+00],
        ])?;
        let mut ai = Matrix::new(6, 6);
        let det = inverse(&mut ai, &mut a)?;
        assert_approx_eq!(det, 7.778940633136385e-19, 1e-15);
        #[rustfmt::skip]
        let ai_correct = Matrix::from(&[
            &[ 6.28811662297464645e+04,  4.23011662297464645e+04,  4.23011662297464645e+04, 0.00000000000000000e+00, -1.05591885817167332e-17, 4.33037966311565489e+07],
            &[ 4.23011662297464645e+04,  6.28811662297464645e+04,  4.23011662297464645e+04, 0.00000000000000000e+00, -1.05591885817167332e-17, 4.33037966311565489e+07],
            &[ 4.23011662297464645e+04,  4.23011662297464645e+04,  6.28811662297464645e+04, 0.00000000000000000e+00, -1.05591885817167348e-17, 4.33037966311565489e+07],
            &[ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 2.05800000000000000e+04,  0.00000000000000000e+00, 0.00000000000000000e+00],
            &[-4.62744616057000471e-13, -4.62744616057000471e-13, -4.62744616057000471e-13, 0.00000000000000000e+00,  1.00000000000000000e+00, 1.93012141894243434e+07],
            &[ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0.00000000000000000e+00,  0.00000000000000000e+00, 1.00000000000000000e+00],
        ])?;
        assert_vec_approx_eq!(ai.data, ai_correct.data, 1e-8);
        let identity = Matrix::identity(6);
        let a_ai = get_a_times_ai(&a, &ai);
        assert_vec_approx_eq!(a_ai.data, identity.data, 1e-12);
        Ok(())
    }

    #[test]
    fn pseudo_inverse_4x3_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let data: &[&[f64]] = &[
            &[-5.773502691896260e-01, -5.773502691896260e-01, 1.000000000000000e+00],
            &[ 5.773502691896260e-01, -5.773502691896260e-01, 1.000000000000000e+00],
            &[-5.773502691896260e-01,  5.773502691896260e-01, 1.000000000000000e+00],
            &[ 5.773502691896260e-01,  5.773502691896260e-01, 1.000000000000000e+00],
        ];
        let a = Matrix::from(data)?;
        let (m, n) = (a.nrow, a.ncol);
        let mut ai = Matrix::new(n, m);
        inverse(&mut ai, &a)?;
        #[rustfmt::skip]
        let ai_correct = Matrix::from(&[
            &[-4.330127018922192e-01,  4.330127018922192e-01, -4.330127018922192e-01, 4.330127018922192e-01],
            &[-4.330127018922192e-01, -4.330127018922192e-01,  4.330127018922192e-01, 4.330127018922192e-01],
            &[ 2.500000000000000e-01,  2.500000000000000e-01,  2.500000000000000e-01, 2.500000000000000e-01],
        ])?;
        assert_vec_approx_eq!(ai.data, ai_correct.data, 1e-15);
        let a_copy = Matrix::from(data)?;
        let a_ai_a = get_a_times_ai_times_a(&a_copy, &ai);
        assert_vec_approx_eq!(a_ai_a.data, a_copy.data, 1e-15);
        Ok(())
    }

    #[test]
    fn pseudo_inverse_4x5_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let data: &[&[f64]] = &[
            &[1.0, 0.0, 0.0, 0.0, 2.0],
            &[0.0, 0.0, 3.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 4.0, 0.0, 0.0, 0.0],
        ];
        let a = Matrix::from(data)?;
        let (m, n) = (a.nrow, a.ncol);
        let mut ai = Matrix::new(n, m);
        inverse(&mut ai, &a)?;
        #[rustfmt::skip]
        let ai_correct = Matrix::from(&[
            &[0.2,     0.0, 0.0,     0.0],
            &[0.0,     0.0, 0.0, 1.0/4.0],
            &[0.0, 1.0/3.0, 0.0,     0.0],
            &[0.0,     0.0, 0.0,     0.0],
            &[0.4,     0.0, 0.0,     0.0],
        ])?;
        assert_vec_approx_eq!(ai.data, ai_correct.data, 1e-15);
        let a_copy = Matrix::from(data)?;
        let a_ai_a = get_a_times_ai_times_a(&a_copy, &ai);
        assert_vec_approx_eq!(a_ai_a.data, a_copy.data, 1e-15);
        Ok(())
    }

    #[test]
    fn pseudo_inverse_5x6_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let data: &[&[f64]] = &[
            &[12.0, 28.0, 22.0, 20.0,  8.0, 1.0],
            &[ 0.0,  3.0,  5.0, 17.0, 28.0, 1.0],
            &[56.0,  0.0, 23.0,  1.0,  0.0, 1.0],
            &[12.0, 29.0, 27.0, 10.0,  1.0, 1.0],
            &[ 9.0,  4.0, 13.0,  8.0, 22.0, 1.0],
        ];
        let a = Matrix::from(data)?;
        let (m, n) = (a.nrow, a.ncol);
        let mut ai = Matrix::new(n, m);
        inverse(&mut ai, &a)?;
        #[rustfmt::skip]
        let ai_correct = Matrix::from(&[
            &[ 5.6387724512344639e-01, -6.0176177188969326e-01, -7.6500652148749224e-02, -5.6389938864086908e-01,  5.8595836573334192e-01],
            &[ 1.2836912791395787e+00, -1.4064756360496755e+00, -2.2890726327210095e-01, -1.2518220058421685e+00,  1.3789338004227019e+00],
            &[-1.2866745075158739e+00,  1.3659857664770796e+00,  2.1392850711928030e-01,  1.2865799982753852e+00, -1.3277457214130808e+00],
            &[-8.8185982449865485e-01,  1.0660542211012198e+00,  1.7123094548599221e-01,  8.9119882164767850e-01, -1.0756926383722674e+00],
            &[ 6.6698814093525072e-01, -7.4815557352521045e-01, -1.2451059750508876e-01, -6.7584431870600359e-01,  7.8530451101142418e-01],
            &[-1.1017522295492406e+00,  1.2149323757487696e+00,  1.9244991110051662e-01,  1.0958269819071325e+00, -1.1998242501940171e+00],
        ])?;
        assert_vec_approx_eq!(ai.data, ai_correct.data, 1e-13);
        let a_copy = Matrix::from(data)?;
        let a_ai_a = get_a_times_ai_times_a(&a_copy, &ai);
        assert_vec_approx_eq!(a_ai_a.data, a_copy.data, 1e-12);
        Ok(())
    }

    #[test]
    fn pseudo_inverse_8x6_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let data :&[&[f64]]= &[
            &[64.0,  2.0,  3.0, 61.0, 60.0,  6.0],
            &[ 9.0, 55.0, 54.0, 12.0, 13.0, 51.0],
            &[17.0, 47.0, 46.0, 20.0, 21.0, 43.0],
            &[40.0, 26.0, 27.0, 37.0, 36.0, 30.0],
            &[32.0, 34.0, 35.0, 29.0, 28.0, 38.0],
            &[41.0, 23.0, 22.0, 44.0, 45.0, 19.0],
            &[49.0, 15.0, 14.0, 52.0, 53.0, 11.0],
            &[ 8.0, 58.0, 59.0,  5.0,  4.0, 62.0],
        ];
        let a = Matrix::from(data)?;
        let (m, n) = (a.nrow, a.ncol);
        let mut ai = Matrix::new(n, m);
        inverse(&mut ai, &a)?;
        let a_copy = Matrix::from(data)?;
        let a_ai_a = get_a_times_ai_times_a(&a_copy, &ai);
        assert_vec_approx_eq!(a_ai_a.data, a_copy.data, 1e-13);
        Ok(())
    }
}
