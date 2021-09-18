use crate::matrix::*;
use russell_openblas::*;

// constants
const ZERO_DETERMINANT: f64 = 1e-15;

/// Computes the inverse of a square matrix and returns its determinant
///
/// ```text
/// ai := a⁻¹
/// ```
///
/// # Output
///
/// * `ai` -- (m,m) inverse matrix
/// * Returns the matrix determinant
///
/// # Input
///
/// * `a` -- (m,m) matrix, symmetric or not
///
/// # Examples
///
/// ## First -- 2 x 2 square matrix
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// // import
/// use russell_lab::*;
///
/// // set matrix
/// let mut a = Matrix::from(&[
///     [-1.0,  1.5],
///     [ 1.0, -1.0],
/// ]);
/// let a_copy = a.get_copy();
///
/// // compute inverse matrix
/// let mut ai = Matrix::new(2, 2);
/// inverse(&mut ai, &mut a)?;
///
/// // compare with solution
/// let ai_correct = "┌     ┐\n\
///                   │ 2 3 │\n\
///                   │ 2 2 │\n\
///                   └     ┘";
/// assert_eq!(format!("{}", ai), ai_correct);
///
/// // check if a⋅ai == identity
/// let (m, n) = a.dims();
/// let mut a_ai = Matrix::new(m, m);
/// for i in 0..m {
///     for j in 0..m {
///         for k in 0..n {
///             a_ai[i][j] += a_copy[i][k] * ai[k][j];
///         }
///     }
/// }
/// let identity = "┌     ┐\n\
///                 │ 1 0 │\n\
///                 │ 0 1 │\n\
///                 └     ┘";
/// assert_eq!(format!("{}", a_ai), identity);
/// # Ok(())
/// # }
/// ```
///
/// ## Second -- 3 x 3 square matrix
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// // import
/// use russell_lab::*;
///
/// // set matrix
/// let mut a = Matrix::from(&[
///     [1.0, 2.0, 3.0],
///     [0.0, 1.0, 4.0],
///     [5.0, 6.0, 0.0],
/// ]);
/// let a_copy = a.get_copy();
///
/// // compute inverse matrix
/// let mut ai = Matrix::new(3, 3);
/// inverse(&mut ai, &mut a)?;
///
/// // compare with solution
/// let ai_correct = "┌             ┐\n\
///                   │ -24  18   5 │\n\
///                   │  20 -15  -4 │\n\
///                   │  -5   4   1 │\n\
///                   └             ┘";
/// assert_eq!(format!("{}", ai), ai_correct);
///
/// // check if a⋅ai == identity
/// let (m, n) = a.dims();
/// let mut a_ai = Matrix::new(m, m);
/// for i in 0..m {
///     for j in 0..m {
///         for k in 0..n {
///             a_ai[i][j] += a_copy[i][k] * ai[k][j];
///         }
///     }
/// }
/// let identity = "┌       ┐\n\
///                 │ 1 0 0 │\n\
///                 │ 0 1 0 │\n\
///                 │ 0 0 1 │\n\
///                 └       ┘";
/// assert_eq!(format!("{}", a_ai), identity);
/// # Ok(())
/// # }
/// ```
pub fn inverse(ai: &mut Matrix, a: &Matrix) -> Result<f64, &'static str> {
    // check
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if ai.nrow() != m || ai.ncol() != n {
        return Err("matrices are incompatible");
    }

    // handle zero-sized matrix
    if m == 0 {
        return Ok(0.0);
    }

    // handle small matrix
    if m == 1 {
        let det = a[0][0];
        if f64::abs(det) <= ZERO_DETERMINANT {
            return Err("cannot compute inverse due to zero determinant");
        }
        ai[0][0] = 1.0 / det;
        return Ok(det);
    }

    if m == 2 {
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        if f64::abs(det) <= ZERO_DETERMINANT {
            return Err("cannot compute inverse due to zero determinant");
        }
        ai[0][0] = a[1][1] / det;
        ai[0][1] = -a[0][1] / det;
        ai[1][0] = -a[1][0] / det;
        ai[1][1] = a[0][0] / det;
        return Ok(det);
    }

    if m == 3 {
        #[rustfmt::skip]
        let det =
              a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        if f64::abs(det) <= ZERO_DETERMINANT {
            return Err("cannot compute inverse due to zero determinant");
        }

        ai[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
        ai[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
        ai[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;

        ai[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
        ai[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
        ai[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;

        ai[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
        ai[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
        ai[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;

        return Ok(det);
    }

    // copy a into ai
    let m_i32 = to_i32(m);
    dcopy(m_i32 * m_i32, a.as_data(), 1, ai.as_mut_data(), 1);

    // handle large matrix
    let mut ipiv = vec![0_i32; m];
    dgetrf(m_i32, m_i32, ai.as_mut_data(), &mut ipiv)?;

    // first, compute the determinant ai.data from dgetrf
    let mut det = 1.0;
    for i in 0..m_i32 {
        let iu = i as usize;
        // NOTE: ipiv are 1-based indices
        if ipiv[iu] - 1 == i {
            det = det * ai[iu][iu];
        } else {
            det = -det * ai[iu][iu];
        }
    }
    // second, perform the inversion
    dgetri(m_i32, ai.as_mut_data(), &ipiv)?;

    // done
    Ok(det)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    /// Computes a⋅ai that should equal I for a square matrix
    fn get_a_times_ai(a: &Matrix, ai: &Matrix) -> Matrix {
        let (m, n) = a.dims();
        let mut a_ai = Matrix::new(m, m);
        for i in 0..m {
            for j in 0..m {
                for k in 0..n {
                    a_ai[i][j] += a[i][k] * ai[k][j];
                }
            }
        }
        a_ai
    }

    #[test]
    fn inverse_fails_on_wrong_dims() {
        let mut a_2x3 = Matrix::new(2, 3);
        let mut a_2x2 = Matrix::new(2, 2);
        let mut ai_1x2 = Matrix::new(1, 2);
        let mut ai_2x1 = Matrix::new(2, 1);
        assert_eq!(inverse(&mut ai_1x2, &mut a_2x3), Err("matrix must be square"));
        assert_eq!(inverse(&mut ai_1x2, &mut a_2x2), Err("matrices are incompatible"));
        assert_eq!(inverse(&mut ai_2x1, &mut a_2x2), Err("matrices are incompatible"));
    }

    #[test]
    fn inverse_0x0_works() -> Result<(), &'static str> {
        let mut a = Matrix::new(0, 0);
        let mut ai = Matrix::new(0, 0);
        let det = inverse(&mut ai, &mut a)?;
        assert_eq!(det, 0.0);
        assert_eq!(ai.as_data().len(), 0);
        Ok(())
    }

    #[test]
    fn inverse_1x1_works() -> Result<(), &'static str> {
        let data = [[2.0]];
        let mut a = Matrix::from(&data);
        let mut ai = Matrix::new(1, 1);
        let det = inverse(&mut ai, &mut a)?;
        assert_eq!(det, 2.0);
        assert_vec_approx_eq!(ai.as_data(), &[0.5], 1e-15);
        let a_copy = Matrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        assert_vec_approx_eq!(a_ai.as_data(), &[1.0], 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_1x1_fails_on_zero_det() -> Result<(), &'static str> {
        let mut a = Matrix::from(&[[ZERO_DETERMINANT / 10.0]]);
        let mut ai = Matrix::new(1, 1);
        let res = inverse(&mut ai, &mut a);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
        Ok(())
    }

    #[test]
    fn inverse_2x2_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let data = [
            [1.0, 2.0],
            [3.0, 2.0],
        ];
        let mut a = Matrix::from(&data);
        let mut ai = Matrix::new(2, 2);
        let det = inverse(&mut ai, &mut a)?;
        assert_eq!(det, -4.0);
        assert_vec_approx_eq!(ai.as_data(), &[-0.5, 0.5, 0.75, -0.25], 1e-15);
        let a_copy = Matrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        assert_vec_approx_eq!(a_ai.as_data(), &[1.0, 0.0, 0.0, 1.0], 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_2x2_fails_on_zero_det() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            [   -1.0, 3.0/2.0],
            [2.0/3.0,    -1.0],
        ]);
        let mut ai = Matrix::new(2, 2);
        let res = inverse(&mut ai, &mut a);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
        Ok(())
    }

    #[test]
    fn inverse_3x3_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let data = [
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [1.0, 0.0, 6.0],
        ];
        let mut a = Matrix::from(&data);
        let mut ai = Matrix::new(3, 3);
        let det = inverse(&mut ai, &mut a)?;
        assert_eq!(det, 22.0);
        #[rustfmt::skip]
        let ai_correct = [
            12.0/11.0, -6.0/11.0, -1.0/11.0,
             2.5/11.0,  1.5/11.0, -2.5/11.0,
            -2.0/11.0,  1.0/11.0,  2.0/11.0,
        ];
        assert_vec_approx_eq!(ai.as_data(), ai_correct, 1e-15);
        let identity = Matrix::identity(3);
        let a_copy = Matrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        assert_vec_approx_eq!(a_ai.as_data(), identity.as_data(), 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_3x3_fails_on_zero_det() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 6.0],
        ]);
        let mut ai = Matrix::new(3, 3);
        let res = inverse(&mut ai, &mut a);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
        Ok(())
    }

    #[test]
    fn inverse_4x4_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let data = [
            [ 3.0,  0.0,  2.0, -1.0],
            [ 1.0,  2.0,  0.0, -2.0],
            [ 4.0,  0.0,  6.0, -3.0],
            [ 5.0,  0.0,  2.0,  0.0],
        ];
        let mut a = Matrix::from(&data);
        let mut ai = Matrix::new(4, 4);
        let det = inverse(&mut ai, &mut a)?;
        assert_approx_eq!(det, 20.0, 1e-14);
        #[rustfmt::skip]
        let ai_correct = [
             0.6,  0.0, -0.2,  0.0,
            -2.5,  0.5,  0.5,  1.0,
            -1.5,  0.0,  0.5,  0.5,
            -2.2,  0.0,  0.4,  1.0,
        ];
        assert_vec_approx_eq!(ai.as_data(), ai_correct, 1e-15);
        let identity = Matrix::identity(4);
        let a_copy = Matrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        assert_vec_approx_eq!(a_ai.as_data(), identity.as_data(), 1e-15);
        Ok(())
    }

    #[test]
    fn inverse_5x5_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let data = [
            [12.0, 28.0, 22.0, 20.0,  8.0],
            [ 0.0,  3.0,  5.0, 17.0, 28.0],
            [56.0,  0.0, 23.0,  1.0,  0.0],
            [12.0, 29.0, 27.0, 10.0,  1.0],
            [ 9.0,  4.0, 13.0,  8.0, 22.0],
        ];
        let mut a = Matrix::from(&data);
        let mut ai = Matrix::new(5, 5);
        let det = inverse(&mut ai, &mut a)?;
        assert_approx_eq!(det, -167402.0, 1e-8);
        #[rustfmt::skip]
        let ai_correct = [
             6.9128803717996279e-01, -7.4226114383340802e-01, -9.8756287260606410e-02, -6.9062496266472417e-01,  7.2471057693456553e-01,
             1.5936129795342968e+00, -1.7482347881148397e+00, -2.8304321334273236e-01, -1.5600769405383470e+00,  1.7164430532490673e+00,
            -1.6345384165063759e+00,  1.7495848317224429e+00,  2.7469205863729274e-01,  1.6325730875377857e+00, -1.7065745928961444e+00,
            -1.1177465024312745e+00,  1.3261729250546601e+00,  2.1243473793622566e-01,  1.1258168958554866e+00, -1.3325766717243535e+00,
             7.9976941733073770e-01, -8.9457712572131853e-01, -1.4770432850264653e-01, -8.0791149448632715e-01,  9.2990525800169743e-01,
        ];
        assert_vec_approx_eq!(ai.as_data(), ai_correct, 1e-13);
        let identity = Matrix::identity(5);
        let a_copy = Matrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        assert_vec_approx_eq!(a_ai.as_data(), identity.as_data(), 1e-13);
        Ok(())
    }

    #[test]
    fn inverse_6x6_works() -> Result<(), &'static str> {
        // NOTE: this matrix is nearly non-invertible; it originated from an FEM analysis
        #[rustfmt::skip]
        let data = [
            [ 3.46540497998689445e-05, -1.39368151175265866e-05, -1.39368151175265866e-05,  0.00000000000000000e+00, 7.15957288480514429e-23, -2.93617909908697186e+02],
            [-1.39368151175265866e-05,  3.46540497998689445e-05, -1.39368151175265866e-05,  0.00000000000000000e+00, 7.15957288480514429e-23, -2.93617909908697186e+02],
            [-1.39368151175265866e-05, -1.39368151175265866e-05,  3.46540497998689445e-05,  0.00000000000000000e+00, 7.15957288480514429e-23, -2.93617909908697186e+02],
            [ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  4.85908649173955311e-05, 0.00000000000000000e+00,  0.00000000000000000e+00],
            [ 3.13760264822604860e-18,  3.13760264822604860e-18,  3.13760264822604860e-18,  0.00000000000000000e+00, 1.00000000000000000e+00, -1.93012141894243434e+07],
            [ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, -0.00000000000000000e+00, 0.00000000000000000e+00,  1.00000000000000000e+00],
        ];
        let mut a = Matrix::from(&data);
        let mut ai = Matrix::new(6, 6);
        let det = inverse(&mut ai, &mut a)?;
        assert_approx_eq!(det, 7.778940633136385e-19, 1e-15);
        #[rustfmt::skip]
        let ai_correct = [
             6.28811662297464645e+04,  4.23011662297464645e+04,  4.23011662297464645e+04, 0.00000000000000000e+00, -1.05591885817167332e-17, 4.33037966311565489e+07,
             4.23011662297464645e+04,  6.28811662297464645e+04,  4.23011662297464645e+04, 0.00000000000000000e+00, -1.05591885817167332e-17, 4.33037966311565489e+07,
             4.23011662297464645e+04,  4.23011662297464645e+04,  6.28811662297464645e+04, 0.00000000000000000e+00, -1.05591885817167348e-17, 4.33037966311565489e+07,
             0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 2.05800000000000000e+04,  0.00000000000000000e+00, 0.00000000000000000e+00,
            -4.62744616057000471e-13, -4.62744616057000471e-13, -4.62744616057000471e-13, 0.00000000000000000e+00,  1.00000000000000000e+00, 1.93012141894243434e+07,
             0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0.00000000000000000e+00,  0.00000000000000000e+00, 1.00000000000000000e+00,
        ];
        assert_vec_approx_eq!(ai.as_data(), ai_correct, 1e-8);
        let identity = Matrix::identity(6);
        let a_copy = Matrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        assert_vec_approx_eq!(a_ai.as_data(), identity.as_data(), 1e-12);
        Ok(())
    }
}
