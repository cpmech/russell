use super::{complex_mat_copy, ComplexMatrix};
use crate::{cpx, to_i32, StrError};
use num_complex::Complex64;

extern "C" {
    // Computes the LU factorization of a general (m,n) matrix
    // <https://www.netlib.org/lapack/explore-html/dd/dd1/zgetrf_8f.html>
    fn c_zgetrf(m: *const i32, n: *const i32, a: *mut Complex64, lda: *const i32, ipiv: *mut i32, info: *mut i32);

    // Computes the inverse of a matrix using the LU factorization computed by zgetrf
    // <https://www.netlib.org/lapack/explore-html/d0/db3/zgetri_8f.html>
    fn c_zgetri(
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        info: *mut i32,
    );

}

// constants
const ZERO_DETERMINANT_NORM: f64 = 1e-15;

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
/// ```
/// use num_complex::Complex64;
/// use russell_lab::{cpx, complex_mat_inverse, ComplexMatrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // set matrix
///     let mut a = ComplexMatrix::from(&[
///         [cpx!(1.0, 0.0), cpx!(2.0, 0.0), cpx!(3.0, 0.0)],
///         [cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(4.0, 0.0)],
///         [cpx!(5.0, 0.0), cpx!(6.0, 0.0), cpx!(0.0, 0.0)],
///     ]);
///     let a_copy = a.clone();
///
///     // compute inverse matrix
///     let mut ai = ComplexMatrix::new(3, 3);
///     complex_mat_inverse(&mut ai, &mut a)?;
///
///     // compare with solution
///     let ai_correct = "┌                      ┐\n\
///                       │ -24+0i  18+0i   5+0i │\n\
///                       │  20+0i -15+0i  -4+0i │\n\
///                       │  -5+0i   4+0i   1+0i │\n\
///                       └                      ┘";
///     assert_eq!(format!("{}", ai), ai_correct);
///
///     // check if a⋅ai == identity
///     let (m, n) = a.dims();
///     let mut a_ai = ComplexMatrix::new(m, m);
///     for i in 0..m {
///         for j in 0..m {
///             for k in 0..n {
///                 a_ai.add(i, j, a_copy.get(i, k) * ai.get(k, j));
///             }
///         }
///     }
///     let identity = "┌                ┐\n\
///                     │ 1+0i 0+0i 0+0i │\n\
///                     │ 0+0i 1+0i 0+0i │\n\
///                     │ 0+0i 0+0i 1+0i │\n\
///                     └                ┘";
///     assert_eq!(format!("{}", a_ai), identity);
///     Ok(())
/// }
/// ```
pub fn complex_mat_inverse(ai: &mut ComplexMatrix, a: &ComplexMatrix) -> Result<Complex64, StrError> {
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
        return Ok(Complex64::new(0.0, 0.0));
    }

    // handle small matrix
    if m == 1 {
        let det = a.get(0, 0);
        if det.norm() <= ZERO_DETERMINANT_NORM {
            return Err("cannot compute inverse due to zero determinant");
        }
        ai.set(0, 0, 1.0 / det);
        return Ok(det);
    }

    if m == 2 {
        let det = a.get(0, 0) * a.get(1, 1) - a.get(0, 1) * a.get(1, 0);
        if det.norm() <= ZERO_DETERMINANT_NORM {
            return Err("cannot compute inverse due to zero determinant");
        }
        ai.set(0, 0, a.get(1, 1) / det);
        ai.set(0, 1, -a.get(0, 1) / det);
        ai.set(1, 0, -a.get(1, 0) / det);
        ai.set(1, 1, a.get(0, 0) / det);
        return Ok(det);
    }

    if m == 3 {
        #[rustfmt::skip]
        let det =
              a.get(0,0) * (a.get(1,1) * a.get(2,2) - a.get(1,2) * a.get(2,1))
            - a.get(0,1) * (a.get(1,0) * a.get(2,2) - a.get(1,2) * a.get(2,0))
            + a.get(0,2) * (a.get(1,0) * a.get(2,1) - a.get(1,1) * a.get(2,0));

        if det.norm() <= ZERO_DETERMINANT_NORM {
            return Err("cannot compute inverse due to zero determinant");
        }

        ai.set(0, 0, (a.get(1, 1) * a.get(2, 2) - a.get(1, 2) * a.get(2, 1)) / det);
        ai.set(0, 1, (a.get(0, 2) * a.get(2, 1) - a.get(0, 1) * a.get(2, 2)) / det);
        ai.set(0, 2, (a.get(0, 1) * a.get(1, 2) - a.get(0, 2) * a.get(1, 1)) / det);

        ai.set(1, 0, (a.get(1, 2) * a.get(2, 0) - a.get(1, 0) * a.get(2, 2)) / det);
        ai.set(1, 1, (a.get(0, 0) * a.get(2, 2) - a.get(0, 2) * a.get(2, 0)) / det);
        ai.set(1, 2, (a.get(0, 2) * a.get(1, 0) - a.get(0, 0) * a.get(1, 2)) / det);

        ai.set(2, 0, (a.get(1, 0) * a.get(2, 1) - a.get(1, 1) * a.get(2, 0)) / det);
        ai.set(2, 1, (a.get(0, 1) * a.get(2, 0) - a.get(0, 0) * a.get(2, 1)) / det);
        ai.set(2, 2, (a.get(0, 0) * a.get(1, 1) - a.get(0, 1) * a.get(1, 0)) / det);

        return Ok(det);
    }

    // copy a into ai
    complex_mat_copy(ai, a).unwrap();

    // compute LU factorization
    let m_i32 = to_i32(m);
    let lda = m_i32;
    let mut ipiv = vec![0; m];
    let mut info = 0;
    unsafe {
        c_zgetrf(
            &m_i32,
            &m_i32,
            ai.as_mut_data().as_mut_ptr(),
            &lda,
            ipiv.as_mut_ptr(),
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (dgetrf): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (dgetrf): An argument had an illegal value");
    } else if info > 0 {
        println!("LAPACK ERROR (dgetrf): U({},{}) is exactly zero", info - 1, info - 1);
        return Err(
            "LAPACK ERROR (dgetrf): The factorization has been completed, but the factor U is exactly singular",
        );
    }

    // first, compute the determinant ai.data from dgetrf
    let mut det = cpx!(1.0, 0.0);
    for i in 0..m_i32 {
        let iu = i as usize;
        // NOTE: ipiv are 1-based indices
        if ipiv[iu] - 1 == i {
            det = det * ai.get(iu, iu);
        } else {
            det = -det * ai.get(iu, iu);
        }
    }
    // second, perform the inversion
    let lwork = m_i32;
    let mut work = vec![cpx!(0.0, 0.0); lwork as usize];
    unsafe {
        c_zgetri(
            &m_i32,
            ai.as_mut_data().as_mut_ptr(),
            &lda,
            ipiv.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }

    // done
    Ok(det)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_inverse, ComplexMatrix, ZERO_DETERMINANT_NORM};
    use crate::{complex_approx_eq, complex_mat_approx_eq, cpx};
    use num_complex::Complex64;

    /// Computes a⋅ai that should equal I for a square matrix
    fn get_a_times_ai(a: &ComplexMatrix, ai: &ComplexMatrix) -> ComplexMatrix {
        let (m, n) = a.dims();
        let mut a_ai = ComplexMatrix::new(m, m);
        for i in 0..m {
            for j in 0..m {
                for k in 0..n {
                    a_ai.add(i, j, a.get(i, k) * ai.get(k, j));
                }
            }
        }
        a_ai
    }

    #[test]
    fn complex_inverse_fails_on_wrong_dims() {
        let mut a_2x3 = ComplexMatrix::new(2, 3);
        let mut a_2x2 = ComplexMatrix::new(2, 2);
        let mut ai_1x2 = ComplexMatrix::new(1, 2);
        let mut ai_2x1 = ComplexMatrix::new(2, 1);
        assert_eq!(
            complex_mat_inverse(&mut ai_1x2, &mut a_2x3),
            Err("matrix must be square")
        );
        assert_eq!(
            complex_mat_inverse(&mut ai_1x2, &mut a_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_inverse(&mut ai_2x1, &mut a_2x2),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn complex_inverse_0x0_works() {
        let mut a = ComplexMatrix::new(0, 0);
        let mut ai = ComplexMatrix::new(0, 0);
        let det = complex_mat_inverse(&mut ai, &mut a).unwrap();
        assert_eq!(det, cpx!(0.0, 0.0));
        assert_eq!(ai.as_data().len(), 0);
    }

    #[test]
    fn complex_inverse_1x1_works() {
        let data = [[2.0]];
        let mut a = ComplexMatrix::from(&data);
        let mut ai = ComplexMatrix::new(1, 1);
        let det = complex_mat_inverse(&mut ai, &mut a).unwrap();
        assert_eq!(det, cpx!(2.0, 0.0));
        complex_mat_approx_eq(&ai, &[[cpx!(0.5, 0.0)]], 1e-15);
        let a_copy = ComplexMatrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        complex_mat_approx_eq(&a_ai, &[[cpx!(1.0, 0.0)]], 1e-15);
    }

    #[test]
    fn complex_inverse_1x1_fails_on_zero_det() {
        let mut a = ComplexMatrix::from(&[[ZERO_DETERMINANT_NORM / 10.0]]);
        let mut ai = ComplexMatrix::new(1, 1);
        let res = complex_mat_inverse(&mut ai, &mut a);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
    }

    #[test]
    fn complex_inverse_2x2_works() {
        #[rustfmt::skip]
        let data = [
            [1.0, 2.0],
            [3.0, 2.0],
        ];
        let mut a = ComplexMatrix::from(&data);
        let mut ai = ComplexMatrix::new(2, 2);
        let det = complex_mat_inverse(&mut ai, &mut a).unwrap();
        assert_eq!(det, cpx!(-4.0, 0.0));
        complex_mat_approx_eq(
            &ai,
            &[[cpx!(-0.5, 0.0), cpx!(0.5, 0.0)], [cpx!(0.75, 0.0), cpx!(-0.25, 0.0)]],
            1e-15,
        );
        let a_copy = ComplexMatrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        complex_mat_approx_eq(
            &a_ai,
            &[[cpx!(1.0, 0.0), cpx!(0.0, 0.0)], [cpx!(0.0, 0.0), cpx!(1.0, 0.0)]],
            1e-15,
        );
    }

    #[test]
    fn complex_inverse_2x2_fails_on_zero_det() {
        #[rustfmt::skip]
        let mut a = ComplexMatrix::from(&[
            [   -1.0, 3.0/2.0],
            [2.0/3.0,    -1.0],
        ]);
        let mut ai = ComplexMatrix::new(2, 2);
        let res = complex_mat_inverse(&mut ai, &mut a);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
    }

    #[test]
    fn complex_inverse_3x3_works() {
        #[rustfmt::skip]
        let data = [
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [1.0, 0.0, 6.0],
        ];
        let mut a = ComplexMatrix::from(&data);
        let mut ai = ComplexMatrix::new(3, 3);
        let det = complex_mat_inverse(&mut ai, &mut a).unwrap();
        assert_eq!(det, cpx!(22.0, 0.0));
        #[rustfmt::skip]
        let ai_correct = &[
            [cpx!(12.0/11.0, 0.0), cpx!(-6.0/11.0, 0.0), cpx!(-1.0/11.0, 0.0)],
            [cpx!( 2.5/11.0, 0.0), cpx!( 1.5/11.0, 0.0), cpx!(-2.5/11.0, 0.0)],
            [cpx!(-2.0/11.0, 0.0), cpx!( 1.0/11.0, 0.0), cpx!( 2.0/11.0, 0.0)],
        ];
        complex_mat_approx_eq(&ai, ai_correct, 1e-15);
        let identity = ComplexMatrix::identity(3);
        let a_copy = ComplexMatrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        complex_mat_approx_eq(&a_ai, &identity, 1e-15);
    }

    #[test]
    fn complex_inverse_3x3_works_imag() {
        #[rustfmt::skip]
        let data = [
            [cpx!( 2.0,  1.0), cpx!(-1.0, -1.0), cpx!( 0.0,  0.0)],
            [cpx!(-1.0, -1.0), cpx!( 2.0,  2.0), cpx!(-1.0,  1.0)],
            [cpx!( 0.0,  0.0), cpx!(-1.0,  1.0), cpx!( 2.0, -1.0)],
        ];
        let mut a = ComplexMatrix::from(&data);
        let mut ai = ComplexMatrix::new(3, 3);
        let det = complex_mat_inverse(&mut ai, &mut a).unwrap();
        assert_eq!(det, cpx!(6.0, 10.0));
        #[rustfmt::skip]
        let ai_correct = &[
            [cpx!(19.0/34.0, -9.0/34.0), cpx!( 7.0/34.0,  -6.0/34.0), cpx!( 3.0/34.0, -5.0/34.0)],
            [cpx!( 7.0/34.0, -6.0/34.0), cpx!(15.0/68.0, -25.0/68.0), cpx!( 2.0/34.0, -9.0/34.0)],
            [cpx!( 3.0/34.0, -5.0/34.0), cpx!( 2.0/34.0,  -9.0/34.0), cpx!(13.0/34.0,  1.0/34.0)],
        ];
        complex_mat_approx_eq(&ai, ai_correct, 1e-15);
        let identity = ComplexMatrix::identity(3);
        let a_copy = ComplexMatrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        complex_mat_approx_eq(&a_ai, &identity, 1e-15);
    }

    #[test]
    fn complex_inverse_3x3_fails_on_zero_det() {
        #[rustfmt::skip]
        let mut a = ComplexMatrix::from(&[
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 6.0],
        ]);
        let mut ai = ComplexMatrix::new(3, 3);
        let res = complex_mat_inverse(&mut ai, &mut a);
        assert_eq!(res, Err("cannot compute inverse due to zero determinant"));
    }

    #[test]
    fn complex_inverse_4x4_works() {
        #[rustfmt::skip]
        let data = [
            [ 3.0,  0.0,  2.0, -1.0],
            [ 1.0,  2.0,  0.0, -2.0],
            [ 4.0,  0.0,  6.0, -3.0],
            [ 5.0,  0.0,  2.0,  0.0],
        ];
        let mut a = ComplexMatrix::from(&data);
        let mut ai = ComplexMatrix::new(4, 4);
        let det = complex_mat_inverse(&mut ai, &mut a).unwrap();
        complex_approx_eq(det, cpx!(20.0, 0.0), 1e-14);
        #[rustfmt::skip]
        let ai_correct = &[
            [cpx!( 0.6, 0.0),  cpx!(0.0, 0.0), cpx!(-0.2, 0.0),  cpx!(0.0, 0.0)],
            [cpx!(-2.5, 0.0),  cpx!(0.5, 0.0), cpx!( 0.5, 0.0),  cpx!(1.0, 0.0)],
            [cpx!(-1.5, 0.0),  cpx!(0.0, 0.0), cpx!( 0.5, 0.0),  cpx!(0.5, 0.0)],
            [cpx!(-2.2, 0.0),  cpx!(0.0, 0.0), cpx!( 0.4, 0.0),  cpx!(1.0, 0.0)],
        ];
        complex_mat_approx_eq(&ai, ai_correct, 1e-15);
        let identity = ComplexMatrix::identity(4);
        let a_copy = ComplexMatrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        complex_mat_approx_eq(&a_ai, &identity, 1e-15);
    }

    #[test]
    fn complex_inverse_4x4_works_imag() {
        #[rustfmt::skip]
        let data = [
		    [cpx!(1.0, 1.0), cpx!(2.0, 0.0), cpx!( 0.0, 0.0), cpx!(1.0, -1.0)],
		    [cpx!(2.0, 1.0), cpx!(3.0, 0.0), cpx!(-1.0, 0.0), cpx!(1.0, -1.0)],
		    [cpx!(1.0, 1.0), cpx!(2.0, 0.0), cpx!( 0.0, 0.0), cpx!(4.0, -1.0)],
		    [cpx!(4.0, 1.0), cpx!(0.0, 0.0), cpx!( 3.0, 0.0), cpx!(1.0, -1.0)],
        ];
        let mut a = ComplexMatrix::from(&data);
        let mut ai = ComplexMatrix::new(4, 4);
        complex_mat_inverse(&mut ai, &mut a).unwrap();
        #[rustfmt::skip]
        let ai_correct = &[
		    [cpx!(-8.442622950819669e-01, -4.644808743169393e-02), cpx!( 5.409836065573769e-01,  4.918032786885240e-02), cpx!( 3.278688524590156e-02, -2.732240437158467e-02), cpx!( 1.803278688524591e-01,  1.639344262295081e-02)],
		    [cpx!( 1.065573770491803e+00,  2.786885245901638e-01), cpx!(-2.459016393442623e-01, -2.950819672131146e-01), cpx!(-1.967213114754096e-01,  1.639344262295082e-01), cpx!(-8.196721311475419e-02, -9.836065573770497e-02)],
		    [cpx!( 1.221311475409836e+00,  2.322404371584698e-01), cpx!(-7.049180327868851e-01, -2.459016393442622e-01), cpx!(-1.639344262295082e-01,  1.366120218579235e-01), cpx!( 9.836065573770481e-02, -8.196721311475411e-02)],
		    [cpx!(-3.333333333333333e-01,  0.000000000000000e+00), cpx!( 0.000000000000000e+00,  0.000000000000000e+00), cpx!( 3.333333333333333e-01,  0.000000000000000e+00), cpx!( 0.000000000000000e+00,  0.000000000000000e+00)],
        ];
        complex_mat_approx_eq(&ai, ai_correct, 1e-15);
        let identity = ComplexMatrix::identity(4);
        let a_copy = ComplexMatrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        complex_mat_approx_eq(&a_ai, &identity, 1e-15);
    }

    #[test]
    fn complex_inverse_5x5_works() {
        #[rustfmt::skip]
        let data = [
            [cpx!(12.0,0.0), cpx!(28.0,0.0), cpx!(22.0,0.0), cpx!(20.0,0.0), cpx!( 8.0,0.0)],
            [cpx!( 0.0,0.0), cpx!( 3.0,0.0), cpx!( 5.0,0.0), cpx!(17.0,0.0), cpx!(28.0,0.0)],
            [cpx!(56.0,0.0), cpx!( 0.0,0.0), cpx!(23.0,0.0), cpx!( 1.0,0.0), cpx!( 0.0,0.0)],
            [cpx!(12.0,0.0), cpx!(29.0,0.0), cpx!(27.0,0.0), cpx!(10.0,0.0), cpx!( 1.0,0.0)],
            [cpx!( 9.0,0.0), cpx!( 4.0,0.0), cpx!(13.0,0.0), cpx!( 8.0,0.0), cpx!(22.0,0.0)],
        ];
        let mut a = ComplexMatrix::from(&data);
        let mut ai = ComplexMatrix::new(5, 5);
        let det = complex_mat_inverse(&mut ai, &mut a).unwrap();
        complex_approx_eq(det, cpx!(-167402.0, 0.0), 1e-8);
        #[rustfmt::skip]
        let ai_correct = &[
            [cpx!( 6.9128803717996279e-01,0.0), cpx!(-7.4226114383340802e-01,0.0), cpx!(-9.8756287260606410e-02,0.0), cpx!(-6.9062496266472417e-01,0.0), cpx!( 7.2471057693456553e-01,0.0)],
            [cpx!( 1.5936129795342968e+00,0.0), cpx!(-1.7482347881148397e+00,0.0), cpx!(-2.8304321334273236e-01,0.0), cpx!(-1.5600769405383470e+00,0.0), cpx!( 1.7164430532490673e+00,0.0)],
            [cpx!(-1.6345384165063759e+00,0.0), cpx!( 1.7495848317224429e+00,0.0), cpx!( 2.7469205863729274e-01,0.0), cpx!( 1.6325730875377857e+00,0.0), cpx!(-1.7065745928961444e+00,0.0)],
            [cpx!(-1.1177465024312745e+00,0.0), cpx!( 1.3261729250546601e+00,0.0), cpx!( 2.1243473793622566e-01,0.0), cpx!( 1.1258168958554866e+00,0.0), cpx!(-1.3325766717243535e+00,0.0)],
            [cpx!( 7.9976941733073770e-01,0.0), cpx!(-8.9457712572131853e-01,0.0), cpx!(-1.4770432850264653e-01,0.0), cpx!(-8.0791149448632715e-01,0.0), cpx!( 9.2990525800169743e-01,0.0)],
        ];
        complex_mat_approx_eq(&ai, ai_correct, 1e-13);
        let identity = ComplexMatrix::identity(5);
        let a_copy = ComplexMatrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        complex_mat_approx_eq(&a_ai, &identity, 1e-13);
    }

    #[test]
    fn complex_inverse_6x6_works() {
        // NOTE: this matrix is nearly non-invertible; it originated from an FEM analysis
        #[rustfmt::skip]
        let data = [
            [cpx!( 3.46540497998689445e-05,0.0), cpx!(-1.39368151175265866e-05,0.0), cpx!(-1.39368151175265866e-05,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!(7.15957288480514429e-23,0.0), cpx!(-2.93617909908697186e+02,0.0)],
            [cpx!(-1.39368151175265866e-05,0.0), cpx!( 3.46540497998689445e-05,0.0), cpx!(-1.39368151175265866e-05,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!(7.15957288480514429e-23,0.0), cpx!(-2.93617909908697186e+02,0.0)],
            [cpx!(-1.39368151175265866e-05,0.0), cpx!(-1.39368151175265866e-05,0.0), cpx!( 3.46540497998689445e-05,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!(7.15957288480514429e-23,0.0), cpx!(-2.93617909908697186e+02,0.0)],
            [cpx!( 0.00000000000000000e+00,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!( 4.85908649173955311e-05,0.0), cpx!(0.00000000000000000e+00,0.0), cpx!( 0.00000000000000000e+00,0.0)],
            [cpx!( 3.13760264822604860e-18,0.0), cpx!( 3.13760264822604860e-18,0.0), cpx!( 3.13760264822604860e-18,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!(1.00000000000000000e+00,0.0), cpx!(-1.93012141894243434e+07,0.0)],
            [cpx!( 0.00000000000000000e+00,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!(-0.00000000000000000e+00,0.0), cpx!(0.00000000000000000e+00,0.0), cpx!( 1.00000000000000000e+00,0.0)],
        ];
        let mut a = ComplexMatrix::from(&data);
        let mut ai = ComplexMatrix::new(6, 6);
        let det = complex_mat_inverse(&mut ai, &mut a).unwrap();
        complex_approx_eq(det, cpx!(7.778940633136385e-19, 0.0), 1e-15);
        #[rustfmt::skip]
        let ai_correct = &[
            [cpx!( 6.28811662297464645e+04,0.0), cpx!( 4.23011662297464645e+04,0.0), cpx!( 4.23011662297464645e+04,0.0), cpx!(0.00000000000000000e+00,0.0), cpx!(-1.05591885817167332e-17,0.0), cpx!(4.33037966311565489e+07,0.0)],
            [cpx!( 4.23011662297464645e+04,0.0), cpx!( 6.28811662297464645e+04,0.0), cpx!( 4.23011662297464645e+04,0.0), cpx!(0.00000000000000000e+00,0.0), cpx!(-1.05591885817167332e-17,0.0), cpx!(4.33037966311565489e+07,0.0)],
            [cpx!( 4.23011662297464645e+04,0.0), cpx!( 4.23011662297464645e+04,0.0), cpx!( 6.28811662297464645e+04,0.0), cpx!(0.00000000000000000e+00,0.0), cpx!(-1.05591885817167348e-17,0.0), cpx!(4.33037966311565489e+07,0.0)],
            [cpx!( 0.00000000000000000e+00,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!(2.05800000000000000e+04,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!(0.00000000000000000e+00,0.0)],
            [cpx!(-4.62744616057000471e-13,0.0), cpx!(-4.62744616057000471e-13,0.0), cpx!(-4.62744616057000471e-13,0.0), cpx!(0.00000000000000000e+00,0.0), cpx!( 1.00000000000000000e+00,0.0), cpx!(1.93012141894243434e+07,0.0)],
            [cpx!( 0.00000000000000000e+00,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!(0.00000000000000000e+00,0.0), cpx!( 0.00000000000000000e+00,0.0), cpx!(1.00000000000000000e+00,0.0)],
        ];
        complex_mat_approx_eq(&ai, ai_correct, 1e-8);
        let identity = ComplexMatrix::identity(6);
        let a_copy = ComplexMatrix::from(&data);
        let a_ai = get_a_times_ai(&a_copy, &ai);
        complex_mat_approx_eq(&a_ai, &identity, 1e-12);
    }
}
