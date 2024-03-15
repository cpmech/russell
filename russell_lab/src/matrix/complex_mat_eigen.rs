use super::ComplexMatrix;
use crate::{to_i32, CcBool, ComplexVector, StrError, C_FALSE, C_TRUE};
use num_complex::Complex64;
use num_traits::Zero;

extern "C" {
    // Computes the eigenvalues and, optionally, the left and/or right eigenvectors for GE matrices
    // <https://www.netlib.org/lapack/explore-html/dd/dba/zgeev_8f.html>
    fn c_zgeev(
        calc_vl: CcBool,
        calc_vr: CcBool,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        w: *mut Complex64,
        vl: *mut Complex64,
        ldvl: *const i32,
        vr: *mut Complex64,
        ldvr: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );
}

/// Performs the eigen-decomposition of a square matrix (complex version)
///
/// Computes the eigenvalues `l` and right eigenvectors `v`, such that:
///
/// ```text
/// a ⋅ vj = lj ⋅ vj
/// ```
///
/// where `lj` is the component j of `l` and `vj` is the column j of `v`.
///
/// # Output
///
/// * `l` -- (m) eigenvalues
/// * `v` -- (m,m) **right** eigenvectors (as columns)
///
/// # Input
///
/// * `a` -- (m,m) general matrix (will be modified)
///
/// # Notes
///
/// * The matrix `a` will be modified
/// * The computed eigenvectors are normalized to have Euclidean norm equal to 1 and largest component real
///
/// # Example
///
/// ```
/// use num_complex::Complex64;
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // set matrix
///     let data = [[2.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 4.0, 9.0]];
///     let mut a = ComplexMatrix::from(&data);
///
///     // allocate output arrays
///     let m = a.nrow();
///     let mut l = ComplexVector::new(m);
///     let mut v = ComplexMatrix::new(m, m);
///
///     // perform the eigen-decomposition
///     complex_mat_eigen(&mut l, &mut v, &mut a)?;
///
///     // check results
///     assert_eq!(
///         format!("{:.1}", l),
///         "┌           ┐\n\
///          │ 11.0+0.0i │\n\
///          │  1.0+0.0i │\n\
///          │  2.0+0.0i │\n\
///          └           ┘"
///     );
///
///     // check eigen-decomposition (similarity transformation) of a
///     // symmetric matrix with real-only eigenvalues and eigenvectors
///     let a_copy = ComplexMatrix::from(&data);
///     let lam = ComplexMatrix::diagonal(l.as_data());
///     let mut a_v = ComplexMatrix::new(m, m);
///     let mut v_l = ComplexMatrix::new(m, m);
///     let mut err = ComplexMatrix::filled(m, m, cpx!(f64::MAX, 0.0));
///     complex_mat_mat_mul(&mut a_v, cpx!(1.0, 0.0), &a_copy, &v)?;
///     complex_mat_mat_mul(&mut v_l, cpx!(1.0, 0.0), &v, &lam)?;
///     complex_mat_add(&mut err, cpx!(1.0, 0.0), &a_v, cpx!(-1.0, 0.0), &v_l)?;
///     approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, 1e-14);
///     Ok(())
/// }
/// ```
pub fn complex_mat_eigen(l: &mut ComplexVector, v: &mut ComplexMatrix, a: &mut ComplexMatrix) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if l.dim() != m {
        return Err("vectors are incompatible");
    }
    if v.nrow() != m || v.ncol() != m {
        return Err("matrices are incompatible");
    }
    let m_i32 = to_i32(m);
    let lda = m_i32;
    let ldu = 1;
    let ldv = m_i32;
    const EXTRA: i32 = 1;
    let lwork = 2 * m_i32 + EXTRA;
    let mut u = vec![Complex64::zero(); ldu as usize];
    let mut work = vec![Complex64::zero(); lwork as usize];
    let mut rwork = vec![0.0; 2 * m];
    let mut info = 0;
    unsafe {
        c_zgeev(
            C_FALSE,
            C_TRUE,
            &m_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            l.as_mut_data().as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            v.as_mut_data().as_mut_ptr(),
            &ldv,
            work.as_mut_ptr(),
            &lwork,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (zgeev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (zgeev): An argument had an illegal value");
    } else if info > 0 {
        println!("LAPACK ERROR (zgeev): The QR algorithm failed. Elements {}+1:N of l contain eigenvalues which have converged", info-1);
        return Err("LAPACK ERROR (zgeev): The QR algorithm failed to compute all the eigenvalues, and no eigenvectors have been computed");
    }
    Ok(())
}

/// Performs the eigen-decomposition of a square matrix (left and right) (complex version)
///
/// Computes the eigenvalues `l` and left eigenvectors `u`, such that:
///
/// ```text
/// ujᴴ ⋅ a = lj ⋅ ujᴴ
/// ```
///
/// where `lj` is the component j of `l` and `ujᴴ` is the column j of `uᴴ`,
/// with `uᴴ` being the conjugate-transpose of `u`.
///
/// Also, computes the right eigenvectors `v`, such that:
///
/// ```text
/// a ⋅ vj = lj ⋅ vj
/// ```
///
/// where `vj` is the column j of `v`.
///
/// # Output
///
/// * `l` -- (m) eigenvalues
/// * `u` -- (m,m) **left** eigenvectors (as columns)
/// * `v` -- (m,m) **right** eigenvectors (as columns)
///
/// # Input
///
/// * `a` -- (m,m) general matrix [will be modified]
///
/// # Notes
///
/// * The matrix `a` will be modified
/// * The computed eigenvectors are normalized to have Euclidean norm equal to 1 and largest component real
///
/// # Example
///
/// ```
/// use num_complex::Complex64;
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // set matrix
///     let data = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
///     let mut a = ComplexMatrix::from(&data);
///
///     // allocate output arrays
///     let m = a.nrow();
///     let mut l = ComplexVector::new(m);
///     let mut u = ComplexMatrix::new(m, m);
///     let mut v = ComplexMatrix::new(m, m);
///
///     // perform the eigen-decomposition
///     complex_mat_eigen_lr(&mut l, &mut u, &mut v, &mut a)?;
///
///     // check results
///     assert_eq!(
///         format!("{:.3}", l),
///         "┌               ┐\n\
///          │ -0.500+0.866i │\n\
///          │ -0.500-0.866i │\n\
///          │  1.000+0.000i │\n\
///          └               ┘"
///     );
///
///     // check the eigen-decomposition (similarity transformation)
///     // ```text
///     // a⋅v = v⋅λ
///     // err := a⋅v - v⋅λ
///     // ```
///     let a = ComplexMatrix::from(&data);
///     let lam = ComplexMatrix::diagonal(l.as_data());
///     let mut a_v = ComplexMatrix::new(m, m);
///     let mut v_l = ComplexMatrix::new(m, m);
///     let mut err = ComplexMatrix::filled(m, m, cpx!(f64::MAX, f64::MAX));
///     let one = cpx!(1.0, 0.0);
///     let m_one = cpx!(-1.0, 0.0);
///     complex_mat_mat_mul(&mut a_v, one, &a, &v)?;
///     complex_mat_mat_mul(&mut v_l, one, &v, &lam)?;
///     complex_mat_add(&mut err, one, &a_v, m_one, &v_l)?;
///     approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, 1e-15);
///     Ok(())
/// }
/// ```
pub fn complex_mat_eigen_lr(
    l: &mut ComplexVector,
    u: &mut ComplexMatrix,
    v: &mut ComplexMatrix,
    a: &mut ComplexMatrix,
) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if l.dim() != m {
        return Err("vectors are incompatible");
    }
    if u.nrow() != m || u.ncol() != m || v.nrow() != m || v.ncol() != m {
        return Err("matrices are incompatible");
    }
    let m_i32 = to_i32(m);
    let lda = m_i32;
    let ldu = m_i32;
    let ldv = m_i32;
    const EXTRA: i32 = 1;
    let lwork = 2 * m_i32 + EXTRA;
    let mut work = vec![Complex64::zero(); lwork as usize];
    let mut rwork = vec![0.0; 2 * m];
    let mut info = 0;
    unsafe {
        c_zgeev(
            C_TRUE,
            C_TRUE,
            &m_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            l.as_mut_data().as_mut_ptr(),
            u.as_mut_data().as_mut_ptr(),
            &ldu,
            v.as_mut_data().as_mut_ptr(),
            &ldv,
            work.as_mut_ptr(),
            &lwork,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (zgeev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (zgeev): An argument had an illegal value");
    } else if info > 0 {
        println!("LAPACK ERROR (zgeev): The QR algorithm failed. Elements {}+1:N of l contain eigenvalues which have converged", info-1);
        return Err("LAPACK ERROR (zgeev): The QR algorithm failed to compute all the eigenvalues, and no eigenvectors have been computed");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_eigen, complex_mat_eigen_lr};
    use crate::matrix::testing::complex_check_eigen;
    use crate::{complex_vec_approx_eq, cpx, ComplexMatrix, ComplexVector};
    use num_complex::Complex64;

    #[test]
    fn complex_mat_eigen_fails_on_non_square() {
        let mut a = ComplexMatrix::new(3, 4);
        let m = a.nrow();
        let mut l = ComplexVector::new(m);
        let mut v = ComplexMatrix::new(m, m);
        assert_eq!(complex_mat_eigen(&mut l, &mut v, &mut a), Err("matrix must be square"));
    }

    #[test]
    fn complex_mat_eigen_fails_on_wrong_dims() {
        let mut a = ComplexMatrix::new(2, 2);
        let m = a.nrow();
        let mut l = ComplexVector::new(m);
        let mut v = ComplexMatrix::new(m, m);
        let mut l_wrong = ComplexVector::new(m + 1);
        let mut v_wrong = ComplexMatrix::new(m + 1, m);
        assert_eq!(
            complex_mat_eigen(&mut l_wrong, &mut v, &mut a),
            Err("vectors are incompatible")
        );
        assert_eq!(
            complex_mat_eigen(&mut l, &mut v_wrong, &mut a),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn complex_mat_eigen_lr_fails_on_non_square() {
        let mut a = ComplexMatrix::new(3, 4);
        let m = a.nrow();
        let mut l = ComplexVector::new(m);
        let mut u = ComplexMatrix::new(m, m);
        let mut v = ComplexMatrix::new(m, m);
        assert_eq!(
            complex_mat_eigen_lr(&mut l, &mut u, &mut v, &mut a,),
            Err("matrix must be square"),
        );
    }

    #[test]
    fn complex_mat_eigen_lr_fails_on_wrong_dims() {
        let mut a = ComplexMatrix::new(2, 2);
        let m = a.nrow();
        let mut l = ComplexVector::new(m);
        let mut u = ComplexMatrix::new(m, m);
        let mut v = ComplexMatrix::new(m, m);
        let mut l_wrong = ComplexVector::new(m + 1);
        let mut u_wrong = ComplexMatrix::new(m + 1, m);
        let mut v_wrong = ComplexMatrix::new(m + 1, m);
        assert_eq!(
            complex_mat_eigen_lr(&mut l_wrong, &mut u, &mut v, &mut a),
            Err("vectors are incompatible"),
        );
        assert_eq!(
            complex_mat_eigen_lr(&mut l, &mut u_wrong, &mut v, &mut a,),
            Err("matrices are incompatible"),
        );
        assert_eq!(
            complex_mat_eigen_lr(&mut l, &mut u, &mut v_wrong, &mut a,),
            Err("matrices are incompatible"),
        );
    }

    #[test]
    fn complex_mat_eigen_works() {
        #[rustfmt::skip]
        let data = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let mut a = ComplexMatrix::from(&data);
        let m = a.nrow();
        let mut l = ComplexVector::new(m);
        let mut v = ComplexMatrix::new(m, m);
        complex_mat_eigen(&mut l, &mut v, &mut a).unwrap();
        let s3 = f64::sqrt(3.0);
        let l_correct = &[cpx!(-0.5, s3 / 2.0), cpx!(-0.5, -s3 / 2.0), cpx!(1.0, 0.0)];
        complex_vec_approx_eq(l.as_data(), l_correct, 1e-15);
        complex_check_eigen(&data, &v, &l, 1e-15);
    }

    #[test]
    fn complex_mat_eigen_repeated_eval_works() {
        // rep: repeated eigenvalues
        #[rustfmt::skip]
        let data = [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],
        ];
        let mut a = ComplexMatrix::from(&data);
        let m = a.nrow();
        let mut l = ComplexVector::new(m);
        let mut v = ComplexMatrix::new(m, m);
        complex_mat_eigen(&mut l, &mut v, &mut a).unwrap();
        let l_correct = &[cpx!(3.0, 0.0), cpx!(3.0, 0.0), cpx!(2.0, 0.0), cpx!(2.0, 0.0)];
        complex_vec_approx_eq(l.as_data(), l_correct, 1e-15);
        complex_check_eigen(&data, &v, &l, 1e-15);
    }

    #[test]
    fn complex_mat_eigen_lr_works() {
        #[rustfmt::skip]
        let data = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let mut a = ComplexMatrix::from(&data);
        let m = a.nrow();
        let mut l = ComplexVector::new(m);
        let mut u = ComplexMatrix::new(m, m);
        let mut v = ComplexMatrix::new(m, m);
        complex_mat_eigen_lr(&mut l, &mut u, &mut v, &mut a).unwrap();
        let s3 = f64::sqrt(3.0);
        let l_correct = &[cpx!(-0.5, s3 / 2.0), cpx!(-0.5, -s3 / 2.0), cpx!(1.0, 0.0)];
        complex_vec_approx_eq(l.as_data(), l_correct, 1e-15);
        complex_check_eigen(&data, &v, &l, 1e-15);
    }
}
