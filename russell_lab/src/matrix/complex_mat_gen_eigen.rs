use super::ComplexMatrix;
use crate::{cpx, to_i32, CcBool, Complex64, ComplexVector, StrError, C_FALSE, C_TRUE};

extern "C" {
    // Computes the eigenvalues and, optionally, the left and/or right eigenvectors for GE matrices
    // <https://www.netlib.org/lapack/explore-html/d3/d47/zggev_8f.html>
    fn c_zggev(
        calc_vl: CcBool,
        calc_vr: CcBool,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        b: *mut Complex64,
        ldb: *const i32,
        alpha: *mut Complex64,
        beta: *mut Complex64,
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

/// (zggev) Computes the generalized eigenvalues and right eigenvectors
///
/// A generalized eigenvalue for a pair of matrices (A,B) is a scalar lambda
/// or a ratio alpha/beta = lambda, such that A - lambda*B is singular.
/// It is usually represented as the pair (alpha,beta), as there is a
/// reasonable interpretation for beta=0, and even for both being zero.
///
/// Computes the eigenvalues `l` and right eigenvectors `v`, such that:
///
/// ```text
/// a ⋅ vj = lj ⋅ b ⋅ vj
/// ```
///
/// where `lj` is the component j of `l` and `vj` is the column j of `v`.
///
/// The eigenvalues are returned in two parts, α (alpha_real, alpha_imag) and β (beta).
///
/// - If alpha_imag(j) = 0, then the j-th eigenvalue is real.
/// - If alpha_imag(j) > 0, then the j-th and (j+1)-th eigenvalues are a complex conjugate pair.
///
/// See also: <https://www.netlib.org/lapack/explore-html/d3/d47/zggev_8f.html>
///
/// # Output
///
/// * `alpha_real` -- (m) real part to compose the eigenvalues
/// * `alpha_imag` -- (m) imaginary part to compose the eigenvalues
/// * `beta` -- (m) coefficients to compose the eigenvalues
/// * `v` -- (m,m) **right** eigenvectors (as columns)
///
/// # Input
///
/// * `a` -- (m,m) general matrix (will be modified)
/// * `b` -- (m,m) general matrix (will be modified)
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // a and b matrices
///     let mut a = ComplexMatrix::from(&[
///         [cpx!(1.0, 2.0), cpx!(3.0, 4.0), cpx!(21.0, 22.0)],
///         [cpx!(43.0, 44.0), cpx!(13.0, 14.0), cpx!(15.0, 16.0)],
///         [cpx!(5.0, 6.0), cpx!(7.0, 8.0), cpx!(25.0, 26.0)],
///     ]);
///     let mut b = ComplexMatrix::from(&[
///         [cpx!(2.0, 0.0), cpx!(0.0, -1.0), cpx!(0.0, 0.0)],
///         [cpx!(0.0, 1.0), cpx!(2.0, 0.0), cpx!(0.0, 0.0)],
///         [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(3.0, 0.0)],
///     ]);
///     let m = a.nrow();
///     let mut alpha = ComplexVector::new(m);
///     let mut beta = ComplexVector::new(m);
///     let mut v = ComplexMatrix::new(m, m);
///     complex_mat_gen_eigen(&mut alpha, &mut beta, &mut v, &mut a, &mut b).unwrap();
///     println!("α =\n{:.6}", alpha);
///     println!("β =\n{:.6}", beta);
///     println!("v =\n{:.6}", v);
///     // compare with reference
///     // (https://www.ibm.com/docs/en/essl/6.2)
///     let alpha_ref = &[
///         cpx!( 15.863783, 41.115283),
///         cpx!(-12.917205, 19.973815),
///         cpx!(  3.215518, -4.912439),
///     ];
///     let beta_ref = &[
///         cpx!(1.668461, 0.0),
///         cpx!(2.024212, 0.0),
///         cpx!(2.664836, 0.0),
///     ];
///     complex_vec_approx_eq(&alpha, alpha_ref, 1e-6);
///     complex_vec_approx_eq(&beta, beta_ref, 1e-6);
///     Ok(())
/// }
/// ```
pub fn complex_mat_gen_eigen(
    alpha: &mut ComplexVector,
    beta: &mut ComplexVector,
    v: &mut ComplexMatrix,
    a: &mut ComplexMatrix,
    b: &mut ComplexMatrix,
) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if alpha.dim() != m || beta.dim() != m {
        return Err("vectors are incompatible");
    }
    if v.nrow() != m || v.ncol() != m || b.nrow() != m || b.ncol() != m {
        return Err("matrices are incompatible");
    }
    let m_i32 = to_i32(m);
    let lda = m_i32;
    let ldb = m_i32;
    let ldu = 1;
    let ldv = m_i32;
    const EXTRA: i32 = 1;
    let lwork = 2 * m_i32 + EXTRA;
    let mut u = vec![cpx!(0.0, 0.0); ldu as usize];
    let mut work = vec![cpx!(0.0, 0.0); lwork as usize];
    let mut rwork = vec![0.0; 8 * m];
    let mut info = 0;
    unsafe {
        c_zggev(
            C_FALSE,
            C_TRUE,
            &m_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            b.as_mut_data().as_mut_ptr(),
            &ldb,
            alpha.as_mut_data().as_mut_ptr(),
            beta.as_mut_data().as_mut_ptr(),
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
        println!("LAPACK ERROR (zggev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (zggev): An argument had an illegal value");
    } else if info > 0 {
        return Err("LAPACK ERROR (zggev): Iterations failed ");
    }
    Ok(())
}

/// Computes the generalized eigenvalues and eigenvectors (left and right)
///
/// A generalized eigenvalue for a pair of matrices (A,B) is a scalar lambda
/// or a ratio alpha/beta = lambda, such that A - lambda*B is singular.
/// It is usually represented as the pair (alpha,beta), as there is a
/// reasonable interpretation for beta=0, and even for both being zero.
/// Performs the eigen-decomposition of a square matrix (left and right)
///
/// Computes the eigenvalues `l` and left eigenvectors `u`, such that:
///
/// ```text
/// ujᴴ ⋅ a = lj ⋅ ujᴴ ⋅ b
/// ```
///
/// where `lj` is the component j of `l` and `ujᴴ` is the column j of `uᴴ`,
/// with `uᴴ` being the conjugate-transpose of `u`.
///
/// Also, computes the right eigenvectors `v`, such that:
///
/// ```text
/// a ⋅ vj = lj ⋅ b ⋅ vj
/// ```
///
/// where `vj` is the column j of `v`.
///
/// The eigenvalues are returned in two parts, α (alpha_real, alpha_imag) and β (beta).
///
/// - If alpha_imag(j) = 0, then the j-th eigenvalue is real.
/// - If alpha_imag(j) > 0, then the j-th and (j+1)-th eigenvalues are a complex conjugate pair.
///
/// See also: <https://www.netlib.org/lapack/explore-html/d3/d47/zggev_8f.html>
///
/// # Output
///
/// * `alpha_real` -- (m) real part to compose the eigenvalues
/// * `alpha_imag` -- (m) imaginary part to compose the eigenvalues
/// * `beta` -- (m) coefficients to compose the eigenvalues
/// * `u` -- (m,m) **left** eigenvectors (as columns)
/// * `v` -- (m,m) **right** eigenvectors (as columns)
///
/// # Input
///
/// * `a` -- (m,m) general matrix (will be modified)
/// * `b` -- (m,m) general matrix (will be modified)
///
/// # Note
///
/// * The matrix `a` will be modified
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // a and b matrices
///     let mut a = ComplexMatrix::from(&[
///         [cpx!(1.0, 2.0), cpx!(3.0, 4.0), cpx!(21.0, 22.0)],
///         [cpx!(43.0, 44.0), cpx!(13.0, 14.0), cpx!(15.0, 16.0)],
///         [cpx!(5.0, 6.0), cpx!(7.0, 8.0), cpx!(25.0, 26.0)],
///     ]);
///     let mut b = ComplexMatrix::from(&[
///         [cpx!(2.0, 0.0), cpx!(0.0, -1.0), cpx!(0.0, 0.0)],
///         [cpx!(0.0, 1.0), cpx!(2.0, 0.0), cpx!(0.0, 0.0)],
///         [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(3.0, 0.0)],
///     ]);
///     let m = a.nrow();
///     let mut alpha = ComplexVector::new(m);
///     let mut beta = ComplexVector::new(m);
///     let mut u = ComplexMatrix::new(m, m);
///     let mut v = ComplexMatrix::new(m, m);
///     complex_mat_gen_eigen_lr(&mut alpha, &mut beta, &mut u, &mut v, &mut a, &mut b).unwrap();
///     println!("α =\n{:.6}", alpha);
///     println!("β =\n{:.6}", beta);
///     println!("u =\n{:.6}", u);
///     println!("v =\n{:.6}", v);
///     // compare with reference
///     // (https://www.ibm.com/docs/en/essl/6.2)
///     #[rustfmt::skip]
///     let alpha_ref = &[
///         cpx!( 15.863783, 41.115283),
///         cpx!(-12.917205, 19.973815),
///         cpx!(  3.215518, -4.912439),
///     ];
///     #[rustfmt::skip]
///     let beta_ref = &[
///         cpx!(1.668461, 0.0),
///         cpx!(2.024212, 0.0),
///         cpx!(2.664836, 0.0),
///     ];
///     complex_vec_approx_eq(&alpha, alpha_ref, 1e-6);
///     complex_vec_approx_eq(&beta, beta_ref, 1e-6);
///     Ok(())
/// }
/// ```
pub fn complex_mat_gen_eigen_lr(
    alpha: &mut ComplexVector,
    beta: &mut ComplexVector,
    u: &mut ComplexMatrix,
    v: &mut ComplexMatrix,
    a: &mut ComplexMatrix,
    b: &mut ComplexMatrix,
) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if alpha.dim() != m || beta.dim() != m {
        return Err("vectors are incompatible");
    }
    if u.nrow() != m || u.ncol() != m || v.nrow() != m || v.ncol() != m || b.nrow() != m || b.ncol() != m {
        return Err("matrices are incompatible");
    }
    let m_i32 = to_i32(m);
    let lda = m_i32;
    let ldb = m_i32;
    let ldu = m_i32;
    let ldv = m_i32;
    const EXTRA: i32 = 1;
    let lwork = 2 * m_i32 + EXTRA;
    let mut work = vec![cpx!(0.0, 0.0); lwork as usize];
    let mut rwork = vec![0.0; 8 * m];
    let mut info = 0;
    unsafe {
        c_zggev(
            C_FALSE,
            C_TRUE,
            &m_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            b.as_mut_data().as_mut_ptr(),
            &ldb,
            alpha.as_mut_data().as_mut_ptr(),
            beta.as_mut_data().as_mut_ptr(),
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
        println!("LAPACK ERROR (zggev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (zggev): An argument had an illegal value");
    } else if info > 0 {
        return Err("LAPACK ERROR (zggev): Iterations failed ");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_gen_eigen, complex_mat_gen_eigen_lr};
    use crate::matrix::testing::complex_check_gen_eigen;
    use crate::{complex_vec_approx_eq, cpx, Complex64, ComplexMatrix, ComplexVector};

    #[test]
    fn complex_mat_gen_eigen_captures_errors() {
        let mut a = ComplexMatrix::new(2, 1);
        let m = a.nrow();
        let mut b = ComplexMatrix::new(m, m);
        let mut alpha = ComplexVector::new(m);
        let mut beta = ComplexVector::new(m);
        let mut v = ComplexMatrix::new(m, m);
        assert_eq!(
            complex_mat_gen_eigen(&mut alpha, &mut beta, &mut v, &mut a, &mut b),
            Err("matrix must be square")
        );
        let mut a = ComplexMatrix::new(2, 2);
        let mut b_wrong1 = ComplexMatrix::new(m + 1, m);
        let mut b_wrong2 = ComplexMatrix::new(m, m + 1);
        let mut alpha_wrong = ComplexVector::new(m + 1);
        let mut beta_wrong = ComplexVector::new(m + 1);
        let mut v_wrong = ComplexMatrix::new(m + 1, m);
        assert_eq!(
            complex_mat_gen_eigen(&mut alpha_wrong, &mut beta, &mut v, &mut a, &mut b),
            Err("vectors are incompatible")
        );
        assert_eq!(
            complex_mat_gen_eigen(&mut alpha, &mut beta_wrong, &mut v, &mut a, &mut b),
            Err("vectors are incompatible")
        );
        assert_eq!(
            complex_mat_gen_eigen(&mut alpha, &mut beta, &mut v_wrong, &mut a, &mut b),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_gen_eigen(&mut alpha, &mut beta, &mut v, &mut a, &mut b_wrong1,),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_gen_eigen(&mut alpha, &mut beta, &mut v, &mut a, &mut b_wrong2,),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn complex_mat_gen_eigen_lr_captures_errors() {
        let mut a = ComplexMatrix::new(2, 1);
        let m = a.nrow();
        let mut b = ComplexMatrix::new(m, m);
        let mut alpha = ComplexVector::new(m);
        let mut beta = ComplexVector::new(m);
        let mut u = ComplexMatrix::new(m, m);
        let mut v = ComplexMatrix::new(m, m);
        assert_eq!(
            complex_mat_gen_eigen_lr(&mut alpha, &mut beta, &mut u, &mut v, &mut a, &mut b),
            Err("matrix must be square")
        );
        let mut a = ComplexMatrix::new(2, 2);
        let mut b_wrong1 = ComplexMatrix::new(m + 1, m);
        let mut b_wrong2 = ComplexMatrix::new(m, m + 1);
        let mut alpha_wrong = ComplexVector::new(m + 1);
        let mut beta_wrong = ComplexVector::new(m + 1);
        let mut u_wrong = ComplexMatrix::new(m + 1, m);
        let mut v_wrong = ComplexMatrix::new(m + 1, m);
        assert_eq!(
            complex_mat_gen_eigen_lr(&mut alpha_wrong, &mut beta, &mut u, &mut v, &mut a, &mut b),
            Err("vectors are incompatible")
        );
        assert_eq!(
            complex_mat_gen_eigen_lr(&mut alpha, &mut beta_wrong, &mut u, &mut v, &mut a, &mut b),
            Err("vectors are incompatible")
        );
        assert_eq!(
            complex_mat_gen_eigen_lr(&mut alpha, &mut beta, &mut u_wrong, &mut v, &mut a, &mut b),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_gen_eigen_lr(&mut alpha, &mut beta, &mut u, &mut v_wrong, &mut a, &mut b),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_gen_eigen_lr(&mut alpha, &mut beta, &mut u, &mut v, &mut a, &mut b_wrong1,),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_gen_eigen_lr(&mut alpha, &mut beta, &mut u, &mut v, &mut a, &mut b_wrong2,),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn complex_mat_gen_eigen_works() {
        #[rustfmt::skip]
        let a_data = &[
            [cpx!(2.0, 4.0), cpx!(1.0, 6.0), cpx!(2.0, 8.0), cpx!(4.0, 4.0)],
            [cpx!(3.0, 3.0), cpx!(6.0, 1.0), cpx!(5.0, 3.0), cpx!(0.0, 0.0)],
            [cpx!(5.0, 1.0), cpx!(8.0, 5.0), cpx!(3.0, 2.0), cpx!(8.0, 5.0)],
            [cpx!(7.0, 6.0), cpx!(3.0, 7.0), cpx!(2.0, 1.0), cpx!(5.0, 4.0)],
        ];
        #[rustfmt::skip]
        let b_data = &[
            [cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(1.0, 0.0)],
            [cpx!(1.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
        ];
        let mut a = ComplexMatrix::from(a_data);
        let mut b = ComplexMatrix::from(b_data);
        let m = a.nrow();
        let mut alpha = ComplexVector::new(m);
        let mut beta = ComplexVector::new(m);
        let mut v = ComplexMatrix::new(m, m);
        complex_mat_gen_eigen(&mut alpha, &mut beta, &mut v, &mut a, &mut b).unwrap();
        // println!("α =\n{}", alpha);
        // println!("β =\n{}", beta);
        // println!("v =\n{}", v);
        // compare with reference (scipy)
        #[rustfmt::skip]
        let alpha_ref = &[
            cpx!(15.8864, 15.0474),
            cpx!( 7.0401,  2.0585),
            cpx!( 1.7083,  4.1133),
            cpx!(-3.6348, -1.2193),
        ];
        #[rustfmt::skip]
        let beta_ref = &[
            cpx!(1.0, 0.0),
            cpx!(1.0, 0.0),
            cpx!(1.0, 0.0),
            cpx!(1.0, 0.0),
        ];
        complex_vec_approx_eq(&alpha, alpha_ref, 1e-4);
        complex_vec_approx_eq(&beta, beta_ref, 1e-14);
        complex_check_gen_eigen(a_data, b_data, &v, &alpha, &beta, 1e-13);
    }

    #[test]
    fn complex_mat_gen_eigen_lr_works() {
        // https://www.ibm.com/docs/en/essl/6.2?topic=eas-sggev-dggev-cggev-zggev-sggevx-dggevx-cggevx-zggevx-eigenvalues-optionally-right-eigenvectors-left-eigenvectors-reciprocal-condition-numbers-eigenvalues-reciprocal-condition-numbers-right-eigenvectors-general-matrix-generalized-eigenproblem
        #[rustfmt::skip]
        let a_data = &[
            [cpx!( 1.0, 2.0), cpx!( 3.0, 4.0), cpx!(21.0,22.0)],
            [cpx!(43.0,44.0), cpx!(13.0,14.0), cpx!(15.0,16.0)],
            [cpx!( 5.0, 6.0), cpx!( 7.0, 8.0), cpx!(25.0,26.0)],
        ];
        #[rustfmt::skip]
        let b_data = &[
            [cpx!( 2.0, 0.0), cpx!( 0.0,-1.0), cpx!( 0.0, 0.0)],
            [cpx!( 0.0, 1.0), cpx!( 2.0, 0.0), cpx!( 0.0, 0.0)],
            [cpx!( 0.0, 0.0), cpx!( 0.0, 0.0), cpx!( 3.0, 0.0)],
        ];
        let mut a = ComplexMatrix::from(a_data);
        let mut b = ComplexMatrix::from(b_data);
        let m = a.nrow();
        let mut alpha = ComplexVector::new(m);
        let mut beta = ComplexVector::new(m);
        let mut u = ComplexMatrix::new(m, m);
        let mut v = ComplexMatrix::new(m, m);
        complex_mat_gen_eigen_lr(&mut alpha, &mut beta, &mut u, &mut v, &mut a, &mut b).unwrap();
        println!("α =\n{}", alpha);
        println!("β =\n{}", beta);
        println!("u =\n{}", u);
        println!("v =\n{}", v);
        // compare with reference
        #[rustfmt::skip]
        let alpha_ref = &[
            cpx!( 15.863783, 41.115283),
            cpx!(-12.917205, 19.973815),
            cpx!(  3.215518, -4.912439),
        ];
        #[rustfmt::skip]
        let beta_ref = &[
            cpx!(1.668461, 0.0),
            cpx!(2.024212, 0.0),
            cpx!(2.664836, 0.0),
        ];
        complex_vec_approx_eq(&alpha, alpha_ref, 1e-6);
        complex_vec_approx_eq(&beta, beta_ref, 1e-6);
        // check the definition: a ⋅ vj = lj ⋅ b ⋅ vj
        complex_check_gen_eigen(a_data, b_data, &v, &alpha, &beta, 1e-13);
    }
}
