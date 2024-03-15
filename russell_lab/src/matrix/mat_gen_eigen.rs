use super::Matrix;
use crate::{to_i32, CcBool, StrError, Vector, C_FALSE, C_TRUE};

extern "C" {
    fn c_dggev(
        calc_vl: CcBool,
        calc_vr: CcBool,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        alphar: *mut f64,
        alphai: *mut f64,
        beta: *mut f64,
        vl: *mut f64,
        ldvl: *const i32,
        vr: *mut f64,
        ldvr: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

/// Computes the generalized eigenvalues and right eigenvectors
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
/// # Note
///
/// * The matrix `a` will be modified
///
/// # Example
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // a and b matrices
///     #[rustfmt::skip]
///     let mut a = Matrix::from(&[
///         [2.0, 1.0,  0.0],
///         [3.0, 2.0,  0.0],
///         [0.0, 0.0, -1.0],
///     ]);
///     #[rustfmt::skip]
///     let mut b = Matrix::from(&[
///         [ 3.0, 0.0, 1.0],
///         [-1.0, 2.0, 0.0],
///         [ 1.0, 0.0, 2.0],
///     ]);
///
///     // perform the eigen-decomposition
///     let m = a.nrow();
///     let mut alpha_real = Vector::new(m);
///     let mut alpha_imag = Vector::new(m);
///     let mut beta = Vector::new(m);
///     let mut v = Matrix::new(m, m);
///     mat_gen_eigen(&mut alpha_real, &mut alpha_imag, &mut beta, &mut v, &mut a, &mut b)?;
///
///     // print the results
///     println!("Re(α) =\n{}", alpha_real);
///     println!("Im(α) =\n{}", alpha_imag);
///     println!("β =\n{}", beta);
///     println!("v =\n{}", v);
///
///     // compare with reference (scipy)
///     let alpha_real_ref = &[3.2763146931828615, 0.3047159998456623, -1.0016572343394645];
///     let alpha_imag_ref = &[0.0, 0.0, 0.0];
///     let beta_ref = &[1.7653454701233677, 3.1289910426065792, 1.810364282360633];
///     vec_approx_eq(alpha_real.as_data(), alpha_real_ref, 1e-14);
///     vec_approx_eq(alpha_imag.as_data(), alpha_imag_ref, 1e-14);
///     vec_approx_eq(beta.as_data(), beta_ref, 1e-14);
///     Ok(())
/// }
/// ```
pub fn mat_gen_eigen(
    alpha_real: &mut Vector,
    alpha_imag: &mut Vector,
    beta: &mut Vector,
    v: &mut Matrix,
    a: &mut Matrix,
    b: &mut Matrix,
) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if alpha_real.dim() != m || alpha_imag.dim() != m || beta.dim() != m {
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
    let lwork = 8 * m_i32 + EXTRA;
    let mut u = vec![0.0; ldu as usize];
    let mut work = vec![0.0; lwork as usize];
    let mut info = 0;
    unsafe {
        c_dggev(
            C_FALSE,
            C_TRUE,
            &m_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            b.as_mut_data().as_mut_ptr(),
            &ldb,
            alpha_real.as_mut_data().as_mut_ptr(),
            alpha_imag.as_mut_data().as_mut_ptr(),
            beta.as_mut_data().as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            v.as_mut_data().as_mut_ptr(),
            &ldv,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (dggev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (dggev): An argument had an illegal value");
    } else if info > 0 {
        return Err("LAPACK ERROR (dggev): Iterations failed ");
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
/// # Example
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // a and b matrices
///     #[rustfmt::skip]
///     let mut a = Matrix::from(&[
///         [0.0, 1.0, 1.0],
///         [1.0, 2.0, 0.0],
///         [0.0, 0.0, 1.0],
///     ]);
///     #[rustfmt::skip]
///     let mut b = Matrix::from(&[
///         [1.0, 0.0, 1.0],
///         [0.0, 2.0, 0.0],
///         [0.0, 0.0, 3.0],
///     ]);
///
///     // perform the eigen-decomposition
///     let m = a.nrow();
///     let mut alpha_real = Vector::new(m);
///     let mut alpha_imag = Vector::new(m);
///     let mut beta = Vector::new(m);
///     let mut u = Matrix::new(m, m);
///     let mut v = Matrix::new(m, m);
///     mat_gen_eigen_lr(
///         &mut alpha_real,
///         &mut alpha_imag,
///         &mut beta,
///         &mut u,
///         &mut v,
///         &mut a,
///         &mut b,
///     )?;
///
///     // print the results
///     println!("Re(α) =\n{}", alpha_real);
///     println!("Im(α) =\n{}", alpha_imag);
///     println!("β =\n{}", beta);
///     println!("u =\n{}", u);
///     println!("v =\n{}", v);
///
///     // compare with reference (scipy)
///     let alpha_real_ref = &[-0.42598156773255, 2.347519413393596, 1.0];
///     let alpha_imag_ref = &[0.0, 0.0, 0.0];
///     let beta_ref = &[1.1638032861331697, 1.718503482358399, 3.0];
///     vec_approx_eq(alpha_real.as_data(), alpha_real_ref, 1e-14);
///     vec_approx_eq(alpha_imag.as_data(), alpha_imag_ref, 1e-14);
///     vec_approx_eq(beta.as_data(), beta_ref, 1e-14);
///     Ok(())
/// }
/// ```
pub fn mat_gen_eigen_lr(
    alpha_real: &mut Vector,
    alpha_imag: &mut Vector,
    beta: &mut Vector,
    u: &mut Matrix,
    v: &mut Matrix,
    a: &mut Matrix,
    b: &mut Matrix,
) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if alpha_real.dim() != m || alpha_imag.dim() != m || beta.dim() != m {
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
    let lwork = 8 * m_i32 + EXTRA;
    let mut work = vec![0.0; lwork as usize];
    let mut info = 0;
    unsafe {
        c_dggev(
            C_TRUE,
            C_TRUE,
            &m_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            b.as_mut_data().as_mut_ptr(),
            &ldb,
            alpha_real.as_mut_data().as_mut_ptr(),
            alpha_imag.as_mut_data().as_mut_ptr(),
            beta.as_mut_data().as_mut_ptr(),
            u.as_mut_data().as_mut_ptr(),
            &ldu,
            v.as_mut_data().as_mut_ptr(),
            &ldv,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (dggev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (dggev): An argument had an illegal value");
    } else if info > 0 {
        return Err("LAPACK ERROR (dggev): Iterations failed ");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_gen_eigen, mat_gen_eigen_lr};
    use crate::matrix::mat_approx_eq;
    use crate::matrix::testing::check_gen_eigen;
    use crate::{vec_approx_eq, Matrix, Vector};

    #[test]
    fn mat_gen_eigen_captures_errors() {
        let mut a = Matrix::new(2, 1);
        let m = a.nrow();
        let mut b = Matrix::new(m, m);
        let mut alpha_real = Vector::new(m);
        let mut alpha_imag = Vector::new(m);
        let mut beta = Vector::new(m);
        let mut v = Matrix::new(m, m);
        assert_eq!(
            mat_gen_eigen(&mut alpha_real, &mut alpha_imag, &mut beta, &mut v, &mut a, &mut b),
            Err("matrix must be square")
        );
        let mut a = Matrix::new(2, 2);
        let mut b_wrong1 = Matrix::new(m + 1, m);
        let mut b_wrong2 = Matrix::new(m, m + 1);
        let mut alpha_real_wrong = Vector::new(m + 1);
        let mut alpha_imag_wrong = Vector::new(m + 1);
        let mut beta_wrong = Vector::new(m + 1);
        let mut v_wrong = Matrix::new(m + 1, m);
        assert_eq!(
            mat_gen_eigen(
                &mut alpha_real_wrong,
                &mut alpha_imag,
                &mut beta,
                &mut v,
                &mut a,
                &mut b
            ),
            Err("vectors are incompatible")
        );
        assert_eq!(
            mat_gen_eigen(
                &mut alpha_real,
                &mut alpha_imag_wrong,
                &mut beta,
                &mut v,
                &mut a,
                &mut b
            ),
            Err("vectors are incompatible")
        );
        assert_eq!(
            mat_gen_eigen(
                &mut alpha_real,
                &mut alpha_imag,
                &mut beta_wrong,
                &mut v,
                &mut a,
                &mut b
            ),
            Err("vectors are incompatible")
        );
        assert_eq!(
            mat_gen_eigen(
                &mut alpha_real,
                &mut alpha_imag,
                &mut beta,
                &mut v_wrong,
                &mut a,
                &mut b
            ),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_gen_eigen(
                &mut alpha_real,
                &mut alpha_imag,
                &mut beta,
                &mut v,
                &mut a,
                &mut b_wrong1,
            ),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_gen_eigen(
                &mut alpha_real,
                &mut alpha_imag,
                &mut beta,
                &mut v,
                &mut a,
                &mut b_wrong2,
            ),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn mat_gen_eigen_lr_captures_errors() {
        let mut a = Matrix::new(2, 1);
        let m = a.nrow();
        let mut b = Matrix::new(m, m);
        let mut alpha_real = Vector::new(m);
        let mut alpha_imag = Vector::new(m);
        let mut beta = Vector::new(m);
        let mut u = Matrix::new(m, m);
        let mut v = Matrix::new(m, m);
        assert_eq!(
            mat_gen_eigen_lr(
                &mut alpha_real,
                &mut alpha_imag,
                &mut beta,
                &mut u,
                &mut v,
                &mut a,
                &mut b
            ),
            Err("matrix must be square")
        );
        let mut a = Matrix::new(2, 2);
        let mut b_wrong1 = Matrix::new(m + 1, m);
        let mut b_wrong2 = Matrix::new(m, m + 1);
        let mut alpha_real_wrong = Vector::new(m + 1);
        let mut alpha_imag_wrong = Vector::new(m + 1);
        let mut beta_wrong = Vector::new(m + 1);
        let mut u_wrong = Matrix::new(m + 1, m);
        let mut v_wrong = Matrix::new(m + 1, m);
        assert_eq!(
            mat_gen_eigen_lr(
                &mut alpha_real_wrong,
                &mut alpha_imag,
                &mut beta,
                &mut u,
                &mut v,
                &mut a,
                &mut b
            ),
            Err("vectors are incompatible")
        );
        assert_eq!(
            mat_gen_eigen_lr(
                &mut alpha_real,
                &mut alpha_imag_wrong,
                &mut beta,
                &mut u,
                &mut v,
                &mut a,
                &mut b
            ),
            Err("vectors are incompatible")
        );
        assert_eq!(
            mat_gen_eigen_lr(
                &mut alpha_real,
                &mut alpha_imag,
                &mut beta_wrong,
                &mut u,
                &mut v,
                &mut a,
                &mut b
            ),
            Err("vectors are incompatible")
        );
        assert_eq!(
            mat_gen_eigen_lr(
                &mut alpha_real,
                &mut alpha_imag,
                &mut beta,
                &mut u_wrong,
                &mut v,
                &mut a,
                &mut b
            ),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_gen_eigen_lr(
                &mut alpha_real,
                &mut alpha_imag,
                &mut beta,
                &mut u,
                &mut v_wrong,
                &mut a,
                &mut b
            ),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_gen_eigen_lr(
                &mut alpha_real,
                &mut alpha_imag,
                &mut beta,
                &mut u,
                &mut v,
                &mut a,
                &mut b_wrong1,
            ),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_gen_eigen_lr(
                &mut alpha_real,
                &mut alpha_imag,
                &mut beta,
                &mut u,
                &mut v,
                &mut a,
                &mut b_wrong2,
            ),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn mat_gen_eigen_works() {
        #[rustfmt::skip]
        let a_data = [
            [10.0, 1.0,  2.0],
            [ 1.0, 2.0, -1.0],
            [ 1.0, 1.0,  2.0],
        ];
        #[rustfmt::skip]
        let b_data = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let mut a = Matrix::from(&a_data);
        let mut b = Matrix::from(&b_data);
        let m = a.nrow();
        let mut alpha_real = Vector::new(m);
        let mut alpha_imag = Vector::new(m);
        let mut beta = Vector::new(m);
        let mut v = Matrix::new(m, m);
        mat_gen_eigen(&mut alpha_real, &mut alpha_imag, &mut beta, &mut v, &mut a, &mut b).unwrap();
        println!("Re(α) =\n{}", alpha_real);
        println!("Im(α) =\n{}", alpha_imag);
        println!("β =\n{}", beta);
        println!("v =\n{}", v);
        // compare with reference (scipy)
        let alpha_real_ref = &[-5.040877033060623, 1.9878762745291856, 4.490731195102491];
        let alpha_imag_ref = &[0.0, 0.0, 0.0];
        let beta_ref = &[1.0510183823362071, 12.076677914155423, 0.0];
        vec_approx_eq(alpha_real.as_data(), alpha_real_ref, 1e-14);
        vec_approx_eq(alpha_imag.as_data(), alpha_imag_ref, 1e-14);
        vec_approx_eq(beta.as_data(), beta_ref, 1e-14);
        let v_ref = Matrix::from(&[
            [0.7711213016981152, 0.1548092699389088, 0.5000000000000012],
            [0.4700458576415154, -1.0, -1.0],
            [-1.0, -0.5655472382416965, 0.4999999999999993],
        ]);
        mat_approx_eq(&v, &v_ref, 1e-14);
    }

    #[test]
    fn mat_gen_eigen_lr_works() {
        // https://www.ibm.com/docs/en/essl/6.2?topic=eas-sggev-dggev-cggev-zggev-sggevx-dggevx-cggevx-zggevx-eigenvalues-optionally-right-eigenvectors-left-eigenvectors-reciprocal-condition-numbers-eigenvalues-reciprocal-condition-numbers-right-eigenvectors-general-matrix-generalized-eigenproblem
        #[rustfmt::skip]
        let a_data = [
            [2.0, 3.0, 4.0, 5.0,  6.0],
            [4.0, 4.0, 5.0, 6.0,  7.0],
            [0.0, 3.0, 6.0, 7.0,  8.0],
            [0.0, 0.0, 2.0, 8.0,  9.0],
            [0.0, 0.0, 0.0, 1.0, 10.0],
        ];
        #[rustfmt::skip]
        let b_data = [
            [1.0, -1.0, -1.0, -1.0, -1.0],
            [0.0,  1.0, -1.0, -1.0, -1.0],
            [0.0,  0.0,  1.0, -1.0, -1.0],
            [0.0,  0.0,  0.0,  1.0, -1.0],
            [0.0,  0.0,  0.0,  0.0,  1.0],
        ];
        let mut a = Matrix::from(&a_data);
        let mut b = Matrix::from(&b_data);
        let m = a.nrow();
        let mut alpha_real = Vector::new(m);
        let mut alpha_imag = Vector::new(m);
        let mut beta = Vector::new(m);
        let mut u = Matrix::new(m, m);
        let mut v = Matrix::new(m, m);
        mat_gen_eigen_lr(
            &mut alpha_real,
            &mut alpha_imag,
            &mut beta,
            &mut u,
            &mut v,
            &mut a,
            &mut b,
        )
        .unwrap();
        // println!("Re(α) =\n{}", alpha_real);
        // println!("Im(α) =\n{}", alpha_imag);
        // println!("β =\n{}", beta);
        // println!("u =\n{}", u);
        // println!("v =\n{}", v);
        // compare with reference
        let alpha_real_ref = &[7.950050, -0.277338, 2.149669, 6.720718, 10.987556];
        let alpha_imag_ref = &[0.0, 0.0, 0.0, 0.0, 0.0];
        let beta_ref = &[0.374183, 1.480299, 1.636872, 1.213574, 0.908837];
        vec_approx_eq(alpha_real.as_data(), alpha_real_ref, 1e-6);
        vec_approx_eq(alpha_imag.as_data(), alpha_imag_ref, 1e-15);
        vec_approx_eq(beta.as_data(), beta_ref, 1e-6);
        // check the definition: a ⋅ vj = lj ⋅ b ⋅ vj
        check_gen_eigen(&a_data, &b_data, &v, &alpha_real, &alpha_imag, &beta, 1e-13);
    }
}
