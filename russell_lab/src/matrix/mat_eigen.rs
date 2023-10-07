use super::Matrix;
use crate::{dgeev_data, dgeev_data_lr, to_i32, CcBool, StrError, Vector, C_FALSE, C_TRUE};

extern "C" {
    // Computes the eigenvalues and eigenvectors of a general matrix
    // <http://www.netlib.org/lapack/explore-html/d9/d28/dgeev_8f.html>
    fn c_dgeev(
        calc_vl: CcBool,
        calc_vr: CcBool,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        wr: *mut f64,
        wi: *mut f64,
        vl: *mut f64,
        ldvl: *const i32,
        vr: *mut f64,
        ldvr: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

/// Performs the eigen-decomposition of a square matrix
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
/// * `l_real` -- (m) eigenvalues; real part
/// * `l_imag` -- (m) eigenvalues; imaginary part
/// * `v_real` -- (m,m) **right** eigenvectors (as columns); real part
/// * `v_imag` -- (m,m) **right** eigenvectors (as columns); imaginary part
///
/// # Input
///
/// * `a` -- (m,m) general matrix [will be modified]
///
/// # Note
///
/// * The matrix `a` will be modified
///
/// # Similarity transformation
///
/// The eigen-decomposition leads to a similarity transformation like so:
///
/// ```text
/// a = v⋅λ⋅v⁻¹
/// ```
///
/// where `v` is a matrix whose columns are the m linearly independent eigenvectors of `a`,
/// and `λ` is a matrix whose diagonal are the eigenvalues of `a`. Thus, the following is valid:
///
/// ```text
/// a⋅v = v⋅λ
/// ```
///
/// Let us define the error `err` as follows:
///
/// ```text
/// err := a⋅v - v⋅λ
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // set matrix
///     let data = [[2.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 4.0, 9.0]];
///     let mut a = Matrix::from(&data);
///
///     // allocate output arrays
///     let m = a.nrow();
///     let mut l_real = Vector::new(m);
///     let mut l_imag = Vector::new(m);
///     let mut v_real = Matrix::new(m, m);
///     let mut v_imag = Matrix::new(m, m);
///
///     // perform the eigen-decomposition
///     mat_eigen(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut a)?;
///
///     // check results
///     assert_eq!(
///         format!("{:.1}", l_real),
///         "┌      ┐\n\
///          │ 11.0 │\n\
///          │  1.0 │\n\
///          │  2.0 │\n\
///          └      ┘"
///     );
///     assert_eq!(
///         format!("{}", l_imag),
///         "┌   ┐\n\
///          │ 0 │\n\
///          │ 0 │\n\
///          │ 0 │\n\
///          └   ┘"
///     );
///
///     // check eigen-decomposition (similarity transformation) of a
///     // symmetric matrix with real-only eigenvalues and eigenvectors
///     let a_copy = Matrix::from(&data);
///     let lam = Matrix::diagonal(l_real.as_data());
///     let mut a_v = Matrix::new(m, m);
///     let mut v_l = Matrix::new(m, m);
///     let mut err = Matrix::filled(m, m, f64::MAX);
///     mat_mat_mul(&mut a_v, 1.0, &a_copy, &v_real)?;
///     mat_mat_mul(&mut v_l, 1.0, &v_real, &lam)?;
///     mat_add(&mut err, 1.0, &a_v, -1.0, &v_l)?;
///     approx_eq(mat_norm(&err, Norm::Max), 0.0, 1e-15);
///     Ok(())
/// }
/// ```
pub fn mat_eigen(
    l_real: &mut Vector,
    l_imag: &mut Vector,
    v_real: &mut Matrix,
    v_imag: &mut Matrix,
    a: &mut Matrix,
) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if l_real.dim() != m || l_imag.dim() != m {
        return Err("vectors are incompatible");
    }
    if v_real.nrow() != m || v_real.ncol() != m || v_imag.nrow() != m || v_imag.ncol() != m {
        return Err("matrices are incompatible");
    }
    let m_i32 = to_i32(m);
    let lda = m_i32;
    let ldu = 1;
    let ldv = m_i32;
    const EXTRA: i32 = 1;
    let lwork = 4 * m_i32 + EXTRA;
    let mut u = vec![0.0; ldu as usize];
    let mut v = vec![0.0; m * m];
    let mut work = vec![0.0; lwork as usize];
    let mut info = 0;
    unsafe {
        c_dgeev(
            C_FALSE,
            C_TRUE,
            &m_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            l_real.as_mut_data().as_mut_ptr(),
            l_imag.as_mut_data().as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            v.as_mut_ptr(),
            &ldv,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (dgeev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (dgeev): An argument had an illegal value");
    } else if info > 0 {
        println!("LAPACK ERROR (dgeev): The QR algorithm failed. Elements {}+1:N of l_real and l_imag contain eigenvalues which have converged", info-1);
        return Err("LAPACK ERROR (dgeev): The QR algorithm failed to compute all the eigenvalues, and no eigenvectors have been computed");
    }
    dgeev_data(v_real.as_mut_data(), v_imag.as_mut_data(), l_imag.as_data(), &v)
}

/// Performs the eigen-decomposition of a square matrix (left and right)
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
/// * `l_real` -- (m) eigenvalues; real part
/// * `l_imag` -- (m) eigenvalues; imaginary part
/// * `u_real` -- (m,m) **left** eigenvectors (as columns); real part
/// * `u_imag` -- (m,m) **left** eigenvectors (as columns); imaginary part
/// * `v_real` -- (m,m) **right** eigenvectors (as columns); real part
/// * `v_imag` -- (m,m) **right** eigenvectors (as columns); imaginary part
///
/// # Input
///
/// * `a` -- (m,m) general matrix [will be modified]
///
/// # Note
///
/// * The matrix `a` will be modified
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
///     let mut a = Matrix::from(&data);
///
///     // allocate output arrays
///     let m = a.nrow();
///     let mut l_real = Vector::new(m);
///     let mut l_imag = Vector::new(m);
///     let mut u_real = Matrix::new(m, m);
///     let mut u_imag = Matrix::new(m, m);
///     let mut v_real = Matrix::new(m, m);
///     let mut v_imag = Matrix::new(m, m);
///
///     // perform the eigen-decomposition
///     mat_eigen_lr(
///         &mut l_real,
///         &mut l_imag,
///         &mut u_real,
///         &mut u_imag,
///         &mut v_real,
///         &mut v_imag,
///         &mut a,
///     )?;
///
///     // check results
///     assert_eq!(
///         format!("{:.3}", l_real),
///         "┌        ┐\n\
///          │ -0.500 │\n\
///          │ -0.500 │\n\
///          │  1.000 │\n\
///          └        ┘"
///     );
///     assert_eq!(
///         format!("{:.3}", l_imag),
///         "┌        ┐\n\
///          │  0.866 │\n\
///          │ -0.866 │\n\
///          │  0.000 │\n\
///          └        ┘"
///     );
///
///     // check the eigen-decomposition (similarity transformation)
///     // ```text
///     // a⋅v = v⋅λ
///     // err := a⋅v - v⋅λ
///     // ```
///     let a = ComplexMatrix::from(&data);
///     let v = complex_mat_zip(&v_real, &v_imag)?;
///     let d = complex_vec_zip(&l_real, &l_imag)?;
///     let lam = ComplexMatrix::diagonal(d.as_data());
///     let mut a_v = ComplexMatrix::new(m, m);
///     let mut v_l = ComplexMatrix::new(m, m);
///     let mut err = ComplexMatrix::filled(m, m, Complex64::new(f64::MAX, f64::MAX));
///     let one = Complex64::new(1.0, 0.0);
///     let m_one = Complex64::new(-1.0, 0.0);
///     complex_mat_mat_mul(&mut a_v, one, &a, &v)?;
///     complex_mat_mat_mul(&mut v_l, one, &v, &lam)?;
///     complex_mat_add(&mut err, one, &a_v, m_one, &v_l)?;
///     approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, 1e-15);
///     Ok(())
/// }
/// ```
pub fn mat_eigen_lr(
    l_real: &mut Vector,
    l_imag: &mut Vector,
    u_real: &mut Matrix,
    u_imag: &mut Matrix,
    v_real: &mut Matrix,
    v_imag: &mut Matrix,
    a: &mut Matrix,
) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if l_real.dim() != m || l_imag.dim() != m {
        return Err("vectors are incompatible");
    }
    if u_real.nrow() != m
        || u_real.ncol() != m
        || u_imag.nrow() != m
        || u_imag.ncol() != m
        || v_real.nrow() != m
        || v_real.ncol() != m
        || v_imag.nrow() != m
        || v_imag.ncol() != m
    {
        return Err("matrices are incompatible");
    }
    let m_i32 = to_i32(m);
    let lda = m_i32;
    let ldu = m_i32;
    let ldv = m_i32;
    const EXTRA: i32 = 1;
    let lwork = 4 * m_i32 + EXTRA;
    let mut u = vec![0.0; m * m];
    let mut v = vec![0.0; m * m];
    let mut work = vec![0.0; lwork as usize];
    let mut info = 0;
    unsafe {
        c_dgeev(
            C_TRUE,
            C_TRUE,
            &m_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            l_real.as_mut_data().as_mut_ptr(),
            l_imag.as_mut_data().as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            v.as_mut_ptr(),
            &ldv,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (dgeev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (dgeev): An argument had an illegal value");
    } else if info > 0 {
        println!("LAPACK ERROR (dgeev): The QR algorithm failed. Elements {}+1:N of l_real and l_imag contain eigenvalues which have converged", info-1);
        return Err("LAPACK ERROR (dgeev): The QR algorithm failed to compute all the eigenvalues, and no eigenvectors have been computed");
    }
    dgeev_data_lr(
        u_real.as_mut_data(),
        u_imag.as_mut_data(),
        v_real.as_mut_data(),
        v_imag.as_mut_data(),
        l_imag.as_data(),
        &u,
        &v,
    )?;
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_eigen, mat_eigen_lr};
    use crate::mat_approx_eq;
    use crate::matrix::testing::{check_eigen_general, check_eigen_real};
    use crate::{vec_approx_eq, Matrix, Vector};

    #[test]
    fn mat_eigen_fails_on_non_square() {
        let mut a = Matrix::new(3, 4);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        assert_eq!(
            mat_eigen(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut a),
            Err("matrix must be square")
        );
    }

    #[test]
    fn mat_eigen_fails_on_wrong_dims() {
        let mut a = Matrix::new(2, 2);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        let mut l_real_wrong = Vector::new(m + 1);
        let mut l_imag_wrong = Vector::new(m + 1);
        let mut v_real_wrong = Matrix::new(m + 1, m);
        let mut v_imag_wrong = Matrix::new(m, m + 1);
        assert_eq!(
            mat_eigen(&mut l_real_wrong, &mut l_imag, &mut v_real, &mut v_imag, &mut a),
            Err("vectors are incompatible")
        );
        assert_eq!(
            mat_eigen(&mut l_real, &mut l_imag_wrong, &mut v_real, &mut v_imag, &mut a),
            Err("vectors are incompatible")
        );
        assert_eq!(
            mat_eigen(&mut l_real, &mut l_imag, &mut v_real_wrong, &mut v_imag, &mut a),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_eigen(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag_wrong, &mut a),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn mat_eigen_lr_fails_on_non_square() {
        let mut a = Matrix::new(3, 4);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        let mut u_real = Matrix::new(m, m);
        let mut u_imag = Matrix::new(m, m);
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        assert_eq!(
            mat_eigen_lr(
                &mut l_real,
                &mut l_imag,
                &mut u_real,
                &mut u_imag,
                &mut v_real,
                &mut v_imag,
                &mut a,
            ),
            Err("matrix must be square"),
        );
    }

    #[test]
    fn mat_eigen_lr_fails_on_wrong_dims() {
        let mut a = Matrix::new(2, 2);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        let mut u_real = Matrix::new(m, m);
        let mut u_imag = Matrix::new(m, m);
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        let mut l_real_wrong = Vector::new(m + 1);
        let mut l_imag_wrong = Vector::new(m + 1);
        let mut u_real_wrong = Matrix::new(m + 1, m);
        let mut u_imag_wrong = Matrix::new(m, m + 1);
        let mut v_real_wrong = Matrix::new(m + 1, m);
        let mut v_imag_wrong = Matrix::new(m, m + 1);
        assert_eq!(
            mat_eigen_lr(
                &mut l_real_wrong,
                &mut l_imag,
                &mut u_real,
                &mut u_imag,
                &mut v_real,
                &mut v_imag,
                &mut a,
            ),
            Err("vectors are incompatible"),
        );
        assert_eq!(
            mat_eigen_lr(
                &mut l_real,
                &mut l_imag_wrong,
                &mut u_real,
                &mut u_imag,
                &mut v_real,
                &mut v_imag,
                &mut a,
            ),
            Err("vectors are incompatible"),
        );
        assert_eq!(
            mat_eigen_lr(
                &mut l_real,
                &mut l_imag,
                &mut u_real_wrong,
                &mut u_imag,
                &mut v_real,
                &mut v_imag,
                &mut a,
            ),
            Err("matrices are incompatible"),
        );
        assert_eq!(
            mat_eigen_lr(
                &mut l_real,
                &mut l_imag,
                &mut u_real,
                &mut u_imag_wrong,
                &mut v_real,
                &mut v_imag,
                &mut a,
            ),
            Err("matrices are incompatible"),
        );
        assert_eq!(
            mat_eigen_lr(
                &mut l_real,
                &mut l_imag,
                &mut u_real,
                &mut u_imag,
                &mut v_real_wrong,
                &mut v_imag,
                &mut a,
            ),
            Err("matrices are incompatible"),
        );
        assert_eq!(
            mat_eigen_lr(
                &mut l_real,
                &mut l_imag,
                &mut u_real,
                &mut u_imag,
                &mut v_real,
                &mut v_imag_wrong,
                &mut a,
            ),
            Err("matrices are incompatible"),
        );
    }

    #[test]
    fn mat_eigen_works() {
        #[rustfmt::skip]
        let data = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let mut a = Matrix::from(&data);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        mat_eigen(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut a).unwrap();
        let s3 = f64::sqrt(3.0);
        let l_real_correct = &[-0.5, -0.5, 1.0];
        let l_imag_correct = &[s3 / 2.0, -s3 / 2.0, 0.0];
        vec_approx_eq(l_real.as_data(), l_real_correct, 1e-15);
        vec_approx_eq(l_imag.as_data(), l_imag_correct, 1e-15);
        check_eigen_general(&data, &v_real, &l_real, &v_imag, &l_imag, 1e-15);
    }

    #[test]
    fn mat_eigen_repeated_eval_works() {
        // rep: repeated eigenvalues
        #[rustfmt::skip]
        let data = [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],
        ];
        let mut a = Matrix::from(&data);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        mat_eigen(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut a).unwrap();
        let l_real_correct = &[3.0, 3.0, 2.0, 2.0];
        let l_imag_correct = &[0.0, 0.0, 0.0, 0.0];
        let os3 = 1.0 / f64::sqrt(3.0);
        #[rustfmt::skip]
        let v_real_correct = &[
            [0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  os3, -os3],
            [0.0,  0.0, -os3,  os3],
            [1.0, -1.0,  os3, -os3],
        ];
        let v_imag_correct = Matrix::new(4, 4);
        vec_approx_eq(l_real.as_data(), l_real_correct, 1e-15);
        vec_approx_eq(l_imag.as_data(), l_imag_correct, 1e-15);
        mat_approx_eq(&v_real, v_real_correct, 1e-15);
        mat_approx_eq(&v_imag, &v_imag_correct, 1e-15);
        check_eigen_real(&data, &v_real, &l_real, 1e-15);
    }

    #[test]
    fn mat_eigen_lr_works() {
        #[rustfmt::skip]
        let data = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let mut a = Matrix::from(&data);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        let mut u_real = Matrix::new(m, m);
        let mut u_imag = Matrix::new(m, m);
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        mat_eigen_lr(
            &mut l_real,
            &mut l_imag,
            &mut u_real,
            &mut u_imag,
            &mut v_real,
            &mut v_imag,
            &mut a,
        )
        .unwrap();
        let s3 = f64::sqrt(3.0);
        let l_real_correct = &[-0.5, -0.5, 1.0];
        let l_imag_correct = &[s3 / 2.0, -s3 / 2.0, 0.0];
        vec_approx_eq(l_real.as_data(), l_real_correct, 1e-15);
        vec_approx_eq(l_imag.as_data(), l_imag_correct, 1e-15);
        check_eigen_general(&data, &v_real, &l_real, &v_imag, &l_imag, 1e-15);
    }
}
