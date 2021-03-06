use super::Matrix;
use crate::StrError;
use russell_openblas::{dgeev, dgeev_data, dgeev_data_lr, to_i32};

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
/// // import
/// use russell_lab::{add_matrices, eigen_decomp, mat_mat_mul, matrix_norm, Matrix, NormMat, StrError};
/// use russell_chk::assert_approx_eq;
///
/// fn main() -> Result<(), StrError> {
///     // set matrix
///     let data = [
///         [2.0, 0.0, 0.0],
///         [0.0, 3.0, 4.0],
///         [0.0, 4.0, 9.0],
///     ];
///     let mut a = Matrix::from(&data);
///
///     // allocate output arrays
///     let m = a.nrow();
///     let mut l_real = vec![0.0; m];
///     let mut l_imag = vec![0.0; m];
///     let mut v_real = Matrix::new(m, m);
///     let mut v_imag = Matrix::new(m, m);
///
///     // perform the eigen-decomposition
///     eigen_decomp(
///         &mut l_real,
///         &mut l_imag,
///         &mut v_real,
///         &mut v_imag,
///         &mut a,
///     )?;
///
///     // check results
///     let l_real_correct = "[11.0, 1.0, 2.0]";
///     let l_imag_correct = "[0.0, 0.0, 0.0]";
///     assert_eq!(format!("{:?}", l_real), l_real_correct);
///     assert_eq!(format!("{:?}", l_imag), l_imag_correct);
///
///     // check eigen-decomposition (similarity transformation) of a
///     // symmetric matrix with real-only eigenvalues and eigenvectors
///     let a_copy = Matrix::from(&data);
///     let lam = Matrix::diagonal(&l_real);
///     let mut a_v = Matrix::new(m, m);
///     let mut v_l = Matrix::new(m, m);
///     let mut err = Matrix::filled(m, m, f64::MAX);
///     mat_mat_mul(&mut a_v, 1.0, &a_copy, &v_real)?;
///     mat_mat_mul(&mut v_l, 1.0, &v_real, &lam)?;
///     add_matrices(&mut err, 1.0, &a_v, -1.0, &v_l)?;
///     assert_approx_eq!(matrix_norm(&err, NormMat::Max), 0.0, 1e-15);
///     Ok(())
/// }
/// ```
pub fn eigen_decomp(
    l_real: &mut [f64],
    l_imag: &mut [f64],
    v_real: &mut Matrix,
    v_imag: &mut Matrix,
    a: &mut Matrix,
) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if l_real.len() != m || l_imag.len() != m {
        return Err("vectors are incompatible");
    }
    if v_real.nrow() != m || v_real.ncol() != m || v_imag.nrow() != m || v_imag.ncol() != m {
        return Err("matrices are incompatible");
    }
    let m_i32 = to_i32(m);
    let mut v = vec![0.0; m * m];
    let mut empty: Vec<f64> = Vec::new();
    dgeev(false, true, m_i32, a.as_mut_data(), l_real, l_imag, &mut empty, &mut v)?;
    dgeev_data(v_real.as_mut_data(), v_imag.as_mut_data(), l_imag, &v)?;
    Ok(())
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
/// use russell_chk::assert_approx_eq;
/// use russell_lab::{
///     complex_add_matrices, complex_mat_mat_mul, complex_mat_zip,
///     complex_matrix_norm, complex_vec_zip, ComplexMatrix, NormMat,
/// };
/// use russell_lab::{eigen_decomp_lr, Matrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // set matrix
///     let data = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
///     let mut a = Matrix::from(&data);
///
///     // allocate output arrays
///     let m = a.nrow();
///     let mut l_real = vec![0.0; m];
///     let mut l_imag = vec![0.0; m];
///     let mut u_real = Matrix::new(m, m);
///     let mut u_imag = Matrix::new(m, m);
///     let mut v_real = Matrix::new(m, m);
///     let mut v_imag = Matrix::new(m, m);
///
///     // perform the eigen-decomposition
///     eigen_decomp_lr(
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
///     let l_real_correct = "[-0.5, -0.5, 0.9999999999999998]";
///     let l_imag_correct = "[0.8660254037844389, -0.8660254037844389, 0.0]";
///     assert_eq!(format!("{:?}", l_real), l_real_correct);
///     assert_eq!(format!("{:?}", l_imag), l_imag_correct);
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
///     complex_add_matrices(&mut err, one, &a_v, m_one, &v_l)?;
///     assert_approx_eq!(complex_matrix_norm(&err, NormMat::Max), 0.0, 1e-15);
///     Ok(())
/// }
/// ```
pub fn eigen_decomp_lr(
    l_real: &mut [f64],
    l_imag: &mut [f64],
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
    if l_real.len() != m || l_imag.len() != m {
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
    let mut u = vec![0.0; m * m];
    let mut v = vec![0.0; m * m];
    dgeev(true, true, m_i32, a.as_mut_data(), l_real, l_imag, &mut u, &mut v)?;
    dgeev_data_lr(
        u_real.as_mut_data(),
        u_imag.as_mut_data(),
        v_real.as_mut_data(),
        v_imag.as_mut_data(),
        l_imag,
        &u,
        &v,
    )?;
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{eigen_decomp, eigen_decomp_lr};
    use crate::{
        add_matrices, complex_add_matrices, complex_mat_mat_mul, complex_mat_zip, complex_matrix_norm, complex_vec_zip,
        mat_mat_mul, matrix_norm, AsArray2D, ComplexMatrix, Matrix, NormMat, StrError,
    };
    use num_complex::Complex64;
    use russell_chk::{assert_approx_eq, assert_vec_approx_eq};

    // Checks the eigen-decomposition (similarity transformation) of a
    // symmetric matrix with real-only eigenvalues and eigenvectors
    // ```text
    // a⋅v = v⋅λ
    // err := a⋅v - v⋅λ
    // ```
    fn check_real_eigen<'a, T>(data: &'a T, v: &Matrix, l: &[f64]) -> Result<(), StrError>
    where
        T: AsArray2D<'a, f64>,
    {
        let a = Matrix::from(data);
        let m = a.nrow();
        let lam = Matrix::diagonal(&l);
        let mut a_v = Matrix::new(m, m);
        let mut v_l = Matrix::new(m, m);
        let mut err = Matrix::filled(m, m, f64::MAX);
        mat_mat_mul(&mut a_v, 1.0, &a, &v)?;
        mat_mat_mul(&mut v_l, 1.0, &v, &lam)?;
        add_matrices(&mut err, 1.0, &a_v, -1.0, &v_l)?;
        assert_approx_eq!(matrix_norm(&err, NormMat::Max), 0.0, 1e-15);
        Ok(())
    }

    // Checks the eigen-decomposition (similarity transformation)
    // ```text
    // a⋅v = v⋅λ
    // err := a⋅v - v⋅λ
    // ```
    fn check_general_eigen<'a, T>(
        data: &'a T,
        v_real: &Matrix,
        l_real: &[f64],
        v_imag: &Matrix,
        l_imag: &[f64],
    ) -> Result<(), StrError>
    where
        T: AsArray2D<'a, f64>,
    {
        let a = ComplexMatrix::from(data);
        let m = a.nrow();
        let v = complex_mat_zip(v_real, v_imag)?;
        let d = complex_vec_zip(l_real, l_imag)?;
        let lam = ComplexMatrix::diagonal(d.as_data());
        let mut a_v = ComplexMatrix::new(m, m);
        let mut v_l = ComplexMatrix::new(m, m);
        let mut err = ComplexMatrix::filled(m, m, Complex64::new(f64::MAX, f64::MAX));
        let one = Complex64::new(1.0, 0.0);
        let m_one = Complex64::new(-1.0, 0.0);
        complex_mat_mat_mul(&mut a_v, one, &a, &v)?;
        complex_mat_mat_mul(&mut v_l, one, &v, &lam)?;
        complex_add_matrices(&mut err, one, &a_v, m_one, &v_l)?;
        assert_approx_eq!(complex_matrix_norm(&err, NormMat::Max), 0.0, 1e-15);
        Ok(())
    }

    #[test]
    fn eigen_decomp_fails_on_non_square() {
        let mut a = Matrix::new(3, 4);
        let m = a.nrow();
        let mut l_real = vec![0.0; m];
        let mut l_imag = vec![0.0; m];
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        assert_eq!(
            eigen_decomp(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut a),
            Err("matrix must be square")
        );
    }

    #[test]
    fn eigen_decomp_fails_on_wrong_dims() {
        let mut a = Matrix::new(2, 2);
        let m = a.nrow();
        let mut l_real = vec![0.0; m];
        let mut l_imag = vec![0.0; m];
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        let mut l_real_wrong = vec![0.0; m + 1];
        let mut l_imag_wrong = vec![0.0; m + 1];
        let mut v_real_wrong = Matrix::new(m + 1, m);
        let mut v_imag_wrong = Matrix::new(m, m + 1);
        assert_eq!(
            eigen_decomp(&mut l_real_wrong, &mut l_imag, &mut v_real, &mut v_imag, &mut a),
            Err("vectors are incompatible")
        );
        assert_eq!(
            eigen_decomp(&mut l_real, &mut l_imag_wrong, &mut v_real, &mut v_imag, &mut a),
            Err("vectors are incompatible")
        );
        assert_eq!(
            eigen_decomp(&mut l_real, &mut l_imag, &mut v_real_wrong, &mut v_imag, &mut a),
            Err("matrices are incompatible")
        );
        assert_eq!(
            eigen_decomp(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag_wrong, &mut a),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn eigen_decomp_lr_fails_on_non_square() {
        let mut a = Matrix::new(3, 4);
        let m = a.nrow();
        let mut l_real = vec![0.0; m];
        let mut l_imag = vec![0.0; m];
        let mut u_real = Matrix::new(m, m);
        let mut u_imag = Matrix::new(m, m);
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        assert_eq!(
            eigen_decomp_lr(
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
    fn eigen_decomp_lr_fails_on_wrong_dims() {
        let mut a = Matrix::new(2, 2);
        let m = a.nrow();
        let mut l_real = vec![0.0; m];
        let mut l_imag = vec![0.0; m];
        let mut u_real = Matrix::new(m, m);
        let mut u_imag = Matrix::new(m, m);
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        let mut l_real_wrong = vec![0.0; m + 1];
        let mut l_imag_wrong = vec![0.0; m + 1];
        let mut u_real_wrong = Matrix::new(m + 1, m);
        let mut u_imag_wrong = Matrix::new(m, m + 1);
        let mut v_real_wrong = Matrix::new(m + 1, m);
        let mut v_imag_wrong = Matrix::new(m, m + 1);
        assert_eq!(
            eigen_decomp_lr(
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
            eigen_decomp_lr(
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
            eigen_decomp_lr(
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
            eigen_decomp_lr(
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
            eigen_decomp_lr(
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
            eigen_decomp_lr(
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
    fn eigen_decomp_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let data = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let mut a = Matrix::from(&data);
        let m = a.nrow();
        let mut l_real = vec![0.0; m];
        let mut l_imag = vec![0.0; m];
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        eigen_decomp(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut a)?;
        let s3 = f64::sqrt(3.0);
        let l_real_correct = &[-0.5, -0.5, 1.0];
        let l_imag_correct = &[s3 / 2.0, -s3 / 2.0, 0.0];
        assert_vec_approx_eq!(l_real, l_real_correct, 1e-15);
        assert_vec_approx_eq!(l_imag, l_imag_correct, 1e-15);
        check_general_eigen(&data, &v_real, &l_real, &v_imag, &l_imag)?;
        Ok(())
    }

    #[test]
    fn eigen_decomp_repeated_eval_works() -> Result<(), StrError> {
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
        let mut l_real = vec![0.0; m];
        let mut l_imag = vec![0.0; m];
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        eigen_decomp(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut a)?;
        let l_real_correct = &[3.0, 3.0, 2.0, 2.0];
        let l_imag_correct = &[0.0, 0.0, 0.0, 0.0];
        let os3 = 1.0 / f64::sqrt(3.0);
        #[rustfmt::skip]
        let v_real_correct = [
            0.0,  0.0,  0.0,  0.0,
            0.0,  0.0,  os3, -os3,
            0.0,  0.0, -os3,  os3,
            1.0, -1.0,  os3, -os3,
        ];
        #[rustfmt::skip]
        let v_imag_correct = [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_vec_approx_eq!(l_real, l_real_correct, 1e-15);
        assert_vec_approx_eq!(l_imag, l_imag_correct, 1e-15);
        assert_vec_approx_eq!(v_real.as_data(), v_real_correct, 1e-15);
        assert_vec_approx_eq!(v_imag.as_data(), v_imag_correct, 1e-15);
        check_real_eigen(&data, &v_real, &l_real)?;
        Ok(())
    }

    #[test]
    fn eigen_decomp_lr_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let data = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let mut a = Matrix::from(&data);
        let m = a.nrow();
        let mut l_real = vec![0.0; m];
        let mut l_imag = vec![0.0; m];
        let mut u_real = Matrix::new(m, m);
        let mut u_imag = Matrix::new(m, m);
        let mut v_real = Matrix::new(m, m);
        let mut v_imag = Matrix::new(m, m);
        eigen_decomp_lr(
            &mut l_real,
            &mut l_imag,
            &mut u_real,
            &mut u_imag,
            &mut v_real,
            &mut v_imag,
            &mut a,
        )?;
        let s3 = f64::sqrt(3.0);
        let l_real_correct = &[-0.5, -0.5, 1.0];
        let l_imag_correct = &[s3 / 2.0, -s3 / 2.0, 0.0];
        assert_vec_approx_eq!(l_real, l_real_correct, 1e-15);
        assert_vec_approx_eq!(l_imag, l_imag_correct, 1e-15);
        check_general_eigen(&data, &v_real, &l_real, &v_imag, &l_imag)?;
        Ok(())
    }
}
