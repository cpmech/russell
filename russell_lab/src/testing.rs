#![allow(dead_code)]

use crate::{
    complex_mat_add, complex_mat_mat_mul, complex_mat_norm, complex_mat_zip, complex_vec_zip, mat_add, mat_mat_mul,
    mat_norm, AsArray2D, ComplexMatrix, Matrix, Norm, Vector,
};
use num_complex::Complex64;
use russell_chk::approx_eq;

/// Checks the eigen-decomposition (similarity transformation) of a
/// symmetric matrix with real-only eigenvalues and eigenvectors
///
/// ```text
/// a⋅v = v⋅λ
/// err := a⋅v - v⋅λ
/// ```
pub(crate) fn check_eigen_real<'a, T>(data: &'a T, v: &Matrix, l: &Vector, tolerance: f64)
where
    T: AsArray2D<'a, f64>,
{
    let a = Matrix::from(data);
    let m = a.nrow();
    let lam = Matrix::diagonal(l.as_data());
    let mut a_v = Matrix::new(m, m);
    let mut v_l = Matrix::new(m, m);
    let mut err = Matrix::filled(m, m, f64::MAX);
    mat_mat_mul(&mut a_v, 1.0, &a, &v).unwrap();
    mat_mat_mul(&mut v_l, 1.0, &v, &lam).unwrap();
    mat_add(&mut err, 1.0, &a_v, -1.0, &v_l).unwrap();
    approx_eq(mat_norm(&err, Norm::Max), 0.0, tolerance);
}

/// Checks the eigen-decomposition (similarity transformation) of a
/// general matrix
///
/// ```text
/// a⋅v = v⋅λ
/// err := a⋅v - v⋅λ
/// ```
pub(crate) fn check_eigen_general<'a, T>(
    data: &'a T,
    v_real: &Matrix,
    l_real: &Vector,
    v_imag: &Matrix,
    l_imag: &Vector,
    tolerance: f64,
) where
    T: AsArray2D<'a, f64>,
{
    let a = ComplexMatrix::from(data);
    let m = a.nrow();
    let v = complex_mat_zip(v_real, v_imag).unwrap();
    let d = complex_vec_zip(l_real, l_imag).unwrap();
    let lam = ComplexMatrix::diagonal(d.as_data());
    let mut a_v = ComplexMatrix::new(m, m);
    let mut v_l = ComplexMatrix::new(m, m);
    let mut err = ComplexMatrix::filled(m, m, Complex64::new(f64::MAX, f64::MAX));
    let one = Complex64::new(1.0, 0.0);
    let m_one = Complex64::new(-1.0, 0.0);
    complex_mat_mat_mul(&mut a_v, one, &a, &v).unwrap();
    complex_mat_mat_mul(&mut v_l, one, &v, &lam).unwrap();
    complex_mat_add(&mut err, one, &a_v, m_one, &v_l).unwrap();
    approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, tolerance);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{check_eigen_general, check_eigen_real};
    use crate::{Matrix, Vector};

    #[test]
    #[should_panic]
    fn check_eigen_real_panics_on_wrong_values() {
        const WRONG: f64 = 123.0;
        let data = &[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let v = Matrix::from(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let l = Vector::from(&[2.0, 2.0, WRONG]);
        check_eigen_real(data, &v, &l, 1e-15);
    }

    #[test]
    fn check_eigen_real_works() {
        let data = &[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let v = Matrix::from(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let l = Vector::from(&[2.0, 2.0, 2.0]);
        check_eigen_real(data, &v, &l, 1e-15);
    }

    #[test]
    #[should_panic]
    fn check_eigen_general_panics_on_wrong_values() {
        let data = &[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let s3 = f64::sqrt(3.0);
        let l_real = Vector::from(&[-0.5, -0.5, 1.0]);
        let l_imag = Vector::from(&[s3 / 2.0, -s3 / 2.0, 0.0]);
        const WRONG: f64 = 123.0;
        let v_real = Matrix::from(&[
            [1.0 / s3, 1.0 / s3, -1.0 / s3],
            [-0.5 / s3, -0.5 / s3, -1.0 / s3],
            [-0.5 / s3, -0.5 / s3, WRONG / s3],
        ]);
        let v_imag = Matrix::from(&[[0.0, 0.0, 0.0], [0.5, -0.5, 0.0], [-0.5, 0.5, 0.0]]);
        check_eigen_general(data, &v_real, &l_real, &v_imag, &l_imag, 1e-15);
    }

    #[test]
    fn check_eigen_general_works() {
        let data = &[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let s3 = f64::sqrt(3.0);
        let l_real = Vector::from(&[-0.5, -0.5, 1.0]);
        let l_imag = Vector::from(&[s3 / 2.0, -s3 / 2.0, 0.0]);
        let v_real = Matrix::from(&[
            [1.0 / s3, 1.0 / s3, -1.0 / s3],
            [-0.5 / s3, -0.5 / s3, -1.0 / s3],
            [-0.5 / s3, -0.5 / s3, -1.0 / s3],
        ]);
        let v_imag = Matrix::from(&[[0.0, 0.0, 0.0], [0.5, -0.5, 0.0], [-0.5, 0.5, 0.0]]);
        check_eigen_general(data, &v_real, &l_real, &v_imag, &l_imag, 1e-15);
    }
}
