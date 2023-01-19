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
pub(crate) fn check_eigen_real<'a, T>(data: &'a T, v: &Matrix, l: &Vector)
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
    approx_eq(mat_norm(&err, Norm::Max), 0.0, 1e-15);
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
    approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, 1e-15);
}
