use crate::{
    approx_eq, complex_mat_add, complex_mat_approx_eq, complex_mat_mat_mul, complex_mat_norm, complex_mat_zip,
    complex_vec_zip, cpx, mat_add, mat_mat_mul, mat_norm, AsArray2D, Complex64, ComplexMatrix, ComplexVector, Matrix,
    Norm, Vector,
};

/// Checks Hermitian matrix given by the lower and upper parts
#[allow(dead_code)]
pub(crate) fn check_hermitian_uplo(full: &ComplexMatrix, lower: &ComplexMatrix, upper: &ComplexMatrix) {
    let (m, n) = full.dims();
    let (mm, nn) = lower.dims();
    let (mmm, nnn) = upper.dims();
    assert_eq!(m, n);
    assert!(mm == m && mmm == m && nn == m && nnn == m);
    let mut cc = ComplexMatrix::new(m, m);
    for i in 0..m {
        for j in 0..m {
            if i == j {
                cc.set(i, j, lower.get(i, j));
                assert_eq!(full.get(i, j).im, 0.0); // hermitian
            } else {
                cc.set(i, j, lower.get(i, j) + upper.get(i, j));
                assert_eq!(full.get(i, j).re, full.get(j, i).re); // hermitian
                assert_eq!(full.get(i, j).im, -full.get(j, i).im); // hermitian
            }
        }
    }
    complex_mat_approx_eq(&full, &cc, 1e-15);
}

/// Checks the eigen-decomposition of a symmetric matrix
///
/// ```text
/// a⋅v = v⋅λ
/// err := a⋅v - v⋅λ
/// ```
#[allow(dead_code)]
pub(crate) fn check_eigen_sym<'a, T>(data: &'a T, v: &Matrix, l: &Vector, tolerance: f64)
where
    T: AsArray2D<'a, f64>,
{
    let a = Matrix::from(data);
    let m = a.nrow();
    let lam = Matrix::diagonal(l.as_data());
    let mut a_v = Matrix::new(m, m);
    let mut v_l = Matrix::new(m, m);
    let mut err = Matrix::filled(m, m, f64::MAX);
    mat_mat_mul(&mut a_v, 1.0, &a, &v, 0.0).unwrap();
    let norm_a_v = mat_norm(&a_v, Norm::Max);
    if norm_a_v <= f64::EPSILON {
        panic!("norm(a⋅v) cannot be zero");
    }
    mat_mat_mul(&mut v_l, 1.0, &v, &lam, 0.0).unwrap();
    mat_add(&mut err, 1.0, &a_v, -1.0, &v_l).unwrap();
    approx_eq(mat_norm(&err, Norm::Max), 0.0, tolerance);
}

/// Checks the eigen-decomposition of a general matrix
///
/// ```text
/// a⋅v = v⋅λ
/// err := a⋅v - v⋅λ
/// ```
#[allow(dead_code)]
pub(crate) fn check_eigen<'a, T>(
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
    let mut v = ComplexMatrix::new(m, m);
    let mut d = ComplexVector::new(m);
    complex_mat_zip(&mut v, v_real, v_imag).unwrap();
    complex_vec_zip(&mut d, l_real, l_imag).unwrap();
    let lam = ComplexMatrix::diagonal(d.as_data());
    let mut a_v = ComplexMatrix::new(m, m);
    let mut v_l = ComplexMatrix::new(m, m);
    let mut err = ComplexMatrix::filled(m, m, Complex64::new(f64::MAX, f64::MAX));
    let one = Complex64::new(1.0, 0.0);
    let m_one = Complex64::new(-1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    complex_mat_mat_mul(&mut a_v, one, &a, &v, zero).unwrap();
    let norm_a_v = complex_mat_norm(&a_v, Norm::Max);
    if norm_a_v <= f64::EPSILON {
        panic!("norm(a⋅v) cannot be zero");
    }
    complex_mat_mat_mul(&mut v_l, one, &v, &lam, zero).unwrap();
    complex_mat_add(&mut err, one, &a_v, m_one, &v_l).unwrap();
    approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, tolerance);
}

/// Checks the eigen-decomposition of a general matrix
///
/// ```text
/// a⋅v = v⋅λ
/// err := a⋅v - v⋅λ
/// ```
#[allow(dead_code)]
pub(crate) fn complex_check_eigen<'a, T>(data: &'a T, v: &ComplexMatrix, l: &ComplexVector, tolerance: f64)
where
    T: AsArray2D<'a, Complex64>,
{
    let a = ComplexMatrix::from(data);
    let m = a.nrow();
    let lam = ComplexMatrix::diagonal(l.as_data());
    let mut a_v = ComplexMatrix::new(m, m);
    let mut v_l = ComplexMatrix::new(m, m);
    let mut err = ComplexMatrix::filled(m, m, Complex64::new(f64::MAX, f64::MAX));
    let one = Complex64::new(1.0, 0.0);
    let m_one = Complex64::new(-1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    complex_mat_mat_mul(&mut a_v, one, &a, &v, zero).unwrap();
    let norm_a_v = complex_mat_norm(&a_v, Norm::Max);
    if norm_a_v <= f64::EPSILON {
        panic!("norm(a⋅v) cannot be zero");
    }
    complex_mat_mat_mul(&mut v_l, one, &v, &lam, zero).unwrap();
    complex_mat_add(&mut err, one, &a_v, m_one, &v_l).unwrap();
    approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, tolerance);
}

/// Checks the generalized eigen-decomposition of a general matrix
///
/// ```text
/// a ⋅ vj = lj ⋅ b ⋅ vj
/// err := a⋅v - b⋅v⋅λ
/// ```
#[allow(dead_code)]
pub(crate) fn check_gen_eigen<'a, T>(
    a_data: &'a T,
    b_data: &'a T,
    v: &Matrix,
    alpha_real: &Vector,
    alpha_imag: &Vector,
    beta: &Vector,
    tolerance: f64,
) where
    T: AsArray2D<'a, f64>,
{
    let aa = ComplexMatrix::from(a_data);
    let bb = ComplexMatrix::from(b_data);
    let m = aa.nrow();
    let mut vv = ComplexMatrix::new(m, m);
    let mut dd = ComplexMatrix::new(m, m);
    for i in 0..m {
        assert!(f64::abs(beta[i]) > 10.0 * f64::EPSILON);
        dd.set(i, i, cpx!(alpha_real[i] / beta[i], alpha_imag[i] / beta[i]));
        for j in 0..m {
            vv.set(i, j, cpx!(v.get(i, j), 0.0));
        }
    }
    let mut a_v = ComplexMatrix::new(m, m);
    let mut v_l = ComplexMatrix::new(m, m);
    let mut b_v_l = ComplexMatrix::new(m, m);
    let mut err = ComplexMatrix::filled(m, m, cpx!(f64::MAX, 0.0));
    let zero = Complex64::new(0.0, 0.0);
    complex_mat_mat_mul(&mut a_v, cpx!(1.0, 0.0), &aa, &vv, zero).unwrap();
    let norm_a_v = complex_mat_norm(&a_v, Norm::Max);
    if norm_a_v <= f64::EPSILON {
        panic!("norm(a⋅v) cannot be zero");
    }
    complex_mat_mat_mul(&mut v_l, cpx!(1.0, 0.0), &vv, &dd, zero).unwrap();
    complex_mat_mat_mul(&mut b_v_l, cpx!(1.0, 0.0), &bb, &v_l, zero).unwrap();
    complex_mat_add(&mut err, cpx!(1.0, 0.0), &a_v, cpx!(-1.0, 0.0), &b_v_l).unwrap();
    approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, tolerance);
}

/// Checks the generalized eigen-decomposition of a general matrix
///
/// ```text
/// a ⋅ vj = lj ⋅ b ⋅ vj
/// err := a⋅v - b⋅v⋅λ
/// ```
#[allow(dead_code)]
pub(crate) fn complex_check_gen_eigen<'a, T>(
    a_data: &'a T,
    b_data: &'a T,
    v: &ComplexMatrix,
    alpha: &ComplexVector,
    beta: &ComplexVector,
    tolerance: f64,
) where
    T: AsArray2D<'a, Complex64>,
{
    let aa = ComplexMatrix::from(a_data);
    let bb = ComplexMatrix::from(b_data);
    let m = aa.nrow();
    let mut dd = ComplexMatrix::new(m, m);
    for i in 0..m {
        assert!(beta[i].norm() > 10.0 * f64::EPSILON);
        dd.set(i, i, alpha[i] / beta[i]);
    }
    let mut a_v = ComplexMatrix::new(m, m);
    let mut v_l = ComplexMatrix::new(m, m);
    let mut b_v_l = ComplexMatrix::new(m, m);
    let mut err = ComplexMatrix::filled(m, m, cpx!(f64::MAX, 0.0));
    let zero = Complex64::new(0.0, 0.0);
    complex_mat_mat_mul(&mut a_v, cpx!(1.0, 0.0), &aa, &v, zero).unwrap();
    let norm_a_v = complex_mat_norm(&a_v, Norm::Max);
    if norm_a_v <= f64::EPSILON {
        panic!("norm(a⋅v) cannot be zero");
    }
    complex_mat_mat_mul(&mut v_l, cpx!(1.0, 0.0), &v, &dd, zero).unwrap();
    complex_mat_mat_mul(&mut b_v_l, cpx!(1.0, 0.0), &bb, &v_l, zero).unwrap();
    complex_mat_add(&mut err, cpx!(1.0, 0.0), &a_v, cpx!(-1.0, 0.0), &b_v_l).unwrap();
    approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, tolerance);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{check_eigen, check_eigen_sym, complex_check_eigen};
    use crate::{cpx, Complex64, ComplexMatrix, ComplexVector, Matrix, Vector};

    #[test]
    #[should_panic]
    fn check_eigen_real_panics_on_wrong_values() {
        const WRONG: f64 = 123.0;
        let data = &[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let v = Matrix::from(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let l = Vector::from(&[2.0, 2.0, WRONG]);
        check_eigen_sym(data, &v, &l, 1e-15);
    }

    #[test]
    #[should_panic]
    fn check_eigen_real_panics_on_zero_matrix() {
        let data = &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let v = Matrix::from(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let l = Vector::from(&[0.0, 0.0, 0.0]);
        check_eigen_sym(data, &v, &l, 1e-15);
    }

    #[test]
    fn check_eigen_real_works() {
        let data = &[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let v = Matrix::from(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let l = Vector::from(&[2.0, 2.0, 2.0]);
        check_eigen_sym(data, &v, &l, 1e-15);
    }

    #[test]
    #[should_panic]
    fn check_eigen_general_panics_on_wrong_values() {
        let data = &[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let s3 = f64::sqrt(3.0);
        let l_real = Vector::from(&[-0.5, -0.5, 1.0]);
        let l_imag = Vector::from(&[s3 / 2.0, -s3 / 2.0, 0.0]);
        const WRONG: f64 = 123.0; // correct = -1.0
        let v_real = Matrix::from(&[
            [1.0 / s3, 1.0 / s3, -1.0 / s3],
            [-0.5 / s3, -0.5 / s3, -1.0 / s3],
            [-0.5 / s3, -0.5 / s3, WRONG / s3],
        ]);
        let v_imag = Matrix::from(&[[0.0, 0.0, 0.0], [0.5, -0.5, 0.0], [-0.5, 0.5, 0.0]]);
        check_eigen(data, &v_real, &l_real, &v_imag, &l_imag, 1e-15);
    }

    #[test]
    #[should_panic]
    fn check_eigen_general_panics_on_zero_matrix() {
        let data = &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let l_real = Vector::from(&[0.0, 0.0, 0.0]);
        let l_imag = Vector::from(&[0.0, 0.0, 0.0]);
        let v_real = Matrix::from(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let v_imag = Matrix::from(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        check_eigen(data, &v_real, &l_real, &v_imag, &l_imag, 1e-15);
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
        check_eigen(data, &v_real, &l_real, &v_imag, &l_imag, 1e-15);
    }

    #[test]
    #[should_panic]
    fn complex_check_eigen_panics_on_wrong_values() {
        let data = &[
            [cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(1.0, 0.0)],
            [cpx!(1.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
        ];
        let s3 = f64::sqrt(3.0);
        let l = ComplexVector::from(&[cpx!(-0.5, s3 / 2.0), cpx!(-0.5, -s3 / 2.0), cpx!(1.0, 0.0)]);
        const WRONG: f64 = 123.0; // correct = -1.0
        let v = ComplexMatrix::from(&[
            [cpx!(1.0 / s3, 0.0), cpx!(1.0 / s3, 0.0), cpx!(-1.0 / s3, 0.0)],
            [cpx!(-0.5 / s3, 0.5), cpx!(-0.5 / s3, -0.5), cpx!(-1.0 / s3, 0.0)],
            [cpx!(-0.5 / s3, -0.5), cpx!(-0.5 / s3, 0.5), cpx!(WRONG / s3, 0.0)],
        ]);
        complex_check_eigen(data, &v, &l, 1e-15);
    }

    #[test]
    fn complex_check_eigen_works() {
        let data = &[
            [cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(1.0, 0.0)],
            [cpx!(1.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
        ];
        let s3 = f64::sqrt(3.0);
        let l = ComplexVector::from(&[cpx!(-0.5, s3 / 2.0), cpx!(-0.5, -s3 / 2.0), cpx!(1.0, 0.0)]);
        let v = ComplexMatrix::from(&[
            [cpx!(1.0 / s3, 0.0), cpx!(1.0 / s3, 0.0), cpx!(-1.0 / s3, 0.0)],
            [cpx!(-0.5 / s3, 0.5), cpx!(-0.5 / s3, -0.5), cpx!(-1.0 / s3, 0.0)],
            [cpx!(-0.5 / s3, -0.5), cpx!(-0.5 / s3, 0.5), cpx!(-1.0 / s3, 0.0)],
        ]);
        complex_check_eigen(data, &v, &l, 1e-15);
    }
}
