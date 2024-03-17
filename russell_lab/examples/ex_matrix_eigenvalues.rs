use num_complex::Complex64;
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // data for matrix "A"
    #[rustfmt::skip]
    let data = [
        [ 0.35,  0.45, -0.14, -0.17],
        [ 0.09,  0.07, -0.54,  0.35],
        [-0.44, -0.33, -0.03,  0.17],
        [ 0.25, -0.32, -0.13,  0.11],
    ];

    // "A" matrix (will be modified)
    let mut a = Matrix::from(&data);

    // allocate output arrays
    let m = a.nrow();
    let mut l_real = Vector::new(m);
    let mut l_imag = Vector::new(m);
    let mut u_real = Matrix::new(m, m);
    let mut u_imag = Matrix::new(m, m);
    let mut v_real = Matrix::new(m, m);
    let mut v_imag = Matrix::new(m, m);

    // perform the eigen-decomposition
    mat_eigen_lr(
        &mut l_real,
        &mut l_imag,
        &mut u_real,
        &mut u_imag,
        &mut v_real,
        &mut v_imag,
        &mut a,
    )?;

    // check eigenvalues
    #[rustfmt::skip]
    let lambda_real_correct = &[
         7.994821225862098e-01,
        -9.941245329507467e-02,
        -9.941245329507467e-02,
        -1.006572159960587e-01,
    ];
    #[rustfmt::skip]
    let lambda_imag_correct = &[
        0.0,
        4.007924719897546e-01,
        -4.007924719897546e-01,
        0.0,
    ];
    vec_approx_eq(l_real.as_data(), lambda_real_correct, 1e-14);
    vec_approx_eq(l_imag.as_data(), lambda_imag_correct, 1e-14);

    // check the eigen-decomposition (similarity transformation)
    // ```text
    // a⋅v = v⋅λ
    // err := a⋅v - v⋅λ
    // ```
    let a = ComplexMatrix::from(&data);
    let mut v = ComplexMatrix::new(m, m);
    let mut d = ComplexVector::new(m);
    complex_mat_zip(&mut v, &v_real, &v_imag)?;
    complex_vec_zip(&mut d, &l_real, &l_imag)?;
    let lam = ComplexMatrix::diagonal(d.as_data());
    let mut a_v = ComplexMatrix::new(m, m);
    let mut v_l = ComplexMatrix::new(m, m);
    let mut err = ComplexMatrix::filled(m, m, cpx!(f64::MAX, f64::MAX));
    let one = cpx!(1.0, 0.0);
    let m_one = cpx!(-1.0, 0.0);
    let zero = cpx!(0.0, 0.0);
    complex_mat_mat_mul(&mut a_v, one, &a, &v, zero)?;
    complex_mat_mat_mul(&mut v_l, one, &v, &lam, zero)?;
    complex_mat_add(&mut err, one, &a_v, m_one, &v_l)?;
    approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, 1e-14);
    Ok(())
}
