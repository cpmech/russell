use russell_chk::approx_eq;
use russell_lab::prelude::*;
use russell_lab::StrError;

fn main() -> Result<(), StrError> {
    // set matrix
    let data = [[2.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 4.0, 9.0]];
    let mut a = Matrix::from(&data);

    // allocate output arrays
    let m = a.nrow();
    let mut l_real = Vector::new(m);
    let mut l_imag = Vector::new(m);
    let mut v_real = Matrix::new(m, m);
    let mut v_imag = Matrix::new(m, m);

    // perform the eigen-decomposition
    mat_eigen(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut a)?;

    // check results
    assert_eq!(
        format!("{:.1}", l_real),
        "┌      ┐\n\
         │ 11.0 │\n\
         │  1.0 │\n\
         │  2.0 │\n\
         └      ┘"
    );
    assert_eq!(
        format!("{}", l_imag),
        "┌   ┐\n\
         │ 0 │\n\
         │ 0 │\n\
         │ 0 │\n\
         └   ┘"
    );

    // check eigen-decomposition (similarity transformation) of a
    // symmetric matrix with real-only eigenvalues and eigenvectors
    let a_copy = Matrix::from(&data);
    let lam = Matrix::diagonal(l_real.as_data());
    let mut a_v = Matrix::new(m, m);
    let mut v_l = Matrix::new(m, m);
    let mut err = Matrix::filled(m, m, f64::MAX);
    mat_mat_mul(&mut a_v, 1.0, &a_copy, &v_real)?;
    mat_mat_mul(&mut v_l, 1.0, &v_real, &lam)?;
    mat_add(&mut err, 1.0, &a_v, -1.0, &v_l)?;
    approx_eq(mat_norm(&err, Norm::Max), 0.0, 1e-15);
    Ok(())
}
