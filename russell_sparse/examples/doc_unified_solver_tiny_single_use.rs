use russell_chk::vec_approx_eq;
use russell_lab::{Matrix, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // constants
    let ndim = 3; // number of rows = number of columns
    let nnz = 5; // number of non-zero values

    // allocate the coefficient matrix
    let mut coo = CooMatrix::new(None, ndim, ndim, nnz)?;
    coo.put(0, 0, 0.2)?;
    coo.put(0, 1, 0.2)?;
    coo.put(1, 0, 0.5)?;
    coo.put(1, 1, -0.25)?;
    coo.put(2, 2, 0.25)?;

    // print matrix
    let mut a = Matrix::new(ndim, ndim);
    coo.to_matrix(&mut a)?;
    let correct = "┌                   ┐\n\
                   │   0.2   0.2     0 │\n\
                   │   0.5 -0.25     0 │\n\
                   │     0     0  0.25 │\n\
                   └                   ┘";
    assert_eq!(format!("{}", a), correct);

    // solve the linear system
    let mut x = Vector::new(ndim);
    let rhs = Vector::from(&[1.0, 1.0, 1.0]);
    Solver::compute(Genie::Umfpack, &coo, &mut x, &rhs, false)?;

    // check the results
    let correct = vec![3.0, 2.0, 4.0];
    vec_approx_eq(x.as_data(), &correct, 1e-14);
    Ok(())
}
