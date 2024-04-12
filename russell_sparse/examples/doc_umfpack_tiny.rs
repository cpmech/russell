use russell_lab::{vec_approx_eq, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // constants
    let ndim = 3; // number of rows = number of columns
    let nnz = 5; // number of non-zero values

    // allocate solver
    let mut umfpack = SolverUMFPACK::new()?;

    // allocate the coefficient matrix
    let mut coo = SparseMatrix::new_coo(ndim, ndim, nnz, Sym::No)?;
    coo.put(0, 0, 0.2)?;
    coo.put(0, 1, 0.2)?;
    coo.put(1, 0, 0.5)?;
    coo.put(1, 1, -0.25)?;
    coo.put(2, 2, 0.25)?;

    // print matrix
    let a = coo.as_dense();
    let correct = "┌                   ┐\n\
                   │   0.2   0.2     0 │\n\
                   │   0.5 -0.25     0 │\n\
                   │     0     0  0.25 │\n\
                   └                   ┘";
    assert_eq!(format!("{}", a), correct);

    // call factorize
    umfpack.factorize(&mut coo, None)?;

    // allocate two right-hand side vectors
    let b = Vector::from(&[1.0, 1.0, 1.0]);

    // calculate the solution
    let mut x = Vector::new(ndim);
    umfpack.solve(&mut x, &coo, &b, false)?;
    let correct = vec![3.0, 2.0, 4.0];
    vec_approx_eq(&x, &correct, 1e-14);
    Ok(())
}
