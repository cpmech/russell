use russell_chk::vec_approx_eq;
use russell_lab::{Matrix, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // allocate a square matrix
    let nrow = 3; // number of equations
    let ncol = nrow; // number of equations
    let nnz = 5; // number of non-zeros
    let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz)?;
    coo.put(0, 0, 0.2)?;
    coo.put(0, 1, 0.2)?;
    coo.put(1, 0, 0.5)?;
    coo.put(1, 1, -0.25)?;
    coo.put(2, 2, 0.25)?;

    // print matrix
    let mut a = Matrix::new(nrow, ncol);
    coo.to_matrix(&mut a)?;
    let correct = "┌                   ┐\n\
                   │   0.2   0.2     0 │\n\
                   │   0.5 -0.25     0 │\n\
                   │     0     0  0.25 │\n\
                   └                   ┘";
    assert_eq!(format!("{}", a), correct);

    // allocate solver, initialize, and factorize
    let mut solver = SolverUMFPACK::new()?;
    solver.initialize(&coo, false)?;
    solver.factorize(&coo, false)?;

    // allocate rhs
    let rhs1 = Vector::from(&[1.0, 1.0, 1.0]);
    let rhs2 = Vector::from(&[2.0, 2.0, 2.0]);

    // calculate solution
    let mut x1 = Vector::new(nrow);
    solver.solve(&mut x1, &rhs1, false)?;
    let correct = vec![3.0, 2.0, 4.0];
    vec_approx_eq(x1.as_data(), &correct, 1e-14);

    // solve again
    let mut x2 = Vector::new(nrow);
    solver.solve(&mut x2, &rhs2, false)?;
    let correct = vec![6.0, 4.0, 8.0];
    vec_approx_eq(x2.as_data(), &correct, 1e-14);
    Ok(())
}
