use russell_lab::{vec_approx_eq, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // constants
    let ndim = 3; // number of rows = number of columns
    let nnz = 5; // number of non-zero values

    // allocate the linear solver
    let mut solver = LinSolver::new(Genie::Umfpack)?;

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
    solver.actual.factorize(&mut coo, None)?;

    // allocate two right-hand side vectors
    let rhs1 = Vector::from(&[1.0, 1.0, 1.0]);
    let rhs2 = Vector::from(&[2.0, 2.0, 2.0]);

    // calculate the solution
    let mut x1 = Vector::new(ndim);
    solver.actual.solve(&mut x1, &coo, &rhs1, false)?;
    let correct = vec![3.0, 2.0, 4.0];
    vec_approx_eq(x1.as_data(), &correct, 1e-14);

    // calculate the solution again
    let mut x2 = Vector::new(ndim);
    solver.actual.solve(&mut x2, &coo, &rhs2, false)?;
    let correct = vec![6.0, 4.0, 8.0];
    vec_approx_eq(x2.as_data(), &correct, 1e-14);
    Ok(())
}
