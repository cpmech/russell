use russell_lab::{vec_approx_eq, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // constants
    let ndim = 3; // number of rows = number of columns
    let nnz = 5; // number of non-zero values

    // allocate the coefficient matrix
    let mut mat = SparseMatrix::new_coo(ndim, ndim, nnz, None)?;
    mat.put(0, 0, 0.2)?;
    mat.put(0, 1, 0.2)?;
    mat.put(1, 0, 0.5)?;
    mat.put(1, 1, -0.25)?;
    mat.put(2, 2, 0.25)?;

    // print matrix
    let a = mat.as_dense();
    let correct = "┌                   ┐\n\
                   │   0.2   0.2     0 │\n\
                   │   0.5 -0.25     0 │\n\
                   │     0     0  0.25 │\n\
                   └                   ┘";
    assert_eq!(format!("{}", a), correct);

    // allocate the right-hand side vector
    let rhs = Vector::from(&[1.0, 1.0, 1.0]);

    // calculate the solution
    let mut x = Vector::new(ndim);
    LinSolver::compute(Genie::Umfpack, &mut x, &mut mat, &rhs, None)?;
    let correct = vec![3.0, 2.0, 4.0];
    vec_approx_eq(x.as_data(), &correct, 1e-14);
    Ok(())
}
