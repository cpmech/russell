use russell_chk::vec_approx_eq;
use russell_lab::{Matrix, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // solving A · x = b for x
    // allocate the coefficient matrix A
    let n = 5;
    let ap = vec![0, 2, 5, 9, 10, 12];
    let ai = vec![0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4];
    let ax = vec![2.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0];
    let mut csc = SparseMatrix::new_csc(n, n, ap, ai, ax, None)?;

    // print the coefficient matrix
    let mut a = Matrix::new(n, n);
    csc.to_dense(&mut a)?;
    let correct = "┌                ┐\n\
                   │  2  3  0  0  0 │\n\
                   │  3  0  4  0  6 │\n\
                   │  0 -1 -3  2  0 │\n\
                   │  0  0  1  0  0 │\n\
                   │  0  4  2  0  1 │\n\
                   └                ┘";
    assert_eq!(format!("{}", a), correct);

    // allocate the solver
    let mut umfpack = SolverUMFPACK::new()?;

    // parameters
    let mut params = LinSolParams::new();
    params.verbose = true;

    // call factorize
    umfpack.factorize(&mut csc, Some(params))?;

    // allocate x and b
    let mut x = Vector::new(n);
    let b = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    // calculate the solution
    // x = inv(A) · b
    umfpack.solve(&mut x, &csc, &b, false)?;

    // check the results
    let correct = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    vec_approx_eq(x.as_data(), &correct, 1e-14);

    // solve again with doubled rhs vector
    let b_times_2 = b.get_mapped(|x| x * 2.0);
    umfpack.solve(&mut x, &csc, &b_times_2, false)?;

    // check the results (again)
    let correct = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    vec_approx_eq(x.as_data(), &correct, 1e-14);
    Ok(())
}
