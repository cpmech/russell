use russell_lab::{ComplexVector, complex_vec_approx_eq, cpx};
use russell_sparse::StrError;
use russell_sparse::prelude::*;

fn main() -> Result<(), StrError> {
    let ndim = 3;
    let nnz = 7;

    // allocate the coefficient matrix
    //   ┌                      ┐
    //   │  2+1i  -1-1i     0   │
    //   │ -1-1i   2+2i  -1+1i  │
    //   │   0    -1+1i   2-1i  │
    //   └                      ┘
    let mut mat = ComplexCooMatrix::new(ndim, ndim, nnz, Sym::No)?;
    mat.put(0, 0, cpx!(2.0, 1.0))?;
    mat.put(0, 1, cpx!(-1.0, -1.0))?;
    mat.put(1, 0, cpx!(-1.0, -1.0))?;
    mat.put(1, 1, cpx!(2.0, 2.0))?;
    mat.put(1, 2, cpx!(-1.0, 1.0))?;
    mat.put(2, 1, cpx!(-1.0, 1.0))?;
    mat.put(2, 2, cpx!(2.0, -1.0))?;

    // right-hand side vector
    let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);

    // solve (single-use convenience)
    let mut x = ComplexVector::new(ndim);
    ComplexLinSolver::compute(Genie::Umfpack, &mut x, &mat, &rhs, None)?;
    println!("x =\n{}", x);

    let correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
    complex_vec_approx_eq(&x, correct, 1e-14);
    Ok(())
}
