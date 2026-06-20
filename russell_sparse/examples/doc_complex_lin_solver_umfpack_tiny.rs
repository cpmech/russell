use russell_lab::{ComplexVector, complex_vec_approx_eq, cpx};
use russell_sparse::StrError;
use russell_sparse::prelude::*;

fn main() -> Result<(), StrError> {
    let ndim = 3;
    let nnz = 7;

    let mut solver = ComplexLinSolver::new(Genie::Umfpack)?;

    // allocate the coefficient matrix
    //   ┌                      ┐
    //   │  2+1i  -1-1i     0   │
    //   │ -1-1i   2+2i  -1+1i  │
    //   │   0    -1+1i   2-1i  │
    //   └                      ┘
    let mut coo = ComplexCooMatrix::new(ndim, ndim, nnz, Sym::No)?;
    coo.put(0, 0, cpx!(2.0, 1.0))?;
    coo.put(0, 1, cpx!(-1.0, -1.0))?;
    coo.put(1, 0, cpx!(-1.0, -1.0))?;
    coo.put(1, 1, cpx!(2.0, 2.0))?;
    coo.put(1, 2, cpx!(-1.0, 1.0))?;
    coo.put(2, 1, cpx!(-1.0, 1.0))?;
    coo.put(2, 2, cpx!(2.0, -1.0))?;

    // call factorize
    solver.actual.factorize(&coo, None)?;

    // right-hand side vector
    let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);

    // calculate solution (first rhs)
    let mut x = ComplexVector::new(ndim);
    solver.actual.solve(&mut x, &rhs, false)?;
    let correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
    complex_vec_approx_eq(&x, correct, 1e-14);

    // calculate solution again (second rhs, factorized once)
    let rhs2 = ComplexVector::from(&[cpx!(-6.0, 6.0), cpx!(4.0, -4.0), cpx!(18.0, 14.0)]);
    solver.actual.solve(&mut x, &rhs2, false)?;
    let correct = &[cpx!(2.0, 2.0), cpx!(4.0, -4.0), cpx!(6.0, 6.0)];
    complex_vec_approx_eq(&x, correct, 1e-14);
    Ok(())
}
