use russell_lab::{ComplexVector, complex_vec_approx_eq, cpx};
use russell_sparse::StrError;
use russell_sparse::prelude::*;

fn main() -> Result<(), StrError> {
    let ndim = 3;
    let nnz = 7;

    let mut klu = ComplexSolverKLU::new()?;

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
    klu.factorize(&coo, None)?;

    // right-hand side vector
    let b = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);

    // calculate the solution
    let mut x = ComplexVector::new(ndim);
    klu.solve(&mut x, &b, false)?;
    println!("x =\n{}", x);

    // check the result
    let correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
    complex_vec_approx_eq(&x, correct, 1e-14);
    println!("Complex KLU solved the tiny system successfully!");

    Ok(())
}
