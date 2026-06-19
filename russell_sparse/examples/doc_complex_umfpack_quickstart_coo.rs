use russell_lab::{ComplexVector, complex_vec_approx_eq, cpx};
use russell_sparse::StrError;
use russell_sparse::prelude::*;

fn main() -> Result<(), StrError> {
    let ndim = 5;
    let nnz = 13;

    let mut umfpack = ComplexSolverUMFPACK::new()?;

    // allocate the coefficient matrix
    // ┌                                              ┐
    // │  2+0.5i   3+1i       0         0         0   │
    // │  3+1i       0      4+1i        0       6+1i  │
    // │     0    -1+0.5i  -3+1i     2+0.5i      0    │
    // │     0       0      1+0.5i      0         0   │
    // │     0     4+0.5i   2+0.5i      0      1+0.5i │
    // └                                              ┘
    let mut coo = ComplexCooMatrix::new(ndim, ndim, nnz, Sym::No)?;
    coo.put(0, 0, cpx!(1.0, 0.25))?; // << (0, 0, a00/2) duplicate
    coo.put(0, 0, cpx!(1.0, 0.25))?; // << (0, 0, a00/2) duplicate
    coo.put(1, 0, cpx!(3.0, 1.0))?;
    coo.put(0, 1, cpx!(3.0, 1.0))?;
    coo.put(2, 1, cpx!(-1.0, 0.5))?;
    coo.put(4, 1, cpx!(4.0, 0.5))?;
    coo.put(1, 2, cpx!(4.0, 1.0))?;
    coo.put(2, 2, cpx!(-3.0, 1.0))?;
    coo.put(3, 2, cpx!(1.0, 0.5))?;
    coo.put(4, 2, cpx!(2.0, 0.5))?;
    coo.put(2, 3, cpx!(2.0, 0.5))?;
    coo.put(1, 4, cpx!(6.0, 1.0))?;
    coo.put(4, 4, cpx!(1.0, 0.5))?;

    // parameters
    let mut params = LinSolParams::new();
    params.verbose = false;
    params.compute_determinant = true;

    // call factorize
    umfpack.factorize(&coo, Some(params))?;

    // allocate x and rhs
    let mut x = ComplexVector::new(ndim);
    let rhs = ComplexVector::from(&[
        cpx!(5.5, 10.5),
        cpx!(36.0, 54.0),
        cpx!(-9.0, 3.0),
        cpx!(1.5, 4.5),
        cpx!(14.0, 24.0),
    ]);

    // calculate the solution
    umfpack.solve(&mut x, &rhs, false)?;
    println!("x =\n{}", x);

    // check the results
    let correct = &[
        cpx!(1.0, 1.0),
        cpx!(2.0, 2.0),
        cpx!(3.0, 3.0),
        cpx!(4.0, 4.0),
        cpx!(5.0, 5.0),
    ];
    complex_vec_approx_eq(&x, correct, 1e-14);

    // analysis
    let mut stats = StatsLinSol::new();
    umfpack.update_stats(&mut stats);
    let (mx, my) = (stats.determinant.mantissa_real, stats.determinant.mantissa_imag);
    let ex = stats.determinant.exponent;
    println!("det(a) = {:?}", cpx!(mx, my) * cpx!(f64::powf(10.0, ex), 0.0));
    println!("rcond  = {:?}", stats.output.umfpack_rcond_estimate);
    Ok(())
}
