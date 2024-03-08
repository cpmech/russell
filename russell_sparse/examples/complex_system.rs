use num_complex::Complex64;
use russell_lab::{complex_vec_approx_eq, cpx, ComplexVector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn solve(genie: Genie) -> Result<(), StrError> {
    // Given the matrix of complex numbers:
    //
    //     ┌                                                    ┐
    //     │  19.73    12.11-i      5i        0          0      │
    //     │  -0.51i   32.3+7i    23.07       i          0      │
    // A = │    0      -0.51i    70+7.3i     3.95    19+31.83i  │
    //     │    0        0        1+1.1i    50.17      45.51    │
    //     │    0        0          0      -9.351i       55     │
    //     └                                                    ┘
    //
    // and the vector:
    //
    //     ┌                   ┐
    //     │    77.38+8.82i    │
    //     │   157.48+19.8i    │
    // b = │  1175.62+20.69i   │
    //     │   912.12-801.75i  │
    //     │     550-1060.4i   │
    //     └                   ┘
    //
    // find x such that:
    //
    //         A.x = b
    //
    // The solution is approximately:
    //
    //     ┌              ┐
    //     │     3.3-i    │
    //     │    1+0.17i   │
    // x = │      5.5     │
    //     │       9      │
    //     │  10-17.75i   │
    //     └              ┘

    // constants
    let ndim = 5; // number of rows = number of columns
    let nnz = 16; // number of non-zero values, including duplicates

    // input matrix in Complex Triplet format
    let mut coo = ComplexSparseMatrix::new_coo(ndim, ndim, nnz, None)?;

    // first column
    coo.put(0, 0, cpx!(19.73, 0.00))?;
    coo.put(1, 0, cpx!(0.00, -0.51))?;

    // second column
    coo.put(0, 1, cpx!(12.11, -1.00))?;
    coo.put(1, 1, cpx!(32.30, 7.00))?;
    coo.put(2, 1, cpx!(0.00, -0.51))?;

    // third column
    coo.put(0, 2, cpx!(0.00, 5.0))?;
    coo.put(1, 2, cpx!(23.07, 0.0))?;
    coo.put(2, 2, cpx!(70.00, 7.3))?;
    coo.put(3, 2, cpx!(1.00, 1.1))?;

    // fourth column
    coo.put(1, 3, cpx!(0.00, 1.000))?;
    coo.put(2, 3, cpx!(3.95, 0.000))?;
    coo.put(3, 3, cpx!(50.17, 0.000))?;
    coo.put(4, 3, cpx!(0.00, -9.351))?;

    // fifth column
    coo.put(2, 4, cpx!(19.00, 31.83))?;
    coo.put(3, 4, cpx!(45.51, 0.00))?;
    coo.put(4, 4, cpx!(55.00, 0.00))?;

    // right-hand-side
    let b = ComplexVector::from(&[
        cpx!(77.38, 8.82),
        cpx!(157.48, 19.8),
        cpx!(1175.62, 20.69),
        cpx!(912.12, -801.75),
        cpx!(550.00, -1060.4),
    ]);

    // allocate solver
    let mut solver = ComplexLinSolver::new(genie)?;

    // parameters
    let mut params = LinSolParams::new();
    params.verbose = false;

    // call factorize and solve
    let mut x = ComplexVector::new(ndim);
    solver.actual.factorize(&mut coo, Some(params))?;
    solver.actual.solve(&mut x, &coo, &b, false)?;
    println!("x =\n{}", x);

    // check
    let x_correct = &[
        cpx!(3.3, -1.00),
        cpx!(1.0, 0.17),
        cpx!(5.5, 0.00),
        cpx!(9.0, 0.00),
        cpx!(10.0, -17.75),
    ];
    complex_vec_approx_eq(x.as_data(), x_correct, 1e-3);
    Ok(())
}

fn main() -> Result<(), StrError> {
    println!("MUMPS =====================================");
    solve(Genie::Mumps)?;

    println!("\nUMFPACK =====================================");
    solve(Genie::Umfpack)?;

    Ok(())
}
