use russell_lab::*;
use russell_sparse::prelude::*;
use russell_sparse::StrError;

#[test]
fn test_complex_umfpack() -> Result<(), StrError> {
    let n = 10;
    let d = (n as f64) / 10.0;
    let mut coo = CooMatrix::new(n, n, n, Sym::No)?;
    let mut x_correct = Vector::new(n);
    let mut rhs = Vector::new(n);
    for k in 0..n {
        // put diagonal entries
        let akk = 10.0 + (k as f64) * d;
        coo.put(k, k, akk)?;
        // let the exact solution be k + 0.5i
        x_correct[k] = k as f64;
        // generate RHS to match solution
        rhs[k] = akk * x_correct[k];
    }
    // println!("a =\n{}", coo.as_dense());
    // println!("x =\n{}", x_correct);
    // println!("b =\n{}", rhs);
    let mut x = Vector::new(n);
    let mut solver = SolverUMFPACK::new()?;
    solver.factorize(&coo, None)?;
    solver.solve(&mut x, &rhs, false)?;
    vec_approx_eq(&x, &x_correct, 1e-14);
    Ok(())
}
