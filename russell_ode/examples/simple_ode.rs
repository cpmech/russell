use russell_lab::{format_scientific, vec_max_abs_diff};
use russell_lab::{StrError, Vector};
use russell_ode::prelude::*;

fn main() -> Result<(), StrError> {
    // ODE system
    // dy/dx = x + y    with    y(0) = 0
    let ndim = 1;
    let system = System::new(
        ndim,
        |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = x + y[0];
            Ok(())
        },
        no_jacobian,
        HasJacobian::No,
        None,
        None,
    );

    // solver
    let params = Params::new(Method::DoPri8);
    let mut solver = OdeSolver::new(params, &system)?;

    // initial values
    let x = 0.0;
    let mut y = Vector::from(&[0.0]);

    // solve from x = 0 to x = 1
    let x1 = 1.0;
    let mut args = 0;
    solver.solve(&mut y, x, x1, None, None, &mut args)?;
    println!("y =\n{}", y);

    // check the results
    let y_ana = Vector::from(&[f64::exp(x1) - x1 - 1.0]);
    let (_, error) = vec_max_abs_diff(&y, &y_ana)?;
    println!("error = {}", format_scientific(error, 8, 2));
    assert!(error < 1e-8);

    // print stats
    println!("{}", solver.bench());
    Ok(())
}
