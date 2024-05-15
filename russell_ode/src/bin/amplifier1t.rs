use russell_lab::{approx_eq, StrError};
use russell_ode::prelude::*;

fn main() -> Result<(), StrError> {
    // get ODE system
    let (system, x0, mut y0, mut args) = Samples::amplifier1t();

    // final x
    let x1 = 0.05;

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-6;
    params.set_tolerances(1e-4, 1e-4, None)?;

    // allocate the solver
    let mut solver = OdeSolver::new(params, &system)?;

    // enable dense output
    solver.enable_output().set_dense_recording(true, 0.001, &[0, 4])?;

    // solve the ODE system
    let y = &mut y0;
    solver.solve(y, x0, x1, None, &mut args)?;

    // compare with radau5.f
    approx_eq(y[0], -2.226517868073645E-02, 1e-10);
    approx_eq(y[1], 3.068700099735197E+00, 1e-10);
    approx_eq(y[2], 2.898340496450958E+00, 1e-9);
    approx_eq(y[3], 2.033525366489690E+00, 1e-7);
    approx_eq(y[4], -2.269179823457655E+00, 1e-7);

    // print stat
    let stat = solver.stats();
    println!("{}", stat);
    Ok(())
}
