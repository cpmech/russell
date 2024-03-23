use russell_lab::{approx_eq, StrError};
use russell_ode::prelude::*;

fn main() -> Result<(), StrError> {
    // get get ODE system
    let (system, mut data, mut args) = Samples::amplifier1t();

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-6;
    params.set_tolerances(1e-4, 1e-4, None)?;

    // enable output of accepted steps
    let mut out = Output::new();
    out.enable_dense(0.001, &[0, 4])?;

    // solve the ODE system
    let mut solver = OdeSolver::new(params, &system)?;
    solver.solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)?;

    // compare with radau5.f
    approx_eq(data.y0[0], -2.226517868073645E-02, 1e-10);
    approx_eq(data.y0[1], 3.068700099735197E+00, 1e-10);
    approx_eq(data.y0[2], 2.898340496450958E+00, 1e-9);
    approx_eq(data.y0[3], 2.033525366489690E+00, 1e-7);
    approx_eq(data.y0[4], -2.269179823457655E+00, 1e-7);

    // print stat
    let stat = solver.bench();
    println!("{}", stat);
    Ok(())
}
