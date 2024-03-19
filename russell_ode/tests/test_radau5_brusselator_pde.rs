use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_radau5_brusselator_pde() {
    // get get ODE system
    let alpha = 2e-3;
    let npoint = 3;
    let (system, mut data, mut args) = Samples::brusselator_pde(alpha, npoint, false);

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-4;
    params.set_tolerances(1e-3, 1e-3, None).unwrap();

    // solve the ODE system
    let mut solver = OdeSolver::new(params, &system).unwrap();
    let x1 = 1.0;
    solver.solve(&mut data.y0, data.x0, x1, None, None, &mut args).unwrap();

    // get statistics
    let stat = solver.bench();
    println!("{}", stat);
}
