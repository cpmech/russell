use russell_lab::format_fortran;
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_radau5_robertson_small_h() {
    // get get ODE system
    let (system, mut data, mut args) = Samples::robertson();

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-6;
    params.debug = true;

    // this will cause h to become too small
    params.set_tolerances(1e-2, 1e-2, None).unwrap();

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    let res = solver.solve(&mut data.y0, data.x0, data.x1, None, None, &mut args);
    assert_eq!(res.err(), Some("the stepsize becomes too small"));
    println!("ERROR: THE STEPSIZE BECOMES TOO SMALL");

    // get statistics
    let stat = solver.bench();

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}{}", format_fortran(data.y0[0]), format_fortran(data.y0[1]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 520);
    assert_eq!(stat.n_jacobian, 57);
    assert_eq!(stat.n_factor, 75);
    assert_eq!(stat.n_lin_sol, 153);
    assert_eq!(stat.n_steps, 75);
    assert_eq!(stat.n_accepted, 60);
    assert_eq!(stat.n_rejected, 4);
    assert_eq!(stat.n_iterations_max, 4);
}
