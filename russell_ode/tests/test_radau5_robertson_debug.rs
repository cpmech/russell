use russell_lab::{approx_eq, format_fortran};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_radau5_robertson_debug() {
    // get get ODE system
    let (system, x0, mut y0, mut args) = Samples::robertson();

    // final x
    let x1 = 0.3;

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-6;
    params.set_tolerances(1e-8, 1e-2, None).unwrap();
    params.debug = true;

    // solve the ODE system
    let mut solver = OdeSolver::new(params, &system).unwrap();
    solver.solve(&mut y0, x0, x1, None, None, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with radau5.f
    approx_eq(y0[0], 9.886740138499884E-01, 1e-15);
    approx_eq(y0[1], 3.447720471782070E-05, 1e-15);
    approx_eq(y0[2], 1.129150894529390E-02, 1e-15);
    approx_eq(stat.h_accepted, 8.160578540333708E-01, 1e-10);

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}{}", format_fortran(y0[0]), format_fortran(y0[1]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 88);
    assert_eq!(stat.n_jacobian, 8);
    assert_eq!(stat.n_factor, 15);
    assert_eq!(stat.n_lin_sol, 24);
    assert_eq!(stat.n_steps, 17);
    assert_eq!(stat.n_accepted, 15);
    assert_eq!(stat.n_rejected, 1);
    assert_eq!(stat.n_iterations_max, 2);
}
