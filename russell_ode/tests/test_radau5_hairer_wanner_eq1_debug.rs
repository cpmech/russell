use russell_lab::{approx_eq, format_fortran, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_radau5_hairer_wanner_eq1_debug() {
    // get ODE system
    let (system, x0, mut y0, mut args, y_fn_x) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();

    // final x
    let x1 = 1.5;

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-4;
    params.debug = true;

    // allocate the solver
    let mut solver = OdeSolver::new(params, system).unwrap();

    // solve the ODE system
    solver.solve(&mut y0, x0, x1, None, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with radau5.f
    approx_eq(y0[0], 9.068021382386648E-02, 1e-15);
    approx_eq(stat.h_accepted, 1.272673814374611E+00, 1e-12);

    // compare with the analytical solution
    let mut y1_correct = Vector::new(ndim);
    y_fn_x(&mut y1_correct, x1, &mut args);
    approx_eq(y0[0], y1_correct[0], 3e-5);

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}", format_fortran(y0[0]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 67);
    assert_eq!(stat.n_jacobian, 1);
    assert_eq!(stat.n_factor, 13);
    assert_eq!(stat.n_lin_sol, 17);
    assert_eq!(stat.n_steps, 15);
    assert_eq!(stat.n_accepted, 15);
    assert_eq!(stat.n_rejected, 0);
    assert_eq!(stat.n_iterations, 1);
    assert_eq!(stat.n_iterations_max, 2);
}
