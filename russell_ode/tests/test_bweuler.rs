use russell_lab::{approx_eq, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_bweuler_hairer_wanner_eq1() {
    // get ODE system
    let (system, x0, mut y0, mut args, y_fn_x) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();

    // final x
    let x1 = 1.5;

    // set configuration parameters
    let params = Params::new(Method::BwEuler);
    let mut solver = OdeSolver::new(params, system).unwrap();

    // solve the ODE system
    let h_equal = Some(1.875 / 50.0);
    solver.solve(&mut y0, x0, x1, h_equal, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with a previous implementation
    approx_eq(y0[0], 0.09060476604187756, 1e-15);
    assert_eq!(stat.h_accepted, h_equal.unwrap());

    // compare with the analytical solution
    let mut y1_correct = Vector::new(ndim);
    y_fn_x(&mut y1_correct, x1, &mut args);
    approx_eq(y0[0], y1_correct[0], 5e-5);

    // print and check statistics
    println!("{}", stat);
    assert_eq!(stat.n_function, 80);
    assert_eq!(stat.n_jacobian, 40);
    assert_eq!(stat.n_factor, 40);
    assert_eq!(stat.n_lin_sol, 40);
    assert_eq!(stat.n_steps, 40);
    assert_eq!(stat.n_accepted, 40);
    assert_eq!(stat.n_rejected, 0);
    assert_eq!(stat.n_iterations, 2);
    assert_eq!(stat.n_iterations_max, 2);
}

#[test]
fn test_bweuler_hairer_wanner_eq1_num_jac() {
    // get ODE system
    let (system, x0, mut y0, mut args, y_fn_x) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();

    // final x
    let x1 = 1.5;

    // set configuration parameters
    let mut params = Params::new(Method::BwEuler);
    params.newton.use_numerical_jacobian = true;

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    let h_equal = Some(1.875 / 50.0);
    solver.solve(&mut y0, x0, x1, h_equal, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with a previous implementation
    approx_eq(y0[0], 0.09060476598021044, 1e-11);
    assert_eq!(stat.h_accepted, h_equal.unwrap());

    // compare with the analytical solution
    let mut y1_correct = Vector::new(ndim);
    y_fn_x(&mut y1_correct, x1, &mut args);
    approx_eq(y0[0], y1_correct[0], 5e-5);

    // print and check statistics
    println!("{}", stat);
    assert_eq!(stat.n_function, 120);
    assert_eq!(stat.n_jacobian, 40);
    assert_eq!(stat.n_factor, 40);
    assert_eq!(stat.n_lin_sol, 40);
    assert_eq!(stat.n_steps, 40);
    assert_eq!(stat.n_accepted, 40);
    assert_eq!(stat.n_rejected, 0);
    assert_eq!(stat.n_iterations, 2);
    assert_eq!(stat.n_iterations_max, 2);
}

#[test]
fn test_bweuler_hairer_wanner_eq1_modified_newton() {
    // get ODE system
    let (system, x0, mut y0, mut args, y_fn_x) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();

    // final x
    let x1 = 1.5;

    // set configuration parameters
    let mut params = Params::new(Method::BwEuler);
    params.bweuler.use_modified_newton = true;

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    let h_equal = Some(1.875 / 50.0);
    solver.solve(&mut y0, x0, x1, h_equal, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with a previous implementation
    approx_eq(y0[0], 0.09060476604187756, 1e-15);
    assert_eq!(stat.h_accepted, h_equal.unwrap());

    // compare with the analytical solution
    let mut y1_correct = Vector::new(ndim);
    y_fn_x(&mut y1_correct, x1, &mut args);
    approx_eq(y0[0], y1_correct[0], 5e-5);

    // print and check statistics
    println!("{}", stat);
    assert_eq!(stat.n_function, 80);
    assert_eq!(stat.n_jacobian, 1);
    assert_eq!(stat.n_factor, 1);
    assert_eq!(stat.n_lin_sol, 40);
    assert_eq!(stat.n_steps, 40);
    assert_eq!(stat.n_accepted, 40);
    assert_eq!(stat.n_rejected, 0);
    assert_eq!(stat.n_iterations, 2);
    assert_eq!(stat.n_iterations_max, 2);
}
