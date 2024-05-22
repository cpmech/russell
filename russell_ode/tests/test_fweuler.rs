use russell_lab::{approx_eq, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_fweuler_hairer_wanner_eq1() {
    // get ODE system
    let (system, x0, mut y0, mut args, y_fn_x) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();

    // final x
    let x1 = 1.5;

    // set configuration parameters
    let params = Params::new(Method::FwEuler);

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    let h_equal = Some(1.875 / 50.0);
    solver.solve(&mut y0, x0, x1, h_equal, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with a previous implementation
    approx_eq(y0[0], 0.08589790706616637, 1e-15);
    assert_eq!(stat.h_accepted, h_equal.unwrap());

    // compare with the analytical solution
    let mut y1_correct = Vector::new(ndim);
    y_fn_x(&mut y1_correct, x1, &mut args);
    approx_eq(y0[0], y1_correct[0], 0.004753);

    // print and check statistics
    println!("{}", stat);
    assert_eq!(stat.n_function, 40);
    assert_eq!(stat.n_jacobian, 0);
    assert_eq!(stat.n_factor, 0);
    assert_eq!(stat.n_lin_sol, 0);
    assert_eq!(stat.n_steps, 40);
    assert_eq!(stat.n_accepted, 40);
    assert_eq!(stat.n_rejected, 0);
    assert_eq!(stat.n_iterations, 0);
    assert_eq!(stat.n_iterations_max, 0);
}
