use russell_lab::{approx_eq, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_fweuler_hairer_wanner_eq1() {
    // get ODE system
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();

    // set configuration parameters
    let params = Params::new(Method::FwEuler);

    // solve the ODE system
    let mut solver = OdeSolver::new(params, &system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, data.h_equal, None, &mut args)
        .unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with a previous implementation
    approx_eq(data.y0[0], 0.08589790706616637, 1e-15);
    assert_eq!(stat.h_accepted, data.h_equal.unwrap());

    // compare with the analytical solution
    let analytical = data.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, data.x1);
    approx_eq(data.y0[0], y1_correct[0], 0.004753);

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
