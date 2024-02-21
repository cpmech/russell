use russell_lab::{approx_eq, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_dopri5_hairer_wanner_eq1() {
    // get ODE system
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();

    // set configuration parameters
    let mut params = Params::new(Method::DoPri5);
    params.h_ini = 1e-4;
    params.erk.m_max = 5.0;

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, None, &mut args)
        .unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with a previous implementation
    approx_eq(data.y0[0], 0.09061967747782482, 1e-15);

    // compare with the analytical solution
    let mut analytical = data.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, data.x1);
    approx_eq(data.y0[0], y1_correct[0], 4e-5);

    // print and check statistics
    println!("{}", stat);
    assert_eq!(stat.n_function, 241);
    assert_eq!(stat.n_jacobian, 0);
    assert_eq!(stat.n_factor, 0);
    assert_eq!(stat.n_lin_sol, 0);
    assert_eq!(stat.n_steps, 40);
    assert_eq!(stat.n_accepted, 40);
    assert_eq!(stat.n_rejected, 0);
    assert_eq!(stat.n_iterations, 0);
    assert_eq!(stat.n_iterations_max, 0);
}
