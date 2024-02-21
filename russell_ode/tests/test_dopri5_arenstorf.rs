use russell_lab::vec_approx_eq;
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_dopri5_arenstorf() {
    // get ODE system
    let (system, mut data, mut args) = Samples::arenstorf();

    // set configuration parameters
    let mut params = Params::new(Method::DoPri5);
    params.h_ini = 1e-4;
    params.set_tolerances(1e-7, 1e-7).unwrap();
    let mut solver = OdeSolver::new(params, system).unwrap();

    // solve the ODE system
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, None, &mut args)
        .unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with a previous implementation
    let y_ref = [
        0.9940021704037415,
        9.040893396741956e-06,
        0.0014597586885445324,
        -2.0012455157289244,
    ];
    vec_approx_eq(data.y0.as_data(), &y_ref, 1e-9);

    // print and check statistics
    println!("{}", stat);
    assert_eq!(stat.n_function, 1429);
    assert_eq!(stat.n_jacobian, 0);
    assert_eq!(stat.n_steps, 238);
    assert_eq!(stat.n_accepted, 217);
    assert_eq!(stat.n_rejected, 21);
    assert_eq!(stat.n_iterations, 0);
    assert_eq!(stat.n_iterations_max, 0);
}
