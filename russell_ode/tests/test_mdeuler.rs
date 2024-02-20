use russell_lab::{approx_eq, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_mdeuler_hairer_wanner_eq1() {
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let mut params = Params::new(Method::MdEuler);
    params.h_ini = 1e-4;
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, None, &mut args)
        .unwrap();

    let b = solver.bench();
    let mut analytical = data.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, data.x1);
    println!("{}", b);
    approx_eq(data.y0[0], 0.09062475637905158, 1e-16);
    approx_eq(data.y0[0], y1_correct[0], 1e-4);
    assert_eq!(b.n_function, 424);
    assert_eq!(b.n_jacobian, 0);
    assert_eq!(b.n_steps, 212);
    assert_eq!(b.n_accepted, 212);
    assert_eq!(b.n_rejected, 0);
    assert_eq!(b.n_iterations, 0);
    assert_eq!(b.n_iterations_max, 0);
    approx_eq(b.h_optimal, 0.015661248295711694, 1e-13);
}
