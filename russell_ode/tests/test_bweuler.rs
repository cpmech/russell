use russell_lab::{approx_eq, Vector};
use russell_ode::{no_dense_output, no_step_output, Method, OdeSolver, Params, Samples};

#[test]
fn test_bweuler_hairer_wanner_eq1() {
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let params = Params::new(Method::BwEuler);
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(
            &mut data.y0,
            data.x0,
            data.x1,
            data.h_equal,
            &mut args,
            no_step_output,
            no_dense_output,
        )
        .unwrap();
    let mut analytical = data.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, data.x1);
    approx_eq(data.y0[0], 0.09060476604187756, 1e-15);
    approx_eq(data.y0[0], y1_correct[0], 5e-5);

    let b = solver.bench();
    println!("{}", b);
    assert_eq!(b.n_function, 80);
    assert_eq!(b.n_jacobian, 40);
    assert_eq!(b.n_steps, 40);
    assert_eq!(b.n_accepted, 40);
    assert_eq!(b.n_rejected, 0);
    assert_eq!(b.n_iterations, 2);
    assert_eq!(b.n_iterations_max, 2);
    assert_eq!(b.h_optimal, data.h_equal.unwrap());
}

#[test]
fn test_bweuler_hairer_wanner_eq1_num_jac() {
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let mut params = Params::new(Method::BwEuler);
    params.bweuler.use_numerical_jacobian = true;
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(
            &mut data.y0,
            data.x0,
            data.x1,
            data.h_equal,
            &mut args,
            no_step_output,
            no_dense_output,
        )
        .unwrap();
    let mut analytical = data.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, data.x1);

    approx_eq(data.y0[0], 0.09060476598021044, 1e-11);
    approx_eq(data.y0[0], y1_correct[0], 5e-5);

    let b = solver.bench();
    println!("{}", b);
    assert_eq!(b.n_function, 120);
    assert_eq!(b.n_jacobian, 40);
    assert_eq!(b.n_steps, 40);
    assert_eq!(b.n_accepted, 40);
    assert_eq!(b.n_rejected, 0);
    assert_eq!(b.n_iterations, 2);
    assert_eq!(b.n_iterations_max, 2);
    assert_eq!(b.h_optimal, data.h_equal.unwrap());
}

#[test]
fn test_bweuler_hairer_wanner_eq1_modified_newton() {
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let mut params = Params::new(Method::BwEuler);
    params.bweuler.use_modified_newton = true;
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(
            &mut data.y0,
            data.x0,
            data.x1,
            data.h_equal,
            &mut args,
            no_step_output,
            no_dense_output,
        )
        .unwrap();
    let mut analytical = data.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, data.x1);

    approx_eq(data.y0[0], 0.09060476604187756, 1e-15);
    approx_eq(data.y0[0], y1_correct[0], 5e-5);

    let b = solver.bench();
    println!("{}", b);
    assert_eq!(b.n_function, 80);
    assert_eq!(b.n_jacobian, 1);
    assert_eq!(b.n_steps, 40);
    assert_eq!(b.n_accepted, 40);
    assert_eq!(b.n_rejected, 0);
    assert_eq!(b.n_iterations, 2);
    assert_eq!(b.n_iterations_max, 2);
    assert_eq!(b.h_optimal, data.h_equal.unwrap());
}
