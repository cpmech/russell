use russell_lab::{approx_eq, Vector};
use russell_ode::{no_dense_output, no_step_output, Method, OdeParams, OdeSolver, Samples};

#[test]
fn test_bweuler_hairer_wanner_eq1() {
    let (system, mut control, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let params = OdeParams::new(Method::BwEuler, None, None);
    let mut solver = OdeSolver::new(&params, system).unwrap();
    solver
        .solve(
            &mut control.y0,
            control.x0,
            control.x1,
            control.h_equal,
            &mut args,
            no_step_output,
            no_dense_output,
        )
        .unwrap();
    let mut analytical = control.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, control.x1);
    approx_eq(control.y0[0], 0.09060476604187756, 1e-16);
    approx_eq(control.y0[0], y1_correct[0], 1e-4);

    let b = solver.bench();
    println!("{}", solver.bench());
    assert_eq!(b.n_function_eval, 80);
    assert_eq!(b.n_jacobian_eval, 40);
    assert_eq!(b.n_performed_steps, 40);
    assert_eq!(b.n_accepted_steps, 40);
    assert_eq!(b.n_rejected_steps, 0);
    assert_eq!(b.n_iterations_last, 2);
    assert_eq!(b.n_iterations_max, 2);
    assert_eq!(b.h_optimal, control.h_equal.unwrap());
}

#[test]
fn test_bweuler_hairer_wanner_eq1_num_jac() {
    let (mut system, mut control, mut args) = Samples::hairer_wanner_eq1();
    system.set_use_num_jacobian(true).unwrap();
    let ndim = system.get_ndim();
    let params = OdeParams::new(Method::BwEuler, None, None);
    let mut solver = OdeSolver::new(&params, system).unwrap();
    solver
        .solve(
            &mut control.y0,
            control.x0,
            control.x1,
            control.h_equal,
            &mut args,
            no_step_output,
            no_dense_output,
        )
        .unwrap();
    let mut analytical = control.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, control.x1);

    approx_eq(control.y0[0], 0.09060476587452296, 1e-10);
    approx_eq(control.y0[0], y1_correct[0], 1e-4);

    let b = solver.bench();
    println!("{}", solver.bench());
    assert_eq!(b.n_function_eval, 120);
    assert_eq!(b.n_jacobian_eval, 40);
    assert_eq!(b.n_performed_steps, 40);
    assert_eq!(b.n_accepted_steps, 40);
    assert_eq!(b.n_rejected_steps, 0);
    assert_eq!(b.n_iterations_last, 2);
    assert_eq!(b.n_iterations_max, 2);
    assert_eq!(b.h_optimal, control.h_equal.unwrap());
}

#[test]
fn test_bweuler_hairer_wanner_eq1_modified_newton() {
    let (system, mut control, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let mut params = OdeParams::new(Method::BwEuler, None, None);
    params.CteTg = true;
    let mut solver = OdeSolver::new(&params, system).unwrap();
    solver
        .solve(
            &mut control.y0,
            control.x0,
            control.x1,
            control.h_equal,
            &mut args,
            no_step_output,
            no_dense_output,
        )
        .unwrap();
    let mut analytical = control.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, control.x1);

    approx_eq(control.y0[0], 0.09060476604187756, 1e-15);
    approx_eq(control.y0[0], y1_correct[0], 1e-4);

    let b = solver.bench();
    println!("{}", solver.bench());
    assert_eq!(b.n_function_eval, 80);
    assert_eq!(b.n_jacobian_eval, 1);
    assert_eq!(b.n_performed_steps, 40);
    assert_eq!(b.n_accepted_steps, 40);
    assert_eq!(b.n_rejected_steps, 0);
    assert_eq!(b.n_iterations_last, 2);
    assert_eq!(b.n_iterations_max, 2);
    assert_eq!(b.h_optimal, control.h_equal.unwrap());
}
