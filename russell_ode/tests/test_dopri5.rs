use russell_lab::{approx_eq, vec_approx_eq, Vector};
use russell_ode::{no_dense_output, no_step_output, Method, OdeSolver, Params, Samples};

#[test]
fn test_dopri5_hairer_wanner_eq1() {
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let params = Params::new(Method::DoPri5);
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(
            &mut data.y0,
            data.x0,
            data.x1,
            None,
            &mut args,
            no_step_output,
            no_dense_output,
        )
        .unwrap();

    let b = solver.bench();
    let mut analytical = data.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, data.x1);
    println!("{}", b);
    approx_eq(data.y0[0], 0.09061967747782482, 1e-15);
    approx_eq(data.y0[0], y1_correct[0], 1e-4);
    assert_eq!(b.n_function_eval, 241);
    assert_eq!(b.n_jacobian_eval, 0);
    assert_eq!(b.n_performed_steps, 40);
    assert_eq!(b.n_accepted_steps, 40);
    assert_eq!(b.n_rejected_steps, 0);
    assert_eq!(b.n_iterations_last, 0);
    assert_eq!(b.n_iterations_max, 0);
    approx_eq(b.h_optimal, 0.006336413253392292, 1e-12);
}

#[test]
fn test_dopri5_arenstorf() {
    let (system, mut data, mut args) = Samples::arenstorf();
    let mut params = Params::new(Method::DoPri5);
    params.set_tolerances(1e-7, 1e-7).unwrap();
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(
            &mut data.y0,
            data.x0,
            data.x1,
            None,
            &mut args,
            no_step_output,
            no_dense_output,
        )
        .unwrap();

    let b = solver.bench();
    println!("{}", b);
    assert_eq!(b.n_function_eval, 1429);
    assert_eq!(b.n_jacobian_eval, 0);
    assert_eq!(b.n_performed_steps, 238);
    assert_eq!(b.n_accepted_steps, 217);
    assert_eq!(b.n_rejected_steps, 21);
    assert_eq!(b.n_iterations_last, 0);
    assert_eq!(b.n_iterations_max, 0);
    let y_ref = [
        0.9940021704037415,
        9.040893396741956e-06,
        0.0014597586885445324,
        -2.0012455157289244,
    ];
    vec_approx_eq(data.y0.as_data(), &y_ref, 1e-9);
    approx_eq(b.h_optimal, 0.00043950148933546984, 1e-10);
}
