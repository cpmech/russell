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
    assert_eq!(b.n_factorization, 40);
    assert_eq!(b.n_iterations_last, 2);
    assert_eq!(b.n_iterations_max, 2);
    assert_eq!(b.h_optimal, control.h_equal.unwrap());
}
