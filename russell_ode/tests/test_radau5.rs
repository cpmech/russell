use russell_lab::{approx_eq, Vector};
use russell_ode::{no_dense_output, no_step_output, Method, Params, Samples, Solver};

#[test]
fn test_radau5_hairer_wanner_eq1() {
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let params = Params::new(Method::Radau5);
    let mut solver = Solver::new(params, system).unwrap();
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
    approx_eq(data.y0[0], y1_correct[0], 1e-4);

    let b = solver.bench();
    println!("{}", b);
    assert_eq!(b.n_function_eval, 80);
    assert_eq!(b.n_jacobian_eval, 40);
    assert_eq!(b.n_performed_steps, 40);
    assert_eq!(b.n_accepted_steps, 40);
    assert_eq!(b.n_rejected_steps, 0);
    assert_eq!(b.n_iterations_last, 2);
    assert_eq!(b.n_iterations_max, 2);
    assert_eq!(b.h_optimal, data.h_equal.unwrap());
}
