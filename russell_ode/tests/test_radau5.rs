use russell_lab::{approx_eq, Vector};
use russell_ode::{no_dense_output, no_step_output, Method, OdeSolver, Params, Samples};

#[test]
fn test_radau5_hairer_wanner_eq1() {
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let params = Params::new(Method::Radau5);
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
    let mut analytical = data.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, data.x1);
    approx_eq(data.y0[0], 0.09067973091719728, 1e-6);
    approx_eq(data.y0[0], y1_correct[0], 3e-5); // << todo
    let b = solver.bench();
    println!("{}", b);
    assert_eq!(b.n_function, 67);
    assert_eq!(b.n_jacobian, 1);
    assert_eq!(b.n_factor, 13); // << new
    assert_eq!(b.n_lin_sol, 17); // << new
    assert_eq!(b.n_steps, 15);
    assert_eq!(b.n_accepted, 15);
    assert_eq!(b.n_rejected, 0);
    assert_eq!(b.n_iterations, 1);
    assert_eq!(b.n_iterations_max, 2);
    approx_eq(b.h_optimal, 0.7212025758141315, 0.0032); // << todo
}
