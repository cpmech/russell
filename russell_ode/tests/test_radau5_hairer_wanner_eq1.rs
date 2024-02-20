use russell_lab::{approx_eq, format_fortran, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_radau5_hairer_wanner_eq1() {
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let mut params = Params::new(Method::Radau5);
    params.h_ini = 1e-4;
    params.radau5.logging = true;
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, None, &mut args)
        .unwrap();
    let mut analytical = data.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, data.x1);
    approx_eq(data.y0[0], 9.068021382386648E-02, 1e-15);
    approx_eq(data.y0[0], y1_correct[0], 3e-5);
    let b = solver.bench();
    println!("{}", b.summary());
    println!("y ={}", format_fortran(data.y0[0]));
    assert_eq!(b.n_function, 67);
    assert_eq!(b.n_jacobian, 1);
    assert_eq!(b.n_factor, 13); // << new
    assert_eq!(b.n_lin_sol, 17); // << new
    assert_eq!(b.n_steps, 15);
    assert_eq!(b.n_accepted, 15);
    assert_eq!(b.n_rejected, 0);
    assert_eq!(b.n_iterations, 1);
    assert_eq!(b.n_iterations_max, 2);
    approx_eq(b.h_optimal, 0.721202575813698, 0.0032);
}
