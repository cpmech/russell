use russell_lab::{approx_eq, format_fortran, Vector};
use russell_ode::{Method, OdeSolver, Output, Params, Samples};

#[test]
fn test_radau5_hairer_wanner_eq1() {
    // get ODE system
    let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-4;

    // enable dense output with 0.2 spacing
    let mut out = Output::new();
    out.enable_dense(0.2, &[0]).unwrap();

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)
        .unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with radau5.f
    approx_eq(data.y0[0], 9.068021382386648E-02, 1e-15);
    approx_eq(stat.h_accepted, 1.272673814374611E+00, 1e-12);

    // compare with the analytical solution
    let mut analytical = data.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, data.x1);
    approx_eq(data.y0[0], y1_correct[0], 3e-5);

    // print dense output
    let n_dense = out.dense_step_index.len();
    for i in 0..n_dense {
        println!(
            "step ={:>4}, x ={:5.2}, y ={}",
            out.dense_step_index[i],
            out.dense_x[i],
            format_fortran(out.dense_y.get(&0).unwrap()[i])
        )
    }

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}", format_fortran(data.y0[0]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 67);
    assert_eq!(stat.n_jacobian, 1);
    assert_eq!(stat.n_factor, 13);
    assert_eq!(stat.n_lin_sol, 17);
    assert_eq!(stat.n_steps, 15);
    assert_eq!(stat.n_accepted, 15);
    assert_eq!(stat.n_rejected, 0);
    assert_eq!(stat.n_iterations, 1);
    assert_eq!(stat.n_iterations_max, 2);
}