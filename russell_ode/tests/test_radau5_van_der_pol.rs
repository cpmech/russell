use russell_lab::{approx_eq, format_fortran};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_radau5_van_der_pol() {
    // get get ODE system
    const EPS: f64 = 1e-6;
    let (system, x0, mut y0, x1, mut args) = Samples::van_der_pol(EPS, false);

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-6;

    // allocate the solver
    let mut solver = OdeSolver::new(params, &system).unwrap();

    // enable dense output
    solver
        .enable_output()
        .set_dense_h_out(0.2)
        .unwrap()
        .set_dense_recording(&[0, 1]);

    // solve the ODE system
    solver.solve(&mut y0, x0, x1, None, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with radau5.f
    approx_eq(y0[0], 1.706163410178079E+00, 1e-13);
    approx_eq(y0[1], -8.927971289301175E-01, 1e-12);
    approx_eq(stat.h_accepted, 1.510987221365367E-01, 1.2e-8);

    // print dense output
    let n_dense = solver.out_dense_x().len();
    for i in 0..n_dense {
        println!(
            "x ={:5.2}, y ={}{}",
            solver.out_dense_x()[i],
            format_fortran(solver.out_dense_y(0)[i]),
            format_fortran(solver.out_dense_y(1)[i]),
        );
    }

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}{}", format_fortran(y0[0]), format_fortran(y0[1]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 2248 + 1); // +1 because the fist step is a reject, thus initialize is called again
    assert_eq!(stat.n_jacobian, 162);
    assert_eq!(stat.n_factor, 253);
    assert_eq!(stat.n_lin_sol, 668);
    assert_eq!(stat.n_steps, 280);
    assert_eq!(stat.n_accepted, 242);
    assert_eq!(stat.n_rejected, 8);
    assert_eq!(stat.n_iterations, 2);
    assert_eq!(stat.n_iterations_max, 6);
}
