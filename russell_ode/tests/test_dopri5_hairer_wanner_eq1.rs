use russell_lab::{approx_eq, format_fortran, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_dopri5_hairer_wanner_eq1() {
    // get ODE system
    let (system, x0, mut y0, mut args, y_fn_x) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();

    // final x
    let x1 = 1.5;

    // set configuration parameters
    let mut params = Params::new(Method::DoPri5);
    params.step.h_ini = 1e-4;

    // allocate the solver
    let mut solver = OdeSolver::new(params, system).unwrap();

    // enable dense output
    solver
        .enable_output()
        .set_dense_h_out(0.1)
        .unwrap()
        .set_dense_recording(&[0]);

    // solve the ODE system
    solver.solve(&mut y0, x0, x1, None, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with dopri5.f
    approx_eq(y0[0], 9.063921649310544E-02, 1e-13);

    // compare with the analytical solution
    let mut y1_correct = Vector::new(ndim);
    y_fn_x(&mut y1_correct, x1, &mut args);
    approx_eq(y0[0], y1_correct[0], 4e-5);

    // print dense output
    let n_dense = solver.out_dense_x().len();
    for i in 0..n_dense {
        println!(
            "x ={:6.2}, y ={}",
            solver.out_dense_x()[i],
            format_fortran(solver.out_dense_y(0)[i]),
        );
    }

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}", format_fortran(y0[0]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 235);
    assert_eq!(stat.n_steps, 39);
    assert_eq!(stat.n_accepted, 39);
    assert_eq!(stat.n_rejected, 0);
}
