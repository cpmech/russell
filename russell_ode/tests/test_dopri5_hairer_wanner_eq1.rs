use russell_lab::{approx_eq, format_fortran, Vector};
use russell_ode::{Method, OdeSolver, Output, Params, Samples};

#[test]
fn test_dopri5_hairer_wanner_eq1() {
    // get ODE system
    let (system, mut data, mut args, y_fn_x) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();

    // set configuration parameters
    let mut params = Params::new(Method::DoPri5);
    params.step.h_ini = 1e-4;

    // enable dense output with 0.1 spacing
    let mut out = Output::new();
    out.set_dense_recording(true, 0.1, &[0]).unwrap();

    // solve the ODE system
    let mut solver = OdeSolver::new(params, &system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)
        .unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with dopri5.f
    approx_eq(data.y0[0], 9.063921649310544E-02, 1e-13);

    // compare with the analytical solution
    let mut y1_correct = Vector::new(ndim);
    y_fn_x(&mut y1_correct, data.x1, &mut args);
    approx_eq(data.y0[0], y1_correct[0], 4e-5);

    // print dense output
    let n_dense = out.dense_step_index.len();
    for i in 0..n_dense {
        println!(
            "step ={:>4}, x ={:6.2}, y ={}",
            out.dense_step_index[i],
            out.dense_x[i],
            format_fortran(out.dense_y.get(&0).unwrap()[i]),
        );
    }

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}", format_fortran(data.y0[0]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 235);
    assert_eq!(stat.n_steps, 39);
    assert_eq!(stat.n_accepted, 39);
    assert_eq!(stat.n_rejected, 0);
}
