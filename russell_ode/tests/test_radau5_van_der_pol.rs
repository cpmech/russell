use russell_lab::{approx_eq, format_fortran};
use russell_ode::{Method, OdeSolver, Output, Params, Samples};

#[test]
fn test_radau5_van_der_pol() {
    // get get ODE system
    const EPS: f64 = 1e-6;
    let (system, mut data, mut args) = Samples::van_der_pol(Some(EPS), false);

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-6;

    // enable dense output with 0.2 spacing
    let mut out = Output::new();
    out.enable_dense(0.2, &[0, 1]).unwrap();

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)
        .unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with radau5.f
    approx_eq(data.y0[0], 1.706163410178079E+00, 1e-14);
    approx_eq(data.y0[1], -8.927971289301175E-01, 1e-12);
    approx_eq(stat.h_accepted, 1.510987221365367E-01, 1.1e-8);

    // print dense output
    let n_dense = out.dense_step_index.len();
    for i in 0..n_dense {
        println!(
            "step ={:>4}, x ={:5.2}, y ={}{}",
            out.dense_step_index[i],
            out.dense_x[i],
            format_fortran(out.dense_y.get(&0).unwrap()[i]),
            format_fortran(out.dense_y.get(&1).unwrap()[i]),
        );
    }

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}{}", format_fortran(data.y0[0]), format_fortran(data.y0[1]));
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
