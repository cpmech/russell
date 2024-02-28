use russell_lab::{approx_eq, format_fortran, vec_approx_eq, Vector};
use russell_ode::{Method, OdeSolver, Output, Params, Samples};

#[test]
fn test_dopri8_van_der_pol_debug() {
    // get get ODE system
    const EPS: f64 = 0.003;
    let (system, _, mut args) = Samples::van_der_pol(Some(EPS), false);

    // set configuration parameters
    let mut params = Params::new(Method::DoPri8);
    params.step.h_ini = 1e-4;
    params.set_tolerances(1e-3, 1e-3, None).unwrap();
    params.debug = true;
    params.stiffness.skip_first_n_accepted_step = 0;
    params.stiffness.enabled = true;
    params.stiffness.stop_with_error = false;
    params.stiffness.save_results = true;

    // output (to save stiff stations)
    let mut out = Output::new();

    // solve the ODE system
    let mut y0 = Vector::from(&[2.0, 0.0]);
    let x0 = 0.0;
    let x1 = 2.0;
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver.solve(&mut y0, x0, x1, None, Some(&mut out), &mut args).unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with dop853.f
    approx_eq(y0[0], 1.819907445729370E+00, 1e-9);
    approx_eq(y0[1], -7.866363461162956E-01, 1e-8);
    approx_eq(stat.h_accepted, 6.908420682852039E-03, 1e-8);

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}{}", format_fortran(y0[0]), format_fortran(y0[1]));
    println!("h ={}", format_fortran(stat.h_accepted));
    let n_extra = stat.n_accepted; // because dopri8 calls the function in the stiffness detection code
    assert_eq!(stat.n_function, 2802 - 2 + n_extra); // -2 when compared with dop853
    assert_eq!(stat.n_steps, 235);
    assert_eq!(stat.n_accepted, 215);
    assert_eq!(stat.n_rejected, 20);

    // check stiffness results
    assert_eq!(out.stiff_step_index, &[21, 109, 196]);
    vec_approx_eq(
        &out.stiff_x,
        &[1.141460201067603E-01, 9.881349085950795E-01, 1.853542594920877E+00],
        1e-8,
    );
}
