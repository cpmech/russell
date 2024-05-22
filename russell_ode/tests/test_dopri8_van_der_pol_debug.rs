use russell_lab::{approx_eq, array_approx_eq, format_fortran, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_dopri8_van_der_pol_debug() {
    // get get ODE system
    const EPS: f64 = 0.003;
    let (system, _, _, _, mut args) = Samples::van_der_pol(EPS, false);

    // set configuration parameters
    let mut params = Params::new(Method::DoPri8);
    params.step.h_ini = 1e-4;
    params.set_tolerances(1e-3, 1e-3, None).unwrap();
    params.debug = true;
    params.stiffness.skip_first_n_accepted_step = 0;
    params.stiffness.enabled = true;
    params.stiffness.stop_with_error = false;
    params.stiffness.save_results = true;

    // allocate the solver
    let mut solver = OdeSolver::new(params, system).unwrap();

    // output (to save stiff stations)
    solver.enable_output();

    // solve the ODE system
    let mut y0 = Vector::from(&[2.0, 0.0]);
    let x0 = 0.0;
    let x1 = 2.0;
    solver.solve(&mut y0, x0, x1, None, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

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
    assert_eq!(solver.out_stiff_step_index(), &[21, 109, 196]);
    array_approx_eq(
        &solver.out_stiff_x(),
        &[1.563905377322407E-02, 8.759592223459979E-01, 1.749270939102191E+00],
        1e-7,
    );
    println!("h·ρ = {:?}", solver.out_stiff_h_times_rho()[21]);
    println!("h·ρ = {:?}", solver.out_stiff_h_times_rho()[109]);
    println!("h·ρ = {:?}", solver.out_stiff_h_times_rho()[196]);
    let max_h_rho = params.stiffness.get_h_times_rho_max();
    assert_eq!(max_h_rho, 6.1);
    assert!(solver.out_stiff_h_times_rho()[0] < max_h_rho);
    assert!(solver.out_stiff_h_times_rho()[21] > max_h_rho);
    assert!(solver.out_stiff_h_times_rho()[109] > max_h_rho);
    assert!(solver.out_stiff_h_times_rho()[196] > max_h_rho);
    assert!(*solver.out_stiff_h_times_rho().last().unwrap() < max_h_rho);
}
