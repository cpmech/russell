use russell_lab::{approx_eq, array_approx_eq, format_fortran, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_dopri5_van_der_pol_debug() {
    // get get ODE system
    const EPS: f64 = 0.003;
    let (system, _, _, _, mut args) = Samples::van_der_pol(EPS, false);

    // set configuration parameters
    let mut params = Params::new(Method::DoPri5);
    params.step.h_ini = 1e-4;
    params.set_tolerances(1e-3, 1e-3, None).unwrap();
    params.debug = true;
    params.stiffness.skip_first_n_accepted_step = 0;
    params.stiffness.enabled = true;
    params.stiffness.stop_with_error = false;
    params.stiffness.save_results = true;

    // allocate the solver
    let mut solver = OdeSolver::new(params, system).unwrap();

    // enable output (to save stiff stations)
    solver.enable_output();

    // solve the ODE system
    let mut y0 = Vector::from(&[2.0, 0.0]);
    let x0 = 0.0;
    let x1 = 2.0;

    solver.solve(&mut y0, x0, x1, None, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with dopri5.f
    approx_eq(y0[0], 1.820788982019278E+00, 1e-12);
    approx_eq(y0[1], -7.853646714272298E-01, 1e-12);
    approx_eq(stat.h_accepted, 4.190371271724428E-03, 1e-13);

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}{}", format_fortran(y0[0]), format_fortran(y0[1]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 2558 - 1); // -1 when compared with dopri5.f
    assert_eq!(stat.n_steps, 426);
    assert_eq!(stat.n_accepted, 406);
    assert_eq!(stat.n_rejected, 20);

    // check stiffness results
    assert_eq!(solver.out_stiff_step_index(), &[32, 189, 357]);
    array_approx_eq(
        &solver.out_stiff_x(),
        &[1.216973774601867E-02, 8.717646581250652E-01, 1.744401291692531E+00],
        1e-12,
    );
    println!("h·ρ = {:?}", solver.out_stiff_h_times_rho()[32]);
    println!("h·ρ = {:?}", solver.out_stiff_h_times_rho()[189]);
    println!("h·ρ = {:?}", solver.out_stiff_h_times_rho()[357]);
    let max_h_rho = params.stiffness.get_h_times_rho_max();
    assert_eq!(max_h_rho, 3.25);
    assert!(solver.out_stiff_h_times_rho()[0] < max_h_rho);
    assert!(solver.out_stiff_h_times_rho()[32] > max_h_rho);
    assert!(solver.out_stiff_h_times_rho()[189] > max_h_rho);
    assert!(solver.out_stiff_h_times_rho()[357] > max_h_rho);
    assert!(*solver.out_stiff_h_times_rho().last().unwrap() < max_h_rho);
}
