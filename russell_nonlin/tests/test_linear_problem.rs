use russell_lab::array_approx_eq;
use russell_nonlin::{AutoStep, Config, IniDir, Output, Samples, Solver, Stop};

#[test]
fn test_linear_no_auto_ana_jac() {
    // system
    let with_ggu = true; // with ∂G/∂u => analytical Jacobian
    let with_ggl = false; // no ∂G/∂λ
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(with_ggu, with_ggl);

    // configuration
    let mut config = Config::new();
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_record_iterations_residuals(true);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[]);

    // solve
    solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.0),
            AutoStep::No(0.1),
            Some(out),
        )
        .unwrap();

    // check
    assert_eq!(
        out.get_h_values(),
        &[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    );
    array_approx_eq(
        out.get_l_values(),
        &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        1e-15,
    );
    array_approx_eq(
        out.get_u_values(0),
        &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        1e-15,
    );

    // check stats
    let nstep = 10;
    let niter = 10 * 2;
    let stats = solver.get_stats();
    assert_eq!(stats.n_function, niter);
    assert_eq!(stats.n_jacobian, nstep);
    assert_eq!(stats.n_factor, nstep);
    assert_eq!(stats.n_lin_sol, nstep);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iteration_total, niter);
    assert!(stats.nanos_step_max > 0);
    assert!(stats.nanos_jacobian_max > 0);
    assert!(stats.nanos_factor_max > 0);
    assert!(stats.nanos_lin_sol_max > 0);
    assert!(stats.nanos_total > 0);
}

#[test]
fn test_linear_no_auto_num_jac() {
    // system
    let with_ggu = false; // no ∂G/∂u => numerical Jacobian
    let with_ggl = false; // no ∂G/∂λ
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(with_ggu, with_ggl);

    // configuration
    let mut config = Config::new();
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_tol_delta(1e-9, 1e-7);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[]);

    // solve
    solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.0),
            AutoStep::No(0.1),
            Some(out),
        )
        .unwrap();

    // check
    assert_eq!(
        out.get_h_values(),
        &[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    );
    array_approx_eq(
        out.get_l_values(),
        &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        1e-15,
    );
    array_approx_eq(
        out.get_u_values(0),
        &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        1e-9,
    );

    // check stats
    let nstep = 10;
    let n_conv_du = 5; // number of converged steps on δu
    let n_num_jac = 10 + 1; // +1 because the first iteration does not converge immediately
    let niter = 10 * 2 + 1; // +1 because the first iteration does not converge immediately
    let stats = solver.get_stats();
    assert_eq!(stats.n_function, niter + n_num_jac + n_conv_du);
    assert_eq!(stats.n_jacobian, 0);
    assert_eq!(stats.n_factor, n_num_jac + n_conv_du);
    assert_eq!(stats.n_lin_sol, n_num_jac + n_conv_du);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iteration_total, niter);
    assert!(stats.nanos_step_max > 0);
    assert!(stats.nanos_jacobian_max > 0);
    assert!(stats.nanos_factor_max > 0);
    assert!(stats.nanos_lin_sol_max > 0);
    assert!(stats.nanos_total > 0);
}

#[test]
fn test_linear_auto_ana_jac() {
    // system
    let with_ggu = true; // with ∂G/∂u => analytical Jacobian
    let with_ggl = false; // no ∂G/∂λ
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(with_ggu, with_ggl);

    // configuration
    let mut config = Config::new();
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_h_ini(0.1);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[]);

    // solve
    solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.0),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();

    // check stats
    let stats = solver.get_stats();
    assert_eq!(stats.n_accepted, 9);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_steps, 9);
}

#[test]
fn test_linear_no_auto_ana_jac_backward() {
    // system
    let with_ggu = true; // with ∂G/∂u => analytical Jacobian
    let with_ggl = false; // no ∂G/∂λ
    let (system, mut u, _, mut args) = Samples::simple_linear_problem(with_ggu, with_ggl);

    // initial state
    u[0] = 1.0;
    let mut l = 1.0;

    // configuration
    let mut config = Config::new();
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_record_iterations_residuals(true);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[]);

    // solve
    solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Neg,
            Stop::MinLambda(0.0),
            AutoStep::No(0.1),
            Some(out),
        )
        .unwrap();

    // check
    assert_eq!(
        out.get_h_values(),
        &[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    );
    array_approx_eq(
        out.get_l_values(),
        &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        1e-15,
    );
    array_approx_eq(
        out.get_u_values(0),
        &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        1e-15,
    );

    // check stats
    let nstep = 10;
    let niter = 10 * 2;
    let stats = solver.get_stats();
    assert_eq!(stats.n_function, niter);
    assert_eq!(stats.n_jacobian, nstep);
    assert_eq!(stats.n_factor, nstep);
    assert_eq!(stats.n_lin_sol, nstep);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iteration_total, niter);
    assert!(stats.nanos_step_max > 0);
    assert!(stats.nanos_jacobian_max > 0);
    assert!(stats.nanos_factor_max > 0);
    assert!(stats.nanos_lin_sol_max > 0);
    assert!(stats.nanos_total > 0);
}
