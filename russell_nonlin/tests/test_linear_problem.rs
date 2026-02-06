use russell_lab::array_approx_eq;
use russell_nonlin::{Config, DeltaLambda, IniDir, Output, Samples, Solver, Stop};
use russell_sparse::Sym;

#[test]
fn test_linear_constant() {
    // system
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);

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
            DeltaLambda::constant(0.1),
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
    let niter = 10; // the Euler predictor makes it converge in 1 iteration per step
    let stats = solver.get_stats();
    assert_eq!(stats.n_function, niter);
    assert_eq!(stats.n_jacobian, 1);
    assert_eq!(stats.n_factor, 1);
    assert_eq!(stats.n_lin_sol, 0);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iteration_total, niter);
    assert!(stats.nanos_step_max > 0);
    assert!(stats.nanos_jacobian_max > 0);
    assert!(stats.nanos_factor_max > 0);
    assert!(stats.nanos_lin_sol_max == 0);
    assert!(stats.nanos_total > 0);
}

#[test]
fn test_linear_auto() {
    // system
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);

    // configuration
    let mut config = Config::new();
    config.set_verbose(true, true, true).set_hide_timings(true);

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
            DeltaLambda::auto(0.1),
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
fn test_linear_constant_backward() {
    // system
    let (system, mut u, _, mut args) = Samples::simple_linear_problem(Sym::No);

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
            DeltaLambda::constant(0.1),
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
    let niter = nstep;
    let stats = solver.get_stats();
    assert_eq!(stats.n_function, niter);
    assert_eq!(stats.n_jacobian, 1); // because no iterations happen due to linear problem and Euler predictor needs this
    assert_eq!(stats.n_factor, 1); // same reason as above
    assert_eq!(stats.n_lin_sol, 0);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iteration_total, niter);
    assert!(stats.nanos_step_max > 0);
    assert!(stats.nanos_jacobian_max > 0);
    assert!(stats.nanos_factor_max > 0);
    assert!(stats.nanos_total > 0);
}

#[test]
fn test_linear_list() {
    // system
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);

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
            Stop::MaxLambda(100.0),
            DeltaLambda::list(&[0.1, 0.2, 0.4, 0.8]),
            Some(out),
        )
        .unwrap();

    // check
    array_approx_eq(out.get_h_values(), &[0.1, 0.2, 0.4, 0.8, 0.8], 1e-15);
    array_approx_eq(out.get_l_values(), &[0.0, 0.1, 0.3, 0.7, 1.5], 1e-15);
    array_approx_eq(out.get_u_values(0), &[0.0, 0.1, 0.3, 0.7, 1.5], 1e-15);

    let nstep = 4;
    let niter = nstep; // the Euler predictor makes it converge in 1 iteration per step
    let stats = solver.get_stats();
    assert_eq!(stats.n_function, niter);
    assert_eq!(stats.n_jacobian, 1);
    assert_eq!(stats.n_factor, 1);
    assert_eq!(stats.n_lin_sol, 0);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iteration_total, niter);
}
