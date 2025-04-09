use russell_lab::vec_approx_eq;
use russell_nonlin::{Config, Method, Samples, Solver, Stop};

#[test]
fn test_newton_problems_ok_1() {
    // problem
    let (system, u_trial_ok, _, _, u_ok, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    let mut u = u_trial_ok;
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).unwrap();

    // check
    let stats = solver.stats();
    assert_eq!(stats.n_iterations_max, 8);
    vec_approx_eq(&u, &u_ok, 1e-10);
}

#[test]
fn test_newton_problems_fail_due_to_max_iter() {
    // problem
    let (system, u_trial_ok, _, _, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_n_iteration_max(3);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    let mut u = u_trial_ok;
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    assert_eq!(
        solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).err(),
        Some("failed to solve the nonlinear problem with equal stepsize")
    );
    assert_eq!(solver.errors(), &["max number of iterations reached"]);
}

#[test]
fn test_newton_problems_fail_oscillation() {
    // problem
    let (system, _, u_trial_oscillation, _, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    let mut u = u_trial_oscillation;
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    assert_eq!(
        solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).err(),
        Some("failed to solve the nonlinear problem with equal stepsize")
    );
    assert_eq!(solver.errors(), &["max number of iterations reached"]);
}

#[test]
fn test_newton_problems_indeterminate() {
    // problem
    let (system, _, _, u_trial_indeterminate, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    let mut u = u_trial_indeterminate;
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    assert_eq!(
        solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).err(),
        Some("failed to solve the nonlinear problem with equal stepsize")
    );
    assert_eq!(solver.errors(), &["max(‖δu‖∞,|δλ|) is too large"]);
}

#[test]
fn test_newton_problems_ok_2() {
    // problem
    let (system, mut u, u_ref, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_n_iteration_max(20)
        .set_n_allowed_cont_div_ul(2);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).unwrap();

    // check
    let stats = solver.stats();
    assert_eq!(stats.n_iterations_max, 20);
    vec_approx_eq(&u, &u_ref, 1e-12);
}

#[test]
fn test_simple_fixed_continued_divergence() {
    // problem
    let (system, mut u, _, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    assert_eq!(
        solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).err(),
        Some("failed to solve the nonlinear problem with equal stepsize")
    );
    assert_eq!(solver.errors(), &["continued divergence detected"]);
}
