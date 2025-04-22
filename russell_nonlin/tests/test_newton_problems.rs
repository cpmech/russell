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
    let n_iter = 8;
    let n_jac = n_iter; // because it converges on ‖δu‖∞ thus the last Jacobian is computed
    assert_eq!(stats.n_function, n_iter);
    assert_eq!(stats.n_jacobian, n_jac);
    assert_eq!(stats.n_factor, n_jac);
    assert_eq!(stats.n_lin_sol, n_jac);
    assert_eq!(stats.n_steps, 1);
    assert_eq!(stats.n_accepted, 1);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_large_delta, 0);
    assert_eq!(stats.n_max_iterations_reached, 0);
    assert_eq!(stats.n_iterations_max, n_iter);
    assert_eq!(stats.n_iterations_total, n_iter);
    vec_approx_eq(&u, &u_ok, 1e-10);
}

#[test]
fn test_newton_problems_fail_due_to_max_iter() {
    // problem
    let (system, u_trial_ok, _, _, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_allowed_iterations(3);

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
        .set_allowed_iterations(20)
        .set_allowed_continued_divergence(2);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).unwrap();

    // check
    let stats = solver.stats();
    let n_iter = 20;
    let n_jac = n_iter - 1; // because it converges on ‖G‖∞ thus the last Jacobian is NOT computed
    assert_eq!(stats.n_function, n_iter);
    assert_eq!(stats.n_jacobian, n_jac);
    assert_eq!(stats.n_factor, n_jac);
    assert_eq!(stats.n_lin_sol, n_jac);
    assert_eq!(stats.n_steps, 1);
    assert_eq!(stats.n_accepted, 1);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_large_delta, 0);
    assert_eq!(stats.n_max_iterations_reached, 0);
    assert_eq!(stats.n_iterations_max, n_iter);
    assert_eq!(stats.n_iterations_total, n_iter);
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

#[test]
fn test_two_eq_nr_prob_1_singular() {
    // problem
    let (system, mut u, _, mut args) = Samples::two_eq_nr_prob_1();

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
        Some("Error(1): Matrix is singular")
    );
}

#[test]
fn test_two_eq_nr_prob_2() {
    // problem
    let (system, u_trial_ok1, u_trial_ok2, u_trial_bad, u_ref1, u_ref2, mut args) = Samples::two_eq_nr_prob_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, false).set_allowed_iterations(20);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem: first solution
    let mut u = u_trial_ok1.clone();
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).unwrap();
    vec_approx_eq(&u, &u_ref1, 1e-9);

    // solve problem: second solution
    let mut u = u_trial_ok2.clone();
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).unwrap();
    vec_approx_eq(&u, &u_ref2, 1e-9);

    // singular case
    let mut u = u_trial_bad.clone();
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    assert_eq!(
        solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).err(),
        Some("Error(1): Matrix is singular")
    );
}
