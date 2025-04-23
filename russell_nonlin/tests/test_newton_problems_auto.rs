use russell_lab::vec_approx_eq;
use russell_nonlin::{AutoStep, Config, Direction, Method, Samples, Solver, Stop};

#[test]
fn test_newton_problems_ok_1_auto() {
    // problem
    let (system, mut state, _, _, u_ref, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    solver
        .solve(
            &mut args,
            &mut state,
            Direction::Pos,
            Stop::Steps(1),
            AutoStep::Yes,
            None,
        )
        .unwrap();

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
    vec_approx_eq(&state.u, &u_ref, 1e-10);
}

#[test]
fn test_newton_problems_fail_due_to_max_iter_auto() {
    // problem
    let (system, mut state, _, _, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_allowed_iterations(3);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    assert_eq!(
        solver
            .solve(
                &mut args,
                &mut state,
                Direction::Pos,
                Stop::Steps(1),
                AutoStep::Yes,
                None,
            )
            .err(),
        Some("failed to solve the nonlinear problem with automatic stepsize")
    );
    assert_eq!(
        solver.errors(),
        &["max number of iterations reached", "too many continued rejections"]
    );
}

#[test]
fn test_newton_problems_fail_oscillation_auto() {
    // problem
    let (system, _, mut state, _, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_n_cont_reject_allowed(4);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    assert_eq!(
        solver
            .solve(
                &mut args,
                &mut state,
                Direction::Pos,
                Stop::Steps(1),
                AutoStep::Yes,
                None,
            )
            .err(),
        Some("failed to solve the nonlinear problem with automatic stepsize")
    );
    assert_eq!(
        solver.errors(),
        &["max number of iterations reached", "too many continued rejections"]
    );
}

#[test]
fn test_newton_problems_indeterminate_auto() {
    // problem
    let (system, _, _, mut state, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    assert_eq!(
        solver
            .solve(
                &mut args,
                &mut state,
                Direction::Pos,
                Stop::Steps(1),
                AutoStep::Yes,
                None,
            )
            .err(),
        Some("failed to solve the nonlinear problem with automatic stepsize")
    );
    assert_eq!(
        solver.errors(),
        &["max(‖δu‖∞,|δλ|) is too large", "too many continued rejections"]
    );
}

#[test]
fn test_newton_problems_ok_2_auto() {
    // problem
    let (system, mut state, u_ref, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_allowed_iterations(20)
        .set_allowed_continued_divergence(2);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    solver
        .solve(
            &mut args,
            &mut state,
            Direction::Pos,
            Stop::Steps(1),
            AutoStep::Yes,
            None,
        )
        .unwrap();

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
    vec_approx_eq(&state.u, &u_ref, 1e-12);
}

#[test]
fn test_simple_fixed_continued_divergence_auto() {
    // problem
    let (system, mut state, _, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_n_cont_reject_allowed(4);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    assert_eq!(
        solver
            .solve(
                &mut args,
                &mut state,
                Direction::Pos,
                Stop::Steps(1),
                AutoStep::Yes,
                None,
            )
            .err(),
        Some("failed to solve the nonlinear problem with automatic stepsize")
    );
    assert_eq!(
        solver.errors(),
        &["continued divergence detected", "too many continued rejections"]
    );
}

#[test]
fn test_newton_problems_stepsize_becomes_small() {
    // problem
    let (system, mut state, _, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_allowed_iterations(1)
        .set_m_failure(0.01);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    assert_eq!(
        solver
            .solve(
                &mut args,
                &mut state,
                Direction::Pos,
                Stop::Steps(1),
                AutoStep::Yes,
                None,
            )
            .err(),
        Some("failed to solve the nonlinear problem with automatic stepsize")
    );
    assert_eq!(
        solver.errors(),
        &["max number of iterations reached", "the stepsize becomes too small"]
    );
}
