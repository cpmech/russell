use russell_lab::vec_approx_eq;
use russell_nonlin::{AutoStep, Config, Direction, Method, Samples, Solver, Status, Stop};

#[test]
fn test_newton_problems_ok_1() {
    // problem
    let (system, mut state, _, _, u_ref, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_hide_timings(true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem
    solver
        .solve(
            &mut args,
            &mut state,
            Direction::Pos,
            Stop::Steps(1),
            AutoStep::No(1.0),
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
fn test_newton_problems_fail_due_to_max_iter() {
    // problem
    let (system, mut state, _, _, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_allowed_iterations(3);

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
                AutoStep::No(1.0),
                None,
            )
            .unwrap(),
        Status::Failure
    );
    assert_eq!(solver.errors(), &["max number of iterations reached"]);
}

#[test]
fn test_newton_problems_fail_oscillation() {
    // problem
    let (system, _, mut state, _, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_hide_timings(true);

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
                AutoStep::No(1.0),
                None,
            )
            .unwrap(),
        Status::Failure
    );
    assert_eq!(solver.errors(), &["max number of iterations reached"]);
}

#[test]
fn test_newton_problems_indeterminate() {
    // problem
    let (system, _, _, mut state, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_hide_timings(true);

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
                AutoStep::No(1.0),
                None,
            )
            .unwrap(),
        Status::Failure
    );
    assert_eq!(solver.errors(), &["‖(δu,δλ)‖∞ is too large"]);
}

#[test]
fn test_newton_problems_ok_2() {
    // problem
    let (system, mut state, u_ref, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
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
            AutoStep::No(1.0),
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
fn test_simple_fixed_continued_divergence() {
    // problem
    let (system, mut state, _, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_hide_timings(true);

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
                AutoStep::No(1.0),
                None,
            )
            .unwrap(),
        Status::Failure
    );
    assert_eq!(solver.errors(), &["continued divergence detected"]);
}

#[test]
fn test_two_eq_nr_prob_1_singular() {
    // problem
    let (system, mut state, _, mut args) = Samples::two_eq_nr_prob_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_hide_timings(true);

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
                AutoStep::No(1.0),
                None,
            )
            .err(),
        Some("Error(1): Matrix is singular")
    );
}

#[test]
fn test_two_eq_nr_prob_2() {
    // problem
    let (system, mut state_ok1, mut state_ok2, mut state_bad, u_ref1, u_ref2, mut args) = Samples::two_eq_nr_prob_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, false)
        .set_hide_timings(true)
        .set_allowed_iterations(20);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // solve problem: first solution
    solver
        .solve(
            &mut args,
            &mut state_ok1,
            Direction::Pos,
            Stop::Steps(1),
            AutoStep::No(1.0),
            None,
        )
        .unwrap();
    vec_approx_eq(&state_ok1.u, &u_ref1, 1e-9);

    // solve problem: second solution
    solver
        .solve(
            &mut args,
            &mut state_ok2,
            Direction::Pos,
            Stop::Steps(1),
            AutoStep::No(1.0),
            None,
        )
        .unwrap();
    vec_approx_eq(&state_ok2.u, &u_ref2, 1e-9);

    // singular case
    assert_eq!(
        solver
            .solve(
                &mut args,
                &mut state_bad,
                Direction::Pos,
                Stop::Steps(1),
                AutoStep::No(1.0),
                None
            )
            .err(),
        Some("Error(1): Matrix is singular")
    );
}
