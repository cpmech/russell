use russell_lab::vec_approx_eq;
use russell_nonlin::{AutoStep, Config, IniDir, Method, Samples, Solver, Status, Stop};

#[test]
fn test_newton_problems_ok_1_auto() {
    // problem
    let (system, mut u, _, _, u_ref, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_hide_timings(true);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // solve problem
    let mut l = 0.0;
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::Yes,
            None,
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // check
    let stats = solver.get_stats();
    let n_iter = 8;
    let n_jac = n_iter; // because it converges on ‖δu‖∞ thus the last Jacobian is computed
    assert_eq!(stats.n_function, n_iter);
    assert_eq!(stats.n_jacobian, n_jac);
    assert_eq!(stats.n_factor, n_jac);
    assert_eq!(stats.n_lin_sol, n_jac);
    assert_eq!(stats.n_steps, 1);
    assert_eq!(stats.n_accepted, 1);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iteration_total, n_iter);
    vec_approx_eq(&u, &u_ref, 1e-10);
}

#[test]
fn test_newton_problems_fail_due_to_continued_failure_auto() {
    // problem
    let (system, mut u, _, _, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_n_iteration_max(3);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // solve problem
    let mut l = 0.0;
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::Yes,
            None,
        )
        .unwrap();
    assert_eq!(status, Status::ContinuedFailure);
}

#[test]
fn test_newton_problems_fail_oscillation_auto() {
    // problem
    let (system, _, mut u, _, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_hide_timings(true);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // solve problem
    let mut l = 0.0;
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::Yes,
            None,
        )
        .unwrap();
    assert_eq!(status, Status::ContinuedFailure);
}

#[test]
fn test_newton_problems_indeterminate_auto() {
    // problem
    let (system, _, _, mut u, _, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_hide_timings(true);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // solve problem
    let mut l = 0.0;
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::Yes,
            None,
        )
        .unwrap();
    assert_eq!(status, Status::ContinuedFailure);
}

#[test]
fn test_newton_problems_ok_2_auto() {
    // problem
    let (system, mut u, u_ref, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_n_iteration_max(20)
        .set_n_cont_divergence_max(2);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // solve problem
    let mut l = 0.0;
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::Yes,
            None,
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // check
    let stats = solver.get_stats();
    let n_iter = 20;
    let n_jac = n_iter - 1; // because it converges on ‖G‖∞ thus the last Jacobian is NOT computed
    assert_eq!(stats.n_function, n_iter);
    assert_eq!(stats.n_jacobian, n_jac);
    assert_eq!(stats.n_factor, n_jac);
    assert_eq!(stats.n_lin_sol, n_jac);
    assert_eq!(stats.n_steps, 1);
    assert_eq!(stats.n_accepted, 1);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iteration_total, n_iter);
    vec_approx_eq(&u, &u_ref, 1e-12);
}

#[test]
fn test_simple_fixed_continued_divergence_auto() {
    // problem
    let (system, mut u, _, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_n_cont_divergence_max(1);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // solve problem
    let mut l = 0.0;
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::Yes,
            None,
        )
        .unwrap();
    assert_eq!(status, Status::ContinuedFailure);
}

#[test]
fn test_newton_problems_stepsize_becomes_small() {
    // problem
    let (system, mut u, _, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_n_iteration_max(1)
        .set_m_failure(0.01);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // solve problem
    let mut l = 0.0;
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::Yes,
            None,
        )
        .unwrap();
    assert_eq!(status, Status::SmallStepsize);
}
