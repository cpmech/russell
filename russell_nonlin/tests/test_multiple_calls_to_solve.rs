use russell_lab::vec_approx_eq;
use russell_nonlin::{AutoStep, Config, IniDir, Samples, Solver, Stop};

#[test]
fn test_multiple_calls_to_solve_1() {
    // problem
    let (system, mut u, _, _, u_ref, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new();
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // --------------- first call to solve ---------------

    // solve problem
    let mut l = 0.0;
    solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::No(1.0),
            None,
        )
        .unwrap();

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

    // --------------- second call to solve ---------------

    // solve again
    println!();
    solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::No(1.0),
            None,
        )
        .unwrap();

    // check again
    let stats = solver.get_stats();
    let n_iter = 1;
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
fn test_multiple_calls_to_solve_2() {
    // problem
    let (system, mut u, u_ref, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new();
    config
        .set_verbose(true, true, true)
        .set_n_iteration_max(20)
        .set_n_cont_delta_div_max(2);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // --------------- first call to solve ---------------

    // solve problem
    let mut l = 0.0;
    solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::No(1.0),
            None,
        )
        .unwrap();

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
    vec_approx_eq(&u, &u_ref, 1e-10);

    // --------------- second call to solve ---------------

    // solve again
    println!();
    solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(1),
            AutoStep::No(1.0),
            None,
        )
        .unwrap();

    // check
    let stats = solver.get_stats();
    let n_iter = 1;
    let n_jac = n_iter - 1; // because it converges on ‖G‖∞ thus the last Jacobian is NOT computed
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
fn test_multiple_calls_to_solve_1_auto() {
    // problem
    let (system, mut u, _, _, u_ref, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new();
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // --------------- first call to solve ---------------

    // solve problem
    let mut l = 0.0;
    solver
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

    // --------------- second call to solve ---------------

    // solve again
    println!();
    solver
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

    // check again
    let stats = solver.get_stats();
    let n_iter = 1;
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
fn test_multiple_calls_to_solve_2_auto() {
    // problem
    let (system, mut u, u_ref, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new();
    config
        .set_verbose(true, true, true)
        .set_n_iteration_max(20)
        .set_n_cont_delta_div_max(2);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // --------------- first call to solve ---------------

    // solve problem
    let mut l = 0.0;
    solver
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
    vec_approx_eq(&u, &u_ref, 1e-10);

    // --------------- second call to solve ---------------

    // solve again
    println!();
    solver
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

    // check
    let stats = solver.get_stats();
    let n_iter = 1;
    let n_jac = n_iter - 1; // because it converges on ‖G‖∞ thus the last Jacobian is NOT computed
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
