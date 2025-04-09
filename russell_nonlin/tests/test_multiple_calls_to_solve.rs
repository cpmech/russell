use russell_lab::vec_approx_eq;
use russell_nonlin::{Config, Method, Samples, Solver, Stop};

#[test]
fn test_multiple_calls_to_solve_1() {
    // problem
    let (system, u_trial_ok, _, _, u_ok, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // --------------- first call to solve ---------------

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
    assert_eq!(stats.n_large_du_dl, 0);
    assert_eq!(stats.n_max_iterations_reached, 0);
    assert_eq!(stats.n_iterations_max, n_iter);
    assert_eq!(stats.n_iterations_total, n_iter);
    vec_approx_eq(&u, &u_ok, 1e-10);

    // --------------- second call to solve ---------------

    // solve again
    println!();
    solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).unwrap();

    // check again
    let stats = solver.stats();
    let n_iter = 1;
    let n_jac = n_iter; // because it converges on ‖δu‖∞ thus the last Jacobian is computed
    assert_eq!(stats.n_function, n_iter);
    assert_eq!(stats.n_jacobian, n_jac);
    assert_eq!(stats.n_factor, n_jac);
    assert_eq!(stats.n_lin_sol, n_jac);
    assert_eq!(stats.n_steps, 1);
    assert_eq!(stats.n_accepted, 1);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_large_du_dl, 0);
    assert_eq!(stats.n_max_iterations_reached, 0);
    assert_eq!(stats.n_iterations_max, n_iter);
    assert_eq!(stats.n_iterations_total, n_iter);
    vec_approx_eq(&u, &u_ok, 1e-10);
}

#[test]
fn test_multiple_calls_to_solve_2() {
    // problem
    let (system, mut u, u_ref, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_n_iteration_max(20)
        .set_n_cont_div_ul_allowed(2);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // --------------- first call to solve ---------------

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
    assert_eq!(stats.n_large_du_dl, 0);
    assert_eq!(stats.n_max_iterations_reached, 0);
    assert_eq!(stats.n_iterations_max, n_iter);
    assert_eq!(stats.n_iterations_total, n_iter);
    vec_approx_eq(&u, &u_ref, 1e-10);

    // --------------- second call to solve ---------------

    // solve again
    println!();
    solver.solve(&mut u, &mut l, stop, Some(1.0), &mut args).unwrap();

    // check
    let stats = solver.stats();
    let n_iter = 1;
    let n_jac = n_iter - 1; // because it converges on ‖G‖∞ thus the last Jacobian is NOT computed
    assert_eq!(stats.n_function, n_iter);
    assert_eq!(stats.n_jacobian, n_jac);
    assert_eq!(stats.n_factor, n_jac);
    assert_eq!(stats.n_lin_sol, n_jac);
    assert_eq!(stats.n_steps, 1);
    assert_eq!(stats.n_accepted, 1);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_large_du_dl, 0);
    assert_eq!(stats.n_max_iterations_reached, 0);
    assert_eq!(stats.n_iterations_max, n_iter);
    assert_eq!(stats.n_iterations_total, n_iter);
    vec_approx_eq(&u, &u_ref, 1e-10);
}

#[test]
fn test_multiple_calls_to_solve_1_auto() {
    // problem
    let (system, u_trial_ok, _, _, u_ok, mut args) = Samples::cubic_poly_1();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // --------------- first call to solve ---------------

    // solve problem
    let mut u = u_trial_ok;
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    solver.solve(&mut u, &mut l, stop, None, &mut args).unwrap();

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
    assert_eq!(stats.n_large_du_dl, 0);
    assert_eq!(stats.n_max_iterations_reached, 0);
    assert_eq!(stats.n_iterations_max, n_iter);
    assert_eq!(stats.n_iterations_total, n_iter);
    vec_approx_eq(&u, &u_ok, 1e-10);

    // --------------- second call to solve ---------------

    // solve again
    println!();
    solver.solve(&mut u, &mut l, stop, None, &mut args).unwrap();

    // check again
    let stats = solver.stats();
    let n_iter = 1;
    let n_jac = n_iter; // because it converges on ‖δu‖∞ thus the last Jacobian is computed
    assert_eq!(stats.n_function, n_iter);
    assert_eq!(stats.n_jacobian, n_jac);
    assert_eq!(stats.n_factor, n_jac);
    assert_eq!(stats.n_lin_sol, n_jac);
    assert_eq!(stats.n_steps, 1);
    assert_eq!(stats.n_accepted, 1);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_large_du_dl, 0);
    assert_eq!(stats.n_max_iterations_reached, 0);
    assert_eq!(stats.n_iterations_max, n_iter);
    assert_eq!(stats.n_iterations_total, n_iter);
    vec_approx_eq(&u, &u_ok, 1e-10);
}

#[test]
fn test_multiple_calls_to_solve_2_auto() {
    // problem
    let (system, mut u, u_ref, mut args) = Samples::cubic_poly_2();

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_n_iteration_max(20)
        .set_n_cont_div_ul_allowed(2);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // --------------- first call to solve ---------------

    // solve problem
    let mut l = 0.0;
    let stop = Stop::Steps(1); // just one step
    solver.solve(&mut u, &mut l, stop, None, &mut args).unwrap();

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
    assert_eq!(stats.n_large_du_dl, 0);
    assert_eq!(stats.n_max_iterations_reached, 0);
    assert_eq!(stats.n_iterations_max, n_iter);
    assert_eq!(stats.n_iterations_total, n_iter);
    vec_approx_eq(&u, &u_ref, 1e-10);

    // --------------- second call to solve ---------------

    // solve again
    println!();
    solver.solve(&mut u, &mut l, stop, None, &mut args).unwrap();

    // check
    let stats = solver.stats();
    let n_iter = 1;
    let n_jac = n_iter - 1; // because it converges on ‖G‖∞ thus the last Jacobian is NOT computed
    assert_eq!(stats.n_function, n_iter);
    assert_eq!(stats.n_jacobian, n_jac);
    assert_eq!(stats.n_factor, n_jac);
    assert_eq!(stats.n_lin_sol, n_jac);
    assert_eq!(stats.n_steps, 1);
    assert_eq!(stats.n_accepted, 1);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_large_du_dl, 0);
    assert_eq!(stats.n_max_iterations_reached, 0);
    assert_eq!(stats.n_iterations_max, n_iter);
    assert_eq!(stats.n_iterations_total, n_iter);
    vec_approx_eq(&u, &u_ref, 1e-10);
}
