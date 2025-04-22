use russell_lab::{array_approx_eq, Vector};
use russell_nonlin::{Config, Method, NoArgs, Solver, State, Stop, System, TgVec};
use russell_sparse::{CooMatrix, Sym};

#[test]
fn test_linear_no_auto_ana_jac() {
    // define nonlinear system: G(u, λ) = u - λ
    let ndim = 1;
    let mut system = System::new(ndim, |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
        gg[0] = u[0] - l;
        Ok(())
    })
    .unwrap();

    // set analytical Jacobian
    let nnz = Some(1);
    let sym = Sym::No;
    system
        .set_calc_ggu(
            nnz,
            sym,
            |ggu: &mut CooMatrix, _l: f64, _u: &Vector, _args: &mut NoArgs| {
                ggu.reset();
                // dG/du = 1
                ggu.put(0, 0, 1.0).unwrap();
                Ok(())
            },
        )
        .unwrap();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true);

    // define solver
    let mut solver = Solver::new(config, system).unwrap();
    solver.enable_output().set_step_recording(&[0]);

    // initial guess
    let mut state = State::new(ndim, false);
    state.u[0] = 0.0;
    state.l = 0.0;

    // solve
    let tg = TgVec::Positive;
    let args = &mut 0;
    solver
        .solve(&mut state, tg, Stop::Lambda(1.0), Some(0.1), args)
        .unwrap();

    // results
    // println!("u[0] = {:?}", solver.out_step_u(0));

    // check
    assert_eq!(
        solver.out_step_h(),
        &[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    );
    array_approx_eq(
        solver.out_step_l(),
        &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        1e-15,
    );
    array_approx_eq(
        solver.out_step_u(0),
        &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        1e-15,
    );

    // check stats
    let nstep = 10;
    let niter = 10 * 2;
    let stats = solver.stats();
    assert_eq!(stats.n_function, niter);
    assert_eq!(stats.n_jacobian, nstep);
    assert_eq!(stats.n_factor, nstep);
    assert_eq!(stats.n_lin_sol, nstep);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iterations_max, 2);
    assert_eq!(stats.n_iterations_total, niter);
    assert!(stats.nanos_step_max > 0);
    assert!(stats.nanos_jacobian_max > 0);
    assert!(stats.nanos_factor_max > 0);
    assert!(stats.nanos_lin_sol_max > 0);
    assert!(stats.nanos_total > 0);
}

#[test]
fn test_linear_no_auto_num_jac() {
    // define nonlinear system: G(u, λ) = u - λ
    let ndim = 1;
    let system = System::new(ndim, |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
        gg[0] = u[0] - l;
        Ok(())
    })
    .unwrap();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, true).set_tol_delta(1e-9, 1e-7);

    // define solver
    let mut solver = Solver::new(config, system).unwrap();
    solver.enable_output().set_step_recording(&[0]);

    // initial guess
    let mut state = State::new(ndim, false);
    state.u[0] = 0.0;
    state.l = 0.0;

    // solve
    let tg = TgVec::Positive;
    let args = &mut 0;
    solver
        .solve(&mut state, tg, Stop::Lambda(1.0), Some(0.1), args)
        .unwrap();

    // results
    // println!("u[0] = {:?}", solver.out_step_u(0));

    // check
    assert_eq!(
        solver.out_step_h(),
        &[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    );
    array_approx_eq(
        solver.out_step_l(),
        &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        1e-15,
    );
    array_approx_eq(
        solver.out_step_u(0),
        &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        1e-9,
    );

    // check stats
    let nstep = 10;
    let n_conv_du = 5; // number of converged steps on δu
    let n_num_jac = 10 + 1; // +1 because the first iteration does not converge immediately
    let niter = 10 * 2 + 1; // +1 because the first iteration does not converge immediately
    let stats = solver.stats();
    assert_eq!(stats.n_function, niter + n_num_jac + n_conv_du);
    assert_eq!(stats.n_jacobian, 0);
    assert_eq!(stats.n_factor, n_num_jac + n_conv_du);
    assert_eq!(stats.n_lin_sol, n_num_jac + n_conv_du);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iterations_max, 3);
    assert_eq!(stats.n_iterations_total, niter);
    assert!(stats.nanos_step_max > 0);
    assert!(stats.nanos_jacobian_max > 0);
    assert!(stats.nanos_factor_max > 0);
    assert!(stats.nanos_lin_sol_max > 0);
    assert!(stats.nanos_total > 0);
}
