use russell_lab::{array_approx_eq, Vector};
use russell_nonlin::{Config, Method, NoArgs, Solver, Stop, System};
use russell_sparse::{CooMatrix, Sym};

#[test]
fn test_linear_no_auto_num_jac() {
    // define nonlinear system: G(u, λ) = u - λ
    let system = System::new(1, |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
        gg[0] = u[0] - l;
        Ok(())
    })
    .unwrap();

    // configuration
    let mut config = Config::new(Method::Natural);
    config.set_verbose(true, true, false, true).set_tol_ul(1e-9);

    // define solver
    let mut solver = Solver::new(config, system).unwrap();
    solver.enable_output().set_step_recording(&[0]);

    // initial guess
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;

    // solve
    let args = &mut 0;
    solver
        .solve(&mut u, &mut l, Stop::Lambda(1.0), Some(0.1), args)
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
}

#[test]
fn test_linear_no_auto_ana_jac() {
    // define nonlinear system: G(u, λ) = u - λ
    let mut system = System::new(1, |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
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
    config.set_verbose(true, true, false, true);

    // define solver
    let mut solver = Solver::new(config, system).unwrap();
    solver.enable_output().set_step_recording(&[0]);

    // initial guess
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;

    // solve
    let args = &mut 0;
    solver
        .solve(&mut u, &mut l, Stop::Lambda(1.0), Some(0.1), args)
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
}
