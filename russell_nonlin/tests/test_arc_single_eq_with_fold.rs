use russell_lab::array_approx_eq;
use russell_nonlin::{AutoStep, Config, Direction, Method, Output, Samples, Solver, Stop};

#[test]
fn test_arc_single_eq_with_fold() {
    // nonlinear problem
    let (system, mut state, lambda_ana, mut args) = Samples::single_eq_with_fold_point();

    // configuration
    let mut config = Config::new(Method::Arclength);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // output data
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    solver
        .solve(
            &mut args,
            &mut state,
            Direction::Pos,
            Stop::Steps(5),
            AutoStep::No(0.5),
            Some(out),
        )
        .unwrap();

    // check results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    let uu_mathematica = &[
        0.0,
        0.428095787401572,
        0.928938368657503,
        1.42982786403821,
        1.92613250768946,
        2.42179932045308,
    ];
    let ll_mathematica = &[
        0.0,
        0.279010993784976,
        0.366905391632565,
        0.342229470016786,
        0.280658009972863,
        0.214963176909604,
    ];
    array_approx_eq(&uu, uu_mathematica, 1e-6);
    array_approx_eq(&ll, ll_mathematica, 1e-7);
}
