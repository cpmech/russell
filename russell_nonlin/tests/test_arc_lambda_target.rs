use russell_lab::math::SQRT_2;
use russell_nonlin::{Config, DeltaLambda, IniDir, Method, Samples, Solver, Status, Stop};
use russell_sparse::Sym;

const RADIUS: f64 = SQRT_2;

#[test]
fn test_arc_lambda_target_max_lambda_one() {
    // system
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);

    // configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // numerical continuation
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.0),
            &DeltaLambda::list(&[0.1; 10]),
            None,
        )
        .unwrap();
    assert_eq!(status, Status::Success);
    assert!(f64::abs(l - 1.0) <= 1e-13);
    assert!(f64::abs(u[0] - l) <= 1e-13);
}

#[test]
fn test_arc_lambda_target_mid_path() {
    // system
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);

    // configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // numerical continuation
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(0.5),
            &DeltaLambda::list(&[0.1; 5]),
            None,
        )
        .unwrap();
    assert_eq!(status, Status::Success);
    assert!(f64::abs(l - 0.5) <= 1e-13);
}

/// The discriminating case: on a curved path `dλ/ds` changes within each step, so a listed Δλ
/// (converted to a step in s through the previous |dλ/ds|) accumulates a shortfall of ~1e-3 over
/// eight steps. Without the reach-based trigger the target is missed; with it, it is exact.
#[test]
fn test_arc_lambda_target_circle_list() {
    // system: circle u² + λ² = R², starting at u = λ = R/√2 = 1 and running to λ = 1.2
    let (system, mut u, mut l, mut args) = Samples::circle_ul(RADIUS);

    // configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // numerical continuation
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.2),
            &DeltaLambda::list(&[0.025; 8]),
            None,
        )
        .unwrap();
    assert_eq!(status, Status::Success);
    assert!(f64::abs(l - 1.2) <= 1e-13, "lambda = {:.17}, short by {:.3e}", l, f64::abs(l - 1.2));
    let radius = f64::sqrt(u[0] * u[0] + l * l);
    // the point stays on the path to the solver's own convergence tolerance
    assert!(f64::abs(radius - RADIUS) <= 1e-8, "radius = {:.17}", radius);
}

/// With automatic Δλ the step grows and can overshoot the target; the cap and the reach-based
/// trigger must make the last step land exactly, from either side.
#[test]
fn test_arc_lambda_target_circle_auto() {
    let (system, mut u, mut l, mut args) = Samples::circle_ul(RADIUS);

    let mut config = Config::new();
    config.set_method(Method::Arclength);

    let mut solver = Solver::new(&config, system).unwrap();
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.2),
            &DeltaLambda::auto(0.05),
            None,
        )
        .unwrap();
    assert_eq!(status, Status::Success);
    assert!(f64::abs(l - 1.2) <= 1e-13, "lambda = {:.17}, short by {:.3e}", l, f64::abs(l - 1.2));
    let radius = f64::sqrt(u[0] * u[0] + l * l);
    // the point stays on the path to the solver's own convergence tolerance
    assert!(f64::abs(radius - RADIUS) <= 1e-8, "radius = {:.17}", radius);
}
