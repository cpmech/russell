#![allow(unused)]

use russell_lab::Vector;
use russell_nonlin::{NlMethod, NlParams, NlSolver, NlStop, NlSystem, NoArgs};
use russell_sparse::CooMatrix;

#[test]
fn test_simple() {
    // define nonlinear system
    let system = NlSystem::new(1, |gg: &mut Vector, u: &Vector, _l: f64, _args: &mut NoArgs| {
        gg[0] = u[0];
        Ok(())
    })
    .unwrap();

    // parameters
    let mut params = NlParams::new(NlMethod::Natural);
    params.verbose = true;

    // define solver
    let mut solver = NlSolver::new(params, system).unwrap();
    solver.enable_output();

    // initial guess
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;

    // solve
    let args = &mut 0;
    solver
        .solve(&mut u, &mut l, NlStop::Lambda(1.0), Some(0.1), args)
        .unwrap();

    // check
    // assert_eq!(u[0], 1.5);
}
