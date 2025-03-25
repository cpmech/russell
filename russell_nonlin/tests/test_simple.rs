#![allow(unused)]

use russell_lab::Vector;
use russell_nonlin::{NlMethod, NlParams, NlSolver, NlSystem, NoArgs};
use russell_sparse::CooMatrix;

#[test]
fn test_simple() {
    // define nonlinear system
    let system = NlSystem::new(1, |gg: &mut Vector, u: &Vector, _l: f64, _args: &mut NoArgs| {
        gg[0] = u[0];
        Ok(())
    })
    .unwrap();

    // define solver
    let params = NlParams::new(NlMethod::Simple);
    let mut solver = NlSolver::new(params, system).unwrap();

    // initial guess
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;

    // solve
    let args = &mut 0;
    solver.solve(&mut u, &mut l, 1.5, None, args).unwrap();

    // check
    assert_eq!(u[0], 1.5);
}
