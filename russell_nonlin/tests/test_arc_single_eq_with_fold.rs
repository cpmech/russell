#![allow(unused)]

use russell_lab::{array_approx_eq, mat_approx_eq, num_jacobian, Vector};
use russell_nonlin::{AutoStep, Config, Direction, Method, NoArgs, Samples, Solver, Stop, System};
use russell_pde::FdmLaplacian2d;
use russell_sparse::{CooMatrix, Sym};

#[test]
fn test_arc_single_eq_with_fold() {
    // nonlinear problem
    let (system, mut state, lambda_ana, mut args) = Samples::single_eq_with_fold_point();

    // configuration
    let mut config = Config::new(Method::Arclength);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // numerical continuation
    solver
        .solve(
            &mut args,
            &mut state,
            Direction::Pos,
            Stop::Steps(5),
            AutoStep::No(0.5),
            None,
        )
        .unwrap();
}
