/*
use russell_lab::{approx_eq, Vector};
use russell_ode::{output_dense_none, output_step_none, Method, OdeParams, OdeSolver, Samples};

#[test]
fn test_fweuler_hairer_wanner_eq1() {
    let mut sample = Samples::hairer_wanner_eq1();
    let params = OdeParams::new(Method::FwEuler, None, None);
    let mut solver = OdeSolver::new(&params, sample.ndim, sample.system).unwrap();
    solver
        .solve(
            &mut sample.y0,
            sample.x0,
            sample.x1,
            sample.h_equal,
            output_step_none,
            output_dense_none,
        )
        .unwrap();
    let mut analytical = sample.analytical.unwrap();
    let mut y1_correct = Vector::new(sample.ndim);
    analytical(&mut y1_correct, sample.x1);
    approx_eq(sample.y0[0], 0.08589790706616637, 1e-16);
    approx_eq(sample.y0[0], y1_correct[0], 0.004753);
}
*/
