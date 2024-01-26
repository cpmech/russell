use russell_lab::approx_eq;
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

    println!("y1 = {:?}", sample.y0);
    // approx_eq(sample.y0[0], 0.08589790706616637, 1e-17);
}
