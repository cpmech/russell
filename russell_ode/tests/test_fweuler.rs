use russell_lab::{approx_eq, Vector};
use russell_ode::{no_dense_output, no_step_output, Method, OdeParams, OdeSolver, Samples};

#[test]
fn test_fweuler_hairer_wanner_eq1() {
    let (system, mut control, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let params = OdeParams::new(Method::FwEuler, None, None);
    let mut solver = OdeSolver::new(&params, system).unwrap();
    solver
        .solve(
            &mut control.y0,
            control.x0,
            control.x1,
            control.h_equal,
            &mut args,
            no_step_output,
            no_dense_output,
        )
        .unwrap();
    let mut analytical = control.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, control.x1);
    approx_eq(control.y0[0], 0.08589790706616637, 1e-15);
    approx_eq(control.y0[0], y1_correct[0], 0.004753);
}
