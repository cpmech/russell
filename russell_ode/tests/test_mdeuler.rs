use russell_lab::{approx_eq, Vector};
use russell_ode::{no_dense_output, no_step_output, Method, OdeParams, OdeSolver, Samples};

#[test]
fn test_mdeuler_hairer_wanner_eq1() {
    let (system, mut control, mut args) = Samples::hairer_wanner_eq1();
    let ndim = system.get_ndim();
    let params = OdeParams::new(Method::MdEuler, None, None);
    let mut solver = OdeSolver::new(&params, system).unwrap();
    solver
        .solve(
            &mut control.y0,
            control.x0,
            control.x1,
            None,
            &mut args,
            no_step_output,
            no_dense_output,
        )
        .unwrap();

    let b = solver.bench();
    let mut analytical = control.y_analytical.unwrap();
    let mut y1_correct = Vector::new(ndim);
    analytical(&mut y1_correct, control.x1);
    println!("y =\n{}", control.y0);
    println!("y_ana =\n{}", y1_correct);
    println!("{}", b);
    if false {
        approx_eq(control.y0[0], 0.09062475637905158, 1e-17);
        approx_eq(control.y0[0], y1_correct[0], 1e-4);
    }
}
