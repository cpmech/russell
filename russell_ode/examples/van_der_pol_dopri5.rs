use plotpy::{Curve, Plot, RayEndpoint};
use russell_lab::{StrError, Vector};
use russell_ode::prelude::*;

// This example solves Van der Pol's equation
//
// This example corresponds to Fig 2.6 on page 23 of the reference.
// The problem is defined in Eq (1.5') on page 5 of the reference.
//
// This example shows how to enable the stiffness detection.
//
// # Reference
//
// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
//   Stiff and Differential-Algebraic Problems. Second Revised Edition.
//   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p

fn main() -> Result<(), StrError> {
    // get the ODE system
    const EPS: f64 = 0.003;
    let (system, x0, _, x1, mut args) = Samples::van_der_pol(EPS, false);
    let mut y0 = Vector::from(&[2.0, 0.0]);

    // set configuration parameters
    let mut params = Params::new(Method::DoPri5);
    params.stiffness.enabled = true;
    params.stiffness.stop_with_error = false;
    params.stiffness.save_results = true;
    params.step.h_ini = 1e-4;
    params.set_tolerances(1e-5, 1e-5, None)?;

    // allocate the solver
    let mut solver = OdeSolver::new(params, &system)?;

    // enable step and dense output
    let h_out = 0.01;
    let selected_y_components = &[0, 1];
    solver
        .enable_output()
        .set_step_recording(true, selected_y_components)
        .set_dense_recording(true, h_out, selected_y_components)?;

    // solve the problem
    solver.solve(&mut y0, x0, x1, None, &mut args)?;
    println!("y =\n{}", y0);

    // print stats
    println!("{}", solver.stats());

    // plot the results
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    let mut curve3 = Curve::new();
    let mut curve4 = Curve::new();
    curve1
        .set_line_color("black")
        .draw(&solver.out().dense_x, solver.out().dense_y.get(&0).unwrap());
    curve2
        .set_marker_style(".")
        .set_marker_color("cyan")
        .draw(&solver.out().step_x, solver.out().step_y.get(&0).unwrap());
    curve3.set_line_color("red").set_line_style("--");
    for i in 0..solver.out().stiff_x.len() {
        curve3.draw_ray(solver.out().stiff_x[i], 0.0, RayEndpoint::Vertical);
    }
    let fac: Vec<_> = solver.out().stiff_h_times_rho.iter().map(|hr| hr / 3.3).collect();
    curve4.set_marker_style(".").draw(&solver.out().step_x, &fac);

    // save figure
    let mut plot = Plot::new();
    plot.set_subplot(2, 1, 1)
        .set_title("Van der Pol ($\\varepsilon = 0.003$) - DoPri5 - Tol = 1e-5")
        .add(&curve1)
        .add(&curve2)
        .add(&curve3)
        .set_xrange(x0, x1)
        .grid_and_labels("$x$", "$y_0$")
        .set_subplot(2, 1, 2)
        .set_log_y(true)
        .set_yrange(7e-3, 2.0)
        .add(&curve4)
        .add(&curve3)
        .set_xrange(x0, x1)
        .grid_and_labels("$x$", "$h\\,\\rho\\,/\\,3.3$")
        .set_figure_size_points(800.0, 500.0)
        .save("/tmp/russell_ode/van_der_pol_dopri5.svg")
}
