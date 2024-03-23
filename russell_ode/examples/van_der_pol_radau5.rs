use plotpy::{Curve, Plot};
use russell_lab::{StrError, Vector};
use russell_ode::prelude::*;

// This example solves Van der Pol's equation
//
// This example corresponds to Fig 8.1 on page 125 of the reference.
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
    const EPS: f64 = 1e-6;
    let (system, data, mut args) = Samples::van_der_pol(EPS, false);
    let mut y0 = Vector::from(&[2.0, -0.6]);

    // solver
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-4;
    params.set_tolerances(1e-4, 1e-4, None)?;
    let mut solver = OdeSolver::new(params, &system)?;

    // enable step output
    let mut out = Output::new();
    let selected_y_components = &[0, 1];
    out.enable_step(selected_y_components);

    // solve the problem
    solver.solve(&mut y0, data.x0, data.x1, None, Some(&mut out), &mut args)?;
    println!("y =\n{}", y0);

    // print stats
    println!("{}", solver.bench());

    // plot the results
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    curve1
        .set_marker_color("red")
        .set_marker_line_color("red")
        .set_marker_style(".")
        .draw(&out.step_x, out.step_y.get(&0).unwrap());
    curve2
        .set_marker_color("green")
        .set_marker_line_color("green")
        .set_marker_style(".")
        .draw(&out.step_x, &out.step_h);

    // save figure
    let mut plot = Plot::new();
    plot.set_subplot(2, 1, 1)
        .set_title("Van der Pol ($\\varepsilon = 10^{-6}$) - Radau5 - Tol = 1e-4")
        .add(&curve1)
        .set_xrange(data.x0, data.x1)
        .grid_and_labels("$x$", "$y_0$")
        .set_subplot(2, 1, 2)
        .set_log_y(true)
        .set_yrange(3e-7, 2.0)
        .add(&curve2)
        .set_xrange(data.x0, data.x1)
        .grid_and_labels("$x$", "$h$")
        .set_figure_size_points(800.0, 500.0)
        .save("/tmp/russell_ode/van_der_pol_radau5.svg")
}
