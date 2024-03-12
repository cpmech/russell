use plotpy::{Curve, Plot, RayEndpoint};
use russell_lab::{StrError, Vector};
use russell_ode::prelude::*;

// This example solves Van der Pol's equation
//
// See Eq (1.5') of Hairer-Wanner' book (Part II) on page 5
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
    let (system, data, mut args) = Samples::van_der_pol(EPS, false);
    let mut y0 = Vector::from(&[2.0, 0.0]);

    // solver
    let mut params = Params::new(Method::DoPri5);
    params.stiffness.enabled = true;
    params.stiffness.stop_with_error = false;
    params.stiffness.save_results = true;
    params.step.h_ini = 1e-4;
    params.set_tolerances(1e-5, 1e-5, None)?;
    let mut solver = OdeSolver::new(params, &system)?;

    // enable step and dense output
    let mut out = Output::new();
    let h_out = 0.01;
    let selected_y_components = &[0, 1];
    out.enable_step(selected_y_components);
    out.enable_dense(h_out, selected_y_components)?;

    // solve the problem
    solver.solve(&mut y0, data.x0, data.x1, None, Some(&mut out), &mut args)?;
    println!("y =\n{}", y0);

    // print stats
    println!("{}", solver.bench());

    // print the stations where stiffness has been detected
    println!("{:?}", out.stiff_step_index);
    println!("{:?}", out.stiff_x);

    // plot the results
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    let mut curve3 = Curve::new();
    curve1.draw(&out.dense_x, out.dense_y.get(&0).unwrap());
    curve2.set_line_color("red").set_line_style("--");
    for i in 0..out.stiff_x.len() {
        curve2.draw_ray(out.stiff_x[i], 0.0, RayEndpoint::Vertical);
    }
    let fac: Vec<_> = out.stiff_h_times_rho.iter().map(|hr| hr / 3.3).collect();
    curve3.set_marker_style(".").draw(&out.step_x, &fac);

    // save figure
    let mut plot = Plot::new();
    plot.set_subplot(2, 1, 1)
        .set_title("Van der Pol - DoPri5 - Tol = 1e-5")
        .add(&curve1)
        .add(&curve2)
        .set_xrange(data.x0, data.x1)
        .grid_and_labels("$x$", "$y_0$")
        .set_subplot(2, 1, 2)
        .set_log_y(true)
        .set_yrange(7e-3, 2.0)
        .add(&curve3)
        .add(&curve2)
        .set_xrange(data.x0, data.x1)
        .grid_and_labels("$x$", "$h\\,\\rho\\,/\\,3.3$")
        .set_figure_size_points(800.0, 500.0)
        .save("/tmp/russell_ode/van_der_pol_dopri5.svg")
}
