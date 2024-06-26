use plotpy::{Curve, Plot};
use russell_lab::StrError;
use russell_ode::prelude::*;

// This example solves Robertson's equation
//
// This example corresponds to Fig 1.3 on page 4 of the reference.
// The problem is defined in Eq (1.4) on page 3 of the reference.
//
// This example shows how to enable the output of accepted steps.
//
// # Reference
//
// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
//   Stiff and Differential-Algebraic Problems. Second Revised Edition.
//   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p

fn main() -> Result<(), StrError> {
    // get the ODE system
    let (system, x0, y0, mut args) = Samples::robertson();

    // final x
    let x1 = 0.3;

    // parameters
    let h_ini = 1e-6;
    let rel_tol = 1e-2;
    let abs_tol = 1e-6 * rel_tol;
    let mut params1 = Params::new(Method::Radau5);
    let mut params2 = Params::new(Method::DoPri5);
    let mut params3 = Params::new(Method::DoPri5);
    params1.step.h_ini = h_ini;
    params2.step.h_ini = h_ini;
    params3.step.h_ini = h_ini;
    params1.set_tolerances(abs_tol, rel_tol, None)?;
    params2.set_tolerances(abs_tol, rel_tol, None)?;
    let rel_tol = 1e-3;
    let abs_tol = 1e-6 * rel_tol;
    params3.set_tolerances(abs_tol, rel_tol, None)?;

    // disable Lund stabilization for DoPri5
    params2.erk.lund_beta = 0.0;
    params3.erk.lund_beta = 0.0;

    // solvers
    let mut radau5 = OdeSolver::new(params1, system.clone())?;
    let mut dopri5 = OdeSolver::new(params2, system)?;

    // selected component for output
    let sel = 1;

    // solve the problem with Radau5
    radau5.enable_output().set_step_recording(&[sel]);
    let mut y = y0.clone();
    radau5.solve(&mut y, x0, x1, None, &mut args)?;
    println!("{}", radau5.stats());
    let n_accepted1 = radau5.stats().n_accepted;

    // solve the problem with DoPri5 and Tol = 1e-2
    dopri5.enable_output().set_step_recording(&[sel]);
    let mut y = y0.clone();
    dopri5.solve(&mut y, x0, x1, None, &mut args)?;
    println!("\nTol = 1e-2\n{}", dopri5.stats());
    let n_accepted2 = dopri5.stats().n_accepted;

    // save the results for later
    let out2_x = dopri5.out_step_x().clone();
    let out2_y = dopri5.out_step_y(sel).clone();

    // solve the problem again with DoPri5 and Tol = 1e-3
    let mut y = y0.clone();
    dopri5.update_params(params3)?;
    dopri5.solve(&mut y, x0, x1, None, &mut args)?;
    println!("\nTol = 1e-3\n{}", dopri5.stats());
    let n_accepted3 = dopri5.stats().n_accepted;

    // Radau5 curve
    let mut curve1 = Curve::new();
    curve1
        .set_label(&format!("Radau5, n_accepted = {}", n_accepted1))
        .set_marker_style("o")
        .draw(radau5.out_step_x(), radau5.out_step_y(sel));

    // DoPri5 curves
    let mut curve2 = Curve::new();
    let mut curve3 = Curve::new();
    let mut curve4 = Curve::new();
    curve2
        .set_label(&format!("DoPri5, Tol = 1e-2, n_accepted = {}", n_accepted2))
        .draw(&out2_x, &out2_y);
    curve3
        .set_label(&format!("DoPri5, Tol = 1e-3, n_accepted = {}", n_accepted3))
        .draw(dopri5.out_step_x(), dopri5.out_step_y(sel));
    curve4.draw(dopri5.out_step_x(), dopri5.out_step_h());

    // save figures
    let mut plot1 = Plot::new();
    plot1
        .set_title("Robertson - second element of the solution")
        .add(&curve1)
        .add(&curve2)
        .add(&curve3)
        .grid_and_labels("$x$", &format!("$y_{}$", sel))
        .legend()
        .set_ymin(0.000032)
        .set_figure_size_points(600.0, 400.0)
        .save("/tmp/russell_ode/robertson_a.svg")?;

    let mut plot2 = Plot::new();
    plot2
        .set_title("Robertson - step sizes - DoPri5 - Tol = 1e-2")
        .add(&curve4)
        .set_ymin(0.001)
        .grid_and_labels("$x$", "$h$")
        .set_figure_size_points(600.0, 200.0)
        .save("/tmp/russell_ode/robertson_b.svg")
}
