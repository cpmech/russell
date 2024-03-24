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
    let (system, data, mut args) = Samples::robertson();

    // parameters
    let rel_tol = 1e-2;
    let abs_tol = 1e-6 * rel_tol;
    let mut params1 = Params::new(Method::Radau5);
    let mut params2 = Params::new(Method::DoPri5);
    let mut params3 = Params::new(Method::DoPri5);
    params1.set_tolerances(abs_tol, rel_tol, None)?;
    params2.set_tolerances(abs_tol, rel_tol, None)?;
    let rel_tol = 1e-3;
    let abs_tol = 1e-6 * rel_tol;
    params3.set_tolerances(abs_tol, rel_tol, None)?;

    // disable Lund stabilization for DoPri5
    params2.erk.lund_beta = 0.0;
    params3.erk.lund_beta = 0.0;

    // solvers
    let mut radau5 = OdeSolver::new(params1, &system)?;
    let mut dopri5 = OdeSolver::new(params2, &system)?;

    // selected component for output
    let sel = 1;

    // solve the problem with Radau5
    let mut out1 = Output::new();
    out1.enable_step(&[sel]);
    let mut y = data.y0.clone();
    radau5.solve(&mut y, data.x0, data.x1, None, Some(&mut out1), &mut args)?;
    println!("{}", radau5.stats());
    let n_accepted1 = radau5.stats().n_accepted;

    // solve the problem with DoPri5 and Tol = 1e-2
    let mut out2 = Output::new();
    out2.enable_step(&[sel]);
    let mut y = data.y0.clone();
    dopri5.solve(&mut y, data.x0, data.x1, None, Some(&mut out2), &mut args)?;
    println!("\nTol = 1e-2\n{}", dopri5.stats());
    let n_accepted2 = dopri5.stats().n_accepted;

    // solve the problem with DoPri5 and Tol = 1e-3
    let mut out3 = Output::new();
    out3.enable_step(&[sel]);
    let mut y = data.y0.clone();
    dopri5.update_params(params3)?;
    dopri5.solve(&mut y, data.x0, data.x1, None, Some(&mut out3), &mut args)?;
    println!("\nTol = 1e-3\n{}", dopri5.stats());
    let n_accepted3 = dopri5.stats().n_accepted;

    // Radau5 curve
    let mut curve1 = Curve::new();
    curve1
        .set_label(format!("Radau5, n_accepted = {}", n_accepted1).as_str())
        .set_marker_style("o")
        .draw(&out1.step_x, out1.step_y.get(&sel).unwrap());

    // DoPri5 curves
    let mut curve2 = Curve::new();
    let mut curve3 = Curve::new();
    let mut curve4 = Curve::new();
    curve2
        .set_label(format!("DoPri5, Tol = 1e-2, n_accepted = {}", n_accepted2).as_str())
        .draw(&out2.step_x, out2.step_y.get(&sel).unwrap());
    curve3
        .set_label(format!("DoPri5, Tol = 1e-3, n_accepted = {}", n_accepted3).as_str())
        .draw(&out3.step_x, out3.step_y.get(&sel).unwrap());
    curve4.draw(&out3.step_x, &out3.step_h);

    // save figures
    let mut plot1 = Plot::new();
    plot1
        .set_title("Robertson - second element of the solution")
        .add(&curve1)
        .add(&curve2)
        .add(&curve3)
        .grid_and_labels("$x$", format!("$y_{}$", sel).as_str())
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
