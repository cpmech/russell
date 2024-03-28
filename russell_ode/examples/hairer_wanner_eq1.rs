use plotpy::{Curve, Plot};
use russell_lab::{StrError, Vector};
use russell_ode::prelude::*;

// This example solves Eq (1.1) of Hairer-Wanner' book (Part II) on page 2
//
// This example corresponds to Fig 1.1 and Fig 1.2 on page 2 of the reference.
// The problem is defined in Eq (1.1) on page 2 of the reference.
//
// This example solves the problem with the BwEuler and FwEuler using
// a range of stepsizes to illustrate the instability behavior.
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
    let (system, x0, y0, mut args, y_fn_x) = Samples::hairer_wanner_eq1();

    // final x
    let x1 = 1.5;

    // solvers
    let mut bweuler = OdeSolver::new(Params::new(Method::BwEuler), &system)?;
    let mut fweuler = OdeSolver::new(Params::new(Method::FwEuler), &system)?;

    // solve the problem with BwEuler and h = 0.5
    let mut out1 = Output::new();
    out1.set_step_recording(true, &[0]);
    let h = 0.5;
    let mut y = y0.clone();
    bweuler.solve(&mut y, x0, x1, Some(h), Some(&mut out1), &mut args)?;

    // solve the problem with FwEuler and h = 1.974/50.0
    let mut out2 = Output::new();
    out2.set_step_recording(true, &[0]);
    let h = 1.974 / 50.0;
    let mut y = y0.clone();
    fweuler.solve(&mut y, x0, x1, Some(h), Some(&mut out2), &mut args)?;

    // solve the problem with FwEuler and h = 1.875/50.0
    let mut out3 = Output::new();
    out3.set_step_recording(true, &[0]);
    let h = 1.875 / 50.0;
    let mut y = y0.clone();
    fweuler.solve(&mut y, x0, x1, Some(h), Some(&mut out3), &mut args)?;

    // analytical solution
    let mut y_aux = Vector::new(system.get_ndim());
    let mut x_ana1 = Vector::linspace(0.0, 0.2, 20)?; // small stepsizes
    let mut x_ana2 = Vector::linspace(0.201, x1, 20)?; // larger stepsizes
    x_ana1.as_mut_data().append(x_ana2.as_mut_data()); // merge the two vectors
    let x_ana = Vector::from(x_ana1.as_data());
    let y_ana = x_ana.get_mapped(|x| {
        y_fn_x(&mut y_aux, x, &mut args);
        y_aux[0]
    });
    let mut curve0 = Curve::new();
    curve0.set_label("analytical").draw(x_ana.as_data(), y_ana.as_data());

    // BwEuler curve
    let mut curve1 = Curve::new();
    curve1
        .set_label("BwEuler h = 0.5")
        .draw(&out1.step_x, out1.step_y.get(&0).unwrap());

    // FwEuler curves
    let mut curve2 = Curve::new();
    let mut curve3 = Curve::new();
    curve2
        .set_label("FwEuler h = 1.974/50")
        .draw(&out2.step_x, out2.step_y.get(&0).unwrap());
    curve3
        .set_label("FwEuler h = 1.875/50")
        .draw(&out3.step_x, out3.step_y.get(&0).unwrap());

    // save figure
    let mut plot = Plot::new();
    plot.set_subplot(1, 2, 1)
        .set_title("Backward Euler")
        .grid_and_labels("$x$", "$y_0$")
        .add(&curve0)
        .add(&curve1)
        .legend()
        .set_subplot(1, 2, 2)
        .set_title("Forward Euler")
        .grid_and_labels("$x$", "$y_0$")
        .add(&curve0)
        .add(&curve2)
        .add(&curve3)
        .legend()
        .set_super_title("Hairer-Wanner Part II Eq(1.1)", None)
        .set_figure_size_points(800.0, 400.0)
        .save("/tmp/russell_ode/hairer_wanner_eq1.svg")
}
