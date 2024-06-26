use plotpy::{Curve, Plot};
use russell_lab::{format_scientific, vec_max_abs_diff, StrError};
use russell_ode::prelude::*;

// This example solves the brusselator equation
//
// This example corresponds to Fig 16.4 on page 116 of the reference.
// The problem is defined in Eq (16.12) on page 116 of the reference.
//
// This example solves the problem a number of times with variable
// (automatic) steps and a few Runge-Kutta methods.
//
// # Reference
//
// * Hairer E, Nørsett, SP, Wanner G (2008) Solving Ordinary Differential Equations I.
//   Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
//   in Computational Mathematics, 528p

fn main() -> Result<(), StrError> {
    // ODE system
    let (system, x0, y0, mut args, y_ref) = Samples::brusselator_ode();

    // final x
    let x1 = 20.0;

    // tolerances
    let tols = [1e-2, 1e-4, 1e-6, 1e-8];

    // print the header of the table
    let (w1, w2) = (12, 10); // column widths
    println!("{:━<1$}", "", w1 + w2 * tols.len());
    print!("{:>w$}", "tol =", w = w1);
    for h in &tols {
        print!("{}", format_scientific(*h, w2, 2))
    }
    print!("\n{:>w$}", "Method", w = w1);
    println!("{}", format!("{:>w$}", "Error", w = w2).repeat(tols.len()));
    println!("{:─<1$}", "", w1 + w2 * tols.len());

    // allocate the plotting area
    let mut plot = Plot::new();

    // run for all explicit Runge-Kutta methods
    for method in &[Method::Radau5, Method::Merson4, Method::DoPri5, Method::DoPri8] {
        // allocate the solver
        let params = Params::new(*method);
        let mut solver = OdeSolver::new(params, system.clone())?;

        // arrays holding the results
        let mut n_f_eval = vec![0.0; tols.len()];
        let mut errors = vec![0.0; tols.len()];

        // solve the problem for a range of stepsizes
        let name = format!("{:?}", method);
        print!("{:>w$}", name, w = w1);
        for i in 0..tols.len() {
            // set the tolerances
            let mut params = Params::new(*method);
            params.set_tolerances(tols[i], tols[i], None)?;
            solver.update_params(params)?;

            // call solve
            let mut y = y0.clone();
            solver.solve(&mut y, x0, x1, None, &mut args).unwrap();

            // compare with the reference solution
            let (_, err) = vec_max_abs_diff(&y, &y_ref)?;
            print!("{}", format_scientific(err, w2, 1));

            // save the results
            n_f_eval[i] = solver.stats().n_function as f64;
            errors[i] = err;
        }
        println!();

        // plot the results
        let info = method.information();
        let mut curve = Curve::new();
        curve
            .set_label(&name)
            .set_marker_style(&format!("${}_{}$", name.get(..1).unwrap(), info.order))
            .set_marker_size(12.0)
            .draw(&n_f_eval, &errors);
        plot.add(&curve);
    }
    println!("{:━<1$}", "", w1 + w2 * tols.len());

    // save the plot
    plot.legend()
        .set_log_x(true)
        .set_log_y(true)
        .set_figure_size_points(600.0, 500.0)
        .grid_and_labels("N FUNCTION EVALUATIONS", "GLOBAL ERROR")
        .save("/tmp/russell_ode/brusselator_ode_var_step.svg")
}
