use plotpy::{Curve, Plot, SlopeIcon};
use russell_lab::{format_scientific, vec_max_abs_diff, StrError, Vector};
use russell_ode::prelude::*;
use russell_sparse::CooMatrix;

fn main() -> Result<(), StrError> {
    // ODE system
    let ndim = 2;
    let jac_nnz = 4;
    let system = System::new(
        ndim,
        |f: &mut Vector, _x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = 1.0 - 4.0 * y[0] + y[0] * y[0] * y[1];
            f[1] = 3.0 * y[0] - y[0] * y[0] * y[1];
            Ok(())
        },
        |jj: &mut CooMatrix, _x: f64, y: &Vector, m: f64, _args: &mut NoArgs| {
            jj.reset();
            jj.put(0, 0, m * (-4.0 + 2.0 * y[0] * y[1])).unwrap();
            jj.put(0, 1, m * (y[0] * y[0])).unwrap();
            jj.put(1, 0, m * (3.0 - 2.0 * y[0] * y[1])).unwrap();
            jj.put(1, 1, m * (-y[0] * y[0])).unwrap();
            Ok(())
        },
        HasJacobian::Yes,
        Some(jac_nnz),
        None,
    );

    // initial values
    let x0 = 0.0;
    let y0 = Vector::from(&[3.0 / 2.0, 3.0]);

    // stepsize
    let mut args: NoArgs = 0;
    let x1 = 20.0;
    let hh = [0.2, 0.1, 0.05, 0.01, 0.001];

    // print the header of the table
    let (w1, w2) = (11, 9); // column widths
    println!("{:━<1$}", "", w1 + w2 * hh.len());
    print!("{:>w$}", "h =", w = w1);
    for h in &hh {
        print!("{}", format_scientific(*h, w2, 2))
    }
    print!("\n{:>w$}", "Method", w = w1);
    println!("{}", format!("{:>w$}", "Error", w = w2).repeat(hh.len()));
    println!("{:─<1$}", "", w1 + w2 * hh.len());

    // reference solution, from the following Mathematica code:
    // ```Mathematica
    // Needs["DifferentialEquations`NDSolveProblems`"];
    // Needs["DifferentialEquations`NDSolveUtilities`"];
    // sys = GetNDSolveProblem["BrusselatorODE"];
    // sol = NDSolve[sys, Method -> "StiffnessSwitching", WorkingPrecision -> 32];
    // ref = First[FinalSolutions[sys, sol]]
    // ```
    let y_ref = Vector::from(&[0.4986370712683478291402659846476, 4.596780349452011024598321237263]);

    // allocate the plotting area
    let mut plot = Plot::new();

    // run for all explicit Runge-Kutta methods
    // for method in [Method::FwEuler] {
    for method in Method::erk_methods() {
        // allocate the solver
        let params = Params::new(method);
        let mut solver = OdeSolver::new(params, &system)?;

        // arrays holding the results
        let mut n_f_eval = vec![0.0; hh.len()];
        let mut errors = vec![0.0; hh.len()];

        // solve the problem for a range of stepsizes
        let name = format!("{:?}", method);
        print!("{:>w$}", name, w = w1);
        for i in 0..hh.len() {
            let x = x0;
            let mut y = y0.clone();
            solver.solve(&mut y, x, x1, Some(hh[i]), None, &mut args).unwrap();

            // compare with reference solution
            let (_, err) = vec_max_abs_diff(&y, &y_ref)?;
            print!("{}", format_scientific(err, w2, 1));

            // save the results
            n_f_eval[i] = solver.bench().n_function as f64;
            errors[i] = err;
        }
        println!();

        // plot the results
        let info = method.information();
        let mut curve = Curve::new();
        curve
            .set_label(&name)
            .set_marker_style(format!("${}_{}$", name.get(..1).unwrap(), info.order).as_str())
            .set_marker_size(12.0)
            .draw(&n_f_eval, &errors);
        plot.add(&curve);
    }
    println!("{:━<1$}", "", w1 + w2 * hh.len());

    let mut icon2 = SlopeIcon::new();
    let mut icon3 = SlopeIcon::new();
    let mut icon4 = SlopeIcon::new();
    let mut icon8 = SlopeIcon::new();
    icon2.set_above(true).draw(-2.0, 2e4, 3e-6);
    icon3.set_above(true).draw(-3.0, 2e4, 1e-7);
    icon4.set_above(true).draw(-4.0, 2e4, 1.5e-9);
    icon8.draw(-8.0, 2.6e3, 1.5e-10);

    // save the plot
    plot.legend()
        .set_log_x(true)
        .set_log_y(true)
        .add(&icon2) // must add after set_log_x and set_log_y
        .add(&icon3)
        .add(&icon4)
        .add(&icon8)
        .set_figure_size_points(600.0, 500.0)
        .grid_and_labels("N FUNCTION EVALUATIONS", "ERROR")
        .save("/tmp/russell_ode/brusselator_ode_erk_methods.svg")
}
