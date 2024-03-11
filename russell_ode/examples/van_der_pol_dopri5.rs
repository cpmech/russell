use plotpy::{Curve, Plot};
use russell_lab::{format_scientific, StrError};
use russell_ode::prelude::*;

fn main() -> Result<(), StrError> {
    // get the ODE system
    const EPS: f64 = 1e-3; // this problem would need many more steps with ε < 1e-3
    let (system, mut data, mut args) = Samples::van_der_pol(EPS, false);

    // solver
    let mut params = Params::new(Method::DoPri5);
    params.step.h_ini = 1e-6;
    params.set_tolerances(1e-5, 1e-5, None)?;
    let mut solver = OdeSolver::new(params, &system)?;

    // enable dense output
    let mut out = Output::new();
    let h_out = 0.01;
    let selected_y_components = &[0, 1];
    out.enable_dense(h_out, selected_y_components)?;

    // solve the problem
    solver.solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)?;
    println!("y =\n{}", data.y0);

    // print dense output
    let n_dense = out.dense_step_index.len();
    for i in 0..n_dense {
        println!(
            "step ={:>5}, x ={:5.2}, y ={}{}",
            out.dense_step_index[i],
            out.dense_x[i],
            format_scientific(out.dense_y.get(&0).unwrap()[i], 12, 4),
            format_scientific(out.dense_y.get(&1).unwrap()[i], 12, 4),
        );
    }

    // print stats
    println!("{}", solver.bench());

    // plot the results
    let mut curve1 = Curve::new();
    curve1.draw(&out.dense_x, &out.dense_y.get(&0).unwrap());

    // save figure
    let mut plot = Plot::new();
    plot.add(&curve1)
        .set_figure_size_points(600.0, 400.0)
        .grid_and_labels("x", "y0")
        .save("/tmp/russell_ode/van_der_pol_dopri5.svg")
}
