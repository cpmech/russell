use plotpy::{Curve, Plot};
use russell_lab::StrError;
use russell_ode::prelude::*;
use serde::Deserialize;
use std::{env, fs::File, io::BufReader, path::Path};

// This example solves the DAE representing a one-transistor amplifier
//
// This example corresponds to Fig 1.3 on page 377 and Fig 1.4 on page 379 of the reference.
// The problem is defined in Eq (1.14) on page 377 of the reference.
//
// # Reference
//
// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
//   Stiff and Differential-Algebraic Problems. Second Revised Edition.
//   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p

fn main() -> Result<(), StrError> {
    // get the ODE system
    let (system, x0, mut y0, mut args) = Samples::amplifier1t();

    // solver
    let params = Params::new(Method::Radau5);
    let mut solver = OdeSolver::new(params, &system)?;

    // enable dense output
    let mut out = Output::new();
    let h_out = 0.0001;
    let selected_y_components = &[0, 4];
    out.set_dense_recording(true, h_out, selected_y_components)?;

    // solve the problem
    let x1 = 0.2;
    solver.solve(&mut y0, x0, x1, None, Some(&mut out), &mut args)?;

    // print the results and stats
    let y_ref = &[
        -0.0222670928295124,
        3.068708898325494,
        2.898349447438977,
        1.4994388165471992,
        -1.735056659198201,
    ];
    println!("y_russell     = {:.6?}", y0.as_data());
    println!("y_mathematica = {:.6?}", y_ref);
    println!("{}", solver.stats());

    // load reference data (from Mathematica)
    let math = ReferenceData::read("data/reference/amplifier1t_mathematica.json")?;

    // plot the results
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    let mut curve3 = Curve::new();
    let mut curve4 = Curve::new();
    curve1.set_label("y0: russell");
    curve2.set_label("y0: mathematica");
    curve3.set_label("y4: russell");
    curve4.set_label("y4: mathematica");

    let blue = "#307BC2";
    let red = "#C23048";

    curve1.set_line_color(&blue);
    curve2
        .set_marker_color(&blue)
        .set_marker_line_color(&blue)
        .set_marker_style(".")
        .set_line_style("None");
    curve3.set_line_color(&red);
    curve4
        .set_marker_color(&red)
        .set_marker_line_color(&red)
        .set_marker_style("+")
        .set_line_style("None");

    curve1.draw(&out.dense_x, out.dense_y.get(&0).unwrap());
    curve2.draw(&math.x, &math.y0);
    curve3.draw(&out.dense_x, out.dense_y.get(&4).unwrap());
    curve4.draw(&math.x, &math.y4);

    // save figure
    let mut plot = Plot::new();
    plot.add(&curve1)
        .add(&curve2)
        .add(&curve3)
        .add(&curve4)
        .legend()
        .set_title("One-transistor Amplifier - Radau5")
        .set_figure_size_points(800.0, 400.0)
        .grid_and_labels("$y_0$", "$y_1$")
        .save("/tmp/russell_ode/amplifier1t_radau5.svg")
}

#[derive(Deserialize)]
struct ReferenceData {
    pub x: Vec<f64>,
    pub y0: Vec<f64>,
    pub y4: Vec<f64>,
}

impl ReferenceData {
    pub fn read(rel_path: &str) -> Result<Self, StrError> {
        let full_path = format!("{}/{}", env::var("CARGO_MANIFEST_DIR").unwrap(), rel_path);
        let path = Path::new(&full_path).to_path_buf();
        let input = File::open(path).map_err(|_| "cannot open file")?;
        let buffered = BufReader::new(input);
        let data = serde_json::from_reader(buffered).map_err(|_| "cannot parse JSON file")?;
        Ok(data)
    }
}
