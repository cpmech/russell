use plotpy::{Curve, Plot, RayEndpoint};
use russell_lab::algo::linear_fitting;
use russell_lab::math::GOLDEN_RATIO;
use russell_lab::{approx_eq, StrError, Vector};

const OUT_DIR: &str = "/tmp/russell_lab/";

// Reference: Mathematica
// data = {{0, 1}, {1, 0}, {3, 2}, {5, 4}};
// lm = LinearModelFit[data, x, x]
// NumberForm[Normal[lm], 50]

fn main() -> Result<(), StrError> {
    // model: c is the y value @ x = 0; m is the slope
    let x = Vector::from(&[0.0, 1.0, 3.0, 5.0]);
    let y = Vector::from(&[1.0, 0.0, 2.0, 4.0]);
    let (c, m) = linear_fitting(&x, &y, false)?;
    println!("c = {}, m = {}", c, m);
    approx_eq(c, 0.1864406779661015, 1e-15);
    approx_eq(m, 0.6949152542372882, 1e-15);

    // plot
    let mut curve_dat = Curve::new();
    let mut curve_fit = Curve::new();
    curve_dat
        .set_label("data")
        .set_line_style("None")
        .set_marker_style("o")
        .set_marker_line_color("red")
        .set_marker_color("red");
    curve_fit.draw_ray(0.0, c, RayEndpoint::Slope(m));
    curve_dat.draw(x.as_data(), y.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/algo_linear_fitting_1.svg", OUT_DIR);
    plot.add(&curve_dat)
        .add(&curve_fit)
        .grid_and_labels("x", "y")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;
    Ok(())
}
