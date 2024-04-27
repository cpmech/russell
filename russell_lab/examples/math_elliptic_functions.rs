use plotpy::{Curve, Plot};
use russell_lab::math::{GOLDEN_RATIO, PI};
use russell_lab::*;

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    // F
    let xx = Vector::linspace(0.0, PI / 2.0, 101)?;
    let yy = xx.get_mapped(|x| math::elliptic_f(x, 0.5).unwrap());
    let mut curve = Curve::new();
    curve
        .set_line_color("#E9708E")
        .set_line_width(2.5)
        .draw(xx.as_data(), yy.as_data());
    let path = format!("{}/math_elliptic_functions_f.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        .grid_labels_legend("$\\phi$", "$F(\\phi,1/2)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // E
    let xx = Vector::linspace(0.0, PI / 2.0, 101)?;
    let yy = xx.get_mapped(|x| math::elliptic_e(x, 0.5).unwrap());
    let mut curve = Curve::new();
    curve
        .set_line_color("#4C689C")
        .set_line_width(2.5)
        .draw(xx.as_data(), yy.as_data());
    let path = format!("{}/math_elliptic_functions_e.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        .grid_labels_legend("$\\phi$", "$E(\\phi,1/2)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // Π
    let xx = Vector::linspace(0.0, 0.9 * PI / 2.0, 101)?;
    let yy = xx.get_mapped(|x| math::elliptic_pi(1.0, x, 0.5).unwrap());
    let mut curve = Curve::new();
    curve
        .set_line_color("#58B090")
        .set_line_width(2.5)
        .draw(xx.as_data(), yy.as_data());
    let path = format!("{}/math_elliptic_functions_pi.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        .grid_labels_legend("$\\phi$", "$\\Pi(1,\\phi,1/2)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    Ok(())
}
