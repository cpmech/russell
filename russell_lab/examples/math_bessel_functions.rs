use plotpy::{Curve, Plot};
use russell_lab::math::{bessel_j0, bessel_j1, bessel_jn, GOLDEN_RATIO};
use russell_lab::*;

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    // values
    let xx = Vector::linspace(0.0, 15.0, 101)?;
    let j0 = xx.get_mapped(|x| bessel_j0(x));
    let j1 = xx.get_mapped(|x| bessel_j1(x));
    let j2 = xx.get_mapped(|x| bessel_jn(2, x));
    // plot
    let mut curve_j0 = Curve::new();
    let mut curve_j1 = Curve::new();
    let mut curve_j2 = Curve::new();
    curve_j0.set_label("J0").draw(xx.as_data(), j0.as_data());
    curve_j1.set_label("J1").draw(xx.as_data(), j1.as_data());
    curve_j2.set_label("J2").draw(xx.as_data(), j2.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_bessel_functions_1.svg", OUT_DIR);
    plot.add(&curve_j0)
        .add(&curve_j1)
        .add(&curve_j2)
        .grid_labels_legend("$x$", "$J_0(x),\\,J_1(x),\\,J_2(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;
    Ok(())
}
