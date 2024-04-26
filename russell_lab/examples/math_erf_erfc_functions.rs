use plotpy::{Curve, Plot};
use russell_lab::math::GOLDEN_RATIO;
use russell_lab::*;

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    // values
    let np = 201;
    let xa = Vector::linspace(-5.0, 5.0, np)?;
    let xb = Vector::linspace(-1.0, 1.0, np)?;
    let xc = Vector::linspace(0.0, 2.0, np)?;
    let y_erf = xa.get_mapped(|x| math::erf(x));
    let y_erfc = xa.get_mapped(|x| math::erfc(x));
    let mut y_erf_inv = xb.get_mapped(|x| math::erf_inv(x));
    let mut y_erfc_inv = xc.get_mapped(|x| math::erfc_inv(x));

    // replace ±Inf with NaN (ok in Matplotlib)
    y_erf_inv.as_mut_data().iter_mut().for_each(|y| {
        if !y.is_finite() {
            *y = f64::NAN;
        }
    });
    y_erfc_inv.as_mut_data().iter_mut().for_each(|y| {
        if !y.is_finite() {
            *y = f64::NAN;
        }
    });

    // plot erf
    let mut curve = Curve::new();
    curve.set_line_color("#C23B23").set_line_width(2.5);
    curve.draw(xa.as_data(), y_erf.as_data());
    let path = format!("{}/math_erf_erfc_functions_erf.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        .grid_labels_legend("$x$", "$\\mathrm{erf}(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;

    // plot erfc
    let mut curve = Curve::new();
    curve.set_line_color("#579ABE").set_line_width(2.5);
    curve.draw(xa.as_data(), y_erfc.as_data());
    let path = format!("{}/math_erf_erfc_functions_erfc.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        .grid_labels_legend("$x$", "$\\mathrm{erfc}(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;

    // plot erf_inv
    let mut curve = Curve::new();
    curve.set_line_color("#976ED7").set_line_width(2.5);
    curve.draw(xb.as_data(), y_erf_inv.as_data());
    let path = format!("{}/math_erf_erfc_functions_erf_inv.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        .grid_labels_legend("$x$", "$\\mathrm{erf}^{-1}(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;

    // plot erfc_inv
    let mut curve = Curve::new();
    curve.set_line_color("#03C03C").set_line_width(2.5);
    curve.draw(xc.as_data(), y_erfc_inv.as_data());
    let path = format!("{}/math_erf_erfc_functions_erfc_inv.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        .grid_labels_legend("$x$", "$\\mathrm{erfc}^{-1}(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;
    Ok(())
}
