use plotpy::{Curve, Plot};
use russell_lab::math::GOLDEN_RATIO;
use russell_lab::*;

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    // values
    let np = 201;
    let xx = Vector::linspace(-2.0, 3.0, np)?;
    let mut bb = xx.get_mapped(|x| math::beta(0.5, x));
    let mut lb = xx.get_mapped(|x| math::ln_beta(0.5, x));
    let mut gg = xx.get_mapped(|x| math::gamma(x));
    let mut lg = xx.get_mapped(|x| math::ln_gamma(x).0);

    // replace ±Inf with NaN (ok in Matplotlib)
    bb.as_mut_data().iter_mut().for_each(|y| {
        if !y.is_finite() {
            *y = f64::NAN;
        }
    });
    lb.as_mut_data().iter_mut().for_each(|y| {
        if !y.is_finite() {
            *y = f64::NAN;
        }
    });
    gg.as_mut_data().iter_mut().for_each(|y| {
        if !y.is_finite() {
            *y = f64::NAN;
        }
    });
    lg.as_mut_data().iter_mut().for_each(|y| {
        if !y.is_finite() {
            *y = f64::NAN;
        }
    });

    // plot beta
    let mut curve = Curve::new();
    curve.set_line_color("#C23B23").set_line_width(2.5);
    curve.draw(xx.as_data(), bb.as_data());
    let path = format!("{}/math_beta_gamma_functions_bb.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        .set_yrange(-10.0, 10.0)
        .grid_labels_legend("$x$", "$B\\left(\\frac{1}{2},x\\right)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;

    // plot ln_beta
    let mut curve = Curve::new();
    curve.set_line_color("#976ED7").set_line_width(2.5);
    curve.draw(xx.as_data(), lb.as_data());
    let path = format!("{}/math_beta_gamma_functions_lb.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        .grid_labels_legend(
            "$x$",
            "${\\mathrm{Real}}\\left[\\log_e(B\\left(\\frac{1}{2},x\\right)\\right]$",
        )
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;

    // plot gamma
    let mut curve = Curve::new();
    curve.set_line_color("#579ABE").set_line_width(2.5);
    curve.draw(xx.as_data(), gg.as_data());
    let path = format!("{}/math_beta_gamma_functions_gg.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        .set_yrange(-20.0, 20.0)
        .grid_labels_legend("$x$", "$\\Gamma(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;

    // plot ln_gamma
    let mut curve = Curve::new();
    curve.set_line_color("#03C03C").set_line_width(2.5);
    curve.draw(xx.as_data(), lg.as_data());
    let path = format!("{}/math_beta_gamma_functions_lg.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve)
        // .set_yrange(-2.0, 3.0)
        .grid_labels_legend("$x$", "${\\mathrm{Real}}\\left[\\log_e(\\Gamma(x)\\right]$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;
    Ok(())
}
