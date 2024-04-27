use plotpy::{Curve, Plot, Text};
use russell_lab::math::GOLDEN_RATIO;
use russell_lab::*;

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    // x
    let xa = Vector::linspace(-1.0, 1.0, 101)?;
    let xb = Vector::linspace(0.0, 3.0, 3 * 40 + 1)?;
    let xc = Vector::linspace(-4.0, 4.0, 202)?;

    // sign(x)
    let y_sign = xa.get_mapped(|x| math::sign(x));
    let mut curve = Curve::new();
    curve.set_marker_style("o").draw(xa.as_data(), y_sign.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_sign.svg", OUT_DIR);
    plot.add(&curve)
        .grid_and_labels("$x$", "$\\mathrm{sign}(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // ramp(x)
    let y_ramp = xa.get_mapped(|x| math::ramp(x));
    let mut curve = Curve::new();
    curve.set_line_width(2.5).draw(xa.as_data(), y_ramp.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_ramp.svg", OUT_DIR);
    plot.add(&curve)
        .grid_and_labels("$x$", "$\\mathrm{ramp}(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // heaviside(x)
    let y_heaviside = xa.get_mapped(|x| math::heaviside(x));
    let mut curve = Curve::new();
    curve.set_marker_style("o").draw(xa.as_data(), y_heaviside.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_heaviside.svg", OUT_DIR);
    plot.add(&curve)
        .grid_and_labels("$x$", "$\\mathrm{heaviside}(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // boxcar(x)
    let (a, b) = (1.0, 2.0);
    let y_boxcar = xb.get_mapped(|x| math::boxcar(x, a, b));
    let mut curve = Curve::new();
    curve.set_marker_style("o").draw(xb.as_data(), y_boxcar.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_boxcar.svg", OUT_DIR);
    plot.add(&curve)
        .grid_and_labels("$x$", &format!("$\\mathrm{{boxcar}}_{{({},{})}}(x)$", a, b))
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // logistic(x)
    let y_logistic = xc.get_mapped(|x| math::logistic(x));
    let mut curve = Curve::new();
    curve.set_line_width(2.5).draw(xc.as_data(), y_logistic.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_logistic.svg", OUT_DIR);
    plot.add(&curve)
        .grid_and_labels("$x$", "$\\mathrm{logistic}(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // logistic_deriv1(x)
    let y_logistic_deriv1 = xc.get_mapped(|x| math::logistic_deriv1(x));
    let mut curve = Curve::new();
    curve
        .set_line_width(2.5)
        .draw(xc.as_data(), y_logistic_deriv1.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_logistic_deriv1.svg", OUT_DIR);
    plot.add(&curve)
        .grid_and_labels("$x$", "$\\frac{d\\mathrm{logistic}(x)}{dx}$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // smooth_ramp(x)
    let beta = 10.0;
    let y_smooth_ramp = xa.get_mapped(|x| math::smooth_ramp(x, beta));
    let mut curve = Curve::new();
    curve.set_line_width(2.5).draw(xa.as_data(), y_smooth_ramp.as_data());
    let mut text = Text::new();
    text.set_bbox(true)
        .set_bbox_facecolor("pink")
        .set_bbox_edgecolor("black");
    text.draw(-1.0, 0.85, &format!("$\\beta = {}$", beta));
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_smooth_ramp.svg", OUT_DIR);
    plot.add(&curve)
        .add(&text)
        .grid_and_labels("$x$", "$\\mathrm{SmoothRamp}(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // smooth_ramp_deriv1(x)
    let beta = 10.0;
    let y_smooth_ramp_deriv1 = xa.get_mapped(|x| math::smooth_ramp_deriv1(x, beta));
    let mut curve = Curve::new();
    curve
        .set_line_width(2.5)
        .draw(xa.as_data(), y_smooth_ramp_deriv1.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_smooth_ramp_deriv1.svg", OUT_DIR);
    plot.add(&curve)
        .add(&text)
        .grid_and_labels("$x$", "$\\frac{d\\mathrm{SmoothRamp}(x)}{dx}$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // smooth_ramp_deriv2(x)
    let beta = 10.0;
    let y_smooth_ramp_deriv2 = xa.get_mapped(|x| math::smooth_ramp_deriv2(x, beta));
    let mut curve = Curve::new();
    curve
        .set_line_width(2.5)
        .draw(xa.as_data(), y_smooth_ramp_deriv2.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_smooth_ramp_deriv2.svg", OUT_DIR);
    plot.add(&curve)
        .add(&text)
        .grid_and_labels("$x$", "$\\frac{d^2\\mathrm{SmoothRamp}(x)}{dx^2}$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    Ok(())
}
