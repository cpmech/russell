use plotpy::{Curve, Legend, Plot};
use russell_lab::math::GOLDEN_RATIO;
use russell_lab::*;

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    // generate curves
    let np = 201;
    let xx = Vector::linspace(-1.0, 1.0, np)?;
    let mut plot_yy = Plot::new();
    let mut plot_gg = Plot::new();
    let mut plot_hh = Plot::new();
    for n in 0..5 {
        let yy = xx.get_mapped(|x| math::chebyshev_tn(n, x));
        let gg = xx.get_mapped(|x| math::chebyshev_tn_deriv1(n, x));
        let hh = xx.get_mapped(|x| math::chebyshev_tn_deriv2(n, x));
        let mut curve_yy = Curve::new();
        let mut curve_gg = Curve::new();
        let mut curve_hh = Curve::new();
        curve_yy.set_label(&format!("$T_{}$", n));
        curve_gg.set_label(&format!("$\\frac{{dT_{}}}{{dx}}$", n));
        curve_hh.set_label(&format!("$\\frac{{d^2T_{}}}{{dx^2}}$", n));
        curve_yy.draw(xx.as_data(), yy.as_data());
        curve_gg.draw(xx.as_data(), gg.as_data());
        curve_hh.draw(xx.as_data(), hh.as_data());
        plot_yy.add(&curve_yy);
        plot_gg.add(&curve_gg);
        plot_hh.add(&curve_hh);
    }

    // save Tn figure
    let path = format!("{}/math_chebyshev_functions_tn.svg", OUT_DIR);
    let mut legend = Legend::new();
    legend.set_outside(true).set_num_col(5);
    legend.draw();
    plot_yy
        .add(&legend)
        .grid_and_labels("$x$", "$T_n(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // save dTn/dx figure
    let path = format!("{}/math_chebyshev_functions_dtn.svg", OUT_DIR);
    let mut legend = Legend::new();
    legend.set_outside(true).set_num_col(5);
    legend.draw();
    plot_gg
        .add(&legend)
        .grid_and_labels("$x$", "$\\frac{dT_n(x)}{dx}$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    // save d2Tn/dx2 figure
    let path = format!("{}/math_chebyshev_functions_d2tn.svg", OUT_DIR);
    let mut legend = Legend::new();
    legend.set_outside(true).set_num_col(5);
    legend.draw();
    plot_hh
        .add(&legend)
        .grid_and_labels("$x$", "$\\frac{dT^2_n(x)}{dx^2}$")
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;
    Ok(())
}
