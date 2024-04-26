use plotpy::{Curve, Plot, SuperTitleParams};
use russell_lab::*;
use russell_stat::{DistributionFrechet, ProbabilityDistribution};

const OUT_DIR: &str = "/tmp/russell_stat/";

fn main() -> Result<(), StrError> {
    // x
    let xa = Vector::linspace(0.0, 4.0, 201)?;

    // Frechet
    let (location, scale, shape) = (0.0, 1.0, 3.0);
    let dist = DistributionFrechet::new(location, scale, shape)?;
    let pdf = xa.get_mapped(|x| dist.pdf(x));
    let cdf = xa.get_mapped(|x| dist.cdf(x));
    let mut curve_pdf = Curve::new();
    let mut curve_cdf = Curve::new();
    curve_pdf.set_line_color("#976ED7").set_line_width(2.5);
    curve_cdf.set_line_color("#C23B23").set_line_width(2.5);
    curve_pdf.draw(xa.as_data(), pdf.as_data());
    curve_cdf.draw(xa.as_data(), cdf.as_data());
    let mut plot = Plot::new();
    let title = format!("Frechet: location = {}, scale = {}, shape = {}", location, scale, shape);
    let path = format!("{}/plot_distribution_functions_frechet.svg", OUT_DIR);
    let mut params = SuperTitleParams::new();
    params.set_y(0.92);
    plot.set_subplot(2, 1, 1)
        .add(&curve_pdf)
        .grid_and_labels("$x$", "$PDF(x)$")
        .set_subplot(2, 1, 2)
        .add(&curve_cdf)
        .grid_and_labels("$x$", "$CDF(x)$")
        .set_super_title(&title, Some(params))
        .set_figure_size_points(350.0, 500.0)
        .save(&path)?;
    Ok(())
}
