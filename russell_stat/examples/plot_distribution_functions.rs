use plotpy::{Curve, Plot, SuperTitleParams};
use russell_lab::*;
use russell_stat::{
    DistributionFrechet, DistributionGumbel, DistributionLognormal, DistributionNormal, DistributionUniform,
    ProbabilityDistribution,
};

const OUT_DIR: &str = "/tmp/russell_stat/";

fn main() -> Result<(), StrError> {
    // Frechet
    let (location, scale, shape) = (0.0, 1.0, 3.0);
    let dist = DistributionFrechet::new(location, scale, shape)?;
    let xx = Vector::linspace(0.0, 4.0, 201)?;
    let pdf = xx.get_mapped(|x| dist.pdf(x));
    let cdf = xx.get_mapped(|x| dist.cdf(x));
    let mut curve_pdf = Curve::new();
    let mut curve_cdf = Curve::new();
    curve_pdf.set_line_color("#976ED7").set_line_width(2.5);
    curve_cdf.set_line_color("#C23B23").set_line_width(2.5);
    curve_pdf.draw(xx.as_data(), pdf.as_data());
    curve_cdf.draw(xx.as_data(), cdf.as_data());
    let mut plot = Plot::new();
    let title = format!(
        "Frechet: location(min) = {}, scale = {}, shape = {}",
        location, scale, shape
    );
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

    // Gumbel
    let (location, scale) = (1.0, 2.0);
    let dist = DistributionGumbel::new(location, scale)?;
    let xx = Vector::linspace(-5.0, 15.0, 201)?;
    let pdf = xx.get_mapped(|x| dist.pdf(x));
    let cdf = xx.get_mapped(|x| dist.cdf(x));
    let mut curve_pdf = Curve::new();
    let mut curve_cdf = Curve::new();
    curve_pdf.set_line_color("#976ED7").set_line_width(2.5);
    curve_cdf.set_line_color("#C23B23").set_line_width(2.5);
    curve_pdf.draw(xx.as_data(), pdf.as_data());
    curve_cdf.draw(xx.as_data(), cdf.as_data());
    let mut plot = Plot::new();
    let title = format!("Gumbel: location = {}, scale = {}", location, scale);
    let path = format!("{}/plot_distribution_functions_gumbel.svg", OUT_DIR);
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

    // Lognormal
    let (mu_logx, sig_logx) = (0.0, 0.25);
    let dist = DistributionLognormal::new(mu_logx, sig_logx)?;
    let xx = Vector::linspace(0.0, 3.0, 201)?;
    let pdf = xx.get_mapped(|x| dist.pdf(x));
    let cdf = xx.get_mapped(|x| dist.cdf(x));
    let mut curve_pdf = Curve::new();
    let mut curve_cdf = Curve::new();
    curve_pdf.set_line_color("#976ED7").set_line_width(2.5);
    curve_cdf.set_line_color("#C23B23").set_line_width(2.5);
    curve_pdf.draw(xx.as_data(), pdf.as_data());
    curve_cdf.draw(xx.as_data(), cdf.as_data());
    let mut plot = Plot::new();
    let title = format!("Lognormal: mu_logx = {}, sig_logx = {}", mu_logx, sig_logx);
    let path = format!("{}/plot_distribution_functions_lognormal.svg", OUT_DIR);
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

    // Normal
    let (mu, sig) = (0.0, 1.0);
    let dist = DistributionNormal::new(mu, sig)?;
    let xx = Vector::linspace(-4.0, 4.0, 201)?;
    let pdf = xx.get_mapped(|x| dist.pdf(x));
    let cdf = xx.get_mapped(|x| dist.cdf(x));
    let mut curve_pdf = Curve::new();
    let mut curve_cdf = Curve::new();
    curve_pdf.set_line_color("#976ED7").set_line_width(2.5);
    curve_cdf.set_line_color("#C23B23").set_line_width(2.5);
    curve_pdf.draw(xx.as_data(), pdf.as_data());
    curve_cdf.draw(xx.as_data(), cdf.as_data());
    let mut plot = Plot::new();
    let title = format!("Normal: mu = {}, sig = {}", mu, sig);
    let path = format!("{}/plot_distribution_functions_normal.svg", OUT_DIR);
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

    // Uniform
    let (xmin, xmax) = (1.0, 2.0);
    let dist = DistributionUniform::new(xmin, xmax)?;
    let xx = Vector::linspace(0.0, 3.0, 201)?;
    let pdf = xx.get_mapped(|x| dist.pdf(x));
    let cdf = xx.get_mapped(|x| dist.cdf(x));
    let mut curve_pdf = Curve::new();
    let mut curve_cdf = Curve::new();
    curve_pdf.set_line_color("#976ED7").set_line_width(2.5);
    curve_cdf.set_line_color("#C23B23").set_line_width(2.5);
    curve_pdf.draw(xx.as_data(), pdf.as_data());
    curve_cdf.draw(xx.as_data(), cdf.as_data());
    let mut plot = Plot::new();
    let title = format!("Uniform: xmin = {}, xmax = {}", xmin, xmax);
    let path = format!("{}/plot_distribution_functions_uniform.svg", OUT_DIR);
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
