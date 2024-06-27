use plotpy::{Curve, Legend, Plot};
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // generate data with noise
    let generator = |x: f64| f64::cos(16.0 * (x + 0.2)) * (1.0 + x) * f64::exp(x * x) / (1.0 + 9.0 * x * x);
    let (xa, xb) = (-1.0, 1.0);
    let dx = xb - xa;
    let nn_fit = 30;
    let np_fit = nn_fit + 1;
    let zz = InterpChebyshev::points(nn_fit);
    let mut xx_dat = Vector::new(np_fit);
    let mut uu = Vector::new(np_fit);
    let dy = 0.1;
    for i in 0..np_fit {
        let x = (xb + xa + dx * zz[i]) / 2.0;
        let noise = if i % 2 == 0 { dy } else { -dy };
        xx_dat[i] = x;
        uu[i] = generator(x) + noise;
    }

    // interpolant
    let nn_max = 100;
    let tol = 1e-8;
    let mut interp = InterpChebyshev::new(nn_max, xa, xb)?;
    interp.adapt_data(tol, uu.as_data())?;
    let nn = interp.get_degree();

    // plot
    let xx = Vector::linspace(xa, xb, 201).unwrap();
    let yy_ana = xx.get_mapped(|x| generator(x));
    let yy_int = xx.get_mapped(|x| interp.eval(x).unwrap());
    let mut curve_ana = Curve::new();
    let mut curve_int = Curve::new();
    let mut curve_dat = Curve::new();
    curve_ana.set_label("generator");
    curve_int
        .set_label(&format!("interpolated,N={}", nn))
        .set_line_style(":")
        .set_marker_style(".")
        .set_marker_every(5);
    curve_dat
        .set_label("noisy data")
        .set_line_style("None")
        .set_marker_style("+");
    curve_ana.draw(xx.as_data(), yy_ana.as_data());
    curve_int.draw(xx.as_data(), yy_int.as_data());
    curve_dat.draw(xx_dat.as_data(), uu.as_data());
    let mut plot = Plot::new();
    let mut legend = Legend::new();
    legend.set_num_col(4);
    legend.set_outside(true);
    legend.draw();
    plot.add(&curve_ana)
        .add(&curve_int)
        .add(&curve_dat)
        .add(&legend)
        .set_cross(0.0, 0.0, "gray", "-", 1.5)
        .grid_and_labels("x", "f(x)")
        .save("/tmp/russell_lab/algo_interp_chebyshev_noisy_data.svg")
        .unwrap();
    Ok(())
}
