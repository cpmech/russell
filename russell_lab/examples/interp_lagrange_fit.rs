use plotpy::{Curve, Plot};
use russell_lab::{algo, math, StrError, Vector};

fn main() -> Result<(), StrError> {
    // generate data
    let (a, b) = (0.0, 4.0 * math::PI);
    let data_x = Vector::linspace(a, b, 10)?;
    let data_y = data_x.get_mapped(|x| f64::sin(x));

    // calc mapped data_y over [-1, 1]
    let nn = 7;
    let npoint = nn + 1;
    let params = algo::InterpParams::new(nn)?;
    let interp = algo::InterpLagrange::new(params)?;
    let mut uu = Vector::new(npoint);
    for (i, ksi) in interp.get_points().into_iter().enumerate() {
        let x = a + (ksi + 1.0) * (b - a) / 2.0;
        uu[i] = f64::sin(x);
    }

    // evaluate the polynomial over [a, b]
    let n_poly = 100;
    let poly_x = Vector::linspace(a, b, n_poly)?;
    let mut poly_y = Vector::new(n_poly);
    for i in 0..n_poly {
        let ksi = 2.0 * (poly_x[i] - a) / (b - a) - 1.0;
        poly_y[i] = interp.eval(ksi, &uu)?;
    }

    // plot
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    curve1
        .set_line_style("None")
        .set_marker_style("o")
        .set_marker_void(true)
        .draw(data_x.as_data(), data_y.as_data());
    curve2.draw(poly_x.as_data(), poly_y.as_data());
    let mut plot = Plot::new();
    plot.add(&curve1)
        .add(&curve2)
        .save("/tmp/russell_lab/ex_interp_lagrange_fit.svg")?;
    Ok(())
}
