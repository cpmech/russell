use plotpy::{Curve, Legend, Plot};
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // data
    let (xa, xb) = (0.0, 1.0);
    let dx = xb - xa;
    let uu = Vector::from(&[3.0, 0.5, -4.5, -7.0]);
    let np = uu.dim(); // number of points
    let nn = np - 1; // degree
    let mut xx_dat = Vector::new(np);
    let zz = InterpChebyshev::points(nn);
    for i in 0..np {
        xx_dat[i] = (xb + xa + dx * zz[i]) / 2.0;
    }

    // interpolant
    let nn_max = 100;
    let tol = 1e-8;
    let interp = InterpChebyshev::new_adapt_uu(nn_max, tol, xa, xb, uu.as_data())?;
    let nn = interp.get_degree();

    // plot
    let xx = Vector::linspace(xa, xb, 201).unwrap();
    let yy_int = xx.get_mapped(|x| interp.eval(x).unwrap());
    let mut curve_dat = Curve::new();
    let mut curve_int = Curve::new();
    curve_dat
        .set_label("data")
        .set_line_style("None")
        .set_marker_style("o")
        .set_marker_void(true);
    curve_int
        .set_label(&format!("interpolated,N={}", nn))
        .set_marker_every(5);
    curve_dat.draw(xx_dat.as_data(), uu.as_data());
    curve_int.draw(xx.as_data(), yy_int.as_data());
    let mut plot = Plot::new();
    let mut legend = Legend::new();
    legend.set_num_col(4);
    legend.set_outside(true);
    legend.draw();
    plot.add(&curve_dat)
        .add(&curve_int)
        .add(&legend)
        .set_cross(0.0, 0.0, "gray", "-", 1.5)
        .grid_and_labels("x", "f(x)")
        .save("/tmp/russell_lab/algo_interp_chebyshev_data.svg")
        .unwrap();
    Ok(())
}
