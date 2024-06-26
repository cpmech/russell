use plotpy::{Curve, Legend, Plot};
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // data
    let uu = [3.0, 0.5, -4.5, -7.0];
    let (xa, xb) = (0.0, 1.0);

    // interpolant
    let nn_max = 100;
    let tol = 1e-8;
    let interp = InterpChebyshev::new_adapt_uu(nn_max, tol, xa, xb, &uu)?;
    let nn = interp.get_degree();

    // plot
    let xx = Vector::linspace(xa, xb, 201).unwrap();
    let yy_int = xx.get_mapped(|x| interp.eval(x).unwrap());
    let mut curve_int = Curve::new();
    curve_int
        .set_label(&format!("interpolated,N={}", nn))
        .set_line_style(":")
        .set_marker_style(".")
        .set_marker_every(5);
    curve_int.draw(xx.as_data(), yy_int.as_data());
    let mut plot = Plot::new();
    let mut legend = Legend::new();
    legend.set_num_col(4);
    legend.set_outside(true);
    legend.draw();
    plot.add(&curve_int)
        .add(&legend)
        .set_cross(0.0, 0.0, "gray", "-", 1.5)
        .grid_and_labels("x", "f(x)")
        .save("/tmp/russell_lab/algo_interp_chebyshev_data.svg")
        .unwrap();
    Ok(())
}
