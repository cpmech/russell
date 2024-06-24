use plotpy::{Curve, Plot};
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // function
    let f = |x| 1.0 / (1.0 + 16.0 * x * x);

    // interpolant
    let degree = 10;
    let npoint = degree + 1;
    let interp = InterpLagrange::new(degree, None).unwrap();

    // compute data points
    let mut uu = Vector::new(npoint);
    for (i, x) in interp.get_points().into_iter().enumerate() {
        uu[i] = f(*x);
    }

    // plot
    let xx = Vector::linspace(-1.0, 1.0, 101).unwrap();
    let y_original = xx.get_mapped(|x| f(x));
    let y_approx = xx.get_mapped(|x| interp.eval(x, &uu).unwrap());
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    curve1
        .set_label("exact")
        .set_line_width(2.0)
        .draw(xx.as_data(), y_original.as_data());
    curve2
        .set_label("interpolated")
        .set_line_style("--")
        .set_marker_style(".")
        .draw(xx.as_data(), y_approx.as_data());
    let mut plot = Plot::new();
    let grid = format!("{:?}", interp.get_grid_type());
    let path = "/tmp/russell_lab/algo_interp_lagrange.svg";
    plot.add(&curve1)
        .add(&curve2)
        .legend()
        .set_title(&format!("N = {}; {}", degree, grid))
        .grid_and_labels("$x$", "$f(x)$")
        .save(path)
        .unwrap();

    Ok(())
}
