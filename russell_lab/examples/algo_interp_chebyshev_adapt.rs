use plotpy::{Curve, Plot};
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // function
    let f = |x, _: &mut NoArgs| Ok(1.0 / (1.0 + 16.0 * x * x));
    let (xa, xb) = (-1.0, 1.0);

    // interpolant
    let nn_max = 200;
    let tol = 1e-8;
    let args = &mut 0;
    let interp = InterpChebyshev::new_adapt(nn_max, tol, xa, xb, args, f)?;
    println!("N = {}", interp.get_degree());

    // plot
    let xx = Vector::linspace(xa, xb, 101).unwrap();
    let y_original = xx.get_mapped(|x| f(x, args).unwrap());
    let y_approx = xx.get_mapped(|x| interp.eval(x).unwrap());
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
        .set_marker_every(2)
        .draw(xx.as_data(), y_approx.as_data());
    let mut plot = Plot::new();
    let path = "/tmp/russell_lab/algo_interp_chebyshev_adapt.svg";
    plot.add(&curve1)
        .add(&curve2)
        .legend()
        .set_title(&format!("N = {}", interp.get_degree()))
        .grid_and_labels("$x$", "$f(x)$")
        .save(path)
        .unwrap();
    Ok(())
}
