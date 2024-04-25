use plotpy::{Curve, Plot};
use russell_lab::{InterpLagrange, StrError, Vector};

// Runge equation
// Reference:
// * Gourgoulhon E (2005), An introduction to polynomial interpolation,
//   School on spectral methods: Application to General Relativity and Field Theory
//   Meudon, 14-18 November 2005

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

    // stations for plotting
    let nstation = 20;
    let station = Vector::linspace(-1.0, 1.0, nstation).unwrap();

    // plot
    let x_original = Vector::linspace(-1.0, 1.0, 101).unwrap();
    let y_original = x_original.get_mapped(|x| f(x));
    let y_approx = station.get_mapped(|x| interp.eval(x, &uu).unwrap());
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    curve1
        .set_label("exact")
        .set_line_width(2.0)
        .draw(x_original.as_data(), y_original.as_data());
    curve2
        .set_label("interpolated")
        .set_line_style("--")
        .set_marker_style("o")
        .set_marker_void(true)
        .draw(station.as_data(), y_approx.as_data());
    let mut plot = Plot::new();
    let grid = format!("{:?}", interp.get_grid_type());
    let path = "/tmp/russell_lab/algo_interpolation_lagrange.svg";
    plot.add(&curve1)
        .add(&curve2)
        .legend()
        .set_title(format!("N = {}; {}", degree, grid).as_str())
        .grid_and_labels("$x$", "$f(x)$")
        .save(path)
        .unwrap();

    Ok(())
}
