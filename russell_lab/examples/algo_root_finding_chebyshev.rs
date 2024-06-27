use plotpy::{Curve, Legend, Plot};
use russell_lab::math::PI;
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // function
    let f = |x, _: &mut NoArgs| Ok(1.0 / (1.0 - f64::exp(-2.0 * x) * f64::powi(f64::sin(5.0 * PI * x), 2)) - 1.5);
    let (xa, xb) = (0.0, 1.0);
    let args = &mut 0;

    // adaptive interpolation
    let nn_max = 200;
    let tol = 1e-8;
    let mut interp = InterpChebyshev::new(nn_max, xa, xb)?;
    interp.adapt_function(tol, args, f)?;
    let nn = interp.get_degree();
    println!("N = {}", nn);

    // find all roots in the interval
    let solver = RootFinder::new();
    let roots = Vector::from(&solver.chebyshev(&interp)?);
    let f_at_roots = roots.get_mapped(|x| f(x, args).unwrap());
    println!("roots =\n{}", roots);
    println!("f @ roots =\n{}", vec_fmt_scientific(&f_at_roots, 2));

    // refine/polish the roots
    let mut roots_refined = roots.clone();
    solver.refine(roots_refined.as_mut_data(), xa, xb, args, f)?;
    let f_at_roots_refined = roots_refined.get_mapped(|x| f(x, args).unwrap());
    println!("refined roots =\n{}", roots_refined);
    println!("f @ refined roots =\n{}", vec_fmt_scientific(&f_at_roots_refined, 2));

    // plot the results
    let nstation = 301;
    let xx = Vector::linspace(xa, xb, nstation).unwrap();
    let yy_ana = xx.get_mapped(|x| f(x, args).unwrap());
    let yy_int = xx.get_mapped(|x| interp.eval(x).unwrap());
    let mut curve_ana = Curve::new();
    let mut curve_int = Curve::new();
    let mut zeros = Curve::new();
    let mut zeros_refined = Curve::new();
    curve_ana.set_label("analytical");
    curve_int
        .set_label("interpolated")
        .set_line_style("--")
        .set_marker_style(".")
        .set_marker_every(5);
    zeros
        .set_marker_style("o")
        .set_marker_void(true)
        .set_marker_line_color("#00760F")
        .set_line_style("None");
    zeros_refined
        .set_marker_style("s")
        .set_marker_size(10.0)
        .set_marker_void(true)
        .set_marker_line_color("#00760F")
        .set_line_style("None");
    for root in &roots {
        zeros.draw(&[*root], &[interp.eval(*root).unwrap()]);
    }
    for root in &roots_refined {
        zeros_refined.draw(&[*root], &[f(*root, args).unwrap()]);
    }
    curve_int.draw(xx.as_data(), yy_int.as_data());
    curve_ana.draw(xx.as_data(), yy_ana.as_data());
    let mut plot = Plot::new();
    let mut legend = Legend::new();
    legend.set_num_col(2);
    legend.set_outside(true);
    legend.draw();
    plot.add(&curve_ana)
        .add(&curve_int)
        .add(&zeros)
        .add(&zeros_refined)
        .add(&legend)
        .set_cross(0.0, 0.0, "gray", "-", 1.5)
        .grid_and_labels("x", "f(x)")
        .save("/tmp/russell_lab/algo_root_finding_chebyshev.svg")
        .unwrap();
    Ok(())
}
