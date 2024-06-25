use plotpy::{Curve, Legend, Plot};
use russell_lab::math::PI;
use russell_lab::*;
use std::fmt::Write;

fn main() -> Result<(), StrError> {
    // function
    let f = |x, _: &mut NoArgs| Ok(1.0 / (1.0 - f64::exp(-2.0 * x) * f64::powi(f64::sin(5.0 * PI * x), 2)) - 1.5);
    let (xa, xb) = (0.0, 1.0);
    let args = &mut 0;

    // adaptive interpolation
    let nn_max = 200;
    let tol = 1e-8;
    let interp = InterpChebyshev::new_adapt(nn_max, tol, xa, xb, args, f)?;
    let nn = interp.get_degree();
    println!("N = {}", nn);

    // find all roots in the interval
    let mut solver = MultiRootSolverCheby::new(nn)?;
    let roots = Vector::from(&solver.find(&interp)?);
    let f_at_roots = roots.get_mapped(|x| f(x, args).unwrap());
    println!("roots =\n{}", roots);
    println!("f @ roots =\n{}", print_vec_exp(&f_at_roots));

    // polish the roots
    let mut roots_polished = Vector::new(roots.dim());
    solver.polish_roots_newton(roots_polished.as_mut_data(), roots.as_data(), xa, xb, args, f)?;
    let f_at_roots_polished = roots_polished.get_mapped(|x| f(x, args).unwrap());
    println!("polished roots =\n{}", roots_polished);
    println!("f @ polished roots =\n{}", print_vec_exp(&f_at_roots_polished));

    // plot the results
    let nstation = 301;
    let xx = Vector::linspace(xa, xb, nstation).unwrap();
    let yy_ana = xx.get_mapped(|x| f(x, args).unwrap());
    let yy_int = xx.get_mapped(|x| interp.eval(x).unwrap());
    let mut curve_ana = Curve::new();
    let mut curve_int = Curve::new();
    let mut zeros_unpolished = Curve::new();
    let mut zeros_polished = Curve::new();
    curve_ana.set_label("analytical");
    curve_int
        .set_label("interpolated")
        .set_line_style("--")
        .set_marker_style(".")
        .set_marker_every(5);
    zeros_unpolished
        .set_marker_style("o")
        .set_marker_void(true)
        .set_marker_line_color("#00760F")
        .set_line_style("None");
    zeros_polished
        .set_marker_style("s")
        .set_marker_size(10.0)
        .set_marker_void(true)
        .set_marker_line_color("#00760F")
        .set_line_style("None");
    for root in &roots {
        zeros_unpolished.draw(&[*root], &[interp.eval(*root).unwrap()]);
    }
    for root in &roots_polished {
        zeros_polished.draw(&[*root], &[f(*root, args).unwrap()]);
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
        .add(&zeros_unpolished)
        .add(&zeros_polished)
        .add(&legend)
        .set_cross(0.0, 0.0, "gray", "-", 1.5)
        .grid_and_labels("x", "f(x)")
        .save("/tmp/russell_lab/algo_multi_root_solver_cheby.svg")
        .unwrap();
    Ok(())
}

fn print_vec_exp(v: &Vector) -> String {
    let mut buf = String::new();
    for x in v {
        writeln!(&mut buf, "{:>9.2e}", x).unwrap();
    }
    buf
}
