use plotpy::{Curve, Plot};
use russell_lab::{approx_eq, math::PI, StrError, Vector};
use russell_ode::prelude::*;

const SAVE_FIGURE: bool = false;

#[test]
fn test_heat_1d_periodic() -> Result<(), StrError> {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    //  ∂u    ∂²u
    //  ——— = ——— + source(t, x)    with x ∈ [-1, 1)
    //  ∂t    ∂x²
    //
    // The source term is given by:
    //
    // source(t, x) = (25 π² - 1) exp(-t) cos(5 π x)
    //
    // Periodic boundary condition:
    //
    // u(t, -1) = u(t, 1)
    //
    // Initial condition:
    //
    // u(0, x) = cos(5 π x)
    //
    // The analytical solution is:
    //
    // u(t, x) = exp(-t) cos(5 π x)
    //
    // NOTE: This is a 1D problem solved on a 2D grid.

    // system
    let nx = 31;
    let (system, data, mut args) = Samples::heat_1d_periodic(nx);

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.newton.use_numerical_jacobian = false;
    params.set_tolerances(1e-3, 1e-3, None).unwrap();

    // solve the ODE system
    let t0 = data.x0;
    let t1 = data.x1;
    let mut uu = data.y0.clone();
    let mut solver = OdeSolver::new(params, &system).unwrap();
    solver.solve(&mut uu, t0, t1, None, None, &mut args).unwrap();

    // print stats
    println!("{}", solver.bench());

    // check the results
    let mut err_max = 0.0;
    args.fdm.loop_over_grid_points(|m, x, _| {
        err_max = f64::max(err_max, f64::abs(uu[m] - f64::exp(-t1) * f64::cos(5.0 * PI * x)));
    });
    println!("err_max = {}", err_max);
    approx_eq(err_max, 0.01314, 1e-5);

    // graph
    if SAVE_FIGURE {
        let x_ana = Vector::linspace(-1.0, 1.0, 201).unwrap();
        let u_ana = x_ana.get_mapped(|x| f64::exp(-t1) * f64::cos(5.0 * PI * x));
        let mut x_num = Vector::new(nx);
        let mut u_num_bot = Vector::new(nx);
        let mut u_num_top = Vector::new(nx);
        args.fdm.loop_over_grid_points(|m, x, _| {
            let i = m % nx;
            let j = m / nx;
            if j == 0 {
                x_num[i] = x;
                u_num_bot[i] = uu[m];
            } else {
                u_num_top[i] = uu[m];
            }
        });
        let mut curve1 = Curve::new();
        let mut curve2 = Curve::new();
        let mut curve3 = Curve::new();
        curve2
            .set_line_style("None")
            .set_marker_style("o")
            .set_marker_color("red");
        curve3
            .set_line_style("None")
            .set_marker_style("+")
            .set_marker_color("black");
        curve1.draw(x_ana.as_data(), u_ana.as_data());
        curve2.draw(x_num.as_data(), u_num_bot.as_data());
        curve3.draw(x_num.as_data(), u_num_top.as_data());
        let mut plot = Plot::new();
        plot.add(&curve1)
            .add(&curve2)
            .add(&curve3)
            .grid_and_labels("x", "u")
            .set_figure_size_points(600.0, 350.0)
            .save("/tmp/russell_ode/test_heat_1d_periodic.svg")
            .unwrap();
    }
    Ok(())
}
