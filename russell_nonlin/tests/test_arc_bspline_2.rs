#![allow(unused)]

use plotpy::{linspace, Curve, Plot};
use russell_lab::approx_eq;
use russell_nonlin::{AutoStep, Config, Direction, Method, Output, Samples, Solver, Status, Stop};

const SAVE_FIGURE: bool = true;
const NAME: &str = "test_arc_bspline_2";

#[test]
fn test_arc_bspline_2() {
    // nonlinear problem
    let (system, mut state, mut args) = Samples::bspline_problem_1(1.5);

    // configuration
    let mut config = Config::new(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_allowed_continued_divergence(1)
        .set_alpha_max(15.0)
        .set_sigma_max(0.3)
        .set_h_ini(0.04);
    // .set_h_ini(0.4743);
    // .set_allowed_continued_divergence(3);

    // define solver
    let mut solver = Solver::new(config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0, 1], &[0, 1]);

    // numerical continuation
    let status = solver
        .solve(
            &mut args,
            &mut state,
            Direction::Pos,
            Stop::Lambda(1.0),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();

    // check
    // approx_eq(state.l, 1.0, 1e-15);

    // results
    let uu0 = out.get_u_values(0);
    let uu1 = out.get_u_values(1);
    // let ll = out.get_l_values();
    let du0ds = out.get_duds_values(0);
    let du1ds = out.get_duds_values(1);
    // let dlds = out.get_dlds_values();

    // plot
    if SAVE_FIGURE {
        // draw B-spline curve
        let mut curve = Curve::new();
        let n_station = 201;
        let stations = linspace(0.0, 1.0, n_station);
        let mut xx = vec![0.0; stations.len()];
        let mut yy = vec![0.0; stations.len()];
        for i in 0..n_station {
            args.bspline.calc_point(&mut args.coords, stations[i], false).unwrap();
            xx[i] = args.coords[0];
            yy[i] = args.coords[1];
        }
        curve.draw(&xx, &yy);

        let mut curve_num = Curve::new();
        curve_num
            .set_label("numerical")
            .set_line_style("-")
            .set_line_color("green")
            .set_marker_style("o")
            .set_marker_color("red")
            .set_marker_line_color("red");
        curve_num.draw(uu0, uu1);

        let mut plot = Plot::new();
        plot.set_labels("$u_1$", "$u_2$")
            .add(&curve)
            .add(&curve_num)
            .set_range(-0.1, 2.7, -0.1, 1.2)
            .set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save(&format!("/tmp/russell_nonlin/{}.svg", NAME))
            .unwrap()
    }
}
