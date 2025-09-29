#![allow(unused)]

use plotpy::{linspace, Curve, Plot};
use russell_lab::approx_eq;
use russell_nonlin::{AutoStep, Config, Direction, Method, Output, Samples, Solver, Status, Stop};

const SAVE_FIGURE: bool = true;

fn run_test(
    name: &str,
    bordering: bool,
    atol_and_rtol: Option<f64>,
    alpha_max: Option<f64>,
    expected_n_accepted: usize,
    expected_n_rejected: usize,
    expected_n_steps: usize,
    expected_status: Status,
) {
    // nonlinear problem
    let (system, mut state, mut args) = Samples::bspline_problem_1(1.5);

    // configuration
    let mut config = Config::new(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_bordering(bordering)
        .set_log_file(&format!("/tmp/russell_nonlin/{}.txt", name))
        .set_record_iterations_residuals(true)
        .set_allowed_continued_divergence(1)
        .set_h_ini(0.04);
    if let Some(tol) = atol_and_rtol {
        config.set_tg_control_atol_and_rtol(tol);
    }
    if let Some(alpha) = alpha_max {
        config.set_alpha_max(alpha);
    }

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
            Stop::MaxLambda(1.0),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    assert_eq!(status, expected_status);

    // check statistics
    let stats = solver.get_stats();
    assert_eq!(stats.n_accepted, expected_n_accepted);
    assert_eq!(stats.n_rejected, expected_n_rejected);
    assert_eq!(stats.n_steps, expected_n_steps);

    // plot
    if SAVE_FIGURE {
        // results
        let uu0 = out.get_u_values(0);
        let uu1 = out.get_u_values(1);
        let du0ds = out.get_duds_values(0);
        let du1ds = out.get_duds_values(1);

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
            .set_marker_style(".")
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
            .save(&format!("/tmp/russell_nonlin/{}.svg", name))
            .unwrap()
    }
}

#[test]
fn test_arc_bspline_2_default() {
    run_test(
        "test_arc_bspline_2_default",
        false,
        None,
        None,
        87,
        0,
        87,
        Status::Success,
    );
}

#[test]
fn test_arc_bspline_2_custom() {
    run_test(
        "test_arc_bspline_2_custom",
        false,
        Some(1e-1),
        Some(3.0),
        82,
        36,
        118,
        Status::Success,
    );
}

#[test]
fn test_arc_bspline_2_bordering() {
    run_test(
        "test_arc_bspline_2_bordering",
        true,
        None,
        None,
        87,
        0,
        87,
        Status::Success,
    );
}
