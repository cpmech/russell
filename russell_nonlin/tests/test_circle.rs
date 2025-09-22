#![allow(unused)]

use plotpy::{linspace, Canvas, Curve, Plot, RayEndpoint};
use russell_lab::{approx_eq, array_approx_eq, math::SQRT_2};
use russell_nonlin::{AutoStep, Config, Direction, Status, Method, Output, Samples, Solver, Stop};

const RADIUS: f64 = SQRT_2;

const SAVE_FIGURE: bool = true;

fn do_plot(name: &str, uu: &[f64], ll: &[f64], stepsizes: &[f64]) {
    // Draw a reference circle
    let mut circle = Canvas::new();
    circle
        .set_edge_color("#00ba7f")
        .set_face_color("None")
        .draw_circle(0.0, 0.0, RADIUS);

    // Draw the state points
    let mut curve = Curve::new();
    curve.set_line_color("#d60943").set_marker_style(".").draw(&uu, &ll);

    // Plot everything
    let mut plot = Plot::new();
    plot.set_labels("$u$", "$\\lambda$")
        .add(&circle)
        .add(&curve)
        .set_equal_axes(true)
        .set_cross(0.0, 0.0, "#00ba7f", "-", 1.0)
        .set_range(-0.1, 0.1 + SQRT_2, -0.1, 0.1 + SQRT_2)
        .set_figure_size_points(800.0, 800.0)
        .save(&format!("/tmp/russell_nonlin/{}.svg", name))
        .unwrap()
}

fn do_plot_stepsizes(name: &str, stepsizes: &[f64]) {
    let hh = &stepsizes[1..]; // the first one is duplicated
    let n = hh.len();
    let x = linspace(1.0, n as f64, n);

    let mut curve = Curve::new();
    curve.set_label("stepsize").set_line_style("-").set_marker_style(".");
    curve.draw(&x.as_slice(), &hh);

    let mut plot = Plot::new();
    plot.set_labels("step number", "stepsize $h$")
        .add(&curve)
        .save(&format!("/tmp/russell_nonlin/{}_stepsizes.svg", name))
        .unwrap();
}

#[test]
fn test_circle_1() {
    // system
    let (system, mut state, mut args) = Samples::circle_ul(RADIUS);

    // configuration
    let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_h_ini(0.3)
        .set_record_iterations_residuals(true);

    // define solver
    let mut solver = Solver::new(config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let status = solver
        .solve(
            &mut args,
            &mut state,
            Direction::Pos,
            Stop::MaxLambda(RADIUS),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    // assert_eq!(status, Failure::Success);

    // check the results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    println!("u final = {:.3e}", state.u[0]);
    println!("λ final = {}, err = {:.3e}", state.l, f64::abs(state.l - RADIUS));
    approx_eq(state.u[0], 0.0, 1e-5);
    approx_eq(state.l, RADIUS, 1e-14);

    // check stats
    let stats = solver.get_stats();
    assert_eq!(stats.n_accepted, 3);
    assert_eq!(stats.n_rejected, 1);
    assert_eq!(stats.n_steps, 4);

    // plot
    if SAVE_FIGURE {
        let hh = out.get_h_values();
        do_plot("test_circle_1", uu, ll, hh);
        do_plot_stepsizes("test_circle_1", hh);
    }
}
