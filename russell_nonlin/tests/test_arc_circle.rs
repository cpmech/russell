use plotpy::{linspace, Canvas, Curve, Plot, RayEndpoint};
use russell_lab::{approx_eq, math::SQRT_2};
use russell_nonlin::{AutoStep, Config, IniDir, Method, Output, Samples, Solver, Status, Stop};

const RADIUS: f64 = SQRT_2;

const SAVE_FIGURE: bool = false;

#[test]
fn test_arc_circle_max_lambda() {
    // system
    let (system, mut state, mut args) = Samples::circle_ul(RADIUS);

    // configuration
    let mut config = Config::new(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_h_ini(0.3)
        .set_record_iterations_residuals(true)
        .set_log_file("/tmp/russell_nonlin/test_arc_circle_max_lambda.txt");

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
            IniDir::Pos,
            Stop::MaxLambda(RADIUS),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // check the results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    println!("u final = {}", state.u[0]);
    println!("λ final = {}, err = {}", state.l, f64::abs(state.l - RADIUS));
    approx_eq(state.u[0], 0.0, 1e-5);
    approx_eq(state.l, RADIUS, 1e-14);

    // check stats
    let stats = solver.get_stats();
    assert_eq!(stats.n_accepted, 5);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_steps, 5);

    // plot
    if SAVE_FIGURE {
        let hh = out.get_h_values();
        let duds = out.get_duds_values(0);
        let dlds = out.get_dlds_values();
        do_plot("test_arc_circle_max_lambda", uu, ll, duds, dlds, hh);
        do_plot_stepsizes("test_arc_circle_max_lambda", hh);
    }
}

#[test]
fn test_arc_circle_min_lambda() {
    // system
    let (system, mut state, mut args) = Samples::circle_ul(RADIUS);

    // configuration
    let mut config = Config::new(Method::Arclength);
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
            IniDir::Neg,
            Stop::MinLambda(0.0),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // check the results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    println!("u final = {}", state.u[0]);
    println!("λ final = {:.7e}", state.l);
    approx_eq(state.u[0], RADIUS, 1e-7);
    approx_eq(state.l, 0.0, 1e-14);

    // check stats
    let stats = solver.get_stats();
    assert_eq!(stats.n_accepted, 13);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_steps, 13);

    // plot
    if SAVE_FIGURE {
        let hh = out.get_h_values();
        let duds = out.get_duds_values(0);
        let dlds = out.get_dlds_values();
        do_plot("test_arc_circle_min_lambda", uu, ll, duds, dlds, hh);
        do_plot_stepsizes("test_arc_circle_min_lambda", hh);
    }
}

#[test]
fn test_arc_circle_max_u() {
    // system
    let (system, mut state, mut args) = Samples::circle_ul(RADIUS);

    // configuration
    let mut config = Config::new(Method::Arclength);
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
            IniDir::Neg,
            Stop::MaxCompU(0, 1.3),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // check the results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    println!("u final = {}", state.u[0]);
    println!("λ final = {}, err = {}", state.l, f64::abs(state.l - RADIUS));
    approx_eq(state.u[0], 1.3, 1e-5);
    approx_eq(state.u[0] * state.u[0] + state.l * state.l, RADIUS * RADIUS, 1e-10);

    // check stats
    let stats = solver.get_stats();
    assert_eq!(stats.n_accepted, 5);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_steps, 5);

    // plot
    if SAVE_FIGURE {
        let hh = out.get_h_values();
        let duds = out.get_duds_values(0);
        let dlds = out.get_dlds_values();
        do_plot("test_arc_circle_max_u", uu, ll, duds, dlds, hh);
        do_plot_stepsizes("test_arc_circle_max_u", hh);
    }
}

#[test]
fn test_arc_circle_min_u() {
    // system
    let (system, mut state, mut args) = Samples::circle_ul(RADIUS);

    // configuration
    let mut config = Config::new(Method::Arclength);
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
            IniDir::Pos,
            Stop::MinCompU(0, 0.4),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // check the results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    println!("u final = {}", state.u[0]);
    println!("λ final = {:.7e}", state.l);
    assert!(state.u[0] > 0.0 && state.u[0] < 0.4);
    approx_eq(state.u[0] * state.u[0] + state.l * state.l, RADIUS * RADIUS, 1e-8);

    // check stats
    let stats = solver.get_stats();
    assert_eq!(stats.n_accepted, 3);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_steps, 3);

    // plot
    if SAVE_FIGURE {
        let hh = out.get_h_values();
        let duds = out.get_duds_values(0);
        let dlds = out.get_dlds_values();
        do_plot("test_arc_circle_min_u", uu, ll, duds, dlds, hh);
        do_plot_stepsizes("test_arc_circle_min_u", hh);
    }
}

#[test]
fn test_arc_circle_max_lambda_num_jac() {
    // system
    let (system, mut state, mut args) = Samples::circle_ul(RADIUS);

    // configuration
    let mut config = Config::new(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_h_ini(0.3)
        .set_use_numerical_jacobian(true)
        .set_record_iterations_residuals(true)
        .set_log_file("/tmp/russell_nonlin/test_arc_circle_max_lambda_num_jac.txt");

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
            IniDir::Pos,
            Stop::MaxLambda(RADIUS),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // check the results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    println!("u final = {}", state.u[0]);
    println!("λ final = {}, err = {}", state.l, f64::abs(state.l - RADIUS));
    approx_eq(state.u[0], 0.0, 1e-5);
    approx_eq(state.l, RADIUS, 1e-14);

    // check stats
    let stats = solver.get_stats();
    assert_eq!(stats.n_accepted, 5);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_steps, 5);

    // plot
    if SAVE_FIGURE {
        let hh = out.get_h_values();
        let duds = out.get_duds_values(0);
        let dlds = out.get_dlds_values();
        do_plot("test_arc_circle_max_lambda_num_jac", uu, ll, duds, dlds, hh);
        do_plot_stepsizes("test_arc_circle_max_lambda_num_jac", hh);
    }
}

fn do_plot(name: &str, uu: &[f64], ll: &[f64], duds: &[f64], dlds: &[f64], stepsizes: &[f64]) {
    // Draw a reference circle
    let mut circle = Canvas::new();
    circle
        .set_edge_color("#00ba7f")
        .set_face_color("None")
        .draw_circle(0.0, 0.0, RADIUS);

    // Draw a tangent line passing through the starting point
    let mut first_tangent = Canvas::new();
    first_tangent
        .set_edge_color("#00ba7f")
        .set_face_color("None")
        .draw_polyline(&[[2.0, 0.0], [0.0, 2.0]], false);

    // Draw the state points
    let mut curve = Curve::new();
    curve.set_line_color("#d60943").set_marker_style(".").draw(&uu, &ll);

    // There are one less arrow than points. Also, the last increment is on the
    // λ axis, i.e., it is not σ, so we cannot calculate the last arrow length.
    let na = uu.len() - 2;

    // Draw arrows
    let mut arrows = Canvas::new();
    arrows
        .set_arrow_scale(15.0)
        .set_edge_color("None")
        .set_face_color("#515151ff");
    for i in 0..na {
        let sigma = stepsizes[1 + i]; // the first stepsize is duplicated because of the initial state
        let xf = uu[i] + sigma * duds[i];
        let yf = ll[i] + sigma * dlds[i];
        arrows.draw_arrow(uu[i], ll[i], xf, yf);
    }

    // Note that the hyperplanes are drawn for the predictor points,
    // thus, the hyperplanes do not pass through zero.
    let mut hyperplanes = Curve::new();
    hyperplanes.set_line_style("--").set_line_color("gray");
    for i in 0..na {
        let sigma = stepsizes[1 + i]; // the first stepsize is duplicated because of the initial state
        let xa = uu[i] + sigma * duds[i];
        let ya = ll[i] + sigma * dlds[i];
        let phi = f64::atan2(dlds[i], duds[i]);
        let xb = xa - f64::sin(phi);
        let yb = ya + f64::cos(phi);
        let ep = RayEndpoint::Coords(xb, yb);
        hyperplanes.draw_ray(xa, ya, ep);
    }

    // Plot everything
    let mut plot = Plot::new();
    plot.set_labels("$u$", "$\\lambda$")
        .add(&circle)
        .add(&first_tangent)
        .add(&hyperplanes)
        .add(&curve)
        .add(&arrows)
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
