use plotpy::{linspace, Canvas, Curve, Plot, RayEndpoint};
use russell_lab::{approx_eq, array_approx_eq, math::SQRT_2};
use russell_nonlin::{AutoStep, Config, IniDir, Method, Output, Samples, Solver, Status, Stop};
use russell_sparse::{Genie, Sym};
use serial_test::serial;

const SAVE_FIGURE: bool = false;

#[test]
fn test_arc_linear_problem() {
    run_test(Genie::Umfpack, false, false);
    run_test(Genie::Umfpack, true, true);
}

#[cfg(feature = "with_mumps")]
#[test]
#[serial]
fn test_arc_linear_problem_mumps() {
    run_test(Genie::Mumps, false, false);
    run_test(Genie::Mumps, true, true);
}

fn run_test(genie: Genie, symmetric: bool, bordering: bool) {
    // system
    let sym = genie.get_sym(symmetric);
    let with_ggu = true; // with ∂G/∂u
    let with_ggl = true; // with ∂G/∂λ
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(with_ggu, with_ggl, sym);

    // configuration
    let mut config = Config::new();
    config
        .set_method(Method::Arclength)
        .set_bordering(bordering)
        .set_genie(genie)
        .set_verbose(true, true, true)
        .set_hide_timings(true);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let nstep = 5;
    let ddl = 0.5 / SQRT_2; // Δλ
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(nstep),
            AutoStep::No(ddl),
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    let hh = out.get_h_values();

    // check
    let dds = ddl * SQRT_2;
    assert_eq!(hh, &vec![dds; nstep + 1]);
    array_approx_eq(&ll, &[0.0, ddl, 2.0 * ddl, 3.0 * ddl, 4.0 * ddl, 5.0 * ddl], 1e-15);
    array_approx_eq(&uu, &ll, 1e-15);

    // check stats
    let niter = nstep; // 1 iteration per step because the Euler predictor gives the exact answer
    let stats = solver.get_stats();
    assert_eq!(stats.n_function, niter);
    assert_eq!(stats.n_jacobian, niter + 1); // +1 because of the initial tangent vector
    assert_eq!(stats.n_factor, niter + 1);
    assert_eq!(stats.n_lin_sol, niter + 1);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iteration_total, niter);
    assert_eq!(stats.h_accepted, dds);
    assert!(stats.nanos_step_max > 0);
    assert!(stats.nanos_jacobian_max > 0);
    assert!(stats.nanos_factor_max > 0);
    assert!(stats.nanos_lin_sol_max > 0);
    assert!(stats.nanos_total > 0);

    // plot
    if SAVE_FIGURE {
        let duds = out.get_duds_values(0);
        let dlds = out.get_dlds_values();
        let hh = out.get_h_values();
        do_plot("test_arc_linear_problem", uu, ll, duds, dlds, hh);
    }
}

#[test]
fn test_arc_linear_problem_backward() {
    // system
    let with_ggu = true; // with ∂G/∂u
    let with_ggl = true; // with ∂G/∂λ
    let (system, mut u, _, mut args) = Samples::simple_linear_problem(with_ggu, with_ggl, Sym::No);

    // initial state
    let ddl = 0.5 / SQRT_2; // Δλ
    u[0] = 5.0 * ddl;
    let mut l = u[0];

    // configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    config.set_verbose(true, true, true).set_hide_timings(true);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let nstep = 5;
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Neg,
            Stop::Steps(nstep),
            AutoStep::No(ddl),
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    let hh = out.get_h_values();

    // check
    let dds = ddl * SQRT_2;
    assert_eq!(hh, &vec![dds; nstep + 1]);
    array_approx_eq(&ll, &[5.0 * ddl, 4.0 * ddl, 3.0 * ddl, 2.0 * ddl, ddl, 0.0], 1e-15);
    array_approx_eq(&uu, &ll, 1e-15);

    // check stats
    let niter = nstep; // 1 iteration per step because the Euler predictor gives the exact answer
    let stats = solver.get_stats();
    assert_eq!(stats.n_function, niter);
    assert_eq!(stats.n_jacobian, niter + 1); // +1 because of the initial tangent vector
    assert_eq!(stats.n_factor, niter + 1);
    assert_eq!(stats.n_lin_sol, niter + 1);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iteration_total, niter);
    assert_eq!(stats.h_accepted, dds);
    assert!(stats.nanos_step_max > 0);
    assert!(stats.nanos_jacobian_max > 0);
    assert!(stats.nanos_factor_max > 0);
    assert!(stats.nanos_lin_sol_max > 0);
    assert!(stats.nanos_total > 0);

    // plot
    if SAVE_FIGURE {
        let duds = out.get_duds_values(0);
        let dlds = out.get_dlds_values();
        do_plot("test_arc_linear_problem_backward", uu, ll, duds, dlds, hh);
    }
}

#[test]
fn test_arc_linear_problem_large_step() {
    // system
    let with_ggu = true; // with ∂G/∂u
    let with_ggl = true; // with ∂G/∂λ
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(with_ggu, with_ggl, Sym::No);

    // configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    config.set_verbose(true, true, true).set_hide_timings(true);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let ddl = 1.2 / SQRT_2; // Δλ
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.0),
            AutoStep::No(ddl),
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // check the results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    approx_eq(*uu.last().unwrap(), 1.0, 1e-14);
    approx_eq(*ll.last().unwrap(), 1.0, 1e-14);

    // check stats
    let stats = solver.get_stats();
    assert_eq!(stats.n_steps, 2);
    assert_eq!(stats.n_accepted, 2);
    assert_eq!(stats.n_rejected, 0);

    // plot
    if SAVE_FIGURE {
        let duds = out.get_duds_values(0);
        let dlds = out.get_dlds_values();
        let hh = out.get_h_values();
        do_plot("test_arc_linear_problem_large_step", uu, ll, duds, dlds, hh);
    }
}

#[test]
fn test_arc_linear_problem_auto() {
    // system
    let with_ggu = true; // with ∂G/∂u
    let with_ggl = true; // with ∂G/∂λ
    let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(with_ggu, with_ggl, Sym::No);

    // configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_ddl_ini(0.07);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.0),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // check the results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    approx_eq(*uu.last().unwrap(), 1.0, 1e-14);
    approx_eq(*ll.last().unwrap(), 1.0, 1e-14);

    // plot
    if SAVE_FIGURE {
        let duds = out.get_duds_values(0);
        let dlds = out.get_dlds_values();
        let hh = out.get_h_values();
        let name = "test_arc_linear_problem_auto";
        do_plot(name, uu, ll, duds, dlds, hh);
        do_plot_stepsizes(name, hh);
    }
}

#[test]
fn test_arc_linear_problem_auto_backward() {
    // system
    let with_ggu = true; // with ∂G/∂u
    let with_ggl = true; // with ∂G/∂λ
    let (system, mut u, _, mut args) = Samples::simple_linear_problem(with_ggu, with_ggl, Sym::No);

    // initial state
    u[0] = 5.0 * 0.5 / SQRT_2;
    let mut l = 5.0 * 0.5 / SQRT_2;

    // configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_ddl_ini(0.07);

    // define solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
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
    approx_eq(*uu.last().unwrap(), 0.0, 1e-9);
    approx_eq(*ll.last().unwrap(), 0.0, 1e-9);

    // plot
    if SAVE_FIGURE {
        let uu = out.get_u_values(0);
        let ll = out.get_l_values();
        let duds = out.get_duds_values(0);
        let dlds = out.get_dlds_values();
        let hh = out.get_h_values();
        let name = "test_arc_linear_problem_auto_backward";
        do_plot(name, uu, ll, duds, dlds, hh);
        do_plot_stepsizes(name, hh);
    }
}

fn do_plot(name: &str, uu: &[f64], ll: &[f64], duds: &[f64], dlds: &[f64], stepsizes: &[f64]) {
    let mut curve_ana = Curve::new();
    curve_ana.set_label("analytical");
    let uu_ana = linspace(0.0, 3.0, 101);
    let ll_ana = uu_ana.iter().map(|&u| u).collect();
    curve_ana.draw(&uu_ana, &ll_ana);

    let mut curve_num = Curve::new();
    curve_num
        .set_label("numerical")
        .set_line_style("None")
        .set_marker_style(".");
    curve_num.draw(&uu, &ll);

    let mut arrows = Canvas::new();
    arrows
        .set_arrow_scale(10.0)
        .set_edge_color("None")
        .set_face_color("black");
    let na = uu.len() - 1; // there are one less arrow than points.
    for i in 0..na {
        let sigma = stepsizes[1 + i]; // the first stepsize is duplicated because of the initial state
        let xf = uu[i] + sigma * duds[i];
        let yf = ll[i] + sigma * dlds[i];
        arrows.draw_arrow(uu[i], ll[i], xf, yf);
    }

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

    let mut plot = Plot::new();
    plot.set_labels("$u$", "$\\lambda$")
        .add(&hyperplanes)
        .add(&curve_ana)
        .add(&curve_num)
        .add(&arrows)
        .set_range(-0.1, 2.0, -0.1, 2.0)
        .set_equal_axes(true)
        .save(&format!("/tmp/russell_nonlin/{}.svg", name))
        .unwrap()
}

fn do_plot_stepsizes(name: &str, stepsizes: &[f64]) {
    let n = stepsizes.len();
    let x = linspace(1.0, n as f64, n);

    let mut curve = Curve::new();
    curve.set_label("stepsize").set_line_style("-").set_marker_style("o");
    curve.draw(&x.as_slice(), &stepsizes);

    let mut plot = Plot::new();
    plot.grid_labels_legend("step number", "stepsize $h$")
        .set_ticks_x(1.0, -1.0, "%d")
        .add(&curve)
        .save(&format!("/tmp/russell_nonlin/{}_stepsizes.svg", name))
        .unwrap();
}
