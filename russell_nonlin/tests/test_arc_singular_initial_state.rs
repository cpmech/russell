use plotpy::{linspace, Canvas, Curve, Plot, RayEndpoint};
use russell_lab::approx_eq;
use russell_nonlin::{AutoStep, Config, IniDir, Method, Output, Samples, Solver, Status, Stop};

const SAVE_FIGURE: bool = true;
const NAME: &str = "test_arc_singular_initial_state";

fn do_plot(
    index: usize,
    perturbation: f64,
    lambda_ana: impl Fn(f64) -> f64,
    dds: f64,
    uu: &[f64],
    ll: &[f64],
    duds: &[f64],
    dlds: &[f64],
) {
    let mut curve_ana = Curve::new();
    curve_ana.set_label("analytical");
    let uu_ana = linspace(0.0, 3.0, 201);
    let ll_ana = uu_ana.iter().map(|&u| lambda_ana(u)).collect();
    curve_ana.draw(&uu_ana, &ll_ana);

    let mut curve_num = Curve::new();
    curve_num
        .set_label("numerical")
        .set_line_style("None")
        .set_marker_style("o")
        .set_marker_void(true);
    curve_num.draw(&uu, &ll);

    let mut arrows = Canvas::new();
    arrows
        .set_arrow_scale(10.0)
        .set_edge_color("None")
        .set_face_color("black");
    for i in 0..uu.len() {
        let xf = uu[i] + dds * duds[i];
        let yf = ll[i] + dds * dlds[i];
        arrows.draw_arrow(uu[i], ll[i], xf, yf);
    }

    let mut hyperplanes = Curve::new();
    hyperplanes.set_line_style("--").set_line_color("gray");
    for i in 0..uu.len() {
        let xa = uu[i] + dds * duds[i];
        let ya = ll[i] + dds * dlds[i];
        let phi = f64::atan2(dlds[i], duds[i]);
        let xb = xa - f64::sin(phi); // with radius = 1
        let yb = ya + f64::cos(phi);
        let ep = RayEndpoint::Coords(xb, yb);
        hyperplanes.draw_ray(xa, ya, ep);
    }

    let mut plot = Plot::new();
    plot.set_title(&format!("perturbation = {:e}", perturbation))
        .set_labels("$u$", "$\\lambda$")
        .add(&hyperplanes)
        .add(&curve_ana)
        .add(&curve_num)
        .add(&arrows)
        .set_range(-0.1, 3.1, -0.1, 1.5)
        .set_equal_axes(true)
        .set_figure_size_points(600.0, 600.0)
        .save(&format!("/tmp/russell_nonlin/{}_{}.svg", NAME, index))
        .unwrap()
}

#[test]
fn test_arc_singular_initial_state_1() {
    // system
    let alpha = 1.0 / 3.0;
    let perturbation = 1e-6;
    let (system, mut state, lambda_ana, mut args) = Samples::singular_initial_state(alpha, perturbation);

    // configuration
    let mut config = Config::new(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_allowed_continued_divergence(2);

    // define solver
    let mut solver = Solver::new(config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let nstep = 5;
    let dds = 0.5; // Δs ≡ h
    let status = solver
        .solve(
            &mut args,
            &mut state,
            IniDir::Pos,
            Stop::Steps(nstep),
            AutoStep::No(dds),
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    let duds = out.get_duds_values(0);
    let dlds = out.get_dlds_values();

    // compare with analytical solution
    for i in 1..uu.len() {
        approx_eq(ll[i], lambda_ana(uu[i]), 1e-10);
    }

    // plot
    if SAVE_FIGURE {
        do_plot(1, perturbation, lambda_ana, dds, uu, ll, duds, dlds);
    }
}

#[test]
fn test_arc_singular_initial_state_2() {
    // system
    let alpha = 1.0 / 3.0;
    let perturbation = 0.05;
    let (system, mut state, lambda_ana, mut args) = Samples::singular_initial_state(alpha, perturbation);

    // configuration
    let mut config = Config::new(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_allowed_continued_divergence(2);

    // define solver
    let mut solver = Solver::new(config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let nstep = 5;
    let dds = 0.5; // Δs ≡ h
    let status = solver
        .solve(
            &mut args,
            &mut state,
            IniDir::Pos,
            Stop::Steps(nstep),
            AutoStep::No(dds),
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    let duds = out.get_duds_values(0);
    let dlds = out.get_dlds_values();

    // compare with analytical solution
    for i in 1..uu.len() {
        approx_eq(ll[i], lambda_ana(uu[i]), 1e-10);
    }

    // plot
    if SAVE_FIGURE {
        do_plot(2, perturbation, lambda_ana, dds, uu, ll, duds, dlds);
    }
}
