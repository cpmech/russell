use plotpy::{linspace, Canvas, Curve, Plot, RayEndpoint};
use russell_nonlin::{AutoStep, Config, Direction, Method, Output, Samples, Solver, Status, Stop};

const SAVE_FIGURE: bool = true;

fn run_test(name: &str, sigma: f64, expected_status: Status) {
    // nonlinear problem
    let (system, mut state, mut args) = Samples::bspline_problem_1(0.0);

    // configuration
    let mut config = Config::new(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_alpha_max_ultimate(60.0)
        .set_hide_timings(true)
        .set_debug_predictor(true);

    // define solver
    let mut solver = Solver::new(config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0, 1], &[0, 1]);

    // numerical continuation
    let nstep = 6;
    let status = solver
        .solve(
            &mut args,
            &mut state,
            Direction::Pos,
            Stop::Steps(nstep),
            AutoStep::No(sigma),
            Some(out),
        )
        .unwrap();
    assert_eq!(status, expected_status);
    if expected_status == Status::Failure {
        assert_eq!(solver.get_errors(), &["max number of iterations reached"]);
    }

    // plot
    if SAVE_FIGURE {
        // results
        let uu0 = out.get_u_values(0);
        let uu1 = out.get_u_values(1);
        let du0ds = out.get_duds_values(0);
        let du1ds = out.get_duds_values(1);

        // debug data
        let (_, u0_predictor, u1_predictor) = solver.get_debug_predictor_values();

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

        // draw numerical solution
        let mut curve_num = Curve::new();
        curve_num
            .set_label("numerical")
            .set_line_style("-")
            .set_line_color("green")
            .set_marker_style("o")
            .set_marker_color("red")
            .set_marker_line_color("red");
        curve_num.draw(uu0, uu1);

        // draw predictor points
        let mut predictor_curve = Curve::new();
        predictor_curve
            .set_line_style("None")
            .set_marker_style("*")
            .set_marker_color("magenta")
            .set_marker_line_color("magenta");
        predictor_curve.draw(&u0_predictor, &u1_predictor);

        // draw tangent vectors
        let mut arrows = Canvas::new();
        arrows
            .set_arrow_scale(10.0)
            .set_edge_color("None")
            .set_face_color("black");
        for i in 0..uu0.len() {
            let xf = uu0[i] + sigma * du0ds[i];
            let yf = uu1[i] + sigma * du1ds[i];
            arrows.draw_arrow(uu0[i], uu1[i], xf, yf);
        }

        // draw hyperplanes
        let mut hyperplanes = Curve::new();
        hyperplanes.set_line_style("--").set_line_color("#d0d0d0");
        for i in 0..uu0.len() {
            let xa = uu0[i] + sigma * du0ds[i];
            let ya = uu1[i] + sigma * du1ds[i];
            let phi = f64::atan2(du1ds[i], du0ds[i]);
            let xb = xa - f64::sin(phi);
            let yb = ya + f64::cos(phi);
            let ep = RayEndpoint::Coords(xb, yb);
            hyperplanes.draw_ray(xa, ya, ep);
        }

        // plot
        let mut plot = Plot::new();
        plot.set_labels("$u_1$", "$u_2$")
            .add(&hyperplanes)
            .add(&predictor_curve)
            .add(&curve)
            .add(&curve_num)
            .add(&arrows)
            .set_range(-0.1, 2.7, -0.1, 1.2)
            .set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save(&format!("/tmp/russell_nonlin/{}.svg", name))
            .unwrap()
    }
}

#[test]
fn test_arc_bspline_1_success() {
    let sigma = 0.4742; // σ ≡ h
    run_test("test_arc_bspline_1_success", sigma, Status::Success);
}

#[test]
fn test_arc_bspline_1_fail() {
    let sigma = 0.4743; // σ ≡ h (this value causes problems; Newton's method diverges)
    run_test("test_arc_bspline_1_fail", sigma, Status::Failure);
}
