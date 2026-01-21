use plotpy::{linspace, Canvas, Curve, Plot, RayEndpoint};
use russell_lab::math::{NAPIER, SQRT_2};
use russell_lab::{approx_eq, array_approx_eq};
use russell_nonlin::{AutoStep, Config, IniDir, Method, Output, Samples, Solver, Status, Stop};

const SAVE_FIGURE: bool = false;
const NAME: &str = "test_arc_one_eq_with_fold";

#[test]
fn test_arc_one_eq_with_fold_1() {
    // nonlinear problem
    let (system, mut u, mut l, lambda_ana, mut args) = Samples::one_eq_with_fold_point();

    // configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    config.set_verbose(true, true, true).set_hide_timings(true);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output data
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let ddl = 0.3535533; // Δλ
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(5),
            AutoStep::No(ddl),
            Some(out),
        )
        .unwrap();
    assert_eq!(status, Status::Success);

    // compare with Mathematica results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    let duds = out.get_duds_values(0);
    let dlds = out.get_dlds_values();
    let uu_mathematica = &[
        0.0,
        0.428095787401572,
        0.928938368657503,
        1.42982786403821,
        1.92613250768946,
        2.42179932045308,
    ];
    let ll_mathematica = &[
        0.0,
        0.279010993784976,
        0.366905391632565,
        0.342229470016786,
        0.280658009972863,
        0.214963176909604,
    ];
    let duds_mathematica = &[
        1.0 / SQRT_2,
        0.937024175037193,
        0.999606342770028,
        0.994749561659407,
        0.991017108989708,
        0.992130479949147,
    ];
    let dlds_mathematica = &[
        1.0 / SQRT_2,
        0.349264506349943,
        0.0280563628064792,
        -0.102339188869263,
        -0.133735147547983,
        -0.125208269518729,
    ];
    array_approx_eq(&uu, uu_mathematica, 1e-6);
    array_approx_eq(&ll, ll_mathematica, 1e-7);
    array_approx_eq(&duds, duds_mathematica, 1e-5);
    array_approx_eq(&dlds, dlds_mathematica, 1e-4);

    // compare with analytical solution
    for i in 0..uu.len() {
        approx_eq(ll[i], lambda_ana(uu[i]), 1e-10);
    }

    // plot
    if SAVE_FIGURE {
        do_plot(1, lambda_ana, ddl, uu, ll, duds, dlds);
    }
}

#[test]
fn test_arc_one_eq_with_fold_2() {
    // nonlinear problem
    let (system, mut u, mut l, lambda_ana, mut args) = Samples::one_eq_with_fold_point();

    // configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    config.set_verbose(true, true, true).set_hide_timings(true);

    // solver
    let mut solver = Solver::new(&config, system).unwrap();

    // output data
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // fold point:
    // uf = 1  and  λf = 1/e  thus  G = u - λ exp(u) = 1 - (1/e) e = 0
    // Δs = (uf - u0) duds0 + (λf - λ0) dλds0
    // duds0 = 1/√2  and  dλds0 = 1/√2
    // thus  Δs₀ = (1 - 0)/√2 + (1/e - 0)/√2 = (1 + 1/e)/√2
    // and Δλ₀ = Δs₀ dλds0 = Δs₀ / √2 = (1 + 1/e)/2

    // numerical continuation with the first step reaching the fold point
    let ddl = (1.0 + 1.0 / NAPIER) / 2.0; // Δλ₀
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(2),
            AutoStep::No(ddl),
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
    for i in 0..uu.len() {
        approx_eq(ll[i], lambda_ana(uu[i]), 1e-9);
    }

    // check u, λ and Gu at the fold point
    let uf = 1.0;
    let lf = 1.0 / NAPIER;
    let ggu_fold = 1.0 - ll[1] * f64::exp(uu[1]);
    approx_eq(uu[1], uf, 1e-15);
    approx_eq(ll[1], lf, 1e-15);
    approx_eq(duds[1], 1.0, 1e-15);
    approx_eq(dlds[1], 0.0, 1e-9);
    approx_eq(ggu_fold, 0.0, 1e-15);

    // plot
    if SAVE_FIGURE {
        do_plot(2, lambda_ana, ddl, uu, ll, duds, dlds);
    }
}

fn do_plot(
    index: usize,
    lambda_ana: impl Fn(f64) -> f64,
    dds: f64,
    uu: &[f64],
    ll: &[f64],
    duds: &[f64],
    dlds: &[f64],
) {
    let mut curve_ana = Curve::new();
    curve_ana.set_label("analytical");
    let uu_ana = linspace(0.0, 3.0, 101);
    let ll_ana = uu_ana.iter().map(|&u| lambda_ana(u)).collect();
    curve_ana.draw(&uu_ana, &ll_ana);

    let mut curve_num = Curve::new();
    curve_num
        .set_label("numerical")
        .set_line_style("None")
        .set_marker_style("o");
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
        .set_range(-0.1, 3.1, -0.1, 1.0)
        .set_equal_axes(true)
        .set_figure_size_points(800.0, 800.0)
        .save(&format!("/tmp/russell_nonlin/{}_{}.svg", NAME, index))
        .unwrap()
}
