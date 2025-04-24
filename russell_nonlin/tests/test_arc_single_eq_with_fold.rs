use plotpy::{linspace, Canvas, Curve, Plot};
use russell_lab::{approx_eq, array_approx_eq, math::SQRT_2};
use russell_nonlin::{AutoStep, Config, Direction, Method, Output, Samples, Solver, Stop};

const SAVE_FIGURE: bool = false;
const NAME: &str = "test_arc_single_eq_with_fold";

#[test]
fn test_arc_single_eq_with_fold() {
    // nonlinear problem
    let (system, mut state, lambda_ana, mut args) = Samples::single_eq_with_fold_point();

    // configuration
    let mut config = Config::new(Method::Arclength);
    config.set_verbose(true, true, true);

    // solver
    let mut solver = Solver::new(config, system).unwrap();

    // output data
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let dds = 0.5; // Δs ≡ h
    solver
        .solve(
            &mut args,
            &mut state,
            Direction::Pos,
            Stop::Steps(5),
            AutoStep::No(dds),
            Some(out),
        )
        .unwrap();

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
        let mut curve_ana = Curve::new();
        curve_ana.set_label("analytical");
        let uu_ana = linspace(0.0, 8.0, 101);
        let ll_ana = uu_ana.iter().map(|&u| lambda_ana(u)).collect();
        curve_ana.draw(&uu_ana, &ll_ana);

        let mut curve_num = Curve::new();
        curve_num
            .set_label("numerical")
            .set_line_style("None")
            .set_marker_style("o");
        curve_num.draw(uu, ll);

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

        let mut plot = Plot::new();
        plot.set_title("Arclength continuation with fold")
            .grid_labels_legend("$u$", "$\\lambda$")
            .add(&curve_ana)
            .add(&curve_num)
            .add(&arrows)
            .save(&format!("/tmp/russell_nonlin/{}.svg", NAME))
            .unwrap()
    }
}
