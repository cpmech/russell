use plotpy::{linspace, Canvas, Curve, Plot, RayEndpoint};
use russell_lab::{array_approx_eq, math::SQRT_2, Vector};
use russell_nonlin::{AutoStep, Config, Direction, Method, NoArgs, Output, Solver, State, Stop, System};
use russell_sparse::{CooMatrix, Sym};

const SAVE_FIGURE: bool = false;
const NAME: &str = "test_arc_linear_problem";

#[test]
fn test_arc_linear_problem() {
    // define nonlinear system: G(u, λ) = u - λ
    let ndim = 1;
    let mut system = System::new(ndim, |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
        gg[0] = u[0] - l;
        Ok(())
    })
    .unwrap();

    // function to compute Gu = ∂G/∂u
    let nnz = Some(1);
    let sym = Sym::No;
    system
        .set_calc_ggu(
            nnz,
            sym,
            |ggu: &mut CooMatrix, _l: f64, _u: &Vector, _args: &mut NoArgs| {
                ggu.put(0, 0, 1.0).unwrap();
                Ok(())
            },
        )
        .unwrap();

    // function to compute Gl = ∂G/∂λ
    system
        .set_calc_ggl(|ggl: &mut Vector, _l: f64, _u: &Vector, _args: &mut NoArgs| {
            ggl[0] = -1.0;
            Ok(())
        })
        .unwrap();

    // configuration
    let mut config = Config::new(Method::Arclength);
    config.set_verbose(true, true, true).set_hide_timings(true);

    // define solver
    let mut solver = Solver::new(config, system).unwrap();

    // initial state
    let mut state = State::new(ndim, true);
    state.u[0] = 0.0;
    state.l = 0.0;

    // output
    let out = &mut Output::new();
    out.set_recording(true, &[0], &[0]);

    // numerical continuation
    let nstep = 5;
    let dds = 0.5; // Δs ≡ h
    let args = &mut 0;
    solver
        .solve(
            args,
            &mut state,
            Direction::Pos,
            Stop::Steps(nstep),
            AutoStep::No(dds),
            Some(out),
        )
        .unwrap();

    // results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();
    let duds = out.get_duds_values(0);
    let dlds = out.get_dlds_values();

    // check
    let d = dds / SQRT_2;
    assert_eq!(out.get_h_values(), &vec![dds; nstep + 1]);
    array_approx_eq(&ll, &[0.0, d, 2.0 * d, 3.0 * d, 4.0 * d, 5.0 * d], 1e-15);
    array_approx_eq(&uu, &[0.0, d, 2.0 * d, 3.0 * d, 4.0 * d, 5.0 * d], 1e-15);

    // check stats
    let niter = nstep; // 1 iteration per step because the Euler predictor gives the exact answer
    let stats = solver.stats();
    assert_eq!(stats.n_function, niter);
    assert_eq!(stats.n_jacobian, niter + 1); // +1 because of the initial tangent vector
    assert_eq!(stats.n_factor, niter + 1);
    assert_eq!(stats.n_lin_sol, niter + 1);
    assert_eq!(stats.n_steps, nstep);
    assert_eq!(stats.n_accepted, nstep);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_iterations_max, 1); // Euler predictor already gives the exact answer
    assert_eq!(stats.n_iterations_total, niter);
    assert_eq!(stats.h_accepted, dds);
    assert!(stats.nanos_step_max > 0);
    assert!(stats.nanos_jacobian_max > 0);
    assert!(stats.nanos_factor_max > 0);
    assert!(stats.nanos_lin_sol_max > 0);
    assert!(stats.nanos_total > 0);

    // plot
    if SAVE_FIGURE {
        let mut curve_ana = Curve::new();
        curve_ana.set_label("analytical");
        let uu_ana = linspace(0.0, 3.0, 101);
        let ll_ana = uu_ana.iter().map(|&u| u).collect();
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
            .save(&format!("/tmp/russell_nonlin/{}.svg", NAME))
            .unwrap()
    }
}
