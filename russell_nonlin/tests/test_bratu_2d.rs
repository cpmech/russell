#![allow(unused)]

use plotpy::{linspace, Curve, Plot, Text};
use russell_lab::{mat_approx_eq, num_jacobian, Norm, Vector};
use russell_nonlin::{AutoStep, Config, IniDir, Method, NoArgs, Output, Solver, State, Stop, System};
use russell_pde::FdmLaplacian2d;
use russell_sparse::{CooMatrix, Sym};

const CHECK_JACOBIAN: bool = false;
const SAVE_FIGURE: bool = true;

fn run_test(bordering: bool, alpha: f64, npt: usize, stop: Stop, auto: AutoStep) {
    let mut fdm = FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, npt, npt).unwrap();
    fdm.set_homogeneous_boundary_conditions();
    let prescribed = fdm.prescribed_flags();
    let nodes_unknown = fdm.get_nodes_unknown();
    let nodes_prescribed = fdm.get_nodes_prescribed();
    let ndim = nodes_unknown.len();
    assert_eq!(fdm.dim() - fdm.num_prescribed(), ndim);

    let (dx, dy) = fdm.get_dx_dy();
    assert_eq!(dx, dy);
    // let dxx = dx * dx;
    let dxx = 1.0;

    println!(
        "\nfdm.dim = {}, n_unknown = {}, n_prescribed = {}, ndim = {}, dx = {}, dy = {}\n",
        fdm.dim(),
        ndim,
        fdm.num_prescribed(),
        ndim,
        dx,
        dy
    );
    println!("nodes_unknown = {:?}", nodes_unknown);
    println!("nodes_prescribed = {:?}\n", nodes_prescribed);

    let mut node_to_local_index = vec![0; fdm.dim()];
    for (k, m) in nodes_unknown.iter().enumerate() {
        node_to_local_index[*m] = k;
    }

    // let mut node_to_prescribed_value = vec![0.0; fdm.dim()];
    // for m in nodes_prescribed {
    //     node_to_prescribed_value[*m] = fdm.get_prescribed_value(*m).unwrap();
    // }

    let calc_gg = |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
        for (i, m) in nodes_unknown.iter().enumerate() {
            gg[i] = 0.0;
            fdm.loop_over_coef_mat_row_core(*m, |n, amn| {
                let phi = if prescribed[n] {
                    // print!("prescribed");
                    fdm.get_prescribed_value(n).unwrap()
                } else {
                    // print!("----------");
                    u[node_to_local_index[n]]
                };
                gg[i] += dxx * amn * phi;
                // println!(": i = {}, m = {}, n = {}, amn = {}, phi = {}", i, m, n, amn, phi);
            });
            // println!();
            gg[i] += dxx * l * f64::exp(u[i]);
        }
        Ok(())
    };

    let calc_ggu = |ggu_or_aa: &mut CooMatrix, l: f64, u: &Vector, _args: &mut NoArgs| {
        ggu_or_aa.reset();
        for (i, m) in nodes_unknown.iter().enumerate() {
            fdm.loop_over_coef_mat_row_core(*m, |n, amn| {
                if !prescribed[n] {
                    let j = node_to_local_index[n];
                    ggu_or_aa.put(i, j, dxx * amn).unwrap();
                }
            });
            ggu_or_aa.put(i, i, dxx * l * f64::exp(u[i])).unwrap();
        }
        if CHECK_JACOBIAN && bordering && npt <= 21 {
            let ana = ggu_or_aa.as_dense();
            let num = num_jacobian(ndim, l, u, 1.0, &mut 0, calc_gg).unwrap();
            if npt <= 4 {
                println!("ana =\n{:.3}", ana);
                println!("num =\n{:.3}", num);
            }
            mat_approx_eq(&ana, &num, 1e-7);
        }
        Ok(())
    };

    let calc_ggl = |ggl: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
        for i in 0..ndim {
            ggl[i] = dxx * f64::exp(u[i]);
        }
        Ok(())
    };

    let mut system = System::new(ndim, calc_gg).unwrap();

    let nnz = ndim * 5 + ndim; // 5-point stencil + diagonal from nonlinearity
    system.set_calc_ggu(Some(nnz), Sym::No, calc_ggu).unwrap();
    system.set_calc_ggl(calc_ggl);

    let mut config = Config::new(Method::Arclength);
    config
        .set_n_cont_failure_max(5)
        .set_n_cont_rejection_max(5)
        .set_nr_control_enabled(true)
        .set_tg_control_enabled(true)
        .set_tg_control_pid_vcc(true)
        // .set_tg_control_atol_and_rtol(1e-2)
        // .set_tg_control_atol_and_rtol(1.1e-1)
        // .set_alpha_max(2.1)
        // .set_alpha_max(1.0)
        .set_record_iterations_residuals(true)
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_debug_predictor(true)
        .set_bordering(bordering);

    let mut solver = Solver::new(config, system).unwrap();

    let out = &mut Output::new();
    out.set_record_norm_u(true, Norm::Inf, 0, ndim);

    let mut state = State::new(ndim);

    let status = solver
        .solve(
            &mut 0,
            &mut state,
            IniDir::Pos,
            // Stop::Steps(67),
            // Stop::Steps(20),
            // AutoStep::No(4.859),
            // AutoStep::No(1.0),
            stop,
            auto,
            Some(out),
        )
        .unwrap();
    println!("Status: {:?}", status);

    let lam_vals = out.get_l_values();
    let nrm_vals = out.get_norm_u_values();
    let mut lam_crit = f64::NEG_INFINITY; // λCrit
    let mut nrm_crit = 0.0; // ‖ϕ‖∞ @ λCrit
    for i in 0..lam_vals.len() {
        if lam_vals[i] > lam_crit {
            lam_crit = lam_vals[i];
            nrm_crit = nrm_vals[i];
        }
    }

    // plot the results
    if SAVE_FIGURE {
        // define the title
        let title = format!("npt = {}  |  $\\lambda_{{crit}} = {:.8}$", npt, lam_crit);

        // annotations
        let mut annotations = Text::new();
        annotations
            .set_bbox(true)
            .set_bbox_facecolor("white")
            .set_bbox_edgecolor("None")
            .set_bbox_style("round,pad=0.3")
            .set_rotation(90.0)
            .draw(
                lam_crit,
                nrm_crit + 0.5,
                &format!("← ({:.8}, {:.8})", lam_crit, nrm_crit),
            );

        // draw ϕ versus λ
        let mut curve_norm_phi = Curve::new();
        curve_norm_phi.set_marker_style(".").draw(lam_vals, nrm_vals);

        // generate the plot
        let key = if auto.yes() { "auto" } else { "fixed" };
        let mut plot = Plot::new();
        plot.set_title(&title)
            .set_horiz_line(nrm_crit, "#689868ff", "-", 1.0)
            .add(&curve_norm_phi)
            .add(&annotations)
            .grid_and_labels("λ", "‖ϕ‖∞")
            .set_figure_size_points(400.0, 300.0)
            .save(&format!("/tmp/russell_nonlin/test_bratu_2d_npt{}_{}.svg", npt, key))
            .unwrap();

        // plot stepsizes
        let hh = &out.get_h_values()[1..]; // the first one is duplicated
        let n = hh.len();
        let x = linspace(1.0, n as f64, n);
        let mut curve = Curve::new();
        curve.set_label("stepsize").set_line_style("-").set_marker_style(".");
        curve.draw(&x.as_slice(), &hh);
        let mut plot_b = Plot::new();
        plot_b
            .set_labels("step number", "stepsize $h$")
            .add(&curve)
            .save(&format!("/tmp/russell_nonlin/test_bratu_2d_npt{}_h_{}.svg", npt, key))
            .unwrap();
    }
}

#[test]
fn test_bratu_2d_auto() {
    let bordering = false;
    let auto = AutoStep::Yes;
    for alpha in [0.0] {
        for npt in [4, 5, 6, 7] {
            let ndim = (npt - 2) * (npt - 2);
            let stop = Stop::MaxNormU(4.0, Norm::Inf, 0, ndim);
            run_test(bordering, alpha, npt, stop, auto);
        }
    }
}
