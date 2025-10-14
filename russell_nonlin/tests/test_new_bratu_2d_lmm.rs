#![allow(unused)]

use plotpy::{linspace, Curve, Plot, Text};
use russell_lab::{
    approx_eq, mat_approx_eq, mat_eigenvalues, mat_inverse, num_jacobian, read_table, vec_norm, vec_norm_chunk,
};
use russell_lab::{Matrix, Norm, Vector};
use russell_nonlin::{AutoStep, Config, IniDir, Method, NoArgs, Output, Solver, State, Status, Stop, System};
use russell_pde::{EssentialBcs2d, FdmLaplacian2dNew, Grid2d};
use russell_sparse::{CooMatrix, Sym};
use std::collections::HashMap;

// The nonlinear problem originates from the FDM discretization of the following equation:
//
// (Bratu's problem in 2D with Lagrange multipliers method - LMM)
//
// ∂²ϕ   ∂²ϕ
// ——— + ——— + λ exp(ϕ) = 0
// ∂x²   ∂y²
//
// on the unit square (1.0 × 1.0) with homogeneous boundary conditions.
//
// Below, ϕ is a vector, i.e., ϕ = [ϕ₀, ϕ₁, ϕ₂, ..., ϕₙ₋₁]ᵀ, where n is the number of grid points.
// The prescribed values are collected in the vector c = [c₀, c₁, c₂, ..., cₘ₋₁]ᵀ, where m is the
// number of boundary points. The Laplacian operator is represented by the matrix K, thus
// Kϕ is the discretization of the Laplacian operator applied to ϕ(x,y).
//
// The boundary conditions are enforced via Lagrange multipliers ψ = [ψ₀, ψ₁, ψ₂, ..., ψₘ₋₁]ᵀ.
// The prescribed values are all zero (homogeneous boundary conditions), thus c = [0, 0, ... 0]ᵀ.
// The constraints matrix for the Lagrange multipliers method is E, thus Eϕ = c.
//
// The vector of unknowns is expressed by u = [ϕ, ψ]ᵀ and the discretized system is
// expressed by G(u, λ) = [R(u, λ), S(u, λ)]ᵀ, where:
//
// R(u, λ) = K ϕ + λ b + Eᵀψ = 0
// S(u, λ) = E ϕ - c = 0
//
// With bₘ = exp(ϕₘ/(1 + α ϕₘ)), the derivatives are:
//
//                  ⎧ bₘ/(1 + α ϕₘ)²  if m = n
// Bₘₙ := ∂bₘ/∂ϕₙ = ⎨
//                  ⎩ 0   otherwise
//
//      ┌              ┐   ┌              ┐
//      │ ∂R/∂ϕ  ∂R/∂ψ │   │ K + λ B   Eᵀ │
// Gu = │              │ = │              │
//      │ ∂S/∂ϕ  ∂S/∂ψ │   │ E         0  │
//      └              ┘   └              ┘
//
//      ┌       ┐   ┌   ┐
//      │ ∂R/∂λ │   │ b │
// Gλ = │       │ = │   │
//      │ ∂S/∂λ │   │ 0 │
//      └       ┘   └   ┘
//
// References:
//
// 1. Bank RE, Chan TF (1986) PLTMGC: A multi-grid continuation program for parametrized nonlinear elliptic systems.
//    SIAM Journal on Scientific and Statistical Computing, 7(2):540-559. https://doi.org/10.1137/0907036
// 2. Bolstad JH, Keller HB (1986) A multigrid continuation method for elliptic problems with folds.
//    SIAM Journal on Scientific and Statistical Computing, 7(4):1081-1104. https://doi.org/10.1137/0907074

const CHECK_JACOBIAN: bool = false;
const SAVE_FIGURE: bool = true;

fn run_test(bordering: bool, alpha: f64, npt: usize, stop: Stop, auto: AutoStep) {
    // filename stem
    let key = if auto.yes() { "auto" } else { "fixed" };
    let stem = format!(
        "/tmp/russell_nonlin/test_new_bratu_2d_lmm_alpha{}_npt{}_{}",
        alpha, npt, key
    );

    // check: alpha parameter in bₘ = exp(ϕₘ/(1 + α ϕₘ))
    assert!(alpha == 0.0 || alpha == 0.2, "alpha must be either 0.0 or 0.2");

    // allocate the grid
    let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, npt, npt).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new(&grid);
    ebcs.set_homogeneous();

    // auxiliary variables
    let n_phi = ebcs.num_total();
    let n_psi = ebcs.num_prescribed();
    let ndim = n_phi + n_psi;

    // allocate the Laplacian operator
    let fdm = FdmLaplacian2dNew::new(ebcs, 1.0, 1.0).unwrap();

    // augmented coefficient matrix of the Laplacian operator
    let (aa, _) = fdm.get_aa_and_ee_matrices(0, false);

    // function to calculate G(u, λ)
    let calc_gg = |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
        // ┌   ┐   ┌       ┐ ┌   ┐   ┌     ┐
        // │ R │   │ M  Eᵀ │ │ ϕ │   │ λ b │
        // │   │ = │       │ │   │ + │     │
        // │ S │   │ E  0  │ │ ψ │   │ -c  │
        // └   ┘   └       ┘ └   ┘   └     ┘
        //   G         A       u
        aa.mat_vec_mul(gg, 1.0, u).unwrap();
        // update R += λ b
        for m in 0..n_phi {
            let dm = 1.0 + alpha * u[m];
            let bm = f64::exp(u[m] / dm);
            gg[m] += l * bm;
        }
        // update S -= c (not needed since c = 0)
        Ok(())
    };

    // function to calculate Gu = ∂G/∂u (Jacobian matrix)
    let calc_ggu = |ggu_or_aa: &mut CooMatrix, l: f64, u: &Vector, _args: &mut NoArgs| {
        // note that ggu_or_aa may be the pseudo-arclength (larger) matrix
        ggu_or_aa.reset();
        ggu_or_aa.add(1.0, &aa).unwrap();
        // add λ B to the K term
        for m in 0..n_phi {
            let dm = 1.0 + alpha * u[m];
            let bm = f64::exp(u[m] / dm);
            ggu_or_aa.put(m, m, l * bm / (dm * dm)).unwrap();
        }
        // check Jacobian for smaller grids
        if CHECK_JACOBIAN && bordering && npt <= 21 {
            let ana = ggu_or_aa.as_dense();
            let num = num_jacobian(ndim, l, u, 1.0, &mut 0, calc_gg).unwrap();
            if npt <= 3 {
                println!("ana =\n{:.3}", ana);
                println!("num =\n{:.3}", num);
            }
            mat_approx_eq(&ana, &num, 1e-7);
        }
        Ok(())
    };

    // function to calculate Gl = ∂G/∂λ
    let calc_ggl = |ggl: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
        for m in 0..n_phi {
            let dm = 1.0 + alpha * u[m];
            let bm = f64::exp(u[m] / dm);
            ggl[m] = bm;
        }
        Ok(())
    };

    // allocate nonlinear problem
    let mut system = System::new(ndim, calc_gg).unwrap();

    // max number of non-zeros in Gu
    let nnz_aa = aa.get_info().2;
    let nnz = nnz_aa + n_phi; // the λ B term
    let sym = Sym::No;

    // set callback functions
    system.set_calc_ggu(Some(nnz), sym, calc_ggu).unwrap();
    system.set_calc_ggl(calc_ggl);

    // configuration
    let mut config = Config::new(Method::Arclength);
    config
        .set_n_cont_failure_max(5)
        .set_n_cont_rejection_max(5)
        .set_nr_control_enabled(true)
        .set_tg_control_enabled(true)
        .set_tg_control_pid_vcc(true)
        // .set_tg_control_atol_and_rtol(1e-5)
        // .set_tg_control_atol_and_rtol(1.1e-1)
        // .set_alpha_max(2.1)
        .set_alpha_max(3.73)
        .set_record_iterations_residuals(true)
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_debug_predictor(true)
        .set_log_file(&format!("{}.txt", stem))
        .set_bordering(bordering);

    // define the solver
    let mut solver = Solver::new(config, system).unwrap();

    // output
    let out = &mut Output::new();
    let norm_type_out = Norm::Inf;
    out.set_record_norm_u(true, norm_type_out, 0, n_phi);

    // initial state (all zero)
    let mut state = State::new(ndim);

    // numerical continuation
    let nrm_phi_stop = if alpha == 0.0 { 10.0 } else { 30.0 };
    let status = solver
        .solve(
            &mut 0,
            &mut state,
            IniDir::Pos,
            stop,
            auto,
            // Stop::Steps(65),
            // Stop::Steps(67),
            //Stop::MaxNormU(4.0, Norm::Inf, 0, n_phi),
            // Stop::MaxNormU(nrm_phi_stop, Norm::Inf, 0, n_phi),
            // Stop::MaxNormU(60.0, Norm::Euc, 0, n_phi),
            //AutoStep::Yes,
            // AutoStep::No(4.859), // ok
            // AutoStep::No(4.8599), // not ok
            // AutoStep::No(4.89516358573), // ok
            // AutoStep::No(4.89516358574), // not ok
            Some(out),
        )
        .unwrap();
    println!("Status: {:?}", status);
    // assert_eq!(status, Status::Success);

    // reference λCrit and ‖ϕCrit‖∞ values (Bolstad and Keller, 6 order scheme, very fine mesh)
    let ref_alp0 = (6.80812442259, 1.3916612); // α = 0: first critical point
    let ref_alp0d2_a = (9.13638296666, 2.8858004); // α = 0.2: first critical point
    let ref_alp0d2_b = (7.10189894953, 18.192768); // α = 0.2: second critical point

    // search for the critical point(s)
    let lam_vals = out.get_l_values();
    let nrm_vals = out.get_norm_u_values();
    let mut lam_crit_a = f64::NEG_INFINITY; // first λCrit (largest before turning point)
    let mut nrm_crit_a = 0.0; // ‖ϕ‖∞ @ first λCrit
    let mut lam_crit_b = f64::INFINITY; // second λCrit (smallest after turning point)
    let mut nrm_crit_b = 0.0; // ‖ϕ‖∞ @ second λCrit
    let mut found_first = false;
    for i in 0..lam_vals.len() {
        let lam = lam_vals[i];
        if alpha == 0.0 {
            if lam > lam_crit_a {
                lam_crit_a = lam;
                nrm_crit_a = nrm_vals[i];
            }
        } else if alpha == 0.2 {
            if found_first {
                if lam < lam_crit_b {
                    // now we are searching for the smallest value
                    lam_crit_b = lam;
                    nrm_crit_b = nrm_vals[i];
                }
            } else {
                if lam > lam_crit_a {
                    lam_crit_a = lam;
                    nrm_crit_a = nrm_vals[i];
                }
                if i > 0 {
                    if lam < lam_vals[i - 1] {
                        // it start to decrease; i.e., passed the first critical point
                        found_first = true;
                    }
                }
            }
        }
    }
    println!("\nNumerical results for α = {} and npt = {}:", alpha, npt);
    let tolerances = HashMap::from([
        (3, (1.47, 1.49)), // npt => (tol_lam_crit_a, tol_lam_crit_b)
        (5, (0.27, 0.43)),
        (21, (0.011, 0.0073)),
        (101, (0.00044, 0.052)),
    ]);
    if alpha == 0.0 {
        let err_lam_crit_a = f64::abs(lam_crit_a - ref_alp0.0);
        println!("λCrit = {} ({}), err = {}", lam_crit_a, ref_alp0.0, err_lam_crit_a);
        println!("‖ϕCrit‖∞ = {} ({})\n", nrm_crit_a, ref_alp0.1);
        // approx_eq(lam_crit_a, ref_alp0.0, tolerances[&npt].0);
    } else if alpha == 0.2 {
        let err_lam_crit_a = f64::abs(lam_crit_a - ref_alp0d2_a.0);
        let err_lam_crit_b = f64::abs(lam_crit_b - ref_alp0d2_b.0);
        println!(
            "First λCrit = {} ({}). diff = {}",
            lam_crit_a, ref_alp0d2_a.0, err_lam_crit_a
        );
        println!("First ‖ϕCrit‖∞ = {} ({})", nrm_crit_a, ref_alp0d2_a.1);
        println!(
            "Second λCrit = {} ({}). diff = {}",
            lam_crit_b, ref_alp0d2_b.0, err_lam_crit_b
        );
        println!("Second ‖ϕCrit‖∞ = {} ({})\n", nrm_crit_b, ref_alp0d2_b.1);
        // approx_eq(lam_crit_a, ref_alp0d2_a.0, tolerances[&npt].0);
        // approx_eq(lam_crit_b, ref_alp0d2_b.0, tolerances[&npt].1);
    }

    // plot the results
    if SAVE_FIGURE {
        // define the title
        let title = format!(
            "$\\alpha = {}$  |  npt = {}  |  $\\lambda_{{crit}} = {:.8}$",
            alpha, npt, lam_crit_a
        );

        // annotations
        let mut annotations = Text::new();
        let dy = if alpha == 0.0 { 1.0 } else { 2.0 };
        annotations
            .set_align_vertical("center")
            .set_bbox(true)
            .set_bbox_facecolor("white")
            .set_bbox_edgecolor("None")
            .set_bbox_style("round,pad=0.3")
            .draw(0.0, nrm_crit_a, &format!("{:.8}", nrm_crit_a));

        // draw ϕ versus λ
        let mut curve_norm_phi = Curve::new();
        curve_norm_phi.set_marker_style(".").draw(lam_vals, nrm_vals);

        // reference results
        let table: HashMap<String, Vec<f64>> =
            read_table(&"data/ref-bratu-2d-shahab-2025.txt", Some(&["lambda", "u_max"])).unwrap();
        let mut curve_ref = Curve::new();
        let n_ref = 150; // max = 201
        let x_ref = &table["lambda"].as_slice()[..n_ref];
        let y_ref = &table["u_max"].as_slice()[..n_ref];
        curve_ref.set_label("reference").draw(&x_ref, &y_ref);

        // generate the plot
        let mut plot = Plot::new();
        plot.set_horiz_line(nrm_crit_a, "#689868ff", "-", 1.0);
        if alpha == 0.2 {
            plot.set_horiz_line(nrm_crit_b, "#689868ff", "-", 1.0);
            annotations
                .set_align_horizontal("right")
                .set_align_vertical("center")
                .set_rotation(0.0)
                .draw(
                    lam_crit_b - 0.3,
                    nrm_crit_b,
                    &format!("({:.8}, {:.8}) →", lam_crit_b, nrm_crit_b),
                );
        }
        let key = if auto.yes() { "auto" } else { "fixed" };
        plot.set_title(&title)
            .add(&curve_ref)
            .add(&curve_norm_phi)
            .add(&annotations)
            .grid_and_labels("λ", &pretty_norm_phi(norm_type_out))
            .set_figure_size_points(400.0, 300.0)
            .save(&format!("{}.svg", stem))
            .unwrap();

        // plot stepsizes
        if auto.yes() {
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
                .save(&format!("{}_h.svg", stem))
                .unwrap();
        }
    }
}

fn pretty_norm_phi(norm_type: Norm) -> String {
    match norm_type {
        Norm::Euc => "‖ϕ‖₂".to_string(),
        Norm::Fro => "‖ϕ‖F".to_string(),
        Norm::Inf => "‖ϕ‖∞".to_string(),
        Norm::Max => "‖ϕ‖max".to_string(),
        Norm::One => "‖ϕ‖₁".to_string(),
    }
}

#[test]
fn test_bratu_2d_lmm_auto() {
    let bordering = false;
    let auto = AutoStep::Yes;
    for alpha in [0.0] {
        for npt in [8, 9, 10] {
            let n_phi = (npt - 2) * (npt - 2);
            let stop = Stop::MaxNormU(15.0, Norm::Inf, 0, n_phi);
            run_test(bordering, alpha, npt, stop, auto);
        }
    }
}

#[test]
fn test_bratu_2d_lmm_fixed() {
    let bordering = false;
    let auto = AutoStep::No(4.89516358573);
    let stop = Stop::Steps(67);
    for alpha in [0.0] {
        for npt in [5, 6, 7] {
            run_test(bordering, alpha, npt, stop, auto);
        }
    }
}
