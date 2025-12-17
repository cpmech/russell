use plotpy::{linspace, Curve, Plot, Text};
use russell_lab::{find_index_abs_max, find_valleys_and_peaks, mat_approx_eq, num_jacobian, read_table};
use russell_lab::{Norm, Vector};
use russell_nonlin::{AutoStep, Config, IniDir, Method, NoArgs, Output, Solver, State, Status, Stop, System};
use russell_pde::{EssentialBcs2d, Fdm2d, Grid2d};
use russell_sparse::{CooMatrix, Sym};
use std::collections::HashMap;

// The nonlinear problem originates from the FDM discretization of the following equation:
//
// (Bratu's problem in 2D with Lagrange multipliers method - LMM)
//
// ∂²ϕ   ∂²ϕ
// ——— + ——— + λ exp(ϕ/(1 + α ϕ)) = 0
// ∂x²   ∂y²
//
// on the unit square (1.0 × 1.0) with homogeneous boundary conditions.
//
// Below, "a" is a vector with the discretized ϕ values, i.e., a = [a₀, a₁, a₂, ..., aₙ₋₁]ᵀ, where n is the
// number of grid points. The prescribed values are collected in the vector p = [p₀, p₁, p₂, ..., pₘ₋₁]ᵀ,
// where m is the number of boundary points. The Laplacian operator is represented by the matrix K, thus
// "K a" is the discretization of the Laplacian operator applied to ϕ(x,y).
//
// The boundary conditions are enforced via Lagrange multipliers indicated by μ = [μ₀, μ₁, μ₂, ..., μₘ₋₁]ᵀ.
// In this case, the prescribed values are all zero (homogeneous boundary conditions), thus p = [0, 0, ... 0]ᵀ.
// The constraints matrix for the Lagrange multipliers method is C, thus C μ = p.
//
// The solution of the nonlinear problem is expressed by u = [a, μ]ᵀ and the discretized system is
// expressed by G(u, λ) = [R(u, λ), S(u, λ)]ᵀ, where:
//
// R(u, λ) = K a + λ b + Cᵀμ = 0
// S(u, λ) = C μ - p = 0
//
// With bₘ = exp(ϕₘ/(1 + α ϕₘ)), the derivatives are:
//
//                  ⎧ bₘ/(1 + α ϕₘ)²  if m = n
// Bₘₙ := ∂bₘ/∂ϕₙ = ⎨
//                  ⎩ 0   otherwise
//
//      ┌              ┐   ┌              ┐
//      │ ∂R/∂ϕ  ∂R/∂ψ │   │ K + λ B   Cᵀ │
// Gu = │              │ = │              │
//      │ ∂S/∂ϕ  ∂S/∂ψ │   │ C         0  │
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
//    SIAM Journal on Scientific and Statistical Computing, 7(2):540-559. <https://doi.org/10.1137/0907036>
// 2. Bolstad JH, Keller HB (1986) A multigrid continuation method for elliptic problems with folds.
//    SIAM Journal on Scientific and Statistical Computing, 7(4):1081-1104. <https://doi.org/10.1137/0907074>
// 3. Shahab ML, Susanto H, Hatzikirou H (2025) A finite difference method with symmetry properties for the high-dimensional
//    Bratu equation, Applied Mathematics and Computation, 489:129136, <https://doi.org/10.1016/j.amc.2024.129136>

// reference (λCrit, ‖ϕCrit‖∞) values from Bolstad and Keller (6 order scheme, very fine mesh)
const REF_ALP00: f64 = 6.80812442259; // α = 0: first critical point; nrm=1.3916612
const REF_ALP02_A: f64 = 9.13638296666; // α = 0.2: first critical point; nrm=2.8858004
const REF_ALP02_B: f64 = 7.10189894953; // α = 0.2: second critical point; nrm=18.192768

const CHECK_JACOBIAN: bool = false;
const SAVE_FIGURE: bool = true;

#[test]
fn test_bratu_2d_lmm_auto() {
    let bordering = false;
    let auto = AutoStep::Yes;
    for alpha in [0.0] {
        for (npt, tol1, tol2, tol3) in [(8, 0.0672, 0.094, 0.11), (9, 0.101, 0.07, 0.053)] {
            let max_nrm_max = if alpha == 0.0 { 15.0 } else { 40.0 };
            let n_phi = (npt - 2) * (npt - 2);
            let stop = Stop::MaxNormU(max_nrm_max, Norm::Inf, 0, n_phi);
            run_test(bordering, alpha, npt, stop, auto, tol1, tol2, tol3);
        }
    }
}

#[test]
fn test_bratu_2d_lmm_fixed() {
    let bordering = false;
    let auto = AutoStep::No(4.89516358573);
    let stop = Stop::Steps(67);
    for alpha in [0.0] {
        for (npt, tol1, tol2, tol3) in [(8, 0.0332, 0.0, 0.0)] {
            run_test(bordering, alpha, npt, stop, auto, tol1, tol2, tol3);
        }
    }
}

// Runs the test
fn run_test(
    bordering: bool,
    alpha: f64,
    npt: usize,
    stop: Stop,
    auto: AutoStep,
    alpha0_lam_crit_tol: f64,
    alpha02_1st_lam_crit_tol: f64,
    alpha02_2nd_lam_crit_tol: f64,
) {
    // filename stem
    let key = if auto.yes() { "auto" } else { "fixed" };
    let stem = format!(
        "/tmp/russell_nonlin/test_bratu_2d_lmm_alpha{}_npt{}_{}",
        alpha, npt, key
    );

    // allocate the grid
    let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, npt, npt).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set_homogeneous();

    // allocate the Laplacian operator
    let fdm = Fdm2d::new(grid, ebcs, -1.0, -1.0).unwrap();

    // auxiliary variables
    let (neq, _, ndim) = fdm.get_dims_lmm();

    // augmented coefficient matrix of the Laplacian operator
    let (aug_mat, _) = fdm.get_matrices_lmm(0, false);

    // function to calculate G(u, λ)
    let calc_gg = |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
        // ┌   ┐   ┌       ┐ ┌   ┐   ┌     ┐
        // │ R │   │ K  Cᵀ │ │ a │   │ λ b │
        // │   │ = │       │ │   │ + │     │
        // │ S │   │ C  0  │ │ μ │   │ -p  │
        // └   ┘   └       ┘ └   ┘   └     ┘
        //   G      aug_mat    u
        aug_mat.mat_vec_mul(gg, 1.0, u).unwrap();
        // update R += λ b
        for m in 0..neq {
            let dm = 1.0 + alpha * u[m];
            let bm = f64::exp(u[m] / dm);
            gg[m] += l * bm;
        }
        // update S -= p (not needed since p = 0)
        Ok(())
    };

    // function to calculate Gu = ∂G/∂u (Jacobian matrix)
    let calc_ggu = |ggu_or_aa: &mut CooMatrix, l: f64, u: &Vector, _args: &mut NoArgs| {
        // note that ggu_or_aa may be the pseudo-arclength (larger) matrix
        //      ┌       ┐   ┌        ┐
        //      │ K  Cᵀ │   │ λ B  0 │
        // Gu = │       │ + │        │
        //      │ C  0  │   │   0  0 │
        //      └       ┘   └        ┘
        ggu_or_aa.reset();
        ggu_or_aa.add(1.0, &aug_mat).unwrap();
        // add λ B to the K term
        for m in 0..neq {
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
        for m in 0..neq {
            let dm = 1.0 + alpha * u[m];
            let bm = f64::exp(u[m] / dm);
            ggl[m] = bm;
        }
        Ok(())
    };

    // allocate nonlinear problem
    let mut system = System::new(ndim, calc_gg).unwrap();

    // max number of non-zeros in Gu
    let nnz_aa = aug_mat.get_info().2;
    let nnz = nnz_aa + neq; // +na for the λ B term
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
    out.set_record_norm_u(true, Norm::Inf, 0, neq);

    // initial state (all zero)
    let mut state = State::new(ndim);

    // numerical continuation
    let status = solver
        .solve(&mut 0, &mut state, IniDir::Pos, stop, auto, Some(out))
        .unwrap();
    println!("\nStatus: {:?}", status);
    assert_eq!(status, Status::Success);

    // search for the critical point(s)
    println!("Numerical results for α = {} and npt = {}:", alpha, npt);
    let lam_vals = out.get_l_values();
    let nrm_vals = out.get_norm_u_values();
    let (ii_valleys, ii_peaks, _, _) = find_valleys_and_peaks(lam_vals);
    for i in &ii_peaks {
        println!("Peak   @ ({}, {})", lam_vals[*i], nrm_vals[*i]);
    }
    for i in &ii_valleys {
        println!("Valley @ ({}, {})", lam_vals[*i], nrm_vals[*i]);
    }

    // check the results
    if alpha == 0.0 {
        if ii_peaks.len() == 1 {
            let lam_crit = lam_vals[ii_peaks[0]];
            let diff = f64::abs(lam_crit - REF_ALP00);
            if diff > alpha0_lam_crit_tol {
                println!("❌ ERROR ❌ λCrit = {}, ref = {}, diff = {}", lam_crit, REF_ALP00, diff);
            }
        } else {
            println!("WARNING: for alpha = 0.0, one peak must have been found");
        }
    } else if alpha == 0.2 {
        if ii_peaks.len() == 1 && ii_valleys.len() == 1 {
            let lam_crit_a = lam_vals[ii_peaks[0]];
            let lam_crit_b = lam_vals[ii_valleys[0]];
            let diff_a = f64::abs(lam_crit_a - REF_ALP02_A);
            let diff_b = f64::abs(lam_crit_b - REF_ALP02_B);
            if diff_a > alpha02_1st_lam_crit_tol {
                println!(
                    "❌ ERROR ❌ 1st λCrit = {}, ref = {}, diff = {}",
                    lam_crit_a, REF_ALP02_A, diff_a
                );
            }
            if diff_b > alpha02_2nd_lam_crit_tol {
                println!(
                    "❌ ERROR ❌ 2nd λCrit = {}, ref = {}, diff = {}",
                    lam_crit_b, REF_ALP02_B, diff_b
                );
            }
        } else {
            println!("WARNING: for alpha = 0.2, one peak and one valley must have been found");
        }
    }
    println!();

    // plot the results
    if SAVE_FIGURE {
        // allocate the plot
        let mut plot = Plot::new();

        // set the title
        plot.set_title(&format!("{}  |  $\\alpha = {}$  |  npt = {}", key, alpha, npt));

        // maximum ‖ϕ‖∞ value
        let max_nrm_max = nrm_vals[find_index_abs_max(&nrm_vals)];

        // reference results
        if alpha == 0.0 {
            let table: HashMap<String, Vec<f64>> =
                read_table(&"data/ref-bratu-2d-shahab-2025.txt", Some(&["lambda", "u_max"])).unwrap();
            let mut n_ref = 0;
            for u_max in &table["u_max"] {
                if *u_max > max_nrm_max {
                    break;
                }
                n_ref += 1;
            }
            if n_ref + 5 < table["u_max"].len() {
                n_ref += 5; // add a few more points for better visualization
            }
            let mut curve_ref = Curve::new();
            let x_ref = &table["lambda"].as_slice()[..n_ref];
            let y_ref = &table["u_max"].as_slice()[..n_ref];
            curve_ref.set_label("reference").draw(&x_ref, &y_ref);
            plot.add(&curve_ref);
        }

        // numerical results
        let mut curve = Curve::new();
        curve.set_marker_style(".").draw(lam_vals, nrm_vals);
        plot.add(&curve);

        // annotations
        let mut annotations = Text::new();
        annotations
            .set_bbox(true)
            .set_bbox_facecolor("white")
            .set_bbox_edgecolor("None")
            .set_bbox_style("round,pad=0.3");
        let indices = [&ii_valleys[..], &ii_peaks[..]].concat();
        for i in &indices {
            plot.set_horiz_line(nrm_vals[*i], "#689868ff", "-", 1.0);
            annotations
                .set_rotation(0.0)
                .set_align_vertical("center")
                .set_align_horizontal("left")
                .draw(0.0, nrm_vals[*i], &format!("{:.9}", nrm_vals[*i]));
            plot.set_vert_line(lam_vals[*i], "#689868ff", "-", 1.0);
            annotations
                .set_rotation(90.0)
                .set_align_vertical("top")
                .set_align_horizontal("center")
                .draw(lam_vals[*i], max_nrm_max, &format!("{:.9}", lam_vals[*i]));
        }
        plot.add(&annotations);

        // generate ‖ϕ‖∞ versus λ plot
        let key = if auto.yes() { "auto" } else { "fixed" };
        plot.set_labels("λ", "‖ϕ‖∞").save(&format!("{}.svg", stem)).unwrap();

        // plot stepsizes
        if auto.yes() {
            let hh = &out.get_h_values()[1..]; // the first one is duplicated
            let n = hh.len();
            let x = linspace(1.0, n as f64, n);
            let mut curve = Curve::new();
            curve.set_label("stepsize").set_line_style("-").set_marker_style(".");
            curve.draw(&x.as_slice(), &hh);
            let mut plot = Plot::new();
            plot.set_title(&format!("{}  |  $\\alpha = {}$  |  npt = {}", key, alpha, npt))
                .set_labels("step number", "stepsize $h$")
                .add(&curve)
                .save(&format!("{}_h.svg", stem))
                .unwrap();
        }
    }
}
