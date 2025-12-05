use plotpy::{linspace, Curve, Plot, Text};
use russell_lab::{find_index_abs_max, find_valleys_and_peaks, mat_approx_eq, num_jacobian, read_table};
use russell_lab::{Norm, Vector};
use russell_nonlin::{AutoStep, Config, IniDir, Method, NoArgs, Output, Solver, State, Status, Stop, System};
use russell_pde::{EssentialBcs1d, FdmLaplacian1d, Grid1d};
use russell_sparse::{CooMatrix, Sym};
use std::collections::HashMap;

// The nonlinear problem originates from the FDM discretization of the following equation:
//
// (Bratu's problem in 1D with Lagrange multipliers method - LMM)
//
// ∂²ϕ
// ——— + λ exp(ϕ/(1 + α ϕ)) = 0
// ∂x²
//
// on the unit segment with homogeneous boundary conditions.
//
// Below, "a" is a vector with the discretized ϕ values, i.e., a = [a₀, a₁, a₂, ..., aₙ₋₁]ᵀ, where n is the
// number of grid points. The prescribed values are collected in the vector p = [p₀, p₁, p₂, ..., pₘ₋₁]ᵀ,
// where m is the number of boundary points. The Laplacian operator is represented by the matrix M, thus
// "M a" is the discretization of the Laplacian operator applied to ϕ(x,y).
//
// The boundary conditions are enforced via Lagrange multipliers indicated by μ = [μ₀, μ₁, μ₂, ..., μₘ₋₁]ᵀ.
// In this case, the prescribed values are all zero (homogeneous boundary conditions), thus p = [0, 0, ... 0]ᵀ.
// The constraints matrix for the Lagrange multipliers method is E, thus E μ = p.
//
// The solution of the nonlinear problem is expressed by u = [a, μ]ᵀ and the discretized system is
// expressed by G(u, λ) = [R(u, λ), S(u, λ)]ᵀ, where:
//
// R(u, λ) = M a + λ b + Eᵀμ = 0
// S(u, λ) = E μ - p = 0
//
// With bₘ = exp(ϕₘ/(1 + α ϕₘ)), the derivatives are:
//
//                  ⎧ bₘ/(1 + α ϕₘ)²  if m = n
// Bₘₙ := ∂bₘ/∂ϕₙ = ⎨
//                  ⎩ 0   otherwise
//
//      ┌              ┐   ┌              ┐
//      │ ∂R/∂ϕ  ∂R/∂ψ │   │ M + λ B   Eᵀ │
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
// 3. Shahab ML, Susanto H, Hatzikirou H (2025) A finite difference method with symmetry properties for the high-dimensional
//    Bratu equation, Applied Mathematics and Computation, 489:129136, https://doi.org/10.1016/j.amc.2024.129136

// Analytical u(x) profile @ λCrit (from Mathematica)
const REF_ALP00: f64 = 3.51383071912516; // λ critical for the α = 0.0 case
const REF_THETA: f64 = 4.79871456103094; // θ critical (for the analytical profile); α = 0.0 case

// TODO: Reference results for α = 0.2
const REF_ALP02_A: f64 = 4.647906373918411; // 1st λ critical for α = 0.2 (from npt = 500 and tol = 1e-8); nrm=2.3548402404342146
const REF_ALP02_B: f64 = 3.509919925802271; // 2nd λ critical for α = 0.2 (from npt = 500 and tol = 1e-8); nrm=15.440772685670549

// Calculates the analytical solution at λCrit
fn analytical_profile(x: f64) -> f64 {
    -2.0 * f64::ln(f64::cosh((x - 0.5) * REF_THETA / 2.0) / f64::cosh(REF_THETA / 4.0))
}

const CHECK_JACOBIAN: bool = false;
const SAVE_FIGURE: bool = true;

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
        "/tmp/russell_nonlin/test_bratu_1d_lmm_alpha{}_npt{}_{}",
        alpha, npt, key
    );

    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, 1.0, npt).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set_homogeneous(&grid);

    // allocate the Laplacian operator
    let fdm = FdmLaplacian1d::new(grid, ebcs, 1.0).unwrap();

    // auxiliary variables
    let (_, _, na, _, ndim) = fdm.get_info();

    // A matrix: augmented coefficient matrix of the Laplacian operator
    let (aa, _) = fdm.get_matrices_lmm(0, false);

    // function to calculate G(u, λ)
    let calc_gg = |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
        // ┌   ┐   ┌       ┐ ┌   ┐   ┌     ┐
        // │ R │   │ M  Eᵀ │ │ a │   │ λ b │
        // │   │ = │       │ │   │ + │     │
        // │ S │   │ E  0  │ │ μ │   │ -p  │
        // └   ┘   └       ┘ └   ┘   └     ┘
        //   G         A       u
        aa.mat_vec_mul(gg, 1.0, u).unwrap();
        // update R += λ b
        for m in 0..na {
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
        //      │ M  Eᵀ │   │ λ B  0 │
        // Gu = │       │ + │        │
        //      │ E  0  │   │   0  0 │
        //      └       ┘   └        ┘
        ggu_or_aa.reset();
        ggu_or_aa.add(1.0, &aa).unwrap();
        // add λ B to the M term
        for m in 0..na {
            let dm = 1.0 + alpha * u[m];
            let bm = f64::exp(u[m] / dm);
            ggu_or_aa.put(m, m, l * bm / (dm * dm)).unwrap();
        }
        // check Jacobian for smaller grids
        if CHECK_JACOBIAN && bordering && npt <= 21 {
            let ana = ggu_or_aa.as_dense();
            let num = num_jacobian(ndim, l, u, 1.0, &mut 0, calc_gg).unwrap();
            if npt <= 5 {
                println!("ana =\n{:.3}", ana);
                println!("num =\n{:.3}", num);
            }
            mat_approx_eq(&ana, &num, 1e-7);
        }
        Ok(())
    };

    // function to calculate Gl = ∂G/∂λ
    let calc_ggl = |ggl: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
        for m in 0..na {
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
    let nnz = nnz_aa + na; // +na for the λ B term
    let sym = Sym::No;

    // set callback functions
    system.set_calc_ggu(Some(nnz), sym, calc_ggu).unwrap();
    system.set_calc_ggl(calc_ggl);

    // configuration
    let mut config = Config::new(Method::Arclength);
    config
        .set_n_cont_failure_max(5)
        .set_n_cont_rejection_max(5)
        // .set_tg_control_atol_and_rtol(1e-5)
        // .set_alpha_max(0.01)
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_debug_predictor(true)
        .set_log_file(&format!("{}.txt", stem))
        .set_bordering(bordering);

    // define the solver
    let mut solver = Solver::new(config, system).unwrap();

    // output
    let out = &mut Output::new();
    let all_indices: Vec<usize> = (0..npt).collect();
    out.set_recording(true, &all_indices, &[])
        .set_record_norm_u(true, Norm::Inf, 0, na);

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
                read_table(&"data/ref-bratu-1d-shahab-2025.txt", Some(&["lambda", "u_max"])).unwrap();
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

        // profile: draw ϕ along x @ λCrit
        if alpha == 0.0 && ii_peaks.len() == 1 {
            let mut curve = Curve::new();
            let mut curve_profile_crit_num = Curve::new();
            let xx_ana = linspace(0.0, 1.0, 201);
            let phi_crit_ana: Vec<_> = xx_ana.iter().map(|&x| analytical_profile(x)).collect();
            curve
                .set_label("Mathematica")
                .set_line_style("-")
                .set_line_color("#1f53d6ff")
                .draw(&xx_ana, &phi_crit_ana);
            let mut xx_num = vec![0.0; npt];
            let mut phi_crit_num = vec![0.0; npt];
            fdm.loop_over_grid_points(|m, x| {
                xx_num[m] = x;
                phi_crit_num[m] = out.get_u_values(m)[ii_peaks[0]];
            });
            curve_profile_crit_num
                .set_label("Russell")
                .set_line_style("None")
                .set_marker_style(".")
                .set_marker_color("#d8211aff")
                .set_marker_line_color("#d8211aff")
                .draw(&xx_num, &phi_crit_num);
            let mut plot = Plot::new();
            plot.set_title(&format!("{}  |  $\\alpha = {}$  |  npt = {}", key, alpha, npt))
                .add(&curve)
                .add(&curve_profile_crit_num)
                .grid_labels_legend("x", "$\\phi_{crit}(x)$")
                .save(&format!("{}_profile.svg", stem))
                .unwrap();
        }
    }
}

#[test]
fn test_bratu_1d_lmm_auto() {
    let bordering = false;
    let auto = AutoStep::Yes;
    for alpha in [0.0, 0.2] {
        for (npt, tol1, tol2, tol3) in [(8, 0.0646, 0.06, 0.0071)] {
            let max_nrm_max = if alpha == 0.0 { 9.0 } else { 30.0 };
            let stop = Stop::MaxNormU(max_nrm_max, Norm::Inf, 0, npt);
            run_test(bordering, alpha, npt, stop, auto, tol1, tol2, tol3);
        }
    }
}
