use plotpy::{linspace, Curve, Plot, Text};
use russell_lab::{mat_approx_eq, num_jacobian, read_table, Norm, Vector};
use russell_nonlin::{AutoStep, Config, IniDir, Method, NoArgs, Output, Solver, State, Stop, System};
use russell_pde::{EssentialBcs2d, Fdm2d, Grid2d, NaturalBcs2d};
use russell_sparse::{CooMatrix, Sym};
use std::collections::HashMap;

// The nonlinear problem originates from the FDM discretization of the following equation:
//
// (Bratu's problem in 2D)
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
// The essential boundary conditions are handled by partitioning the linear system into known (bar) and
// unknown parts (check), with `ā` (a-bar) being the unknown values and `ǎ` (a-check) being the prescribed
// values. In this case, since the boundary conditions are homogeneous, `ǎ` = 0.
//
// The solution of the nonlinear problem is expressed by u = ā (unknowns) and nonlinear problem is
//
// G(u, λ) = K̄ ̄a + λ b = K̄ ̄a + λ b
//
// With bₘ = exp(ϕₘ/(1 + α ϕₘ)), the derivatives are:
//
//                  ⎧ bₘ/(1 + α ϕₘ)²  if m = n
// Bₘₙ := ∂bₘ/∂ϕₙ = ⎨
//                  ⎩ 0   otherwise
//
// Gu = ∂G/∂u = K̄ + λ B
// Gλ = ∂G/∂λ = b
//
// References:
//
// 1. Bank RE, Chan TF (1986) PLTMGC: A multi-grid continuation program for parametrized nonlinear elliptic systems.
//    SIAM Journal on Scientific and Statistical Computing, 7(2):540-559. <https://doi.org/10.1137/0907036>
// 2. Bolstad JH, Keller HB (1986) A multigrid continuation method for elliptic problems with folds.
//    SIAM Journal on Scientific and Statistical Computing, 7(4):1081-1104. <https://doi.org/10.1137/0907074>
// 3. Shahab ML, Susanto H, Hatzikirou H (2025) A finite difference method with symmetry properties for the high-dimensional
//    Bratu equation, Applied Mathematics and Computation, 489:129136, <https://doi.org/10.1016/j.amc.2024.129136>

const CHECK_JACOBIAN: bool = false;
const SAVE_FIGURE: bool = true;

#[test]
fn test_bratu_2d_auto() {
    let bordering = false;
    let auto = AutoStep::Yes;
    for alpha in [0.0] {
        for npt in [8, 9, 10] {
            let ndim = (npt - 2) * (npt - 2);
            let stop = Stop::MaxNormU(10.0, Norm::Inf, 0, ndim);
            run_test(bordering, alpha, npt, stop, auto);
        }
    }
}

fn run_test(bordering: bool, _alpha: f64, npt: usize, stop: Stop, auto: AutoStep) {
    // filename stem
    let key = if auto.yes() { "auto" } else { "fixed" };
    let stem = format!("/tmp/russell_nonlin/test_bratu_2d_npt{}_{}", npt, key);

    // allocate the grid
    let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, npt, npt).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set_homogeneous();

    // natural boundary conditions (NBCs) handler
    let nbcs = NaturalBcs2d::new();

    // allocate the Laplacian operator
    let fdm = Fdm2d::new(grid, ebcs, nbcs, -1.0, -1.0).unwrap();

    // number of unknowns
    let ndim = fdm.get_dims_sps().0;

    // get the discrete operator
    let (kk_bar, _) = fdm.get_matrices_sps(0, Sym::No);

    // function to calculate G(u, λ)
    let calc_gg = |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
        kk_bar.mat_vec_mul(gg, 1.0, &u).unwrap(); // G := K̄ ̄a
        for i in 0..ndim {
            gg[i] += l * f64::exp(u[i]); // G += λ exp(u)
        }
        Ok(())
    };

    // function to calculate Gu = ∂G/∂u (Jacobian matrix)
    let calc_ggu = |ggu_or_aa: &mut CooMatrix, l: f64, u: &Vector, _args: &mut NoArgs| {
        ggu_or_aa.reset();
        ggu_or_aa.add(1.0, &kk_bar).unwrap(); // Gu := K̄
        for i in 0..ndim {
            ggu_or_aa.put(i, i, l * f64::exp(u[i])).unwrap(); // Gu += λ B (diagonal)
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

    // function to calculate Gl = ∂G/∂λ
    let calc_ggl = |ggl: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
        for i in 0..ndim {
            ggl[i] = f64::exp(u[i]); // Gλ = b
        }
        Ok(())
    };

    // allocate nonlinear problem
    let mut system = System::new(ndim, calc_gg).unwrap();

    // max number of non-zeros in Gu
    let nnz_kk = kk_bar.get_info().2;
    let nnz = nnz_kk + ndim; // + diagonal from nonlinearity

    // set callback functions
    system.set_calc_ggu(Some(nnz), Sym::No, calc_ggu).unwrap();
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
        // .set_tol_residual(1e-10)
        // .set_tol_delta(1e-10, 1e-10)
        // .set_alpha_max(2.1)
        // .set_alpha_max(1.0)
        .set_alpha_max(3.73)
        .set_record_iterations_residuals(true)
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_debug_predictor(true)
        .set_log_file(&format!("{}.txt", stem))
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
        let key = if auto.yes() { "auto" } else { "fixed" };
        let title = format!("{}  |  npt = {}  |  $\\lambda_{{crit}} = {:.8}$", key, npt, lam_crit);

        // annotations
        let mut annotations = Text::new();
        annotations
            .set_align_vertical("center")
            .set_bbox(true)
            .set_bbox_facecolor("white")
            .set_bbox_edgecolor("None")
            .set_bbox_style("round,pad=0.3")
            .draw(0.0, nrm_crit, &format!("{:.8}", nrm_crit));

        // draw ϕ versus λ
        let mut curve_norm_phi = Curve::new();
        curve_norm_phi.set_marker_style(".").draw(lam_vals, nrm_vals);

        // reference results
        let table: HashMap<String, Vec<f64>> =
            read_table(&"data/ref-bratu-2d-shahab-2025.txt", Some(&["lambda", "u_max"])).unwrap();
        let mut curve_ref = Curve::new();
        let n_ref = 100; //150; // max = 201
        let x_ref = &table["lambda"].as_slice()[..n_ref];
        let y_ref = &table["u_max"].as_slice()[..n_ref];
        curve_ref.set_label("reference").draw(&x_ref, &y_ref);

        // generate the plot
        let mut plot = Plot::new();
        plot.set_title(&title)
            .add(&curve_ref)
            .set_horiz_line(nrm_crit, "#689868ff", "-", 1.0)
            .add(&curve_norm_phi)
            .add(&annotations)
            .grid_and_labels("λ", "‖ϕ‖∞")
            .set_figure_size_points(400.0, 300.0)
            .save(&format!("{}.svg", stem))
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
            .save(&format!("{}_h.svg", stem))
            .unwrap();
    }
}
