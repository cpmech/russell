use plotpy::{Curve, Plot, Text};
use russell_lab::{approx_eq, mat_approx_eq, num_jacobian, Norm, Vector};
use russell_nonlin::{AutoStep, Config, IniDir, Method, NoArgs, Output, Solver, State, Stop, System};
use russell_pde::FdmLaplacian2d;
use russell_sparse::{CooMatrix, Sym};
use std::collections::HashMap;

const CHECK_JACOBIAN: bool = false;
const SAVE_FIGURE: bool = true;

#[test]
fn test_reaction_diffusion_2d() {
    // The nonlinear problem originates from the FDM discretization of the following equation:
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

    // alpha parameter in bₘ = exp(ϕₘ/(1 + α ϕₘ))
    const ALPHA: f64 = 0.2;
    assert!(ALPHA == 0.0 || ALPHA == 0.2, "ALPHA must be either 0.0 or 0.2");

    // number of points along each axis of the FDM grid (must be ODD)
    const NPT: usize = 3;
    // const NPT: usize = 5;
    // const NPT: usize = 21;
    // const NPT: usize = 101;
    assert_eq!(NPT % 2, 1, "NPT must be odd");

    // allocate the Laplacian operator
    let mut fdm = FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, NPT, NPT).unwrap();
    fdm.set_homogeneous_boundary_conditions();

    // check if there is a middle point
    let i_middle = NPT / 2;
    let j_middle = NPT / 2;
    let m_middle = i_middle + j_middle * NPT;
    fdm.loop_over_grid_points(|m, x, y| {
        if m == m_middle {
            assert_eq!(x, 0.5, "the middle point must be at x = 0.5");
            assert_eq!(y, 0.5, "the middle point must be at y = 0.5");
        }
    });

    // auxiliary variables
    let n_phi = fdm.dim(); // number of unknowns
    let n_psi = fdm.num_prescribed(); // number of Lagrange multipliers
    let ndim = n_phi + n_psi;

    // augmented coefficient matrix of the Laplacian operator
    //     ┌       ┐
    //     │ K  Eᵀ │
    // A = │       │
    //     │ E  0  │
    //     └       ┘
    let aa = fdm.augmented_coefficient_matrix(0).unwrap();

    // function to calculate G(u, λ)
    let calc_gg = |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
        // ┌   ┐   ┌       ┐ ┌   ┐   ┌     ┐
        // │ R │   │ K  Eᵀ │ │ ϕ │   │ λ b │
        // │   │ = │       │ │   │ + │     │
        // │ S │   │ E  0  │ │ ψ │   │ -c  │
        // └   ┘   └       ┘ └   ┘   └     ┘
        //   G         A       u
        aa.mat_vec_mul(gg, 1.0, u).unwrap();
        // update R += λ b
        for m in 0..n_phi {
            let dm = 1.0 + ALPHA * u[m];
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
            let dm = 1.0 + ALPHA * u[m];
            let bm = f64::exp(u[m] / dm);
            ggu_or_aa.put(m, m, l * bm / (dm * dm)).unwrap();
        }
        // check Jacobian for smaller grids
        if CHECK_JACOBIAN && NPT <= 21 {
            let ana = ggu_or_aa.as_dense();
            let num = num_jacobian(ndim, l, u, 1.0, &mut 0, calc_gg).unwrap();
            if NPT <= 3 {
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
            let dm = 1.0 + ALPHA * u[m];
            let bm = f64::exp(u[m] / dm);
            ggl[m] = bm;
        }
        Ok(())
    };

    // allocate nonlinear problem
    let mut system = System::new(ndim, calc_gg).unwrap();

    // max number of non-zeros in Gu
    let nnz_a = aa.get_info().2;
    let nnz = nnz_a + n_phi; // the λ B term
    let sym = Sym::No;

    // set callback functions
    system.set_calc_ggu(Some(nnz), sym, calc_ggu).unwrap();
    system.set_calc_ggl(calc_ggl);

    // configuration
    // (need to use bordering if checking the Jacobian because the
    // matrix provided contains is not Gu, but the augmented one)
    let mut config = Config::new(Method::Arclength);
    config
        .set_n_cont_failure_max(5)
        .set_n_cont_rejection_max(5)
        .set_tg_control_atol_and_rtol(1e-4)
        // .set_alpha_max(0.01)
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_debug_predictor(true)
        .set_bordering(CHECK_JACOBIAN);

    // define the solver
    let mut solver = Solver::new(config, system).unwrap();

    // output
    let out = &mut Output::new();
    let norm_type_out = Norm::Inf;
    out.set_record_norm_u(true, norm_type_out, 0, n_phi);

    // initial state (all zero)
    let mut state = State::new(ndim);

    // numerical continuation
    let npt_f64 = NPT as f64;
    let max_norm = if ALPHA == 0.0 { 2.0 * npt_f64 } else { 15.0 * npt_f64 };
    let status = solver
        .solve(
            &mut 0,
            &mut state,
            IniDir::Pos,
            Stop::MaxNormU(max_norm, Norm::Euc, 0, n_phi),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    println!("Status: {:?}", status);

    // reference λCrit and ‖ϕCrit‖∞ values (Bolstad and Keller, 6 order scheme, very fine mesh)
    let ref_alp0 = (6.80812442259, 1.3916612); // α = 0: first critical point
    let ref_alp0d2_a = (9.13638296666, 2.8858004); // α = 0.2: first critical point
    let ref_alp0d2_b = (7.10189894953, 18.192768); // α = 0.2: second critical point

    // numerical results
    let lam_vals = out.get_l_values();
    let nrm_vals = out.get_norm_u_values();
    let mut lam_crit_a = f64::NEG_INFINITY; // first λCrit (largest before turning point)
    let mut nrm_crit_a = 0.0; // ‖ϕ‖∞ @ first λCrit
    let mut lam_crit_b = f64::INFINITY; // second λCrit (smallest after turning point)
    let mut nrm_crit_b = 0.0; // ‖ϕ‖∞ @ second λCrit
    let mut found_first = false;
    for i in 0..lam_vals.len() {
        let lam = lam_vals[i];
        if ALPHA == 0.0 {
            if lam > lam_crit_a {
                lam_crit_a = lam;
                nrm_crit_a = nrm_vals[i];
            }
        } else if ALPHA == 0.2 {
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
    println!("\nNumerical results:");
    let tolerances = HashMap::from([
        (3, (1.47, 1.49)), // npt => (tol_lam_crit_a, tol_lam_crit_b)
        (5, (0.27, 0.43)),
        (21, (0.011, 0.0073)),
        (101, (0.0004, 0.052)),
    ]);
    if ALPHA == 0.0 {
        println!("λCrit = {} ({})", lam_crit_a, ref_alp0.0);
        println!("‖ϕCrit‖∞ = {} ({})\n", nrm_crit_a, ref_alp0.1);
    } else if ALPHA == 0.2 {
        println!("First λCrit = {} ({})", lam_crit_a, ref_alp0d2_a.0);
        println!("First ‖ϕCrit‖∞ = {} ({})", nrm_crit_a, ref_alp0d2_a.1);
        println!("Second λCrit = {} ({})", lam_crit_b, ref_alp0d2_b.0);
        println!("Second ‖ϕCrit‖∞ = {} ({})\n", nrm_crit_b, ref_alp0d2_b.1);
        approx_eq(lam_crit_a, ref_alp0d2_a.0, tolerances[&NPT].0);
        approx_eq(lam_crit_b, ref_alp0d2_b.0, tolerances[&NPT].1);
    }

    // plot the results
    if SAVE_FIGURE {
        // define the title
        let title = format!("$\\alpha = {}$  |  npt = {}", ALPHA, NPT,);

        // annotations
        let mut annotations = Text::new();
        annotations
            .set_bbox(true)
            .set_bbox_facecolor("white")
            .set_bbox_edgecolor("None")
            .set_bbox_style("round,pad=0.3")
            .set_rotation(90.0)
            .draw(
                lam_crit_a,
                nrm_crit_a + 2.0,
                &format!("← ({:.8}, {:.8})", lam_crit_a, nrm_crit_a),
            );

        // draw ϕ versus λ
        let mut curve_norm_phi = Curve::new();
        curve_norm_phi.set_marker_style(".").draw(lam_vals, nrm_vals);

        // generate the plot
        let mut plot = Plot::new();
        plot.set_horiz_line(nrm_crit_a, "#689868ff", "-", 1.0);
        if ALPHA == 0.2 {
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
        plot.set_title(&title)
            .add(&curve_norm_phi)
            .add(&annotations)
            .grid_and_labels("λ", &pretty_norm_phi(norm_type_out))
            .set_figure_size_points(400.0, 300.0)
            .save(&format!(
                "/tmp/russell_nonlin/test_reaction_diffusion_2d_alpha{}_npt{}.svg",
                ALPHA, NPT
            ))
            .unwrap();
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
