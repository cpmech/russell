use plotpy::{linspace, Curve, Plot, Text};
use russell_lab::{find_index_abs_max, find_valleys_and_peaks, mat_approx_eq, num_jacobian, read_table};
use russell_lab::{Norm, Vector};
use russell_nonlin::{AutoStep, Config, IniDir, Method, Output, Solver, Status, Stop, StrError, System};
use russell_pde::{
    EssentialBcs1d, EssentialBcs2d, Fdm1d, Fdm2d, Grid1d, Grid2d, NaturalBcs1d, NaturalBcs2d, Spc1d, Spc2d,
};
use russell_sparse::{CooMatrix, Sym};
use std::collections::HashMap;

// Bratu problem
//
// The nonlinear problem originates from the discretization of the following equation:
//
// ∂²ϕ   ∂²ϕ
// ——— + ——— + λ exp(ϕ/(1 + α ϕ)) = 0
// ∂x²   ∂y²
//
// on the unit square `[0,0]×[1,1]` with homogeneous boundary conditions.
//
// ϕ is discretized into values a = [a₀, a₁, a₂, ..., aₙ₋₁]ᵀ, where n is the number of grid points.
// The prescribed values are collected in the vector ǎ = [ǎ₀, ǎ₁, ǎ₂, ..., ǎₘ₋₁]ᵀ, where m is
// the number of boundary points. Here, all components of ǎ are zero. Consider the following definitions:
//
// * `nu` - Number of unknown degrees of freedom
// * `np` - Number of prescribed degrees of freedom
// * `neq` - `nu + np`
//
// The Laplacian operator is represented by the matrix K, thus "K a" is the discretization of the
// Laplacian operator applied to ϕ(x,y).
//
// Two approaches are considered to handle the essential boundary conditions (EBCs):
//
// 1. SPS (System partitioning strategy) - handled by partitioning the linear system as follows:
//
// ┌       ┐ ┌   ┐   ┌   ┐
// │ K̄   Ǩ │ │ ̄a │   │ f̄ │    (nu = nphi)
// │       │ │   │ = │   │
// │ Ḵ   ̰K │ │ ǎ │   │ f̌ │    (np)
// └       ┘ └   ┘   └   ┘
//     K       a       f
//
// where ̄a (a-bar) are the unknown values and ǎ (a-check) are the prescribed values.
//
// 2. LMM (Lagrange multipliers method) - handled by introducing Lagrange multipliers as follows:
//
// ┌       ┐ ┌   ┐   ┌   ┐
// │ K  Cᵀ │ │ a │   │ f │   (neq = nphi)
// │       │ │   │ = │   │
// │ C  0  │ │ ℓ │   │ ǎ │   (np)
// └       ┘ └   ┘   └   ┘
//     M       A       F
//
// where ℓ are the Lagrange multipliers and C is the constraints matrix for the EBCs.
//
// The nonlinear problem for the SPS approach is (with u = ā):
//
// G(u, λ) = K̄ ̄a + λ b = 0    (ndim = nu = nphi)
//
// whereas the problem for the LMM approach is (with u = (a, ℓ)):
//
//           ⎧ R(u, λ) = K a + λ b + Cᵀℓ = 0  (neq = nphi)
// G(u, λ) = ⎨
//           ⎩ S(u, λ) = C ℓ - ǎ = 0          (np)
//
// where ndim = neq + np.
//
// The required derivatives are discussed next.
//
// With bₘ = exp(ϕₘ/(1 + α ϕₘ)), the derivatives are:
//
//                  ⎧ bₘ/(1 + α ϕₘ)²  if m = n
// Bₘₙ := ∂bₘ/∂ϕₙ = ⎨
//                  ⎩ 0   otherwise
//
// For the SPS approach (with u = ā):
//
// Gu = ∂G/∂u = K̄ + λ B       (nu = ndim)
// Gλ = ∂G/∂λ = b             (nu = ndim)
//
// For the LMM approach (with u = (a, ℓ)):
//
//      ┌              ┐   ┌              ┐
//      │ ∂R/∂ϕ  ∂R/∂ψ │   │ K + λ B   Cᵀ │    (neq)
// Gu = │              │ = │              │
//      │ ∂S/∂ϕ  ∂S/∂ψ │   │ C         0  │    (np)
//      └              ┘   └              ┘
//
//      ┌       ┐   ┌   ┐
//      │ ∂R/∂λ │   │ b │    (neq)
// Gλ = │       │ = │   │
//      │ ∂S/∂λ │   │ 0 │    (np)
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

// 1D: Analytical u(x) profile @ λCrit (from Mathematica)
const D1_REF_ALP00: f64 = 3.51383071912516; // λ critical for the α = 0.0 case
const D1_REF_THETA: f64 = 4.79871456103094; // θ critical (for the analytical profile); α = 0.0 case
const D1_REF_ALP02_A: f64 = 4.647906373918411; // 1st λ critical for α = 0.2 (from npt = 500 and tol = 1e-8); nrm=2.3548402404342146
const D1_REF_ALP02_B: f64 = 3.509919925802271; // 2nd λ critical for α = 0.2 (from npt = 500 and tol = 1e-8); nrm=15.440772685670549

// 2D:reference (λCrit, ‖ϕCrit‖∞) values from Bolstad and Keller (6 order scheme, very fine mesh)
const D2_REF_ALP00: f64 = 6.80812442259; // α = 0: first critical point; nrm=1.3916612
const D2_REF_ALP02_A: f64 = 9.13638296666; // α = 0.2: first critical point; nrm=2.8858004
const D2_REF_ALP02_B: f64 = 7.10189894953; // α = 0.2: second critical point; nrm=18.192768

// Calculates the analytical solution at λCrit
fn d1_analytical_profile(x: f64) -> f64 {
    -2.0 * f64::ln(f64::cosh((x - 0.5) * D1_REF_THETA / 2.0) / f64::cosh(D1_REF_THETA / 4.0))
}

const CHECK_JACOBIAN: bool = false;
const SAVE_FIGURE: bool = false;

#[test]
fn test_bratu_1d_spc_auto() -> Result<(), StrError> {
    let spc = true;
    let one_dim = true;
    let auto = AutoStep::Yes;
    for (npt, tol1, tol2, tol3) in [
        (8, 0.00029, 0.0014, 0.00044), //
                                       // (20, 0.00019, 0.00024, 0.0021), //
    ] {
        for alpha in [0.0, 0.2] {
            for lmm in [true, false] {
                for bordering in [true, false] {
                    println!("{}", gen_file_stem(spc, one_dim, npt, alpha, lmm, bordering, auto));
                    run_test(spc, one_dim, lmm, bordering, alpha, npt, auto, tol1, tol2, tol3)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_bratu_1d_fdm_auto() -> Result<(), StrError> {
    let spc = false;
    let one_dim = true;
    let auto = AutoStep::Yes;
    for (npt, tol1, tol2, tol3) in [
        (8, 0.039, 0.061, 0.06), //
                                 // (100, 0.00089, 0.00047, 0.0011), //
    ] {
        for alpha in [0.0, 0.2] {
            for lmm in [true, false] {
                for bordering in [true, false] {
                    println!("{}", gen_file_stem(spc, one_dim, npt, alpha, lmm, bordering, auto));
                    run_test(spc, one_dim, lmm, bordering, alpha, npt, auto, tol1, tol2, tol3)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_bratu_2d_spc_auto() -> Result<(), StrError> {
    let spc = true;
    let one_dim = false;
    let auto = AutoStep::Yes;
    for (npt, tol1, tol2, tol3) in [
        (8, 0.0011, 0.00054, 0.002), //
                                     // (20, 0.0012, 0.00003, 0.0002), //
    ] {
        for alpha in [0.0, 0.2] {
            for lmm in [true, false] {
                for bordering in [true, false] {
                    println!("{}", gen_file_stem(spc, one_dim, npt, alpha, lmm, bordering, auto));
                    run_test(spc, one_dim, lmm, bordering, alpha, npt, auto, tol1, tol2, tol3)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_bratu_2d_fdm_auto() -> Result<(), StrError> {
    let spc = false;
    let one_dim = false;
    let auto = AutoStep::Yes;
    for (npt, tol1, tol2, tol3) in [
        (8, 0.034, 0.083, 0.123), //
                                  // (9, 0.027, 0.062, 0.09),        //
                                  // (20, 0.0052, 0.012, 0.015),     //
                                  // (22, 0.0059, 0.012, 0.013),     //
                                  // (40, 0.0026, 0.032, 0.034),     //
                                  // (41, 0.0020, 0.0052, 0.0034),   //
                                  // (100, 0.00095, 0.0016, 0.0008), //
                                  // (101, 0.0012, 0.0017, 0.0009),  //
    ] {
        for alpha in [0.0, 0.2] {
            for lmm in [true, false] {
                for bordering in [true, false] {
                    println!("{}", gen_file_stem(spc, one_dim, npt, alpha, lmm, bordering, auto));
                    run_test(spc, one_dim, lmm, bordering, alpha, npt, auto, tol1, tol2, tol3)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_bratu_2d_fdm_fixed() -> Result<(), StrError> {
    let spc = false;
    let one_dim = false;
    let lmm = true;
    let bordering = false;
    let auto = AutoStep::No(10.0);
    for alpha in [0.0] {
        for (npt, tol1, tol2, tol3) in [(8, 0.034, 0.0, 0.0)] {
            run_test(spc, one_dim, lmm, bordering, alpha, npt, auto, tol1, tol2, tol3)?;
        }
    }
    Ok(())
}

// Runs the test
fn run_test(
    spc: bool,
    one_dim: bool,
    lmm: bool,
    bordering: bool,
    alpha: f64,
    npt: usize,
    auto: AutoStep,
    alpha0_lam_crit_tol: f64,
    alpha02_1st_lam_crit_tol: f64,
    alpha02_2nd_lam_crit_tol: f64,
) -> Result<(), StrError> {
    // stem
    let stem = gen_file_stem(spc, one_dim, npt, alpha, lmm, bordering, auto);

    // configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    config
        .set_n_cont_failure_max(8)
        // .set_tg_control_atol(0.04)
        // .set_tg_control_rtol(0.04)
        .set_tg_control_atol_and_rtol(0.04)
        .set_record_iterations_residuals(true)
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_debug_predictor(true)
        .set_log_file(&format!("{}.txt", stem))
        .set_bordering(bordering);

    // calculate the coefficient matrix
    let sym = Sym::No;
    let (nu, np, coo) = if one_dim {
        let mut ebcs = EssentialBcs1d::new();
        let nbcs = NaturalBcs1d::new();
        ebcs.set_homogeneous();
        if spc {
            let spectral = Spc1d::new(0.0, 1.0, npt, ebcs, nbcs, -1.0)?;
            let coo = if lmm {
                spectral.get_matrices_lmm(0.0, 0, false).0
            } else {
                spectral.get_matrices_sps(0.0, 0).0
            };
            let (nu, np) = spectral.get_dims_sps();
            (nu, np, coo)
        } else {
            let grid = Grid1d::new_uniform(0.0, 1.0, npt)?;
            let fdm = Fdm1d::new(grid, ebcs, nbcs, -1.0)?;
            let coo = if lmm {
                fdm.get_matrices_lmm(0.0, 0, false, sym).0
            } else {
                fdm.get_matrices_sps(0.0, 0, sym).0
            };
            let (nu, np) = fdm.get_dims_sps();
            (nu, np, coo)
        }
    } else {
        let mut ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        ebcs.set_homogeneous();
        if spc {
            let spectral = Spc2d::new(0.0, 1.0, 0.0, 1.0, npt, npt, ebcs, nbcs, -1.0, -1.0)?;
            let coo = if lmm {
                spectral.get_matrices_lmm(0.0, 0, false).0
            } else {
                spectral.get_matrices_sps(0.0, 0).0
            };
            let (nu, np) = spectral.get_dims_sps();
            (nu, np, coo)
        } else {
            let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, npt, npt)?;
            let fdm = Fdm2d::new(grid, ebcs, nbcs, -1.0, -1.0)?;
            let coo = if lmm {
                fdm.get_matrices_lmm(0.0, 0, false, sym).0
            } else {
                fdm.get_matrices_sps(0.0, 0, sym).0
            };
            let (nu, np) = fdm.get_dims_sps();
            (nu, np, coo)
        }
    };
    let nnz_coo = coo.get_info().2;

    // allocate arguments struct
    let neq = nu + np;
    let nphi = if lmm { neq } else { nu };
    let ndim = if lmm { neq + np } else { nu };
    let mut args = Args {
        alpha,
        bordering,
        npt,
        nphi,
        ndim,
        coo,
    };

    // allocate nonlinear problem
    let mut system = System::new(ndim, calc_gg)?;

    // max number of non-zeros in Gu
    let nnz = if lmm {
        nnz_coo + neq // +neq due to the λ B term
    } else {
        nnz_coo + nu // +nu due to the λ B term
    };

    // set callback functions
    system.set_calc_ggu(Some(nnz), sym, calc_ggu)?;
    system.set_calc_ggl(calc_ggl);

    // define the solver
    let mut solver = Solver::new(&config, system)?;

    // output
    let output = &mut Output::new();
    output.set_record_norm_u(true, Norm::Inf, 0, nphi);
    if one_dim {
        let all_indices: Vec<usize> = (0..nphi).collect();
        output.set_recording(true, &all_indices, &[]);
    }

    // initial state (all zero)
    let mut u = Vector::new(ndim);
    let mut l = 0.0;

    // stop criterion
    let max_nrm_max = if alpha == 0.0 { 15.0 } else { 40.0 };
    let stop = Stop::MaxNormU(max_nrm_max, Norm::Inf, 0, nphi);

    // numerical continuation
    let status = solver.solve(&mut args, &mut u, &mut l, IniDir::Pos, stop, auto, Some(output))?;
    println!("Status: {:?}", status);
    assert_eq!(status, Status::Success);

    // search for the critical point(s)
    println!("Numerical results for α = {} and npt = {}:", alpha, npt);
    let lam_vals = output.get_l_values();
    let nrm_vals = output.get_norm_u_values();
    let (mut ii_valleys, mut ii_peaks, _, _) = find_valleys_and_peaks(lam_vals);
    ii_valleys.retain(|&i| lam_vals[i] > 0.3);
    ii_peaks.retain(|&i| lam_vals[i] > 0.3);
    for i in &ii_peaks {
        println!("Peak   @ ({}, {})", lam_vals[*i], nrm_vals[*i]);
    }
    for i in &ii_valleys {
        println!("Valley @ ({}, {})", lam_vals[*i], nrm_vals[*i]);
    }

    // check the results
    let (ref_alp00, ref_alp02a, ref_alp02b) = if one_dim {
        (D1_REF_ALP00, D1_REF_ALP02_A, D1_REF_ALP02_B)
    } else {
        (D2_REF_ALP00, D2_REF_ALP02_A, D2_REF_ALP02_B)
    };
    if alpha == 0.0 {
        if ii_peaks.len() == 1 {
            let lam_crit = lam_vals[ii_peaks[0]];
            let diff = f64::abs(lam_crit - ref_alp00);
            println!("diff = {}", diff);
            if diff > alpha0_lam_crit_tol {
                println!("❌ ERROR ❌ λCrit = {}, ref = {}, diff = {}", lam_crit, ref_alp00, diff);
            }
        } else {
            println!("WARNING: for alpha = 0.0, one peak must have been found");
        }
    } else if alpha == 0.2 {
        if ii_peaks.len() == 1 && ii_valleys.len() == 1 {
            let lam_crit_a = lam_vals[ii_peaks[0]];
            let lam_crit_b = lam_vals[ii_valleys[0]];
            let diff_a = f64::abs(lam_crit_a - ref_alp02a);
            let diff_b = f64::abs(lam_crit_b - ref_alp02b);
            println!("diff_a = {}, diff_b = {}", diff_a, diff_b);
            if diff_a > alpha02_1st_lam_crit_tol {
                println!(
                    "❌ ERROR ❌ 1st λCrit = {}, ref = {}, diff = {}",
                    lam_crit_a, ref_alp02a, diff_a
                );
            }
            if diff_b > alpha02_2nd_lam_crit_tol {
                println!(
                    "❌ ERROR ❌ 2nd λCrit = {}, ref = {}, diff = {}",
                    lam_crit_b, ref_alp02b, diff_b
                );
            }
        } else {
            println!("WARNING: for alpha = 0.2, one peak and one valley must have been found");
        }
    }
    println!();

    // plot the results
    if SAVE_FIGURE {
        do_plot(
            spc,
            one_dim,
            lmm,
            bordering,
            alpha,
            npt,
            auto,
            &stem,
            &lam_vals,
            &nrm_vals,
            &ii_peaks,
            &ii_valleys,
            &output,
        )?;
    }
    Ok(())
}

struct Args {
    alpha: f64,
    bordering: bool,
    npt: usize,
    nphi: usize,
    ndim: usize,
    coo: CooMatrix,
}

// function to calculate G(u, λ)
fn calc_gg(gg: &mut Vector, l: f64, u: &Vector, args: &mut Args) -> Result<(), StrError> {
    // calculate G := K̄ ̄a  or  G: = M A
    args.coo.mat_vec_mul(gg, 1.0, u).unwrap();
    // update G += λ b (diagonal terms)
    for m in 0..args.nphi {
        let dm = 1.0 + args.alpha * u[m];
        let bm = f64::exp(u[m] / dm);
        gg[m] += l * bm;
    }
    Ok(())
}

// function to calculate Gu = ∂G/∂u (Jacobian matrix)
fn calc_ggu(ggu_or_aa: &mut CooMatrix, l: f64, u: &Vector, args: &mut Args) -> Result<(), StrError> {
    // note that ggu_or_aa may be the pseudo-arclength (larger) matrix
    ggu_or_aa.reset();
    // set Gu := K̄  or Gu := M
    ggu_or_aa.add(1.0, &args.coo).unwrap();
    // add λ B to the Gu
    for m in 0..args.nphi {
        let dm = 1.0 + args.alpha * u[m];
        let bm = f64::exp(u[m] / dm);
        ggu_or_aa.put(m, m, l * bm / (dm * dm)).unwrap();
    }
    // check Jacobian for smaller grids
    if CHECK_JACOBIAN && args.bordering && args.npt <= 21 {
        let ana = ggu_or_aa.as_dense();
        let num = num_jacobian(args.ndim, l, u, 1.0, args, calc_gg).unwrap();
        if args.npt <= 4 {
            println!("ana =\n{:.3}", ana);
            println!("num =\n{:.3}", num);
        }
        mat_approx_eq(&ana, &num, 1e-7);
    }
    Ok(())
}

// function to calculate Gl = ∂G/∂λ
fn calc_ggl(ggl: &mut Vector, _l: f64, u: &Vector, args: &mut Args) -> Result<(), StrError> {
    for m in 0..args.nphi {
        let dm = 1.0 + args.alpha * u[m];
        let bm = f64::exp(u[m] / dm);
        ggl[m] = bm;
    }
    Ok(())
}

fn gen_file_stem(
    spc: bool,
    one_dim: bool,
    npt: usize,
    alpha: f64,
    lmm: bool,
    bordering: bool,
    auto: AutoStep,
) -> String {
    let mut key0 = format!("a{:.1}", alpha);
    key0 = key0.replace('.', "d");
    let key1 = if lmm { "lmm" } else { "sps" };
    let key2 = if bordering { "brd" } else { "full" };
    let key3 = if auto.no() { "fix" } else { "auto" };
    format!(
        "/tmp/russell_nonlin/test_bratu_{}_{}_n{}_{}_{}_{}_{}",
        if spc { "spc" } else { "fdm" },
        if one_dim { "1d" } else { "2d" },
        npt,
        key0,
        key1,
        key2,
        key3
    )
}

fn do_plot<'a>(
    spc: bool,
    one_dim: bool,
    lmm: bool,
    bordering: bool,
    alpha: f64,
    npt: usize,
    auto: AutoStep,
    stem: &str,
    lam_vals: &Vec<f64>,
    nrm_vals: &Vec<f64>,
    ii_peaks: &[usize],
    ii_valleys: &[usize],
    output: &Output<'a, Args>,
) -> Result<(), StrError> {
    // allocate the plot
    let mut plot = Plot::new();

    // set the title
    let mut title = if spc { "SPC".to_string() } else { "FDM".to_string() };
    title += if one_dim { " | 1D" } else { " | 2D" };
    title += &format!(" | n = {}", npt);
    if lmm {
        title += " | LMM";
    }
    if bordering {
        title += " | BRD";
    }
    title += &format!(" | $\\alpha$ = {:.2}", alpha);

    // maximum ‖ϕ‖∞ value
    let max_nrm_max = nrm_vals[find_index_abs_max(&nrm_vals)];

    // reference results
    if alpha == 0.0 {
        let table: HashMap<String, Vec<f64>> = if one_dim {
            read_table(&"data/ref-bratu-1d-shahab-2025.txt", Some(&["lambda", "u_max"]))?
        } else {
            read_table(&"data/ref-bratu-2d-shahab-2025.txt", Some(&["lambda", "u_max"]))?
        };
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
    plot.set_title(&title)
        .set_labels("λ", "‖ϕ‖∞")
        .save(&format!("{}.svg", stem))?;

    // plot stepsizes
    if auto.yes() {
        let stepsizes = &output.get_h_values()[1..];
        let n = stepsizes.len();
        let x = linspace(1.0, n as f64, n);
        let mut curve = Curve::new();
        curve.set_label("stepsize").set_line_style("-").set_marker_style(".");
        curve.draw(&x.as_slice(), &stepsizes);
        let mut plot = Plot::new();
        plot.set_title(&title)
            .set_labels("step number", "stepsize $h$")
            .add(&curve)
            .save(&format!("{}_h.svg", stem))?;
    }

    // profile: draw ϕ along x @ λCrit
    if one_dim && alpha == 0.0 && ii_peaks.len() == 1 {
        let mut curve = Curve::new();
        let mut curve_profile_crit_num = Curve::new();
        let xx_ana = linspace(0.0, 1.0, 201);
        let phi_crit_ana: Vec<_> = xx_ana.iter().map(|&x| d1_analytical_profile(x)).collect();
        curve
            .set_label("Mathematica")
            .set_line_style("-")
            .set_line_color("#1f53d6ff")
            .draw(&xx_ana, &phi_crit_ana);
        let mut xx_num = vec![0.0; npt];
        let mut phi_crit_num = vec![0.0; npt];
        let grid = Grid1d::new_uniform(0.0, 1.0, npt)?;
        grid.for_each_coord(|m, x| {
            xx_num[m] = x;
            if lmm {
                phi_crit_num[m] = output.get_u_values(m)[ii_peaks[0]];
            } else {
                if m > 0 && m < npt - 1 {
                    let iu = m - 1;
                    phi_crit_num[m] = output.get_u_values(iu)[ii_peaks[0]];
                }
            }
        });
        curve_profile_crit_num
            .set_label("Russell")
            .set_line_style("None")
            .set_marker_style(".")
            .set_marker_color("#d8211aff")
            .set_marker_line_color("#d8211aff")
            .draw(&xx_num, &phi_crit_num);
        let mut plot = Plot::new();
        plot.set_title(&title)
            .add(&curve)
            .add(&curve_profile_crit_num)
            .grid_labels_legend("x", "$\\phi_{crit}(x)$")
            .save(&format!("{}_profile.svg", stem))?;
    }
    Ok(())
}
