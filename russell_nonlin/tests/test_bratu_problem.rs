use plotpy::{linspace, Curve, Plot, Text};
use russell_lab::{
    find_index_abs_max, find_valleys_and_peaks, mat_approx_eq, num_jacobian, read_table, InterpChebyshev, MinSolver,
};
use russell_lab::{Norm, Vector};
use russell_nonlin::{Config, DeltaLambda, IniDir, Method, Output, Solver, Status, Stop, StrError, System};
use russell_pde::{
    EssentialBcs1d, EssentialBcs2d, Fdm1d, Fdm2d, Grid1d, Grid2d, NaturalBcs1d, NaturalBcs2d, Spc1d, Spc2d,
};
use russell_sparse::{CooMatrix, Genie};
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

struct Args {
    alpha: f64,
    bordering: bool,
    npt: usize,
    nphi: usize,
    ndim: usize,
    coo: CooMatrix,
}

// ----------------------------- Problem definition and derivatives ----------------------------

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

// function to calculate the Jacobian; Gu = ∂G/∂u and Gl = ∂G/∂λ
fn calc_jac(ggu: &mut CooMatrix, ggl: &mut Vector, l: f64, u: &Vector, args: &mut Args) -> Result<(), StrError> {
    // note that ggu_or_aa may be the pseudo-arclength (larger) matrix
    ggu.reset();
    // set Gu := K̄  or Gu := M
    ggu.add(1.0, &args.coo).unwrap();
    // add λ B to the Gu
    for m in 0..args.nphi {
        let dm = 1.0 + args.alpha * u[m];
        let bm = f64::exp(u[m] / dm);
        ggu.put(m, m, l * bm / (dm * dm)).unwrap();
        ggl[m] = bm;
    }
    // check Jacobian for smaller grids
    if CHECK_JACOBIAN && args.bordering && args.npt <= 21 {
        let ana = ggu.as_dense();
        let num = num_jacobian(args.ndim, l, u, 1.0, args, calc_gg).unwrap();
        if args.npt <= 4 {
            println!("ana =\n{:.3}", ana);
            println!("num =\n{:.3}", num);
        }
        mat_approx_eq(&ana, &num, 1e-5);
    }
    Ok(())
}

// ----------------------------- Constants ----------------------------

// 1D: Analytical u(x) profile @ λCrit (from Mathematica)
const D1_REF_ALP00: f64 = 3.51383071912516; // λ critical for the α = 0.0 case
const D1_REF_ALP02_A: f64 = 4.647906373918411; // 1st λ critical for α = 0.2 (from npt = 500 and tol = 1e-8); nrm=2.3548402404342146
const D1_REF_ALP02_B: f64 = 3.509919925802271; // 2nd λ critical for α = 0.2 (from npt = 500 and tol = 1e-8); nrm=15.440772685670549

// 2D:reference (λCrit, ‖ϕCrit‖∞) values from Bolstad and Keller (6 order scheme, very fine mesh)
const D2_REF_ALP00: f64 = 6.80812442259; // α = 0: first critical point; nrm=1.3916612
const D2_REF_ALP02_A: f64 = 9.13638296666; // α = 0.2: first critical point; nrm=2.8858004
const D2_REF_ALP02_B: f64 = 7.10189894953; // α = 0.2: second critical point; nrm=18.192768

const TG_CONTROL_TOL: f64 = 0.1;
const DDL_INI: f64 = 0.5;

const CHECK_JACOBIAN: bool = false;
const SAVE_FIGURE: bool = false;
const PLOT_STEPSIZES: bool = false;
const PLOT_OBJECTIVE_FUNCTION: bool = false;

const DELTA_INDEX: usize = 5; // delta-index to bracket critical points

// ----------------------------- Test combinations ----------------------------

/// Holds all test combinations
#[derive(Debug, Clone)]
struct Combo {
    pub spc: bool,
    pub one_dim: bool,
    pub npt: usize,
    pub alpha: f64,
    pub lmm: bool,
    pub bordering: bool,
    pub genie: Genie,
    pub symmetric: bool,
    pub ddl: DeltaLambda,
}

impl Combo {
    /// Generates the file stem for the given combination of parameters
    pub fn stem(&self) -> String {
        let mut key0 = format!("a{:.1}", self.alpha);
        key0 = key0.replace('.', "d");
        let key1 = if self.lmm { "lmm" } else { "sps" };
        let key2 = if self.bordering { "brd" } else { "full" };
        let key3 = if self.ddl.is_auto() { "auto" } else { "fix" };
        let key4 = format!("sym-{:?}", self.genie.get_sym(self.symmetric)).to_lowercase();
        format!(
            "/tmp/russell_nonlin/test_bratu_{}_{}_n{}_{}_{}_{}_{}_{}_{}",
            if self.spc { "spc" } else { "fdm" },
            if self.one_dim { "1d" } else { "2d" },
            self.npt,
            key0,
            key1,
            key2,
            key3,
            self.genie.to_string(),
            key4
        )
    }

    /// Generates the plot title for the given combination of parameters
    pub fn title(&self, tex: bool) -> String {
        let mut sym = format!("{:?}", self.genie.get_sym(self.symmetric)).to_uppercase();
        if sym == "NO" {
            sym = "NO-SYM".to_string();
        }
        let alp = if tex {
            format!("$\\alpha = {:.2}$", self.alpha)
        } else {
            format!("alpha = {:.2}", self.alpha)
        };
        format!(
            "{} | {} | {} | N = {} | {} | {} | {} | {} | {}",
            if self.spc { "SPC" } else { "FDM" },
            if self.one_dim { "1D" } else { "2D" },
            alp,
            self.npt,
            if self.lmm { "LMM" } else { "SPS" },
            if self.bordering { "BRD" } else { "FULL" },
            if self.ddl.is_auto() { "AUTO" } else { "FIX" },
            self.genie.to_string().to_uppercase(),
            sym
        )
    }
}

// ----------------------------- Test cases ----------------------------

#[test]
fn test_bratu_1d_spc_auto_step() -> Result<(), StrError> {
    let spc = true;
    let one_dim = true;
    let ddl = DeltaLambda::auto(DDL_INI);
    let genie = Genie::Umfpack;
    let symmetric = false;
    for (npt, tol1, tol2, tol3) in [
        (8, 0.001, 0.0009, 0.0005),      //
        (20, 0.00047, 0.00033, 0.00003), //
    ] {
        for alpha in [0.0, 0.2] {
            for lmm in [true, false] {
                for bordering in [true, false] {
                    let combo = Combo {
                        spc,
                        one_dim,
                        npt,
                        alpha,
                        lmm,
                        bordering,
                        genie,
                        symmetric,
                        ddl: ddl.clone(),
                    };
                    run_test(combo, tol1, tol2, tol3)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_bratu_1d_fdm_auto_step() -> Result<(), StrError> {
    let spc = false;
    let one_dim = true;
    let ddl = DeltaLambda::auto(DDL_INI);
    let genie = Genie::Umfpack;
    for (npt, tol1, tol2, tol3) in [
        (8, 0.0381, 0.06, 0.06), //
        (17, 0.00789, 0.012, 0.012), //
                                 // (100, 0.00024, 0.00029, 0.00032), //
    ] {
        for alpha in [0.0, 0.2] {
            for lmm in [true, false] {
                for bordering in [true, false] {
                    let flags = if bordering { vec![true, false] } else { vec![false] }; // symmetric only if bordering
                    for symmetric in flags {
                        let combo = Combo {
                            spc,
                            one_dim,
                            npt,
                            alpha,
                            lmm,
                            bordering,
                            genie,
                            symmetric,
                            ddl: ddl.clone(),
                        };
                        run_test(combo, tol1, tol2, tol3)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_bratu_2d_spc_auto_step() -> Result<(), StrError> {
    let spc = true;
    let one_dim = false;
    let ddl = DeltaLambda::auto(DDL_INI);
    let genie = Genie::Umfpack;
    let symmetric = false;
    for (npt, tol1, tol2, tol3) in [
        (8, 0.00256, 0.00029, 0.002), //
                                      // (10, 0.00073, 0.000034, 0.000024), //
                                      // (11, 0.00073, 0.000034, 0.000024), //
    ] {
        for alpha in [0.0, 0.2] {
            for lmm in [false] {
                for bordering in [true, false] {
                    let combo = Combo {
                        spc,
                        one_dim,
                        npt,
                        alpha,
                        lmm,
                        bordering,
                        genie,
                        symmetric,
                        ddl: ddl.clone(),
                    };
                    run_test(combo, tol1, tol2, tol3)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_bratu_2d_fdm_auto_step() -> Result<(), StrError> {
    let spc = false;
    let one_dim = false;
    let ddl = DeltaLambda::auto(DDL_INI);
    let genie = Genie::Umfpack;
    for (npt, tol1, tol2, tol3) in [
        (8, 0.034, 0.082, 0.123), //
                                  // (9, 0.0253, 0.062, 0.092), //
                                  // (20, 0.0043, 0.011, 0.016), //
                                  // (22, 0.00355, 0.0089, 0.013), //
                                  // (40, 0.0011, 0.0032, 0.0034), //
    ] {
        for alpha in [0.0, 0.2] {
            for lmm in [false] {
                for bordering in [true, false] {
                    let flags = if bordering { vec![true, false] } else { vec![false] }; // symmetric only if bordering
                    for symmetric in flags {
                        let combo = Combo {
                            spc,
                            one_dim,
                            npt,
                            alpha,
                            lmm,
                            bordering,
                            genie,
                            symmetric,
                            ddl: ddl.clone(),
                        };
                        run_test(combo, tol1, tol2, tol3)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_bratu_2d_fdm_fix_step() -> Result<(), StrError> {
    let spc = false;
    let one_dim = false;
    let lmm = true;
    let bordering = false;
    let ddl = DeltaLambda::constant(1.0);
    let genie = Genie::Umfpack;
    let symmetric = false;
    for alpha in [0.0] {
        for (npt, tol1, tol2, tol3) in [(8, 0.23, 0.0, 0.0)] {
            let combo = Combo {
                spc,
                one_dim,
                npt,
                alpha,
                lmm,
                bordering,
                genie,
                symmetric,
                ddl: ddl.clone(),
            };
            run_test(combo, tol1, tol2, tol3)?;
        }
    }
    Ok(())
}

// Runs the test
fn run_test(
    combo: Combo,
    alpha0_lam_crit_tol: f64,
    alpha02_1st_lam_crit_tol: f64,
    alpha02_2nd_lam_crit_tol: f64,
) -> Result<(), StrError> {
    // Filename/path stem
    let stem = combo.stem();
    let caption = combo.title(false);
    println!("{}", "=".repeat(caption.len()));
    println!("{}", caption);

    // Configuration
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    config
        .set_n_cont_failure_max(8)
        .set_tg_control_tol(TG_CONTROL_TOL)
        .set_record_iterations_residuals(true)
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_debug_predictor(true)
        .set_log_file(&format!("{}.txt", stem))
        .set_bordering(combo.bordering)
        .set_genie(combo.genie);

    // Calculate the coefficient matrix
    let sym = combo.genie.get_sym(combo.symmetric);
    let (nu, np, coo) = if combo.one_dim {
        let mut ebcs = EssentialBcs1d::new();
        let nbcs = NaturalBcs1d::new();
        ebcs.set_homogeneous();
        if combo.spc {
            let spectral = Spc1d::new(0.0, 1.0, combo.npt, ebcs, nbcs, -1.0)?;
            let coo = if combo.lmm {
                spectral.get_matrices_lmm(0.0, 0, false).0
            } else {
                spectral.get_matrices_sps(0.0, 0).0
            };
            let (nu, np) = spectral.get_dims_sps();
            (nu, np, coo)
        } else {
            let grid = Grid1d::new_uniform(0.0, 1.0, combo.npt)?;
            let fdm = Fdm1d::new(grid, ebcs, nbcs, -1.0)?;
            let coo = if combo.lmm {
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
        if combo.spc {
            let spectral = Spc2d::new(0.0, 1.0, 0.0, 1.0, combo.npt, combo.npt, ebcs, nbcs, -1.0, -1.0)?;
            let coo = if combo.lmm {
                spectral.get_matrices_lmm(0.0, 0, false).0
            } else {
                spectral.get_matrices_sps(0.0, 0).0
            };
            let (nu, np) = spectral.get_dims_sps();
            (nu, np, coo)
        } else {
            let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, combo.npt, combo.npt)?;
            let fdm = Fdm2d::new(grid, ebcs, nbcs, -1.0, -1.0)?;
            let coo = if combo.lmm {
                fdm.get_matrices_lmm(0.0, 0, false, sym).0
            } else {
                fdm.get_matrices_sps(0.0, 0, sym).0
            };
            let (nu, np) = fdm.get_dims_sps();
            (nu, np, coo)
        }
    };
    let nnz_coo = coo.get_info().2;

    // Allocate arguments struct
    let neq = nu + np;
    let nphi = if combo.lmm { neq } else { nu };
    let ndim = if combo.lmm { neq + np } else { nu };
    let mut args = Args {
        alpha: combo.alpha,
        bordering: combo.bordering,
        npt: combo.npt,
        nphi,
        ndim,
        coo,
    };

    // Max number of non-zeros in Gu
    let nnz = if combo.lmm {
        nnz_coo + neq // +neq due to the λ B term
    } else {
        nnz_coo + nu // +nu due to the λ B term
    };

    // Allocate nonlinear problem
    let system = System::new(ndim, Some(nnz), sym, calc_gg, calc_jac)?;

    // Define the solver
    let mut solver = Solver::new(&config, system)?;

    // Output
    let output = &mut Output::new();
    output.set_record_norm_u(true, Norm::Inf, 0, nphi);
    if combo.one_dim {
        let all_indices: Vec<usize> = (0..nphi).collect();
        output.set_recording(true, &all_indices, &[]);
    }

    // Initial state (all zero)
    let mut u = Vector::new(ndim);
    let mut l = 0.0;

    // Stop criterion
    let max_nrm_max = if combo.alpha == 0.0 { 15.0 } else { 40.0 };
    let stop = Stop::MaxNormU(max_nrm_max, Norm::Inf, 0, nphi);

    // ----------------- numerical solution ----------------

    // Perform the numerical continuation
    let status = solver.solve(&mut args, &mut u, &mut l, IniDir::Pos, stop, &combo.ddl, Some(output))?;
    println!("Status: {:?}", status);
    assert_eq!(status, Status::Success);

    // numerical results
    let lam_vals = output.get_l_values();
    let nrm_vals = output.get_norm_u_values();

    // ----------------- search for critical points ----------------

    let (mut ii_valleys, mut ii_peaks, _, _) = find_valleys_and_peaks(lam_vals);
    ii_valleys.retain(|&i| lam_vals[i] > 0.3);
    ii_peaks.retain(|&i| lam_vals[i] > 0.3);
    for i in &ii_peaks {
        println!("Peak: ‖u‖∞ = {}, λ = {}", nrm_vals[*i], lam_vals[*i]);
    }
    for i in &ii_valleys {
        println!("Valley: ‖u‖∞ = {}, λ = {}", nrm_vals[*i], lam_vals[*i]);
    }
    if combo.alpha == 0.0 {
        if ii_peaks.len() != 1 {
            return Err("for alpha = 0.0, one peak must have been found");
        }
    } else if combo.alpha == 0.2 {
        if !(ii_peaks.len() == 1 && ii_valleys.len() == 1) {
            return Err("for alpha = 0.2, one peak and one valley must have been found");
        }
    } else {
        return Err("invalid alpha value");
    }

    // ------------- polynomial fitting and optimization ----------------

    // Fit a polynomial (using Chebyshev interpolation) to all the numerical (‖u‖∞, λ) results.
    // Then, we can use the interpolant to find the critical points with higher accuracy (e.g., using Brent's method).
    // The interpolant is λ = f(‖u‖∞)
    let npoint = lam_vals.len();
    let last_u_max = nrm_vals[npoint - 1];
    let degree_max = npoint - 1;
    let mut interp = InterpChebyshev::new(degree_max, 0.0, last_u_max).unwrap();
    interp.set_gen_data(&nrm_vals, &lam_vals)?; // (U, L) data

    // Find the critical points with higher accuracy
    let ((nrm_crit_1, lam_crit_1), (nrm_crit_2, lam_crit_2)) = if combo.alpha == 0.0 {
        // For α = 0.0, we have a single critical point (peak)
        let (u_crit, l_crit) = find_critical_point(&nrm_vals, &lam_vals, ii_peaks[0], &interp, true)?;
        println!("Critical point: ‖u‖∞ = {}, λ = {}", u_crit, l_crit);
        ((u_crit, l_crit), (0.0, 0.0)) // dummy values for the second critical point
    } else {
        // For α = 0.2, we have two critical points (one peak and one valley)
        let (u_crit1, l_crit1) = find_critical_point(&nrm_vals, &lam_vals, ii_peaks[0], &interp, true)?;
        let (u_crit2, l_crit2) = find_critical_point(&nrm_vals, &lam_vals, ii_valleys[0], &interp, false)?;
        println!("1st critical point: ‖u‖∞ = {}, λ = {}", u_crit1, l_crit1);
        println!("2nd critical point: ‖u‖∞ = {}, λ = {}", u_crit2, l_crit2);
        ((u_crit1, l_crit1), (u_crit2, l_crit2)) // dummy values for the critical points; we won't use them in the checks below
    };

    // ----------------- check the results ----------------

    // reference critical points
    let (ref_alp00, ref_alp02a, ref_alp02b) = if combo.one_dim {
        (D1_REF_ALP00, D1_REF_ALP02_A, D1_REF_ALP02_B)
    } else {
        (D2_REF_ALP00, D2_REF_ALP02_A, D2_REF_ALP02_B)
    };

    // check the critical points
    if combo.alpha == 0.0 {
        let err_lambda = f64::abs(lam_crit_1 - ref_alp00);
        println!("err_lambda = {}", err_lambda);
        if err_lambda > alpha0_lam_crit_tol {
            panic!("❌ λ* = {} ({}) err = {} ❌", lam_crit_1, ref_alp00, err_lambda);
        }
    } else if combo.alpha == 0.2 {
        let err_lambda_1 = f64::abs(lam_crit_1 - ref_alp02a);
        let err_lambda_2 = f64::abs(lam_crit_2 - ref_alp02b);
        println!("err_lambda_1 = {}", err_lambda_1);
        println!("err_lambda_2 = {}", err_lambda_2);
        if err_lambda_1 > alpha02_1st_lam_crit_tol {
            panic!("❌ 1st λ* = {} ({}) err = {} ❌", lam_crit_1, ref_alp02a, err_lambda_1);
        }
        if err_lambda_2 > alpha02_2nd_lam_crit_tol {
            panic!("❌ 2nd λ* = {} ({}) err = {} ❌", lam_crit_2, ref_alp02b, err_lambda_2);
        }
    }
    println!();

    // -------------------- plot the results --------------------
    if SAVE_FIGURE {
        do_plot(
            &combo, &lam_vals, &nrm_vals, lam_crit_1, nrm_crit_1, lam_crit_2, nrm_crit_2, &output, &interp,
        )?;
    }
    Ok(())
}

/// Generates the plot
fn do_plot<'a>(
    combo: &Combo,
    lam_vals: &Vec<f64>,
    nrm_vals: &Vec<f64>,
    lam_crit_1: f64,
    nrm_crit_1: f64,
    lam_crit_2: f64,
    nrm_crit_2: f64,
    output: &Output<'a, Args>,
    interp: &InterpChebyshev,
) -> Result<(), StrError> {
    // allocate the plot
    let mut plot = Plot::new();

    // set the filename stem and title
    let stem = combo.stem();
    let title = combo.title(true);

    // maximum ‖ϕ‖∞ value
    let max_nrm_max = nrm_vals[find_index_abs_max(&nrm_vals)];

    // reference results
    if combo.alpha == 0.0 {
        // load reference data
        let reference_data: HashMap<String, Vec<f64>> = if combo.one_dim {
            read_table(&"data/ref-bratu-1d-shahab-2025.txt", Some(&["lambda", "u_max"]))?
        } else {
            read_table(&"data/ref-bratu-2d-shahab-2025.txt", Some(&["lambda", "u_max"]))?
        };

        // draw reference curve
        let mut n_ref = 0;
        for u_max in &reference_data["u_max"] {
            if *u_max > max_nrm_max {
                break;
            }
            n_ref += 1;
        }
        if n_ref + 5 < reference_data["u_max"].len() {
            n_ref += 5; // add a few more points for better visualization
        }
        let xx_ref_pts = &reference_data["u_max"].as_slice()[..n_ref];
        let yy_ref_pts = &reference_data["lambda"].as_slice()[..n_ref];
        let mut curve_ref_pts = Curve::new();
        curve_ref_pts
            .set_label("reference")
            .set_line_color("#009500")
            .draw(&xx_ref_pts, &yy_ref_pts);
        plot.add(&curve_ref_pts);
    }

    // numerical results: interpolated curve
    let mut curve_interp = Curve::new();
    let xx_interp = Vector::linspace(0.0, max_nrm_max, 100).unwrap();
    let yy_interp = xx_interp.get_mapped(|x| interp.eval(x).unwrap());
    curve_interp
        .set_label("this code")
        .set_line_style("-")
        .set_line_color("#ba6ed3")
        .draw(xx_interp.as_data(), yy_interp.as_data());
    plot.add(&curve_interp);

    // numerical results: points
    let mut curve_points = Curve::new();
    curve_points
        .set_line_style("None")
        .set_marker_size(4.0)
        .set_marker_style(".")
        .set_line_color("#79158d")
        .draw(nrm_vals, lam_vals);
    plot.add(&curve_points);

    // annotations
    let mut annotations = Text::new();
    annotations
        .set_bbox(true)
        .set_bbox_facecolor("white")
        .set_bbox_edgecolor("None")
        .set_bbox_style("round,pad=0.3");
    let last = nrm_vals.len() - 1;

    // First critical point
    plot.set_horiz_line(lam_crit_1, "#4f4f4f", "--", 1.0);
    plot.set_vert_line(nrm_crit_1, "#4f4f4f", "--", 1.0);
    annotations
        .set_rotation(0.0)
        .set_align_vertical("center")
        .set_align_horizontal("right")
        .draw(nrm_vals[last], lam_crit_1, &format!("{:.9}", nrm_crit_1));
    annotations
        .set_rotation(90.0)
        .set_align_vertical("bottom")
        .set_align_horizontal("center")
        .draw(nrm_crit_1, 0.0, &format!("{:.9}", lam_crit_1));

    // Second critical point
    if combo.alpha == 0.2 {
        plot.set_horiz_line(lam_crit_2, "#4f4f4f", "--", 1.0);
        plot.set_vert_line(nrm_crit_2, "#4f4f4f", "--", 1.0);
        annotations
            .set_rotation(0.0)
            .set_align_vertical("center")
            .set_align_horizontal("right")
            .draw(nrm_vals[last], lam_crit_2, &format!("{:.9}", nrm_crit_2));
        annotations
            .set_rotation(90.0)
            .set_align_vertical("bottom")
            .set_align_horizontal("center")
            .draw(nrm_crit_2, 0.0, &format!("{:.9}", lam_crit_2));
    }
    plot.add(&annotations);

    // Save the plot
    plot.extra(&format!("plt.title(r'{}',fontsize=10)\n", title))
        .legend()
        .set_labels("‖ϕ‖∞", "λ")
        .save(&format!("{}.svg", stem))?;

    // Plot stepsizes
    if combo.ddl.is_auto() && PLOT_STEPSIZES {
        let stepsizes = &output.get_h_values()[1..];
        let n = stepsizes.len();
        let x = linspace(1.0, n as f64, n);
        let mut curve = Curve::new();
        curve.set_label("stepsize").set_line_style("-").set_marker_style(".");
        curve.draw(&x.as_slice(), &stepsizes);
        let mut plot = Plot::new();
        plot.extra(&format!("plt.title(r'{}',fontsize=10)\n", title))
            .set_labels("step number", "stepsize $h$")
            .add(&curve)
            .save(&format!("{}_h.svg", stem))?;
    }

    // Done
    Ok(())
}

/// Finds the critical point (peak or valley) with higher accuracy using the interpolant and Brent's method.
fn find_critical_point(
    xx: &[f64],
    yy: &[f64],
    index_crit: usize,
    interp: &InterpChebyshev,
    maximization: bool,
) -> Result<(f64, f64), StrError> {
    // Check that we have enough points around the critical point
    let npoint = xx.len();
    if index_crit <= DELTA_INDEX || index_crit >= npoint - DELTA_INDEX {
        println!("npoint = {}, index_crit = {}", npoint, index_crit);
        return Err("There aren't enough points around the critical point to perform the optimization");
    }

    // Define the objective function
    // If maximization, the objective function is a shifted/reversed version of f(x)
    let y_crit = yy[index_crit];
    let objective = |x: f64, _args: &mut u8| -> Result<f64, StrError> {
        let y = interp.eval(x)?;
        let yo = if maximization { y_crit - y } else { y };
        Ok(yo)
    };

    // Solve the minimization problem
    let prm = &mut 0;
    let xa = xx[index_crit - DELTA_INDEX];
    let xb = xx[index_crit + DELTA_INDEX];
    let min_solver = MinSolver::new();
    let (xo, _) = min_solver.brent(xa, xb, prm, objective)?;

    // Draw the objective function
    if PLOT_OBJECTIVE_FUNCTION {
        let mut curve = Curve::new();
        let xx_pts = Vector::linspace(xa, xb, 20)?;
        let yy_pts = xx_pts.get_mapped(|u| objective(u, prm).unwrap());
        curve.draw(xx_pts.as_data(), yy_pts.as_data());
        let mut plot = Plot::new();
        plot.add(&curve)
            .grid_and_labels("x", "objective")
            .save("/tmp/russell_nonlin/temp.svg")?;
    }

    // Done
    Ok((xo, interp.eval(xo)?))
}
