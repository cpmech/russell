use plotpy::{Curve, Plot};
use russell_lab::{mat_approx_eq, num_jacobian, Vector};
use russell_nonlin::{AutoStep, Config, IniDir, Method, NoArgs, Output, Solver, State, Stop, System};
use russell_pde::FdmLaplacian2d;
use russell_sparse::{CooMatrix, Sym};

const CHECK_JACOBIAN: bool = true;
const SAVE_FIGURE: bool = true;

#[test]
fn test_reaction_diffusion_2d() {
    // The nonlinear problem originates from the FDM discretization of the following equation:
    //
    // ∂²ϕ   ∂²ϕ
    // ——— + ——— + λ exp(ϕ/(1 + α ϕ)) = 0
    // ∂x²   ∂y²
    //
    // on the unit square (1.0 × 1.0) with homogeneous boundary conditions.
    //
    // Below, ϕ is a vector, i.e., ϕ = [ϕ₁, ϕ₂, ..., ϕₙ]ᵀ, where n is the number of grid points.
    // The prescribed values are collected in the vector c = [c₁, c₂, ..., cₘ]ᵀ, where m is the number
    // of boundary points. The Laplacian operator is represented by the matrix K, thus
    // Kϕ is the discretization of the Laplacian operator applied to ϕ(x,y).
    //
    // The boundary conditions are enforced via Lagrange multipliers ψ = [ψ₁, ψ₂, ..., ψₘ]ᵀ,
    // where m is the number of prescribed values (boundary points). The prescribed
    // values are all zero (homogeneous boundary conditions), thus c = [0, 0, ..., 0]ᵀ.
    //
    // The vector of unknowns is expressed by u = [ϕ, ψ]ᵀ and the discretized system is
    // expressed by G = [R, S]ᵀ, where:
    //
    // R = K ϕ + λ b + Eᵀψ = 0
    // S = E ϕ - c = 0
    //
    // with: bₘ = exp(ϕₘ/(1 + α ϕₘ))  (no sum on m)
    //
    // The derivatives are:
    //
    //                  ⎧ bₘ/(1 + α ϕₘ)²  if m = n
    // Bₘₙ := ∂bₘ/∂ϕₙ = ⎨
    //                  ⎩ 0  otherwise
    //
    // ∂R/∂ϕ = K + λ B    ∂R/∂ψ = Eᵀ
    // ∂S/∂ϕ = E          ∂S/∂ψ = 0
    //
    // ∂R/∂λ = b
    // ∂S/∂λ = 0
    //
    // Thus:
    //      ┌              ┐
    //      │ K + λ B   Eᵀ │
    // Gu = │              │
    //      │ E         0  │
    //      └              ┘
    // And:
    //      ┌   ┐
    //      │ b │
    // Gλ = │   │
    //      │ 0 │
    //      └   ┘

    // constants
    const ALPHA: f64 = 0.0;
    const NDIV: usize = 5; // number of divisions along each axis of the FDM grid

    // allocate the Laplacian operator
    let mut fdm = FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, NDIV, NDIV).unwrap();
    fdm.set_homogeneous_boundary_conditions();

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
        // ┌   ┐   ┌       ┐ ┌   ┐   ┌    ┐
        // │ R │   │ K  Eᵀ │ │ ϕ │   │ λb │
        // │   │ = │       │ │   │ + │    │
        // │ S │   │ E  0  │ │ ψ │   │ -c │
        // └   ┘   └       ┘ └   ┘   └    ┘
        //   G         A       u
        aa.mat_vec_mul(gg, 1.0, u).unwrap();
        // update R += λ b
        for m in 0..n_phi {
            let dm = 1.0 + ALPHA * u[m];
            let bm = f64::exp(u[m] / dm);
            gg[m] += l * bm;
        }
        // update S -= c (not really needed since c = 0)
        Ok(())
    };

    // function to calculate Gu = ∂G/∂u (Jacobian matrix)
    let calc_ggu = |ggu: &mut CooMatrix, l: f64, u: &Vector, _args: &mut NoArgs| {
        ggu.reset();
        ggu.add(1.0, &aa).unwrap();
        // add λ B to the K term
        for m in 0..n_phi {
            let dm = 1.0 + ALPHA * u[m];
            let bm = f64::exp(u[m] / dm);
            let bmm = bm / (dm * dm); // no sum on m
            ggu.put(m, m, l * bmm).unwrap();
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

    // check Jacobian matrix
    if CHECK_JACOBIAN {
        let mut u0 = Vector::new(ndim);
        let l0 = 0.01;
        u0.fill(0.01);
        let mut ggu = CooMatrix::new(ndim, ndim, nnz, sym).unwrap();
        calc_ggu(&mut ggu, l0, &u0, &mut 0).unwrap();
        let ana = ggu.as_dense();
        let num = num_jacobian(ndim, l0, &u0, 1.0, &mut 0, calc_gg).unwrap();
        if NDIV <= 3 {
            println!("ana =\n{:.3}", ana);
            println!("num =\n{:.3}", num);
        }
        mat_approx_eq(&ana, &num, 1e-9);
    }

    // configuration
    let mut config = Config::new(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_debug_predictor(true)
        .set_bordering(false);

    // define the solver
    let mut solver = Solver::new(config, system).unwrap();

    // output
    let out = &mut Output::new();
    out.set_record_norm_u(true);

    // initial state (all zero)
    let mut state = State::new(ndim);

    // numerical continuation
    let status = solver
        .solve(
            &mut 0,
            &mut state,
            IniDir::Pos,
            Stop::MaxNormU(100.0),
            AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    println!("Status: {:?}", status);

    // plot the results
    if SAVE_FIGURE {
        let mut curve = Curve::new();
        curve.draw(out.get_l_values(), out.get_norm_u_values());
        let mut plot = Plot::new();
        plot.add(&curve)
            .grid_and_labels("λ", "‖u‖₂")
            .save(&format!("/tmp/russell_nonlin/test_reaction_diffusion_2d.svg"))
            .unwrap();
    }
}
