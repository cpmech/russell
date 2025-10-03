use plotpy::{linspace, Curve, Plot};
use russell_lab::{approx_eq, Vector};
use russell_pde::{FdmLaplacian1d, Side};
use russell_sparse::{Genie, LinSolver};

const SAVE_FIGURE: bool = false;

#[test]
fn test_laplace1d_3() {
    // This problem simulates the heat conduction-confection of 1D rod.
    //
    // The rod has a length of lx = 1.0 m and the conductivity coefficient
    // is 1.0 W/m/°C. The surrounding environment has a temperature
    // of 0°C and the convection coefficient is 1 W/m/°C. The left-hand
    // side of the rod is kept at a constant temperature of 2°C, while
    // the right-hand side has a flux input of 3 W. The rod has also a heat
    // (source) generation equal to x². The goal is to find the temperature
    // distribution along the rod.
    //
    // The Model is:
    //
    //      ∂²ϕ
    // - kx ——— + (ϕ - ϕ∞) β = x²
    //      ∂x²
    //
    // where ϕ∞ = 0°C is the temperature of the surrounding environment and
    // β = 1 W/m/°C is the convection coefficient. The essential boundary
    // condition is:
    //
    // ϕ(0) = ϕₐ = 2°C
    //
    // The closed-form solution is:
    //
    //        sinh(x)
    // ϕ(x) = ——————— + x² + 2
    //        cosh(1)

    // constants
    let lx = 1.0;
    let kx = 1.0;
    let beta = 1.0;
    let phi_a = 2.0;
    let phi_inf = 0.0;
    let flux = 3.0;

    // allocate the Laplacian operator
    // (note that we have to use negative kx)
    let nx = 21;
    let mut fdm = FdmLaplacian1d::new(-kx, 0.0, lx, nx, Some(beta)).unwrap();

    // set essential boundary conditions
    fdm.set_essential_boundary_condition(Side::Xmin, |_| phi_a);

    // compute the modified coefficient matrix and the correction matrix
    let (aa, cc) = fdm.mod_coefficient_matrix().unwrap();

    // allocate the left- and right-hand side vectors
    let dim = fdm.dim();
    let mut phi = Vector::new(dim);
    let mut rhs = Vector::new(dim);

    // set the 'prescribed' part of the left-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        phi[i] = value; // u2 := ϕ₀
    });

    // initialize the right-hand side vector with the correction
    cc.mat_vec_mul(&mut rhs, -1.0, &phi).unwrap(); // f1 := -K12⋅u2

    // set the right-hand side vector with the convection, source, and flux terms
    fdm.loop_over_grid_points(|i, x| {
        rhs[i] += phi_inf * beta + x * x;
    });

    // set the right-hand side with the flux boundary condition
    // (the flux is calculated using central differences)
    //
    // The FDM stencil uses the "molecule" {α, β, β} such that:
    //
    // α ϕᵢ + β ϕᵢ₋₁ + β ϕᵢ₊₁ = sᵢ
    //
    // where α = -2kx/Δx² and β = kx/Δx².
    //
    // The central difference formula for the flux is:
    //
    // ∂ϕ │    ϕ₁ - ϕ₋₁
    // —— │  ≈ ———————— := flux
    // ∂x │0     2 Δx
    //
    // Thus, ϕ₋₁ = ϕ₁ - 2 Δx flux
    //
    // Substituting ϕ₋₁ in the FDM stencil gives (for i = 0):
    //
    // α ϕ₀ + β (ϕ₁ - 2 Δx flux) + β ϕ₁ = s₀
    // or
    // α ϕ₀ + 2 β ϕ₁ = s₀ + 2 β Δx flux
    //                       extra_term
    //
    // Substituting the value of β for the extra term gives:
    //
    // extra_term = 2 kx flux / Δx
    //
    // Similar expression can be derived for the right-hand side of the rod
    let dx = fdm.grid_spacing();
    rhs[nx - 1] += 2.0 * kx * flux / dx;

    // set the 'prescribed' part of the right-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        rhs[i] = value; // f2 := ϕ₀
    });

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&aa, None).unwrap();
    solver.actual.solve(&mut phi, &rhs, false).unwrap();

    // results
    let d = f64::cosh(1.0);
    let ana_phi = |x| f64::sinh(x) / d + x * x + 2.0;
    fdm.loop_over_grid_points(|i, x| {
        println!("{}: ϕ = {} ({})", i, phi[i], ana_phi(x));
        approx_eq(phi[i], ana_phi(x), 0.000282);
    });

    // plot
    if SAVE_FIGURE {
        let mut curve_ana = Curve::new();
        let mut curve_num = Curve::new();
        curve_ana.set_label("Analytical");
        curve_num
            .set_label("Numerical")
            .set_marker_style("o")
            .set_line_style("None");
        let xx_ana = linspace(0.0, lx, 101);
        let uu_ana = xx_ana.iter().map(|&x| ana_phi(x)).collect::<Vec<_>>();
        let mut xx_num = vec![0.0; nx];
        fdm.loop_over_grid_points(|i, x| {
            xx_num[i] = x;
        });
        curve_ana.draw(&xx_ana, &uu_ana);
        curve_num.draw(&xx_num, &phi.as_data());
        let mut plot = Plot::new();
        plot.add(&curve_ana)
            .add(&curve_num)
            .grid_labels_legend("$x$", "$\\phi$")
            .save("/tmp/russell_pde/test_laplace1d_3.svg")
            .unwrap();
    }
}
