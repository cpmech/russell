use plotpy::{linspace, Curve, Plot};
use russell_lab::approx_eq;
use russell_pde::{EssentialBcs1d, Fdm1d, Grid1d, Side};
use russell_sparse::{Genie, LinSolver, Sym};

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
    let nx = 21;
    let lx = 1.0;
    let kx = 1.0;
    let beta = 1.0;
    let phi_a = 2.0;
    let phi_inf = 0.0;
    let flux = 3.0;

    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, lx, nx).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set(&grid, Side::Xmin, |_| phi_a);

    // allocate the Laplacian operator
    // (note that we have to use negative kx)
    let fdm = Fdm1d::new(grid, ebcs, -kx).unwrap();

    // assemble the coefficient matrix and the lhs and rhs vectors
    let nu = fdm.get_dims_sps().0;
    let extra_nnz = nu; // diagonal entries due to ϕ β
    let (mut kk_bar, kk_check) = fdm.get_matrices_sps(extra_nnz, Sym::No);
    let (mut a_bar, a_check, mut f_bar) = fdm.get_vectors_sps(|x| x * x + phi_inf * beta); // (- ϕ∞ β) goes to the rhs
    let kk_check = kk_check.unwrap();

    // add the diagonal entries due to ϕ β
    for i in 0..nu {
        kk_bar.put(i, i, beta).unwrap();
    }

    // update the right-hand side with the prescribed values
    kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check).unwrap(); // f̄ -= Ǩ ǎ

    // add the flux term to the right-hand side vector
    //
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
    let dx = fdm.get_grid().get_dx().unwrap();
    let m_right = fdm.get_grid().node_xmax(); // global index of the rightmost node (@ xmax)
    let i_right = fdm.get_equations().iu(m_right); // local index of the rightmost node
    f_bar[i_right] += 2.0 * kx * flux / dx;

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk_bar, None).unwrap();
    solver.actual.solve(&mut a_bar, &f_bar, false).unwrap();

    // results
    let a = fdm.get_joined_vector_sps(&a_bar, &a_check);

    // analytical solution
    let d = f64::cosh(1.0);
    let analytical = |x| f64::sinh(x) / d + x * x + 2.0;
    fdm.for_each_coord(|m, x| {
        println!("{}: ϕ = {} ({})", m, a[m], analytical(x));
        approx_eq(a[m], analytical(x), 0.000282);
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
        let uu_ana = xx_ana.iter().map(|&x| analytical(x)).collect::<Vec<_>>();
        let mut xx_num = vec![0.0; nx];
        fdm.for_each_coord(|i, x| {
            xx_num[i] = x;
        });
        let uu_num = a.as_data();
        curve_ana.draw(&xx_ana, &uu_ana);
        curve_num.draw(&xx_num, &uu_num);
        let mut plot = Plot::new();
        plot.add(&curve_ana)
            .add(&curve_num)
            .grid_labels_legend("$x$", "$\\phi$")
            .save("/tmp/russell_pde/test_laplace1d_3.svg")
            .unwrap();
    }
}
