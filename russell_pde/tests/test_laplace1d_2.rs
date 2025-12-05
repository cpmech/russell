use plotpy::{linspace, Curve, Plot};
use russell_lab::{approx_eq, math::PI};
use russell_pde::{EssentialBcs1d, FdmLaplacian1d, Grid1d, Side};
use russell_sparse::{Genie, LinSolver, Sym};

const SAVE_FIGURE: bool = false;

#[test]
fn test_laplace1d_2() {
    // This problem simulates the heat conduction-confection of 1D rod.
    //
    // The rod has a length of lx = 0.05 m and the conductivity coefficient
    // is 0.01571 W/m/°C. The surrounding environment has a temperature
    // of 20°C and the convection coefficient is 2 π W/m/°C. The left-hand
    // side of the rod is kept at a constant temperature of 320°C, while
    // the right-hand side is insulated. The goal is to find the temperature
    // distribution along the rod.
    //
    // The Model is:
    //
    //      ∂²ϕ
    // - kx ——— + (ϕ - ϕ∞) β = 0
    //      ∂x²
    //
    // where ϕ∞ = 20°C is the temperature of the surrounding environment and
    // β = 2 π W/m/°C is the convection coefficient. The essential boundary
    // condition is:
    //
    // ϕ(0) = ϕₐ = 320°C
    //
    // The closed-form solution is:
    //
    //                       cosh(m (lx - x))
    // ϕ(x) = ϕ∞ + (ϕₐ - ϕ∞) —————————————————
    //                           cosh(ck lx)
    //
    // where m = √(β / kx)

    // constants
    let nx = 21;
    let lx = 0.05;
    let kx = 0.01571;
    let beta = 2.0 * PI;
    let phi_a = 320.0;
    let phi_inf = 20.0;

    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, lx, nx).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set(&grid, Side::Xmin, |_| phi_a);

    // allocate the Laplacian operator
    // (note that we have to use negative kx)
    let fdm = FdmLaplacian1d::new(grid, ebcs, -kx).unwrap();

    // solving K u = F from:
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ K   C │ │ u │   │ f │
    // │       │ │   │ = │   │
    // │ c   k │ │ p │   │ g │
    // └       ┘ └   ┘   └   ┘
    // where F = f - C p

    // assemble the coefficient matrix and the lhs and rhs vectors
    let nu = fdm.get_dims_sps().0;
    let extra_nnz = nu; // diagonal entries due to ϕ β
    let (mut kk, cc_mat) = fdm.get_matrices_sps(extra_nnz, Sym::No);
    let (mut u, p, mut ff) = fdm.get_vectors(|_| phi_inf * beta); // (- ϕ∞ β) goes to the rhs
    let cc = cc_mat.unwrap();

    // add the diagonal entries due to ϕ β
    for i in 0..nu {
        kk.put(i, i, beta).unwrap();
    }

    // update the right-hand side with the prescribed values
    cc.mat_vec_mul_update(&mut ff, -1.0, &p).unwrap(); // F -= C p

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk, None).unwrap();
    solver.actual.solve(&mut u, &ff, false).unwrap();

    // results: a = (u, p)
    let a = fdm.get_composed_vector(&u, &p);

    // analytical solution
    let m = f64::sqrt(beta / kx);
    let analytical = |x| phi_inf + (phi_a - phi_inf) * f64::cosh(m * (lx - x)) / f64::cosh(m * lx);
    fdm.loop_over_grid_points(|m, x| {
        println!("{}: ϕ = {} ({})", m, a[m], analytical(x));
        approx_eq(a[m], analytical(x), 0.0155);
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
        fdm.loop_over_grid_points(|i, x| {
            xx_num[i] = x;
        });
        let uu_num = a.as_data();
        curve_ana.draw(&xx_ana, &uu_ana);
        curve_num.draw(&xx_num, &uu_num);
        let mut plot = Plot::new();
        plot.add(&curve_ana)
            .add(&curve_num)
            .grid_labels_legend("$x$", "$\\phi$")
            .save("/tmp/russell_pde/test_laplace1d_2.svg")
            .unwrap();
    }
}
