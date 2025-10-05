use plotpy::{linspace, Curve, Plot};
use russell_lab::{approx_eq, math::PI, Vector};
use russell_pde::{FdmLaplacian1d, Side};
use russell_sparse::{Genie, LinSolver};

const SAVE_FIGURE: bool = false;

#[test]
fn test_laplace1d_2_lag() {
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
    let lx = 0.05;
    let kx = 0.01571;
    let beta = 2.0 * PI;
    let phi_a = 320.0;
    let phi_inf = 20.0;

    // allocate the Laplacian operator
    // (note that we have to use negative kx)
    let nx = 21;
    let mut fdm = FdmLaplacian1d::new(-kx, 0.0, lx, nx, Some(beta)).unwrap();

    // set essential boundary conditions
    fdm.set_essential_boundary_condition(Side::Xmin, |_| phi_a);

    // compute the augmented coefficient matrix for the Lagrange multipliers method
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ K  Eᵀ │ │ u │   │ f │
    // │       │ │   │ = │   │
    // │ E  0  │ │ w │   │ ū │
    // └       ┘ └   ┘   └   ┘
    //     A      lhs     rhs
    let aa = fdm.augmented_coefficient_matrix().unwrap();

    // allocate the left- and right-hand side vectors
    let np = fdm.num_prescribed();
    let dim = fdm.dim();
    let mut lhs = Vector::new(dim + np);
    let mut rhs = Vector::new(dim + np);

    // add the source term to the right-hand side vector
    fdm.loop_over_grid_points(|m, _| {
        rhs[m] = phi_inf * beta;
    });

    // add the prescribed values to the right-hand side vector
    fdm.loop_over_prescribed_values(|ip, _, value| {
        rhs[dim + ip] = value;
    });

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&aa, None).unwrap();
    solver.actual.solve(&mut lhs, &rhs, false).unwrap();

    // results
    let m = f64::sqrt(beta / kx);
    let ana_phi = |x| phi_inf + (phi_a - phi_inf) * f64::cosh(m * (lx - x)) / f64::cosh(m * lx);
    fdm.loop_over_grid_points(|m, x| {
        println!("{}: ϕ = {} ({})", m, lhs[m], ana_phi(x));
        approx_eq(lhs[m], ana_phi(x), 0.0155);
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
        let uu_num = Vec::from(&lhs.as_data()[..dim]);
        curve_ana.draw(&xx_ana, &uu_ana);
        curve_num.draw(&xx_num, &uu_num);
        let mut plot = Plot::new();
        plot.add(&curve_ana)
            .add(&curve_num)
            .grid_labels_legend("$x$", "$\\phi$")
            .save("/tmp/russell_pde/test_laplace1d_2_lag.svg")
            .unwrap();
    }
}
