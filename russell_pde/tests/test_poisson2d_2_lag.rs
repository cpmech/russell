use plotpy::{Contour, Plot};
use russell_lab::{array_approx_eq, math::PI, Vector};
use russell_pde::{FdmLaplacian2d, Side};
use russell_sparse::{Genie, LinSolver};

const SAVE_FIGURE: bool = false;

#[test]
fn test_poisson2d_2_lag() {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    // ∂²ϕ   ∂²ϕ
    // ——— + ——— = - π² y sin(π x)
    // ∂x²   ∂y²
    //
    // on a (1.0 × 1.0) square with the following essential boundary conditions:
    //
    // left:    ϕ(0.0, y) = 0.0
    // right:   ϕ(1.0, y) = 0.0
    // bottom:  ϕ(x, 0.0) = 0.0
    // top:     ϕ(x, 1.0) = sin(π x)
    //
    // The analytical solution is:
    //
    // ϕ(x, y) = y sin(π x)
    //
    // Reference: Olver PJ (2020) - page 210 - Introduction to Partial Differential Equations, Springer

    // allocate the Laplacian operator
    let (nx, ny) = (17, 17);
    let mut fdm = FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny).unwrap();

    // set essential boundary conditions
    fdm.set_essential_boundary_condition(Side::Xmin, |_, _| 0.0);
    fdm.set_essential_boundary_condition(Side::Xmax, |_, _| 0.0);
    fdm.set_essential_boundary_condition(Side::Ymin, |_, _| 0.0);
    fdm.set_essential_boundary_condition(Side::Ymax, |x, _| f64::sin(PI * x));

    // compute the augmented coefficient matrix for the Lagrange multipliers method
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ K  Eᵀ │ │ u │   │ f │
    // │       │ │   │ = │   │
    // │ E  0  │ │ w │   │ ū │
    // └       ┘ └   ┘   └   ┘
    //     A      lhs     rhs
    let aa = fdm.augmented_coefficient_matrix(0).unwrap();

    // allocate the left- and right-hand side vectors
    let np = fdm.num_prescribed();
    let dim = fdm.dim();
    let mut lhs = Vector::new(dim + np);
    let mut rhs = Vector::new(dim + np);

    // add the source term to the right-hand side vector
    fdm.loop_over_grid_points(|m, x, y| {
        rhs[m] = -PI * PI * y * f64::sin(PI * x); // f1 += source
    });

    // add the prescribed values to the right-hand side vector
    fdm.loop_over_prescribed_values(|ip, _, value| {
        rhs[dim + ip] = value;
    });

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&aa, None).unwrap();
    solver.actual.solve(&mut lhs, &rhs, false).unwrap();

    // check
    let mut phi_correct = Vector::new(dim);
    let analytical = |x, y| y * f64::sin(PI * x);
    fdm.loop_over_grid_points(|m, x, y| {
        phi_correct[m] = analytical(x, y);
    });
    array_approx_eq(&lhs.as_data()[..dim], phi_correct.as_data(), 0.001036);

    // plot results
    if SAVE_FIGURE {
        let mut contour_num = Contour::new();
        let mut contour_ana = Contour::new();
        let mut xx = vec![vec![0.0; nx]; ny];
        let mut yy = vec![vec![0.0; nx]; ny];
        let mut zz_num = vec![vec![0.0; nx]; ny];
        let mut zz_ana = vec![vec![0.0; nx]; ny];
        fdm.loop_over_grid_points(|m, x, y| {
            let row = m / nx;
            let col = m % nx;
            xx[row][col] = x;
            yy[row][col] = y;
            zz_num[row][col] = lhs[m];
            zz_ana[row][col] = analytical(x, y);
        });
        contour_num.set_no_lines(false).draw(&xx, &yy, &zz_num);
        contour_ana
            .set_colors(&["None"])
            .set_no_colorbar(true)
            .set_no_labels(true)
            .set_line_color("yellow")
            .set_line_style(":")
            .set_line_width(2.0)
            .draw(&xx, &yy, &zz_ana);
        let mut plot = Plot::new();
        plot.add(&contour_num).add(&contour_ana);
        plot.set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save("/tmp/russell_pde/test_poisson2d_2_lag.svg")
            .unwrap();
    }
}
