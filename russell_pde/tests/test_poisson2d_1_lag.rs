use plotpy::{Contour, Plot};
use russell_lab::{array_approx_eq, Vector};
use russell_pde::FdmLaplacian2d;
use russell_sparse::{Genie, LinSolver};

const SAVE_FIGURE: bool = false;

#[test]
fn test_poisson2d_1_lag() {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    // ∂²ϕ   ∂²ϕ
    // ——— + ——— = 2 x (y - 1) (y - 2 x + x y + 2) exp(x - y)
    // ∂x²   ∂y²
    //
    // on a (1.0 × 1.0) square with the homogeneous boundary conditions.
    //
    // The analytical solution is:
    //
    // ϕ(x, y) = x y (x - 1) (y - 1) exp(x - y)

    // allocate the Laplacian operator
    let (nx, ny) = (9, 9);
    let mut fdm = FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny).unwrap();

    // set zero essential boundary conditions
    fdm.set_homogeneous_boundary_conditions();

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
        rhs[m] = 2.0 * x * (y - 1.0) * (y - 2.0 * x + x * y + 2.0) * f64::exp(x - y);
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
    let analytical = |x, y| x * y * (x - 1.0) * (y - 1.0) * f64::exp(x - y);
    fdm.loop_over_grid_points(|m, x, y| {
        phi_correct[m] = analytical(x, y);
    });
    array_approx_eq(&lhs.as_data()[..dim], phi_correct.as_data(), 1e-3);

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
            .save("/tmp/russell_pde/test_poisson2d_1_lag.svg")
            .unwrap();
    }
}
