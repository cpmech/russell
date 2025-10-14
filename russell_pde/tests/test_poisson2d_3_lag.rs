use plotpy::{Contour, Plot};
use russell_lab::approx_eq;
use russell_pde::{EssentialBcs2d, FdmLaplacian2dNew, Grid2d};
use russell_sparse::{Genie, LinSolver};

const SAVE_FIGURE: bool = false;

#[test]
fn test_poisson2d_3_lag() {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    //  ∂²ϕ     ∂²ϕ
    //  ———  +  ——— =  source(x, y)
    //  ∂x²     ∂y²
    //
    // on a (1.0 × 1.0) square with homogeneous essential boundary conditions
    //
    // The source term is given by (for a manufactured solution):
    //
    // source(x, y) = 14y³ - (16 - 12x) y² - (-42x² + 54x - 2) y + 4x³ - 16x² + 12x
    //
    // The analytical solution is:
    //
    // ϕ(x, y) = x (1 - x) y (1 - y) (1 + 2x + 7y)

    // allocate the grid
    let (nx, ny) = (11, 11);
    let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, nx, ny).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new(&grid);
    ebcs.set_homogeneous();

    // allocate the Laplacian operator
    let (kx, ky) = (1.0, 1.0);
    let fdm = FdmLaplacian2dNew::new(&ebcs, kx, ky).unwrap();

    // solving:
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ M  Eᵀ │ │ a │   │ r │
    // │       │ │   │ = │   │
    // │ E  0  │ │ w │   │ ū │
    // └       ┘ └   ┘   └   ┘
    //     A       x       b
    // where a = (u, p) and w are the Lagrange multipliers

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (aa, _) = fdm.get_aa_and_ee_matrices(0, false);
    let (mut x, mut b) = ebcs.get_system_vectors_lmm();

    // add the source term to the right-hand side vector
    grid.for_each_coord(|m, x, y| {
        let (xx, yy) = (x * x, y * y);
        let (xxx, yyy) = (xx * x, yy * y);
        b[m] =
            14.0 * yyy - (16.0 - 12.0 * x) * yy - (-42.0 * xx + 54.0 * x - 2.0) * y + 4.0 * xxx - 16.0 * xx + 12.0 * x;
    });

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&aa, None).unwrap();
    solver.actual.solve(&mut x, &b, false).unwrap();

    // results
    let na = ebcs.get_grid().size(); // dimension of a = (u, p)
    let a = &x.as_data()[..na];

    // check
    let analytical = |x, y| x * (1.0 - x) * y * (1.0 - y) * (1.0 + 2.0 * x + 7.0 * y);
    grid.for_each_coord(|m, x, y| {
        approx_eq(a[m], analytical(x, y), 1e-15);
    });

    // plot results
    if SAVE_FIGURE {
        let mut contour_num = Contour::new();
        let mut contour_ana = Contour::new();
        let mut xx = vec![vec![0.0; nx]; ny];
        let mut yy = vec![vec![0.0; nx]; ny];
        let mut zz_num = vec![vec![0.0; nx]; ny];
        let mut zz_ana = vec![vec![0.0; nx]; ny];
        grid.for_each_coord(|m, x, y| {
            let row = m / nx;
            let col = m % nx;
            xx[row][col] = x;
            yy[row][col] = y;
            zz_num[row][col] = a[m];
            zz_ana[row][col] = analytical(x, y);
        });
        contour_num.set_no_lines(false).draw(&xx, &yy, &zz_num);
        contour_ana
            .set_colors(&["None"])
            .set_no_colorbar(true)
            .set_no_labels(true)
            .set_line_color("yellow")
            .set_line_style(":")
            .draw(&xx, &yy, &zz_ana);
        let mut plot = Plot::new();
        plot.add(&contour_num).add(&contour_ana);
        plot.set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save("/tmp/russell_pde/test_poisson2d_3_lag.svg")
            .unwrap();
    }
}
