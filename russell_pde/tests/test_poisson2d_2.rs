use plotpy::{Contour, Plot};
use russell_lab::{approx_eq, math::PI};
use russell_pde::{EssentialBcs2d, FdmLaplacian2d, Grid2d, Side};
use russell_sparse::{Genie, LinSolver, Sym};

const SAVE_FIGURE: bool = false;

#[test]
fn test_poisson2d_2() {
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

    // define the source term
    let source = |x, y| -PI * PI * y * f64::sin(PI * x);

    // allocate the grid
    let (nx, ny) = (17, 17);
    let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, nx, ny).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(&grid, Side::Xmin, |_, _| 0.0);
    ebcs.set(&grid, Side::Xmax, |_, _| 0.0);
    ebcs.set(&grid, Side::Ymin, |_, _| 0.0);
    ebcs.set(&grid, Side::Ymax, |x, _| f64::sin(PI * x));

    // allocate the Laplacian operator
    let (kx, ky) = (1.0, 1.0);
    let fdm = FdmLaplacian2d::new(grid, ebcs, kx, ky).unwrap();

    // solving K u = h from:
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ K   C │ │ u │   │ f │
    // │       │ │   │ = │   │
    // │ c   k │ │ p │   │ g │
    // └       ┘ └   ┘   └   ┘
    // where h = f - C p

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk, cc_mat) = fdm.get_kk_and_cc_matrices(0, Sym::No);
    let (mut u, p, mut h) = fdm.get_vectors(source);
    let cc = cc_mat.unwrap();

    // initialize the right-hand side
    cc.mat_vec_mul_update(&mut h, -1.0, &p).unwrap(); // h -= C p

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk, None).unwrap();
    solver.actual.solve(&mut u, &h, false).unwrap();

    // results: a = (u, p)
    let a = fdm.get_composed_vector(&u, &p);

    // check
    let analytical = |x, y| y * f64::sin(PI * x);
    fdm.loop_over_grid_points(|m, x, y| {
        approx_eq(a[m], analytical(x, y), 0.001036);
    });

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
            .set_line_width(2.0)
            .draw(&xx, &yy, &zz_ana);
        let mut plot = Plot::new();
        plot.add(&contour_num).add(&contour_ana);
        plot.set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save("/tmp/russell_pde/test_poisson2d_2.svg")
            .unwrap();
    }
}
