use plotpy::{Contour, Plot};
use russell_lab::{StrError, Vector};
use russell_pde::{EssentialBcs2d, FdmLaplacian2d, Grid2d, Side};
use russell_sparse::{Genie, LinSolver, Sym};

fn main() -> Result<(), StrError> {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    //  ∂²ϕ     ∂²ϕ
    //  ———  +  ——— = 0
    //  ∂x²     ∂y²
    //
    // on a (1.0 × 1.0) rectangle with the following
    // essential (Dirichlet) boundary conditions:
    //
    // left:    ϕ(0.0, y) = 50.0
    // right:   ϕ(1.0, y) =  0.0
    // bottom:  ϕ(x, 0.0) =  0.0
    // top:     ϕ(x, 1.0) = 50.0

    // allocate the grid
    let (nx, ny) = (31, 31);
    let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 3.0, nx, ny)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(&grid, Side::Xmin, |_, _| 50.0);
    ebcs.set(&grid, Side::Xmax, |_, _| 0.0);
    ebcs.set(&grid, Side::Ymin, |_, _| 0.0);
    ebcs.set(&grid, Side::Ymax, |_, _| 50.0);

    // allocate the Laplacian operator
    let (kx, ky) = (1.0, 1.0);
    let fdm = FdmLaplacian2d::new(grid, ebcs, kx, ky)?;

    // solving K u = h from:
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ K   C │ │ u │   │ f │
    // │       │ │   │ = │   │
    // │ c   k │ │ p │   │ g │
    // └       ┘ └   ┘   └   ┘
    // where h = f - C p

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk, cc_mat) = fdm.get_matrices_sps(0, Sym::No);
    let (mut u, p, mut h) = fdm.get_vectors(|_, _| 0.0);
    let cc = cc_mat.unwrap();

    // set the right-hand side (note that f = 0)
    cc.mat_vec_mul(&mut h, -1.0, &p)?; // h = - C p

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack)?;
    solver.actual.factorize(&kk, None)?;
    solver.actual.solve(&mut u, &h, false)?;

    // results: a = (u, p)
    let a = fdm.get_composed_vector(&u, &p);

    // plot results
    let mut contour = Contour::new();
    let mut xx = vec![vec![0.0; nx]; ny];
    let mut yy = vec![vec![0.0; nx]; ny];
    let mut zz_num = vec![vec![0.0; nx]; ny];
    fdm.loop_over_grid_points(|m, x, y| {
        let row = m / nx;
        let col = m % nx;
        xx[row][col] = x;
        yy[row][col] = y;
        zz_num[row][col] = a[m];
    });
    let levels = Vector::linspace(0.0, 50.0, 11)?;
    contour.set_levels(levels.as_data()).draw(&xx, &yy, &zz_num);
    let mut plot = Plot::new();
    plot.add(&contour);
    plot.set_equal_axes(true)
        .set_figure_size_points(600.0, 600.0)
        .save("/tmp/russell_pde/laplace_equation.svg")?;
    Ok(())
}
