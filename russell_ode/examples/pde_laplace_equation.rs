use plotpy::{Contour, Plot};
use russell_lab::{StrError, Vector};
use russell_ode::{PdeDiscreteLaplacian2d, Side};
use russell_sparse::{Genie, LinSolver, SparseMatrix};

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

    // allocate the Laplacian operator
    let (nx, ny) = (31, 31);
    let mut fdm = PdeDiscreteLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny).unwrap();

    // set essential boundary conditions
    fdm.set_essential_boundary_condition(Side::Left, |_, _| 50.0);
    fdm.set_essential_boundary_condition(Side::Right, |_, _| 0.0);
    fdm.set_essential_boundary_condition(Side::Bottom, |_, _| 0.0);
    fdm.set_essential_boundary_condition(Side::Top, |_, _| 50.0);

    // compute the augmented coefficient matrix and the correction matrix
    let (aa, cc) = fdm.coefficient_matrix().unwrap();

    // allocate the left- and right-hand side vectors
    let dim = fdm.dim();
    let mut phi = Vector::new(dim);
    let mut rhs = Vector::new(dim);

    // set the 'prescribed' part of the left-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        phi[i] = value; // xp := xp
    });

    // initialize the right-hand side vector with the correction
    cc.mat_vec_mul(&mut rhs, -1.0, &phi)?; // bu := -Aup⋅xp

    // if there were natural (Neumann) boundary conditions,
    // we could set `bu := natural()` here

    // set the 'prescribed' part of the right-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        rhs[i] = value; // bp := xp
    });

    // solve the linear system
    let mut mat = SparseMatrix::from_coo(aa);
    let mut solver = LinSolver::new(Genie::Umfpack)?;
    solver.actual.factorize(&mut mat, None)?;
    solver.actual.solve(&mut phi, &mut mat, &rhs, false)?;

    // plot results
    let mut contour = Contour::new();
    let mut xx = vec![vec![0.0; nx]; ny];
    let mut yy = vec![vec![0.0; nx]; ny];
    let mut zz_num = vec![vec![0.0; nx]; ny];
    fdm.loop_over_grid_points(|i, x, y| {
        let row = i / nx;
        let col = i % nx;
        xx[row][col] = x;
        yy[row][col] = y;
        zz_num[row][col] = phi[i];
    });
    let levels = Vector::linspace(0.0, 50.0, 11)?;
    contour.set_levels(levels.as_data()).draw(&xx, &yy, &zz_num);
    let mut plot = Plot::new();
    plot.add(&contour);
    plot.set_equal_axes(true)
        .set_figure_size_points(600.0, 600.0)
        .save("/tmp/russel_ode/pde_laplace_equation.svg")?;
    Ok(())
}
