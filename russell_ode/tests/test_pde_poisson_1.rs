use plotpy::{Contour, Plot};
use russell_lab::{vec_approx_eq, Vector};
use russell_ode::PdeDiscreteLaplacian2d;
use russell_sparse::{Genie, LinSolver, SparseMatrix};

const SAVE_FIGURE: bool = false;

#[test]
fn test_pde_poisson_1() {
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
    let mut fdm = PdeDiscreteLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny).unwrap();

    // set zero essential boundary conditions
    fdm.set_homogeneous_boundary_conditions();

    // compute the augmented coefficient matrix and the correction matrix
    let (aa, _) = fdm.coefficient_matrix().unwrap();

    // allocate the left- and right-hand side vectors
    let dim = fdm.dim();
    let mut phi = Vector::new(dim);
    let mut rhs = Vector::new(dim);

    // set the 'prescribed' part of the left-hand side vector with the essential values
    // (this step is not needed with homogeneous boundary conditions)

    // initialize the right-hand side vector with the correction
    // (this step is not needed with homogeneous boundary conditions)

    // set the right-hand side vector with the source term
    fdm.loop_over_grid_points(|i, x, y| {
        rhs[i] = 2.0 * x * (y - 1.0) * (y - 2.0 * x + x * y + 2.0) * f64::exp(x - y);
    });

    // set the 'prescribed' part of the right-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        rhs[i] = value; // bp := ϕp
    });

    // solve the linear system
    let mut mat = SparseMatrix::from_coo(aa);
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&mut mat, None).unwrap();
    solver.actual.solve(&mut phi, &mut mat, &rhs, false).unwrap();

    // check
    let mut phi_correct = Vector::new(dim);
    let analytical = |x, y| x * y * (x - 1.0) * (y - 1.0) * f64::exp(x - y);
    fdm.loop_over_grid_points(|i, x, y| {
        phi_correct[i] = analytical(x, y);
    });
    vec_approx_eq(&phi, &phi_correct, 1e-3);

    // plot results
    if SAVE_FIGURE {
        let mut contour_num = Contour::new();
        let mut contour_ana = Contour::new();
        let mut xx = vec![vec![0.0; nx]; ny];
        let mut yy = vec![vec![0.0; nx]; ny];
        let mut zz_num = vec![vec![0.0; nx]; ny];
        let mut zz_ana = vec![vec![0.0; nx]; ny];
        fdm.loop_over_grid_points(|i, x, y| {
            let row = i / nx;
            let col = i % nx;
            xx[row][col] = x;
            yy[row][col] = y;
            zz_num[row][col] = phi[i];
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
            .save("/tmp/russell_ode/test_pde_poisson_1.svg")
            .unwrap();
    }
}
