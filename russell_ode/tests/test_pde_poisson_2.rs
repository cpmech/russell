use plotpy::{Contour, Plot};
use russell_lab::{math::PI, vec_approx_eq, StrError, Vector};
use russell_ode::{PdeDiscreteLaplacian2d, Side};
use russell_sparse::{Genie, LinSolver, SparseMatrix};

const SAVE_FIGURE: bool = false;

#[test]
fn main() -> Result<(), StrError> {
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
    let mut fdm = PdeDiscreteLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny).unwrap();

    // set essential boundary conditions
    fdm.set_essential_boundary_condition(Side::Left, |_, _| 0.0);
    fdm.set_essential_boundary_condition(Side::Right, |_, _| 0.0);
    fdm.set_essential_boundary_condition(Side::Bottom, |_, _| 0.0);
    fdm.set_essential_boundary_condition(Side::Top, |x, _| f64::sin(PI * x));

    // compute the augmented coefficient matrix and the correction matrix
    //
    // ┌          ┐ ┌    ┐   ┌                 ┐
    // │ Auu   0  │ │ ϕu │   │ source - Aup⋅ϕp │
    // │          │ │    │ = │                 │
    // │  0    1  │ │ ϕp │   │        ϕp       │
    // └          ┘ └    ┘   └                 ┘
    // A := augmented(Auu)
    //
    // ┌          ┐ ┌    ┐   ┌        ┐
    // │  0   Aup │ │ .. │   │ Aup⋅ϕp │
    // │          │ │    │ = │        │
    // │  0    0  │ │ ϕp │   │   0    │
    // └          ┘ └    ┘   └        ┘
    // C := augmented(Aup)
    let (aa, cc) = fdm.coefficient_matrix().unwrap();

    // allocate the left- and right-hand side vectors
    let dim = fdm.dim();
    let mut phi = Vector::new(dim);
    let mut rhs = Vector::new(dim);

    // set the 'prescribed' part of the left-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        phi[i] = value; // ϕp := ϕp
    });

    // initialize the right-hand side vector with the correction
    cc.mat_vec_mul(&mut rhs, -1.0, &phi)?; // bu := -Aup⋅ϕp

    // set the right-hand side vector with the source term (note plus-equal)
    fdm.loop_over_grid_points(|i, x, y| {
        rhs[i] += -PI * PI * y * f64::sin(PI * x); // bu += source
    });

    // set the 'prescribed' part of the right-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        rhs[i] = value; // bp := ϕp
    });

    // solve the linear system
    let mut mat = SparseMatrix::from_coo(aa);
    let mut solver = LinSolver::new(Genie::Umfpack)?;
    solver.actual.factorize(&mut mat, None)?;
    solver.actual.solve(&mut phi, &mut mat, &rhs, false)?;

    // check
    let mut phi_correct = Vector::new(dim);
    let analytical = |x, y| y * f64::sin(PI * x);
    fdm.loop_over_grid_points(|i, x, y| {
        phi_correct[i] = analytical(x, y);
    });
    vec_approx_eq(phi.as_data(), phi_correct.as_data(), 0.001036);

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
            .save("/tmp/russell_ode/test_pde_poisson_2.svg")?;
    }
    Ok(())
}
