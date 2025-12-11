use plotpy::{Contour, Plot};
use russell_lab::approx_eq;
use russell_pde::{EssentialBcs2d, Grid2d, SpectralLaplacianCurv2d, TransfiniteSamples};
use russell_sparse::{CscMatrix, Genie, LinSolver};

const SAVE_FIGURE: bool = true;

#[test]
fn test_spectral_curv_quarter_ring() {
    // Example 7.1.4 on page 259 of Kopriva's book
    //
    // Approximate the solution of
    //
    //         16 ln(r)
    // ∇²ϕ = - ———————— sin(4θ)
    //            r²
    //
    // where: r = √(x² + y²) and θ = arctan(y/x)
    //
    // on a transfinite mapped domain with homogeneous essential boundary conditions.
    // The mapped domain is quarter ring defined by 1 ≤ r ≤ 3 and 0 ≤ θ ≤ π/2.
    //
    // The analytical solution is:
    //
    // ϕ = ln(r) sin(4θ)
    //
    // # Reference
    //
    // * Kopriva LN (2009) - Implementing Spectral Methods for Partial Differential Equations, Springer

    // define the source term
    let source = |x, y| {
        let r = f64::sqrt(x * x + y * y);
        let theta = f64::atan2(y, x);
        -16.0 * f64::ln(r) * f64::sin(4.0 * theta) / (r * r)
    };

    // allocate the grid on [-1, 1] × [-1, 1] and then map to a quarter ring
    let (nx, ny) = (9, 9);
    let grid = Grid2d::new_chebyshev_gauss_lobatto(-1.0, 1.0, -1.0, 1.0, nx, ny).unwrap();
    let map = TransfiniteSamples::quarter_ring_2d(1.0, 3.0);

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set_homogeneous(&grid);

    // allocate the Laplacian operator
    let mut spectral = SpectralLaplacianCurv2d::new(grid, ebcs, map).unwrap();

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk_bar, kk_check) = spectral.get_matrices();
    let (mut a_bar, a_check, mut f_bar) = spectral.get_vectors(source);

    let csc_mat = CscMatrix::from_coo(&kk_bar).unwrap();
    csc_mat
        .write_matrix_market("/tmp/russell_pde/cuv_quarter.smat", true, 1e-16)
        .unwrap();

    // initialize the right-hand side
    kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check).unwrap(); // f̄ -= Ǩ ǎ

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk_bar, None).unwrap();
    solver.actual.solve(&mut a_bar, &f_bar, false).unwrap();

    // results
    let a = spectral.get_joined_vector(&a_bar, &a_check);

    // check
    let analytical = |x, y| {
        let r = f64::sqrt(x * x + y * y);
        let theta = f64::atan2(y, x);
        f64::ln(r) * f64::sin(4.0 * theta)
    };
    spectral.for_each_coord(|m, x, y| {
        // approx_eq(a[m], analytical(x, y), 2.1e-3);
    });

    // plot results
    if SAVE_FIGURE {
        let mut contour_num = Contour::new();
        let mut contour_ana = Contour::new();
        let mut xx = vec![vec![0.0; nx]; ny];
        let mut yy = vec![vec![0.0; nx]; ny];
        let mut zz_num = vec![vec![0.0; nx]; ny];
        let mut zz_ana = vec![vec![0.0; nx]; ny];
        spectral.for_each_coord(|m, x, y| {
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
        plot.add(&contour_num)
            .add(&contour_ana)
            .set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save("/tmp/russell_pde/test_spectral_curv_quarter_ring.svg")
            .unwrap();
    }
}
