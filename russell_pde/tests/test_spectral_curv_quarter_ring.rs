use plotpy::{Contour, Plot};
use russell_lab::approx_eq;
use russell_pde::{EssentialBcs2d, Grid2d, NaturalBcs2d, Side, SpcMap2d, StrError, TransfiniteSamples};
use russell_sparse::{Genie, LinSolver};

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
// on a transfinite mapped domain. The mapped domain is quarter ring
// defined by 1 ≤ r ≤ 3 and 0 ≤ θ ≤ π/2. The boundary conditions are:
//
// R-min (aka Xmin): ϕ = 0               on r = 1
// R-max (aka Xmax): ϕ = ln(3) sin(4θ)   on r = 3
// S-min (aka Ymin): ϕ = 0               on θ = 0
// S-max (aka Ymax): ϕ = 0               on θ = π/2
//
// The analytical solution is:
//
// ϕ = ln(r) sin(4θ)
//
// # Reference
//
// * Kopriva LN (2009) - Implementing Spectral Methods for Partial Differential Equations, Springer

const SAVE_FIGURE: bool = false;

#[test]
fn test_spectral_curv_quarter_ring() -> Result<(), StrError> {
    for (nn, tol, correct_log10_err_max) in vec![
        (8, 1.03e-1, -4.0), // Table 7.1 (Orthogonal column), page 261, Kopriva's book
                            // (12, 3.05e-8, -8.0),
                            // (16, 1.02e-11, -11.0),
                            // (20, 3.47e-14, -14.0),
    ] {
        let err_max = run_test(nn, tol)?;
        let log10_err_max = f64::log10(err_max);
        println!(
            "N = {:>2}, log10(max(error)) = {:>8.4} ({:>})",
            nn, log10_err_max, correct_log10_err_max
        );
        approx_eq(log10_err_max, correct_log10_err_max, 0.55);
    }
    Ok(())
}

/// Runs the test and returns max(error)
fn run_test(nn: usize, tol: f64) -> Result<f64, StrError> {
    // define the analytical solution
    let analytical = |x, y| {
        let r = f64::sqrt(x * x + y * y);
        let theta = f64::atan2(y, x);
        f64::ln(r) * f64::sin(4.0 * theta)
    };

    // define the source term
    let source = |x, y| {
        let r = f64::sqrt(x * x + y * y);
        let theta = f64::atan2(y, x);
        -16.0 * f64::ln(r) * f64::sin(4.0 * theta) / (r * r)
    };

    // allocate the grid on [-1, 1] × [-1, 1] and then map to a quarter ring
    let (nx, ny) = (nn + 1, nn + 1);
    let grid = Grid2d::new_chebyshev_gauss_lobatto(-1.0, 1.0, -1.0, 1.0, nx, ny).unwrap();
    let map = TransfiniteSamples::quarter_ring_2d(1.0, 3.0);

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set_homogeneous();
    ebcs.set(Side::Xmax, |x, y| analytical(x, y));

    // natural boundary conditions
    let nbcs = NaturalBcs2d::new();

    // allocate the Laplacian operator
    let mut spectral = SpcMap2d::new(grid, ebcs, nbcs, map).unwrap();

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk_bar, kk_check) = spectral.get_matrices();
    let (mut a_bar, a_check, mut f_bar) = spectral.get_vectors(source);

    // initialize the right-hand side
    kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check).unwrap(); // f̄ -= Ǩ ǎ

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk_bar, None).unwrap();
    solver.actual.solve(&mut a_bar, &f_bar, false).unwrap();

    // results
    let a = spectral.get_joined_vector(&a_bar, &a_check);

    // check
    let mut err_max = 0.0;
    spectral.for_each_coord(|m, x, y| {
        let err = f64::abs(a[m] - analytical(x, y));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x, y), tol);
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
    Ok(err_max)
}
