use plotpy::{Plot, Surface};
use russell_lab::approx_eq;
use russell_lab::math::PI;
use russell_pde::{EssentialBcs2d, Grid2d, Side, Spc2d, StrError};
use russell_sparse::{Genie, LinSolver};

// This is the benchmark solution 5.2.1.7 on page 170 of Kopriva's book.
//
// Approximate the solution of the problem:
//
// ∂²ϕ   ∂²ϕ
// ——— + ——— = -8 π² cos(2πx) sin(2πy)
// ∂x²   ∂y²
//
// on a [-1,1] x [-1,1] square with the following essential boundary conditions:
//
// Xmin: ϕ(-1, y) = sin(2πy)
// Xmax: ϕ( 1, y) = sin(2πy)
// Ymin: ϕ(x, -1) = 0
// Ymax: ϕ(x,  1) = 0
//
// The analytical solution is:
//
// ϕ(x, y) = cos(2πx) sin(2πy)
//
// # Reference
//
// * Kopriva LN (2009) - Implementing Spectral Methods for Partial Differential Equations, Springer

const SAVE_FIGURE: bool = false;

#[test]
fn test_spectral_poisson2d_5() -> Result<(), StrError> {
    for (nn, tol, correct_log10_err_max) in vec![
        (8, 1e-1, -1.5375), // Table 5.1 (Chebyshev row), page 171, Kopriva's book
        (12, 1e-3, -3.8044),
        // (16, 1e-6, -6.6535),
        // (20, 1e-9, -9.8774),
    ] {
        let err_max = run_test(nn, tol)?;
        let log10_err_max = f64::log10(err_max);
        println!(
            "N = {:>2}, log10(max(error)) = {:>8.4} ({:>})",
            nn, log10_err_max, correct_log10_err_max
        );
        approx_eq(log10_err_max, correct_log10_err_max, 1e-4);
    }
    Ok(())
}

/// Runs the test and returns max(error)
fn run_test(nn: usize, tol: f64) -> Result<f64, StrError> {
    // allocate the grid
    let (nx, ny) = (nn + 1, nn + 1);
    let grid = Grid2d::new_chebyshev_gauss_lobatto(-1.0, 1.0, -1.0, 1.0, nx, ny)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(Side::Xmin, |_, y| f64::sin(2.0 * PI * y));
    ebcs.set(Side::Xmax, |_, y| f64::sin(2.0 * PI * y));
    ebcs.set(Side::Ymin, |_, _| 0.0);
    ebcs.set(Side::Ymax, |_, _| 0.0);

    // allocate the Laplacian operator
    let (kx, ky) = (1.0, 1.0);
    let spectral = Spc2d::new(grid, ebcs, kx, ky)?;

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk_bar, kk_check) = spectral.get_matrices();
    let (mut a_bar, a_check, mut f_bar) =
        spectral.get_vectors(|x, y| -8.0 * PI * PI * f64::cos(2.0 * PI * x) * f64::sin(2.0 * PI * y));

    // fix the right-hand side
    kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check)?; // f̄ -= Ǩ ǎ

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk_bar, None).unwrap();
    solver.actual.solve(&mut a_bar, &f_bar, false).unwrap();

    // results
    let a = spectral.get_joined_vector(&a_bar, &a_check);

    // check
    let analytical = |x, y| f64::cos(2.0 * PI * x) * f64::sin(2.0 * PI * y);
    let mut err_max = 0.0;
    spectral.for_each_coord(|m, x, y| {
        let err = f64::abs(a[m] - analytical(x, y));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x, y), tol);
    });

    // plot
    if SAVE_FIGURE {
        let mut xx = vec![vec![0.0; nx]; ny];
        let mut yy = vec![vec![0.0; nx]; ny];
        let mut zz = vec![vec![0.0; nx]; ny];
        spectral.for_each_coord(|m, x, y| {
            let row = m / nx;
            let col = m % nx;
            xx[row][col] = x;
            yy[row][col] = y;
            zz[row][col] = a[m];
        });
        let mut surf = Surface::new();
        surf.set_with_surface(false)
            .set_with_wireframe(true)
            .draw(&xx, &yy, &zz);
        let mut plot = Plot::new();
        plot.add(&surf)
            .set_figure_size_points(600.0, 600.0)
            .set_camera(30.0, 30.0)
            .save("/tmp/russell_pde/test_spectral_poisson2d_6.svg")?;
    }
    Ok(err_max)
}
