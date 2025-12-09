use plotpy::{Plot, Surface};
use russell_lab::vec_approx_eq;
use russell_pde::{EssentialBcs2d, Grid2d, SpectralLaplacian2d, StrError};
use russell_sparse::{Genie, LinSolver};

const SAVE_FIGURE: bool = false;

#[test]
fn test_spectral_poisson2d_4() -> Result<(), StrError> {
    // Approximate the solution of the problem:
    //
    // ∂²ϕ   ∂²ϕ
    // ——— + ——— = 10 sin(8x⋅(y-1))
    // ∂x²   ∂y²
    //
    // on a [-1,1] x [-1,1] square with homogeneous boundary conditions.
    //
    // This is Problem 16 on page 70 of Trefethen's book.
    //
    // * Trefethen LN (2000) - Spectral Methods in MATLAB, SIAM

    // polynomial degree
    let nn = 4;

    // allocate the grid
    let (nx, ny) = (nn + 1, nn + 1);
    let grid = Grid2d::new_chebyshev_gauss_lobatto(-1.0, 1.0, -1.0, 1.0, nx, ny)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set_homogeneous(&grid);

    // allocate the Laplacian operator
    let (kx, ky) = (1.0, 1.0);
    let spectral = SpectralLaplacian2d::new(grid, ebcs, kx, ky)?;

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk_bar, kk_check) = spectral.get_matrices();
    let (mut a_bar, a_check, mut f_bar) = spectral.get_vectors(|x, y| 10.0 * f64::sin(8.0 * x * (y - 1.0)));

    // fix the right-hand side
    kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check)?; // f̄ -= Ǩ ǎ

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk_bar, None).unwrap();
    solver.actual.solve(&mut a_bar, &f_bar, false).unwrap();

    // results
    let a = spectral.get_joined_vector(&a_bar, &a_check);

    // check
    #[rustfmt::skip]
    let a_correct = &[
		0.0,  0.000000000000000,  0.0,  0.000000000000000, 0.0,
		0.0,  0.181363633964132,  0.0, -0.181363633964131, 0.0,
		0.0,  0.292713394079481,  0.0, -0.292713394079479, 0.0,
		0.0, -0.329593843114906,  0.0,  0.329593843114906, 0.0,
		0.0,  0.000000000000000,  0.0,  0.000000000000000, 0.0,
    ];
    vec_approx_eq(&a, a_correct, 1e-14);

    // plot
    if SAVE_FIGURE {
        let mut xx = vec![vec![0.0; nx]; ny];
        let mut yy = vec![vec![0.0; nx]; ny];
        let mut zz = vec![vec![0.0; nx]; ny];
        spectral.get_grid().for_each_coord(|m, x, y| {
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
            .set_camera(30.0, -120.0)
            .save("/tmp/russell_pde/test_spectral_poisson2d_4.svg")?;
    }
    Ok(())
}
