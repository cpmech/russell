#![allow(unused)]

use plotpy::{Contour, Plot};
use russell_lab::{approx_eq, mat_approx_eq, math::PI, vec_approx_eq};
use russell_pde::{EssentialBcs2d, Grid2d, Side, SpectralLaplacian2d, StrError};
use russell_sparse::{Genie, LinSolver, Sym};

const SAVE_FIGURE: bool = false;

#[test]
fn test_spectral_laplace2d_1() -> Result<(), StrError> {
    // Approximate the solution of the problem:
    //
    //    ∂²ϕ     ∂²ϕ
    //    ———  +  ——— = 10 sin(8x⋅(y-1))
    //    ∂x²     ∂y²
    //
    // on a unit square with homogeneous essential boundary conditions.
    //
    // This is Problem 16 on page 70 of Trefethen's book.
    //
    // * Trefethen LN (2000) - Spectral Methods in MATLAB, SIAM

    // polynomial degree
    let nn = 4;

    // allocate the grid
    let grid = Grid2d::new_chebyshev_gauss_lobatto(-1.0, 1.0, -1.0, 1.0, nn + 1, nn + 1)?;

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
    Ok(())
}
