use russell_lab::{mat_approx_eq, vec_approx_eq};
use russell_pde::{EssentialBcs2d, Grid2d, Side, SpectralLaplacian2d, StrError};
use russell_sparse::{Genie, LinSolver};

#[test]
fn test_spectral_laplace2d_4() -> Result<(), StrError> {
    // Approximate the solution of the problem:
    //
    // ∂²ϕ   ∂²ϕ
    // ——— + ——— = 0
    // ∂x²   ∂y²
    //
    // on a [-1,1] × [-1,1] square with the following
    // essential (Dirichlet) boundary conditions:
    //
    // left:    ϕ(-1.0, y) = 1.0
    // right:   ϕ( 1.0, y) = 2.0
    // bottom:  ϕ(x, -1.0) = 1.0
    // top:     ϕ(x,  1.0) = 2.0

    // polynomial degree
    let nn = 3;

    // allocate the grid
    let (nx, ny) = (nn + 1, nn + 1);
    let grid = Grid2d::new_chebyshev_gauss_lobatto(-1.0, 1.0, -1.0, 1.0, nx, ny)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(&grid, Side::Xmin, |_, _| 1.0);
    ebcs.set(&grid, Side::Xmax, |_, _| 2.0);
    ebcs.set(&grid, Side::Ymin, |_, _| 1.0);
    ebcs.set(&grid, Side::Ymax, |_, _| 2.0);

    // allocate the Laplacian operator
    let (kx, ky) = (1.0, 1.0);
    let spectral = SpectralLaplacian2d::new(grid, ebcs, kx, ky)?;

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk_bar, kk_check) = spectral.get_matrices();
    let (mut a_bar, a_check, mut f_bar) = spectral.get_vectors(|_, _| 0.0);
    let kk_bar_dense = kk_bar.as_dense();
    let kk_bar_correct = &[
        [-10.666666666666668, 2.666666666666667, 2.666666666666667, 0.0],
        [2.666666666666667, -10.666666666666668, 0.0, 2.666666666666667],
        [2.666666666666667, 0.0, -10.666666666666668, 2.666666666666667],
        [0.0, 2.666666666666667, 2.666666666666667, -10.666666666666668],
    ];
    mat_approx_eq(&kk_bar_dense, kk_bar_correct, 1e-15);

    // fix the right-hand side (note that f = 0)
    kk_check.mat_vec_mul(&mut f_bar, -1.0, &a_check).unwrap(); // f̄ -= Ǩ ǎ

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk_bar, None).unwrap();
    solver.actual.solve(&mut a_bar, &f_bar, false).unwrap();

    // results
    let a = spectral.get_joined_vector(&a_bar, &a_check);

    // check
    let a_ref = &[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.125, 1.5, 2.0, 1.0, 1.5, 1.875, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    vec_approx_eq(&a, a_ref, 1e-15);

    Ok(())
}
