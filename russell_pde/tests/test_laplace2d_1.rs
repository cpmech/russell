use russell_lab::vec_approx_eq;
use russell_pde::{EssentialBcs2d, FdmLaplacian2d, Grid2d, Side};
use russell_sparse::{Genie, LinSolver, Sym};

#[test]
fn test_laplace2d_1() {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    //  ∂²ϕ     ∂²ϕ
    //  ———  +  ——— = 0
    //  ∂x²     ∂y²
    //
    // on a [-1,1] × [-1,1] square with the following
    // essential (Dirichlet) boundary conditions:
    //
    // left:    ϕ(-1.0, y) = 1.0
    // right:   ϕ( 1.0, y) = 2.0
    // bottom:  ϕ(x, -1.0) = 1.0
    // top:     ϕ(x,  1.0) = 2.0

    // allocate the grid
    let grid = Grid2d::new_uniform(-1.0, 1.0, -1.0, 1.0, 4, 4).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(&grid, Side::Xmin, |_, _| 1.0);
    ebcs.set(&grid, Side::Xmax, |_, _| 2.0);
    ebcs.set(&grid, Side::Ymin, |_, _| 1.0);
    ebcs.set(&grid, Side::Ymax, |_, _| 2.0);

    // allocate the Laplacian operator
    let (kx, ky) = (1.0, 1.0);
    let fdm = FdmLaplacian2d::new(grid, ebcs, kx, ky).unwrap();

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk_bar, kk_check) = fdm.get_matrices_sps(0, Sym::No);
    let (mut a_bar, a_check, mut f_bar) = fdm.get_vectors_sps(|_, _| 0.0);
    let kk_check = kk_check.unwrap();

    // set the right-hand side (note that f = 0)
    kk_check.mat_vec_mul(&mut f_bar, -1.0, &a_check).unwrap(); // f̄ -= Ǩ ǎ

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk_bar, None).unwrap();
    solver.actual.solve(&mut a_bar, &f_bar, false).unwrap();

    // results
    let a = fdm.get_joined_vector_sps(&a_bar, &a_check);

    // check
    let a_correct = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.25, 1.5, 2.0, 1.0, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    vec_approx_eq(&a, &a_correct, 1e-15);
}
