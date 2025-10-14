use russell_lab::vec_approx_eq;
use russell_pde::{EssentialBcs2d, FdmLaplacian2dNew, Grid2d, Side};
use russell_sparse::{Genie, LinSolver, Sym};

#[test]
fn test_laplace2d_1() {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    //  ∂²ϕ     ∂²ϕ
    //  ———  +  ——— = 0
    //  ∂x²     ∂y²
    //
    // on a (3.0 × 3.0) rectangle with the following
    // essential (Dirichlet) boundary conditions:
    //
    // left:    ϕ(0.0, y) = 1.0
    // right:   ϕ(3.0, y) = 2.0
    // bottom:  ϕ(x, 0.0) = 1.0
    // top:     ϕ(x, 3.0) = 2.0

    // allocate the grid
    let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new(grid);
    ebcs.set(Side::Xmin, |_, _| 1.0);
    ebcs.set(Side::Xmax, |_, _| 2.0);
    ebcs.set(Side::Ymin, |_, _| 1.0);
    ebcs.set(Side::Ymax, |_, _| 2.0);

    // allocate the Laplacian operator
    let (kx, ky) = (1.0, 1.0);
    let fdm = FdmLaplacian2dNew::new(ebcs, kx, ky).unwrap();

    // solving K u = h from:
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ K   C │ │ u │   │ f │
    // │       │ │   │ = │   │
    // │ c   k │ │ p │   │ g │
    // └       ┘ └   ┘   └   ┘
    // where h = f - C p

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk, cc_mat) = fdm.get_kk_and_cc_matrices(0, Sym::No);
    let (mut u, p, mut h) = fdm.get_vectors(|_, _| 0.0);
    let cc = cc_mat.unwrap();

    // set the right-hand side (note that f = 0)
    cc.mat_vec_mul(&mut h, -1.0, &p).unwrap(); // h = - C p

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk, None).unwrap();
    solver.actual.solve(&mut u, &h, false).unwrap();

    // results: a = (u, p)
    let a = fdm.get_composed_vector(&u, &p);

    // check
    let a_correct = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.25, 1.5, 2.0, 1.0, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    vec_approx_eq(&a, &a_correct, 1e-15);
}
