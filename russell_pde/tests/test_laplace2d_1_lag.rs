use russell_lab::array_approx_eq;
use russell_pde::{EssentialBcs2d, FdmLaplacian2d, Grid2d, Side};
use russell_sparse::{Genie, LinSolver};

#[test]
fn test_laplace2d_1_lag() {
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
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(&grid, Side::Xmin, |_, _| 1.0);
    ebcs.set(&grid, Side::Xmax, |_, _| 2.0);
    ebcs.set(&grid, Side::Ymin, |_, _| 1.0);
    ebcs.set(&grid, Side::Ymax, |_, _| 2.0);

    // allocate the Laplacian operator
    let (kx, ky) = (1.0, 1.0);
    let fdm = FdmLaplacian2d::new(grid, ebcs, kx, ky).unwrap();

    // solving:
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ M  Eᵀ │ │ a │   │ r │
    // │       │ │   │ = │   │
    // │ E  0  │ │ w │   │ ū │
    // └       ┘ └   ┘   └   ┘
    //     A       x       b
    // where a = (u, p) and w are the Lagrange multipliers

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (aa, _) = fdm.get_aa_and_ee_matrices(0, false);
    let (mut x, b) = fdm.get_vectors_lmm(|_, _| 0.0);

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&aa, None).unwrap();
    solver.actual.solve(&mut x, &b, false).unwrap();

    // results
    let na = fdm.get_info().2;
    let a = &x.as_data()[..na];

    // check
    let a_correct = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.25, 1.5, 2.0, 1.0, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    array_approx_eq(a, &a_correct, 1e-15);
}
