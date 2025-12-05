use russell_lab::vec_approx_eq;
use russell_pde::{EssentialBcs1d, FdmLaplacian1d, Grid1d};
use russell_sparse::{Genie, LinSolver, Sym};

#[test]
fn test_laplace1d_1() {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    //   ∂²ϕ
    // - ——— = x
    //   ∂x²
    //
    // on a unit interval with homogeneous boundary conditions

    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, 1.0, 5).unwrap();

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set_homogeneous(&grid);

    // allocate the Laplacian operator
    // (note that we have to use negative kx)
    let kx = 1.0;
    let fdm = FdmLaplacian1d::new(grid, ebcs, -kx).unwrap();

    // solving K u = F from:
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ K   C │ │ u │   │ f │
    // │       │ │   │ = │   │
    // │ c   k │ │ p │   │ g │
    // └       ┘ └   ┘   └   ┘
    // where F = f - C p = f because p = 0 (homogeneous EBCs)

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk, _) = fdm.get_matrices_sps(0, Sym::No);
    let (mut u, p, ff) = fdm.get_vectors(|x| x);

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk, None).unwrap();
    solver.actual.solve(&mut u, &ff, false).unwrap();

    // results: a = (u, p)
    let a = fdm.get_composed_vector(&u, &p);

    // analytical solution
    let analytical = |x| (x - f64::powi(x, 3)) / 6.0;
    fdm.loop_over_grid_points(|m, x| {
        println!("{}: 128 ϕ = {} ({})", m, 128.0 * a[m], 128.0 * analytical(x));
    });

    // check
    let correct = [0.0, 5.0 / 128.0, 8.0 / 128.0, 7.0 / 128.0, 0.0];
    vec_approx_eq(&a, &correct, 1e-15);
}
