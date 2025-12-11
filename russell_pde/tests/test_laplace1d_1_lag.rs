use russell_lab::array_approx_eq;
use russell_pde::{EssentialBcs1d, FdmLaplacian1d, Grid1d};
use russell_sparse::{Genie, LinSolver};

#[test]
fn test_laplace1d_1_lag() {
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

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (mm, _) = fdm.get_matrices_lmm(0, false);
    let (mut aa, ff) = fdm.get_vectors_lmm(|x| x);

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&mm, None).unwrap();
    solver.actual.solve(&mut aa, &ff, false).unwrap();

    // results
    let neq = fdm.get_dims_lmm().0;
    let a = &aa.as_data()[..neq];

    // analytical solution
    let analytical = |x| (x - f64::powi(x, 3)) / 6.0;
    fdm.for_each_coord(|m, x| {
        println!("{}: 128 ϕ = {} ({})", m, 128.0 * a[m], 128.0 * analytical(x));
    });

    // check
    let correct = [0.0, 5.0 / 128.0, 8.0 / 128.0, 7.0 / 128.0, 0.0];
    array_approx_eq(a, &correct, 1e-15);
}
