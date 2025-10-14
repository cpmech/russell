use russell_lab::{array_approx_eq, Vector};
use russell_pde::{EssentialBcs2d, FdmLaplacian2dNew, Grid2d, Side};
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

    // set the essential boundary conditions
    let mut ebcs = EssentialBcs2d::new(&grid);
    ebcs.set(Side::Xmin, |_, _| 1.0);
    ebcs.set(Side::Xmax, |_, _| 2.0);
    ebcs.set(Side::Ymin, |_, _| 1.0);
    ebcs.set(Side::Ymax, |_, _| 2.0);

    // allocate the Laplacian operator
    let (kx, ky) = (1.0, 1.0);
    let fdm = FdmLaplacian2dNew::new(&ebcs, kx, ky).unwrap();

    // Solving:
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ M  Eᵀ │ │ a │   │ r │
    // │       │ │   │ = │   │
    // │ E  0  │ │ w │   │ ū │
    // └       ┘ └   ┘   └   ┘
    //     A       x       b
    // where a = (u, p) and w are the Lagrange multipliers

    // auxiliary variables
    let nu = ebcs.num_unknown();
    let np = ebcs.num_prescribed();
    let na = nu + np; // dimension of a = (u, p)
    let nw = np; // number of Lagrange multipliers
    let nx = na + nw; // dimension of x = (u, p, w)

    // assemble the coefficient matrix
    let (aa, _) = fdm.get_aa_matrix(0, true);

    // allocate the left- and right-hand side vectors
    let mut x = Vector::new(nx);
    let mut b = Vector::new(nx);

    // add the prescribed values to the right-hand side vector
    ebcs.for_each_prescribed_node(|ip, _, _, _, u_bar| {
        b[na + ip] = u_bar;
    });

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&aa, None).unwrap();
    solver.actual.solve(&mut x, &b, false).unwrap();

    // results
    let a = &x.as_data()[..na];
    println!("a = {:?}", a);

    // check
    let a_correct = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.25, 1.5, 2.0, 1.0, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    array_approx_eq(a, &a_correct, 1e-15);
}
