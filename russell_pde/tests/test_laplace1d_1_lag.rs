use russell_lab::{array_approx_eq, Vector};
use russell_pde::FdmLaplacian1d;
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

    // allocate the Laplacian operator
    // (note that we have to use negative kx)
    let mut fdm = FdmLaplacian1d::new(-1.0, 0.0, 1.0, 5, None).unwrap();

    // set essential boundary conditions
    fdm.set_homogeneous_boundary_conditions();

    // compute the augmented coefficient matrix for the Lagrange multipliers method
    // ┌       ┐ ┌   ┐   ┌   ┐
    // │ K  Eᵀ │ │ u │   │ f │
    // │       │ │   │ = │   │
    // │ E  0  │ │ w │   │ ū │
    // └       ┘ └   ┘   └   ┘
    //     A      lhs     rhs
    let aa = fdm.augmented_coefficient_matrix(0).unwrap();

    // allocate the left- and right-hand side vectors
    let np = fdm.num_prescribed();
    let dim = fdm.dim();
    let mut lhs = Vector::new(dim + np);
    let mut rhs = Vector::new(dim + np);

    // add the source term to the right-hand side vector
    fdm.loop_over_grid_points(|m, x| {
        rhs[m] = x;
    });

    // add the prescribed values to the right-hand side vector
    fdm.loop_over_prescribed_values(|ip, _, value| {
        rhs[dim + ip] = value;
    });

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&aa, None).unwrap();
    solver.actual.solve(&mut lhs, &rhs, false).unwrap();

    // results
    let ana_phi = |x| (x - f64::powi(x, 3)) / 6.0;
    fdm.loop_over_grid_points(|m, x| {
        println!("{}: 128 ϕ = {} ({})", m, 128.0 * lhs[m], 128.0 * ana_phi(x));
    });

    // check
    let correct = [0.0, 5.0 / 128.0, 8.0 / 128.0, 7.0 / 128.0, 0.0];
    array_approx_eq(&lhs.as_data()[..dim], &correct, 1e-15);
}
