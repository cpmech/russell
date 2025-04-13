use russell_lab::{vec_approx_eq, Vector};
use russell_pde::FdmLaplacian1d;
use russell_sparse::{Genie, LinSolver};

#[test]
fn test_laplace1d_1() {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    //    ∂²ϕ
    //  - ——— = x
    //    ∂x²
    //
    // on a unit interval with homogeneous boundary conditions

    // allocate the Laplacian operator
    let mut fdm = FdmLaplacian1d::new(-1.0, 0.0, 1.0, 5).unwrap();

    // set essential boundary conditions
    fdm.set_homogeneous_boundary_conditions();

    // compute the augmented coefficient matrix and the correction matrix
    let (aa, _) = fdm.coefficient_matrix().unwrap();

    // allocate the left- and right-hand side vectors
    let dim = fdm.dim();
    let mut phi = Vector::new(dim);
    let mut rhs = Vector::new(dim);

    // set the 'prescribed' part of the left-hand side vector with the essential values
    // (this step is not needed with homogeneous boundary conditions)

    // initialize the right-hand side vector with the correction
    // (this step is not needed with homogeneous boundary conditions)

    // set the right-hand side vector with the source term
    fdm.loop_over_grid_points(|i, x| {
        rhs[i] = x;
    });

    // set the 'prescribed' part of the right-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        rhs[i] = value; // bp := ϕp
    });

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&aa, None).unwrap();
    solver.actual.solve(&mut phi, &rhs, false).unwrap();

    // results
    let ana_phi = |x| (x - f64::powi(x, 3)) / 6.0;
    fdm.loop_over_grid_points(|i, x| {
        println!("{}: 128 ϕ = {} ({})", i, 128.0 * phi[i], 128.0 * ana_phi(x));
    });

    // check
    let x_correct = [0.0, 5.0 / 128.0, 8.0 / 128.0, 7.0 / 128.0, 0.0];
    vec_approx_eq(&phi, &x_correct, 1e-15);
}
