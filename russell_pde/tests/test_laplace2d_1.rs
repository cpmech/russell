use russell_lab::{vec_approx_eq, Vector};
use russell_pde::{FdmLaplacian2d, Side};
use russell_sparse::{Genie, LinSolver};

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

    // allocate the Laplacian operator
    let mut fdm = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

    // set essential boundary conditions
    fdm.set_essential_boundary_condition(Side::Xmin, |_, _| 1.0);
    fdm.set_essential_boundary_condition(Side::Xmax, |_, _| 2.0);
    fdm.set_essential_boundary_condition(Side::Ymin, |_, _| 1.0);
    fdm.set_essential_boundary_condition(Side::Ymax, |_, _| 2.0);

    // compute the augmented coefficient matrix and the correction matrix
    let (aa, cc) = fdm.coefficient_matrix().unwrap();

    // allocate the left- and right-hand side vectors
    let dim = fdm.dim();
    let mut phi = Vector::new(dim);
    let mut rhs = Vector::new(dim);

    // set the 'prescribed' part of the left-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        phi[i] = value;
    });

    // initialize the right-hand side vector with the correction
    cc.mat_vec_mul(&mut rhs, -1.0, &phi).unwrap(); // f1 := -K12⋅u2

    // set the 'prescribed' part of the right-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        rhs[i] = value; // f2 := ebc
    });

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&aa, None).unwrap();
    solver.actual.solve(&mut phi, &rhs, false).unwrap();

    // check
    let x_correct = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.25, 1.5, 2.0, 1.0, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    vec_approx_eq(&phi, &x_correct, 1e-15);
}
