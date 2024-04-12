use russell_lab::{vec_approx_eq, Vector};
use russell_ode::{PdeDiscreteLaplacian2d, Side};
use russell_sparse::{Genie, LinSolver, SparseMatrix};

#[test]
fn test_pde_laplace_1() {
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
    let mut fdm = PdeDiscreteLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

    // set essential boundary conditions
    fdm.set_essential_boundary_condition(Side::Left, |_, _| 1.0);
    fdm.set_essential_boundary_condition(Side::Right, |_, _| 2.0);
    fdm.set_essential_boundary_condition(Side::Bottom, |_, _| 1.0);
    fdm.set_essential_boundary_condition(Side::Top, |_, _| 2.0);

    // compute the augmented coefficient matrix and the correction matrix
    //
    // ┌          ┐ ┌    ┐   ┌             ┐
    // │ Auu   0  │ │ xu │   │ bu - Aup⋅xp │
    // │          │ │    │ = │             │
    // │  0    1  │ │ xp │   │     xp      │
    // └          ┘ └    ┘   └             ┘
    // A := augmented(Auu)
    //
    // ┌          ┐ ┌    ┐   ┌        ┐
    // │  0   Aup │ │ .. │   │ Aup⋅xp │
    // │          │ │    │ = │        │
    // │  0    0  │ │ xp │   │   0    │
    // └          ┘ └    ┘   └        ┘
    // C := augmented(Aup)
    let (aa, cc) = fdm.coefficient_matrix().unwrap();

    // allocate the left- and right-hand side vectors
    let dim = fdm.dim();
    let mut x = Vector::new(dim);
    let mut b = Vector::new(dim);

    // set the 'prescribed' part of the left-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        x[i] = value; // xp := xp
    });

    // initialize the right-hand side vector with the correction
    cc.mat_vec_mul(&mut b, -1.0, &x).unwrap(); // bu := -Aup⋅xp

    // if there were natural (Neumann) boundary conditions,
    // we could set `bu := natural()` here

    // set the 'prescribed' part of the right-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        b[i] = value; // bp := xp
    });

    // solve the linear system
    let mut mat = SparseMatrix::from_coo(aa);
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&mut mat, None).unwrap();
    solver.actual.solve(&mut x, &mut mat, &b, false).unwrap();

    // check
    let x_correct = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.25, 1.5, 2.0, 1.0, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    vec_approx_eq(&x, &x_correct, 1e-15);
}
