use russell_lab::vec_approx_eq;
use russell_pde::{EssentialBcs1d, Fdm1d, Grid1d, NaturalBcs1d};

// Solve the following problem:
//
//   ∂²ϕ
// - ——— = x
//   ∂x²
//
// on a unit interval with homogeneous boundary conditions
//
// The analytical solution is:
//
//        x - x³
// ϕ(x) = ——————
//          6

#[test]
fn test_1d_prob01_fdm_sps() -> Result<(), String> {
    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, 1.0, 5)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set_homogeneous();

    // natural boundary conditions
    let nbcs = NaturalBcs1d::new();

    // allocate the solver
    let kx = 1.0;
    let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // solve the problem
    let a = fdm.solve(|x| x)?;

    // check
    let correct = [0.0, 5.0 / 128.0, 8.0 / 128.0, 7.0 / 128.0, 0.0];
    vec_approx_eq(&a, &correct, 1e-15);
    Ok(())
}

#[test]
fn test_1d_prob01_fdm_lmm() -> Result<(), String> {
    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, 1.0, 5)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set_homogeneous();

    // natural boundary conditions
    let nbcs = NaturalBcs1d::new();

    // allocate the solver
    let kx = 1.0;
    let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // solve the problem
    let a = fdm.solve_lmm(|x| x)?;

    // check
    let correct = [0.0, 5.0 / 128.0, 8.0 / 128.0, 7.0 / 128.0, 0.0];
    vec_approx_eq(&a, &correct, 1e-15);
    Ok(())
}
