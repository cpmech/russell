use russell_lab::approx_eq;
use russell_pde::{Fdm1d, Grid1d, ProblemSamples, StrError};

#[test]
fn test_1d_prob01_fdm_sps() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 5)?;

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // solve the problem
    let a = fdm.solve(source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), 1e-15);
    });
    Ok(())
}

#[test]
fn test_1d_prob01_fdm_lmm() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 5)?;

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // solve the problem
    let a = fdm.solve_lmm(source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), 1e-15);
    });
    Ok(())
}
