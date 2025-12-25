use russell_lab::approx_eq;
use russell_pde::{Fdm1d, Grid1d, ProblemSamples, StrError};
use russell_sparse::Genie;
use serial_test::serial;

#[test]
fn test_1d_prob01_fdm_sps_umfpack() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 5)?;

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // solve the problem
    let a = fdm.solve_sps(0.0, source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), 1e-15);
    });
    Ok(())
}

#[test]
fn test_1d_prob01_fdm_lmm_umfpack() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 5)?;

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // solve the problem
    let a = fdm.solve_lmm(0.0, source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), 1e-15);
    });
    Ok(())
}

#[test]
fn test_1d_prob01_fdm_sps_umfpack_symmetric() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 5)?;

    // allocate the solver
    let mut fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;
    fdm.set_solver_options(Genie::Umfpack, true);

    // solve the problem
    let a = fdm.solve_sps(0.0, source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), 1e-15);
    });
    Ok(())
}

#[test]
fn test_1d_prob01_fdm_lmm_umfpack_symmetric() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 5)?;

    // allocate the solver
    let mut fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;
    fdm.set_solver_options(Genie::Umfpack, true);

    // solve the problem
    let a = fdm.solve_lmm(0.0, source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), 1e-15);
    });
    Ok(())
}

#[cfg(feature = "with_mumps")]
#[test]
#[serial]
fn test_1d_prob01_fdm_sps_mumps() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 5)?;

    // allocate the solver
    let mut fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;
    fdm.set_solver_options(Genie::Mumps, false);

    // solve the problem
    let a = fdm.solve_sps(0.0, source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), 1e-15);
    });
    Ok(())
}

#[cfg(feature = "with_mumps")]
#[test]
#[serial]
fn test_1d_prob01_fdm_lmm_mumps() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 5)?;

    // allocate the solver
    let mut fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;
    fdm.set_solver_options(Genie::Mumps, false);

    // solve the problem
    let a = fdm.solve_lmm(0.0, source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), 1e-15);
    });
    Ok(())
}

#[cfg(feature = "with_mumps")]
#[test]
#[serial]
fn test_1d_prob01_fdm_sps_mumps_symmetric() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 5)?;

    // allocate the solver
    let mut fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;
    fdm.set_solver_options(Genie::Mumps, true);

    // solve the problem
    let a = fdm.solve_sps(0.0, source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), 1e-15);
    });
    Ok(())
}

#[cfg(feature = "with_mumps")]
#[test]
#[serial]
fn test_1d_prob01_fdm_lmm_mumps_symmetric() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 5)?;

    // allocate the solver
    let mut fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;
    fdm.set_solver_options(Genie::Mumps, true);

    // solve the problem
    let a = fdm.solve_lmm(0.0, source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), 1e-15);
    });
    Ok(())
}
