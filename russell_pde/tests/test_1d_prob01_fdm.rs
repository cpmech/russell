use russell_lab::approx_eq;
use russell_pde::{Fdm1d, Grid1d, ProblemSamples, StrError};
use russell_sparse::Genie;
use serial_test::serial;

#[test]
fn test_1d_prob01_fdm() -> Result<(), StrError> {
    // SPS
    run_fdm(false, 5, 1e-15, Genie::Umfpack, false)?;
    // LMM
    run_fdm(true, 5, 1e-15, Genie::Umfpack, false)?;
    // SPS symmetric
    run_fdm(false, 5, 1e-15, Genie::Umfpack, true)?;
    // LMM symmetric
    run_fdm(true, 5, 1e-15, Genie::Umfpack, true)?;
    Ok(())
}

#[cfg(feature = "with_mumps")]
#[test]
#[serial]
fn test_1d_prob01_fdm_mumps() -> Result<(), StrError> {
    // SPS
    run_fdm(false, 5, 1e-15, Genie::Mumps, false)?;
    // LMM
    run_fdm(true, 5, 1e-15, Genie::Mumps, false)?;
    // SPS symmetric
    run_fdm(false, 5, 1e-15, Genie::Mumps, true)?;
    // LMM symmetric
    run_fdm(true, 5, 1e-15, Genie::Mumps, true)?;
    Ok(())
}

fn run_fdm(lmm: bool, nx: usize, tol: f64, genie: Genie, symmetric: bool) -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical, _) = ProblemSamples::d1_problem_01();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, nx)?;

    // allocate the solver
    let mut fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;
    fdm.set_solver_options(genie, symmetric);

    // solve the problem
    let a = if lmm {
        fdm.solve_lmm(0.0, source)?
    } else {
        fdm.solve_sps(0.0, source)?
    };

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], analytical(x), tol);
    });
    Ok(())
}
