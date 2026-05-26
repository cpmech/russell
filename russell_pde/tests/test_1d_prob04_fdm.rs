use russell_lab::approx_eq;
use russell_pde::{Fdm1d, Grid1d, ProblemSamples, StrError};

#[test]
fn test_1d_prob04_fdm_sps() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_04b();

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, 11)?;

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // solve the problem
    let a = fdm.solve_sps(0.0, source)?;

    // analytical solution
    let mut max_err = 0.0;
    fdm.for_each_coord(|m, x| {
        let diff = f64::abs(a[m] - analytical(x));
        if diff > max_err {
            max_err = diff;
        }
        approx_eq(a[m], analytical(x), 0.177);
    });
    println!("max_err = {:e}", max_err);
    Ok(())
}
