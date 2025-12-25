use russell_lab::approx_eq;
use russell_pde::{ProblemSamples, Spc1d, StrError};

#[test]
fn test_1d_prob04_spc_sps() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_04();

    // allocate the solver
    let nx = 7;
    let spc = Spc1d::new(xmin, xmax, nx, ebcs, nbcs, kx)?;

    // solve the problem
    let a = spc.solve_sps(0.0, source)?;

    // analytical solution
    let mut err_max = 0.0;
    spc.for_each_coord(|m, x| {
        let err = f64::abs(a[m] - analytical(x));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x), 8.48e-2);
    });
    println!("N = {}, max(err) = {:>10.5e}", nx - 1, err_max);
    Ok(())
}
