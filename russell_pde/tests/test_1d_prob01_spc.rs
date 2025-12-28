use russell_lab::approx_eq;
use russell_pde::{ProblemSamples, Spc1d, StrError};

#[test]
fn test_1d_prob01_spc_sps() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical, ana_flow) = ProblemSamples::d1_problem_01();

    // allocate the solver
    let nx = 4;
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
        approx_eq(a[m], analytical(x), 1e-15);
    });

    // check flow vectors
    let mut flow_err_max = 0.0;
    let wwx = spc.calculate_flow_vectors(&a)?;
    spc.for_each_coord(|m, x| {
        let ana_wx = ana_flow(x);
        approx_eq(wwx[m], ana_wx, 1e-15);
        let err_wx = f64::abs(wwx[m] - ana_wx);
        if err_wx > flow_err_max {
            flow_err_max = err_wx;
        }
    });
    let nn = nx - 1;
    println!(
        "N = {:>2}, max(err) = {:>10.5e}, max(flow_err) = {:>10.5e}",
        nn, err_max, flow_err_max
    );
    Ok(())
}

#[test]
fn test_1d_prob01_spc_lmm() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical, ana_flow) = ProblemSamples::d1_problem_01();

    // allocate the solver
    let nx = 4;
    let spc = Spc1d::new(xmin, xmax, nx, ebcs, nbcs, kx)?;

    // solve the problem
    let a = spc.solve_lmm(0.0, source)?;

    // analytical solution
    let mut err_max = 0.0;
    spc.for_each_coord(|m, x| {
        let err = f64::abs(a[m] - analytical(x));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x), 1e-15);
    });

    // check flow vectors
    let mut flow_err_max = 0.0;
    let wwx = spc.calculate_flow_vectors(&a)?;
    spc.for_each_coord(|m, x| {
        let ana_wx = ana_flow(x);
        approx_eq(wwx[m], ana_wx, 1e-15);
        let err_wx = f64::abs(wwx[m] - ana_wx);
        if err_wx > flow_err_max {
            flow_err_max = err_wx;
        }
    });
    let nn = nx - 1;
    println!(
        "N = {:>2}, max(err) = {:>10.5e}, max(flow_err) = {:>10.5e}",
        nn, err_max, flow_err_max
    );
    Ok(())
}
