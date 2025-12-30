use russell_lab::approx_eq;
use russell_pde::{ProblemSamples, Spc1d, StrError};
use russell_sparse::Genie;
use serial_test::serial;

#[test]
fn test_1d_prob01_spc() -> Result<(), StrError> {
    // SPS
    run_spc(false, 4, 1e-15, Genie::Umfpack)?;
    // LMM
    run_spc(true, 4, 1e-15, Genie::Umfpack)?;
    Ok(())
}

#[cfg(feature = "with_mumps")]
#[test]
#[serial]
fn test_1d_prob01_spc_mumps() -> Result<(), StrError> {
    // SPS
    run_spc(false, 4, 1e-15, Genie::Mumps)?;
    // LMM
    run_spc(true, 4, 1e-15, Genie::Mumps)?;
    Ok(())
}

fn run_spc(lmm: bool, nx: usize, tol: f64, genie: Genie) -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical, ana_flow) = ProblemSamples::d1_problem_01();

    // allocate the solver
    let mut spc = Spc1d::new(xmin, xmax, nx, ebcs, nbcs, kx)?;
    spc.set_solver_options(genie);

    // solve the problem
    let a = if lmm {
        spc.solve_lmm(0.0, source)?
    } else {
        spc.solve_sps(0.0, source)?
    };

    // analytical solution
    let mut err_max = 0.0;
    spc.for_each_coord(|m, x| {
        let err = f64::abs(a[m] - analytical(x));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x), tol);
    });

    // check flow vectors
    let mut flow_err_max = 0.0;
    let wwx = spc.calculate_flow_vectors(&a)?;
    spc.for_each_coord(|m, x| {
        let ana_wx = ana_flow(x);
        approx_eq(wwx[m], ana_wx, tol);
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
