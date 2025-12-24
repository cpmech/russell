use russell_lab::approx_eq;
use russell_pde::{ProblemSamples, Spc2d, SpcMap2d, StrError, TransfiniteSamples};

// Table 5.1 (Chebyshev row), page 171, Kopriva's book
//  N   tol  log10(max(err))
//  8  1e-1  -1.5375
// 12  1e-3  -3.8044
// 16  1e-6  -6.6535
// 20  1e-9  -9.8774

#[test]
fn test_2d_prob07_spc() -> Result<(), StrError> {
    // parameters
    let (nn, tol, l10) = (8, 1e-1, -1.5375);
    // let (nn, tol, l10) = (12, 1e-3, -3.8044);
    // let (nn, tol, l10) = (16, 1e-6, -6.6535);
    // let (nn, tol, l10) = (20, 1e-9, -9.8774);

    // get the problem data
    let (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical) = ProblemSamples::d2_problem_07();

    // allocate the solver
    let (nx, ny) = (nn + 1, nn + 1);
    let spc = Spc2d::new(xmin, xmax, ymin, ymax, nx, ny, ebcs, nbcs, kx, ky)?;

    // solve the problem
    let a = spc.solve_poisson_sps(&source)?;

    // check
    let mut err_max = 0.0;
    spc.for_each_coord(|m, x, y| {
        let err = f64::abs(a[m] - analytical(x, y));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x, y), tol);
    });
    println!("max(err) = {:>10.5e}", err_max);
    println!("log10(max(err)) = {:>10.5}", f64::log10(err_max));
    approx_eq(f64::log10(err_max), l10, 1e-4);

    Ok(())
}

#[test]
fn test_2d_prob05_spc_map() -> Result<(), StrError> {
    // parameters
    let (nn, tol, l10) = (8, 1e-1, -1.5375);
    // let (nn, tol, l10) = (12, 1e-3, -3.8044);
    // let (nn, tol, l10) = (16, 1e-6, -6.6535);
    // let (nn, tol, l10) = (20, 1e-9, -9.8774);

    // get the problem data
    let (xmin, xmax, ymin, ymax, k, _, ebcs, nbcs, source, analytical) = ProblemSamples::d2_problem_07();

    // transfinite map
    let map = TransfiniteSamples::quadrilateral_2d(&[xmin, ymin], &[xmax, ymin], &[xmax, ymax], &[xmin, ymax]);

    // allocate the solver
    let (nr, ns) = (nn + 1, nn + 1);
    let mut spc = SpcMap2d::new(map, nr, ns, ebcs, nbcs, k)?;

    // solve the problem
    let a = spc.solve_poisson_sps(&source)?;

    // check
    let mut err_max = 0.0;
    spc.for_each_coord(|m, x, y| {
        let err = f64::abs(a[m] - analytical(x, y));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x, y), tol);
    });
    println!("max(err) = {:>10.5e}", err_max);
    println!("log10(max(err)) = {:>10.5}", f64::log10(err_max));
    approx_eq(f64::log10(err_max), l10, 1e-4);
    Ok(())
}
