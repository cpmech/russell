use russell_lab::approx_eq;
use russell_pde::{ProblemSamples, SpcMap2d, StrError};

// Table 7.1 (Orthogonal column), page 261, Kopriva's book
//   N  max(err)
//   8  1.03e-04
//  12  3.05e-08
//  16  1.02e-11
//  20  3.47e-14

#[test]
fn test_2d_prob08_spc_map() -> Result<(), StrError> {
    for nn_tol in &[
        (8, 1.03e-4), //
                      // (12, 3.05e-8),  //
                      // (16, 1.02e-11), //
                      // (20, 3.47e-14), //
    ] {
        let (nn, tol) = *nn_tol;
        // SPS
        run_spc_map(false, nn, tol)?;
        // LMM
        run_spc_map(true, nn, tol)?;
    }
    Ok(())
}

fn run_spc_map(lmm: bool, nn: usize, tol: f64) -> Result<(), StrError> {
    // get the problem data
    let ra = 1.0;
    let rb = 3.0;
    let (map, k, ebcs, nbcs, source, analytical) = ProblemSamples::d2_problem_08(ra, rb);

    // allocate the solver
    let (nr, ns) = (nn + 1, nn + 1);
    let mut spc = SpcMap2d::new(map, nr, ns, ebcs, nbcs, k)?;

    // solve the problem
    let a = if lmm {
        spc.solve_lmm(0.0, &source)?
    } else {
        spc.solve_sps(0.0, &source)?
    };

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
    Ok(())
}
