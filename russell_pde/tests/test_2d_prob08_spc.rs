use plotpy::{Contour, Plot};
use russell_lab::approx_eq;
use russell_pde::{ProblemSamples, SpcMap2d, StrError};

const SAVE_FIGURE: bool = false;

// Table 7.1 (Orthogonal column), page 261, Kopriva's book
//   N  max(err)  max(err)
//        ring     lozenge
//   8  1.03e-04  2.10e-03
//  12  3.05e-08  4.75e-05
//  16  1.02e-11  8.90e-07
//  20  3.47e-14  1.80e-08
//  24    N/A     5.00e-10
//  28    N/A     2.50e-11
//  32    N/A     1.20e-12
//  36    N/A     5.00e-14

#[test]
fn test_2d_prob08_spc_map() -> Result<(), StrError> {
    // quarter ring domain
    for nn_tol in &[
        (8, 1.03e-4), //
                      // (12, 3.05e-8),  //
                      // (16, 1.02e-11), //
                      // (20, 3.47e-14), //
    ] {
        let (nn, tol) = *nn_tol;
        // SPS
        run_spc_map(false, nn, tol, false)?;
        // LMM
        run_spc_map(true, nn, tol, false)?;
    }

    // quarter perforated lozenge domain
    for nn_tol in &[
        (8, 2.24e-3), //
                      // (12, 4.77e-5),  //
                      // (16, 8.90e-7),  //
                      // (20, 2.10e-8),  //
                      // (24, 9.90e-10), //
                      // (28, 4.52e-11), //
                      // (32, 1.94e-12), //
                      // (36, 7.91e-14), //
    ] {
        let (nn, tol) = *nn_tol;
        // SPS
        run_spc_map(false, nn, tol, true)?;
        // LMM
        run_spc_map(true, nn, tol, true)?;
    }
    Ok(())
}

fn run_spc_map(lmm: bool, nn: usize, tol: f64, lozenge: bool) -> Result<(), StrError> {
    // get the problem data
    let ra = 1.0;
    let rb = 3.0;
    let (map, k, ebcs, nbcs, source, analytical) = ProblemSamples::d2_problem_08(ra, rb, lozenge);

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
    println!("N = {} max(err) = {:>10.5e}", nn, err_max);

    // plot contour
    if SAVE_FIGURE {
        let key = if lozenge { "lozenge" } else { "ring" };
        let (xx_num, yy_num, tri_num) = spc.get_map().triangulate(nr, ns, true, true);
        let mut zz_num = vec![0.0; xx_num.len()];
        spc.for_each_coord(|m, x, y| {
            approx_eq(x, xx_num[m], 1.0e-15);
            approx_eq(y, yy_num[m], 1.0e-15);
            zz_num[m] = a[m];
        });
        let n_ana = 41;
        let (xx_ana, yy_ana, tri_ana) = spc.get_map().triangulate(n_ana, n_ana, false, false);
        let mut zz = vec![0.0; xx_ana.len()];
        for m in 0..xx_ana.len() {
            zz[m] = analytical(xx_ana[m], yy_ana[m]);
        }
        let levels = &[-1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25];
        let mut contour_ana = Contour::new();
        contour_ana
            .set_tri_show_edges(false)
            .set_no_lines(false)
            .set_levels(levels)
            .draw_tri(&xx_ana, &yy_ana, &zz, &tri_ana);
        let mut contour_num = Contour::new();
        contour_num
            .set_tri_show_edges(false)
            .set_no_colorbar(true)
            .set_no_fill(true)
            .set_no_lines(false)
            .set_line_color("#30ff45ff")
            .set_fontsize_labels(14.0)
            .set_levels(levels)
            .draw_tri(&xx_num, &yy_num, &zz_num, &tri_num);
        let mut plot = Plot::new();
        plot.add(&contour_ana)
            .add(&contour_num)
            .set_range(0.0, rb, 0.0, rb)
            .set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save(&format!("/tmp/russell_pde/test_2d_prob08_spc_map_{}.svg", key))?;
    }
    Ok(())
}
