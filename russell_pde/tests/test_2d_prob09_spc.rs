use plotpy::{Contour, Plot};
use russell_lab::approx_eq;
use russell_pde::{ProblemSamples, SpcMap2d, StrError};

const SAVE_FIGURE: bool = false;

// Figure 7.6 on page 266, Kopriva's book
//  N   Log10(max(err))
// 10        -2
// 15        -3
// 20        -4
// 25        -5
// 30        -6

#[test]
fn test_2d_prob09_spc_map() -> Result<(), StrError> {
    // quarter ring domain
    for nn_tol in &[
        (10, 2.22e-2), //
                       // (15, 2.62e-3), //
                       // (20, 3.21e-4), //
                       // (25, 3.89e-5), //
                       // (30, 4.64e-6), //
    ] {
        let (nn, tol) = *nn_tol;
        // SPS
        run_spc_map(false, nn, tol)?;
        // LMM
        // run_spc_map(true, nn, tol, false)?;
    }
    Ok(())
}

fn run_spc_map(lmm: bool, nn: usize, tol: f64) -> Result<(), StrError> {
    // get the problem data
    let ra = 0.5;
    let rb = 10.0;
    let v_inf = 0.5;
    let (map, k, ebcs, nbcs, source, analytical) = ProblemSamples::d2_problem_09(ra, rb, v_inf);

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
        let mut contour_ana = Contour::new();
        contour_ana
            .set_tri_show_edges(false)
            .set_no_lines(false)
            .draw_tri(&xx_ana, &yy_ana, &zz, &tri_ana);
        let mut contour_num = Contour::new();
        contour_num
            .set_tri_show_edges(false)
            .set_no_colorbar(true)
            .set_no_fill(true)
            .set_no_lines(false)
            .set_line_color("#30ff45ff")
            .set_fontsize_labels(14.0)
            .draw_tri(&xx_num, &yy_num, &zz_num, &tri_num);
        let mut plot = Plot::new();
        plot.add(&contour_ana)
            .add(&contour_num)
            .set_range(-rb, rb, 0.0, rb)
            .set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save("/tmp/russell_pde/test_2d_prob09_spc_map.svg")?;
    }
    Ok(())
}
