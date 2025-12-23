use plotpy::{Contour, Curve, Plot, Surface};
use russell_lab::approx_eq;
use russell_pde::{ProblemSamples, Spc2d, SpcMap2d, StrError, TransfiniteSamples};

const SAVE_FIGURE: bool = false;

#[test]
fn test_2d_prob03_spc() -> Result<(), StrError> {
    for nn_tol in &[
        (8, 4.2e-2), //
                     // (12, 1.0e-5), //
                     // (15, 1e-8),   //
                     // (20, 1e-13),  //
    ] {
        let (nn, tol) = *nn_tol;
        // SPS
        run_spc(true, true, false, nn, tol)?;
        run_spc(true, false, false, nn, tol)?;
        run_spc(false, true, false, nn, tol)?;
        run_spc(false, false, false, nn, tol)?;
        // LMM
        run_spc(true, false, true, nn, tol)?;
        run_spc(false, false, true, nn, tol)?;
    }
    Ok(())
}

#[test]
fn test_2d_prob03_spc_map() -> Result<(), StrError> {
    for nn_tol in &[
        (8, 4.2e-2), //
                     // (12, 1.0e-5), //
                     // (15, 1e-8),   //
                     // (20, 1e-13),  //
    ] {
        let (nn, tol) = *nn_tol;
        run_spc_map(true, true, nn, tol)?;
        run_spc_map(true, false, nn, tol)?;
        run_spc_map(false, true, nn, tol)?;
        run_spc_map(false, false, nn, tol)?;
    }
    Ok(())
}

fn run_spc(case_a: bool, helmholtz: bool, lmm: bool, nn: usize, tol: f64) -> Result<(), StrError> {
    // get the problem data
    let alpha = if helmholtz { 1.0 } else { 0.0 };
    let (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical) =
        ProblemSamples::d2_problem_03(1.0, alpha, case_a);

    // allocate the solver
    let (nx, ny) = (nn + 1, nn + 1);
    let spc = Spc2d::new(xmin, xmax, ymin, ymax, nx, ny, ebcs, nbcs, kx, ky)?;

    // solve the problem
    let a = if helmholtz {
        spc.solve_helmholtz_sps(alpha, &source)?
    } else {
        if lmm {
            spc.solve_poisson_lmm(&source)?
        } else {
            spc.solve_poisson_sps(&source)?
        }
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

    // plot results
    if SAVE_FIGURE {
        let case = if case_a { "a" } else { "b" };
        let prob = if helmholtz { "hz" } else { "ps" };
        let fn_a = format!("/tmp/russell_pde/test_2d_prob03_spc_{}_{}_contour.svg", case, prob);
        let fn_b = format!("/tmp/russell_pde/test_2d_prob03_spc_{}_{}_surface.svg", case, prob);
        let mut points = Curve::new();
        let mut surf_num = Surface::new();
        let mut surf_ana = Surface::new();
        let mut contour_num = Contour::new();
        let mut contour_ana = Contour::new();
        let mut xx = vec![vec![0.0; nx]; ny];
        let mut yy = vec![vec![0.0; nx]; ny];
        let mut zz_num = vec![vec![0.0; nx]; ny];
        let mut zz_ana = vec![vec![0.0; nx]; ny];
        let mut xx_serial = Vec::with_capacity(nx * ny);
        let mut yy_serial = Vec::with_capacity(nx * ny);
        spc.for_each_coord(|m, x, y| {
            let row = m / nx;
            let col = m % nx;
            xx[row][col] = x;
            yy[row][col] = y;
            zz_num[row][col] = a[m];
            zz_ana[row][col] = analytical(x, y);
            xx_serial.push(x);
            yy_serial.push(y);
        });
        points
            .set_line_style("None")
            .set_marker_size(2.0)
            .set_marker_style(".")
            .set_marker_color("red")
            .set_marker_line_color("red")
            .draw(&xx_serial, &yy_serial);
        contour_num
            .set_colors(&["None"])
            .set_no_colorbar(true)
            .set_line_color("black")
            .set_line_width(5.0)
            .set_line_style("-")
            .draw(&xx, &yy, &zz_num);
        contour_ana
            .set_colors(&["None"])
            .set_no_colorbar(true)
            .set_no_labels(true)
            .set_line_color("orange")
            .set_line_style("-")
            .draw(&xx, &yy, &zz_ana);
        surf_num
            .set_with_surface(false)
            .set_with_wireframe(false)
            .set_point_style(".")
            .set_with_points(true)
            .set_point_color("black")
            .set_point_size(70.0)
            .set_wire_line_width(1.0)
            .set_wire_line_color("black")
            .draw(&xx, &yy, &zz_num);
        surf_ana
            .set_with_surface(false)
            .set_with_wireframe(true)
            .set_wire_line_width(2.0)
            .set_wire_line_color("orange")
            .draw(&xx, &yy, &zz_ana);
        let mut plot = Plot::new();
        plot.add(&contour_num)
            .add(&contour_ana)
            .add(&points)
            .set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save(&fn_a)
            .unwrap();
        plot.add(&surf_ana)
            .add(&surf_num)
            .set_figure_size_points(600.0, 600.0)
            .save(&fn_b)?;
    }
    Ok(())
}

fn run_spc_map(case_a: bool, helmholtz: bool, nn: usize, tol: f64) -> Result<(), StrError> {
    // get the problem data
    let alpha = if helmholtz { 1.0 } else { 0.0 };
    let (xmin, xmax, ymin, ymax, k, _, ebcs, nbcs, source, analytical) =
        ProblemSamples::d2_problem_03(1.0, alpha, case_a);

    // transfinite map
    let map = TransfiniteSamples::quadrilateral_2d(&[xmin, ymin], &[xmax, ymin], &[xmax, ymax], &[xmin, ymax]);

    // allocate the solver
    let (nx, ny) = (nn + 1, nn + 1);
    let mut spc = SpcMap2d::new(map, nx, ny, ebcs, nbcs, k)?;

    // solve the problem
    let a = if helmholtz {
        spc.solve_hz(alpha, &source)?
    } else {
        spc.solve(&source)?
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

    // plot results
    if SAVE_FIGURE {
        let case = if case_a { "a" } else { "b" };
        let prob = if helmholtz { "hz" } else { "ps" };
        let fn_a = format!("/tmp/russell_pde/test_2d_prob03_spc_map_{}_{}_contour.svg", case, prob);
        let fn_b = format!("/tmp/russell_pde/test_2d_prob03_spc_map_{}_{}_surface.svg", case, prob);
        let mut points = Curve::new();
        let mut surf_num = Surface::new();
        let mut surf_ana = Surface::new();
        let mut contour_num = Contour::new();
        let mut contour_ana = Contour::new();
        let mut xx = vec![vec![0.0; nx]; ny];
        let mut yy = vec![vec![0.0; nx]; ny];
        let mut zz_num = vec![vec![0.0; nx]; ny];
        let mut zz_ana = vec![vec![0.0; nx]; ny];
        let mut xx_serial = Vec::with_capacity(nx * ny);
        let mut yy_serial = Vec::with_capacity(nx * ny);
        spc.for_each_coord(|m, x, y| {
            let row = m / nx;
            let col = m % nx;
            xx[row][col] = x;
            yy[row][col] = y;
            zz_num[row][col] = a[m];
            zz_ana[row][col] = analytical(x, y);
            xx_serial.push(x);
            yy_serial.push(y);
        });
        points
            .set_line_style("None")
            .set_marker_size(2.0)
            .set_marker_style(".")
            .set_marker_color("red")
            .set_marker_line_color("red")
            .draw(&xx_serial, &yy_serial);
        contour_num
            .set_colors(&["None"])
            .set_no_colorbar(true)
            .set_line_color("black")
            .set_line_width(5.0)
            .set_line_style("-")
            .draw(&xx, &yy, &zz_num);
        contour_ana
            .set_colors(&["None"])
            .set_no_colorbar(true)
            .set_no_labels(true)
            .set_line_color("orange")
            .set_line_style("-")
            .draw(&xx, &yy, &zz_ana);
        surf_num
            .set_with_surface(false)
            .set_with_wireframe(false)
            .set_point_style(".")
            .set_with_points(true)
            .set_point_color("black")
            .set_point_size(70.0)
            .set_wire_line_width(1.0)
            .set_wire_line_color("black")
            .draw(&xx, &yy, &zz_num);
        surf_ana
            .set_with_surface(false)
            .set_with_wireframe(true)
            .set_wire_line_width(2.0)
            .set_wire_line_color("orange")
            .draw(&xx, &yy, &zz_ana);
        let mut plot = Plot::new();
        plot.add(&contour_num)
            .add(&contour_ana)
            .add(&points)
            .set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save(&fn_a)
            .unwrap();
        plot.add(&surf_ana)
            .add(&surf_num)
            .set_figure_size_points(600.0, 600.0)
            .save(&fn_b)?;
    }
    Ok(())
}
