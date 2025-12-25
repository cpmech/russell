use plotpy::{Contour, Curve, Plot, Surface};
use russell_lab::approx_eq;
use russell_pde::{Fdm2d, Grid2d, ProblemSamples, StrError};
use russell_sparse::Genie;
use serial_test::serial;

const SAVE_FIGURE: bool = false;

#[test]
fn test_2d_prob03_fdm() -> Result<(), StrError> {
    for nd_tol in &[
        (11, 1.0055e-1), //
                         // (101, 1.043e-3), //
    ] {
        let (nd, tol) = *nd_tol;
        // SPS
        run(true, true, false, nd, tol, false)?;
        run(true, false, false, nd, tol, false)?;
        run(false, true, false, nd, tol, false)?;
        run(false, false, false, nd, tol, false)?;
        // LMM
        run(true, true, true, nd, tol, false)?;
        run(true, false, true, nd, tol, false)?;
        run(false, true, true, nd, tol, false)?;
        run(false, false, true, nd, tol, false)?;
    }
    Ok(())
}

#[cfg(feature = "with_mumps")]
#[test]
#[serial]
fn test_2d_prob03_fdm_mumps_sym() -> Result<(), StrError> {
    for nd_tol in &[
        (11, 1.0055e-1), //
                         // (101, 1.043e-3), //
    ] {
        let (nd, tol) = *nd_tol;
        // SPS
        run(true, true, false, nd, tol, true)?;
        // run(true, false, false, nd, tol, true)?;
        // run(false, true, false, nd, tol, true)?;
        // run(false, false, false, nd, tol, true)?;
        // LMM
        run(true, true, true, nd, tol, true)?;
        // run(true, false, true, nd, tol, true)?;
        // run(false, true, true, nd, tol, true)?;
        // run(false, false, true, nd, tol, true)?;
    }
    Ok(())
}

fn run(case_a: bool, helmholtz: bool, lmm: bool, nd: usize, tol: f64, mumps_sym: bool) -> Result<(), StrError> {
    // get the problem data
    let alpha = if helmholtz { 1.0 } else { 0.0 };
    let (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical) =
        ProblemSamples::d2_problem_03(1.0, alpha, case_a);

    // allocate the grid
    let (nx, ny) = (nd, nd);
    let grid = Grid2d::new_uniform(xmin, xmax, ymin, ymax, nx, ny)?;

    // allocate the solver
    let mut fdm = Fdm2d::new(grid, ebcs, nbcs, kx, ky)?;
    if mumps_sym {
        fdm.set_solver_options(Genie::Mumps, true);
    }

    // solve the problem
    let a = if lmm {
        fdm.solve_lmm(alpha, &source)?
    } else {
        fdm.solve_sps(alpha, &source)?
    };

    // check
    let mut err_max = 0.0;
    fdm.for_each_coord(|m, x, y| {
        let err = f64::abs(a[m] - analytical(x, y));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x, y), tol);
    });
    println!("max(err) = {:>10.5e}", err_max);

    // plot results
    if SAVE_FIGURE {
        let fn_a = "/tmp/russell_pde/test_2d_prob03_fdm_a.svg";
        let fn_b = "/tmp/russell_pde/test_2d_prob03_fdm_b.svg";
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
        fdm.for_each_coord(|m, x, y| {
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
            .set_zrange(-1.0, 1.0)
            .set_figure_size_points(600.0, 600.0)
            .save(&fn_b)?;
    }
    Ok(())
}
