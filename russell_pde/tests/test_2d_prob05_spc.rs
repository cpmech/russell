use plotpy::{Contour, Curve, Plot, Surface};
use russell_lab::approx_eq;
use russell_pde::{ProblemSamples, Spc2d, SpcMap2d, StrError, TransfiniteSamples};

const SAVE_FIGURE: bool = false;

#[test]
fn test_2d_prob05_spc() -> Result<(), StrError> {
    // parameters
    let (nn, tol) = (5, 1e-14);

    // get the problem data
    let (_, _, _, _, kx, ky, ebcs, nbcs, source, analytical) = ProblemSamples::d2_problem_05();

    // allocate the solver
    let (nx, ny) = (nn + 1, nn + 1);
    let spc = Spc2d::new(nx, ny, ebcs, nbcs, kx, ky)?;

    // solve the problem
    let a = spc.solve(&source)?;

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
        let fn_a = format!("/tmp/russell_pde/test_2d_prob05_spc_a.svg");
        let fn_b = format!("/tmp/russell_pde/test_2d_prob05_spc_b.svg");
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
            .set_with_wireframe(true)
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
        plot.add(&surf_num)
            .add(&surf_ana)
            .set_figure_size_points(600.0, 600.0)
            .save(&fn_b)?;
    }
    Ok(())
}

#[test]
fn test_2d_prob05_spc_map() -> Result<(), StrError> {
    // parameters
    let (nn, tol) = (5, 1e-14);

    // get the problem data
    let (_, _, _, _, k, _, ebcs, nbcs, source, analytical) = ProblemSamples::d2_problem_05();

    // transfinite map
    let map = TransfiniteSamples::quadrilateral_2d(&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]);

    // allocate the solver
    let (nx, ny) = (nn + 1, nn + 1);
    let mut spc = SpcMap2d::new(nx, ny, ebcs, nbcs, k, map)?;

    // solve the problem
    let a = spc.solve(&source)?;

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
