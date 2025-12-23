use plotpy::{Contour, Plot};
use russell_lab::approx_eq;
use russell_pde::{Fdm2d, Grid2d, ProblemSamples, StrError};

const SAVE_FIGURE: bool = false;

#[test]
fn test_2d_prob02_fdm_sps() -> Result<(), StrError> {
    // get the problem data
    let (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical) = ProblemSamples::d2_problem_02();

    // allocate the grid
    let n = 10;
    let (nx, ny) = (n, n);
    let grid = Grid2d::new_uniform(xmin, xmax, ymin, ymax, nx, ny)?;

    // allocate the solver
    let fdm = Fdm2d::new(grid, ebcs, nbcs, kx, ky)?;

    // solve the problem
    let a = fdm.solve_poisson_sps(&source)?;

    // check
    let mut err_max = 0.0;
    fdm.for_each_coord(|m, x, y| {
        let err = f64::abs(a[m] - analytical(x, y));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x, y), 3.19e-3);
    });
    println!("max(err) = {:>10.5e}", err_max);

    // plot results
    if SAVE_FIGURE {
        let mut contour_num = Contour::new();
        let mut contour_ana = Contour::new();
        let mut xx = vec![vec![0.0; nx]; ny];
        let mut yy = vec![vec![0.0; nx]; ny];
        let mut zz_num = vec![vec![0.0; nx]; ny];
        let mut zz_ana = vec![vec![0.0; nx]; ny];
        fdm.for_each_coord(|m, x, y| {
            let row = m / nx;
            let col = m % nx;
            xx[row][col] = x;
            yy[row][col] = y;
            zz_num[row][col] = a[m];
            zz_ana[row][col] = analytical(x, y);
        });
        contour_num.set_no_lines(false).draw(&xx, &yy, &zz_num);
        contour_ana
            .set_colors(&["None"])
            .set_no_colorbar(true)
            .set_no_labels(true)
            .set_line_color("yellow")
            .set_line_style(":")
            .set_line_width(2.0)
            .draw(&xx, &yy, &zz_ana);
        let mut plot = Plot::new();
        plot.add(&contour_num).add(&contour_ana);
        plot.set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save("/tmp/russell_pde/test_2d_prob02_fdm_sps.svg")?;
    }
    Ok(())
}

#[test]
fn test_2d_prob02_fdm_lmm() -> Result<(), StrError> {
    // get the problem data
    let (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical) = ProblemSamples::d2_problem_02();

    // allocate the grid
    let (nx, ny) = (17, 17);
    let grid = Grid2d::new_uniform(xmin, xmax, ymin, ymax, nx, ny)?;

    // allocate the solver
    let fdm = Fdm2d::new(grid, ebcs, nbcs, kx, ky)?;

    // solve the problem
    let a = fdm.solve_poisson_lmm(&source)?;

    // check
    fdm.for_each_coord(|m, x, y| {
        approx_eq(a[m], analytical(x, y), 0.001036);
    });
    Ok(())
}
