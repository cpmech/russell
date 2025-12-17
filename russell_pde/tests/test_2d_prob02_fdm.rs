use plotpy::{Contour, Plot};
use russell_lab::{approx_eq, math::PI};
use russell_pde::{EssentialBcs2d, Fdm2d, Grid2d, Side, StrError};

const SAVE_FIGURE: bool = false;

// Solve the following problem:
//
// ∂²ϕ   ∂²ϕ
// ——— + ——— = - π² y sin(π x)
// ∂x²   ∂y²
//
// on a (1.0 × 1.0) square with the following essential boundary conditions:
//
// left:    ϕ(0.0, y) = 0.0
// right:   ϕ(1.0, y) = 0.0
// bottom:  ϕ(x, 0.0) = 0.0
// top:     ϕ(x, 1.0) = sin(π x)
//
// The analytical solution is:
//
// ϕ(x, y) = y sin(π x)
//
// Reference: Olver PJ (2020) - page 210 - Introduction to Partial Differential Equations, Springer

fn source(x: f64, y: f64) -> f64 {
    -PI * PI * y * f64::sin(PI * x)
}

fn analytical(x: f64, y: f64) -> f64 {
    y * f64::sin(PI * x)
}

#[test]
fn test_2d_prob02_fdm_sps() -> Result<(), StrError> {
    // allocate the grid
    let (nx, ny) = (17, 17);
    let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, nx, ny)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(Side::Xmin, |_, _| 0.0);
    ebcs.set(Side::Xmax, |_, _| 0.0);
    ebcs.set(Side::Ymin, |_, _| 0.0);
    ebcs.set(Side::Ymax, |x, _| f64::sin(PI * x));

    // allocate the solver
    let (kx, ky) = (-1.0, -1.0);
    let fdm = Fdm2d::new(grid, ebcs, kx, ky)?;

    // solve the problem
    let a = fdm.solve(source)?;

    // check
    fdm.for_each_coord(|m, x, y| {
        approx_eq(a[m], analytical(x, y), 0.001036);
    });

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
            .save("/tmp/russell_pde/test_poisson2d_2.svg")?;
    }
    Ok(())
}

#[test]
fn test_2d_prob02_fdm_lmm() -> Result<(), StrError> {
    // allocate the grid
    let (nx, ny) = (17, 17);
    let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, nx, ny)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(Side::Xmin, |_, _| 0.0);
    ebcs.set(Side::Xmax, |_, _| 0.0);
    ebcs.set(Side::Ymin, |_, _| 0.0);
    ebcs.set(Side::Ymax, |x, _| f64::sin(PI * x));

    // allocate the solver
    let (kx, ky) = (-1.0, -1.0);
    let fdm = Fdm2d::new(grid, ebcs, kx, ky)?;

    // solve the problem
    let a = fdm.solve_lmm(source)?;

    // check
    fdm.for_each_coord(|m, x, y| {
        approx_eq(a[m], analytical(x, y), 0.001036);
    });
    Ok(())
}
