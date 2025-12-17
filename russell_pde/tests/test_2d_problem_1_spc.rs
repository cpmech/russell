use plotpy::{Contour, Curve, Plot, Surface};
use russell_lab::{approx_eq, math::PI};
use russell_pde::{EssentialBcs2d, Grid2d, SpectralLaplacian2d, StrError};
use russell_sparse::{Genie, LinSolver};

// Approximate the solution of
//
// ∇²ϕ = -1
//
// on a transfinite mapped domain. The mapped domain is a unit square.
// The boundary conditions are:
//
// R-min (aka Xmin): ∇ϕ = 0   on r = 0
// R-max (aka Xmax):  ϕ = 0   on r = 1
// S-min (aka Ymin): ∇ϕ = 0   on s = 0
// S-max (aka Ymax):  ϕ = 0   on s = 1
//
// The analytical solution is:
//
//          1 - x²   16    ∞
// ϕ(x,y) = —————— - ——    Σ    mₖ(x,y)
//            2      π³  k = 1
//                       k odd
// where
//           sin(aₖ) (sinh(bₖ) + sinh(cₖ))
// mₖ(x,y) = —————————————————————————————
//                   k³ sinh(k π)
//
// aₖ = k π (1 + x) / 2
// bₖ = k π (1 + y) / 2
// cₖ = k π (1 - y) / 2

const SAVE_FIGURE: bool = false;

#[test]
fn test_2d_problem_1_spc() -> Result<(), StrError> {
    for (nn, tol) in vec![
        (8, 1e-4), //
                   // (24, 1e-5), // cannot get better precision because the analytical solution is approximated
    ] {
        let err_max = run_test(nn, tol)?;
        println!("N = {:>2}, max(err) = {:>10.5e}", nn, err_max);
    }
    Ok(())
}

/// Runs the test and returns max(error)
fn run_test(nn: usize, tol: f64) -> Result<f64, StrError> {
    // define the analytical solution
    let analytical = |x, y| {
        let nk = 101; // for nk > 200, infinite values may appear in the sum
        let mut sum = 0.0;
        for k in (1..nk).step_by(2) {
            let k3 = (k * k * k) as f64;
            let kp = (k as f64) * PI;
            let ak = kp * (1.0 + x) / 2.0;
            let bk = kp * (1.0 + y) / 2.0;
            let ck = kp * (1.0 - y) / 2.0;
            sum += f64::sin(ak) * (f64::sinh(bk) + f64::sinh(ck)) / (k3 * f64::sinh(kp));
        }
        (1.0 - x * x) / 2.0 - 16.0 * sum / (PI * PI * PI)
    };

    // define the source term
    let source = |_x, _y| -1.0;

    // allocate the grid on [-1, 1] × [-1, 1] and then map to a quarter ring
    let (nx, ny) = (nn + 1, nn + 1);
    let grid = Grid2d::new_chebyshev_gauss_lobatto(-1.0, 1.0, -1.0, 1.0, nx, ny)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set_homogeneous(&grid);

    // allocate the Laplacian operator
    let spc = SpectralLaplacian2d::new(grid, ebcs, 1.0, 1.0)?;

    // assemble the coefficient matrix and the lhs and rhs vectors
    let (kk_bar, kk_check) = spc.get_matrices();
    let (mut a_bar, a_check, mut f_bar) = spc.get_vectors(source);

    // initialize the right-hand side
    kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check).unwrap(); // f̄ -= Ǩ ǎ

    // solve the linear system
    let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
    solver.actual.factorize(&kk_bar, None).unwrap();
    solver.actual.solve(&mut a_bar, &f_bar, false).unwrap();

    // results
    let a = spc.get_joined_vector(&a_bar, &a_check);

    // check
    let mut err_max = 0.0;
    spc.for_each_coord(|m, x, y| {
        let err = f64::abs(a[m] - analytical(x, y));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x, y), tol);
    });

    // plot results
    if SAVE_FIGURE {
        let fn_a = format!("/tmp/russell_pde/test_2d_problem_1_spc_{}_a.svg", nn);
        let fn_b = format!("/tmp/russell_pde/test_2d_problem_1_spc_{}_b.svg", nn);
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
            .save(&fn_b)
            .unwrap();
    }
    Ok(err_max)
}
