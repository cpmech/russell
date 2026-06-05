//! Example: Solving a 2D Poisson equation on a rotated square domain
//!
//! This example demonstrates the use of `SpcMap2d` (spectral collocation with
//! transfinite mapping) to solve the Poisson equation:
//!
//! ```text
//!   -k · ∇²u = f    on a unit square rotated by angle α
//!   u = g           on the boundary (Dirichlet conditions)
//! ```
//!
//! The analytical solution used for verification is:
//! ```text
//!   u(x,y) = sin(π·x·cos(α) + π·y·sin(α)) · exp(π·y·cos(α) - π·x·sin(α))
//! ```
//!
//! The domain is mapped from the reference square (r,s) ∈ [-1,1]×[-1,1] to
//! the physical rotated square via transfinite interpolation.

use plotpy::{linspace, Canvas, Contour, Plot, PolyCode};
use russell_lab::math::PI;
use russell_lab::{approx_eq, Vector};
use russell_pde::{EssentialBcs2d, NaturalBcs2d, Side, SpcMap2d, StrError, Transfinite2d, TransfiniteSamples};

fn main() -> Result<(), StrError> {
    // ---- Parameters --------------------------------------------------
    let nn = 20; // number of interior nodes in each direction
    let tol = 1.0e-5; // tolerance for error checking
    let alpha = PI / 6.0; // rotation angle of the square (30 degrees)
    let (ca, sa) = (f64::cos(alpha), f64::sin(alpha)); // precompute cos/sin of alpha

    // ---- Define the transfinite map (unit square rotated by alpha) ----
    // The four corner points of the rotated unit square:
    let xa = &[0.0, 0.0]; // bottom-left corner (origin)
    let xb = &[ca, sa]; // bottom-right corner
    let xc = &[ca - sa, ca + sa]; // top-right corner
    let xd = &[-sa, ca]; // top-left corner
    let mut map = TransfiniteSamples::quadrilateral_2d(xa, xb, xc, xd);

    // ---- Draw the computational grid for visualization ---------------
    let mut canvas = Canvas::new();
    draw_lines_2d(&mut canvas, &mut map, 21);
    let mut plot = Plot::new();
    plot.add(&canvas)
        .set_range(-1.0, 1.0, 0.0, 2.0)
        .set_equal_axes(true)
        .set_figure_size_points(600.0, 600.0)
        .save("/tmp/russell_pde/doc_example_spc_map_grid.svg")?;

    // ---- Analytical solution (closure capturing alpha) ----------------
    let analytical = move |x: f64, y: f64| f64::sin(PI * x * ca + PI * y * sa) * f64::exp(PI * y * ca - PI * x * sa);

    // ---- Essential (Dirichlet) boundary conditions --------------------
    // Set u = 0 on the left and right sides; on bottom and top sides,
    // set u to match the analytical solution for verification.
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(Side::Xmin, |_, _| 0.0);
    ebcs.set(Side::Xmax, |_, _| 0.0);
    ebcs.set(Side::Ymin, move |x, _| f64::sin(PI * x / ca));
    ebcs.set(Side::Ymax, move |x, _| f64::sin(PI * (x + sa) / ca) * f64::exp(PI));

    // ---- Natural (Neumann) boundary conditions -----------------------
    // No natural BCs needed — all sides have Dirichlet conditions.
    let nbcs = NaturalBcs2d::new();

    // ---- Allocate and assemble the spectral collocation solver --------
    let (nr, ns) = (nn + 1, nn + 1); // total nodes (interior + boundary) in r and s
    let k = 1.0; // diffusion coefficient
    let mut spc = SpcMap2d::new(map, nr, ns, ebcs, nbcs, k)?;

    // ---- Solve the linear system (source term f = 0) -----------------
    let a = spc.solve_sps(0.0, |_, _| 0.0)?;

    // ---- Verify numerical solution against analytical solution --------
    let mut err_max = 0.0;
    spc.for_each_coord(|m, x, y| {
        let err = f64::abs(a[m] - analytical(x, y));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x, y), tol);
    });
    println!("N = {} max(err) = {:>10.5e}", nn, err_max);

    // ---- Plot comparison: numerical vs analytical contours ------------
    // Triangulate the numerical grid for filled contour plotting.
    let (xx_num, yy_num, tri_num) = spc.get_map().triangulate(nr, ns, true, true);
    let neq = a.dim();
    let mut x_arr = vec![0.0; neq];
    let mut y_arr = vec![0.0; neq];
    let mut zz_num = vec![0.0; xx_num.len()];
    spc.for_each_coord(|m, x, y| {
        approx_eq(x, xx_num[m], 1.0e-15);
        approx_eq(y, yy_num[m], 1.0e-15);
        x_arr[m] = x;
        y_arr[m] = y;
        zz_num[m] = a[m];
    });

    // Generate a finer analytical grid for smooth contour lines.
    let n_ana = 41;
    let (xx_ana, yy_ana, tri_ana) = spc.get_map().triangulate(n_ana, n_ana, false, false);
    let mut zz = vec![0.0; xx_ana.len()];
    for m in 0..xx_ana.len() {
        zz[m] = analytical(xx_ana[m], yy_ana[m]);
    }

    // Filled contour: analytical solution.
    let mut contour_ana = Contour::new();
    contour_ana
        .set_extra_filled("levels=21")
        .set_no_colorbar(true)
        .set_colormap_index(4)
        .set_tri_show_edges(false)
        .set_no_lines(false)
        .draw_tri(&xx_ana, &yy_ana, &zz, &tri_ana);

    // Line contour overlay: numerical solution (magenta lines).
    let mut contour_num = Contour::new();
    contour_num
        .set_extra_line("levels=21")
        .set_tri_show_edges(false)
        .set_no_colorbar(true)
        .set_no_fill(true)
        .set_no_lines(false)
        .set_line_color("#ea14dfff")
        .set_fontsize_labels(14.0)
        .draw_tri(&xx_num, &yy_num, &zz_num, &tri_num);

    let mut plot = Plot::new();
    plot.add(&contour_ana)
        .add(&contour_num)
        .set_labels("x", "y")
        .set_equal_axes(true)
        .set_figure_size_points(600.0, 600.0)
        .save("/tmp/russell_pde/doc_example_spc_map.svg")?;
    Ok(())
}

/// Draws the transfinite interpolation grid lines (r- and s-directions).
///
/// * `canvas` – the plot canvas to draw on.
/// * `map`    – the transfinite map defining the domain.
/// * `np`     – number of sample points along each coordinate direction.
fn draw_lines_2d(canvas: &mut Canvas, map: &mut Transfinite2d, np: usize) {
    canvas.set_face_color("None");
    let mut x = Vector::new(2);
    let tt = linspace(-1.0, 1.0, np);

    // Draw lines of constant s (r-direction curves)
    for j in 0..np {
        let s = tt[j];
        map.point(&mut x, tt[0], s);
        canvas.polycurve_begin();
        canvas.polycurve_add(x[0], x[1], PolyCode::MoveTo);
        for i in 1..np {
            let r = tt[i];
            map.point(&mut x, r, s);
            canvas.polycurve_add(x[0], x[1], PolyCode::LineTo);
        }
        canvas.polycurve_end(false);
    }

    // Draw lines of constant r (s-direction curves)
    for i in 0..np {
        let r = tt[i];
        map.point(&mut x, r, tt[0]);
        canvas.polycurve_begin();
        canvas.polycurve_add(x[0], x[1], PolyCode::MoveTo);
        for j in 1..np {
            let s = tt[j];
            map.point(&mut x, r, s);
            canvas.polycurve_add(x[0], x[1], PolyCode::LineTo);
        }
        canvas.polycurve_end(false);
    }
}
