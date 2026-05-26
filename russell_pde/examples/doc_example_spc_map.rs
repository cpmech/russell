use plotpy::{linspace, Canvas, Contour, Plot, PolyCode};
use russell_lab::math::PI;
use russell_lab::{approx_eq, Vector};
use russell_pde::{EssentialBcs2d, NaturalBcs2d, Side, SpcMap2d, StrError, Transfinite2d, TransfiniteSamples};

const SAVE_FIGURE: bool = false;

fn main() -> Result<(), StrError> {
    // Polynomial degree and tolerance for error checking
    let nn = 20;
    let tol = 1.0e-5;
    let alpha = PI / 6.0;
    let (ca, sa) = (f64::cos(alpha), f64::sin(alpha));

    // Transfinite map
    let xa = &[0.0, 0.0];
    let xb = &[ca, sa];
    let xc = &[ca - sa, ca + sa];
    let xd = &[-sa, ca];
    let mut map = TransfiniteSamples::quadrilateral_2d(xa, xb, xc, xd);

    // Draw the domain
    if SAVE_FIGURE {
        let mut canvas = Canvas::new();
        draw_lines_2d(&mut canvas, &mut map, 21);
        let mut plot = Plot::new();
        plot.add(&canvas)
            .set_range(-1.0, 1.0, 0.0, 2.0)
            .set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0)
            .save("/tmp/russell_pde/doc_example_spc_map_grid.svg")?;
    }

    // Analytical solution
    let analytical = move |x: f64, y: f64| f64::sin(PI * x * ca + PI * y * sa) * f64::exp(PI * y * ca - PI * x * sa);

    // Essential boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(Side::Xmin, |_, _| 0.0);
    ebcs.set(Side::Xmax, |_, _| 0.0);
    ebcs.set(Side::Ymin, move |x, _| f64::sin(PI * x / ca));
    ebcs.set(Side::Ymax, move |x, _| f64::sin(PI * (x + sa) / ca) * f64::exp(PI));

    // Natural boundary conditions
    let nbcs = NaturalBcs2d::new();

    // Allocate the solver
    let (nr, ns) = (nn + 1, nn + 1);
    let k = 1.0;
    let mut spc = SpcMap2d::new(map, nr, ns, ebcs, nbcs, k)?;

    // Solve the problem
    let a = spc.solve_sps(0.0, |_, _| 0.0)?;

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

    // plot the results
    if SAVE_FIGURE {
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
        let n_ana = 41;
        let (xx_ana, yy_ana, tri_ana) = spc.get_map().triangulate(n_ana, n_ana, false, false);
        let mut zz = vec![0.0; xx_ana.len()];
        for m in 0..xx_ana.len() {
            zz[m] = analytical(xx_ana[m], yy_ana[m]);
        }
        let mut contour_ana = Contour::new();
        contour_ana
            .set_extra_filled("levels=21")
            .set_no_colorbar(true)
            .set_colormap_index(4)
            .set_tri_show_edges(false)
            .set_no_lines(false)
            .draw_tri(&xx_ana, &yy_ana, &zz, &tri_ana);
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
    }
    Ok(())
}

fn draw_lines_2d(canvas: &mut Canvas, map: &mut Transfinite2d, np: usize) {
    canvas.set_face_color("None");
    let mut x = Vector::new(2);
    let tt = linspace(-1.0, 1.0, np);
    // lines in r-direction
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
    // lines in s-direction
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
