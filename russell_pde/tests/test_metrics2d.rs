use plotpy::{linspace, Canvas, Plot, PolyCode, Text};
use russell_lab::{approx_eq, vec_approx_eq, vec_inner, Matrix, Vector};
use russell_pde::{FnVec1Param1, Metrics, StrError, Transfinite2d};

const SAVE_FIGURE: bool = false;

fn map_fun(x: &mut Vector, t: f64, xa: &[f64], xb: &[f64], cf: f64) {
    let u = [xb[0] - xa[0], xb[1] - xa[1]];
    let n = f64::sqrt(u[0] * u[0] + u[1] * u[1]);
    let v = [-u[1] / n, u[0] / n];
    for d in 0..2 {
        x[d] = xa[d] + (xb[d] - xa[d]) * ((1.0 + t) / 2.0) + cf * (t * t - 1.0) * v[d];
    }
}

fn deriv1_map_fun(x: &mut Vector, t: f64, xa: &[f64], xb: &[f64], cf: f64) {
    let u = [xb[0] - xa[0], xb[1] - xa[1]];
    let n = f64::sqrt(u[0] * u[0] + u[1] * u[1]);
    let v = [-u[1] / n, u[0] / n];
    for d in 0..2 {
        x[d] = (xb[d] - xa[d]) / 2.0 + cf * 2.0 * t * v[d];
    }
}

fn deriv2_map_fun(x: &mut Vector, _t: f64, xa: &[f64], xb: &[f64], cf: f64) {
    let u = [xb[0] - xa[0], xb[1] - xa[1]];
    let n = f64::sqrt(u[0] * u[0] + u[1] * u[1]);
    let v = [-u[1] / n, u[0] / n];
    for d in 0..2 {
        x[d] = cf * 2.0 * v[d];
    }
}

#[test]
fn test_metrics_2d() -> Result<(), StrError> {
    let a = 1.0 / 3.0;
    let b = -1.0 / 2.0;
    let xx = [[2.0, 1.0], [6.0, 3.0], [5.0, 7.0], [1.0, 4.0]];

    let boundary_functions: Vec<FnVec1Param1> = vec![
        // B0(s) with s ϵ [-1,+1]
        Box::new(move |x, s| {
            map_fun(x, s, &xx[0], &xx[3], a);
        }),
        // B1(s) with s ϵ [-1,+1]
        Box::new(move |x, s| {
            map_fun(x, s, &xx[1], &xx[2], a);
        }),
        // B2(r) with r ϵ [-1,+1]
        Box::new(move |x, r| {
            map_fun(x, r, &xx[0], &xx[1], b);
        }),
        // B3(r) with r ϵ [-1,+1]
        Box::new(move |x, r| {
            map_fun(x, r, &xx[3], &xx[2], b);
        }),
    ];

    let mut x = Vector::new(2);
    boundary_functions[0](&mut x, -1.0);
    assert_eq!(x.as_data(), &xx[0]);
    boundary_functions[0](&mut x, 1.0);
    assert_eq!(x.as_data(), &xx[3]);
    boundary_functions[1](&mut x, -1.0);
    assert_eq!(x.as_data(), &xx[1]);
    boundary_functions[1](&mut x, 1.0);
    assert_eq!(x.as_data(), &xx[2]);
    boundary_functions[2](&mut x, -1.0);
    assert_eq!(x.as_data(), &xx[0]);
    boundary_functions[2](&mut x, 1.0);
    assert_eq!(x.as_data(), &xx[1]);
    boundary_functions[3](&mut x, -1.0);
    assert_eq!(x.as_data(), &xx[3]);
    boundary_functions[3](&mut x, 1.0);
    assert_eq!(x.as_data(), &xx[2]);

    let deriv1_boundary_functions: Vec<FnVec1Param1> = vec![
        // dB0/ds
        Box::new(move |dx_ds, s| {
            deriv1_map_fun(dx_ds, s, &xx[0], &xx[3], a);
        }),
        // dB1/ds
        Box::new(move |dx_ds, s| {
            deriv1_map_fun(dx_ds, s, &xx[1], &xx[2], a);
        }),
        // dB2/dr
        Box::new(move |dx_dr, r| {
            deriv1_map_fun(dx_dr, r, &xx[0], &xx[1], b);
        }),
        // dB3/dr
        Box::new(move |dx_dr, r| {
            deriv1_map_fun(dx_dr, r, &xx[3], &xx[2], b);
        }),
    ];

    let deriv2_boundary_functions: Vec<FnVec1Param1> = vec![
        // d²B0/ds²
        Box::new(move |d2x_ds2, s| {
            deriv2_map_fun(d2x_ds2, s, &xx[0], &xx[3], a);
        }),
        // d²B1/ds²
        Box::new(move |d2x_ds2, s| {
            deriv2_map_fun(d2x_ds2, s, &xx[1], &xx[2], a);
        }),
        // d²B2/dr²
        Box::new(move |d2x_dr2, r| {
            deriv2_map_fun(d2x_dr2, r, &xx[0], &xx[1], b);
        }),
        // d²B3/dr²
        Box::new(move |d2x_dr2, r| {
            deriv2_map_fun(d2x_dr2, r, &xx[3], &xx[2], b);
        }),
    ];

    let mut map = Transfinite2d::new(
        boundary_functions,
        deriv1_boundary_functions,
        Some(deriv2_boundary_functions),
    )?;

    let mut met = Metrics::new(2, false);

    let mut dx_dr = Vector::new(2);
    let mut dx_ds = Vector::new(2);
    let mut d2x_dr2 = Vector::new(2);
    let mut d2x_ds2 = Vector::new(2);
    let mut d2x_drs = Vector::new(2);

    for (r, s) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.0, 0.0)] {
        map.point_and_derivs(
            &mut x,
            &mut dx_dr,
            &mut dx_ds,
            Some(&mut d2x_dr2),
            Some(&mut d2x_ds2),
            Some(&mut d2x_drs),
            r,
            s,
        );

        let det_g = met.calculate_2d(&dx_dr, &dx_ds, Some(&d2x_dr2), Some(&d2x_ds2), Some(&d2x_drs))?;

        let (g_cov_0, g_cov_1, g_ctr_0, g_ctr_1, det_g_ref) = mathematica(r, s, &xx, a, b);
        vec_approx_eq(&met.g_cov[0], &g_cov_0, 1e-15);
        vec_approx_eq(&met.g_cov[1], &g_cov_1, 1e-15);
        vec_approx_eq(&met.g_ctr[0], &g_ctr_0, 1e-15);
        vec_approx_eq(&met.g_ctr[1], &g_ctr_1, 1e-15);
        approx_eq(det_g, det_g_ref, 1e-14);
    }

    if SAVE_FIGURE {
        map.point_and_derivs(
            &mut x,
            &mut dx_dr,
            &mut dx_ds,
            Some(&mut d2x_dr2),
            Some(&mut d2x_ds2),
            Some(&mut d2x_drs),
            0.0,
            0.0,
        );

        met.calculate_2d(&dx_dr, &dx_ds, Some(&d2x_dr2), Some(&d2x_ds2), Some(&d2x_drs))?;

        let mut canvas_lines = Canvas::new();
        let mut canvas_cov = Canvas::new();
        let mut canvas_ctr = Canvas::new();
        let mut canvas_cov_many = Canvas::new();
        let mut canvas_ctr_many = Canvas::new();
        let mut text_cov = Text::new();
        let mut text_ctr = Text::new();
        canvas_lines.set_edge_color("#606060ff");
        canvas_cov.set_arrow_scale(10.0);
        canvas_ctr.set_arrow_scale(10.0);
        canvas_cov_many.set_arrow_scale(10.0);
        canvas_ctr_many.set_arrow_scale(10.0);
        text_cov
            .set_fontsize(14.0)
            .set_bbox(true)
            .set_bbox_style("square,pad=0.1")
            .set_bbox_facecolor("white")
            .set_bbox_edgecolor("None");
        text_ctr
            .set_fontsize(14.0)
            .set_bbox(true)
            .set_bbox_style("square,pad=0.1")
            .set_bbox_facecolor("white")
            .set_bbox_edgecolor("None");
        draw_lines_2d(&mut canvas_lines, &mut map, 11, 0.03);
        draw_cov_vectors_2d(&mut canvas_cov, Some(&mut text_cov), &x, &met, 1.0, false);
        draw_ctr_vectors_2d(&mut canvas_ctr, Some(&mut text_ctr), &x, &met, 2.0, false);

        let rr = linspace(-1.0, 1.0, 6);
        let ss = rr.clone();
        for &r in &rr {
            for &s in &ss {
                map.point_and_derivs(
                    &mut x,
                    &mut dx_dr,
                    &mut dx_ds,
                    Some(&mut d2x_dr2),
                    Some(&mut d2x_ds2),
                    Some(&mut d2x_drs),
                    r,
                    s,
                );
                met.calculate_2d(&dx_dr, &dx_ds, Some(&d2x_dr2), Some(&d2x_ds2), Some(&d2x_drs))?;
                draw_cov_vectors_2d(&mut canvas_cov_many, None, &x, &met, 0.5, true);
                draw_ctr_vectors_2d(&mut canvas_ctr_many, None, &x, &met, 0.5, true);
            }
        }

        let mut plot = Plot::new();
        plot.set_gaps(0.0, 0.0)
            .set_subplot(2, 2, 1)
            .add(&canvas_lines)
            .add(&canvas_cov)
            .add(&text_cov)
            .set_range(0.5, 6.5, 0.5, 7.5)
            .set_equal_axes(true)
            .set_hide_axes(true)
            // 2
            .set_subplot(2, 2, 2)
            .add(&canvas_lines)
            .add(&canvas_ctr)
            .add(&text_ctr)
            .set_range(0.5, 6.5, 0.5, 7.5)
            .set_equal_axes(true)
            .set_hide_axes(true)
            // 3
            .set_subplot(2, 2, 3)
            .add(&canvas_lines)
            .add(&canvas_cov_many)
            .set_range(0.5, 6.5, 0.5, 7.5)
            .set_equal_axes(true)
            .set_hide_axes(true)
            // 4
            .set_subplot(2, 2, 4)
            .add(&canvas_lines)
            .add(&canvas_ctr_many)
            .set_range(0.5, 6.5, 0.5, 7.5)
            .set_equal_axes(true)
            .set_hide_axes(true)
            //
            .extra(
                "plt.figtext(0.33,0.52,'(a)',ha='center',va='center',fontsize=14)\n\
                 plt.figtext(0.70,0.52,'(b)',ha='center',va='center',fontsize=14)\n\
                 plt.figtext(0.33,0.12,'(c)',ha='center',va='center',fontsize=14)\n\
                 plt.figtext(0.70,0.12,'(d)',ha='center',va='center',fontsize=14)\n",
            )
            .set_figure_size_points(800.0, 800.0)
            .save("/tmp/russell_pde/test_metrics_2d.svg")
            .unwrap();
    }

    Ok(())
}

/// Calculates the base vectors using Mathematica's expressions
///
/// Returns `(g_cov_0, g_cov_1, g_ctr_0, g_ctr_1, det_g)`
fn mathematica(r: f64, s: f64, xx: &[[f64; 2]; 4], a: f64, b: f64) -> (Vector, Vector, Vector, Vector, f64) {
    let x0 = xx[0][0];
    let y0 = xx[0][1];
    let x1 = xx[1][0];
    let y1 = xx[1][1];
    let x2 = xx[2][0];
    let y2 = xx[2][1];
    let x3 = xx[3][0];
    let y3 = xx[3][1];

    let p30 = x3 - x0;
    let q30 = y3 - y0;
    let p21 = x2 - x1;
    let q21 = y2 - y1;
    let p10 = x1 - x0;
    let q10 = y1 - y0;
    let p23 = x2 - x3;
    let q23 = y2 - y3;

    let d30 = f64::sqrt(f64::powi(p30, 2) + f64::powi(q30, 2));
    let d21 = f64::sqrt(f64::powi(p21, 2) + f64::powi(q21, 2));
    let d10 = f64::sqrt(f64::powi(p10, 2) + f64::powi(q10, 2));
    let d23 = f64::sqrt(f64::powi(p23, 2) + f64::powi(q23, 2));

    let dx_dr_0 = ((4.0 * b * q10 * r * (-1.0 + s)) / d10 - (4.0 * b * q23 * r * (1.0 + s)) / d23
        + (2.0 * a * (-(d30 * q21) + d21 * q30) * (-1.0 + f64::powi(s, 2))) / (d21 * d30)
        + (-1.0 + s) * x0
        + x1
        - s * x1
        + (1.0 + s) * (x2 - x3))
        / 4.0;

    let dx_dr_1 = ((-4.0 * b * p10 * r * (-1.0 + s)) / d10
        + (4.0 * b * p23 * r * (1.0 + s)) / d23
        + (2.0 * a * (d30 * p21 - d21 * p30) * (-1.0 + f64::powi(s, 2))) / (d21 * d30)
        + (-1.0 + s) * y0
        + y1
        - s * y1
        + (1.0 + s) * (y2 - y3))
        / 4.0;

    let dx_ds_0 = ((2.0 * b * (d23 * q10 - d10 * q23) * (-1.0 + f64::powi(r, 2))) / (d10 * d23)
        + (4.0 * a * q30 * (-1.0 + r) * s) / d30
        - (4.0 * a * q21 * (1.0 + r) * s) / d21
        + (-1.0 + r) * x0
        - x1
        - r * x1
        + x2
        + r * x2
        + x3
        - r * x3)
        / 4.0;

    let dx_ds_1 = ((2.0 * b * (-(d23 * p10) + d10 * p23) * (-1.0 + f64::powi(r, 2))) / (d10 * d23)
        - (4.0 * a * p30 * (-1.0 + r) * s) / d30
        + (4.0 * a * p21 * (1.0 + r) * s) / d21
        + (-1.0 + r) * y0
        - (1.0 + r) * y1
        + y2
        + r * y2
        + y3
        - r * y3)
        / 4.0;

    let g_cov_0 = Vector::from(&[dx_dr_0, dx_dr_1]);
    let g_cov_1 = Vector::from(&[dx_ds_0, dx_ds_1]);

    let g_mat = Matrix::from(&[
        [vec_inner(&g_cov_0, &g_cov_0), vec_inner(&g_cov_0, &g_cov_1)], // g0.g0, g0.g1
        [vec_inner(&g_cov_1, &g_cov_0), vec_inner(&g_cov_1, &g_cov_1)], // g1.g0, g1.g1
    ]);

    let det_g = g_mat.get(0, 0) * g_mat.get(1, 1) - g_mat.get(0, 1) * g_mat.get(1, 0);

    let jj = f64::sqrt(det_g);

    let g_ctr_0 = Vector::from(&[g_cov_1[1] / jj, -g_cov_1[0] / jj]);

    let g_ctr_1 = Vector::from(&[-g_cov_0[1] / jj, g_cov_0[0] / jj]);

    (g_cov_0, g_cov_1, g_ctr_0, g_ctr_1, det_g)
}

fn draw_cov_vectors_2d(
    canvas: &mut Canvas,
    text: Option<&mut Text>,
    x: &Vector,
    met: &Metrics,
    scale: f64,
    normalize: bool,
) {
    let txt_scale = 1.1 * scale;

    // covariant base vector in r-direction
    let g0 = if normalize {
        let norm = f64::sqrt(vec_inner(&met.g_cov[0], &met.g_cov[0]));
        Vector::from(&[met.g_cov[0][0] / norm, met.g_cov[0][1] / norm])
    } else {
        met.g_cov[0].clone()
    };
    canvas.set_edge_color("red").set_face_color("red").draw_arrow(
        x[0],
        x[1],
        x[0] + scale * g0[0],
        x[1] + scale * g0[1],
    );

    // covariant base vector in s-direction
    let g1 = if normalize {
        let norm = f64::sqrt(vec_inner(&met.g_cov[1], &met.g_cov[1]));
        Vector::from(&[met.g_cov[1][0] / norm, met.g_cov[1][1] / norm])
    } else {
        met.g_cov[1].clone()
    };
    canvas.set_edge_color("blue").set_face_color("blue").draw_arrow(
        x[0],
        x[1],
        x[0] + scale * g1[0],
        x[1] + scale * g1[1],
    );

    if let Some(txt) = text {
        txt.draw(x[0] + txt_scale * g0[0], x[1] + txt_scale * g0[1], "$\\vec{g}_1$");
        txt.draw(x[0] + txt_scale * g1[0], x[1] + txt_scale * g1[1], "$\\vec{g}_2$");
    }
}

fn draw_ctr_vectors_2d(
    canvas: &mut Canvas,
    text: Option<&mut Text>,
    x: &Vector,
    met: &Metrics,
    scale: f64,
    normalize: bool,
) {
    let txt_scale = 1.2 * scale;

    // contravariant base vector in r-direction
    let g0 = if normalize {
        let norm = f64::sqrt(vec_inner(&met.g_ctr[0], &met.g_ctr[0]));
        Vector::from(&[met.g_ctr[0][0] / norm, met.g_ctr[0][1] / norm])
    } else {
        met.g_ctr[0].clone()
    };
    canvas.set_edge_color("green").set_face_color("green").draw_arrow(
        x[0],
        x[1],
        x[0] + scale * g0[0],
        x[1] + scale * g0[1],
    );

    // contravariant base vector in s-direction
    let g1 = if normalize {
        let norm = f64::sqrt(vec_inner(&met.g_ctr[1], &met.g_ctr[1]));
        Vector::from(&[met.g_ctr[1][0] / norm, met.g_ctr[1][1] / norm])
    } else {
        met.g_ctr[1].clone()
    };
    canvas.set_edge_color("orange").set_face_color("orange").draw_arrow(
        x[0],
        x[1],
        x[0] + scale * g1[0],
        x[1] + scale * g1[1],
    );

    if let Some(txt) = text {
        txt.draw(x[0] + txt_scale * g0[0], x[1] + txt_scale * g0[1], "$2\\vec{g}^1$");
        txt.draw(x[0] + txt_scale * g1[0], x[1] + txt_scale * g1[1], "$2\\vec{g}^2$");
    }
}

fn draw_lines_2d(canvas: &mut Canvas, map: &mut Transfinite2d, np: usize, dot_size: f64) {
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
    // points at corners
    map.point(&mut x, -1.0, -1.0);
    canvas.draw_circle(x[0], x[1], dot_size);
    map.point(&mut x, 1.0, -1.0);
    canvas.draw_circle(x[0], x[1], dot_size);
    map.point(&mut x, 1.0, 1.0);
    canvas.draw_circle(x[0], x[1], dot_size);
    map.point(&mut x, -1.0, 1.0);
    canvas.draw_circle(x[0], x[1], dot_size);
    // point in the center
    map.point(&mut x, 0.0, 0.0);
    canvas.draw_circle(x[0], x[1], dot_size);
}
