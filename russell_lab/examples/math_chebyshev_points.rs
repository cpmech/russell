use plotpy::{Curve, Plot, RayEndpoint, Text};
use russell_lab::math::GOLDEN_RATIO;
use russell_lab::*;

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    let nn = 8;
    let npoint = nn + 1;
    let y_cg: Vec<_> = (0..npoint).into_iter().map(|_| 1.0).collect();
    let y_cgl: Vec<_> = (0..npoint).into_iter().map(|_| 0.0).collect();

    // Chebyshev-Gauss
    let x_cg = math::chebyshev_gauss_points(nn);
    let mut curve_cg = Curve::new();
    curve_cg
        .set_label("Chebyshev-Gauss")
        .set_line_style("None")
        .set_marker_style("o")
        .set_marker_line_color("blue")
        .set_marker_void(true)
        .draw(x_cg.as_data(), &y_cg);

    println!("\nChebyshev-Gauss points =\n{:.3?}", x_cg.as_data());

    // Chebyshev-Gauss-Lobatto
    let x_cgl = math::chebyshev_lobatto_points(nn);
    let mut curve_cgl = Curve::new();
    curve_cgl
        .set_label("Chebyshev-Gauss-Lobatto")
        .set_line_style("None")
        .set_marker_style("o")
        .set_marker_void(true)
        .draw(x_cgl.as_data(), &y_cgl);

    println!("\nChebyshev-Gauss-Lobatto points =\n{:.3?}\n", x_cgl.as_data());

    // vertical lines
    let mut v_lines = Curve::new();
    v_lines.set_line_color("#83C3A1").set_line_style("--");
    for x in &x_cgl {
        v_lines.draw_ray(*x, 0.0, RayEndpoint::Vertical);
    }

    // text
    let mut text = Text::new();
    text.set_bbox(true)
        .set_bbox_facecolor("pink")
        .set_bbox_edgecolor("black");
    text.draw(-1.0, 4.0, &format!("$N = {}$", nn));

    // save figure
    let path = format!("{}/math_chebyshev_points.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&v_lines)
        .add(&curve_cg)
        .add(&curve_cgl)
        .add(&text)
        .legend()
        .set_hide_yticks()
        .set_yrange(-5.0, 5.0)
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(&path)?;

    Ok(())
}
