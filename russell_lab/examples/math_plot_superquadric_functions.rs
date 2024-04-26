use plotpy::{Curve, Plot, Surface};
use russell_lab::math::{GOLDEN_RATIO, PI};
use russell_lab::*;

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    // x
    let xa = Vector::linspace(-PI / 2.0, PI / 2.0, 101)?;

    // suq_sin(x)
    let k = 2.0;
    let y_suq_sin = xa.get_mapped(|x| math::suq_sin(x, k));
    let mut curve = Curve::new();
    curve.set_line_width(2.5).draw(xa.as_data(), y_suq_sin.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_suq_sin.svg", OUT_DIR);
    plot.add(&curve)
        .grid_and_labels("$x$", format!("$\\mathrm{{SuqSin}}_{}(x)$", k).as_str())
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;

    // suq_cos(x)
    let y_suq_cos = xa.get_mapped(|x| math::suq_cos(x, k));
    let mut curve = Curve::new();
    curve.set_line_width(2.5).draw(xa.as_data(), y_suq_cos.as_data());
    let mut plot = Plot::new();
    let path = format!("{}/math_plot_functions_suq_cos.svg", OUT_DIR);
    plot.add(&curve)
        .grid_and_labels("$x$", format!("$\\mathrm{{SuqCos}}_{}(x)$", k).as_str())
        .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
        .save(path.as_str())?;

    // superquadric
    if false {
        let (n_alpha, n_theta) = (201, 201);
        let (alpha_min, alpha_max) = (-PI, PI);
        let (theta_min, theta_max) = (-PI / 2.0, PI / 2.0);
        let (alp, the) = generate2d(alpha_min, alpha_max, theta_min, theta_max, n_alpha, n_theta);
        let mut xx = vec![vec![0.0; n_alpha]; n_theta];
        let mut yy = vec![vec![0.0; n_alpha]; n_theta];
        let mut zz = vec![vec![0.0; n_alpha]; n_theta];
        let mut plot = Plot::new();
        let colors = &["#E9708E", "#4C689C", "#58B090", "#F39A27", "#976ED7", "#C23B23"];
        let mut index = 0;
        for (r, s, t) in &[(0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (10.0, 10.0, 10.0)] {
            let (a, b, c) = (2.0 / r, 2.0 / s, 2.0 / t);
            let dx = (index as f64) * 2.0;
            let dy = dx;
            for i in 0..n_theta {
                for j in 0..n_alpha {
                    xx[i][j] = math::suq_cos(the.get(i, j), a) * math::suq_cos(alp.get(i, j), a) + dx;
                    yy[i][j] = math::suq_cos(the.get(i, j), b) * math::suq_sin(alp.get(i, j), b) + dy;
                    zz[i][j] = math::suq_sin(the.get(i, j), c);
                }
            }
            let mut surf = Surface::new();
            surf.set_surf_color(colors[index]).draw(&xx, &yy, &zz);
            plot.add(&surf);
            index += 1;
        }
        let path = format!("{}/math_plot_functions_superquadric.svg", OUT_DIR);
        plot.set_equal_axes(true)
            .set_figure_size_points(800.0, 800.0)
            .save(path.as_str())?;
    }
    Ok(())
}
