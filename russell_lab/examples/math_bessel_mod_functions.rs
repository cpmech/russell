use plotpy::{Curve, Plot};
use russell_lab::math::GOLDEN_RATIO;
use russell_lab::*;

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    // values
    let xi = Vector::linspace(-3.0, 3.0, 101)?;
    let xk = Vector::linspace(0.5, 3.0, 101)?;
    let data = &[
        ("I", &xi, xi.get_mapped(|x| math::bessel_i0(x))),
        ("I", &xi, xi.get_mapped(|x| math::bessel_i1(x))),
        ("I", &xi, xi.get_mapped(|x| math::bessel_in(2, x))),
        ("K", &xk, xk.get_mapped(|x| math::bessel_k0(x))),
        ("K", &xk, xk.get_mapped(|x| math::bessel_k1(x))),
        ("K", &xk, xk.get_mapped(|x| math::bessel_kn(2, x))),
    ];
    // plots
    let colors = &["#E9708E", "#4C689C", "#58B090", "#F39A27", "#976ED7", "#C23B23"];
    for (k, dat) in data.iter().enumerate() {
        let (s, x, y) = dat;
        let n = k % 3;
        let mut curve = Curve::new();
        curve
            .set_label(format!("{}{}", s, n).as_str())
            .set_line_color(colors[k])
            .set_line_width(2.5)
            .draw(x.as_data(), y.as_data());
        let path = format!("{}/math_bessel_mod_functions_{}{}.svg", OUT_DIR, s.to_lowercase(), n);
        let mut plot = Plot::new();
        // if *s == "Y" && n == 2 {
        //     plot.set_yrange(-1.0, 0.5);
        // }
        plot.add(&curve)
            .grid_labels_legend("$x$", format!("${}_{}(x)$", s, n).as_str())
            .set_figure_size_points(GOLDEN_RATIO * 280.0, 280.0)
            .save(path.as_str())?;
    }
    Ok(())
}
