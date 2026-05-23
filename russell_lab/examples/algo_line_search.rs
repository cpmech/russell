//! Demonstrates line search for optimization
//!
//! This example shows how to use the backtracking Armijo line search algorithm.
//!
//! Line search is typically used in conjunction with gradient-based optimization
//! methods (e.g., gradient descent, Newton's method) to find an appropriate step size.

use plotpy::{Curve, Legend, Plot};
use russell_lab::math::GOLDEN_RATIO;
use russell_lab::{StrError, Vector};

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    // Function: f(x) = (x-1)^4 + (x-1)^2, minimum at x=1
    // At x=0: f(0) = 2, gradient = 4*(0-1)^3 + 2*(0-1) = -4-2 = -6
    let f = |x: f64| {
        let d = x - 1.0;
        d.powi(4) + d.powi(2)
    };

    let x0 = 0.0_f64;
    let fx0 = f(x0); // 2.0
    let p = 1.0_f64; // descent direction (positive: toward minimum at x=1)
    let slope = -6.0_f64; // directional derivative grad_f(x0) * p

    // Use c1=0.5 (strict) so that alpha=1 is rejected and backtracking is visible.
    // In practice c1 is typically 1e-4; here 0.5 is chosen for illustration.
    let c1 = 0.5_f64;
    let rho = 0.5_f64;

    // Trace the backtracking manually to collect tried alpha values
    // alpha=1.0: f(1.0)=0, target=2+0.5*1*(-6)=-1 → 0 > -1 → rejected
    // alpha=0.5: f(0.5)=0.3125, target=2+0.5*0.5*(-6)=0.5 → 0.3125 ≤ 0.5 → accepted
    let mut tried_alphas: Vec<f64> = Vec::new();
    let mut accepted_alpha = 1.0_f64;
    let mut alpha = 1.0_f64;
    for _ in 0..20 {
        let x_try = x0 + alpha * p;
        let target = fx0 + c1 * alpha * slope;
        tried_alphas.push(alpha);
        if f(x_try) <= target {
            accepted_alpha = alpha;
            break;
        }
        alpha *= rho;
    }

    let x_new = x0 + accepted_alpha * p;
    let fx_new = f(x_new);
    let n_rejected = tried_alphas.len() - 1;

    println!("f(x) = (x-1)^4 + (x-1)^2  (minimum at x=1)");
    println!("Initial point : x0={:.4}  f(x0)={:.4}", x0, fx0);
    println!("Slope         : {:.4}", slope);
    println!("c1 (Armijo)   : {}", c1);
    println!("Tried alphas  : {:?}", tried_alphas);
    println!("Accepted alpha: {:.4}", accepted_alpha);
    println!("New point     : x_new={:.4}  f(x_new)={:.4}", x_new, fx_new);
    println!("Rejected steps: {}", n_rejected);

    // -------------------------------------------------------------------------
    // Plot
    // -------------------------------------------------------------------------

    // 1. Function curve
    let xx = Vector::linspace(-0.15, 1.65, 300)?;
    let yy = xx.get_mapped(|x| f(x));
    let mut curve_f = Curve::new();
    curve_f
        .set_label("$f(x)=(x-1)^4+(x-1)^2$")
        .set_line_color("C0")
        .set_line_width(2.5);
    curve_f.draw(xx.as_data(), yy.as_data());

    // 2. Tangent line (full linear model): y = f(x0) + slope*(x - x0)
    let x_end = 1.3_f64;
    let tan_xx = vec![x0, x_end];
    let tan_yy: Vec<f64> = tan_xx.iter().map(|&x| fx0 + slope * (x - x0)).collect();
    let mut curve_tan = Curve::new();
    curve_tan
        .set_label("tangent (linear model)")
        .set_line_style("--")
        .set_line_color("#888888")
        .set_line_width(1.5);
    curve_tan.draw(&tan_xx, &tan_yy);

    // 3. Armijo acceptance line: y = f(x0) + c1*slope*(x - x0)
    let arm_xx = vec![x0, x_end];
    let arm_yy: Vec<f64> = arm_xx.iter().map(|&x| fx0 + c1 * slope * (x - x0)).collect();
    let mut curve_arm = Curve::new();
    curve_arm
        .set_label(&format!("Armijo line ($c_1={:.1}$)", c1))
        .set_line_style("-.")
        .set_line_color("C2")
        .set_line_width(1.5);
    curve_arm.draw(&arm_xx, &arm_yy);

    // 4. Initial point
    let mut curve_init = Curve::new();
    curve_init
        .set_label(&format!("initial $(x_0={},\\ f_0={})$", x0, fx0))
        .set_line_style("None")
        .set_marker_style("o")
        .set_marker_size(10.0)
        .set_marker_color("C0")
        .set_marker_line_color("C0");
    curve_init.draw(&[x0], &[fx0]);

    // 5. Rejected points (all in one curve)
    let rej_xx: Vec<f64> = tried_alphas[..n_rejected].iter().map(|&a| x0 + a * p).collect();
    let rej_yy: Vec<f64> = rej_xx.iter().map(|&x| f(x)).collect();
    let mut curve_rej = Curve::new();
    curve_rej
        .set_label(&format!(
            "rejected ($\\alpha$={})",
            tried_alphas[..n_rejected]
                .iter()
                .map(|a| format!("{:.2}", a))
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .set_line_style("None")
        .set_marker_style("x")
        .set_marker_size(14.0)
        .set_marker_line_color("C3")
        .set_marker_color("C3");
    curve_rej.draw(&rej_xx, &rej_yy);

    // 6. Accepted (new) point
    let mut curve_new = Curve::new();
    curve_new
        .set_label(&format!(
            "accepted $(x_{{\\rm new}}={:.2},\\ \\alpha={:.1})$",
            x_new, accepted_alpha
        ))
        .set_line_style("None")
        .set_marker_style("*")
        .set_marker_size(16.0)
        .set_marker_color("C2")
        .set_marker_line_color("C2");
    curve_new.draw(&[x_new], &[fx_new]);

    let mut legend = Legend::new();
    legend.draw();

    let path = format!("{}/algo_line_search.svg", OUT_DIR);
    let mut plot = Plot::new();
    plot.add(&curve_f)
        .add(&curve_tan)
        .add(&curve_arm)
        .add(&curve_init)
        .add(&curve_rej)
        .add(&curve_new)
        .add(&legend)
        .set_yrange(-1.5, 2.5)
        .grid_and_labels("$x$", "$f(x)$")
        .set_figure_size_points(GOLDEN_RATIO * 450.0, 450.0)
        .save(&path)?;

    Ok(())
}
