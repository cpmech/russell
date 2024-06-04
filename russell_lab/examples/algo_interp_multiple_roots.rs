use plotpy::{Curve, Legend, Plot};
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // function
    let f = |x| x * x - 1.0;
    let (xa, xb) = (-4.0, 4.0);

    // interpolant
    let degree = 2;
    let npoint = degree + 1;
    let mut params = InterpParams::new();
    params.no_eta_normalization = true;
    let interp = InterpLagrange::new(degree, Some(params)).unwrap();

    // compute data points
    let mut uu = Vector::new(npoint);
    let yy = interp.get_points();
    for (i, y) in yy.into_iter().enumerate() {
        let x = (xb + xa + (xb - xa) * y) / 2.0;
        uu[i] = f(x);
    }
    println!("U = \n{}", uu);

    // interpolation
    let lambda = interp.get_lambda();
    let p_interp = |x| {
        let y = (2.0 * x - xb - xa) / (xb - xa);
        let mut prod = 1.0;
        for i in 0..npoint {
            prod *= y - yy[i];
        }
        let mut sum = 0.0;
        for i in 0..npoint {
            let wi = lambda[i];
            sum += (wi * uu[i]) / (y - yy[i]);
        }
        prod * sum
    };

    // companion matrix
    let np = npoint;
    let na = 1 + np;
    let mut aa = Matrix::new(na, na);
    let mut bb = Matrix::new(na, na);
    for k in 0..np {
        let wk = lambda[k];
        aa.set(0, 1 + k, -uu[k]);
        aa.set(1 + k, 0, wk);
        aa.set(1 + k, 1 + k, yy[k]);
        bb.set(1 + k, 1 + k, 1.0);
    }
    println!("A =\n{:.3}", aa);
    println!("B =\n{:.3}", bb);

    // interpolation using the companion matrix
    let mut cc = Matrix::new(na, na);
    let mut cc_inv = Matrix::new(na, na);
    let mut p_companion = |x, a_mat, b_mat| {
        // C := y B - A
        let y = (2.0 * x - xb - xa) / (xb - xa);
        mat_add(&mut cc, y, b_mat, -1.0, a_mat).unwrap();
        let det = mat_inverse(&mut cc_inv, &cc).unwrap();
        det
    };

    // generalized eigenvalues
    let mut alpha_real = Vector::new(na);
    let mut alpha_imag = Vector::new(na);
    let mut beta = Vector::new(na);
    let mut v = Matrix::new(na, na);
    mat_gen_eigen(&mut alpha_real, &mut alpha_imag, &mut beta, &mut v, &mut aa, &mut bb)?;

    // print the results
    println!("Re(α) =\n{}", alpha_real);
    println!("Im(α) =\n{}", alpha_imag);
    println!("β =\n{}", beta);
    // println!("v =\n{:.2}", v);

    // roots = real eigenvalues
    let mut roots = Vector::new(na);
    let mut nroot = 0;
    for i in 0..na {
        let imaginary = f64::abs(alpha_imag[i]) > f64::EPSILON;
        let infinite = f64::abs(beta[i]) < 10.0 * f64::EPSILON;
        if !imaginary && !infinite {
            let y_root = alpha_real[i] / beta[i];
            roots[nroot] = (xb + xa + (xb - xa) * y_root) / 2.0;
            nroot += 1;
        }
    }
    println!("nroot = {}", nroot);
    for i in 0..nroot {
        println!("root # {} = {}", i, roots[i]);
    }

    // plot
    let x_original = Vector::linspace(xa, xb, 101).unwrap();
    let y_original = x_original.get_mapped(|x| f(x));
    let y_interp = x_original.get_mapped(|x| p_interp(x));
    let y_companion = x_original.get_mapped(|x| p_companion(x, &aa, &bb));
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    let mut curve3 = Curve::new();
    curve1
        .set_label("original")
        .set_line_color("grey")
        .set_line_width(15.0)
        .draw(x_original.as_data(), y_original.as_data());
    curve2
        .set_label("interpolated")
        .set_line_color("yellow")
        .set_line_width(7.0)
        .draw(x_original.as_data(), y_interp.as_data());
    curve3
        .set_label("companion")
        .set_line_color("black")
        .draw(x_original.as_data(), y_companion.as_data());
    let mut plot = Plot::new();
    let path = "/tmp/russell_lab/algo_interp_multiple_roots.svg";
    let mut legend = Legend::new();
    legend.set_outside(true).set_num_col(3);
    legend.draw();
    plot.set_cross(0.0, 0.0, "grey", "-", 1.5)
        .add(&curve1)
        .add(&curve2)
        .add(&curve3)
        .add(&legend)
        .grid_and_labels("$x$", "$f(x)$")
        .save(path)
        .unwrap();

    Ok(())
}
