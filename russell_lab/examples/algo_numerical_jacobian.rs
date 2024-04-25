use russell_lab::math::PI;
use russell_lab::{mat_approx_eq, mat_scale, num_jacobian};
use russell_lab::{Matrix, StrError, Vector};

fn main() -> Result<(), StrError> {
    // Given the vector function:
    //
    // ```text
    // {f}(x, {y})
    // ```
    //
    // Calculate the (scaled) Jacobian matrix:
    //
    // ```text
    //                 ∂{f} │
    // [J](x, {y}) = α ———— │
    //                 ∂{y} │(x=x_at, y=y_at)
    // ```

    // arguments for the function
    struct Args {
        count: usize,
    }
    let args = &mut Args { count: 0 };

    // current time-position
    let x = 0.5;
    let y = Vector::from(&[PI, -PI / 2.0, PI / 4.0, PI / 2.0]);

    // function
    let function = |f: &mut Vector, x: f64, y: &Vector, args: &mut Args| {
        args.count += 1;
        f[0] = x + f64::sin(y[0]) - f64::cos(y[1]) + f64::exp(y[2]) - y[3];
        f[1] = x * y[1] * y[2] * y[2] + y[3];
        f[2] = x - y[0] + y[2];
        f[3] = y[3] - x;
        Ok(())
    };

    // (non-scaled) analytical Jacobian
    let mut jj_ana = Matrix::from(&[
        [f64::cos(y[0]), f64::sin(y[1]), f64::exp(y[2]), -1.0],
        [0.0, x * y[2] * y[2], 2.0 * x * y[1] * y[2], 1.0],
        [-1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    // scaling factor
    let alpha = 2.0;

    // scaled analytical Jacobian
    mat_scale(&mut jj_ana, alpha);

    // numerical Jacobian
    let jj_num = num_jacobian(y.dim(), x, &y, alpha, args, function)?;

    // check analytical versus numerical
    mat_approx_eq(&jj_ana, &jj_num, 1e-10);

    // print how many times function was called
    println!("count = {}", args.count);
    Ok(())
}
