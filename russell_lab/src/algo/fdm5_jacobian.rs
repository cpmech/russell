use crate::StrError;
use crate::{deriv_central5, Matrix, Vector};

/// Computes the Jacobian matrix of a vector function using 5-point finite differences
///
/// Given the vector function:
///
/// ```text
/// {f}(x, {y})
/// ```
///
/// Calculate the (scaled) Jacobian matrix:
///
/// ```text
///                 ∂{f} │
/// [J](x, {y}) = α ———— │
///                 ∂{y} │(x=x_at, y=y_at)
/// ```
///
/// # Input
///
/// * `ndim` -- the dimension of the vector function
/// * `x_at` -- the scalar argument corresponding to the time/station when/where the Jacobian is to be computed
/// * `y_at` -- the vector argument corresponding to the time/station when/where the Jacobian is to be computed
/// * `function` -- the vector function `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)` where:
///     * `f` -- the result of the the function evaluation
///     * `x` -- the scalar argument (e.g., time)
///     * `y` -- the vector argument (e.g., position)
///     * `args` -- some additional arguments for the function
/// * `alpha` -- a scaling parameter to scale all elements of the Jacobian matrix
/// * `args` -- the additional arguments to be passed to the function
///
/// # Panics
///
/// A panic will occur if `function` returns an error.
///
/// # Examples
///
/// ```
/// use russell_lab::{algo, mat_approx_eq, Matrix, StrError, Vector};
///
/// fn main() -> Result<(), StrError> {
///     // arguments for the function
///     struct Args {}
///     let args = &mut Args {};
///
///     // current values
///     let x = 0.5;
///     let y = Vector::from(&[-1.0, 2.0]);
///
///     // scaling factor
///     let alpha = 2.0;
///
///     // analytical Jacobian
///     #[rustfmt::skip]
///     let jj_ana = Matrix::from(&[
///         [alpha * (1.0),  alpha * (-1.0)],
///         [alpha * (y[1]), alpha * (y[0])],
///     ]);
///
///     // numerical Jacobian
///     let jj_num = algo::fdm5_jacobian(y.dim(), x, &y, alpha, args, |f, x, y, _| {
///         f[0] = x + y[0] - y[1];
///         f[1] = y[0] * y[1];
///         Ok(())
///     });
///
///     // check the results
///     mat_approx_eq(&jj_ana, &jj_num, 1e-11);
///     Ok(())
/// }
/// ```
pub fn fdm5_jacobian<F, A>(ndim: usize, x_at: f64, y_at: &Vector, alpha: f64, args: &mut A, function: F) -> Matrix
where
    F: Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
{
    struct Extra {
        x: f64,    // "time"
        y: Vector, // "position"
        f: Vector, // output of f(x, y)
        i: usize,  // index i of ∂fᵢ/∂yⱼ
        j: usize,  // index j of ∂fᵢ/∂yⱼ
    }
    let mut extra = Extra {
        x: x_at,
        y: y_at.clone(),
        f: Vector::new(ndim),
        i: 0,
        j: 0,
    };
    let mut jac = Matrix::new(ndim, ndim);
    for i in 0..ndim {
        extra.i = i;
        for j in 0..ndim {
            extra.j = j;
            let res = deriv_central5(y_at[j], &mut extra, |yj: f64, extra: &mut Extra| {
                let original = extra.y[extra.j];
                extra.y[extra.j] = yj;
                function(&mut extra.f, extra.x, &extra.y, args).unwrap();
                extra.y[extra.j] = original;
                extra.f[extra.i]
            });
            jac.set(i, j, res * alpha);
        }
    }
    jac
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::fdm5_jacobian;
    use crate::{mat_approx_eq, Matrix, Vector};

    #[test]
    fn fdm5_jacobian_works() {
        struct Args {
            count: usize,
        }
        let args = &mut Args { count: 0 };
        let x = 1.5;
        let y = Vector::from(&[1.0, 2.0, 3.0]);
        let alpha = 2.0;
        let jj_ana = Matrix::from(&[
            [alpha * (1.0), alpha * (-1.0), alpha * (1.0)],
            [0.0, alpha * (x * y[2] * y[2]), alpha * (2.0 * x * y[1] * y[2])],
            [alpha * (-y[1] * y[2]), alpha * (-y[0] * y[2]), alpha * (-y[0] * y[1])],
        ]);
        let jj_num = fdm5_jacobian(y.dim(), x, &y, 2.0, args, |f, x, y, args| {
            args.count += 1;
            f[0] = x + y[0] - y[1] + y[2];
            f[1] = x * y[1] * y[2] * y[2];
            f[2] = x - y[0] * y[1] * y[2];
            Ok(())
        });
        mat_approx_eq(&jj_ana, &jj_num, 1e-10);
        assert_eq!(args.count, 36);
    }
}
