use crate::CooMatrix;
use crate::StrError;
use russell_lab::vec_copy;
use russell_lab::Vector;

/// Computes the Jacobian matrix of a vector function using first-order finite differences
///
/// Given the vector function:
///
/// ```text
/// {f}(x, {y})
/// ```
///
/// Compute the Jacobian matrix:
///
/// ```text
///                            ∂{f}
/// [J](x, {y}) = multiplier · ————
///                            ∂{y}
/// ```
///
/// The elements of the Jacobian matrix are approximated by:
///
/// ```text
///                    Δfᵢ
/// Jᵢⱼ ≈ multiplier · ———
///                    Δyⱼ
/// ```
///
/// **Note:** `function` will be called `1 + ndim` times.
///
/// # Output
///
/// * `jj` -- Is the resulting numerical Jacobian matrix, which must be square, i.e., `nrow = ncol = ndim`.
///   The condition `max_nnz ≥ ndim · ndim` is required even though there may be zero values.
///   Note that the numerical algorithm cannot know which elements are zero in advance.
///
/// # Input
///
/// * `multiplier` -- A coefficient to multiply all elements of the Jacobian
/// * `x` -- The station (e.g., time) where the `f` function is called
/// * `y` -- The vector `{y}` for which the `f` function is called
/// * `f` -- A workspace to calculate the `{f}` vector @ `(x, y)`
/// * `y_new` -- A workspace to calculate `y_new = y + Δy`
/// * `f_new` -- A workspace to calculate the `{f_new}` vector @ `(x, {y_new})`
/// * `function` -- The `f(f: &mut Vector, x: f64, y: &Vector, args: &mut A)` function
/// * `args` -- Extra arguments for the `f` function
pub fn numerical_jacobian<F, A>(
    jj: &mut CooMatrix,
    multiplier: f64,
    x: f64,
    y: &Vector,
    f: &mut Vector,
    y_new: &mut Vector,
    f_new: &mut Vector,
    mut function: F,
    args: &mut A,
) -> Result<(), StrError>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
{
    if jj.nrow != jj.ncol {
        return Err("the Jacobian matrix must be square");
    }
    let ndim = jj.nrow;
    if jj.max_nnz < ndim * ndim {
        return Err("the max number of non-zero values in the numerical Jacobian matrix must be at least ndim * ndim");
    }
    if y_new.dim() != ndim || f.dim() != ndim || f_new.dim() != ndim {
        return Err("the workspace vectors must be ndim");
    }
    const THRESHOLD: f64 = 1e-5;
    function(f, x, y, args)?; // f := f(x, y)
    vec_copy(y_new, &y).unwrap(); // Y := y
    jj.reset();
    for j in 0..ndim {
        let delta_yj = f64::sqrt(f64::EPSILON * f64::max(THRESHOLD, f64::abs(y[j])));
        y_new[j] = y[j] + delta_yj; // Yⱼ := yⱼ + Δyⱼ
        function(f_new, x, y_new, args)?; // F := f(x, y + Δy)
        for i in 0..ndim {
            let delta_fi = f_new[i] - f[i]; // Δfᵢ := Fᵢ - fᵢ
            jj.put(i, j, multiplier * delta_fi / delta_yj).unwrap(); // Δfᵢ/Δyⱼ
        }
        y_new[j] = y[j]; // restore value: Yⱼ := yⱼ
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::numerical_jacobian;
    use crate::{CooMatrix, Sym};
    use russell_lab::{mat_approx_eq, Vector};

    #[test]
    fn numerical_jacobian_works() {
        struct Args {
            n_function_calls: usize,
        }
        let mut args = Args { n_function_calls: 0 };

        let function = |f: &mut Vector, _x: f64, y: &Vector, args: &mut Args| {
            f[0] = 2.0 * y[0] + 3.0 * y[1] * y[2] - 4.0 * f64::cos(y[3]);
            f[1] = -3.0 * y[1] - 4.0 * f64::exp(y[3] / (1.0 + y[1]));
            f[2] = y[2] * y[2];
            f[3] = -y[0] + 5.0 * (1.0 - y[0] * y[0]) * y[1] - 6.0 * y[2];
            args.n_function_calls += 1;
            Ok(())
        };

        let jacobian = |jj: &mut CooMatrix, m: f64, _x: f64, y: &Vector, _args: &mut Args| {
            let d = 1.0 + y[1];
            let e = f64::exp(y[3] / d);
            let dd = d * d;

            jj.reset();

            jj.put(0, 0, m * (2.0)).unwrap();
            jj.put(0, 1, m * (3.0 * y[2])).unwrap();
            jj.put(0, 2, m * (3.0 * y[1])).unwrap();
            jj.put(0, 3, m * (4.0 * f64::sin(y[3]))).unwrap();

            jj.put(1, 1, m * (-3.0 + 4.0 * e * y[3] / dd)).unwrap();
            jj.put(1, 3, m * (-4.0 * e / d)).unwrap();

            jj.put(2, 2, m * (2.0 * y[2])).unwrap();

            jj.put(3, 0, m * (-1.0 - 10.0 * y[0] * y[1])).unwrap();
            jj.put(3, 1, m * (5.0 * (1.0 - y[0] * y[0]))).unwrap();
            jj.put(3, 2, m * (-6.0)).unwrap();
        };

        let ndim = 4;
        let x = 1.0;
        let y = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let multiplier = 0.5;

        let mut jj_ana = CooMatrix::new(ndim, ndim, 10, Sym::No).unwrap();
        jacobian(&mut jj_ana, multiplier, x, &y, &mut args);
        let mat_jj_ana = jj_ana.as_dense();
        // println!("analytical:\n{}", mat_jj_ana);

        let mut jj_num = CooMatrix::new(ndim, ndim, ndim * ndim, Sym::No).unwrap();
        let mut f = Vector::new(ndim);
        let mut y_new = Vector::new(ndim);
        let mut f_new = Vector::new(ndim);
        numerical_jacobian(
            &mut jj_num,
            multiplier,
            x,
            &y,
            &mut f,
            &mut y_new,
            &mut f_new,
            function,
            &mut args,
        )
        .unwrap();
        assert_eq!(args.n_function_calls, 1 + ndim);

        let mat_jj_num = jj_num.as_dense();
        // println!("numerical:\n{}", mat_jj_num);
        mat_approx_eq(&mat_jj_num, &mat_jj_ana, 1e-6);
    }
}
