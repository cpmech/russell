use crate::CooMatrix;
use crate::StrError;
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
/// * `y` -- The vector `{y}` for which the `f` function is called.
///   **Note:** Although this variable is mutable, the original values are restored on exit
/// * `w1` -- A workspace vector with `len ≥ ndim`
/// * `w2` -- A workspace vector with `len ≥ ndim`
/// * `function` -- The `f(f: &mut Vector, x: f64, y: &Vector, args: &mut A)` function
/// * `args` -- Extra arguments for the `f` function
///
/// # Example
///
/// ```
/// use russell_lab::mat_approx_eq;
/// use russell_lab::Vector;
/// use russell_sparse::prelude::*;
/// use russell_sparse::StrError;
///
/// fn main() -> Result<(), StrError> {
///     struct Args {
///         n_function_calls: usize,
///     }
///     let mut args = Args { n_function_calls: 0 };
///
///     let function = |f: &mut Vector, _x: f64, y: &Vector, args: &mut Args| {
///         f[0] = 2.0 * y[0] + 3.0 * y[1] * y[2];
///         f[1] = -3.0 * y[1];
///         f[2] = y[2] * y[2];
///         args.n_function_calls += 1;
///         Ok(())
///     };
///
///     let jacobian = |jj: &mut CooMatrix, m: f64, _x: f64, y: &Vector, _args: &mut Args| {
///         jj.reset();
///         jj.put(0, 0, m * (2.0)).unwrap();
///         jj.put(0, 1, m * (3.0 * y[2])).unwrap();
///         jj.put(0, 2, m * (3.0 * y[1])).unwrap();
///         jj.put(1, 1, m * (-3.0)).unwrap();
///         jj.put(2, 2, m * (2.0 * y[2])).unwrap();
///     };
///
///     let ndim = 3;
///     let x = 1.0;
///     let mut y = Vector::from(&[1.0, 2.0, 3.0]);
///     let multiplier = 0.5;
///
///     let mut jj_ana = CooMatrix::new(ndim, ndim, 5, Sym::No).unwrap();
///     jacobian(&mut jj_ana, multiplier, x, &y, &mut args);
///     let mat_jj_ana = jj_ana.as_dense();
///
///     let mut jj_num = CooMatrix::new(ndim, ndim, ndim * ndim, Sym::No).unwrap();
///     let mut w1 = Vector::new(ndim);
///     let mut w2 = Vector::new(ndim);
///     numerical_jacobian(
///         &mut jj_num,
///         multiplier,
///         x,
///         &mut y,
///         &mut w1,
///         &mut w2,
///         &mut args,
///         function,
///     )
///     .unwrap();
///     assert_eq!(args.n_function_calls, 1 + ndim);
///
///     let mat_jj_num = jj_num.as_dense();
///     mat_approx_eq(&mat_jj_num, &mat_jj_ana, 1e-7);
///     Ok(())
/// }
/// ```
pub fn numerical_jacobian<F, A>(
    jj: &mut CooMatrix,
    multiplier: f64,
    x: f64,
    y: &mut Vector,
    w1: &mut Vector,
    w2: &mut Vector,
    args: &mut A,
    mut function: F,
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
    if y.dim() != ndim {
        return Err("the y-vector must have dim = ndim");
    }
    if w1.dim() < ndim || w2.dim() < ndim {
        return Err("the workspace vectors must have dim ≥ ndim");
    }
    const THRESHOLD: f64 = 1e-5;
    function(w1, x, y, args)?; // w1 := f(x, y)
    jj.reset();
    for j in 0..ndim {
        let original_yj = y[j];
        let delta_yj = f64::sqrt(f64::EPSILON * f64::max(THRESHOLD, f64::abs(y[j])));
        y[j] += delta_yj; // Yⱼ := yⱼ + Δyⱼ
        function(w2, x, y, args)?; // F := f(x, y + Δy)
        for i in 0..ndim {
            let delta_fi = w2[i] - w1[i]; // Δfᵢ := Fᵢ - fᵢ
            jj.put(i, j, multiplier * delta_fi / delta_yj).unwrap(); // Δfᵢ/Δyⱼ
        }
        y[j] = original_yj; // restore yⱼ
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
    fn numerical_jacobian_captures_errors() {
        struct Args {}
        let mut args = Args {};
        let mut jj = CooMatrix::new(2, 3, 1, Sym::No).unwrap();
        let mut y = Vector::new(2);
        let mut w1 = Vector::new(2);
        let mut w2 = Vector::new(2);
        let function = |_f: &mut Vector, _x: f64, _y: &Vector, _args: &mut Args| Ok(());
        assert_eq!(
            numerical_jacobian(&mut jj, 2.0, 1.0, &mut y, &mut w1, &mut w2, &mut args, function).err(),
            Some("the Jacobian matrix must be square")
        );
        let mut jj = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        assert_eq!(
            numerical_jacobian(&mut jj, 2.0, 1.0, &mut y, &mut w1, &mut w2, &mut args, function).err(),
            Some("the y-vector must have dim = ndim")
        );
        let mut y = Vector::new(1);
        assert_eq!(
            numerical_jacobian(&mut jj, 2.0, 1.0, &mut y, &mut w1, &mut w2, &mut args, function).err(),
            None
        );
        let mut w1 = Vector::new(0);
        assert_eq!(
            numerical_jacobian(&mut jj, 2.0, 1.0, &mut y, &mut w1, &mut w2, &mut args, function).err(),
            Some("the workspace vectors must have dim ≥ ndim")
        );
        let mut w1 = Vector::new(2);
        let mut w2 = Vector::new(0);
        assert_eq!(
            numerical_jacobian(&mut jj, 2.0, 1.0, &mut y, &mut w1, &mut w2, &mut args, function).err(),
            Some("the workspace vectors must have dim ≥ ndim")
        );
    }

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
        let mut y = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let multiplier = 0.5;

        let mut jj_ana = CooMatrix::new(ndim, ndim, 10, Sym::No).unwrap();
        jacobian(&mut jj_ana, multiplier, x, &y, &mut args);
        let mat_jj_ana = jj_ana.as_dense();
        // println!("analytical:\n{}", mat_jj_ana);

        let mut jj_num = CooMatrix::new(ndim, ndim, ndim * ndim, Sym::No).unwrap();
        let mut w1 = Vector::new(ndim);
        let mut w2 = Vector::new(ndim);
        numerical_jacobian(
            &mut jj_num,
            multiplier,
            x,
            &mut y,
            &mut w1,
            &mut w2,
            &mut args,
            function,
        )
        .unwrap();
        assert_eq!(args.n_function_calls, 1 + ndim);

        let mat_jj_num = jj_num.as_dense();
        // println!("numerical:\n{}", mat_jj_num);
        mat_approx_eq(&mat_jj_num, &mat_jj_ana, 1e-6);
    }
}
