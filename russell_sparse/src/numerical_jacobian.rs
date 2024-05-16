use crate::CooMatrix;
use crate::StrError;
use crate::Sym;
use russell_lab::Vector;

/// Computes the Jacobian matrix of a vector function using first-order finite differences
///
/// Given the vector function:
///
/// ```text
/// {f}(x, {y})
/// ```
///
/// Compute the (scaled) Jacobian matrix:
///
/// ```text
///                 ∂{f}
/// [J](x, {y}) = α ————
///                 ∂{y}
/// ```
///
/// The elements of the Jacobian matrix are approximated by:
///
/// ```text
///         Δfᵢ
/// Jᵢⱼ ≈ α ———
///         Δyⱼ
/// ```
///
/// # Notes
///
/// 1. `function` will be called `1 + ndim` times.
/// 2. The numerical Jacobian is only first-order accurate.
///
/// # Output
///
/// * `jj` -- Is the resulting numerical Jacobian matrix, which must be square, i.e., `nrow = ncol = ndim`.
///   The number of non-zeros must be at least `ndim * ndim` for [Sym::No] and [Sym::YesFull]
///   or at least `(ndim + ndim * ndim) / 2` for [Sym::YesLower] and [Sym::YesUpper]; i.e.,
///   the number of non-zeros (`max_nnz`) allocated in `jj` must satisfy the following constraints
///   (see the table below):
///
/// ```text
///           ⎧ (ndim + ndim²) / 2  if triangular
/// max_nnz ≥ ⎨
///           ⎩ ndim²               otherwise
/// ```
///
/// The above constraints are required because, even though there may be zero values,
/// the numerical algorithm cannot detect zero values in advance.
///
/// | matrix (`n = ndim`)                                          |`n`|`n²`|`(n + n²) / 2`|
/// |:-----------------------------------------------------------------:|:-:|:--:|:------------:|
/// |<pre>1</pre>                                                       | 1 |  1 |            1 |
/// |<pre>1 2<br>· 3</pre>                                              | 2 |  4 |            3 |
/// |<pre>1 2 3<br>· 4 5<br>· · 6</pre>                                 | 3 |  9 |            6 |
/// |<pre> 1  2  3  4<br> ·  5  6  7<br> ·  ·  8  9<br> ·  ·  · 10</pre>| 4 | 16 |           10 |
///
/// # Input
///
/// * `alpha` -- A coefficient to multiply all elements of the Jacobian
/// * `x` -- The station (e.g., time) where the `f` function is called
/// * `y` -- The vector `{y}` for which the `f` function is called.
///   **Note:** Although this variable is mutable, the original values are restored on exit
/// * `w1` -- A workspace vector with `len ≥ ndim`
/// * `w2` -- A workspace vector with `len ≥ ndim`
/// * `function` -- The `f(f: &mut Vector, x: f64, y: &Vector, args: &mut A)` function
/// * `args` -- Extra arguments for the `f` function
///
/// # Examples
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
///     let jacobian = |jj: &mut CooMatrix, alpha: f64, _x: f64, y: &Vector, _args: &mut Args| {
///         jj.reset();
///         jj.put(0, 0, alpha * (2.0)).unwrap();
///         jj.put(0, 1, alpha * (3.0 * y[2])).unwrap();
///         jj.put(0, 2, alpha * (3.0 * y[1])).unwrap();
///         jj.put(1, 1, alpha * (-3.0)).unwrap();
///         jj.put(2, 2, alpha * (2.0 * y[2])).unwrap();
///     };
///
///     let ndim = 3;
///     let x = 1.0;
///     let mut y = Vector::from(&[1.0, 2.0, 3.0]);
///     let alpha = 0.5;
///
///     let mut jj_ana = CooMatrix::new(ndim, ndim, 5, Sym::No).unwrap();
///     jacobian(&mut jj_ana, alpha, x, &y, &mut args);
///     let mat_jj_ana = jj_ana.as_dense();
///
///     let mut jj_num = CooMatrix::new(ndim, ndim, ndim * ndim, Sym::No).unwrap();
///     let mut w1 = Vector::new(ndim);
///     let mut w2 = Vector::new(ndim);
///     numerical_jacobian(&mut jj_num, alpha, x, &mut y, &mut w1, &mut w2, &mut args, function)?;
///     assert_eq!(args.n_function_calls, 1 + ndim);
///
///     let mat_jj_num = jj_num.as_dense();
///     mat_approx_eq(&mat_jj_num, &mat_jj_ana, 1e-7);
///     Ok(())
/// }
/// ```
pub fn numerical_jacobian<F, A>(
    jj: &mut CooMatrix,
    alpha: f64,
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
    if jj.symmetric.triangular() {
        if jj.max_nnz < (ndim + ndim * ndim) / 2 {
            return Err(
                "the max number of non-zero values in the numerical (triangular) Jacobian matrix must be at least (ndim + ndim²) / 2",
            );
        }
    } else {
        if jj.max_nnz < ndim * ndim {
            return Err("the max number of non-zero values in the numerical Jacobian matrix must be at least ndim²");
        }
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
        let (start, endp1) = match jj.symmetric {
            Sym::YesLower => (j, ndim),
            Sym::YesUpper => (0, j + 1),
            Sym::YesFull | Sym::No => (0, ndim),
        };
        for i in start..endp1 {
            let delta_fi = w2[i] - w1[i]; // Δfᵢ := Fᵢ - fᵢ
            jj.put(i, j, alpha * delta_fi / delta_yj).unwrap(); // Δfᵢ/Δyⱼ
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
        let nnz_wrong = 2; // nnz_correct = 3 = (ndim + ndim * ndim)/2
        let mut jj = CooMatrix::new(2, 2, nnz_wrong, Sym::YesLower).unwrap();
        assert_eq!(
            numerical_jacobian(&mut jj, 2.0, 1.0, &mut y, &mut w1, &mut w2, &mut args, function).err(),
            Some("the max number of non-zero values in the numerical (triangular) Jacobian matrix must be at least (ndim + ndim²) / 2")
        );
        let mut jj = CooMatrix::new(2, 2, nnz_wrong, Sym::YesUpper).unwrap();
        assert_eq!(
            numerical_jacobian(&mut jj, 2.0, 1.0, &mut y, &mut w1, &mut w2, &mut args, function).err(),
            Some("the max number of non-zero values in the numerical (triangular) Jacobian matrix must be at least (ndim + ndim²) / 2")
        );
        let nnz_wrong = 3; // nnz_correct = 4 = ndim * ndim
        let mut jj = CooMatrix::new(2, 2, nnz_wrong, Sym::No).unwrap();
        assert_eq!(
            numerical_jacobian(&mut jj, 2.0, 1.0, &mut y, &mut w1, &mut w2, &mut args, function).err(),
            Some("the max number of non-zero values in the numerical Jacobian matrix must be at least ndim²")
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

        let function = |f: &mut Vector, _x: f64, y: &Vector, a: &mut Args| {
            f[0] = 2.0 * y[0] + 3.0 * y[1] * y[2] - 4.0 * f64::cos(y[3]);
            f[1] = -3.0 * y[1] - 4.0 * f64::exp(y[3] / (1.0 + y[1]));
            f[2] = y[2] * y[2];
            f[3] = -y[0] + 5.0 * (1.0 - y[0] * y[0]) * y[1] - 6.0 * y[2];
            a.n_function_calls += 1;
            Ok(())
        };

        let jacobian = |jj: &mut CooMatrix, alpha: f64, _x: f64, y: &Vector, _a: &mut Args| {
            let d = 1.0 + y[1];
            let e = f64::exp(y[3] / d);
            let dd = d * d;

            jj.reset();

            jj.put(0, 0, alpha * (2.0)).unwrap();
            jj.put(0, 1, alpha * (3.0 * y[2])).unwrap();
            jj.put(0, 2, alpha * (3.0 * y[1])).unwrap();
            jj.put(0, 3, alpha * (4.0 * f64::sin(y[3]))).unwrap();

            jj.put(1, 1, alpha * (-3.0 + 4.0 * e * y[3] / dd)).unwrap();
            jj.put(1, 3, alpha * (-4.0 * e / d)).unwrap();

            jj.put(2, 2, alpha * (2.0 * y[2])).unwrap();

            jj.put(3, 0, alpha * (-1.0 - 10.0 * y[0] * y[1])).unwrap();
            jj.put(3, 1, alpha * (5.0 * (1.0 - y[0] * y[0]))).unwrap();
            jj.put(3, 2, alpha * (-6.0)).unwrap();
        };

        let ndim = 4;
        let x = 1.0;
        let mut y = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let alpha = 0.5;

        let mut jj_ana = CooMatrix::new(ndim, ndim, 10, Sym::No).unwrap();
        jacobian(&mut jj_ana, alpha, x, &y, &mut args);
        let mat_jj_ana = jj_ana.as_dense();
        // println!("analytical:\n{}", mat_jj_ana);

        let mut jj_num = CooMatrix::new(ndim, ndim, ndim * ndim, Sym::No).unwrap();
        let mut w1 = Vector::new(ndim);
        let mut w2 = Vector::new(ndim);
        numerical_jacobian(&mut jj_num, alpha, x, &mut y, &mut w1, &mut w2, &mut args, function).unwrap();
        assert_eq!(args.n_function_calls, 1 + ndim);

        let mat_jj_num = jj_num.as_dense();
        // println!("numerical:\n{}", mat_jj_num);
        mat_approx_eq(&mat_jj_num, &mat_jj_ana, 1e-6);
    }

    #[test]
    fn numerical_jacobian_symmetric_works() {
        struct Args {
            n_function_calls: usize,
        }
        let mut args = Args { n_function_calls: 0 };

        // system
        let function = |f: &mut Vector, x: f64, y: &Vector, a: &mut Args| {
            f[0] = -y[0] + y[1];
            f[1] = y[0] + y[1];
            f[2] = 1.0 / (1.0 + x);
            a.n_function_calls += 1;
            Ok(())
        };

        // ```text
        //          ┌          ┐   ┌          ┐
        //     df   │ -1  1  0 │   │ -1  *  * │
        // J = —— = │  1  1  0 │ = │  1  1  * │
        //     dy   │  0  0  0 │   │  0  0  0 │
        //          └          ┘   └          ┘
        // ```
        let jacobian = |jj: &mut CooMatrix, alpha: f64, _x: f64, _y: &Vector, _a: &mut Args| {
            jj.reset();
            jj.put(0, 0, alpha * (-1.0)).unwrap();
            jj.put(1, 1, alpha * (1.0)).unwrap();
            if jj.symmetric == Sym::YesLower {
                jj.put(1, 0, alpha * (1.0)).unwrap();
            } else {
                jj.put(0, 1, alpha * (1.0)).unwrap();
            }
        };

        let ndim = 3;
        let jac_nnz = 3;
        let x = 1.0;
        let mut y = Vector::from(&[1.0, 2.0, 3.0]);
        let alpha = 0.5;

        // lower triangular -----------------------------------------------------------
        let symmetry = Sym::YesLower;

        let mut jj_ana = CooMatrix::new(ndim, ndim, jac_nnz, symmetry).unwrap();
        jacobian(&mut jj_ana, alpha, x, &y, &mut args);
        let mat_jj_ana = jj_ana.as_dense();
        println!("analytical:\n{}", mat_jj_ana);

        let jac_num_nnz = (ndim + ndim * ndim) / 2;
        let mut jj_num = CooMatrix::new(ndim, ndim, jac_num_nnz, symmetry).unwrap();
        let mut w1 = Vector::new(ndim);
        let mut w2 = Vector::new(ndim);
        numerical_jacobian(&mut jj_num, alpha, x, &mut y, &mut w1, &mut w2, &mut args, function).unwrap();
        assert_eq!(args.n_function_calls, 1 + ndim);

        let mat_jj_num = jj_num.as_dense();
        println!("numerical:\n{}", mat_jj_num);
        mat_approx_eq(&mat_jj_num, &mat_jj_ana, 1e-8);

        // upper triangular -----------------------------------------------------------
        args.n_function_calls = 0;
        let symmetry = Sym::YesUpper;

        let mut jj_ana = CooMatrix::new(ndim, ndim, jac_nnz, symmetry).unwrap();
        jacobian(&mut jj_ana, alpha, x, &y, &mut args);
        let mat_jj_ana = jj_ana.as_dense();
        println!("analytical:\n{}", mat_jj_ana);

        let jac_num_nnz = (ndim + ndim * ndim) / 2;
        let mut jj_num = CooMatrix::new(ndim, ndim, jac_num_nnz, symmetry).unwrap();
        let mut w1 = Vector::new(ndim);
        let mut w2 = Vector::new(ndim);
        numerical_jacobian(&mut jj_num, alpha, x, &mut y, &mut w1, &mut w2, &mut args, function).unwrap();
        assert_eq!(args.n_function_calls, 1 + ndim);

        let mat_jj_num = jj_num.as_dense();
        println!("numerical:\n{}", mat_jj_num);
        mat_approx_eq(&mat_jj_num, &mat_jj_ana, 1e-8);
    }
}
