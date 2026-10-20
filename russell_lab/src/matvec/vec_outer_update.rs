use crate::StrError;
use crate::matrix::Matrix;
use crate::vector::Vector;

/// (dger) Performs the outer (tensor) product between two vectors resulting in a matrix (with update)
///
/// ```text
///   a  +=   α ⋅ u  outer  v
/// (m,n)        (m)       (n)
/// ```
///
/// See also: <https://www.netlib.org/lapack/explore-html/dc/da8/dger_8f.html>
///
/// # Note
///
/// The rows of matrix a must equal the length of vector u and
/// the columns of matrix a must equal the length of vector v
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_outer_update, Matrix, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[1.0, 2.0, 3.0]);
///     let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
///     let mut a = Matrix::from(&[
///         [1000.0, 0.0, 10.0, 100.0],
///         [2000.0, 0.0, 20.0, 200.0],
///         [3000.0, 0.0, 30.0, 300.0],
///     ]);
///     vec_outer_update(&mut a, 1.0, &u, &v)?;
///     let correct = "┌                     ┐\n\
///                    │ 1005   -2   10  101 │\n\
///                    │ 2010   -4   20  202 │\n\
///                    │ 3015   -6   30  303 │\n\
///                    └                     ┘";
///     assert_eq!(format!("{}", a), correct);
///     Ok(())
/// }
/// ```
pub fn vec_outer_update(a: &mut Matrix, alpha: f64, u: &Vector, v: &Vector) -> Result<(), StrError> {
    let m = u.dim();
    let n = v.dim();
    if a.nrow() != m || a.ncol() != n {
        return Err("matrix and vectors are incompatible");
    }
    // One dispatch for both backends: `dger` in the default build, the linked
    // library's through the shim otherwise. CBLAS's `dger` is the unconjugated
    // rank-1 update, which is what BLAS's own `dger` is and what the native
    // kernel computes.
    crate::base::blas_dispatch::dger(alpha, u.as_data(), v.as_data(), a.as_mut_data(), m, n);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Matrix, Vector, vec_outer_update};
    use crate::mat_approx_eq;

    #[test]
    fn mat_vec_mul_fail_on_wrong_dims() {
        let u = Vector::new(2);
        let v = Vector::new(3);
        let mut a_1x3 = Matrix::new(1, 3);
        let mut a_2x1 = Matrix::new(2, 1);
        assert_eq!(
            vec_outer_update(&mut a_1x3, 1.0, &u, &v),
            Err("matrix and vectors are incompatible")
        );
        assert_eq!(
            vec_outer_update(&mut a_2x1, 1.0, &u, &v),
            Err("matrix and vectors are incompatible")
        );
    }

    #[test]
    fn vec_outer_update_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
        let (m, n) = (u.dim(), v.dim());
        let mut a = Matrix::new(m, n);
        vec_outer_update(&mut a, 3.0, &u, &v).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [15.0,  -6.0, 0.0, 3.0],
            [30.0, -12.0, 0.0, 6.0],
            [45.0, -18.0, 0.0, 9.0],
        ];
        mat_approx_eq(&a, correct, 1e-15);
    }

    #[test]
    fn vec_outer_update_works_1() {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let v = Vector::from(&[1.0, 1.0, -2.0]);
        let (m, n) = (u.dim(), v.dim());
        let mut a = Matrix::new(m, n);
        vec_outer_update(&mut a, 2.0, &u, &v).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [2.0, 2.0,  -4.0],
            [4.0, 4.0,  -8.0],
            [6.0, 6.0, -12.0],
            [8.0, 8.0, -16.0],
        ];
        mat_approx_eq(&a, correct, 1e-15);

        // call it again to make sure that summation is happening
        #[rustfmt::skip]
        let correct2 = &[
            [ 4.0,  4.0,  -8.0],
            [ 8.0,  8.0, -16.0],
            [12.0, 12.0, -24.0],
            [16.0, 16.0, -32.0],
        ];
        vec_outer_update(&mut a, 2.0, &u, &v).unwrap();
        mat_approx_eq(&a, correct2, 1e-15);
    }
}
