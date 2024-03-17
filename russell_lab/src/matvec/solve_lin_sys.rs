use crate::matrix::Matrix;
use crate::vector::Vector;
use crate::{to_i32, StrError};

extern "C" {
    // Computes the solution to a system of linear equations
    // <https://www.netlib.org/lapack/explore-html/d8/d72/dgesv_8f.html>
    fn c_dgesv(
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
}

/// (dgesv) Solves a general linear system (real numbers)
///
/// For a general matrix `a` (square, symmetric, non-symmetric, dense,
/// sparse), find `x` such that:
///
/// ```text
///   a   ⋅  x  =  b
/// (m,m)   (m)   (m)
/// ```
///
/// However, the right-hand-side will hold the solution:
///
/// ```text
/// b := a⁻¹⋅b == x
/// ```
///
/// The solution is obtained via LU decomposition using Lapack dgesv routine.
///
/// See also: <https://www.netlib.org/lapack/explore-html/d8/d72/dgesv_8f.html>
///
/// # Note
///
/// 1. The matrix `a` will be modified
/// 2. The right-hand-side `b` will contain the solution `x`
///
/// # Examples
///
/// ```
/// use russell_lab::{solve_lin_sys, Matrix, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // set matrix and right-hand side
///     let mut a = Matrix::from(&[
///         [1.0,  3.0, -2.0],
///         [3.0,  5.0,  6.0],
///         [2.0,  4.0,  3.0],
///     ]);
///     let mut b = Vector::from(&[5.0, 7.0, 8.0]);
///
///     // solve linear system b := a⁻¹⋅b
///     solve_lin_sys(&mut b, &mut a)?;
///
///     // check
///     let x_correct = "┌         ┐\n\
///                      │ -15.000 │\n\
///                      │   8.000 │\n\
///                      │   2.000 │\n\
///                      └         ┘";
///     assert_eq!(format!("{:.3}", b), x_correct);
///     Ok(())
/// }
/// ```
pub fn solve_lin_sys(b: &mut Vector, a: &mut Matrix) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if b.dim() != m {
        return Err("vector has wrong dimension");
    }
    if m == 0 {
        return Ok(());
    }
    let mut ipiv = vec![0; m];
    let m_i32 = to_i32(m);
    let nrhs = 1;
    let lda = to_i32(m);
    let ldb = lda;
    let mut info = 0;
    unsafe {
        c_dgesv(
            &m_i32,
            &nrhs,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            ipiv.as_mut_ptr(),
            b.as_mut_data().as_mut_ptr(),
            &ldb,
            &mut info,
        )
    }
    if info < 0 {
        println!("LAPACK ERROR (dgesv): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (dgesv): An argument had an illegal value");
    } else if info > 0 {
        println!("LAPACK ERROR (dgesv): U({},{}) is exactly zero", info - 1, info - 1);
        return Err("LAPACK ERROR (dgesv): The factorization has been completed, but the factor U is exactly singular");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{solve_lin_sys, Matrix, Vector};
    use crate::vec_approx_eq;

    #[test]
    fn solve_lin_sys_fails_on_non_square() {
        let mut a = Matrix::new(2, 3);
        let mut b = Vector::new(3);
        assert_eq!(solve_lin_sys(&mut b, &mut a), Err("matrix must be square"));
    }

    #[test]
    fn solve_lin_sys_fails_on_wrong_dims() {
        let mut a = Matrix::new(2, 2);
        let mut b = Vector::new(3);
        assert_eq!(solve_lin_sys(&mut b, &mut a), Err("vector has wrong dimension"));
    }

    #[test]
    fn solve_lin_sys_0x0_works() {
        let mut a = Matrix::new(0, 0);
        let mut b = Vector::new(0);
        solve_lin_sys(&mut b, &mut a).unwrap();
        assert_eq!(b.dim(), 0);
    }

    #[test]
    fn solve_lin_sys_works() {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            [2.0, 1.0, 1.0, 3.0, 2.0],
            [1.0, 2.0, 2.0, 1.0, 1.0],
            [1.0, 2.0, 9.0, 1.0, 5.0],
            [3.0, 1.0, 1.0, 7.0, 1.0],
            [2.0, 1.0, 5.0, 1.0, 8.0],
        ]);
        #[rustfmt::skip]
        let mut b = Vector::from(&[
            -2.0,
             4.0,
             3.0,
            -5.0,
             1.0,
        ]);
        solve_lin_sys(&mut b, &mut a).unwrap();
        #[rustfmt::skip]
        let x_correct = &[
            -629.0 / 98.0,
             237.0 / 49.0,
             -53.0 / 49.0,
              62.0 / 49.0,
              23.0 / 14.0,
        ];
        vec_approx_eq(b.as_data(), x_correct, 1e-13);
    }

    #[test]
    fn solve_lin_sys_1_works() {
        // example from https://numericalalgorithmsgroup.github.io/LAPACK_Examples/examples/doc/dgesv_example.html
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            [ 1.80,  2.88,  2.05, -0.89],
            [ 5.25, -2.95, -0.95, -3.80],
            [ 1.58, -2.69, -2.90, -1.04],
            [-1.11, -0.66, -0.59,  0.80],
        ]);
        #[rustfmt::skip]
        let mut b = Vector::from(&[
             9.52,
            24.35,
             0.77,
            -6.22,
        ]);
        solve_lin_sys(&mut b, &mut a).unwrap();
        #[rustfmt::skip]
        let x_correct = &[
             1.0,
            -1.0,
             3.0,
            -5.0,
        ];
        vec_approx_eq(b.as_data(), x_correct, 1e-14);
    }

    #[test]
    fn solve_lin_sys_singular_handles_error() {
        let mut a = Matrix::from(&[
            [0.0, 0.0], //
            [0.0, 1.0], //
        ]);
        let mut b = Vector::from(&[1.0, 1.0]);
        assert_eq!(
            solve_lin_sys(&mut b, &mut a).err(),
            Some("LAPACK ERROR (dgesv): The factorization has been completed, but the factor U is exactly singular")
        );
    }
}
