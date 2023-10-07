use crate::matrix::ComplexMatrix;
use crate::vector::ComplexVector;
use crate::{to_i32, StrError};
use num_complex::Complex64;

extern "C" {
    // Computes the solution to a system of linear equations (complex version)
    // <http://www.netlib.org/lapack/explore-html/d1/ddc/zgesv_8f.html>
    fn c_zgesv(
        n: *const i32,
        nrhs: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut Complex64,
        ldb: *const i32,
        info: *mut i32,
    );
}

/// Solves a general linear system (complex version)
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
pub fn complex_solve_lin_sys(b: &mut ComplexVector, a: &mut ComplexMatrix) -> Result<(), StrError> {
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
        c_zgesv(
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
    use super::complex_solve_lin_sys;
    use crate::{complex_vec_approx_eq, ComplexMatrix, ComplexVector};
    use num_complex::Complex64;

    #[test]
    fn solve_lin_sys_fails_on_non_square() {
        let mut a = ComplexMatrix::new(2, 3);
        let mut b = ComplexVector::new(3);
        assert_eq!(complex_solve_lin_sys(&mut b, &mut a), Err("matrix must be square"));
    }

    #[test]
    fn complex_solve_lin_sys_fails_on_wrong_dims() {
        let mut a = ComplexMatrix::new(2, 2);
        let mut b = ComplexVector::new(3);
        assert_eq!(complex_solve_lin_sys(&mut b, &mut a), Err("vector has wrong dimension"));
    }

    #[test]
    fn complex_solve_lin_sys_0x0_works() {
        let mut a = ComplexMatrix::new(0, 0);
        let mut b = ComplexVector::new(0);
        complex_solve_lin_sys(&mut b, &mut a).unwrap();
        assert_eq!(b.dim(), 0);
    }

    #[test]
    fn complex_solve_lin_sys_works() {
        #[rustfmt::skip]
        let mut a = ComplexMatrix::from(&[
            [2.0, 1.0, 1.0, 3.0, 2.0],
            [1.0, 2.0, 2.0, 1.0, 1.0],
            [1.0, 2.0, 9.0, 1.0, 5.0],
            [3.0, 1.0, 1.0, 7.0, 1.0],
            [2.0, 1.0, 5.0, 1.0, 8.0],
        ]);
        #[rustfmt::skip]
        let mut b = ComplexVector::from(&[
            -2.0,
             4.0,
             3.0,
            -5.0,
             1.0,
        ]);
        complex_solve_lin_sys(&mut b, &mut a).unwrap();
        #[rustfmt::skip]
        let x_correct = &[
            Complex64::new(-629.0 / 98.0, 0.0),
            Complex64::new( 237.0 / 49.0, 0.0),
            Complex64::new( -53.0 / 49.0, 0.0),
            Complex64::new(  62.0 / 49.0, 0.0),
            Complex64::new(  23.0 / 14.0, 0.0),
        ];
        complex_vec_approx_eq(b.as_data(), x_correct, 1e-13);
    }

    #[test]
    fn complex_solve_lin_sys_1_works() {
        // example from https://numericalalgorithmsgroup.github.io/LAPACK_Examples/examples/doc/dgesv_example.html
        #[rustfmt::skip]
        let mut a = ComplexMatrix::from(&[
            [ 1.80,  2.88,  2.05, -0.89],
            [ 5.25, -2.95, -0.95, -3.80],
            [ 1.58, -2.69, -2.90, -1.04],
            [-1.11, -0.66, -0.59,  0.80],
        ]);
        #[rustfmt::skip]
        let mut b = ComplexVector::from(&[
             9.52,
            24.35,
             0.77,
            -6.22,
        ]);
        complex_solve_lin_sys(&mut b, &mut a).unwrap();
        #[rustfmt::skip]
        let x_correct = &[
            Complex64::new( 1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new( 3.0, 0.0),
            Complex64::new(-5.0, 0.0),
        ];
        complex_vec_approx_eq(b.as_data(), x_correct, 1e-14);
    }

    #[test]
    fn complex_solve_lin_sys_singular_handles_error() {
        let mut a = ComplexMatrix::from(&[
            [0.0, 0.0], //
            [0.0, 1.0], //
        ]);
        let mut b = ComplexVector::from(&[1.0, 1.0]);
        assert_eq!(
            complex_solve_lin_sys(&mut b, &mut a).err(),
            Some("LAPACK ERROR (dgesv): The factorization has been completed, but the factor U is exactly singular")
        );
    }

    #[test]
    fn complex_solve_lin_sys_challenge() {
        // NOTE: zgesv performs poorly in this problem.
        // The same problem happens in python (likely using lapack too)

        // matrix
        #[rustfmt::skip]
        let mut a = ComplexMatrix::from(&[
            [Complex64::new(19.730,  0.000), Complex64::new(12.110, - 1.000), Complex64::new( 0.000, 5.000), Complex64::new( 0.000,  0.000), Complex64::new( 0.000,  0.000)],
            [Complex64::new( 0.000, -0.510), Complex64::new(32.300,   7.000), Complex64::new(23.070, 0.000), Complex64::new( 0.000,  1.000), Complex64::new( 0.000,  0.000)],
            [Complex64::new( 0.000,  0.000), Complex64::new( 0.000, - 0.510), Complex64::new(70.000, 7.300), Complex64::new( 3.950,  0.000), Complex64::new(19.000, 31.830)],
            [Complex64::new( 0.000,  0.000), Complex64::new( 0.000,   0.000), Complex64::new( 1.000, 1.100), Complex64::new(50.170,  0.000), Complex64::new(45.510,  0.000)],
            [Complex64::new( 0.000,  0.000), Complex64::new( 0.000,   0.000), Complex64::new( 0.000, 0.000), Complex64::new( 0.000, -9.351), Complex64::new(55.000,  0.000)],
        ]);

        // right-hand-side
        let mut b = ComplexVector::from(&[
            Complex64::new(77.38, 8.82),
            Complex64::new(157.48, 19.8),
            Complex64::new(1175.62, 20.69),
            Complex64::new(912.12, -801.75),
            Complex64::new(550.00, -1060.4),
        ]);

        // solution
        let x_correct = &[
            Complex64::new(3.3, -1.00),
            Complex64::new(1.0, 0.17),
            Complex64::new(5.5, 0.00),
            Complex64::new(9.0, 0.00),
            Complex64::new(10.0, -17.75),
        ];

        // run test
        // solve b := x := A⁻¹ b
        complex_solve_lin_sys(&mut b, &mut a).unwrap();
        complex_vec_approx_eq(b.as_data(), x_correct, 0.00049);

        // compare with python results
        let x_python = &[
            Complex64::new(3.299687426933794e+00, -1.000372829305209e+00),
            Complex64::new(9.997606020636992e-01, 1.698383755401385e-01),
            Complex64::new(5.500074759292877e+00, -4.556001293922560e-05),
            Complex64::new(8.999787912842375e+00, -6.662818244209770e-05),
            Complex64::new(1.000001132800243e+01, -1.774987242230929e+01),
        ];
        complex_vec_approx_eq(b.as_data(), x_python, 1e-13);
    }
}
