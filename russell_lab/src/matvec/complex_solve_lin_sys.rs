use crate::matrix::ComplexMatrix;
use crate::vector::ComplexVector;
use crate::{to_i32, StrError};
use num_complex::Complex64;

extern "C" {
    // Computes the solution to a system of linear equations
    // <https://www.netlib.org/lapack/explore-html/d1/ddc/zgesv_8f.html>
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

/// (zgesv) Solves a general linear system
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
/// See also: <https://www.netlib.org/lapack/explore-html/d1/ddc/zgesv_8f.html>
///
/// # Note
///
/// 1. The matrix `a` will be modified
/// 2. The right-hand-side `b` will contain the solution `x`
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // Example from:
///     // https://numericalalgorithmsgroup.github.io/LAPACK_Examples/examples/doc/zgesv_example.html
///
///     #[rustfmt::skip]
///     let mut a = ComplexMatrix::from(&[
///         [cpx!(-1.34, 2.55), cpx!( 0.28, 3.17), cpx!(-6.39,-2.20), cpx!( 0.72,-0.92)],
///         [cpx!(-0.17,-1.41), cpx!( 3.31,-0.15), cpx!(-0.15, 1.34), cpx!( 1.29, 1.38)],
///         [cpx!(-3.29,-2.39), cpx!(-1.91, 4.42), cpx!(-0.14,-1.35), cpx!( 1.72, 1.35)],
///         [cpx!( 2.41, 0.39), cpx!(-0.56, 1.47), cpx!(-0.83,-0.69), cpx!(-1.96, 0.67)],
///     ]);
///
///     let mut b = ComplexVector::from(&[
///         cpx!(26.26, 51.78),
///         cpx!(6.43, -8.68),
///         cpx!(-5.75, 25.31),
///         cpx!(1.16, 2.57),
///     ]);
///
///     // solve b := x := A⁻¹ b
///     complex_solve_lin_sys(&mut b, &mut a).unwrap();
///
///     // print results
///     println!("a (after) =\n{:.3}", a);
///     println!("b (after) =\n{:.3}", b);
///
///     // expected results
///     let correct = ComplexVector::from(&[
///         cpx!(1.0, 1.0),
///         cpx!(2.0, -3.0),
///         cpx!(-4.0, -5.0),
///         cpx!(0.0, 6.0),
///     ]);
///     println!("expected =\n{:.3}", correct);
///     complex_vec_approx_eq(b.as_data(), correct.as_data(), 1e-13);
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
    use crate::{complex_vec_approx_eq, cpx, ComplexMatrix, ComplexVector};
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
            cpx!(-629.0 / 98.0, 0.0),
            cpx!( 237.0 / 49.0, 0.0),
            cpx!( -53.0 / 49.0, 0.0),
            cpx!(  62.0 / 49.0, 0.0),
            cpx!(  23.0 / 14.0, 0.0),
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
            cpx!( 1.0, 0.0),
            cpx!(-1.0, 0.0),
            cpx!( 3.0, 0.0),
            cpx!(-5.0, 0.0),
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
            [cpx!(19.730,  0.000), cpx!(12.110, - 1.000), cpx!( 0.000, 5.000), cpx!( 0.000,  0.000), cpx!( 0.000,  0.000)],
            [cpx!( 0.000, -0.510), cpx!(32.300,   7.000), cpx!(23.070, 0.000), cpx!( 0.000,  1.000), cpx!( 0.000,  0.000)],
            [cpx!( 0.000,  0.000), cpx!( 0.000, - 0.510), cpx!(70.000, 7.300), cpx!( 3.950,  0.000), cpx!(19.000, 31.830)],
            [cpx!( 0.000,  0.000), cpx!( 0.000,   0.000), cpx!( 1.000, 1.100), cpx!(50.170,  0.000), cpx!(45.510,  0.000)],
            [cpx!( 0.000,  0.000), cpx!( 0.000,   0.000), cpx!( 0.000, 0.000), cpx!( 0.000, -9.351), cpx!(55.000,  0.000)],
        ]);

        // right-hand-side
        let mut b = ComplexVector::from(&[
            cpx!(77.38, 8.82),
            cpx!(157.48, 19.8),
            cpx!(1175.62, 20.69),
            cpx!(912.12, -801.75),
            cpx!(550.00, -1060.4),
        ]);

        // solution
        let x_correct = &[
            cpx!(3.3, -1.00),
            cpx!(1.0, 0.17),
            cpx!(5.5, 0.00),
            cpx!(9.0, 0.00),
            cpx!(10.0, -17.75),
        ];

        // run test
        // solve b := x := A⁻¹ b
        complex_solve_lin_sys(&mut b, &mut a).unwrap();
        complex_vec_approx_eq(b.as_data(), x_correct, 0.00049);

        // compare with python results
        let x_python = &[
            cpx!(3.299687426933794e+00, -1.000372829305209e+00),
            cpx!(9.997606020636992e-01, 1.698383755401385e-01),
            cpx!(5.500074759292877e+00, -4.556001293922560e-05),
            cpx!(8.999787912842375e+00, -6.662818244209770e-05),
            cpx!(1.000001132800243e+01, -1.774987242230929e+01),
        ];
        complex_vec_approx_eq(b.as_data(), x_python, 1e-13);
    }
}
