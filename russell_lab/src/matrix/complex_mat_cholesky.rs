use super::ComplexMatrix;
use crate::{to_i32, CcBool, Complex64, StrError, C_FALSE, C_TRUE};

extern "C" {
    // Computes the Cholesky factorization of a complex Hermitian positive definite matrix A.
    // <https://www.netlib.org/lapack/explore-html/d1/db9/zpotrf_8f.html>
    fn c_zpotrf(upper: CcBool, n: *const i32, a: *mut Complex64, lda: *const i32, info: *mut i32);
}

/// (zpotrf) Performs the Cholesky factorization of a Hermitian positive-definite matrix
///
/// Finds `L` or `U` such that:
///
/// ```text
/// A = L ⋅ Lᴴ
///
/// or
///
/// A = Uᴴ ⋅ U
/// ```
///
/// where `L` is lower triangular and `U` is upper triangular.
///
/// See also: <https://www.netlib.org/lapack/explore-html/d1/db9/zpotrf_8f.html>
///
/// # Input/Output
///
/// * `A → L` or `A → U` -- On input, `A` is a **Hermitian/positive-definite** matrix
///   with either the lower or upper triangular part given, according to the `upper` flag.
///   On output, `A = L` or `A = U` with the other side of the triangle unmodified.
/// * `upper` -- Whether the upper triangle of `A` must be considered instead
///    of the lower triangle. This will cause the computation of either `L` or `U`.
///
/// # Notes
///
/// * Either the lower triangle or the upper triangle is considered according to the `upper` flag.
/// * All elements of `A` may be provided and the unnecessary "triangle" will be ignored.
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // set matrix
///     let _______________ = cpx!(0.0, 0.0);
///     #[rustfmt::skip]
///     let mut a = ComplexMatrix::from(&[
///         [cpx!(  4.0, 0.0), _______________,  _______________],
///         [cpx!( 12.0, 0.0), cpx!( 37.0, 0.0), _______________],
///         [cpx!(-16.0, 0.0), cpx!(-43.0, 0.0), cpx!(98.0, 0.0)],
///     ]);
///
///     // perform factorization
///     complex_mat_cholesky(&mut a, false)?;
///
///     // define alias (for convenience)
///     let l = &a;
///
///     // compare with solution
///     let l_correct = "┌                   ┐\n\
///                      │  2+0i  0+0i  0+0i │\n\
///                      │  6+0i  1+0i  0+0i │\n\
///                      │ -8+0i  5+0i  3+0i │\n\
///                      └                   ┘";
///     assert_eq!(format!("{}", l), l_correct);
///
///     // check:  l ⋅ lᵀ = a
///     let m = a.nrow();
///     let mut l_lt = ComplexMatrix::new(m, m);
///     for i in 0..m {
///         for j in 0..m {
///             for k in 0..m {
///                 l_lt.add(i, j, l.get(i, k) * l.get(j, k));
///             }
///         }
///     }
///     let l_lt_correct = "┌                      ┐\n\
///                         │   4+0i  12+0i -16+0i │\n\
///                         │  12+0i  37+0i -43+0i │\n\
///                         │ -16+0i -43+0i  98+0i │\n\
///                         └                      ┘";
///     assert_eq!(format!("{}", l_lt), l_lt_correct);
///     Ok(())
/// }
/// ```
pub fn complex_mat_cholesky(a: &mut ComplexMatrix, upper: bool) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("the matrix must be square");
    }
    let c_upper = if upper { C_TRUE } else { C_FALSE };
    let m_i32 = to_i32(m);
    let lda = m_i32;
    let mut info = 0;
    unsafe { c_zpotrf(c_upper, &m_i32, a.as_mut_data().as_mut_ptr(), &lda, &mut info) }
    if info < 0 {
        println!("LAPACK ERROR (zpotrf): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (zpotrf): An argument had an illegal value");
    } else if info > 0 {
        println!(
            "LAPACK ERROR (zpotrf): The leading minor of order {} is not positive definite",
            info
        );
        return Err("LAPACK ERROR (zpotrf): Positive definite check failed");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_mat_cholesky;
    use crate::math::SQRT_2;
    use crate::matrix::testing::check_hermitian_uplo;
    use crate::{complex_mat_approx_eq, cpx, Complex64, ComplexMatrix};

    fn calc_l_times_lt(l_and_a: &ComplexMatrix) -> ComplexMatrix {
        let m = l_and_a.nrow();
        let mut l_lt = ComplexMatrix::new(m, m);
        let zero = cpx!(0.0, 0.0);
        for i in 0..m {
            for j in 0..m {
                for k in 0..m {
                    let l_ik = if i >= k { l_and_a.get(i, k) } else { zero };
                    let l_jk = if j >= k { l_and_a.get(j, k) } else { zero };
                    l_lt.add(i, j, l_ik * l_jk);
                }
            }
        }
        l_lt
    }

    fn calc_ut_times_u(u_and_a: &ComplexMatrix) -> ComplexMatrix {
        let m = u_and_a.nrow();
        let mut ut_u = ComplexMatrix::new(m, m);
        let zero = cpx!(0.0, 0.0);
        for i in 0..m {
            for j in 0..m {
                for k in 0..m {
                    let u_ki = if i >= k { u_and_a.get(k, i) } else { zero };
                    let u_kj = if j >= k { u_and_a.get(k, j) } else { zero };
                    ut_u.add(i, j, u_ki * u_kj);
                }
            }
        }
        ut_u
    }

    #[test]
    fn complex_mat_cholesky_fails_on_wrong_dims() {
        let mut a_wrong = ComplexMatrix::new(2, 3);
        assert_eq!(
            complex_mat_cholesky(&mut a_wrong, false),
            Err("the matrix must be square")
        );
    }

    #[test]
    fn complex_mat_cholesky_3x3_lower_works() {
        // define matrix
        let (a01, a02) = (15.0, -5.0);
        let a12 = 0.0;
        #[rustfmt::skip]
        let a_full = ComplexMatrix::from(&[
            [25.0,  a01,  a02],
            [ a01, 18.0,  a12],
            [ a02,  a12, 11.0],
        ]);
        #[rustfmt::skip]
        let a_lower = ComplexMatrix::from(&[
            [25.0,  0.0,  0.0],
            [ a01, 18.0,  0.0],
            [ a02,  a12, 11.0],
        ]);

        // Cholesky factorization with full matrix => lower
        let mut l_and_a = a_full.clone();
        complex_mat_cholesky(&mut l_and_a, false).unwrap(); // l := lower(l_and_a), a := upper(l_and_a)
        #[rustfmt::skip]
        let l_and_a_correct = ComplexMatrix::from(&[
            [ 5.0, a01, a02],
            [ 3.0, 3.0, a12],
            [-1.0, 1.0, 3.0],
        ]);
        complex_mat_approx_eq(&l_and_a, &l_and_a_correct, 1e-15);
        let l_lt = calc_l_times_lt(&l_and_a);
        complex_mat_approx_eq(&l_lt, &a_full, 1e-15);

        // Cholesky factorization with lower matrix
        let mut l = a_lower.clone();
        complex_mat_cholesky(&mut l, false).unwrap();
        let nil = 0.0;
        #[rustfmt::skip]
        let l_correct = ComplexMatrix::from(&[
            [ 5.0, nil, nil],
            [ 3.0, 3.0, nil],
            [-1.0, 1.0, 3.0],
        ]);
        complex_mat_approx_eq(&l, &l_correct, 1e-15);
        let l_lt = calc_l_times_lt(&l);
        complex_mat_approx_eq(&l_lt, &a_full, 1e-15);
    }

    #[test]
    fn complex_mat_cholesky_3x3_upper_works() {
        // define matrix
        let (a01, a02) = (15.0, -5.0);
        let a12 = 0.0;
        #[rustfmt::skip]
        let a_full = ComplexMatrix::from(&[
            [25.0,  a01,  a02],
            [ a01, 18.0,  a12],
            [ a02,  a12, 11.0],
        ]);
        #[rustfmt::skip]
        let a_upper = ComplexMatrix::from(&[
            [25.0,  a01,  a02],
            [ 0.0, 18.0,  a12],
            [ 0.0,  0.0, 11.0],
        ]);

        // Cholesky factorization with full matrix => upper
        let mut u_and_a = a_full.clone();
        complex_mat_cholesky(&mut u_and_a, true).unwrap(); // u := upper(u_and_a), a := lower(l_and_a)
        #[rustfmt::skip]
        let u_and_a_correct = ComplexMatrix::from(&[
            [5.0, 3.0,-1.0],
            [a01, 3.0, 1.0],
            [a02, a12, 3.0],
        ]);
        complex_mat_approx_eq(&u_and_a, &u_and_a_correct, 1e-15);
        let ut_u = calc_ut_times_u(&u_and_a);
        complex_mat_approx_eq(&ut_u, &a_full, 1e-15);

        // Cholesky factorization with upper matrix
        let mut u = a_upper.clone();
        complex_mat_cholesky(&mut u, true).unwrap();
        let nil = 0.0;
        #[rustfmt::skip]
        let u_and_a_correct = ComplexMatrix::from(&[
            [5.0, 3.0,-1.0],
            [nil, 3.0, 1.0],
            [nil, nil, 3.0],
        ]);
        complex_mat_approx_eq(&u, &u_and_a_correct, 1e-15);
        let ut_u = calc_ut_times_u(&u);
        complex_mat_approx_eq(&ut_u, &a_full, 1e-15);
    }

    #[test]
    fn complex_mat_cholesky_5x5_lower_works() {
        // define matrix
        let nil = 0.0;
        let (a01, a02, a03, a04) = (1.0, 1.0, 3.0, 2.0);
        let (___, a12, a13, a14) = (nil, 2.0, 1.0, 1.0);
        let (___, __p, a23, a24) = (nil, nil, 1.0, 5.0);
        let (___, __p, __q, a34) = (nil, nil, nil, 1.0);
        #[rustfmt::skip]
        let a_full = ComplexMatrix::from(&[
            [2.0, a01, a02, a03, a04],
            [a01, 2.0, a12, a13, a14],
            [a02, a12, 9.0, a23, a24],
            [a03, a13, a23, 7.0, a34],
            [a04, a14, a24, a34, 8.0],
        ]);
        #[rustfmt::skip]
        let a_lower = ComplexMatrix::from(&[
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [a01, 2.0, 0.0, 0.0, 0.0],
            [a02, a12, 9.0, 0.0, 0.0],
            [a03, a13, a23, 7.0, 0.0],
            [a04, a14, a24, a34, 8.0],
        ]);

        // Cholesky factorization with full matrix => lower
        let mut l_and_a = a_full.clone();
        complex_mat_cholesky(&mut l_and_a, false).unwrap(); // l := lower(l_and_a), a := upper(l_and_a)
        #[rustfmt::skip]
        let l_and_a_correct = ComplexMatrix::from(&[
            [    SQRT_2,                 a01,                a02,                     a03,    a04],
            [1.0/SQRT_2,  f64::sqrt(3.0/2.0),                a12,                     a13,    a14],
            [1.0/SQRT_2,  f64::sqrt(3.0/2.0),     f64::sqrt(7.0),                     a23,    a24],
            [3.0/SQRT_2, -1.0/f64::sqrt(6.0),                0.0,      f64::sqrt(7.0/3.0),    a34],
            [    SQRT_2,                 0.0, 4.0/f64::sqrt(7.0), -2.0*f64::sqrt(3.0/7.0), SQRT_2],
        ]);
        complex_mat_approx_eq(&l_and_a, &l_and_a_correct, 1e-14);
        let l_lt = calc_l_times_lt(&l_and_a);
        complex_mat_approx_eq(&l_lt, &a_full, 1e-14);

        // Cholesky factorization with lower matrix
        let mut l = a_lower.clone();
        complex_mat_cholesky(&mut l, false).unwrap();
        let l_lt = calc_l_times_lt(&l);
        complex_mat_approx_eq(&l_lt, &a_full, 1e-14);
    }

    #[test]
    fn complex_mat_cholesky_5x5_upper_works() {
        // define matrix
        let nil = 0.0;
        let (a01, a02, a03, a04) = (1.0, 1.0, 3.0, 2.0);
        let (___, a12, a13, a14) = (nil, 2.0, 1.0, 1.0);
        let (___, __p, a23, a24) = (nil, nil, 1.0, 5.0);
        let (___, __p, __q, a34) = (nil, nil, nil, 1.0);
        #[rustfmt::skip]
        let a_full = ComplexMatrix::from(&[
            [2.0, a01, a02, a03, a04],
            [a01, 2.0, a12, a13, a14],
            [a02, a12, 9.0, a23, a24],
            [a03, a13, a23, 7.0, a34],
            [a04, a14, a24, a34, 8.0],
        ]);
        #[rustfmt::skip]
        let a_upper = ComplexMatrix::from(&[
            [2.0, a01, a02, a03, a04],
            [0.0, 2.0, a12, a13, a14],
            [0.0, 0.0, 9.0, a23, a24],
            [0.0, 0.0, 0.0, 7.0, a34],
            [0.0, 0.0, 0.0, 0.0, 8.0],
        ]);

        // Cholesky factorization with full matrix => upper
        let mut u_and_a = a_full.clone();
        complex_mat_cholesky(&mut u_and_a, true).unwrap(); // u := upper(u_and_a), a := lower(l_and_a)
        let ut_u = calc_ut_times_u(&u_and_a);
        complex_mat_approx_eq(&ut_u, &a_full, 1e-14);

        // Cholesky factorization with upper matrix
        let mut u = a_upper.clone();
        complex_mat_cholesky(&mut u, true).unwrap();
        let ut_u = calc_ut_times_u(&u);
        complex_mat_approx_eq(&ut_u, &a_full, 1e-14);
    }

    #[test]
    fn complex_mat_cholesky_captures_non_positive_definite() {
        // define matrix
        let (a01, a02) = (15.0, -5.0);
        let a12 = 0.0;
        #[rustfmt::skip]
        let a_full = ComplexMatrix::from(&[
            [25.0,   a01,  a02],
            [ a01, -18.0,  a12],
            [ a02,   a12, 11.0],
        ]);
        let mut res = a_full.clone();
        assert_eq!(
            complex_mat_cholesky(&mut res, true).err(),
            Some("LAPACK ERROR (zpotrf): Positive definite check failed")
        );
    }

    #[test]
    fn complex_mat_cholesky_hermitian() {
        #[rustfmt::skip]
        let a= ComplexMatrix::from(&[
            [cpx!( 4.0,  0.0), cpx!(0.0, 1.0), cpx!(-3.0, 1.0), cpx!(0.0,  2.0)],
            [cpx!( 0.0, -1.0), cpx!(3.0, 0.0), cpx!( 1.0, 0.0), cpx!(2.0,  0.0)],
            [cpx!(-3.0, -1.0), cpx!(1.0, 0.0), cpx!( 4.0, 0.0), cpx!(1.0, -1.0)],
            [cpx!( 0.0, -2.0), cpx!(2.0, 0.0), cpx!( 1.0, 1.0), cpx!(4.0,  0.0)],
        ]);
        #[rustfmt::skip]
        let mut a_lower = ComplexMatrix::from(&[
            [cpx!( 4.0,  0.0), cpx!(0.0, 0.0),  cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!( 0.0, -1.0), cpx!(3.0, 0.0),  cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(-3.0, -1.0), cpx!(1.0, 0.0),  cpx!(4.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!( 0.0, -2.0), cpx!(2.0, 0.0),  cpx!(1.0, 1.0), cpx!(4.0, 0.0)],
        ]);
        #[rustfmt::skip]
        let mut a_upper = ComplexMatrix::from(&[
            [cpx!(4.0, 0.0), cpx!(0.0, 1.0), cpx!(-3.0, 1.0), cpx!(0.0,  2.0)],
            [cpx!(0.0, 0.0), cpx!(3.0, 0.0), cpx!( 1.0, 0.0), cpx!(2.0,  0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!( 4.0, 0.0), cpx!(1.0, -1.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!( 0.0, 0.0), cpx!(4.0,  0.0)],
        ]);
        check_hermitian_uplo(&a, &a_lower, &a_upper);

        // reference results
        #[rustfmt::skip]
        let a_ref_lower = ComplexMatrix::from(&[
            [cpx!( 2.0,  0.0e+00), cpx!(0.000000000000000e+00, 0.000000000000000e+00), cpx!(0.000000000000000e+00,  0.000000000000000e+00), cpx!(0.000000000000000e+00, 0.0)],
            [cpx!( 0.0, -5.0e-01), cpx!(1.658312395177700e+00, 0.000000000000000e+00), cpx!(0.000000000000000e+00,  0.000000000000000e+00), cpx!(0.000000000000000e+00, 0.0)],
            [cpx!(-1.5, -5.0e-01), cpx!(4.522670168666454e-01, 4.522670168666454e-01), cpx!(1.044465935734187e+00,  0.000000000000000e+00), cpx!(0.000000000000000e+00, 0.0)],
            [cpx!( 0.0, -1.0e+00), cpx!(9.045340337332909e-01, 0.000000000000000e+00), cpx!(8.703882797784884e-02, -8.703882797784884e-02), cpx!(1.471960144387974e+00, 0.0)],
        ]);
        #[rustfmt::skip]
        let a_ref_upper = ComplexMatrix::from(&[
            [cpx!(2.0, 0.0), cpx!(0.000000000000000e+00, 5.0e-01), cpx!(-1.500000000000000e+00,  5.000000000000000e-01), cpx!(0.000000000000000e+00, 1.000000000000000e+00)],
            [cpx!(0.0, 0.0), cpx!(1.658312395177700e+00, 0.0e+00), cpx!( 4.522670168666454e-01, -4.522670168666454e-01), cpx!(9.045340337332909e-01, 0.000000000000000e+00)],
            [cpx!(0.0, 0.0), cpx!(0.000000000000000e+00, 0.0e+00), cpx!( 1.044465935734187e+00,  0.000000000000000e+00), cpx!(8.703882797784884e-02, 8.703882797784884e-02)],
            [cpx!(0.0, 0.0), cpx!(0.000000000000000e+00, 0.0e+00), cpx!( 0.000000000000000e+00,  0.000000000000000e+00), cpx!(1.471960144387974e+00, 0.000000000000000e+00)],
        ]);

        // lower
        complex_mat_cholesky(&mut a_lower, false).unwrap();
        complex_mat_approx_eq(&a_lower, &a_ref_lower, 1e-15);

        // upper
        complex_mat_cholesky(&mut a_upper, true).unwrap();
        complex_mat_approx_eq(&a_upper, &a_ref_upper, 1e-15);
    }
}
