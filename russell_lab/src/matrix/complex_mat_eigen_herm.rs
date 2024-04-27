use super::ComplexMatrix;
use crate::{cpx, to_i32, CcBool, Complex64, StrError, Vector, C_FALSE, C_TRUE};

extern "C" {
    // Computes the eigenvalues and, optionally, the left and/or right eigenvectors for HE matrices
    // <https://www.netlib.org/lapack/explore-html/d6/dee/zheev_8f.html>
    fn c_zheev(
        calc_v: CcBool,
        upper: CcBool,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        w: *mut f64,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );
}

/// (zheev) Performs the eigen-decomposition of a hermitian matrix
///
/// Computes the eigenvalues `l` and right eigenvectors `v`, such that:
///
/// ```text
/// a ⋅ vj = lj ⋅ vj
///
/// where  a = aᴴ
/// ```
///
/// where `lj` is the component j of `l` and `vj` is the column j of `v`.
///
/// See also <https://www.netlib.org/lapack/explore-html/d6/dee/zheev_8f.html>
///
/// # Output
///
/// * `l` -- (lambda) will hold the eigenvalues (dim = m)
/// * `a` -- will hold the eigenvectors as columns (dim = m * m)
///
/// # Input
///
/// * `a` -- (m,m) general matrix (will be modified) (HERMITIAN and SQUARE)
/// * `upper` -- Whether the upper triangle of `A` must be considered instead
///    of the lower triangle.
///
/// # Notes
///
/// * The matrix `a` will be modified
/// * The computed eigenvectors are normalized to have Euclidean norm equal to 1 and largest component real
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // Hermitian matrix (upper-tridiagonal)
///     let ______________ = cpx!(0.0, 0.0);
///     let mut a_upper = ComplexMatrix::from(&[
///         [cpx!(2.0, 0.0), cpx!(0.0, 1.0), cpx!(0.0, 0.0)],
///         [______________, cpx!(2.0, 0.0), cpx!(0.0, 0.0)],
///         [______________, ______________, cpx!(3.0, 0.0)],
///     ]);
///
///     // allocate the eigenvector array
///     let m = a_upper.nrow();
///     let mut l = Vector::new(m);
///
///     // perform the eigen-decomposition
///     complex_mat_eigen_herm(&mut l, &mut a_upper, true)?;
///
///     // the matrix is modified and will contain the eigenvectors
///     let v = &a_upper;
///
///     // check the eigenvalues
///     assert_eq!(
///         format!("{}", l),
///         "┌   ┐\n\
///          │ 1 │\n\
///          │ 3 │\n\
///          │ 3 │\n\
///          └   ┘"
///     );
///
///     // check the eigen-decomposition
///     let a_herm = ComplexMatrix::from(&[
///         [cpx!(2.0, 0.0), cpx!(0.0, 1.0), cpx!(0.0, 0.0)],
///         [cpx!(0.0, -1.0), cpx!(2.0, 0.0), cpx!(0.0, 0.0)],
///         [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(3.0, 0.0)],
///     ]);
///     let lam = ComplexVector::from(&l);
///     let d = ComplexMatrix::diagonal(lam.as_data());
///     let mut a_v = ComplexMatrix::new(m, m);
///     let mut v_l = ComplexMatrix::new(m, m);
///     let mut err = ComplexMatrix::filled(m, m, cpx!(f64::MAX, 0.0));
///     let zero = cpx!(0.0, 0.0);
///     complex_mat_mat_mul(&mut a_v, cpx!(1.0, 0.0), &a_herm, &v, zero)?;
///     complex_mat_mat_mul(&mut v_l, cpx!(1.0, 0.0), &v, &d, zero)?;
///     complex_mat_add(&mut err, cpx!(1.0, 0.0), &a_v, cpx!(-1.0, 0.0), &v_l)?;
///     approx_eq(complex_mat_norm(&err, Norm::Max), 0.0, 1e-15);
///     Ok(())
/// }
/// ```
pub fn complex_mat_eigen_herm(l: &mut Vector, a: &mut ComplexMatrix, upper: bool) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if m == 0 {
        return Err("matrix dimension must be ≥ 1");
    }
    if l.dim() != m {
        return Err("l vector is incompatible");
    }
    let c_upper = if upper { C_TRUE } else { C_FALSE };
    let m_i32 = to_i32(m);
    let lda = m_i32;
    const EXTRA: i32 = 1;
    let lwork = 2 * m_i32 + EXTRA;
    let mut work = vec![cpx!(0.0, 0.0); lwork as usize];
    let mut rwork = vec![0.0; 3 * m];
    let mut info = 0;
    unsafe {
        c_zheev(
            C_TRUE,
            c_upper,
            &m_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            l.as_mut_data().as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (zheev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (zheev): An argument had an illegal value");
    } else if info > 0 {
        println!("LAPACK ERROR (zheev): {} off-diagonal elements of an intermediate tridiagonal form did not converge to zero.",info-1);
        return Err("LAPACK ERROR (zheev): The algorithm failed to converge");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_mat_eigen_herm;
    use crate::math::SQRT_2;
    use crate::matrix::testing::complex_check_eigen;
    use crate::{complex_mat_approx_eq, cpx, vec_approx_eq};
    use crate::{AsArray2D, Complex64, ComplexMatrix, ComplexVector, Vector};

    fn calc_eigen_lower<'a, T>(data: &'a T) -> (Vector, ComplexMatrix)
    where
        T: AsArray2D<'a, Complex64>,
    {
        let mut a = ComplexMatrix::from_lower(data).unwrap();
        let n = a.ncol();
        let mut l = Vector::new(n);
        complex_mat_eigen_herm(&mut l, &mut a, false).unwrap();
        (l, a)
    }

    fn calc_eigen_upper<'a, T>(data: &'a T) -> (Vector, ComplexMatrix)
    where
        T: AsArray2D<'a, Complex64>,
    {
        let mut a = ComplexMatrix::from_upper(data).unwrap();
        let n = a.ncol();
        let mut l = Vector::new(n);
        complex_mat_eigen_herm(&mut l, &mut a, true).unwrap();
        (l, a)
    }

    #[test]
    fn complex_mat_eigen_herm_handles_errors() {
        let mut a = ComplexMatrix::new(0, 1);
        let mut l = Vector::new(0);
        assert_eq!(
            complex_mat_eigen_herm(&mut l, &mut a, false).err(),
            Some("matrix must be square")
        );
        let mut a = ComplexMatrix::new(0, 0);
        assert_eq!(
            complex_mat_eigen_herm(&mut l, &mut a, false).err(),
            Some("matrix dimension must be ≥ 1")
        );
        let mut a = ComplexMatrix::new(1, 1);
        assert_eq!(
            complex_mat_eigen_herm(&mut l, &mut a, false).err(),
            Some("l vector is incompatible")
        );
    }

    #[test]
    fn complex_mat_eigen_herm_works_0() {
        // 1x1 matrix
        let data = &[[cpx!(2.0, 0.0)]];
        let (l, v) = calc_eigen_lower(data);
        complex_mat_approx_eq(&v, &[[cpx!(1.0, 0.0)]], 1e-15);
        vec_approx_eq(&l, &[2.0], 1e-15);

        // 2x2 matrix
        let data = &[[cpx!(2.0, 0.0), cpx!(1.0, 0.0)], [cpx!(1.0, 0.0), cpx!(2.0, 0.0)]];
        let (l, v) = calc_eigen_lower(data);
        complex_mat_approx_eq(
            &v,
            &[
                [cpx!(-1.0 / SQRT_2, 0.0), cpx!(1.0 / SQRT_2, 0.0)],
                [cpx!(1.0 / SQRT_2, 0.0), cpx!(1.0 / SQRT_2, 0.0)],
            ],
            1e-15,
        );
        vec_approx_eq(&l, &[1.0, 3.0], 1e-15);
    }

    #[test]
    fn complex_mat_eigen_herm_works_1() {
        // all zero
        #[rustfmt::skip]
        let data = &[
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
        ];
        let (l, v) = calc_eigen_lower(data);
        let (ll, vv) = calc_eigen_upper(data);
        #[rustfmt::skip]
        let correct = &[
            [cpx!(1.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(1.0, 0.0)],
        ];
        complex_mat_approx_eq(&v, correct, 1e-15);
        complex_mat_approx_eq(&vv, correct, 1e-15);
        vec_approx_eq(&l, &[0.0, 0.0, 0.0], 1e-15);
        vec_approx_eq(&ll, &[0.0, 0.0, 0.0], 1e-15);

        // 2-repeated, with one zero diagonal entry
        #[rustfmt::skip]
        let data = &[
            [cpx!(2.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(2.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
        ];
        let (l, v) = calc_eigen_lower(data);
        let (ll, vv) = calc_eigen_upper(data);
        #[rustfmt::skip]
        let correct = &[
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(1.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(1.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
        ];
        complex_mat_approx_eq(&v, correct, 1e-15);
        complex_mat_approx_eq(&vv, correct, 1e-15);
        vec_approx_eq(&l, &[0.0, 2.0, 2.0], 1e-15);
        vec_approx_eq(&ll, &[0.0, 2.0, 2.0], 1e-15);
        let l_cpx = ComplexVector::from(&l);
        complex_check_eigen(data, &v, &l_cpx, 1e-15);

        // 3-repeated / diagonal
        #[rustfmt::skip]
        let data = &[
            [cpx!(2.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(2.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(2.0, 0.0)],
        ];
        let (l, v) = calc_eigen_lower(data);
        let (ll, vv) = calc_eigen_upper(data);
        #[rustfmt::skip]
        let correct = ComplexMatrix::from(&[
            [cpx!(1.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(1.0, 0.0)],
        ]);
        complex_mat_approx_eq(&v, &correct, 1e-15);
        complex_mat_approx_eq(&vv, &correct, 1e-15);
        vec_approx_eq(&l, &[2.0, 2.0, 2.0], 1e-15);
        vec_approx_eq(&ll, &[2.0, 2.0, 2.0], 1e-15);
        let l_cpx = ComplexVector::from(&l);
        complex_check_eigen(data, &v, &l_cpx, 1e-15);
    }

    #[test]
    fn complex_mat_eigen_herm_works_2() {
        #[rustfmt::skip]
        let data = &[
		    [cpx!(2.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)],
		    [cpx!(0.0, 0.0), cpx!(3.0, 0.0), cpx!(4.0, 0.0)],
		    [cpx!(0.0, 0.0), cpx!(4.0, 0.0), cpx!(9.0, 0.0)],
        ];
        let (l, v) = calc_eigen_lower(data);
        let d = 1.0 / f64::sqrt(5.0);
        #[rustfmt::skip]
        let correct = ComplexMatrix::from(&[
            [cpx!( 0.0, 0.0),   cpx!(1.0, 0.0), cpx!(0.0, 0.0)  ],
            [cpx!(-2.0*d, 0.0), cpx!(0.0, 0.0), cpx!(1.0*d, 0.0)],
            [cpx!( 1.0*d, 0.0), cpx!(0.0, 0.0), cpx!(2.0*d, 0.0)],
        ]);
        complex_mat_approx_eq(&v, &correct, 1e-15);
        vec_approx_eq(&l, &[1.0, 2.0, 11.0], 1e-15);
        let l_cpx = ComplexVector::from(&l);
        complex_check_eigen(data, &v, &l_cpx, 1e-15);
    }

    #[test]
    fn complex_mat_eigen_herm_works_3() {
        // https://www.ibm.com/docs/en/essl/6.2?topic=eas-ssyev-dsyev-cheev-zheev-sspevx-dspevx-chpevx-zhpevx-ssyevx-dsyevx-cheevx-zheevx-ssyevr-dsyevr-cheevr-zheevr-eigenvalues-optionally-eigenvectors-real-symmetric-complex-hermitian-matrix
        #[rustfmt::skip]
        let a_herm = &[
            [cpx!(2.0,  0.0), cpx!(0.0, 1.0), cpx!(0.0, 0.0)], // <<< note: a(0,1) = 0+1j and not 0-1j (because it's Hermitian)
            [cpx!(0.0, -1.0), cpx!(2.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(0.0,  0.0), cpx!(0.0, 0.0), cpx!(3.0, 0.0)],
        ];
        let ______________ = cpx!(0.0, 0.0);
        #[rustfmt::skip]
        let mut a_lower = ComplexMatrix::from(&[
            [cpx!(2.0,  0.0), ______________, ______________],
            [cpx!(0.0, -1.0), cpx!(2.0, 0.0), ______________],
            [cpx!(0.0,  0.0), cpx!(0.0, 0.0), cpx!(3.0, 0.0)],
        ]);
        #[rustfmt::skip]
        let mut a_upper = ComplexMatrix::from(&[
            [cpx!(2.0, 0.0), cpx!(0.0, 1.0), cpx!(0.0, 0.0)],
            [______________, cpx!(2.0, 0.0), cpx!(0.0, 0.0)],
            [______________, ______________, cpx!(3.0, 0.0)],
        ]);
        let l_correct = &[1.0, 3.0, 3.0];
        // lower
        let mut l = Vector::new(3);
        complex_mat_eigen_herm(&mut l, &mut a_lower, false).unwrap();
        vec_approx_eq(&l, l_correct, 1e-15);
        let v = &a_lower;
        let l_cpx = ComplexVector::from(&l);
        complex_check_eigen(a_herm, v, &l_cpx, 1e-15);
        // upper
        let mut l = Vector::new(3);
        complex_mat_eigen_herm(&mut l, &mut a_upper, true).unwrap();
        vec_approx_eq(&l, l_correct, 1e-15);
        let v = &a_upper;
        let l_cpx = ComplexVector::from(&l);
        complex_check_eigen(a_herm, v, &l_cpx, 1e-15);
        // full
        let mut l = Vector::new(3);
        let mut a = ComplexMatrix::from(a_herm);
        complex_mat_eigen_herm(&mut l, &mut a, false).unwrap();
        let v = &a;
        let l_cpx = ComplexVector::from(&l);
        complex_check_eigen(a_herm, v, &l_cpx, 1e-15);
    }
}
