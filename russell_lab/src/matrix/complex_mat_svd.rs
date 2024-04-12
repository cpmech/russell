use crate::matrix::ComplexMatrix;
use crate::vector::Vector;
use crate::{to_i32, StrError, SVD_CODE_A};
use num_complex::Complex64;
use num_traits::Zero;

extern "C" {
    // Computes the singular value decomposition (SVD)
    // <https://www.netlib.org/lapack/explore-html/d6/d42/zgesvd_8f.html>
    fn c_zgesvd(
        jobu_code: i32,
        jobvt_code: i32,
        m: *const i32,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        s: *mut f64,
        u: *mut Complex64,
        ldu: *const i32,
        vh: *mut Complex64,
        ldvt: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );
}

/// (zgesvd) Computes the singular value decomposition (SVD) of a matrix
///
/// Finds `u`, `s`, and `v`, such that:
///
/// ```text
///   a  :=  u   ⋅   s   ⋅   vᴴ
/// (m,n)  (m,m)   (m,n)   (n,n)
/// ```
///
/// See also <https://www.netlib.org/lapack/explore-html/d6/d42/zgesvd_8f.html>
///
/// # Output
///
/// * `s` -- min(m,n) vector with the diagonal elements
/// * `u` -- (m,m) orthogonal matrix
/// * `vh` -- (n,n) orthogonal matrix with the conjugate-transpose of v
///
/// # Input
///
/// * `a` -- (m,n) matrix, symmetric or not (will be modified)
///
/// # Notes
///
/// * The matrix `a` will be modified
/// * Note that the routine returns `vᴴ`, not v.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // matrix
///     let data = [
///         [cpx!(1.0, 1.0), cpx!(2.0, -1.0), cpx!(3.0, 0.0)],
///         [cpx!(2.0, -1.0), cpx!(4.0, 1.0), cpx!(5.0, -1.0)],
///         [cpx!(3.0, 0.0), cpx!(5.0, -1.0), cpx!(6.0, 1.0)],
///     ];
///
///     let mut a = ComplexMatrix::from(&data);
///     let a_copy = ComplexMatrix::from(&data);
///
///     // allocate output data
///     let (m, n) = a.dims();
///     let min_mn = if m < n { m } else { n };
///     let mut s = Vector::new(min_mn);
///     let mut u = ComplexMatrix::new(m, m);
///     let mut vh = ComplexMatrix::new(n, n);
///
///     // calculate SVD
///     complex_mat_svd(&mut s, &mut u, &mut vh, &mut a).unwrap();
///
///     // check SVD
///     let mut usv = ComplexMatrix::new(m, n);
///     for i in 0..m {
///         for j in 0..n {
///             for k in 0..min_mn {
///                 usv.add(i, j, u.get(i, k) * s[k] * vh.get(k, j));
///             }
///         }
///     }
///     complex_mat_approx_eq(&usv, &a_copy, 1e-14);
///     Ok(())
/// }
/// ```
pub fn complex_mat_svd(
    s: &mut Vector,
    u: &mut ComplexMatrix,
    vh: &mut ComplexMatrix,
    a: &mut ComplexMatrix,
) -> Result<(), StrError> {
    let (m, n) = a.dims();
    let min_mn = usize::min(m, n);
    let max_mn = usize::max(m, n);
    if s.dim() != min_mn {
        return Err("[s] must be a min(m,n) vector");
    }
    if u.nrow() != m || u.ncol() != m {
        return Err("[u] must be an m-by-m square matrix");
    }
    if vh.nrow() != n || vh.ncol() != n {
        return Err("[vh] must be an n-by-n square matrix");
    }
    let m_i32 = to_i32(m);
    let n_i32 = to_i32(n);
    let lda = m_i32;
    let ldu = m_i32;
    let ldvt = n_i32;
    const EXTRA: i32 = 1;
    let lwork = 2 * to_i32(min_mn) + to_i32(max_mn) + EXTRA;
    let mut work = vec![Complex64::zero(); lwork as usize];
    let mut rwork = vec![0.0; 5 * min_mn];
    let mut info = 0;
    unsafe {
        c_zgesvd(
            SVD_CODE_A,
            SVD_CODE_A,
            &m_i32,
            &n_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            s.as_mut_data().as_mut_ptr(),
            u.as_mut_data().as_mut_ptr(),
            &ldu,
            vh.as_mut_data().as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (zgesvd): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (zgesvd): An argument had an illegal value");
    } else if info > 0 {
        println!("LAPACK ERROR (zgesvd): {} is the number of super-diagonals of an intermediate bi-diagonal form B which did not converge to zero",info);
        return Err("LAPACK ERROR (zgesvd): Algorithm did not converge");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_mat_svd;
    use crate::{complex_mat_approx_eq, cpx, vec_approx_eq, ComplexMatrix, Vector};
    use num_complex::Complex64;

    #[test]
    fn complex_mat_svd_fails_on_wrong_dims() {
        let mut a = ComplexMatrix::new(3, 2);
        let mut s = Vector::new(2);
        let mut u = ComplexMatrix::new(3, 3);
        let mut vt = ComplexMatrix::new(2, 2);
        let mut s_3 = Vector::new(3);
        let mut u_2x2 = ComplexMatrix::new(2, 2);
        let mut u_3x2 = ComplexMatrix::new(3, 2);
        let mut vt_3x3 = ComplexMatrix::new(3, 3);
        let mut vt_2x3 = ComplexMatrix::new(2, 3);
        assert_eq!(
            complex_mat_svd(&mut s_3, &mut u, &mut vt, &mut a),
            Err("[s] must be a min(m,n) vector")
        );
        assert_eq!(
            complex_mat_svd(&mut s, &mut u_2x2, &mut vt, &mut a),
            Err("[u] must be an m-by-m square matrix")
        );
        assert_eq!(
            complex_mat_svd(&mut s, &mut u_3x2, &mut vt, &mut a),
            Err("[u] must be an m-by-m square matrix")
        );
        assert_eq!(
            complex_mat_svd(&mut s, &mut u, &mut vt_3x3, &mut a),
            Err("[vh] must be an n-by-n square matrix")
        );
        assert_eq!(
            complex_mat_svd(&mut s, &mut u, &mut vt_2x3, &mut a),
            Err("[vh] must be an n-by-n square matrix")
        );
    }

    #[test]
    fn complex_mat_svd_works() {
        // matrix
        let s33 = f64::sqrt(3.0) / 3.0;
        #[rustfmt::skip]
        let data = [
            [-s33, -s33, 1.0],
            [ s33, -s33, 1.0],
            [-s33,  s33, 1.0],
            [ s33,  s33, 1.0],
        ];
        let mut a = ComplexMatrix::from(&data);
        let a_copy = ComplexMatrix::from(&data);

        // allocate output data
        let (m, n) = a.dims();
        let min_mn = if m < n { m } else { n };
        let mut s = Vector::new(min_mn);
        let mut u = ComplexMatrix::new(m, m);
        let mut vh = ComplexMatrix::new(n, n);

        // calculate SVD
        complex_mat_svd(&mut s, &mut u, &mut vh, &mut a).unwrap();

        // check S
        #[rustfmt::skip]
        let s_correct = &[
            2.0,
            2.0 / f64::sqrt(3.0),
            2.0 / f64::sqrt(3.0),
        ];
        vec_approx_eq(&s, s_correct, 1e-14);

        // check SVD
        let mut usv = ComplexMatrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                for k in 0..min_mn {
                    usv.add(i, j, u.get(i, k) * s[k] * vh.get(k, j));
                }
            }
        }
        complex_mat_approx_eq(&usv, &a_copy, 1e-14);
    }

    #[test]
    fn complex_mat_svd_1_works() {
        // matrix
        #[rustfmt::skip]
        let data = [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ];
        let mut a = ComplexMatrix::from(&data);
        let a_copy = ComplexMatrix::from(&data);

        // allocate output data
        let (m, n) = a.dims();
        let min_mn = if m < n { m } else { n };
        let mut s = Vector::new(min_mn);
        let mut u = ComplexMatrix::new(m, m);
        let mut vh = ComplexMatrix::new(n, n);

        // calculate SVD
        complex_mat_svd(&mut s, &mut u, &mut vh, &mut a).unwrap();

        // check S
        let sqrt2 = std::f64::consts::SQRT_2;
        #[rustfmt::skip]
        let s_correct = &[
            sqrt2,
            sqrt2,
        ];
        vec_approx_eq(&s, s_correct, 1e-14);

        // check SVD
        let mut usv = ComplexMatrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                for k in 0..min_mn {
                    usv.add(i, j, u.get(i, k) * s[k] * vh.get(k, j));
                }
            }
        }
        complex_mat_approx_eq(&usv, &a_copy, 1e-14);
    }

    #[test]
    fn complex_mat_svd_2_works() {
        // https://www.ibm.com/docs/en/essl/6.2?topic=llss-sgesvd-dgesvd-cgesvd-zgesvd-sgesdd-dgesdd-cgesdd-zgesdd-singular-value-decomposition-general-matrix

        // matrix
        #[rustfmt::skip]
        let data = [
            [cpx!(1.0, 1.0), cpx!(2.0,-1.0), cpx!(3.0, 0.0)],
            [cpx!(2.0,-1.0), cpx!(4.0, 1.0), cpx!(5.0,-1.0)],
            [cpx!(3.0, 0.0), cpx!(5.0,-1.0), cpx!(6.0, 1.0)],
        ];

        let mut a = ComplexMatrix::from(&data);
        let a_copy = ComplexMatrix::from(&data);

        // allocate output data
        let (m, n) = a.dims();
        let min_mn = if m < n { m } else { n };
        let mut s = Vector::new(min_mn);
        let mut u = ComplexMatrix::new(m, m);
        let mut vh = ComplexMatrix::new(n, n);

        // calculate SVD
        complex_mat_svd(&mut s, &mut u, &mut vh, &mut a).unwrap();

        // check SVD
        let mut usv = ComplexMatrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                for k in 0..min_mn {
                    usv.add(i, j, u.get(i, k) * s[k] * vh.get(k, j));
                }
            }
        }
        complex_mat_approx_eq(&usv, &a_copy, 1e-14);

        // compare with reference
        #[rustfmt::skip]
        let s_ref = &[
            11.370686,
             2.386257,
             1.006620,
        ];
        #[rustfmt::skip]
        let u_ref = ComplexMatrix::from(&[
            [cpx!(-0.3265, 0.0409), cpx!( 0.0558, 0.4814), cpx!( 0.3504,-0.7308)],
            [cpx!(-0.5822, 0.0725), cpx!(-0.0823,-0.7730), cpx!( 0.1017,-0.2026)],
            [cpx!(-0.7396, 0.0233), cpx!( 0.0036, 0.4009), cpx!(-0.2805, 0.4616)],
        ]);
        #[rustfmt::skip]
        let vh_ref = ComplexMatrix::from(&[
            [cpx!(-0.3290, 0.0000), cpx!(-0.5867,-0.0004), cpx!(-0.7367,-0.0688)],
            [cpx!( 0.4846, 0.0000), cpx!(-0.7774,-0.0071), cpx!( 0.3987, 0.0425)],
            [cpx!(-0.8105, 0.0000), cpx!(-0.2267,-0.0041), cpx!( 0.5375, 0.0533)],
        ]);
        vec_approx_eq(&s, s_ref, 1e-6);
        complex_mat_approx_eq(&u, &u_ref, 1e-4);
        complex_mat_approx_eq(&vh, &vh_ref, 1e-4);
    }
}
