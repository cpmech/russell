use crate::{SVD_CODE_A, StrError, to_i32};

unsafe extern "C" {
    // Computes the singular value decomposition (SVD)
    // <https://www.netlib.org/lapack/explore-html/d8/d2d/dgesvd_8f.html>
    fn c_dgesvd(
        jobu_code: i32,
        jobvt_code: i32,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        s: *mut f64,
        u: *mut f64,
        ldu: *const i32,
        vt: *mut f64,
        ldvt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

/// (dgesvd) Computes the singular value decomposition (SVD) of a small matrix
///
/// Finds `u`, `s`, and `vt`, such that:
///
/// ```text
///   a  :=  u   ⋅   s   ⋅   vt
/// (m,n)  (m,m)   (m,n)   (n,n)
/// ```
///
/// The input matrix `a` is stored row-major (as a Rust 2D array) and is copied
/// into a column-major buffer before being passed to LAPACK (`dgesvd`).
///
/// See also: [`crate::mat_svd`] (the heap-allocated counterpart), the LAPACK
/// reference <https://www.netlib.org/lapack/explore-html/d8/d2d/dgesvd_8f.html>,
/// and the example below.
///
/// # Output
///
/// * `s` -- the `min(m,n)` singular values in descending order; its length must
///   be exactly `min(m,n)`.
/// * `u` -- (M,M) orthogonal matrix holding the left singular vectors as columns.
/// * `vt` -- (N,N) orthogonal matrix with the transpose of the right singular
///   vectors.
///
/// # Input
///
/// * `a` -- (M,N) matrix (row-major)
///
/// # Errors
///
/// Returns an error if `s.len() != min(M,N)` or if LAPACK fails.
///
/// # Examples
///
/// ```
/// use russell_lab::small_mat_svd;
///
/// // 2 x 3 rectangular matrix
/// let a = [
///     [3.0, 2.0,  2.0],
///     [2.0, 3.0, -2.0],
/// ];
/// let mut s = [0.0; 2];
/// let mut u = [[0.0; 2]; 2];
/// let mut vt = [[0.0; 3]; 3];
/// small_mat_svd(&mut s, &mut u, &mut vt, &a).unwrap();
///
/// // singular values
/// assert!((s[0] - 5.0).abs() < 1e-15);
/// assert!((s[1] - 3.0).abs() < 1e-15);
///
/// // check the decomposition: a == u * diag(s) * vt
/// for i in 0..2 {
///     for j in 0..3 {
///         let mut sum = 0.0;
///         for k in 0..2 {
///             sum += u[i][k] * s[k] * vt[k][j];
///         }
///         assert!((sum - a[i][j]).abs() < 1e-14);
///     }
/// }
/// ```
pub fn small_mat_svd<const M: usize, const N: usize>(
    s: &mut [f64],
    u: &mut [[f64; M]; M],
    vt: &mut [[f64; N]; N],
    a: &[[f64; N]; M],
) -> Result<(), StrError> {
    let min_mn = if M < N { M } else { N };
    if s.len() != min_mn {
        return Err("[s] must be a min(m,n) vector");
    }
    if min_mn == 0 {
        return Ok(());
    }
    let m_i32: i32 = to_i32(M);
    let n_i32: i32 = to_i32(N);
    let lda = m_i32;
    let ldu = m_i32;
    let ldvt = n_i32;

    // copy the row-major input into a column-major buffer (i.e. store the transpose)
    let mut acm = [[0.0; M]; N];
    for i in 0..M {
        for j in 0..N {
            acm[j][i] = a[i][j];
        }
    }

    // column-major output buffers
    let mut ucm = [[0.0; M]; M];
    let mut vtcm = [[0.0; N]; N];

    let mut info: i32 = 0;
    unsafe {
        // first: perform the workspace query
        let lwork_query: i32 = -1;
        let mut work = [0.0f64];
        c_dgesvd(
            SVD_CODE_A,
            SVD_CODE_A,
            &m_i32,
            &n_i32,
            acm[0].as_mut_ptr(),
            &lda,
            s.as_mut_ptr(),
            ucm[0].as_mut_ptr(),
            &ldu,
            vtcm[0].as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork_query,
            &mut info,
        );
        if info != 0 {
            return Err("dgesvd workspace query failed");
        }

        // second: perform the SVD
        let lwork = work[0] as i32;
        let mut work = vec![0.0; lwork as usize];
        c_dgesvd(
            SVD_CODE_A,
            SVD_CODE_A,
            &m_i32,
            &n_i32,
            acm[0].as_mut_ptr(),
            &lda,
            s.as_mut_ptr(),
            ucm[0].as_mut_ptr(),
            &ldu,
            vtcm[0].as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
        if info != 0 {
            return Err("dgesvd failed");
        }
    }

    // copy the column-major outputs back into row-major arrays
    for i in 0..M {
        for j in 0..M {
            u[i][j] = ucm[j][i];
        }
    }
    for i in 0..N {
        for j in 0..N {
            vt[i][j] = vtcm[j][i];
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::small_mat_svd;
    use crate::approx_eq;

    /// Checks that `a == u ⋅ diag(s) ⋅ vt` and that `u`/`vt` are orthogonal
    fn check_svd<const M: usize, const N: usize>(
        a: &[[f64; N]; M],
        s: &[f64],
        u: &[[f64; M]; M],
        vt: &[[f64; N]; N],
        tol: f64,
    ) {
        let min_mn = if M < N { M } else { N };

        // reconstruction: a == u ⋅ diag(s) ⋅ vt
        for i in 0..M {
            for j in 0..N {
                let mut sum = 0.0;
                for k in 0..min_mn {
                    sum += u[i][k] * s[k] * vt[k][j];
                }
                approx_eq(sum, a[i][j], tol);
            }
        }

        // orthogonality of u: uᵀ ⋅ u = I
        for i in 0..M {
            for j in 0..M {
                let mut sum = 0.0;
                for k in 0..M {
                    sum += u[k][i] * u[k][j];
                }
                approx_eq(sum, if i == j { 1.0 } else { 0.0 }, tol);
            }
        }

        // orthogonality of vt: vt ⋅ vtᵀ = I
        for i in 0..N {
            for j in 0..N {
                let mut sum = 0.0;
                for k in 0..N {
                    sum += vt[i][k] * vt[j][k];
                }
                approx_eq(sum, if i == j { 1.0 } else { 0.0 }, tol);
            }
        }

        // singular values are non-negative and sorted in descending order
        for k in 0..min_mn {
            assert!(s[k] >= 0.0);
            if k > 0 {
                assert!(s[k - 1] >= s[k]);
            }
        }
    }

    #[test]
    fn small_mat_svd_fails_on_wrong_s() {
        let a = [[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]];
        let mut s = [0.0; 3]; // should be 2
        let mut u = [[0.0; 2]; 2];
        let mut vt = [[0.0; 3]; 3];
        assert_eq!(
            small_mat_svd(&mut s, &mut u, &mut vt, &a),
            Err("[s] must be a min(m,n) vector")
        );
    }

    #[test]
    fn small_mat_svd_works_2x3() {
        let a = [[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]];
        let mut s = [0.0; 2];
        let mut u = [[0.0; 2]; 2];
        let mut vt = [[0.0; 3]; 3];
        small_mat_svd(&mut s, &mut u, &mut vt, &a).unwrap();
        approx_eq(s[0], 5.0, 1e-15);
        approx_eq(s[1], 3.0, 1e-15);
        check_svd(&a, &s, &u, &vt, 1e-14);
    }

    #[test]
    fn small_mat_svd_works_4x2() {
        #[rustfmt::skip]
        let a = [
            [2.0, 4.0],
            [1.0, 3.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ];
        let mut s = [0.0; 2];
        let mut u = [[0.0; 4]; 4];
        let mut vt = [[0.0; 2]; 2];
        small_mat_svd(&mut s, &mut u, &mut vt, &a).unwrap();
        approx_eq(s[0], 5.464985704219043, 1e-14);
        approx_eq(s[1], 0.365966190626257, 1e-14);
        check_svd(&a, &s, &u, &vt, 1e-14);
    }

    #[test]
    fn small_mat_svd_works_3x3() {
        #[rustfmt::skip]
        let a = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
        ];
        let mut s = [0.0; 3];
        let mut u = [[0.0; 3]; 3];
        let mut vt = [[0.0; 3]; 3];
        small_mat_svd(&mut s, &mut u, &mut vt, &a).unwrap();
        check_svd(&a, &s, &u, &vt, 1e-13);
    }

    #[test]
    fn small_mat_svd_works_0() {
        let a: [[f64; 0]; 0] = [];
        let mut s: [f64; 0] = [];
        let mut u: [[f64; 0]; 0] = [];
        let mut vt: [[f64; 0]; 0] = [];
        small_mat_svd(&mut s, &mut u, &mut vt, &a).unwrap();
    }
}
