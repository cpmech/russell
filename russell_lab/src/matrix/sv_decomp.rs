use crate::matrix::*;
use crate::vector::*;
use russell_openblas::*;
use std::convert::TryFrom;

/// Computes the singular value decomposition (SVD) of a matrix
///
/// ```text
///   a  :=  u  *   s  * trans(v)
/// (m,n)  (m,m)  (m,n)    (n,n)
/// ```
///
/// # Output
///
/// * `s` -- min(m,n) vector with the diagonal elements
/// * `u` -- (m,m) orthogonal matrix
/// * `vt` -- (n,n) orthogonal matrix with the transpose of v
///
/// # Input
///
/// * `a` -- (m,n) matrix, symmetric or not [will be modified]
///
/// # Note
///
/// 1. The matrix [a] will be modified
///
pub fn sv_decomp(
    s: &mut Vector,
    u: &mut Matrix,
    vt: &mut Matrix,
    a: &mut Matrix,
) -> Result<(), &'static str> {
    let (m, n) = (a.nrow, a.ncol);
    let min_mn = if m < n { m } else { n };
    if s.data.len() != min_mn {
        return Err("[s] must be an min(m,n) vector");
    }
    if u.nrow != m || u.ncol != m {
        return Err("[u] must be an m-by-m square matrix");
    }
    if vt.nrow != n || vt.ncol != n {
        return Err("[vt] must be an n-by-n square matrix");
    }
    let m_i32 = to_i32!(m)?;
    let n_i32 = to_i32!(n)?;
    let mut superb = vec![0.0; min_mn];
    dgesvd(
        b'A',
        b'A',
        m_i32,
        n_i32,
        &mut a.data,
        m_i32,
        &mut s.data,
        &mut u.data,
        m_i32,
        &mut vt.data,
        n_i32,
        &mut superb,
    )
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use crate::matrix::*;
    use crate::vector::*;
    use russell_chk::*;

    #[test]
    fn sv_decomp_works() -> Result<(), &'static str> {
        // matrix
        let s33 = f64::sqrt(3.0) / 3.0;
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[-s33, -s33, 1.0],
            &[ s33, -s33, 1.0],
            &[-s33,  s33, 1.0],
            &[ s33,  s33, 1.0],
        ]);

        // allocate output data
        let (m, n) = a.dims();
        let min_mn = if m < n { m } else { n };
        let mut s = Vector::new(min_mn);
        let mut u = Matrix::new(m, m);
        let mut vt = Matrix::new(n, n);

        // calculate SVD
        sv_decomp(&mut s, &mut u, &mut vt, &mut a)?;

        // check
        #[rustfmt::skip]
        let s_correct = Vector::from(&[
            2.0,
            2.0 / f64::sqrt(3.0),
            2.0 / f64::sqrt(3.0),
        ]);
        #[rustfmt::skip]
        let u_correct = Matrix::from(&[
            &[-0.5, -0.5, -0.5,  0.5],
            &[-0.5, -0.5,  0.5, -0.5],
            &[-0.5,  0.5, -0.5, -0.5],
            &[-0.5,  0.5,  0.5,  0.5],
        ]);
        #[rustfmt::skip]
        let vt_correct =Matrix::from(&[
            &[0.0,  0.0, -1.0],
            &[0.0,  1.0,  0.0],
            &[1.0,  0.0,  0.0],
        ]);
        assert_vec_approx_eq!(u.data, u_correct.data, 1e-15);
        assert_vec_approx_eq!(s.data, s_correct.data, 1e-15);
        assert_vec_approx_eq!(vt.data, vt_correct.data, 1e-15);

        // done
        Ok(())
    }
}
