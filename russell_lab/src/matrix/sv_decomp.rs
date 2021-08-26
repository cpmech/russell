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
/// # Examples
///
/// ```
/// # fn main() -> Result<(), &'static str> {
///
/// // import
/// use russell_lab::*;
///
/// // set matrix
/// let mut a = Matrix::from(&[
///     &[3.0, 2.0,  2.0],
///     &[2.0, 3.0, -2.0],
/// ]);
///
/// // allocate output structures
/// let (m, n) = a.dims();
/// let min_mn = if m < n { m } else { n };
/// let mut s = Vector::new(min_mn);
/// let mut u = Matrix::new(m, m);
/// let mut vt = Matrix::new(n, n);
///
/// // perform SVD
/// sv_decomp(&mut s, &mut u, &mut vt, &mut a)?;
///
/// // define correct data
/// let s_correct = "┌       ┐\n\
///                  │ 5.000 │\n\
///                  │ 3.000 │\n\
///                  └       ┘";
/// let u_correct = "┌               ┐\n\
///                  │ -0.707 -0.707 │\n\
///                  │ -0.707  0.707 │\n\
///                  └               ┘";
/// let vt_correct = "┌                      ┐\n\
///                   │ -0.707 -0.707 -0.000 │\n\
///                   │ -0.236  0.236 -0.943 │\n\
///                   │ -0.667  0.667  0.333 │\n\
///                   └                      ┘";
///
/// // check solution
/// assert_eq!(format!("{:.3}", s), s_correct);
/// assert_eq!(format!("{:.3}", u), u_correct);
/// assert_eq!(format!("{:.3}", vt), vt_correct);
///
/// // check SVD: a == u * s * v
/// let mut usv = Matrix::new(m, n);
/// for i in 0..m {
///     for j in 0..n {
///         for k in 0..min_mn {
///             usv.plus_equal(i,j, u.get(i,k) * s.get(k) * vt.get(k,j));
///         }
///     }
/// }
/// let usv_correct = "┌                               ┐\n\
///                    │  3.000000  2.000000  2.000000 │\n\
///                    │  2.000000  3.000000 -2.000000 │\n\
///                    └                               ┘";
/// assert_eq!(format!("{:.6}", usv), usv_correct);
/// # Ok(())
/// # }
/// ```
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// let mut a = Matrix::from(&[
///     &[2.0, 4.0],
///     &[1.0, 3.0],
///     &[0.0, 0.0],
///     &[0.0, 0.0],
/// ]);
/// let (m, n) = a.dims();
/// let min_mn = if m < n { m } else { n };
/// let mut s = Vector::new(min_mn);
/// let mut u = Matrix::new(m, m);
/// let mut vt = Matrix::new(n, n);
/// sv_decomp(&mut s, &mut u, &mut vt, &mut a)?;
/// let s_correct = "┌      ┐\n\
///                  │ 5.46 │\n\
///                  │ 0.37 │\n\
///                  └      ┘";
/// let u_correct = "┌                         ┐\n\
///                  │ -0.82 -0.58  0.00  0.00 │\n\
///                  │ -0.58  0.82  0.00  0.00 │\n\
///                  │  0.00  0.00  1.00  0.00 │\n\
///                  │  0.00  0.00  0.00  1.00 │\n\
///                  └                         ┘";
/// let vt_correct = "┌             ┐\n\
///                   │ -0.40 -0.91 │\n\
///                   │ -0.91  0.40 │\n\
///                   └             ┘";
/// assert_eq!(format!("{:.2}", s), s_correct);
/// assert_eq!(format!("{:.2}", u), u_correct);
/// assert_eq!(format!("{:.2}", vt), vt_correct);
/// # Ok(())
/// # }
/// ```
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
        let data: &[&[f64]] = &[
            &[-s33, -s33, 1.0],
            &[ s33, -s33, 1.0],
            &[-s33,  s33, 1.0],
            &[ s33,  s33, 1.0],
        ];
        let mut a = Matrix::from(data);
        let a_copy = Matrix::from(data);

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

        // check SVD
        let mut usv = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for k in 0..min_mn {
                    usv[i + j * m] += u.data[i + k * m] * s.data[k] * vt.data[k + j * n];
                }
            }
        }
        assert_vec_approx_eq!(usv, a_copy.data, 1e-15);

        // done
        Ok(())
    }

    #[test]
    fn sv_decomp_1_works() -> Result<(), &'static str> {
        // matrix
        #[rustfmt::skip]
        let data: &[&[f64]] = &[
            &[1.0, 0.0, 1.0, 0.0],
            &[0.0, 1.0, 0.0, 1.0],
        ];
        let mut a = Matrix::from(data);
        let a_copy = Matrix::from(data);

        // allocate output data
        let (m, n) = a.dims();
        let min_mn = if m < n { m } else { n };
        let mut s = Vector::new(min_mn);
        let mut u = Matrix::new(m, m);
        let mut vt = Matrix::new(n, n);

        // calculate SVD
        sv_decomp(&mut s, &mut u, &mut vt, &mut a)?;

        // check
        let sqrt2 = std::f64::consts::SQRT_2;
        #[rustfmt::skip]
        let s_correct = Vector::from(&[
            sqrt2,
            sqrt2,
        ]);
        #[rustfmt::skip]
        let u_correct = Matrix::from(&[
            &[1.0, 0.0],
            &[0.0, 1.0],
        ]);
        #[rustfmt::skip]
        let vt_correct =Matrix::from(&[
            &[ 1.0/sqrt2,        0.0, 1.0/sqrt2,       0.0],
            &[       0.0,  1.0/sqrt2,       0.0, 1.0/sqrt2],
            &[-1.0/sqrt2,        0.0, 1.0/sqrt2,       0.0],
            &[       0.0, -1.0/sqrt2,       0.0, 1.0/sqrt2],
        ]);
        assert_vec_approx_eq!(u.data, u_correct.data, 1e-15);
        assert_vec_approx_eq!(s.data, s_correct.data, 1e-15);
        assert_vec_approx_eq!(vt.data, vt_correct.data, 1e-15);

        // check SVD
        let mut usv = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for k in 0..min_mn {
                    usv[i + j * m] += u.data[i + k * m] * s.data[k] * vt.data[k + j * n];
                }
            }
        }
        assert_vec_approx_eq!(usv, a_copy.data, 1e-15);

        // done
        Ok(())
    }
}
