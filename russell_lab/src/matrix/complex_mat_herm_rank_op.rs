use super::ComplexMatrix;
use crate::{to_i32, StrError, CBLAS_COL_MAJOR, CBLAS_CONJ_TRANS, CBLAS_LOWER, CBLAS_NO_TRANS, CBLAS_UPPER};
use num_complex::Complex64;

extern "C" {
    // Performs one of the hermitian rank k operations
    // <https://www.netlib.org/lapack/explore-html/d1/db1/zherk_8f.html>
    fn cblas_zherk(
        layout: i32,
        uplo: i32,
        trans: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const Complex64,
        lda: i32,
        beta: f64,
        c: *mut Complex64,
        ldc: i32,
    );
}

/// (zherk) Performs a hermitian rank k operations
///
/// Performs one of the hermitian rank k operations
///
/// ```text
/// First case:
///
///   c   := α ⋅ a   ⋅  aᴴ + β ⋅ c
/// (n,n)      (n,k)  (k,n)    (n,n)
/// ```
///
/// or
///
/// ```text
/// Second case:
///
///   c   := α ⋅  aᴴ  ⋅  a + β ⋅ c
/// (n,n)       (n,k)  (k,n)   (n,n)
/// ```
///
/// where `c = cᴴ`
///
/// # Input
///
/// * `c` -- the (n,n) **hermitian** matrix (will be modified)
/// * `a` -- the (n,k) matrix on the first case or (k,n) on the second case
/// * `alpha` -- the α coefficient
/// * `beta` -- the β coefficient
/// * `upper` -- whether the upper triangle of `a` must be considered instead of the lower triangle
/// * `second_case` -- indicates the second case illustrated above
///
/// # Example
///
/// ```
/// use num_complex::Complex64;
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let ________________ = cpx!(0.0, 0.0);
///     #[rustfmt::skip]
///     let mut c_lower = ComplexMatrix::from(&[
///         [cpx!(-1.0,  0.0), ________________, ________________],
///         [cpx!( 2.0,  1.0), cpx!( 1.0,  0.0), ________________],
///         [cpx!( 0.0, -1.0), cpx!( 2.0,  3.0), cpx!( 1.0,  0.0)],
///     ]);
///
///     #[rustfmt::skip]
///     let a = ComplexMatrix::from(&[
///         [cpx!( 1.0, 1.0),  cpx!(2.0, -1.0), cpx!(-1.0, 3.0)],
///         [cpx!(-1.0, 2.0),  cpx!(2.0,  0.0), cpx!( 0.0, 2.0)],
///     ]);
///
///     let (alpha, beta) = (2.0, -3.0);
///
///     // c := 2 aᴴ⋅a - 3 c
///     complex_mat_herm_rank_op(&mut c_lower, &a, alpha, beta, false, true).unwrap();
///
///     let __________________ = cpx!(0.0, 0.0);
///     #[rustfmt::skip]
///     let c_ref = ComplexMatrix::from(&[
///         [cpx!( 17.0,   0.0), __________________, __________________],
///         [cpx!( -8.0,  11.0), cpx!( 15.0,   0.0), __________________],
///         [cpx!( 12.0,  -1.0), cpx!(-16.0, -27.0), cpx!( 25.0,   0.0)]
///
///     ]);
///     complex_mat_approx_eq(&c_lower, &c_ref, 1e-15);
///     Ok(())
/// }
/// ```
pub fn complex_mat_herm_rank_op(
    c: &mut ComplexMatrix,
    a: &ComplexMatrix,
    alpha: f64,
    beta: f64,
    upper: bool,
    second_case: bool,
) -> Result<(), StrError> {
    let (m, n) = c.dims();
    if m != n {
        return Err("[c] matrix must be square");
    }
    let (row, col) = a.dims();
    let (lda, k, trans) = if !second_case {
        //   c   := α ⋅ a   ⋅  aᴴ + β ⋅ c
        // (n,n)      (n,k)  (k,n)    (n,n)
        if row != n {
            return Err("[a] matrix is incompatible");
        }
        (row, col, CBLAS_NO_TRANS)
    } else {
        //   c   := α ⋅  aᴴ  ⋅  a + β ⋅ c
        // (n,n)       (n,k)  (k,n)   (n,n)
        if col != n {
            return Err("[a] matrix is incompatible");
        }
        (row, row, CBLAS_CONJ_TRANS)
    };
    let uplo = if upper { CBLAS_UPPER } else { CBLAS_LOWER };
    let n_i32 = to_i32(n);
    let k_i32 = to_i32(k);
    let ldc = n_i32;
    unsafe {
        cblas_zherk(
            CBLAS_COL_MAJOR,
            uplo,
            trans,
            n_i32,
            k_i32,
            alpha,
            a.as_data().as_ptr(),
            to_i32(lda),
            beta,
            c.as_mut_data().as_mut_ptr(),
            ldc,
        );
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_herm_rank_op, ComplexMatrix};
    use crate::{complex_mat_approx_eq, cpx};
    use num_complex::Complex64;

    fn check_matrices(full: &ComplexMatrix, lower: &ComplexMatrix, upper: &ComplexMatrix) {
        let (m, n) = full.dims();
        let (mm, nn) = lower.dims();
        let (mmm, nnn) = upper.dims();
        assert_eq!(m, n);
        assert!(mm == m && mmm == m && nn == m && nnn == m);
        let mut cc = ComplexMatrix::new(m, m);
        for i in 0..m {
            for j in 0..m {
                if i == j {
                    cc.set(i, j, lower.get(i, j));
                    assert_eq!(full.get(i, j).im, 0.0); // hermitian
                } else {
                    cc.set(i, j, lower.get(i, j) + upper.get(i, j));
                    assert_eq!(full.get(i, j).re, full.get(j, i).re); // hermitian
                    assert_eq!(full.get(i, j).im, -full.get(j, i).im); // hermitian
                }
            }
        }
        complex_mat_approx_eq(&full, &cc, 1e-15);
    }

    #[test]
    fn complex_mat_herm_rank_op_fail_on_wrong_dims() {
        let mut c_2x2 = ComplexMatrix::new(2, 2);
        let mut c_3x2 = ComplexMatrix::new(3, 2);
        let a_2x3 = ComplexMatrix::new(2, 3);
        let a_3x2 = ComplexMatrix::new(3, 2);
        let alpha = 2.0;
        let beta = 3.0;
        assert_eq!(
            complex_mat_herm_rank_op(&mut c_3x2, &a_3x2, alpha, beta, false, false).err(),
            Some("[c] matrix must be square")
        );
        assert_eq!(
            complex_mat_herm_rank_op(&mut c_2x2, &a_3x2, alpha, beta, false, false).err(),
            Some("[a] matrix is incompatible")
        );
        assert_eq!(
            complex_mat_herm_rank_op(&mut c_2x2, &a_2x3, alpha, beta, false, true).err(),
            Some("[a] matrix is incompatible")
        );
    }

    #[test]
    fn complex_mat_herm_rank_op_works_first_case() {
        // c matrix
        #[rustfmt::skip]
        let c = ComplexMatrix::from(&[
            [cpx!( 4.0,  0.0), cpx!(0.0, 1.0), cpx!(-3.0, 1.0), cpx!(0.0,  2.0)],
            [cpx!( 0.0, -1.0), cpx!(3.0, 0.0), cpx!( 1.0, 0.0), cpx!(2.0,  0.0)],
            [cpx!(-3.0, -1.0), cpx!(1.0, 0.0), cpx!( 4.0, 0.0), cpx!(1.0, -1.0)],
            [cpx!( 0.0, -2.0), cpx!(2.0, 0.0), cpx!( 1.0, 1.0), cpx!(4.0,  0.0)],
        ]);
        #[rustfmt::skip]
        let mut c_lower = ComplexMatrix::from(&[
            [cpx!( 4.0,  0.0), cpx!(0.0, 0.0), cpx!( 0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!( 0.0, -1.0), cpx!(3.0, 0.0), cpx!( 0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(-3.0, -1.0), cpx!(1.0, 0.0), cpx!( 4.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!( 0.0, -2.0), cpx!(2.0, 0.0), cpx!( 1.0, 1.0), cpx!(4.0, 0.0)],
        ]);
        #[rustfmt::skip]
        let mut c_upper = ComplexMatrix::from(&[
            [cpx!( 4.0, 0.0), cpx!(0.0, 1.0), cpx!(-3.0, 1.0), cpx!(0.0,  2.0)],
            [cpx!( 0.0, 0.0), cpx!(3.0, 0.0), cpx!( 1.0, 0.0), cpx!(2.0,  0.0)],
            [cpx!( 0.0, 0.0), cpx!(0.0, 0.0), cpx!( 4.0, 0.0), cpx!(1.0, -1.0)],
            [cpx!( 0.0, 0.0), cpx!(0.0, 0.0), cpx!( 0.0, 0.0), cpx!(4.0,  0.0)],
        ]);
        check_matrices(&c, &c_lower, &c_upper);

        // a matrix
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [cpx!( 1.0, -1.0), cpx!( 2.0, 0.0), cpx!( 1.0, 0.0), cpx!( 1.0, 0.0), cpx!(-1.0, 0.0), cpx!( 0.0, 0.0)],
            [cpx!( 2.0,  0.0), cpx!( 2.0, 0.0), cpx!( 1.0, 0.0), cpx!( 0.0, 0.0), cpx!( 0.0, 0.0), cpx!( 0.0, 1.0)],
            [cpx!( 3.0,  1.0), cpx!( 1.0, 0.0), cpx!( 3.0, 0.0), cpx!( 1.0, 0.0), cpx!( 2.0, 0.0), cpx!(-1.0, 0.0)],
            [cpx!( 1.0,  0.0), cpx!( 0.0, 0.0), cpx!( 1.0, 0.0), cpx!(-1.0, 0.0), cpx!( 0.0, 0.0), cpx!( 0.0, 1.0)],
        ]);

        // constants
        let (alpha, beta) = (3.0, -2.0);

        // reference data
        #[rustfmt::skip]
        let c_ref_full = ComplexMatrix::from(&[
            [cpx!(19.0,   0.0), cpx!(21.0,  -8.0), cpx!(24.0, -14.0), cpx!( 3.0,  -7.0)],
            [cpx!(21.0,   8.0), cpx!(24.0,   0.0), cpx!(31.0,  -9.0), cpx!( 8.0,   0.0)],
            [cpx!(24.0,  14.0), cpx!(31.0,   9.0), cpx!(70.0,   0.0), cpx!(13.0,   8.0)],
            [cpx!( 3.0,   7.0), cpx!( 8.0,   0.0), cpx!(13.0,  -8.0), cpx!( 4.0,   0.0)], 
        ]);
        #[rustfmt::skip]
        let c_ref_lower = ComplexMatrix::from(&[
            [cpx!(19.0,   0.0), cpx!( 0.0,   0.0), cpx!( 0.0,   0.0), cpx!( 0.0,   0.0)],
            [cpx!(21.0,   8.0), cpx!(24.0,   0.0), cpx!( 0.0,   0.0), cpx!( 0.0,   0.0)],
            [cpx!(24.0,  14.0), cpx!(31.0,   9.0), cpx!(70.0,   0.0), cpx!( 0.0,   0.0)],
            [cpx!( 3.0,   7.0), cpx!( 8.0,   0.0), cpx!(13.0,  -8.0), cpx!( 4.0,   0.0)], 
        ]);
        #[rustfmt::skip]
        let c_ref_upper = ComplexMatrix::from(&[
            [cpx!(19.0,   0.0), cpx!(21.0,  -8.0), cpx!(24.0, -14.0), cpx!( 3.0,  -7.0)],
            [cpx!( 0.0,   0.0), cpx!(24.0,   0.0), cpx!(31.0,  -9.0), cpx!( 8.0,   0.0)],
            [cpx!( 0.0,   0.0), cpx!( 0.0,   0.0), cpx!(70.0,   0.0), cpx!(13.0,   8.0)],
            [cpx!( 0.0,   0.0), cpx!( 0.0,   0.0), cpx!( 0.0,   0.0), cpx!( 4.0,   0.0)], 
        ]);
        check_matrices(&c_ref_full, &c_ref_lower, &c_ref_upper);

        // lower: c := 3 a⋅aᴴ - 2 c
        complex_mat_herm_rank_op(&mut c_lower, &a, alpha, beta, false, false).unwrap();
        // println!("{}", c_lower);
        complex_mat_approx_eq(&c_lower, &c_ref_lower, 1e-15);

        // upper: c := 3 a⋅aᴴ - 2 c
        complex_mat_herm_rank_op(&mut c_upper, &a, alpha, beta, true, false).unwrap();
        // println!("{}", c_upper);
        complex_mat_approx_eq(&c_upper, &c_ref_upper, 1e-15);
    }

    #[test]
    fn complex_mat_herm_rank_op_works_second_case() {
        // c matrix
        #[rustfmt::skip]
        let c = ComplexMatrix::from(&[
            [cpx!( 4.0,  0.0), cpx!(0.0, 1.0), cpx!(-3.0, 1.0), cpx!(0.0,  2.0)],
            [cpx!( 0.0, -1.0), cpx!(3.0, 0.0), cpx!( 1.0, 0.0), cpx!(2.0,  0.0)],
            [cpx!(-3.0, -1.0), cpx!(1.0, 0.0), cpx!( 4.0, 0.0), cpx!(1.0, -1.0)],
            [cpx!( 0.0, -2.0), cpx!(2.0, 0.0), cpx!( 1.0, 1.0), cpx!(4.0,  0.0)],
        ]);
        #[rustfmt::skip]
        let mut c_lower = ComplexMatrix::from(&[
            [cpx!( 4.0,  0.0), cpx!(0.0, 0.0), cpx!( 0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!( 0.0, -1.0), cpx!(3.0, 0.0), cpx!( 0.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!(-3.0, -1.0), cpx!(1.0, 0.0), cpx!( 4.0, 0.0), cpx!(0.0, 0.0)],
            [cpx!( 0.0, -2.0), cpx!(2.0, 0.0), cpx!( 1.0, 1.0), cpx!(4.0, 0.0)],
        ]);
        #[rustfmt::skip]
        let mut c_upper = ComplexMatrix::from(&[
            [cpx!( 4.0, 0.0), cpx!(0.0, 1.0), cpx!(-3.0, 1.0), cpx!(0.0,  2.0)],
            [cpx!( 0.0, 0.0), cpx!(3.0, 0.0), cpx!( 1.0, 0.0), cpx!(2.0,  0.0)],
            [cpx!( 0.0, 0.0), cpx!(0.0, 0.0), cpx!( 4.0, 0.0), cpx!(1.0, -1.0)],
            [cpx!( 0.0, 0.0), cpx!(0.0, 0.0), cpx!( 0.0, 0.0), cpx!(4.0,  0.0)],
        ]);
        check_matrices(&c, &c_lower, &c_upper);

        // a matrix
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [cpx!( 1.0, -1.0), cpx!( 2.0, 0.0), cpx!( 1.0, 0.0), cpx!( 1.0, 0.0)],
            [cpx!( 3.0,  1.0), cpx!( 1.0, 0.0), cpx!( 3.0, 0.0), cpx!( 1.0, 2.0)],
        ]);

        // constants
        let (alpha, beta) = (3.0, -2.0);

        // reference data
        #[rustfmt::skip]
        let c_ref_full = ComplexMatrix::from(&[
            [cpx!(28.0,   0.0), cpx!(15.0,   1.0), cpx!(36.0,  -8.0), cpx!(18.0,  14.0)],
            [cpx!(15.0,  -1.0), cpx!( 9.0,   0.0), cpx!(13.0,   0.0), cpx!( 5.0,   6.0)],
            [cpx!(36.0,   8.0), cpx!(13.0,   0.0), cpx!(22.0,   0.0), cpx!(10.0,  20.0)],
            [cpx!(18.0, -14.0), cpx!( 5.0,  -6.0), cpx!(10.0, -20.0), cpx!(10.0,   0.0)],
        ]);
        #[rustfmt::skip]
        let c_ref_lower = ComplexMatrix::from(&[
            [cpx!(28.0,   0.0), cpx!( 0.0,   0.0), cpx!( 0.0,   0.0), cpx!( 0.0,   0.0)],
            [cpx!(15.0,  -1.0), cpx!( 9.0,   0.0), cpx!( 0.0,   0.0), cpx!( 0.0,   0.0)],
            [cpx!(36.0,   8.0), cpx!(13.0,   0.0), cpx!(22.0,   0.0), cpx!( 0.0,   0.0)],
            [cpx!(18.0, -14.0), cpx!( 5.0,  -6.0), cpx!(10.0, -20.0), cpx!(10.0,   0.0)],
        ]);
        #[rustfmt::skip]
        let c_ref_upper = ComplexMatrix::from(&[
            [cpx!(28.0,   0.0), cpx!(15.0,   1.0), cpx!(36.0,  -8.0), cpx!(18.0,  14.0)],
            [cpx!( 0.0,   0.0), cpx!( 9.0,   0.0), cpx!(13.0,   0.0), cpx!( 5.0,   6.0)],
            [cpx!( 0.0,   0.0), cpx!( 0.0,   0.0), cpx!(22.0,   0.0), cpx!(10.0,  20.0)],
            [cpx!( 0.0,   0.0), cpx!( 0.0,   0.0), cpx!( 0.0,   0.0), cpx!(10.0,   0.0)],
        ]);
        check_matrices(&c_ref_full, &c_ref_lower, &c_ref_upper);

        // lower: c := 3 aᴴ⋅a - 2 c
        complex_mat_herm_rank_op(&mut c_lower, &a, alpha, beta, false, true).unwrap();
        println!("{}", c_lower);
        complex_mat_approx_eq(&c_lower, &c_ref_lower, 1e-15);

        // upper: c := 3 aᴴ⋅a - 2 c
        complex_mat_herm_rank_op(&mut c_upper, &a, alpha, beta, true, true).unwrap();
        println!("{}", c_upper);
        complex_mat_approx_eq(&c_upper, &c_ref_upper, 1e-15);
    }
}
