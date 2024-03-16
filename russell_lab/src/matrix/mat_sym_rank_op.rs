use super::Matrix;
use crate::{to_i32, StrError, CBLAS_COL_MAJOR, CBLAS_LOWER, CBLAS_NO_TRANS, CBLAS_TRANS, CBLAS_UPPER};

extern "C" {
    // Performs one of the symmetric rank k operations
    // <https://www.netlib.org/lapack/explore-html/dc/d05/dsyrk_8f.html>
    fn cblas_dsyrk(
        layout: i32,
        uplo: i32,
        trans: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

/// (dsyrk) Performs a symmetric rank k operation
///
/// Performs one of the symmetric rank k operations:
///
/// ```text
/// First case:
///
///   c   := α ⋅ a   ⋅  aᵀ + β ⋅ c
/// (n,n)      (n,k)  (k,n)    (n,n)
/// ```
///
/// or
///
/// ```text
/// Second case:
///
///   c   := α ⋅  aᵀ  ⋅  a + β ⋅ c
/// (n,n)       (n,k)  (k,n)   (n,n)
/// ```
///
/// # Input
///
/// * `c` -- the (n,n) matrix (will be modified)
/// * `a` -- the (n,k) matrix on the first case or (k,n) on the second case
/// * `alpha` -- the α coefficient
/// * `beta` -- the β coefficient
/// * `upper` -- whether the upper triangle of `a` must be considered instead of the lower triangle
/// * `second_case` -- indicates the second case illustrated above
///
/// # Example
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     //  -1   2   0,
///     //   2   1   2,
///     //   0   2   1,
///     let ___ = 0.0;
///     let mut c_lower = Matrix::from(&[
///         [-1.0, ___, ___],
///         [ 2.0, 1.0, ___],
///         [ 0.0, 2.0, 1.0],
///     ]);
///
///     let a = Matrix::from(&[
///         [ 1.0,  2.0, -1.0],
///         [-1.0,  2.0,  0.0],
///     ]);
///
///     let (alpha, beta) = (-1.0, 2.0);
///
///     // c := -1 aᵀ⋅a + 2 c
///     mat_sym_rank_op(&mut c_lower, &a, alpha, beta, false, true).unwrap();
///
///     let c_ref = Matrix::from(&[
///         [-4.0,  ___,  ___],
///         [ 4.0, -6.0,  ___],
///         [ 1.0,  6.0,  1.0],
///     ]);
///     mat_approx_eq(&c_lower, &c_ref, 1e-15);
///     Ok(())
/// }
/// ```
pub fn mat_sym_rank_op(
    c: &mut Matrix,
    a: &Matrix,
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
        //   c   := α ⋅ a   ⋅  aᵀ + β ⋅ c
        // (n,n)      (n,k)  (k,n)    (n,n)
        if row != n {
            return Err("[a] matrix is incompatible");
        }
        (row, col, CBLAS_NO_TRANS)
    } else {
        //   c   := α ⋅  aᵀ  ⋅  a + β ⋅ c
        // (n,n)       (n,k)  (k,n)   (n,n)
        if col != n {
            return Err("[a] matrix is incompatible");
        }
        (row, row, CBLAS_TRANS)
    };
    let uplo = if upper { CBLAS_UPPER } else { CBLAS_LOWER };
    let n_i32 = to_i32(n);
    let k_i32 = to_i32(k);
    let ldc = n_i32;
    unsafe {
        cblas_dsyrk(
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
    use super::{mat_sym_rank_op, Matrix};
    use crate::mat_approx_eq;

    #[test]
    fn mat_sym_rank_op_fail_on_wrong_dims() {
        let mut c_2x2 = Matrix::new(2, 2);
        let mut c_3x2 = Matrix::new(3, 2);
        let a_2x3 = Matrix::new(2, 3);
        let a_3x2 = Matrix::new(3, 2);
        assert_eq!(
            mat_sym_rank_op(&mut c_3x2, &a_3x2, 2.0, 3.0, false, false).err(),
            Some("[c] matrix must be square")
        );
        assert_eq!(
            mat_sym_rank_op(&mut c_2x2, &a_3x2, 2.0, 3.0, false, false).err(),
            Some("[a] matrix is incompatible")
        );
        assert_eq!(
            mat_sym_rank_op(&mut c_2x2, &a_2x3, 2.0, 3.0, false, true).err(),
            Some("[a] matrix is incompatible")
        );
    }

    #[test]
    fn mat_sym_rank_op_works_first_case() {
        // c matrix
        // #[rustfmt::skip]
        // let c = Matrix::from(&[
        //     [ 3.0,  0.0, -3.0,  0.0],
        //     [ 0.0,  3.0,  1.0,  2.0],
        //     [-3.0,  1.0,  4.0,  1.0],
        //     [ 0.0,  2.0,  1.0,  3.0],
        // ]);
        #[rustfmt::skip]
        let mut c_lower = Matrix::from(&[
            [ 3.0,  0.0,  0.0,  0.0],
            [ 0.0,  3.0,  0.0,  0.0],
            [-3.0,  1.0,  4.0,  0.0],
            [ 0.0,  2.0,  1.0,  3.0],
        ]);
        #[rustfmt::skip]
        let mut c_upper = Matrix::from(&[
            [ 3.0,  0.0, -3.0,  0.0],
            [ 0.0,  3.0,  1.0,  2.0],
            [ 0.0,  0.0,  4.0,  1.0],
            [ 0.0,  0.0,  0.0,  3.0],
        ]);

        // a matrix
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [ 1.0,  2.0,  1.0,  1.0, -1.0,  0.0],
            [ 2.0,  2.0,  1.0,  0.0,  0.0,  0.0],
            [ 3.0,  1.0,  3.0,  1.0,  2.0, -1.0],
            [ 1.0,  0.0,  1.0, -1.0,  0.0,  0.0],
        ]);

        // constants
        let (alpha, beta) = (3.0, -1.0);

        // lower: c := 3⋅a⋅aᵀ - c
        mat_sym_rank_op(&mut c_lower, &a, alpha, beta, false, false).unwrap();
        // println!("{}", c_lower);
        #[rustfmt::skip]
        let c_ref = Matrix::from(&[
            [21.0,  0.0,  0.0,  0.0],
            [21.0, 24.0,  0.0,  0.0],
            [24.0, 32.0, 71.0,  0.0],
            [ 3.0,  7.0, 14.0,  6.0],
        ]);
        mat_approx_eq(&c_lower, &c_ref, 1e-15);

        // upper: c := 3⋅a⋅aᵀ - c
        mat_sym_rank_op(&mut c_upper, &a, alpha, beta, true, false).unwrap();
        // println!("{}", c_upper);
        #[rustfmt::skip]
        let c_ref = Matrix::from(&[
            [21.0, 21.0, 24.0,  3.0],
            [ 0.0, 24.0, 32.0,  7.0],
            [ 0.0,  0.0, 71.0, 14.0],
            [ 0.0,  0.0,  0.0,  6.0],
        ]);
        mat_approx_eq(&c_upper, &c_ref, 1e-15);
    }

    #[test]
    fn mat_sym_rank_op_works_second_case() {
        // c matrix
        // #[rustfmt::skip]
        // let c = Matrix::from(&[
        //     [ 3.0, 0.0, -3.0, 0.0, 0.0, 0.0],
        //     [ 0.0, 3.0,  1.0, 2.0, 2.0, 2.0],
        //     [-3.0, 1.0,  4.0, 1.0, 1.0, 1.0],
        //     [ 0.0, 2.0,  1.0, 3.0, 3.0, 3.0],
        //     [ 0.0, 2.0,  1.0, 3.0, 4.0, 3.0],
        //     [ 0.0, 2.0,  1.0, 3.0, 3.0, 4.0],
        // ]);
        #[rustfmt::skip]
        let mut c_lower = Matrix::from(&[
            [ 3.0, 0.0,  0.0, 0.0, 0.0, 0.0],
            [ 0.0, 3.0,  0.0, 0.0, 0.0, 0.0],
            [-3.0, 1.0,  4.0, 0.0, 0.0, 0.0],
            [ 0.0, 2.0,  1.0, 3.0, 0.0, 0.0],
            [ 0.0, 2.0,  1.0, 3.0, 4.0, 0.0],
            [ 0.0, 2.0,  1.0, 3.0, 3.0, 4.0],
        ]);
        #[rustfmt::skip]
        let mut c_upper = Matrix::from(&[
            [ 3.0, 0.0, -3.0, 0.0, 0.0, 0.0],
            [ 0.0, 3.0,  1.0, 2.0, 2.0, 2.0],
            [ 0.0, 0.0,  4.0, 1.0, 1.0, 1.0],
            [ 0.0, 0.0,  0.0, 3.0, 3.0, 3.0],
            [ 0.0, 0.0,  0.0, 0.0, 4.0, 3.0],
            [ 0.0, 0.0,  0.0, 0.0, 0.0, 4.0],
        ]);

        // a matrix
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [ 1.0,  2.0,  1.0,  1.0, -1.0,  0.0],
            [ 2.0,  2.0,  1.0,  0.0,  0.0,  0.0],
            [ 3.0,  1.0,  3.0,  1.0,  2.0, -1.0],
            [ 1.0,  0.0,  1.0, -1.0,  0.0,  0.0],
        ]);

        // constants
        let (alpha, beta) = (3.0, 1.0);

        // lower: c := 3⋅a⋅aᵀ + c
        mat_sym_rank_op(&mut c_lower, &a, alpha, beta, false, true).unwrap();
        println!("{}", c_lower);
        #[rustfmt::skip]
        let c_ref = Matrix::from(&[
            [48.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [27.0, 30.0,  0.0,  0.0,  0.0,  0.0],
            [36.0, 22.0, 40.0,  0.0,  0.0,  0.0],
            [ 9.0, 11.0, 10.0, 12.0,  0.0,  0.0],
            [15.0,  2.0, 16.0,  6.0, 19.0,  0.0],
            [-9.0, -1.0, -8.0,  0.0, -3.0,  7.0],
        ]);
        mat_approx_eq(&c_lower, &c_ref, 1e-15);

        // upper: c := 3⋅a⋅aᵀ + c
        mat_sym_rank_op(&mut c_upper, &a, alpha, beta, true, true).unwrap();
        println!("{}", c_upper);
        #[rustfmt::skip]
        let c_ref = Matrix::from(&[
            [48.0, 27.0, 36.0,  9.0, 15.0, -9.0],
            [ 0.0, 30.0, 22.0, 11.0,  2.0, -1.0],
            [ 0.0,  0.0, 40.0, 10.0, 16.0, -8.0],
            [ 0.0,  0.0,  0.0, 12.0,  6.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0, 19.0, -3.0],
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  7.0],
        ]);
        mat_approx_eq(&c_upper, &c_ref, 1e-15);
    }
}
