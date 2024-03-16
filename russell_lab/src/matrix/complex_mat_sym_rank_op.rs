use super::ComplexMatrix;
use crate::{to_i32, StrError, CBLAS_COL_MAJOR, CBLAS_LOWER, CBLAS_NO_TRANS, CBLAS_TRANS, CBLAS_UPPER};
use num_complex::Complex64;

extern "C" {
    // Performs one of the symmetric rank k operations
    // <https://www.netlib.org/lapack/explore-html/de/d54/zsyrk_8f.html>
    fn cblas_zsyrk(
        layout: i32,
        uplo: i32,
        trans: i32,
        n: i32,
        k: i32,
        alpha: *const Complex64,
        a: *const Complex64,
        lda: i32,
        beta: *const Complex64,
        c: *mut Complex64,
        ldc: i32,
    );
}

/// (zsyrk) Performs a symmetric rank k operation (complex version)
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
/// use num_complex::Complex64;
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     //  -1   2   0,
///     //   2   1   2,
///     //   0   2   1,
///     let ___ = 0.0;
///     #[rustfmt::skip]
///     let mut c_lower = ComplexMatrix::from(&[
///         [-1.0, ___, ___],
///         [ 2.0, 1.0, ___],
///         [ 0.0, 2.0, 1.0],
///     ]);
///
///     #[rustfmt::skip]
///     let a = ComplexMatrix::from(&[
///         [ 1.0,  2.0, -1.0],
///         [-1.0,  2.0,  0.0],
///     ]);
///
///     let (alpha, beta) = (cpx!(-1.0, 1.0), cpx!(2.0, -1.0));
///
///     // c := (-1+1i) aᵀ⋅a + (2-1i) c
///     complex_mat_sym_rank_op(&mut c_lower, &a, alpha, beta, false, true).unwrap();
///
///     let ________________ = cpx!(0.0, 0.0);
///     #[rustfmt::skip]
///     let c_ref = ComplexMatrix::from(&[
///         [cpx!(-4.0,  3.0), ________________, ________________],
///         [cpx!( 4.0, -2.0), cpx!(-6.0,  7.0), ________________],
///         [cpx!( 1.0, -1.0), cpx!( 6.0, -4.0), cpx!( 1.0,  0.0)],
///     ]);
///     complex_mat_approx_eq(&c_lower, &c_ref, 1e-15);
///     Ok(())
/// }
/// ```
pub fn complex_mat_sym_rank_op(
    c: &mut ComplexMatrix,
    a: &ComplexMatrix,
    alpha: Complex64,
    beta: Complex64,
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
        cblas_zsyrk(
            CBLAS_COL_MAJOR,
            uplo,
            trans,
            n_i32,
            k_i32,
            &alpha,
            a.as_data().as_ptr(),
            to_i32(lda),
            &beta,
            c.as_mut_data().as_mut_ptr(),
            ldc,
        );
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_sym_rank_op, ComplexMatrix};
    use crate::{complex_mat_approx_eq, cpx};
    use num_complex::Complex64;

    #[test]
    fn complex_mat_sym_rank_op_fail_on_wrong_dims() {
        let mut c_2x2 = ComplexMatrix::new(2, 2);
        let mut c_3x2 = ComplexMatrix::new(3, 2);
        let a_2x3 = ComplexMatrix::new(2, 3);
        let a_3x2 = ComplexMatrix::new(3, 2);
        let alpha = cpx!(2.0, 1.0);
        let beta = cpx!(3.0, 1.0);
        assert_eq!(
            complex_mat_sym_rank_op(&mut c_3x2, &a_3x2, alpha, beta, false, false).err(),
            Some("[c] matrix must be square")
        );
        assert_eq!(
            complex_mat_sym_rank_op(&mut c_2x2, &a_3x2, alpha, beta, false, false).err(),
            Some("[a] matrix is incompatible")
        );
        assert_eq!(
            complex_mat_sym_rank_op(&mut c_2x2, &a_2x3, alpha, beta, false, true).err(),
            Some("[a] matrix is incompatible")
        );
    }

    #[test]
    fn complex_mat_sym_rank_op_works_first_case() {
        // c matrix
        // #[rustfmt::skip]
        // let c = ComplexMatrix::from(&[
        //     [cpx!( 3.0,  1.0), cpx!(0.0,  0.0), cpx!(-2.0,  0.0), cpx!(0.0,  0.0)],
        //     [cpx!(-1.0,  0.0), cpx!(3.0,  0.0), cpx!( 0.0,  0.0), cpx!(2.0,  0.0)],
        //     [cpx!(-4.0,  0.0), cpx!(1.0,  0.0), cpx!( 3.0,  0.0), cpx!(1.0,  0.0)],
        //     [cpx!(-1.0,  0.0), cpx!(2.0,  0.0), cpx!( 0.0,  0.0), cpx!(3.0, -1.0)],
        // ]);
        #[rustfmt::skip]
        let mut c_lower = ComplexMatrix::from(&[
            [cpx!( 3.0,  1.0), cpx!(0.0,  0.0),  cpx!(0.0,  0.0), cpx!(0.0,  0.0)],
            [cpx!(-1.0,  0.0), cpx!(3.0,  0.0),  cpx!(0.0,  0.0), cpx!(0.0,  0.0)],
            [cpx!(-4.0,  0.0), cpx!(1.0,  0.0),  cpx!(3.0,  0.0), cpx!(0.0,  0.0)],
            [cpx!(-1.0,  0.0), cpx!(2.0,  0.0),  cpx!(0.0,  0.0), cpx!(3.0, -1.0)],
        ]);
        #[rustfmt::skip]
        let mut c_upper = ComplexMatrix::from(&[
            [cpx!( 3.0,  1.0), cpx!(0.0,  0.0), cpx!(-2.0,  0.0), cpx!(0.0,  0.0)],
            [cpx!( 0.0,  0.0), cpx!(3.0,  0.0), cpx!( 0.0,  0.0), cpx!(2.0,  0.0)],
            [cpx!( 0.0,  0.0), cpx!(0.0,  0.0), cpx!( 3.0,  0.0), cpx!(1.0,  0.0)],
            [cpx!( 0.0,  0.0), cpx!(0.0,  0.0), cpx!( 0.0,  0.0), cpx!(3.0, -1.0)],
        ]);

        // a matrix
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [cpx!( 1.0, -1.0),  cpx!(2.0, 0.0),  cpx!(1.0, 0.0), cpx!( 1.0, 0.0), cpx!(-1.0, 0.0), cpx!( 0.0,  0.0)],
            [cpx!( 2.0,  0.0),  cpx!(2.0, 0.0),  cpx!(1.0, 0.0), cpx!( 0.0, 0.0), cpx!( 0.0, 0.0), cpx!( 0.0,  1.0)],
            [cpx!( 3.0,  1.0),  cpx!(1.0, 0.0),  cpx!(3.0, 0.0), cpx!( 1.0, 0.0), cpx!( 2.0, 0.0), cpx!(-1.0,  0.0)],
            [cpx!( 1.0,  0.0),  cpx!(0.0, 0.0),  cpx!(1.0, 0.0), cpx!(-1.0, 0.0), cpx!( 0.0, 0.0), cpx!( 0.0,  1.0)],
        ]);

        // constants
        let (alpha, beta) = (cpx!(3.0, 0.0), cpx!(1.0, 0.0));

        // lower: c := 3⋅a⋅aᵀ + c
        complex_mat_sym_rank_op(&mut c_lower, &a, alpha, beta, false, false).unwrap();
        // println!("{}", c_lower);
        #[rustfmt::skip]
        let c_ref = ComplexMatrix::from(&[
            [cpx!(24.0, -5.0), cpx!( 0.0,  0.0), cpx!( 0.0,   0.0),  cpx!(0.0,  0.0)],
            [cpx!(20.0, -6.0), cpx!(27.0,  0.0), cpx!( 0.0,   0.0),  cpx!(0.0,  0.0)],
            [cpx!(20.0, -6.0), cpx!(34.0,  3.0), cpx!(75.0,  18.0),  cpx!(0.0,  0.0)],
            [cpx!( 2.0, -3.0), cpx!( 8.0,  0.0), cpx!(15.0,   0.0),  cpx!(9.0, -1.0)],
        ]);
        complex_mat_approx_eq(&c_lower, &c_ref, 1e-15);

        // upper: c := 3⋅a⋅aᵀ + c
        complex_mat_sym_rank_op(&mut c_upper, &a, alpha, beta, true, false).unwrap();
        // println!("{}", c_upper);
        #[rustfmt::skip]
        let c_ref = ComplexMatrix::from(&[
            [cpx!(24.0, -5.0), cpx!(21.0, -6.0), cpx!(22.0,  -6.0),  cpx!(3.0, -3.0)],
            [cpx!( 0.0,  0.0), cpx!(27.0,  0.0), cpx!(33.0,   3.0),  cpx!(8.0,  0.0)],
            [cpx!( 0.0,  0.0), cpx!( 0.0,  0.0), cpx!(75.0,  18.0), cpx!(16.0,  0.0)],
            [cpx!( 0.0,  0.0), cpx!( 0.0,  0.0), cpx!( 0.0,   0.0),  cpx!(9.0, -1.0)],
        ]);
        complex_mat_approx_eq(&c_upper, &c_ref, 1e-15);
    }

    #[test]
    fn complex_mat_sym_rank_op_works_second_case() {
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
        let mut c_lower = ComplexMatrix::from(&[
            [ 3.0, 0.0,  0.0, 0.0, 0.0, 0.0],
            [ 0.0, 3.0,  0.0, 0.0, 0.0, 0.0],
            [-3.0, 1.0,  4.0, 0.0, 0.0, 0.0],
            [ 0.0, 2.0,  1.0, 3.0, 0.0, 0.0],
            [ 0.0, 2.0,  1.0, 3.0, 4.0, 0.0],
            [ 0.0, 2.0,  1.0, 3.0, 3.0, 4.0],
        ]);
        #[rustfmt::skip]
        let mut c_upper = ComplexMatrix::from(&[
            [ 3.0, 0.0, -3.0, 0.0, 0.0, 0.0],
            [ 0.0, 3.0,  1.0, 2.0, 2.0, 2.0],
            [ 0.0, 0.0,  4.0, 1.0, 1.0, 1.0],
            [ 0.0, 0.0,  0.0, 3.0, 3.0, 3.0],
            [ 0.0, 0.0,  0.0, 0.0, 4.0, 3.0],
            [ 0.0, 0.0,  0.0, 0.0, 0.0, 4.0],
        ]);

        // a matrix
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [ 1.0,  2.0,  1.0,  1.0, -1.0,  0.0],
            [ 2.0,  2.0,  1.0,  0.0,  0.0,  0.0],
            [ 3.0,  1.0,  3.0,  1.0,  2.0, -1.0],
            [ 1.0,  0.0,  1.0, -1.0,  0.0,  0.0],
        ]);

        // constants
        let (alpha, beta) = (cpx!(3.0, 0.0), cpx!(1.0, 0.0));

        // lower: c := 3⋅a⋅aᵀ + c
        complex_mat_sym_rank_op(&mut c_lower, &a, alpha, beta, false, true).unwrap();
        // println!("{}", c_lower);
        #[rustfmt::skip]
        let c_ref = ComplexMatrix::from(&[
            [48.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [27.0, 30.0,  0.0,  0.0,  0.0,  0.0],
            [36.0, 22.0, 40.0,  0.0,  0.0,  0.0],
            [ 9.0, 11.0, 10.0, 12.0,  0.0,  0.0],
            [15.0,  2.0, 16.0,  6.0, 19.0,  0.0],
            [-9.0, -1.0, -8.0,  0.0, -3.0,  7.0],
        ]);
        complex_mat_approx_eq(&c_lower, &c_ref, 1e-15);

        // upper: c := 3⋅a⋅aᵀ + c
        complex_mat_sym_rank_op(&mut c_upper, &a, alpha, beta, true, true).unwrap();
        // println!("{}", c_upper);
        #[rustfmt::skip]
        let c_ref = ComplexMatrix::from(&[
            [48.0, 27.0, 36.0,  9.0, 15.0, -9.0],
            [ 0.0, 30.0, 22.0, 11.0,  2.0, -1.0],
            [ 0.0,  0.0, 40.0, 10.0, 16.0, -8.0],
            [ 0.0,  0.0,  0.0, 12.0,  6.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0, 19.0, -3.0],
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  7.0],
        ]);
        complex_mat_approx_eq(&c_upper, &c_ref, 1e-15);
    }
}
