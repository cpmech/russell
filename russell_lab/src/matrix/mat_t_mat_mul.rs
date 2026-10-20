use super::Matrix;
use crate::StrError;

/// (dgemm) Performs the transpose(matrix)-matrix multiplication
///
/// ```text
///   c  :=  α   aᵀ  ⋅   b   +  β  c
/// (m,n)      (m,k)   (k,n)     (m,n)
///          a:(k,m)
/// ```
///
/// See also: <https://www.netlib.org/lapack/explore-html/d7/d2b/dgemm_8f.html>
///
/// See also: [`crate::small_mat_t_mat_mul`] (the stack-allocated counterpart).
///
/// # Examples
///
/// ```
/// use russell_lab::{mat_t_mat_mul, Matrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Matrix::from(&[
///         [1.0, 3.0, 5.0],
///         [2.0, 4.0, 6.0],
///     ]);
///     let b = Matrix::from(&[
///         [-1.0, -2.0, -3.0],
///         [-4.0, -5.0, -6.0],
///     ]);
///     let mut c = Matrix::new(3, 3);
///     mat_t_mat_mul(&mut c, 1.0, &a, &b, 0.0)?;
///     let correct = "┌             ┐\n\
///                    │  -9 -12 -15 │\n\
///                    │ -19 -26 -33 │\n\
///                    │ -29 -40 -51 │\n\
///                    └             ┘";
///     assert_eq!(format!("{}", c), correct);
///     Ok(())
/// }
/// ```
pub fn mat_t_mat_mul(c: &mut Matrix, alpha: f64, a: &Matrix, b: &Matrix, beta: f64) -> Result<(), StrError> {
    let (m, n) = c.dims();
    let k = a.nrow();
    if a.ncol() != m || b.nrow() != k || b.ncol() != n {
        return Err("matrices are incompatible");
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        c.fill(0.0);
        return Ok(());
    }
    // One dispatch for both backends: the transposed native kernel in the default
    // build, the linked library's `dgemm('T','N')` through the shim otherwise.
    crate::base::blas_dispatch::dgemm_tn(alpha, a.as_data(), b.as_data(), beta, c.as_mut_data(), m, n, k);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Matrix, mat_t_mat_mul};
    use crate::mat_approx_eq;

    #[test]
    fn mat_t_mat_mul_fails_on_wrong_dims() {
        let a_2x1 = Matrix::new(2, 1);
        let a_1x2 = Matrix::new(1, 2);
        let b_2x1 = Matrix::new(2, 1);
        let b_1x3 = Matrix::new(1, 3);
        let mut c_2x2 = Matrix::new(2, 2);
        assert_eq!(
            mat_t_mat_mul(&mut c_2x2, 1.0, &a_2x1, &b_2x1, 0.0),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_t_mat_mul(&mut c_2x2, 1.0, &a_1x2, &b_2x1, 0.0),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_t_mat_mul(&mut c_2x2, 1.0, &a_2x1, &b_1x3, 0.0),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn mat_t_mat_mul_0x0_works() {
        let a = Matrix::new(0, 0);
        let b = Matrix::new(0, 0);
        let mut c = Matrix::new(0, 0);
        mat_t_mat_mul(&mut c, 2.0, &a, &b, 0.0).unwrap();

        let a = Matrix::new(0, 1);
        let b = Matrix::new(0, 1);
        let mut c = Matrix::from(&[[123.0]]);
        mat_t_mat_mul(&mut c, 2.0, &a, &b, 0.0).unwrap();
        let correct = &[
            [0.0], //
        ];
        mat_approx_eq(&c, correct, 1e-15);
    }

    /// The naive reference, written from the definition rather than reusing any
    /// production helper.
    fn naive_t_mat_mul(c: &mut Matrix, alpha: f64, a: &Matrix, b: &Matrix, beta: f64, m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                // c[i][j] = sum_l a[l][i] * b[l][j]
                for l in 0..k {
                    sum += a.get(l, i) * b.get(l, j);
                }
                let old = c.get(i, j);
                c.set(
                    i,
                    j,
                    if beta == 0.0 {
                        alpha * sum
                    } else {
                        alpha * sum + beta * old
                    },
                );
            }
        }
    }

    /// Note on the accelerated path: this calls `dgemm('T','N')` -- CBLAS's
    /// `CblasTrans` through the shim, and `native::blas::dgemm_tn` in the default
    /// build, which packs `A`'s columns from `A`'s rows. Until those kernels
    /// existed the native path *transposed `A` into a temporary* and called the
    /// plain `dgemm`; the transpose was measured at ~130 us at m=n=k=400 against
    /// ~25 ms for the multiplication, so it was never the bottleneck, but it was
    /// also a second m*k buffer and an extra pass over it.
    ///
    /// Reinterpreting the leading dimensions instead -- the usual row/column-major
    /// trick of calling `dgemm(b, a, ...)` and reading the result transposed --
    /// does *not* work here, and it fails silently: `dgemm` would write an `n x m`
    /// result with column stride `n`, while `c` has stride `m`, so the layouts
    /// coincide only when `m == n`.
    ///
    /// Sizes chosen to reach every branch and every padding/transpose path.
    ///
    /// This matters more than usual here: a wrong leading dimension in the
    /// transposed kernel is silently a different product. The pre-existing tests
    /// are all 2x2 or 3x2, which cannot distinguish the transposed leading
    /// dimensions from the correct ones.
    #[test]
    fn mat_t_mat_mul_matches_naive_on_every_shape() {
        let shapes = [
            // a is k x m, b is k x n, c is m x n
            (1, 1, 1),
            (1, 1, 5),
            (1, 4, 1),
            (4, 1, 1),
            (2, 3, 4),
            (3, 2, 5),
            (5, 5, 5),
            // non-square and asymmetric: these are what pin the leading dims
            (7, 11, 13),
            (13, 7, 11),
            (11, 13, 7),
            // past the block sizes (MC=128, KC=256, NC=256)
            (130, 6, 8),
            (6, 130, 8),
            (8, 6, 260),
            (260, 9, 7),
            (200, 130, 300),
        ];
        for (m, n, k) in shapes {
            let mut a = Matrix::new(k, m);
            let mut b = Matrix::new(k, n);
            for l in 0..k {
                for i in 0..m {
                    a.set(l, i, (((l * 7 + i * 13) % 17) as f64) - 8.0);
                }
                for j in 0..n {
                    b.set(l, j, (((l * 5 + j * 11) % 19) as f64) - 9.0);
                }
            }
            let c0 = {
                let mut c = Matrix::new(m, n);
                for i in 0..m {
                    for j in 0..n {
                        c.set(i, j, (((i + j) % 4) as f64) * 0.5);
                    }
                }
                c
            };

            for (alpha, beta) in [(1.0, 1.0), (1.0, 0.0), (2.5, -1.5), (0.0, 3.0)] {
                let mut want = c0.clone();
                naive_t_mat_mul(&mut want, alpha, &a, &b, beta, m, n, k);
                let mut got = c0.clone();
                mat_t_mat_mul(&mut got, alpha, &a, &b, beta).unwrap();
                for i in 0..m {
                    for j in 0..n {
                        let scale = want.get(i, j).abs().max(1.0);
                        assert!(
                            (got.get(i, j) - want.get(i, j)).abs() / scale < 1e-11,
                            "m={m} n={n} k={k} alpha={alpha} beta={beta}: ({i},{j}) is {} not {}",
                            got.get(i, j),
                            want.get(i, j)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mat_t_mat_mul_works_1() {
        let a = Matrix::from(&[
            // 3 x 2
            [1.0, 0.5],
            [2.0, 0.75],
            [3.0, 1.5],
        ]);
        let b = Matrix::from(&[
            // 3 x 4
            [0.1, 0.5, 0.5, 0.75],
            [0.2, 2.0, 2.0, 2.00],
            [0.3, 0.5, 0.5, 0.50],
        ]);
        let mut c = Matrix::new(2, 4);
        // c := 2⋅aᵀ⋅b
        mat_t_mat_mul(&mut c, 2.0, &a, &b, 0.0).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [2.80, 12.0, 12.0, 12.50],
            [1.30,  5.0,  5.0, 5.25],
        ];
        mat_approx_eq(&c, correct, 1e-15);
    }

    #[test]
    fn mat_t_mat_mul_works_2() {
        let a = Matrix::from(&[
            // 3 x 2
            [1.0, 0.5],
            [2.0, 0.75],
            [3.0, 1.5],
        ]);
        let b = Matrix::from(&[
            // 3 x 4
            [0.1, 0.5, 0.5, 0.75],
            [0.2, 2.0, 2.0, 2.00],
            [0.3, 0.5, 0.5, 0.50],
        ]);
        let mut c = Matrix::filled(2, 4, 100.0);
        // c := 2 aᵀ⋅b + 10 c
        mat_t_mat_mul(&mut c, 2.0, &a, &b, 10.0).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [1002.80, 1012.0, 1012.0, 1012.50],
            [1001.30, 1005.0, 1005.0, 1005.25],
        ];
        mat_approx_eq(&c, correct, 1e-15);
    }
}
