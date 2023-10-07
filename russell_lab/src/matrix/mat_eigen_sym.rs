use super::Matrix;
use crate::{to_i32, CcBool, StrError, Vector, C_TRUE};

extern "C" {
    // Computes the eigenvalues and eigenvectors of a symmetric matrix
    // <https://netlib.org/lapack/explore-html/dd/d4c/dsyev_8f.html>
    fn c_dsyev(
        calc_v: CcBool,
        upper: CcBool,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        w: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

/// Calculates the eigenvalues and eigenvectors of a symmetric matrix
///
/// Computes the eigenvalues `l` and eigenvectors `v`, such that:
///
/// ```text
/// a ⋅ vj = lj ⋅ vj
/// ```
///
/// where `lj` is the component j of `l` and `vj` is the column j of `v`.
///
/// The computed eigenvectors are normalized to have Euclidean norm
/// equal to 1 and largest component real.
///
/// # Input
///
/// * `a` -- (modified on exit) matrix to compute eigenvalues (SYMMETRIC and SQUARE)
///
/// # Output
///
/// * `l` -- the eigenvalues
/// * `a` -- will hold the eigenvectors as columns
pub fn mat_eigen_sym(l: &mut Vector, a: &mut Matrix) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if m == 0 {
        return Err("matrix dimension must be ≥ 1");
    }
    if l.dim() != n {
        return Err("l vector has incompatible dimension");
    }
    let n_i32 = to_i32(n);
    let lda = n_i32;
    const EXTRA: i32 = 1;
    let lwork = 3 * n_i32 + EXTRA; // max(1,3*N-1), thus, 2 extra spaces effectively
    let mut work = vec![0.0; lwork as usize];
    let mut info = 0;
    unsafe {
        c_dsyev(
            C_TRUE,
            C_TRUE,
            &n_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            l.as_mut_data().as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (dsyev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (dsyev): An argument had an illegal value");
    } else if info > 0 {
        println!("LAPACK ERROR (dsyev): {} off-diagonal elements of an intermediate tri-diagonal form did not converge to zero", info - 1);
        return Err("LAPACK ERROR (dsyev): The algorithm failed to converge");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_eigen_sym, Matrix};
    use crate::math::SQRT_2;
    use crate::matrix::testing::check_eigen_real;
    use crate::{mat_approx_eq, vec_approx_eq, AsArray2D, Vector};

    fn calc_eigen<'a, T>(data: &'a T) -> (Vector, Matrix)
    where
        T: AsArray2D<'a, f64>,
    {
        let mut a = Matrix::from(data);
        let n = a.ncol();
        let mut l = Vector::new(n);
        mat_eigen_sym(&mut l, &mut a).unwrap();
        (l, a)
    }

    #[test]
    fn mat_eigen_sym_handles_errors() {
        let mut a = Matrix::new(0, 1);
        let mut l = Vector::new(0);
        assert_eq!(mat_eigen_sym(&mut l, &mut a).err(), Some("matrix must be square"));
        let mut a = Matrix::new(0, 0);
        assert_eq!(
            mat_eigen_sym(&mut l, &mut a).err(),
            Some("matrix dimension must be ≥ 1")
        );
        let mut a = Matrix::new(1, 1);
        assert_eq!(
            mat_eigen_sym(&mut l, &mut a).err(),
            Some("l vector has incompatible dimension")
        );
    }

    #[test]
    fn mat_eigen_sym_works_0() {
        // 1x1 matrix
        let data = &[[2.0]];
        let (l, v) = calc_eigen(data);
        mat_approx_eq(&v, &[[1.0]], 1e-15);
        vec_approx_eq(l.as_data(), &[2.0], 1e-15);

        // 2x2 matrix
        let data = &[[2.0, 1.0], [1.0, 2.0]];
        let (l, v) = calc_eigen(data);
        mat_approx_eq(
            &v,
            &[[-1.0 / SQRT_2, 1.0 / SQRT_2], [1.0 / SQRT_2, 1.0 / SQRT_2]],
            1e-15,
        );
        vec_approx_eq(l.as_data(), &[1.0, 3.0], 1e-15);
    }

    #[test]
    fn mat_eigen_sym_works_1() {
        // all zero
        #[rustfmt::skip]
        let data = &[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let (l, v) = calc_eigen(data);
        #[rustfmt::skip]
        let correct = &[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(l.as_data(), &[0.0, 0.0, 0.0], 1e-15);

        // 2-repeated, with one zero diagonal entry
        #[rustfmt::skip]
        let data = &[
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let (l, v) = calc_eigen(data);
        #[rustfmt::skip]
        let correct = &[
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(l.as_data(), &[0.0, 2.0, 2.0], 1e-15);
        check_eigen_real(data, &v, &l, 1e-15);

        // 3-repeated / diagonal
        #[rustfmt::skip]
        let data = &[
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ];
        let (l, v) = calc_eigen(data);
        #[rustfmt::skip]
        let correct = &[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(l.as_data(), &[2.0, 2.0, 2.0], 1e-15);
        check_eigen_real(data, &v, &l, 1e-15);
    }

    #[test]
    fn mat_eigen_sym_works_2() {
        #[rustfmt::skip]
        let data = &[
		    [2.0, 0.0, 0.0],
		    [0.0, 3.0, 4.0],
		    [0.0, 4.0, 9.0],
        ];
        let (l, v) = calc_eigen(data);
        let d = 1.0 / f64::sqrt(5.0);
        #[rustfmt::skip]
        let correct = &[
            [ 0.0,   1.0, 0.0  ],
            [-2.0*d, 0.0, 1.0*d],
            [ 1.0*d, 0.0, 2.0*d],
        ];
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(l.as_data(), &[1.0, 2.0, 11.0], 1e-15);
        check_eigen_real(data, &v, &l, 1e-15);
    }

    #[test]
    fn mat_eigen_sym_works_3() {
        #[rustfmt::skip]
        let data = &[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 2.0],
            [3.0, 2.0, 2.0],
        ];
        let (l, v) = calc_eigen(data);
        check_eigen_real(data, &v, &l, 1e-14);
    }

    #[test]
    fn mat_eigen_sym_works_4() {
        #[rustfmt::skip]
        let data = &[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 0.0, 2.0, 4.0],
            [3.0, 0.0, 2.0, 1.0, 3.0],
            [4.0, 2.0, 1.0, 1.0, 2.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ];
        let (l, v) = calc_eigen(data);
        check_eigen_real(data, &v, &l, 1e-14);
    }

    #[test]
    fn mat_eigen_sym_works_5() {
        let samples = &[
            (
                // 0
                [[1.0, 2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 0.0, -2.0]],
                1e-15,
            ),
            (
                // 1
                [[-100.0, 33.0, 0.0], [33.0, -200.0, 0.0], [0.0, 0.0, 150.0]],
                1e-14,
            ),
            (
                // 2
                [[1.0, 2.0, 4.0], [2.0, -2.0, 3.0], [4.0, 3.0, -2.0]],
                1e-14,
            ),
            (
                // 3
                [[-100.0, -10.0, 20.0], [-10.0, -200.0, 15.0], [20.0, 15.0, -300.0]],
                1e-13,
            ),
            (
                // 4
                [[-100.0, 0.0, -10.0], [0.0, -200.0, 0.0], [-10.0, 0.0, 100.0]],
                1e-13,
            ),
            (
                // 5
                [[0.13, 1.2, 0.0], [1.2, -20.0, 0.0], [0.0, 0.0, -28.0]],
                1e-14,
            ),
            (
                // 6
                [[-10.0, 3.3, 0.0], [3.3, -2.0, 0.0], [0.0, 0.0, 1.5]],
                1e-15,
            ),
            (
                // 7
                [[0.1, 0.2, 0.8], [0.2, -1.3, 0.3], [0.8, 0.3, -0.2]],
                1e-15,
            ),
            (
                // 8
                [[-10.0, -1.0, 2.0], [-1.0, -20.0, 1.0], [2.0, 1.0, -30.0]],
                1e-14,
            ),
            (
                // 9
                [[-10.0, 0.0, -1.0], [0.0, -20.0, 0.0], [-1.0, 0.0, 10.0]],
                1e-14,
            ),
        ];
        // let mut test_id = 0;
        for (data, tol) in samples {
            // println!("test = {}", test_id);
            let (l, v) = calc_eigen(data);
            check_eigen_real(data, &v, &l, *tol);
            // test_id += 1;
        }
    }
}
