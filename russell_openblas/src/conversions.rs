/// Converts nested slice into vector representing a matrix in col-major format
///
/// Example of col-major data:
///
/// ```text
///        _      _
///       |  0  3  |
///   A = |  1  4  |            ⇒     a = [0, 1, 2, 3, 4, 5]
///       |_ 2  5 _|(m x n)
///
///   a[i+j*m] = A[i][j]
///
/// ```
pub fn slice_to_colmajor(a: &[&[f64]]) -> Result<Vec<f64>, &'static str> {
    let nrow = a.len();
    let ncol = a[0].len();
    let mut data = vec![0.0; nrow * ncol];
    for i in 0..nrow {
        if a[i].len() != ncol {
            return Err("all rows must have the same number of columns");
        }
        for j in 0..ncol {
            data[i + j * nrow] = a[i][j];
        }
    }
    Ok(data)
}

/// Extracts LAPACK (dgeev) eigenvectors from its compact representation
///
/// # Output
///
/// * `vl_real` -- pre-allocated, n*n col-major, **left** eigenvectors; real part
/// * `vl_imag` -- pre-allocated, n*n col-major, **left** eigenvectors; imaginary part
/// * `vr_real` -- pre-allocated, n*n col-major, **right** eigenvectors; real part
/// * `vr_imag` -- pre-allocated, n*n col-major, **right** eigenvectors; imaginary part
///
/// # Input
///
/// * `w_imag` -- n, eigenvalues; imaginary part
/// * `vl` -- n*n, output of dgeev
/// * `vr` -- n*n, output of dgeev
///
pub fn extract_lapack_eigenvectors(
    vl_real: &mut [f64],
    vl_imag: &mut [f64],
    vr_real: &mut [f64],
    vr_imag: &mut [f64],
    w_imag: &[f64],
    vl: &[f64],
    vr: &[f64],
) -> Result<(), &'static str> {
    // check
    let n = w_imag.len();
    let nn = n * n;
    if vl_real.len() != nn
        || vl_imag.len() != nn
        || vr_real.len() != nn
        || vr_imag.len() != nn
        || vl.len() != nn
        || vr.len() != nn
        || w_imag.len() != n
    {
        return Err("arrays have wrong dimensions");
    }

    // step and increment for next conjugate pair
    let mut j = 0_usize;
    let mut dj: usize;

    // loop over columns ~ eigenvalues
    while j < n {
        // eigenvalue is complex
        if w_imag[j].abs() > 0.0 {
            if j > n - 2 {
                return Err("last eigenvalue cannot be complex");
            }
            // loop over rows
            for i in 0..n {
                let p = i + j * n;
                let q = i + (j + 1) * n;
                vl_real[p] = vl[p];
                vl_imag[p] = vl[q];
                vr_real[p] = vr[p];
                vr_imag[p] = vr[q];
                vl_real[q] = vl[p];
                vl_imag[q] = -vl[q];
                vr_real[q] = vr[p];
                vr_imag[q] = -vr[q];
            }
            dj = 2;

        // eigenvalue is real
        } else {
            // loop over rows
            for i in 0..n {
                let p = i + j * n;
                vl_real[p] = vl[p];
                vr_real[p] = vr[p];
                vl_imag[p] = 0.0;
                vr_imag[p] = 0.0;
            }
            dj = 1;
        }
        // next step
        j += dj;
    }
    Ok(())
}

/// Extracts LAPACK (dgeev) eigenvectors from its compact representation (single set)
///
/// Single set: extracts either the left eigenvectors or the right eigenvectors
///
/// # Output
///
/// * `v_real` -- pre-allocated, n*n col-major, eigenvectors; real part
/// * `v_imag` -- pre-allocated, n*n col-major, eigenvectors; imaginary part
///
/// # Input
///
/// * `w_imag` -- n, eigenvalues; imaginary part
/// * `v` -- n*n, output of dgeev
///
pub fn extract_lapack_eigenvectors_single(
    v_real: &mut [f64],
    v_imag: &mut [f64],
    w_imag: &[f64],
    v: &[f64],
) -> Result<(), &'static str> {
    // check
    let n = w_imag.len();
    let nn = n * n;
    if v_real.len() != nn || v_imag.len() != nn || v.len() != nn || w_imag.len() != n {
        return Err("arrays have wrong dimensions");
    }

    // step and increment for next conjugate pair
    let mut j = 0_usize;
    let mut dj: usize;

    // loop over columns ~ eigenvalues
    while j < n {
        // eigenvalue is complex
        if w_imag[j].abs() > 0.0 {
            if j > n - 2 {
                return Err("last eigenvalue cannot be complex");
            }
            // loop over rows
            for i in 0..n {
                let p = i + j * n;
                let q = i + (j + 1) * n;
                v_real[p] = v[p];
                v_imag[p] = v[q];
                v_real[q] = v[p];
                v_imag[q] = -v[q];
            }
            dj = 2;

        // eigenvalue is real
        } else {
            // loop over rows
            for i in 0..n {
                let p = i + j * n;
                v_real[p] = v[p];
                v_imag[p] = 0.0;
            }
            dj = 1;
        }
        // next step
        j += dj;
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn slice_to_colmajor_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let data = slice_to_colmajor(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 8.0],
        ])?;
        let correct = &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 8.0];
        assert_vec_approx_eq!(data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn slice_to_colmajor_0_works() -> Result<(), &'static str> {
        let input: &[&[f64]] = &[&[]];
        let data = slice_to_colmajor(input)?;
        assert_eq!(data.len(), 0);
        Ok(())
    }

    #[test]
    fn slice_to_colmajor_fails_on_wrong_columns() {
        #[rustfmt::skip]
        let input_data: &[&[f64]] = &[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0],
            &[7.0, 8.0, 8.0],
        ];
        assert_eq!(
            slice_to_colmajor(input_data),
            Err("all rows must have the same number of columns")
        );
    }

    #[test]
    fn extract_lapack_eigenvectors_works() -> Result<(), &'static str> {
        let n = 5_usize;
        let mut vl_real = vec![0.0; n * n];
        let mut vl_imag = vec![0.0; n * n];
        let mut vr_real = vec![0.0; n * n];
        let mut vr_imag = vec![0.0; n * n];
        let w_imag = [10.76, -10.76, 4.70, -4.70, 0.0];
        let vl = [
            0.04, 0.62, -0.04, 0.28, -0.04, 0.29, 0.00, -0.58, 0.01, 0.34, -0.13, 0.69, -0.39, -0.02, -0.40, -0.33,
            0.00, -0.07, -0.19, 0.22, 0.04, 0.56, -0.13, -0.80, 0.18,
        ];
        let vr = [
            0.11, 0.41, 0.10, 0.40, 0.54, 0.17, -0.26, -0.51, -0.09, 0.00, 0.73, -0.03, 0.19, -0.08, -0.29, 0.00,
            -0.02, -0.29, -0.08, -0.49, 0.46, 0.34, 0.31, -0.74, 0.16,
        ];
        extract_lapack_eigenvectors(
            &mut vl_real,
            &mut vl_imag,
            &mut vr_real,
            &mut vr_imag,
            &w_imag,
            &vl,
            &vr,
        )?;
        let correct_vl_real = &[
            0.04, 0.62, -0.04, 0.28, -0.04, 0.04, 0.62, -0.04, 0.28, -0.04, -0.13, 0.69, -0.39, -0.02, -0.40, -0.13,
            0.69, -0.39, -0.02, -0.40, 0.04, 0.56, -0.13, -0.80, 0.18,
        ];
        let correct_vl_imag = &[
            0.29, 0.00, -0.58, 0.01, 0.34, -0.29, 0.00, 0.58, -0.01, -0.34, -0.33, 0.00, -0.07, -0.19, 0.22, 0.33,
            0.00, 0.07, 0.19, -0.22, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let correct_vr_real = &[
            0.11, 0.41, 0.10, 0.40, 0.54, 0.11, 0.41, 0.10, 0.40, 0.54, 0.73, -0.03, 0.19, -0.08, -0.29, 0.73, -0.03,
            0.19, -0.08, -0.29, 0.46, 0.34, 0.31, -0.74, 0.16,
        ];
        let correct_vr_imag = &[
            0.17, -0.26, -0.51, -0.09, 0.00, -0.17, 0.26, 0.51, 0.09, 0.00, 0.00, -0.02, -0.29, -0.08, -0.49, 0.00,
            0.02, 0.29, 0.08, 0.49, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_vec_approx_eq!(vl_real, correct_vl_real, 1e-15);
        assert_vec_approx_eq!(vl_imag, correct_vl_imag, 1e-15);
        assert_vec_approx_eq!(vr_real, correct_vr_real, 1e-15);
        assert_vec_approx_eq!(vr_imag, correct_vr_imag, 1e-15);
        Ok(())
    }

    #[test]
    fn extract_lapack_eigenvectors_fails_on_wrong_dims() {
        let n = 2_usize;
        let wrong = 1_usize;
        let mut vl_real = vec![0.0; n * n];
        let mut vl_imag = vec![0.0; n * n];
        let mut vr_real = vec![0.0; n * n];
        let mut vr_imag = vec![0.0; n * n];
        let w_imag = vec![0.0; n];
        let vl = vec![0.0; n * n];
        let vr = vec![0.0; n * n];
        let mut vl_real_wrong = vec![0.0; n * wrong];
        let mut vl_imag_wrong = vec![0.0; n * wrong];
        let mut vr_real_wrong = vec![0.0; n * wrong];
        let mut vr_imag_wrong = vec![0.0; n * wrong];
        let w_imag_wrong = vec![0.0; wrong];
        let vl_wrong = vec![0.0; n * wrong];
        let vr_wrong = vec![0.0; n * wrong];
        assert_eq!(
            extract_lapack_eigenvectors(
                &mut vl_real_wrong,
                &mut vl_imag,
                &mut vr_real,
                &mut vr_imag,
                &w_imag,
                &vl,
                &vr,
            ),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            extract_lapack_eigenvectors(
                &mut vl_real,
                &mut vl_imag_wrong,
                &mut vr_real,
                &mut vr_imag,
                &w_imag,
                &vl,
                &vr,
            ),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            extract_lapack_eigenvectors(
                &mut vl_real,
                &mut vl_imag,
                &mut vr_real_wrong,
                &mut vr_imag,
                &w_imag,
                &vl,
                &vr,
            ),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            extract_lapack_eigenvectors(
                &mut vl_real,
                &mut vl_imag,
                &mut vr_real,
                &mut vr_imag_wrong,
                &w_imag,
                &vl,
                &vr,
            ),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            extract_lapack_eigenvectors(
                &mut vl_real,
                &mut vl_imag,
                &mut vr_real,
                &mut vr_imag,
                &w_imag_wrong,
                &vl,
                &vr,
            ),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            extract_lapack_eigenvectors(
                &mut vl_real,
                &mut vl_imag,
                &mut vr_real,
                &mut vr_imag,
                &w_imag,
                &vl_wrong,
                &vr,
            ),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            extract_lapack_eigenvectors(
                &mut vl_real,
                &mut vl_imag,
                &mut vr_real,
                &mut vr_imag,
                &w_imag,
                &vl,
                &vr_wrong,
            ),
            Err("arrays have wrong dimensions")
        );
    }

    #[test]
    fn extract_lapack_eigenvectors_single_works() -> Result<(), &'static str> {
        let n = 5_usize;
        let mut v_real = vec![0.0; n * n];
        let mut v_imag = vec![0.0; n * n];
        let w_imag = [10.76, -10.76, 4.70, -4.70, 0.0];
        let v = [
            0.04, 0.62, -0.04, 0.28, -0.04, 0.29, 0.00, -0.58, 0.01, 0.34, -0.13, 0.69, -0.39, -0.02, -0.40, -0.33,
            0.00, -0.07, -0.19, 0.22, 0.04, 0.56, -0.13, -0.80, 0.18,
        ];
        extract_lapack_eigenvectors_single(&mut v_real, &mut v_imag, &w_imag, &v)?;
        let correct_v_real = &[
            0.04, 0.62, -0.04, 0.28, -0.04, 0.04, 0.62, -0.04, 0.28, -0.04, -0.13, 0.69, -0.39, -0.02, -0.40, -0.13,
            0.69, -0.39, -0.02, -0.40, 0.04, 0.56, -0.13, -0.80, 0.18,
        ];
        let correct_v_imag = &[
            0.29, 0.00, -0.58, 0.01, 0.34, -0.29, 0.00, 0.58, -0.01, -0.34, -0.33, 0.00, -0.07, -0.19, 0.22, 0.33,
            0.00, 0.07, 0.19, -0.22, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_vec_approx_eq!(v_real, correct_v_real, 1e-15);
        assert_vec_approx_eq!(v_imag, correct_v_imag, 1e-15);
        Ok(())
    }

    #[test]
    fn extract_lapack_eigenvectors_single_fails_on_wrong_dim_1() {
        let n = 2_usize;
        let wrong = 1_usize;
        let mut v_real = vec![0.0; n * n];
        let mut v_imag = vec![0.0; n * n];
        let w_imag = vec![0.0; n];
        let v = vec![0.0; n * n];
        let mut v_real_wrong = vec![0.0; n * wrong];
        let mut v_imag_wrong = vec![0.0; n * wrong];
        let w_imag_wrong = vec![0.0; wrong];
        let v_wrong = vec![0.0; n * wrong];
        assert_eq!(
            extract_lapack_eigenvectors_single(&mut v_real_wrong, &mut v_imag, &w_imag, &v),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            extract_lapack_eigenvectors_single(&mut v_real, &mut v_imag_wrong, &w_imag, &v),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            extract_lapack_eigenvectors_single(&mut v_real, &mut v_imag, &w_imag_wrong, &v),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            extract_lapack_eigenvectors_single(&mut v_real, &mut v_imag, &w_imag, &v_wrong),
            Err("arrays have wrong dimensions")
        );
    }

    #[test]
    fn extract_lapack_eigenvectors_single_fails_on_wrong_ev() {
        let n = 2_usize;
        let mut v_real = vec![0.0; n * n];
        let mut v_imag = vec![0.0; n * n];
        const WRONG: f64 = 123.456;
        let w_imag = [0.0, WRONG];
        let v = vec![0.0; n * n];
        assert_eq!(
            extract_lapack_eigenvectors_single(&mut v_real, &mut v_imag, &w_imag, &v),
            Err("last eigenvalue cannot be complex")
        );
    }
}
