use crate::StrError;

/// Extracts LAPACK (dgeev) eigenvectors from its compact representation
///
/// Single set: extracts either the left eigenvectors or the right eigenvectors
///
/// # Output
///
/// * `v_real` -- pre-allocated, n*n row-major, eigenvectors; real part
/// * `v_imag` -- pre-allocated, n*n row-major, eigenvectors; imaginary part
///
/// # Input
///
/// * `w_imag` -- n, eigenvalues; imaginary part
/// * `v` -- n*n, output of dgeev
///
pub fn dgeev_data(v_real: &mut [f64], v_imag: &mut [f64], w_imag: &[f64], v: &[f64]) -> Result<(), StrError> {
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
                let p = i * n + j;
                let q = i * n + (j + 1);
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
                let p = i * n + j;
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

/// Extracts LAPACK (dgeev) eigenvectors from its compact representation (left and right)
///
/// # Output
///
/// * `vl_real` -- pre-allocated, n*n row-major, **left** eigenvectors; real part
/// * `vl_imag` -- pre-allocated, n*n row-major, **left** eigenvectors; imaginary part
/// * `vr_real` -- pre-allocated, n*n row-major, **right** eigenvectors; real part
/// * `vr_imag` -- pre-allocated, n*n row-major, **right** eigenvectors; imaginary part
///
/// # Input
///
/// * `w_imag` -- n, eigenvalues; imaginary part
/// * `vl` -- n*n, output of dgeev
/// * `vr` -- n*n, output of dgeev
///
pub fn dgeev_data_lr(
    vl_real: &mut [f64],
    vl_imag: &mut [f64],
    vr_real: &mut [f64],
    vr_imag: &mut [f64],
    w_imag: &[f64],
    vl: &[f64],
    vr: &[f64],
) -> Result<(), StrError> {
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
                let p = i * n + j;
                let q = i * n + (j + 1);
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
                let p = i * n + j;
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{dgeev_data, dgeev_data_lr};
    use crate::StrError;
    use russell_chk::assert_vec_approx_eq;

    #[test]
    fn dgeev_data_fails_on_wrong_dims() {
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
            dgeev_data(&mut v_real_wrong, &mut v_imag, &w_imag, &v),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            dgeev_data(&mut v_real, &mut v_imag_wrong, &w_imag, &v),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            dgeev_data(&mut v_real, &mut v_imag, &w_imag_wrong, &v),
            Err("arrays have wrong dimensions")
        );
        assert_eq!(
            dgeev_data(&mut v_real, &mut v_imag, &w_imag, &v_wrong),
            Err("arrays have wrong dimensions")
        );
    }

    #[test]
    fn dgeev_data_fails_on_wrong_ev() {
        let n = 2_usize;
        let mut v_real = vec![0.0; n * n];
        let mut v_imag = vec![0.0; n * n];
        const WRONG: f64 = 123.456;
        let w_imag = [0.0, WRONG];
        let v = vec![0.0; n * n];
        assert_eq!(
            dgeev_data(&mut v_real, &mut v_imag, &w_imag, &v),
            Err("last eigenvalue cannot be complex")
        );
    }

    #[test]
    fn dgeev_data_works() -> Result<(), StrError> {
        let n = 5_usize;
        let mut v_real = vec![0.0; n * n];
        let mut v_imag = vec![0.0; n * n];
        let w_imag = [10.76, -10.76, 4.70, -4.70, 0.0];
        let v = [
            0.04, 0.29, -0.13, -0.33, 0.04, 0.62, 0.00, 0.69, 0.00, 0.56, -0.04, -0.58, -0.39, -0.07, -0.13, 0.28,
            0.01, -0.02, -0.19, -0.80, -0.04, 0.34, -0.40, 0.22, 0.18,
        ];
        dgeev_data(&mut v_real, &mut v_imag, &w_imag, &v)?;
        #[rustfmt::skip]
        let correct_v_real = &[
             0.04,  0.04, -0.13, -0.13,  0.04,
             0.62,  0.62,  0.69,  0.69,  0.56,
            -0.04, -0.04, -0.39, -0.39, -0.13,
             0.28,  0.28, -0.02, -0.02, -0.80,
            -0.04, -0.04, -0.40, -0.40,  0.18,
        ];
        #[rustfmt::skip]
        let correct_v_imag = &[
             0.29, -0.29, -0.33,  0.33,  0.00,
             0.00, -0.00,  0.00, -0.00,  0.00,
            -0.58,  0.58, -0.07,  0.07,  0.00,
             0.01, -0.01, -0.19,  0.19,  0.00,
             0.34, -0.34,  0.22, -0.22,  0.00,
        ];
        assert_vec_approx_eq!(v_real, correct_v_real, 1e-15);
        assert_vec_approx_eq!(v_imag, correct_v_imag, 1e-15);
        Ok(())
    }

    #[test]
    fn dgeev_data_lr_fails_on_wrong_dims() {
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
            dgeev_data_lr(
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
            dgeev_data_lr(
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
            dgeev_data_lr(
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
            dgeev_data_lr(
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
            dgeev_data_lr(
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
            dgeev_data_lr(
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
            dgeev_data_lr(
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
    fn dgeev_data_lr_fails_on_wrong_ev() {
        let n = 2_usize;
        let mut vl_real = vec![0.0; n * n];
        let mut vl_imag = vec![0.0; n * n];
        let mut vr_real = vec![0.0; n * n];
        let mut vr_imag = vec![0.0; n * n];
        const WRONG: f64 = 123.456;
        let w_imag = [0.0, WRONG];
        let vl = vec![0.0; n * n];
        let vr = vec![0.0; n * n];
        assert_eq!(
            dgeev_data_lr(
                &mut vl_real,
                &mut vl_imag,
                &mut vr_real,
                &mut vr_imag,
                &w_imag,
                &vl,
                &vr,
            ),
            Err("last eigenvalue cannot be complex")
        );
    }

    #[test]
    fn dgeev_data_lr_works() -> Result<(), StrError> {
        let n = 5_usize;
        let mut vl_real = vec![0.0; n * n];
        let mut vl_imag = vec![0.0; n * n];
        let mut vr_real = vec![0.0; n * n];
        let mut vr_imag = vec![0.0; n * n];
        let w_imag = [10.76, -10.76, 4.70, -4.70, 0.0];
        let vl = [
            0.04, 0.29, -0.13, -0.33, 0.04, 0.62, 0.00, 0.69, 0.00, 0.56, -0.04, -0.58, -0.39, -0.07, -0.13, 0.28,
            0.01, -0.02, -0.19, -0.80, -0.04, 0.34, -0.40, 0.22, 0.18,
        ];
        let vr = [
            0.11, 0.17, 0.73, 0.00, 0.46, 0.41, -0.26, -0.03, -0.02, 0.34, 0.10, -0.51, 0.19, -0.29, 0.31, 0.40, -0.09,
            -0.08, -0.08, -0.74, 0.54, 0.00, -0.29, -0.49, 0.16,
        ];
        dgeev_data_lr(
            &mut vl_real,
            &mut vl_imag,
            &mut vr_real,
            &mut vr_imag,
            &w_imag,
            &vl,
            &vr,
        )?;
        #[rustfmt::skip]
        let correct_vl_real = &[
             0.04,  0.04, -0.13, -0.13,  0.04,
             0.62,  0.62,  0.69,  0.69,  0.56,
            -0.04, -0.04, -0.39, -0.39, -0.13,
             0.28,  0.28, -0.02, -0.02, -0.80,
            -0.04, -0.04, -0.40, -0.40,  0.18,
        ];
        #[rustfmt::skip]
        let correct_vl_imag = &[
             0.29, -0.29, -0.33,  0.33,  0.00,
             0.00, -0.00,  0.00, -0.00,  0.00,
            -0.58,  0.58, -0.07,  0.07,  0.00,
             0.01, -0.01, -0.19,  0.19,  0.00,
             0.34, -0.34,  0.22, -0.22,  0.00,
        ];
        #[rustfmt::skip]
        let correct_vr_real = &[
             0.11,  0.11,  0.73,  0.73,  0.46,
             0.41,  0.41, -0.03, -0.03,  0.34,
             0.10,  0.10,  0.19,  0.19,  0.31,
             0.40,  0.40, -0.08, -0.08, -0.74,
             0.54,  0.54, -0.29, -0.29,  0.16,
        ];
        #[rustfmt::skip]
        let correct_vr_imag = &[
             0.17, -0.17,  0.00, -0.00,  0.00,
            -0.26,  0.26, -0.02,  0.02,  0.00,
            -0.51,  0.51, -0.29,  0.29,  0.00,
            -0.09,  0.09, -0.08,  0.08,  0.00,
             0.00, -0.00, -0.49,  0.49,  0.00,
        ];
        assert_vec_approx_eq!(vl_real, correct_vl_real, 1e-15);
        assert_vec_approx_eq!(vl_imag, correct_vl_imag, 1e-15);
        assert_vec_approx_eq!(vr_real, correct_vr_real, 1e-15);
        assert_vec_approx_eq!(vr_imag, correct_vr_imag, 1e-15);
        Ok(())
    }
}
