use crate::StrError;

extern "C" {
    fn c_using_intel_mkl() -> CcBool;
    fn c_set_num_threads(n: i32);
    fn c_get_num_threads() -> i32;
}

/// Defines the vector size to decide when to use the native Rust code or BLAS
pub(crate) const MAX_DIM_FOR_NATIVE_BLAS: usize = 16;

// -------------------------------------------------------------------------------------------
// IMPORTANT: The constants below must match the corresponding C-code constants in constants.h

// Represents the type of boolean flags interchanged with the C-code
pub(crate) type CcBool = i32;

// Boolean flags
pub(crate) const C_TRUE: i32 = 1;
pub(crate) const C_FALSE: i32 = 0;

// Norm codes
pub(crate) const NORM_EUC: isize = 0;
pub(crate) const NORM_FRO: isize = 1;
pub(crate) const NORM_INF: isize = 2;
pub(crate) const NORM_MAX: isize = 3;
pub(crate) const NORM_ONE: isize = 4;

// From: /usr/include/x86_64-linux-gnu/cblas.h
// From: /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h
pub(crate) const CBLAS_COL_MAJOR: i32 = 102;
pub(crate) const CBLAS_NO_TRANS: i32 = 111;
pub(crate) const CBLAS_TRANS: i32 = 112;
// -------------------------------------------------------------------------------------------

/// Converts usize to i32
///
/// # Panics
///
/// Will panic if usize is too large to be an i32
#[inline]
pub(crate) fn to_i32(num: usize) -> i32 {
    i32::try_from(num).unwrap()
}

/// Returns whether the code was compiled with Intel MKL or not
pub fn using_intel_mkl() -> bool {
    unsafe { c_using_intel_mkl() == C_TRUE }
}

/// Sets the number of threads allowed for the BLAS routines
pub fn set_num_threads(n: usize) {
    let n_i32 = to_i32(n);
    unsafe {
        c_set_num_threads(n_i32);
    }
}

/// Gets the number of threads available to the BLAS routines
pub fn get_num_threads() -> usize {
    unsafe { c_get_num_threads() as usize }
}

/// Extracts LAPACK (dgeev) eigenvectors from its compact representation
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
pub(crate) fn dgeev_data(v_real: &mut [f64], v_imag: &mut [f64], w_imag: &[f64], v: &[f64]) -> Result<(), StrError> {
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

/// Extracts LAPACK (dgeev) eigenvectors from its compact representation (left and right)
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
pub(crate) fn dgeev_data_lr(
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{dgeev_data, dgeev_data_lr, get_num_threads, set_num_threads, using_intel_mkl};
    use crate::Matrix;
    use russell_chk::vec_approx_eq;

    #[test]
    fn using_intel_mkl_works() {
        if cfg!(use_intel_mkl) {
            assert!(using_intel_mkl());
        } else {
            assert!(!using_intel_mkl());
        }
    }

    #[test]
    fn set_num_threads_and_get_num_threads_work() {
        assert!(get_num_threads() > 2);
        set_num_threads(1);
        assert_eq!(get_num_threads(), 1);
    }

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
    fn dgeev_data_works() {
        let n = 5_usize;
        let mut v_real = vec![0.0; n * n];
        let mut v_imag = vec![0.0; n * n];
        let w_imag = [10.76, -10.76, 4.70, -4.70, 0.0];
        let v = [
            0.04, 0.62, -0.04, 0.28, -0.04, 0.29, 0.00, -0.58, 0.01, 0.34, -0.13, 0.69, -0.39, -0.02, -0.40, -0.33,
            0.00, -0.07, -0.19, 0.22, 0.04, 0.56, -0.13, -0.80, 0.18,
        ];
        dgeev_data(&mut v_real, &mut v_imag, &w_imag, &v).unwrap();
        #[rustfmt::skip]
        let correct_v_real = Matrix::from(&[
            [ 0.04,  0.04, -0.13, -0.13,  0.04],
            [ 0.62,  0.62,  0.69,  0.69,  0.56],
            [-0.04, -0.04, -0.39, -0.39, -0.13],
            [ 0.28,  0.28, -0.02, -0.02, -0.80],
            [-0.04, -0.04, -0.40, -0.40,  0.18],
        ]);
        #[rustfmt::skip]
        let correct_v_imag = Matrix::from(&[
            [ 0.29, -0.29, -0.33,  0.33,  0.00],
            [ 0.00, -0.00,  0.00, -0.00,  0.00],
            [-0.58,  0.58, -0.07,  0.07,  0.00],
            [ 0.01, -0.01, -0.19,  0.19,  0.00],
            [ 0.34, -0.34,  0.22, -0.22,  0.00],
        ]);
        vec_approx_eq(&v_real, &correct_v_real.as_data(), 1e-15);
        vec_approx_eq(&v_imag, &correct_v_imag.as_data(), 1e-15);
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
    fn dgeev_data_lr_works() {
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
        dgeev_data_lr(
            &mut vl_real,
            &mut vl_imag,
            &mut vr_real,
            &mut vr_imag,
            &w_imag,
            &vl,
            &vr,
        )
        .unwrap();
        #[rustfmt::skip]
        let correct_vl_real = Matrix::from(&[
            [ 0.04,  0.04, -0.13, -0.13,  0.04],
            [ 0.62,  0.62,  0.69,  0.69,  0.56],
            [-0.04, -0.04, -0.39, -0.39, -0.13],
            [ 0.28,  0.28, -0.02, -0.02, -0.80],
            [-0.04, -0.04, -0.40, -0.40,  0.18],
        ]);
        #[rustfmt::skip]
        let correct_vl_imag = Matrix::from(&[
            [ 0.29, -0.29, -0.33,  0.33,  0.00],
            [ 0.00, -0.00,  0.00, -0.00,  0.00],
            [-0.58,  0.58, -0.07,  0.07,  0.00],
            [ 0.01, -0.01, -0.19,  0.19,  0.00],
            [ 0.34, -0.34,  0.22, -0.22,  0.00],
        ]);
        #[rustfmt::skip]
        let correct_vr_real = Matrix::from(&[
            [0.11,  0.11,  0.73,  0.73,  0.46],
            [0.41,  0.41, -0.03, -0.03,  0.34],
            [0.10,  0.10,  0.19,  0.19,  0.31],
            [0.40,  0.40, -0.08, -0.08, -0.74],
            [0.54,  0.54, -0.29, -0.29,  0.16],
        ]);
        #[rustfmt::skip]
        let correct_vr_imag = Matrix::from(&[
            [ 0.17, -0.17,  0.00, -0.00,  0.00],
            [-0.26,  0.26, -0.02,  0.02,  0.00],
            [-0.51,  0.51, -0.29,  0.29,  0.00],
            [-0.09,  0.09, -0.08,  0.08,  0.00],
            [ 0.00, -0.00, -0.49,  0.49,  0.00],
        ]);
        vec_approx_eq(&vl_real, &correct_vl_real.as_data(), 1e-15);
        vec_approx_eq(&vl_imag, &correct_vl_imag.as_data(), 1e-15);
        vec_approx_eq(&vr_real, &correct_vr_real.as_data(), 1e-15);
        vec_approx_eq(&vr_imag, &correct_vr_imag.as_data(), 1e-15);
    }
}
