use super::Matrix;
use crate::{to_i32, CcBool, StrError, Vector, C_FALSE};

extern "C" {
    // Computes the eigenvalues and eigenvectors of a general matrix
    // <https://www.netlib.org/lapack/explore-html/d9/d28/dgeev_8f.html>
    fn c_dgeev(
        calc_vl: CcBool,
        calc_vr: CcBool,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        wr: *mut f64,
        wi: *mut f64,
        vl: *mut f64,
        ldvl: *const i32,
        vr: *mut f64,
        ldvr: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

/// (dgeev) Computes the eigenvalues of a matrix
///
/// Computes the eigenvalues `l`, such that:
///
/// ```text
/// a ⋅ vj = lj ⋅ vj
/// ```
///
/// where `lj` is the component j of `l` and `vj` is the column j of the eigenvector `v`.
///
/// See also: <https://www.netlib.org/lapack/explore-html/d9/d28/dgeev_8f.html>
///
/// # Output
///
/// * `l_real` -- (m) eigenvalues; real part
/// * `l_imag` -- (m) eigenvalues; imaginary part
///
/// # Input
///
/// * `a` -- (m,m) general matrix (will be modified)
///
/// # Note
///
/// * The matrix `a` will be modified
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     // set matrix
///     let data = [[2.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 4.0, 9.0]];
///     let mut a = Matrix::from(&data);
///
///     // allocate output arrays
///     let m = a.nrow();
///     let mut l_real = Vector::new(m);
///     let mut l_imag = Vector::new(m);
///
///     // calculate the eigenvalues
///     mat_eigenvalues(&mut l_real, &mut l_imag, &mut a)?;
///
///     // check results
///     assert_eq!(
///         format!("{:.1}", l_real),
///         "┌      ┐\n\
///          │ 11.0 │\n\
///          │  1.0 │\n\
///          │  2.0 │\n\
///          └      ┘"
///     );
///     assert_eq!(
///         format!("{}", l_imag),
///         "┌   ┐\n\
///          │ 0 │\n\
///          │ 0 │\n\
///          │ 0 │\n\
///          └   ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn mat_eigenvalues(l_real: &mut Vector, l_imag: &mut Vector, a: &mut Matrix) -> Result<(), StrError> {
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if l_real.dim() != m || l_imag.dim() != m {
        return Err("vectors are incompatible");
    }
    let m_i32 = to_i32(m);
    let lda = m_i32;
    let ldu = 1;
    let ldv = 1;
    const EXTRA: i32 = 1;
    let lwork = 4 * m_i32 + EXTRA;
    let mut u = vec![0.0; ldu as usize];
    let mut v = vec![0.0; ldv as usize];
    let mut work = vec![0.0; lwork as usize];
    let mut info = 0;
    unsafe {
        c_dgeev(
            C_FALSE,
            C_FALSE,
            &m_i32,
            a.as_mut_data().as_mut_ptr(),
            &lda,
            l_real.as_mut_data().as_mut_ptr(),
            l_imag.as_mut_data().as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            v.as_mut_ptr(),
            &ldv,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    if info < 0 {
        println!("LAPACK ERROR (dgeev): Argument #{} had an illegal value", -info);
        return Err("LAPACK ERROR (dgeev): An argument had an illegal value");
    } else if info > 0 {
        println!("LAPACK ERROR (dgeev): The QR algorithm failed. Elements {}+1:N of l_real and l_imag contain eigenvalues which have converged", info-1);
        return Err("LAPACK ERROR (dgeev): The QR algorithm failed to compute all the eigenvalues, and no eigenvectors have been computed");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::mat_eigenvalues;
    use crate::{vec_approx_eq, Matrix, Vector};

    #[test]
    fn mat_eigenvalues_fails_on_non_square() {
        let mut a = Matrix::new(3, 4);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        assert_eq!(
            mat_eigenvalues(&mut l_real, &mut l_imag, &mut a),
            Err("matrix must be square")
        );
    }

    #[test]
    fn mat_eigenvalues_fails_on_wrong_dims() {
        let mut a = Matrix::new(2, 2);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        let mut l_real_wrong = Vector::new(m + 1);
        let mut l_imag_wrong = Vector::new(m + 1);
        assert_eq!(
            mat_eigenvalues(&mut l_real_wrong, &mut l_imag, &mut a),
            Err("vectors are incompatible")
        );
        assert_eq!(
            mat_eigenvalues(&mut l_real, &mut l_imag_wrong, &mut a),
            Err("vectors are incompatible")
        );
    }

    #[test]
    fn mat_eigenvalues_works() {
        #[rustfmt::skip]
        let data = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ];
        let mut a = Matrix::from(&data);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        mat_eigenvalues(&mut l_real, &mut l_imag, &mut a).unwrap();
        let s3 = f64::sqrt(3.0);
        let l_real_correct = &[-0.5, -0.5, 1.0];
        let l_imag_correct = &[s3 / 2.0, -s3 / 2.0, 0.0];
        vec_approx_eq(&l_real, l_real_correct, 1e-15);
        vec_approx_eq(&l_imag, l_imag_correct, 1e-15);
    }

    #[test]
    fn mat_eigenvalues_repeated_eval_works() {
        // rep: repeated eigenvalues
        #[rustfmt::skip]
        let data = [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],
        ];
        let mut a = Matrix::from(&data);
        let m = a.nrow();
        let mut l_real = Vector::new(m);
        let mut l_imag = Vector::new(m);
        mat_eigenvalues(&mut l_real, &mut l_imag, &mut a).unwrap();
        let l_real_correct = &[3.0, 3.0, 2.0, 2.0];
        let l_imag_correct = &[0.0, 0.0, 0.0, 0.0];
        vec_approx_eq(&l_real, l_real_correct, 1e-15);
        vec_approx_eq(&l_imag, l_imag_correct, 1e-15);
    }
}
