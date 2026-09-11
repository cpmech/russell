use super::Tensor2;
use russell_lab::StrError;

const BRANNON_MAX_NIT: usize = 2000;

/// Computes the polar rotation tensor R of a general tensor F
///
/// Uses the iterative fixed-point algorithm by Rebecca Brannon.
///
/// # Arguments
///
/// * `rr` -- (out) R: the rotation tensor
/// * `ff` -- (in) F: the deformation gradient
///
/// # Returns
///
/// Returns the number of iterations taken for convergence.
///
/// # Errors
///
/// Returns an error if the algorithm did not converge.
pub(crate) fn polar_rotation_brannon(rr: &mut Tensor2<9>, ff: &Tensor2<9>) -> Result<usize, StrError> {
    // This is a direct port of Brannon's `POLARROTATION` FORTRAN routine; all
    // tensor operations are written long-hand (as in the original) so that the
    // iteration path, iteration count, and result match it exactly.
    let mut f = [[0.0; 3]; 3];
    ff.to_std_matrix_slice(&mut f);

    // Step 1: E = F^T F
    let mut e11 = f[0][0] * f[0][0] + f[1][0] * f[1][0] + f[2][0] * f[2][0];
    let mut e22 = f[0][1] * f[0][1] + f[1][1] * f[1][1] + f[2][1] * f[2][1];
    let mut e33 = f[0][2] * f[0][2] + f[1][2] * f[1][2] + f[2][2] * f[2][2];
    let mut e23 = f[0][1] * f[0][2] + f[1][1] * f[1][2] + f[2][1] * f[2][2];
    let mut e31 = f[0][2] * f[0][0] + f[1][2] * f[1][0] + f[2][2] * f[2][0];
    let mut e12 = f[0][0] * f[0][1] + f[1][0] * f[1][1] + f[2][0] * f[2][1];

    // Step 2: scale F (via E) to guarantee convergence
    let mut s = 3.0 / (e11 + e22 + e33);
    e11 = 0.5 * (s * e11 - 1.0);
    e22 = 0.5 * (s * e22 - 1.0);
    e33 = 0.5 * (s * e33 - 1.0);
    e23 = 0.5 * (s * e23);
    e31 = 0.5 * (s * e31);
    e12 = 0.5 * (s * e12);

    // Step 3: first guess A = sqrt(s) F
    s = f64::sqrt(s);
    let mut a11 = s * f[0][0];
    let mut a21 = s * f[1][0];
    let mut a31 = s * f[2][0];
    let mut a12 = s * f[0][1];
    let mut a22 = s * f[1][1];
    let mut a32 = s * f[2][1];
    let mut a13 = s * f[0][2];
    let mut a23 = s * f[1][2];
    let mut a33 = s * f[2][2];

    // Step 4: initial error (Frobenius norm of E, with the off-diagonal terms counted twice)
    let mut errz = e11 * e11 + e22 * e22 + e33 * e33 + 2.0 * (e12 * e12 + e23 * e23 + e31 * e31);

    // Steps 5-9: iterate until the error stops decreasing (machine precision).
    // The cap BRANNON_MAX_NIT guards against near-singular F.
    //
    // Note: "errz + 1.0 <= 1.0" is Brannon's test for "errz is zero to machine
    // precision"; it covers the case where scaling alone produced a rotation.
    let mut knt = 0;
    let mut converged = errz + 1.0 <= 1.0;
    while !converged && knt < BRANNON_MAX_NIT {
        // Step 6: X = A(I - E)
        let x11 = a11 - (a11 * e11 + a12 * e12 + a13 * e31);
        let x21 = a21 - (a21 * e11 + a22 * e12 + a23 * e31);
        let x31 = a31 - (a31 * e11 + a32 * e12 + a33 * e31);
        let x12 = a12 - (a12 * e22 + a13 * e23 + a11 * e12);
        let x22 = a22 - (a22 * e22 + a23 * e23 + a21 * e12);
        let x32 = a32 - (a32 * e22 + a33 * e23 + a31 * e12);
        let x13 = a13 - (a13 * e33 + a11 * e31 + a12 * e23);
        let x23 = a23 - (a23 * e33 + a21 * e31 + a22 * e23);
        let x33 = a33 - (a33 * e33 + a31 * e31 + a32 * e23);
        a11 = x11;
        a21 = x21;
        a31 = x31;
        a12 = x12;
        a22 = x22;
        a32 = x32;
        a13 = x13;
        a23 = x23;
        a33 = x33;

        // Step 7: E = 1/2(A^T A - I)
        e11 = 0.5 * (a11 * a11 + a21 * a21 + a31 * a31 - 1.0);
        e22 = 0.5 * (a12 * a12 + a22 * a22 + a32 * a32 - 1.0);
        e33 = 0.5 * (a13 * a13 + a23 * a23 + a33 * a33 - 1.0);
        e23 = 0.5 * (a12 * a13 + a22 * a23 + a32 * a33);
        e31 = 0.5 * (a13 * a11 + a23 * a21 + a33 * a31);
        e12 = 0.5 * (a11 * a12 + a21 * a22 + a31 * a32);

        // Step 8: new error
        let err = e11 * e11 + e22 * e22 + e33 * e33 + 2.0 * (e12 * e12 + e23 * e23 + e31 * e31);

        knt += 1;

        // Step 9: stop if the error stopped decreasing
        if err >= errz {
            converged = true;
        } else {
            errz = err;
        }
    }

    if !converged {
        return Err("polar_rotation_brannon did not converge");
    }

    // Step 10: round the rotation to machine precision using Brannon's
    //          "(1 + A) - 1" trick and set the output
    #[rustfmt::skip]
    let r = [
        [(1.0 + a11) - 1.0, (1.0 + a12) - 1.0, (1.0 + a13) - 1.0],
        [(1.0 + a21) - 1.0, (1.0 + a22) - 1.0, (1.0 + a23) - 1.0],
        [(1.0 + a31) - 1.0, (1.0 + a32) - 1.0, (1.0 + a33) - 1.0],
    ];
    rr.set_std_matrix(&r)?;
    Ok(knt)
}

/// Computes the polar rotation tensor R of an in-plane (2D) deformation F
///
/// Uses the closed-form formula of Brannon (Eqs. 12.60a, 12.62):
/// `cos = (F11+F22)/D` and `sin = (F21-F12)/D`, with
/// `D = sqrt((F11+F22)² + (F21-F12)²)`.
///
/// # Output
///
/// * `rr` -- (out) R: the rotation tensor
///
/// # Input
///
/// * `ff` -- (in) F: the deformation gradient
///
/// # Note
///
/// `F` is assumed to be an in-plane (planar) deformation: the third axis is
/// decoupled (`R(3,3) = 1`, and `F(3,3)` is carried through to `U = Rᵀ F`).
pub(crate) fn polar_rotation_brannon2d(rr: &mut Tensor2<9>, ff: &Tensor2<9>) -> Result<(), StrError> {
    // F must be an in-plane (planar) deformation: the out-of-plane shear
    // components F13, F23, F31, F32 must be zero.
    if ff.get_std(0, 2) != 0.0 || ff.get_std(1, 2) != 0.0 || ff.get_std(2, 0) != 0.0 || ff.get_std(2, 1) != 0.0 {
        return Err("ff must be an in-plane deformation (F13 = F23 = F31 = F32 = 0)");
    }

    // Closed-form in-plane rotation
    let mut c = ff.get_std(0, 0) + ff.get_std(1, 1);
    let mut s = ff.get_std(1, 0) - ff.get_std(0, 1);
    let d = (c * c + s * s).sqrt();
    if d == 0.0 {
        return Err("ff has no unique in-plane rotation (singular)");
    }
    c /= d;
    s /= d;
    let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    rr.set_std_matrix(&r)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::polar_rotation_brannon;
    use crate::Tensor2;
    use crate::test_common::{example01, example01_rotation};
    use russell_lab::mat_approx_eq;

    #[test]
    fn polar_rotation_brannon_works() {
        // Example 01: the polar rotation is 60° about E3 (Brannon, Eq. 12.38)
        let ff = example01();
        let mut rr = Tensor2::<9>::new();
        let nit = polar_rotation_brannon(&mut rr, &ff).unwrap();
        assert!(nit > 0);
        mat_approx_eq(&rr.as_std_matrix(), &example01_rotation(), 1e-13);
    }
}
