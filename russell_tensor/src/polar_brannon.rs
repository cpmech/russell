use super::{t2_gen_dot_sym, t2_gen_tra_dot_self, Tensor2};
use crate::Rep;
use russell_lab::StrError;

const BRANNON_MAX_NIT: usize = 2000;

/// Computes the polar rotation tensor R of a general tensor F
///
/// Uses the iterative fixed-point algorithm by Rebecca Brannon.
///
/// # Arguments
///
/// * `rr` -- (out) R: the rotation tensor; must be [Rep::General]
/// * `ff` -- (in) F: the deformation gradient; must be [Rep::General]
///
/// # Returns
///
/// Returns the number of iterations taken for convergence.
///
/// # Panics
///
/// A panic will occur if the required [Rep] enums are incorrect.
pub fn polar_rotation_brannon(rr: &mut Tensor2, ff: &Tensor2) -> Result<usize, StrError> {
    assert_eq!(ff.rep(), Rep::General);
    assert_eq!(rr.rep(), Rep::General);

    // e and i_vec_minus_e are symmetric (Kelvin-Mandel 6-component), matching the
    // Fortran scalars E11, E22, E33, E23, E31, E12; a and x are general (9).
    let mut e = [0.0; 6];
    let mut a = [0.0; 9];
    let mut x = [0.0; 9];
    let mut i_vec_minus_e = [0.0; 6];

    // Step 1: E = F^T F
    t2_gen_tra_dot_self(&mut e, 1.0, ff.as_data());

    // Step 2: Scale F to guarantee convergence
    let mut s = 3.0 / (e[0] + e[1] + e[2]);
    for i in 0..6 {
        if i < 3 {
            e[i] = 0.5 * (s * e[i] - 1.0);
        } else {
            e[i] = 0.5 * (s * e[i]);
        }
    }

    // Step 3: First guess A = sqrt(s) F
    s = f64::sqrt(s);
    for i in 0..9 {
        a[i] = s * ff.get(i);
    }

    // Step 4: Initial error using vector dot product
    let mut errz = 0.0;
    for i in 0..6 {
        errz += e[i] * e[i];
    }

    // Steps 5-9: iterate until the error stops decreasing (machine
    //             precision). The cap BRANNON_MAX_NIT guards against
    //             near-singular F.
    //
    //             Note: "errz + 1.0 <= 1.0" is Brannon's test for
    //             "errz is zero to machine precision"; it covers the
    //             case where scaling alone already produced a rotation.
    let mut knt = 0;
    let mut converged = errz + 1.0 <= 1.0;
    while !converged && knt < BRANNON_MAX_NIT {
        // Step 6: X = A(I - E)
        for i in 0..6 {
            if i < 3 {
                i_vec_minus_e[i] = 1.0 - e[i];
            } else {
                i_vec_minus_e[i] = -e[i];
            }
        }
        t2_gen_dot_sym(&mut x, 1.0, &a, &i_vec_minus_e);
        a.copy_from_slice(&x);

        // Step 7: E = 1/2(A^T A - I)
        t2_gen_tra_dot_self(&mut e, 1.0, &a);
        for i in 0..6 {
            if i < 3 {
                e[i] = 0.5 * (e[i] - 1.0);
            } else {
                e[i] = 0.5 * e[i];
            }
        }

        // Step 8: New error
        let mut err = 0.0;
        for i in 0..6 {
            err += e[i] * e[i];
        }

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
    //           "(1 + A) - 1" trick (a no-op on IEEE-754 f64, but kept
    //           for fidelity) and copy it to the output
    for i in 0..9 {
        rr.vec[i] = (1.0 + a[i]) - 1.0;
    }
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
/// * `rr` -- (out) R: the rotation tensor; must be [Rep::General]
///
/// # Input
///
/// * `ff` -- (in) F: the deformation gradient; must be [Rep::General]
///
/// # Errors
///
/// Returns an error if the required [Rep] enums are incorrect.
///
/// # Note
///
/// `F` is assumed to be an in-plane (planar) deformation: the third axis is
/// decoupled (`R(3,3) = 1`, and `F(3,3)` is carried through to `U = Rᵀ F`).
pub fn polar_rotation_brannon2d(rr: &mut Tensor2, ff: &Tensor2) -> Result<(), StrError> {
    if ff.rep() != Rep::General {
        return Err("ff must be Rep::General");
    }
    if rr.rep() != Rep::General {
        return Err("rr must be Rep::General");
    }

    // Closed-form in-plane rotation
    let mut c = ff.get_std(0, 0) + ff.get_std(1, 1);
    let mut s = ff.get_std(1, 0) - ff.get_std(0, 1);
    let d = (c * c + s * s).sqrt();
    c /= d;
    s /= d;
    let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    rr.set_std_matrix(&r)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::polar_rotation_brannon;
    use crate::test_common::{example01, example01_rotation};
    use crate::{Rep, Tensor2};
    use russell_lab::mat_approx_eq;

    #[test]
    fn polar_rotation_brannon_works() {
        // Example 01: the polar rotation is 60° about E3 (Brannon, Eq. 12.38)
        let ff = example01();
        let mut rr = Tensor2::new(Rep::General);
        let nit = polar_rotation_brannon(&mut rr, &ff).unwrap();
        assert!(nit > 0);
        mat_approx_eq(&rr.as_std_matrix(), &example01_rotation(), 1e-13);
    }
}
