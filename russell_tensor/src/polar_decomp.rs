use super::{Tensor2, t2_gen_dot_gen_tra_chop, t2_gen_dot_sym, t2_gen_tra_dot_gen_chop, t2_gen_tra_dot_self};
use crate::Rep;
use russell_lab::StrError;

const BRANNON_MAX_NIT: usize = 2000;

/// Performs the polar decomposition F = R U = V R
///
/// Note: This is the only function where the output arguments are not the first parameters of the function.
///
/// # Arguments
///
/// * `ff` -- (in) F: the deformation gradient; must be [Rep::General]
/// * `rr` -- (out) R: the rotation tensor; must be [Rep::General]
/// * `uu` -- (out) U: the right stretch tensor; must be [Rep::Symmetric]
/// * `vv` -- (out) V: the left stretch tensor; must be [Rep::Symmetric] -- Optional
///
/// Returns the number of iterations taken for the rotation tensor to converge
///
/// # Panics
///
/// A panic will occur if the required [Rep] enums are incorrect.
pub fn polar_decomp(
    ff: &Tensor2,
    rr: &mut Tensor2,
    uu: &mut Tensor2,
    vv: Option<&mut Tensor2>,
) -> Result<usize, StrError> {
    assert_eq!(ff.rep(), Rep::General);
    assert_eq!(rr.rep(), Rep::General);
    assert_eq!(uu.rep(), Rep::Symmetric);
    let nit = polar_rotation(rr, ff)?;
    t2_gen_tra_dot_gen_chop(uu.as_mut_data(), 1.0, rr.as_data(), ff.as_data()); // U = Rᵀ F
    if let Some(v) = vv {
        assert_eq!(v.rep(), Rep::Symmetric);
        t2_gen_dot_gen_tra_chop(v.as_mut_data(), 1.0, ff.as_data(), rr.as_data()); // V = F Rᵀ
    }
    Ok(nit)
}

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
pub fn polar_rotation(rr: &mut Tensor2, ff: &Tensor2) -> Result<usize, StrError> {
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
        return Err("polar_rotation did not converge");
    }

    // Step 10: round the rotation to machine precision using Brannon's
    //           "(1 + A) - 1" trick (a no-op on IEEE-754 f64, but kept
    //           for fidelity) and copy it to the output
    for i in 0..9 {
        rr.vec[i] = (1.0 + a[i]) - 1.0;
    }
    Ok(knt)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{polar_decomp, polar_rotation};
    use crate::{Rep, Tensor2};
    use russell_lab::{Matrix, mat_approx_eq, mat_mat_mul, mat_t_mat_mul};

    #[test]
    fn polar_rotation_works() {
        // In-plane deformation gradient (Brannon Eq. 12.39, example 01).
        #[rustfmt::skip]
        let ff = Tensor2::from_std_matrix(&[
            [ 0.61784609690826542, -0.70889727457341833, 0.0],
            [ 0.59014083110323967,  0.13215390309173483, 0.0],
            [ 0.0,                  0.0,                 3.0],
        ], Rep::General).unwrap();

        let mut rr = Tensor2::new(Rep::General);
        let nit = polar_rotation(&mut rr, &ff).unwrap();
        assert!(nit > 0);

        // Rotation is 60 degrees about the E3 axis (Eq. 12.38).
        #[rustfmt::skip]
        let r_correct = &[
            [0.5,                -0.8660254037844386, 0.0],
            [0.8660254037844386,  0.5,                0.0],
            [0.0,                 0.0,                1.0],
        ];
        mat_approx_eq(&rr.as_std_matrix(), r_correct, 1e-13);

        // Orthogonality: R^T R = I
        let r = rr.as_std_matrix();
        let mut rtr = Matrix::new(3, 3);
        mat_t_mat_mul(&mut rtr, 1.0, &r, &r, 0.0).unwrap();
        mat_approx_eq(&rtr, &Matrix::diagonal(&[1.0, 1.0, 1.0]), 1e-13);
    }

    #[test]
    fn polar_decomp_works() {
        // Fully 3-D deformation gradient (continuummechanics.org example).
        #[rustfmt::skip]
        let ff = Tensor2::from_std_matrix(&[
            [ 1.000,  0.495,  0.500],
            [-0.333,  1.000, -0.247],
            [ 0.959,  0.000,  1.500],
        ], Rep::General).unwrap();

        let mut rr = Tensor2::new(Rep::General);
        let mut uu = Tensor2::new(Rep::Symmetric);
        let mut vv = Tensor2::new(Rep::Symmetric);
        let nit = polar_decomp(&ff, &mut rr, &mut uu, Some(&mut vv)).unwrap();
        assert!(nit > 0);

        let f = ff.as_std_matrix();
        let r = rr.as_std_matrix();
        let u = uu.as_std_matrix();
        let v = vv.as_std_matrix();

        // F = R U
        let mut ru = Matrix::new(3, 3);
        mat_mat_mul(&mut ru, 1.0, &r, &u, 0.0).unwrap();
        mat_approx_eq(&ru, &f, 1e-13);

        // F = V R
        let mut vr = Matrix::new(3, 3);
        mat_mat_mul(&mut vr, 1.0, &v, &r, 0.0).unwrap();
        mat_approx_eq(&vr, &f, 1e-13);

        // R^T R = I
        let mut rtr = Matrix::new(3, 3);
        mat_t_mat_mul(&mut rtr, 1.0, &r, &r, 0.0).unwrap();
        mat_approx_eq(&rtr, &Matrix::diagonal(&[1.0, 1.0, 1.0]), 1e-13);

        // Reference values (3-decimal published).
        #[rustfmt::skip]
        let r_correct = &[
            [ 0.914,  0.377, -0.148],
            [-0.374,  0.926,  0.049],
            [ 0.156,  0.011,  0.988],
        ];
        mat_approx_eq(&r, r_correct, 1e-3);
        #[rustfmt::skip]
        let u_correct = &[
            [ 1.188,  0.079,  0.783],
            [ 0.079,  1.113, -0.024],
            [ 0.783, -0.024,  1.396],
        ];
        mat_approx_eq(&u, u_correct, 1e-3);
    }
}
