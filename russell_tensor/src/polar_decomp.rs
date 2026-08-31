use crate::polar_brannon::{polar_rotation_brannon, polar_rotation_brannon2d};
use crate::polar_classic::{polar_decomp_eigen, polar_decomp_svd};
use crate::polar_higham::polar_quaternion_higham;
use crate::{Rep, Tensor2, t2_gen_dot_gen_tra_chop, t2_gen_tra_dot_gen_chop};
use russell_lab::StrError;

/// Specifies the polar decomposition algorithm
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolarAlgo {
    /// Classic: Using eigenvalues/eigenvectors
    Eigen,

    /// Classic: Using singular-value-decomposition (SVD)
    SVD,

    /// Rebecca Brannon's iterative fixed-point algorithm (3×3)
    Brannon,

    /// Brannon's closed-form algorithm for 2×2 (in-plane) matrices
    Brannon2d,

    /// Higham & Noferini (2016) quaternion-based direct algorithm (3×3)
    Higham,
}

/// Performs the polar decomposition F = R U = V R
///
/// # Output
///
/// * `rr` -- (out) R: the rotation tensor; must be [Rep::General]
/// * `uu` -- (out) U: the right stretch tensor; must be [Rep::Symmetric]
/// * `vv` -- (out) V: the left stretch tensor; must be [Rep::Symmetric] -- Optional
///
/// # Input
///
/// * `algo` -- the algorithm to use
/// * `ff` -- (in) F: the deformation gradient; must be [Rep::General]
///
/// # Returns
///
/// Returns the number of iterations taken for the rotation tensor to converge.
/// This is always zero for the non-iterative algorithms (`Higham`, `Brannon2d`).
///
/// # Errors
///
/// Returns an error if the required [Rep] enums are incorrect.
pub fn polar_decomp(
    rr: &mut Tensor2,
    uu: &mut Tensor2,
    vv: Option<&mut Tensor2>,
    algo: PolarAlgo,
    ff: &Tensor2,
) -> Result<usize, StrError> {
    if ff.rep() != Rep::General {
        return Err("ff must be Rep::General");
    }
    if rr.rep() != Rep::General {
        return Err("rr must be Rep::General");
    }
    if uu.rep() != Rep::Symmetric {
        return Err("uu must be Rep::Symmetric");
    }
    if let Some(v) = vv.as_deref() {
        if v.rep() != Rep::Symmetric {
            return Err("vv must be Rep::Symmetric");
        }
    }

    // Polar rotation R and right stretch U
    let nit = match algo {
        PolarAlgo::Eigen => {
            polar_decomp_eigen(rr, uu, ff)?; // classic: eigenvalues of C = Fᵀ F
            0
        }
        PolarAlgo::SVD => {
            polar_decomp_svd(rr, uu, ff)?; // classic: singular value decomposition
            0
        }
        PolarAlgo::Brannon => {
            let nit = polar_rotation_brannon(rr, ff)?;
            t2_gen_tra_dot_gen_chop(uu.as_mut_data(), 1.0, rr.as_data(), ff.as_data()); // U = Rᵀ F
            nit
        }
        PolarAlgo::Brannon2d => {
            polar_rotation_brannon2d(rr, ff)?;
            t2_gen_tra_dot_gen_chop(uu.as_mut_data(), 1.0, rr.as_data(), ff.as_data()); // U = Rᵀ F
            0
        }
        PolarAlgo::Higham => {
            polar_quaternion_higham(rr, uu, ff)?; // R = Q, U = H
            0
        }
    };

    // Left stretch V = F Rᵀ (common to all algorithms)
    if let Some(v) = vv {
        t2_gen_dot_gen_tra_chop(v.as_mut_data(), 1.0, ff.as_data(), rr.as_data());
    }

    Ok(nit)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{PolarAlgo, polar_decomp};
    use crate::test_common::{
        case51, case52, check_agree, check_polar, example01, example01_rotation, example01_stretch, example03,
        example03_rotation, example03_stretch,
    };
    use crate::{Rep, Tensor2};
    use russell_lab::{Matrix, mat_approx_eq, mat_mat_mul};

    #[test]
    fn polar_decomp_brannon_works() {
        // Example 03: fully 3-D deformation gradient (McGinty)
        let ff = example03();
        let mut rr = Tensor2::new(Rep::General);
        let mut uu = Tensor2::new(Rep::Symmetric);
        let mut vv = Tensor2::new(Rep::Symmetric);
        let nit = polar_decomp(&mut rr, &mut uu, Some(&mut vv), PolarAlgo::Brannon, &ff).unwrap();
        assert!(nit > 0);

        // F = R U and Q orthogonal
        check_polar(&ff, &rr, &uu, 1e-13);

        // F = V R (left stretch, specific to Brannon's decomposition)
        let f = ff.as_std_matrix();
        let r = rr.as_std_matrix();
        let v = vv.as_std_matrix();
        let mut vr = Matrix::new(3, 3);
        mat_mat_mul(&mut vr, 1.0, &v, &r, 0.0).unwrap();
        mat_approx_eq(&vr, &f, 1e-13);

        // Reference values (3-decimal published)
        mat_approx_eq(&r, &example03_rotation(), 1e-3);
        mat_approx_eq(&uu.as_std_matrix(), &example03_stretch(), 1e-3);
    }

    #[test]
    fn polar_decomp_brannon_on_higham_cases() {
        // Higham & Noferini test (5.1), cross-checked against their algorithm
        check_agree(&case51());

        // Higham & Noferini test (5.2) over a range of condition numbers
        for y in [1.0f64, 1e-2, 1e-4, 1e-6, 1e-8] {
            let a = case52(y);
            let mut rr = Tensor2::new(Rep::General);
            let mut uu = Tensor2::new(Rep::Symmetric);
            let mut vv = Tensor2::new(Rep::Symmetric);
            polar_decomp(&mut rr, &mut uu, Some(&mut vv), PolarAlgo::Brannon, &a).unwrap();
            // Brannon's algorithm is only accurate to ~1e-8 for very
            // ill-conditioned F (kappa ~ 1/y), so loosen the tolerance there.
            let tol = if y == 1.0 { 1e-13 } else { 1e-8 };
            check_polar(&a, &rr, &uu, tol);
            if y == 1.0 {
                check_agree(&a);
            }
        }
    }

    #[test]
    fn polar_decomp_higham_algo_works() {
        // Higham & Noferini test (5.1), via the dispatcher
        let a = case51();
        let mut rr = Tensor2::new(Rep::General);
        let mut uu = Tensor2::new(Rep::Symmetric);
        let nit = polar_decomp(&mut rr, &mut uu, None, PolarAlgo::Higham, &a).unwrap();
        assert_eq!(nit, 0); // Higham is non-iterative
        check_polar(&a, &rr, &uu, 1e-13);
    }

    #[test]
    fn polar_decomp_brannon2d_works() {
        // Example 01 is in-plane; the closed-form 2×2 rotation must match the 3×3 one
        let ff = example01();
        let mut rr = Tensor2::new(Rep::General);
        let mut uu = Tensor2::new(Rep::Symmetric);
        let mut vv = Tensor2::new(Rep::Symmetric);
        let nit = polar_decomp(&mut rr, &mut uu, Some(&mut vv), PolarAlgo::Brannon2d, &ff).unwrap();
        assert_eq!(nit, 0); // Brannon2d is non-iterative
        check_polar(&ff, &rr, &uu, 1e-13);
        mat_approx_eq(&rr.as_std_matrix(), &example01_rotation(), 1e-13);
        mat_approx_eq(&uu.as_std_matrix(), &example01_stretch(), 1e-13);
    }

    #[test]
    fn polar_decomp_brannon2d_rejects_non_planar() {
        // Example 03 is fully 3-D (non-zero out-of-plane shear), so Brannon2d must fail
        let ff = example03();
        let mut rr = Tensor2::new(Rep::General);
        let mut uu = Tensor2::new(Rep::Symmetric);
        let res = polar_decomp(&mut rr, &mut uu, None, PolarAlgo::Brannon2d, &ff);
        assert!(res.is_err());
    }

    #[test]
    fn polar_decomp_brannon2d_rejects_singular() {
        // In-plane reflection F = diag(1, -1, 1): F11+F22 = 0 and F21-F12 = 0,
        // so the in-plane rotation is undefined (d = 0)
        let ff = Tensor2::from_std_matrix(&[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], Rep::General).unwrap();
        let mut rr = Tensor2::new(Rep::General);
        let mut uu = Tensor2::new(Rep::Symmetric);
        let res = polar_decomp(&mut rr, &mut uu, None, PolarAlgo::Brannon2d, &ff);
        assert!(res.is_err());
    }

    #[test]
    fn polar_decomp_eigen_works() {
        // Example 03: fully 3-D deformation gradient (McGinty)
        let ff = example03();
        let mut rr = Tensor2::new(Rep::General);
        let mut uu = Tensor2::new(Rep::Symmetric);
        let mut vv = Tensor2::new(Rep::Symmetric);
        let nit = polar_decomp(&mut rr, &mut uu, Some(&mut vv), PolarAlgo::Eigen, &ff).unwrap();
        assert_eq!(nit, 0); // Eigen is non-iterative

        // F = R U and R orthogonal
        check_polar(&ff, &rr, &uu, 1e-13);

        // F = V R (left stretch)
        let f = ff.as_std_matrix();
        let r = rr.as_std_matrix();
        let v = vv.as_std_matrix();
        let mut vr = Matrix::new(3, 3);
        mat_mat_mul(&mut vr, 1.0, &v, &r, 0.0).unwrap();
        mat_approx_eq(&vr, &f, 1e-13);

        // Reference values (3-decimal published)
        mat_approx_eq(&r, &example03_rotation(), 1e-3);
        mat_approx_eq(&uu.as_std_matrix(), &example03_stretch(), 1e-3);
    }

    #[test]
    fn polar_decomp_svd_works() {
        // Example 03: fully 3-D deformation gradient (McGinty)
        let ff = example03();
        let mut rr = Tensor2::new(Rep::General);
        let mut uu = Tensor2::new(Rep::Symmetric);
        let mut vv = Tensor2::new(Rep::Symmetric);
        let nit = polar_decomp(&mut rr, &mut uu, Some(&mut vv), PolarAlgo::SVD, &ff).unwrap();
        assert_eq!(nit, 0); // SVD is non-iterative

        // F = R U and R orthogonal
        check_polar(&ff, &rr, &uu, 1e-13);

        // F = V R (left stretch)
        let f = ff.as_std_matrix();
        let r = rr.as_std_matrix();
        let v = vv.as_std_matrix();
        let mut vr = Matrix::new(3, 3);
        mat_mat_mul(&mut vr, 1.0, &v, &r, 0.0).unwrap();
        mat_approx_eq(&vr, &f, 1e-13);

        // Reference values (3-decimal published)
        mat_approx_eq(&r, &example03_rotation(), 1e-3);
        mat_approx_eq(&uu.as_std_matrix(), &example03_stretch(), 1e-3);
    }

    #[test]
    fn polar_decomp_eigen_on_higham_cases() {
        // Higham & Noferini test (5.1), cross-checked against Higham's algorithm
        let a = case51();
        let mut r_e = Tensor2::new(Rep::General);
        let mut u_e = Tensor2::new(Rep::Symmetric);
        polar_decomp(&mut r_e, &mut u_e, None, PolarAlgo::Eigen, &a).unwrap();
        check_polar(&a, &r_e, &u_e, 1e-13);
        let mut r_h = Tensor2::new(Rep::General);
        let mut u_h = Tensor2::new(Rep::Symmetric);
        polar_decomp(&mut r_h, &mut u_h, None, PolarAlgo::Higham, &a).unwrap();
        mat_approx_eq(&r_e.as_std_matrix(), &r_h.as_std_matrix(), 1e-13);
        mat_approx_eq(&u_e.as_std_matrix(), &u_h.as_std_matrix(), 1e-13);

        // Higham & Noferini test (5.2), well-conditioned case. Note: the eigen
        // approach squares the condition number of F (via C = Fᵀ F), so it is
        // only reliable for well-conditioned F.
        let a = case52(1.0);
        let mut r_e = Tensor2::new(Rep::General);
        let mut u_e = Tensor2::new(Rep::Symmetric);
        polar_decomp(&mut r_e, &mut u_e, None, PolarAlgo::Eigen, &a).unwrap();
        check_polar(&a, &r_e, &u_e, 1e-13);
    }

    #[test]
    fn polar_decomp_svd_on_higham_cases() {
        // Higham & Noferini test (5.1), cross-checked against Higham's algorithm
        let a = case51();
        let mut r_s = Tensor2::new(Rep::General);
        let mut u_s = Tensor2::new(Rep::Symmetric);
        polar_decomp(&mut r_s, &mut u_s, None, PolarAlgo::SVD, &a).unwrap();
        check_polar(&a, &r_s, &u_s, 1e-13);
        let mut r_h = Tensor2::new(Rep::General);
        let mut u_h = Tensor2::new(Rep::Symmetric);
        polar_decomp(&mut r_h, &mut u_h, None, PolarAlgo::Higham, &a).unwrap();
        mat_approx_eq(&r_s.as_std_matrix(), &r_h.as_std_matrix(), 1e-13);
        mat_approx_eq(&u_s.as_std_matrix(), &u_h.as_std_matrix(), 1e-13);

        // Higham & Noferini test (5.2) over a range of condition numbers
        for y in [1.0f64, 1e-2, 1e-4, 1e-6, 1e-8] {
            let a = case52(y);
            let tol = if y == 1.0 { 1e-13 } else { 1e-8 };
            let mut r_s = Tensor2::new(Rep::General);
            let mut u_s = Tensor2::new(Rep::Symmetric);
            polar_decomp(&mut r_s, &mut u_s, None, PolarAlgo::SVD, &a).unwrap();
            check_polar(&a, &r_s, &u_s, tol);
        }
    }
}
