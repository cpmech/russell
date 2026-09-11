use crate::polar_brannon::polar_rotation_brannon;
use crate::polar_classic::{polar_decomp_eigen, polar_decomp_svd};
use crate::polar_higham::polar_quaternion_higham;
use crate::{Tensor2, t2_gen_dot_gen_tra_chop, t2_gen_tra_dot_gen_chop};
use russell_lab::StrError;

/// Specifies the polar decomposition algorithm
///
/// # References
///
/// 1. Brannon R. M. (2018) Rotation, Reflection, and Frame Changes. IOP Publishing.
///    <https://doi.org/10.1088/978-0-7503-1454-1>
/// 2. Higham N. J. and Noferini V. (2016) An algorithm to compute the polar decomposition of a
///    3×3 matrix. Numerical Algorithms, 73:349-369. <https://doi.org/10.1007/s11075-016-0098-7>
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolarAlgo {
    /// Classic: Using eigenvalues/eigenvectors
    ///
    /// Uses [crate::Spectral2] analytical method.
    Eigen,

    /// Classic: Using singular-value-decomposition (SVD)
    ///
    /// Calls LAPACK dgesvd and may not be the fastest for these 3x3 problems.
    SVD,

    /// Brannon's iterative fixed-point algorithm (3×3)
    ///
    /// * Brannon R. M. (2018) Rotation, Reflection, and Frame Changes. IOP Publishing.
    ///   <https://doi.org/10.1088/978-0-7503-1454-1>
    Iterative,

    /// Higham & Noferini (2016) quaternion-based direct algorithm (3×3)
    ///
    /// * Higham N. J. and Noferini V. (2016) An algorithm to compute the polar decomposition of a
    ///   3×3 matrix. Numerical Algorithms, 73:349-369. <https://doi.org/10.1007/s11075-016-0098-7>
    Quaternion,
}

/// Performs the polar decomposition F = R U = V R (using the default method)
///
/// # Output
///
/// * `rr` -- (out) R: the rotation tensor
/// * `uu` -- (out) U: the right stretch tensor
/// * `vv` -- (out) V: the left stretch tensor
///
/// # Input
///
/// * `ff` -- (in) F: the deformation gradient
///
/// # Returns
///
/// Returns the number of iterations taken for the rotation tensor to converge.
/// This is always zero for the non-iterative algorithm ([PolarAlgo::Quaternion]).
///
/// Default method: [PolarAlgo::Quaternion]
#[inline]
pub fn polar_decomp(
    rr: &mut Tensor2<9>,
    uu: &mut Tensor2<6>,
    vv: Option<&mut Tensor2<6>>,
    ff: &Tensor2<9>,
) -> Result<usize, StrError> {
    polar_decomp_mx(rr, uu, vv, PolarAlgo::Quaternion, ff)
}

/// Performs the polar decomposition F = R U = V R (selectable method version)
///
/// # Output
///
/// * `rr` -- (out) R: the rotation tensor
/// * `uu` -- (out) U: the right stretch tensor
/// * `vv` -- (out) V: the left stretch tensor
///
/// # Input
///
/// * `algo` -- the algorithm to use
/// * `ff` -- (in) F: the deformation gradient
///
/// # Returns
///
/// Returns the number of iterations taken for the rotation tensor to converge.
/// This is always zero for the non-iterative algorithm ([PolarAlgo::Quaternion]).
pub fn polar_decomp_mx(
    rr: &mut Tensor2<9>,
    uu: &mut Tensor2<6>,
    vv: Option<&mut Tensor2<6>>,
    algo: PolarAlgo,
    ff: &Tensor2<9>,
) -> Result<usize, StrError> {
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
        PolarAlgo::Iterative => {
            let nit = polar_rotation_brannon(rr, ff)?;
            t2_gen_tra_dot_gen_chop(uu.as_mut_data(), 1.0, rr.as_data(), ff.as_data()); // U = Rᵀ F
            nit
        }
        PolarAlgo::Quaternion => {
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
    use super::{PolarAlgo, polar_decomp, polar_decomp_mx};
    use crate::Tensor2;
    use crate::test_common::{
        case51, case52, check_agree, check_polar, example03, example03_rotation, example03_stretch,
    };
    use russell_lab::{Matrix, mat_approx_eq, mat_mat_mul};

    #[test]
    fn polar_decomp_default_works() {
        let ff = example03();
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        let mut vv = Tensor2::<6>::new();
        let _ = polar_decomp(&mut rr, &mut uu, Some(&mut vv), &ff).unwrap();
        check_polar(&ff, &rr, &uu, 1e-13);
    }

    #[test]
    fn polar_decomp_brannon_works() {
        // Example 03: fully 3-D deformation gradient (McGinty)
        let ff = example03();
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        let mut vv = Tensor2::<6>::new();
        let nit = polar_decomp_mx(&mut rr, &mut uu, Some(&mut vv), PolarAlgo::Iterative, &ff).unwrap();
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
            let mut rr = Tensor2::<9>::new();
            let mut uu = Tensor2::<6>::new();
            let mut vv = Tensor2::<6>::new();
            polar_decomp_mx(&mut rr, &mut uu, Some(&mut vv), PolarAlgo::Iterative, &a).unwrap();
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
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        let nit = polar_decomp_mx(&mut rr, &mut uu, None, PolarAlgo::Quaternion, &a).unwrap();
        assert_eq!(nit, 0); // Higham is non-iterative
        check_polar(&a, &rr, &uu, 1e-13);
    }

    #[test]
    fn polar_decomp_eigen_works() {
        // Example 03: fully 3-D deformation gradient (McGinty)
        let ff = example03();
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        let mut vv = Tensor2::<6>::new();
        let nit = polar_decomp_mx(&mut rr, &mut uu, Some(&mut vv), PolarAlgo::Eigen, &ff).unwrap();
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
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        let mut vv = Tensor2::<6>::new();
        let nit = polar_decomp_mx(&mut rr, &mut uu, Some(&mut vv), PolarAlgo::SVD, &ff).unwrap();
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
        let mut r_e = Tensor2::<9>::new();
        let mut u_e = Tensor2::<6>::new();
        polar_decomp_mx(&mut r_e, &mut u_e, None, PolarAlgo::Eigen, &a).unwrap();
        check_polar(&a, &r_e, &u_e, 1e-13);
        let mut r_h = Tensor2::<9>::new();
        let mut u_h = Tensor2::<6>::new();
        polar_decomp_mx(&mut r_h, &mut u_h, None, PolarAlgo::Quaternion, &a).unwrap();
        mat_approx_eq(&r_e.as_std_matrix(), &r_h.as_std_matrix(), 1e-13);
        mat_approx_eq(&u_e.as_std_matrix(), &u_h.as_std_matrix(), 1e-13);

        // Higham & Noferini test (5.2), well-conditioned case. Note: the eigen
        // approach squares the condition number of F (via C = Fᵀ F), so it is
        // only reliable for well-conditioned F.
        let a = case52(1.0);
        let mut r_e = Tensor2::<9>::new();
        let mut u_e = Tensor2::<6>::new();
        polar_decomp_mx(&mut r_e, &mut u_e, None, PolarAlgo::Eigen, &a).unwrap();
        check_polar(&a, &r_e, &u_e, 1e-13);
    }

    #[test]
    fn polar_decomp_svd_on_higham_cases() {
        // Higham & Noferini test (5.1), cross-checked against Higham's algorithm
        let a = case51();
        let mut r_s = Tensor2::<9>::new();
        let mut u_s = Tensor2::<6>::new();
        polar_decomp_mx(&mut r_s, &mut u_s, None, PolarAlgo::SVD, &a).unwrap();
        check_polar(&a, &r_s, &u_s, 1e-13);
        let mut r_h = Tensor2::<9>::new();
        let mut u_h = Tensor2::<6>::new();
        polar_decomp_mx(&mut r_h, &mut u_h, None, PolarAlgo::Quaternion, &a).unwrap();
        mat_approx_eq(&r_s.as_std_matrix(), &r_h.as_std_matrix(), 1e-13);
        mat_approx_eq(&u_s.as_std_matrix(), &u_h.as_std_matrix(), 1e-13);

        // Higham & Noferini test (5.2) over a range of condition numbers
        for y in [1.0f64, 1e-2, 1e-4, 1e-6, 1e-8] {
            let a = case52(y);
            let tol = if y == 1.0 { 1e-13 } else { 1e-8 };
            let mut r_s = Tensor2::<9>::new();
            let mut u_s = Tensor2::<6>::new();
            polar_decomp_mx(&mut r_s, &mut u_s, None, PolarAlgo::SVD, &a).unwrap();
            check_polar(&a, &r_s, &u_s, tol);
        }
    }
}
