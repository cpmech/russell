use crate::polar_brannon::{polar_rotation_brannon, polar_rotation_brannon2d};
use crate::polar_higham::polar_quaternion_higham;
use crate::{t2_gen_dot_gen_tra_chop, t2_gen_tra_dot_gen_chop, Rep, Tensor2};
use russell_lab::StrError;

/// Specifies the polar decomposition algorithm
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolarAlgo {
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

    // Polar rotation R and right stretch U
    let nit = match algo {
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
            polar_quaternion_higham(rr, uu, ff); // R = Q, U = H
            0
        }
    };

    // Left stretch V = F Rᵀ (common to all algorithms)
    if let Some(v) = vv {
        if v.rep() != Rep::Symmetric {
            return Err("vv must be Rep::Symmetric");
        }
        t2_gen_dot_gen_tra_chop(v.as_mut_data(), 1.0, ff.as_data(), rr.as_data());
    }

    Ok(nit)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{polar_decomp, PolarAlgo};
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
}
