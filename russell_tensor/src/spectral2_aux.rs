//! Auxiliary methods related to [Spectral2]

use crate::{EigStatus, SQRT_2, SQRT_3, SQRT_6, Spectral2, Tensor2};

/// Rotates the eigenvalues to the principal values space
///
/// Returns `(λ_star_1, λ_star_2, λ_star_3)`
pub fn spectral2_octahedral(spc: &Spectral2) -> (f64, f64, f64) {
    let (s1, s2, s3) = (spc.lam[0], spc.lam[1], spc.lam[2]);
    let ls1 = (2.0 * s1 - s2 - s3) / SQRT_6;
    let ls2 = (s1 + s2 + s3) / SQRT_3;
    let ls3 = (s3 - s2) / SQRT_2;
    (ls1, ls2, ls3)
}

/// Composes a new tensor from the eigenprojectors and diagonal values `d`
///
/// The composition depends on the eigenvalue multiplicity (see [EigStatus]):
///
/// ```text
/// [Distinct]       B = d[0] P[0] + d[1] P[1] + d[2] P[2]
///
/// [Coalescent]     B = d[0] P_dist + d[1] P_coal
///
/// [Spherical]      B = d[0] I
/// ```
///
/// In the coalescent case, `P_dist = P[0]` is the eigenprojector of the distinct eigenvalue while
/// `P_coal = P[1] = I - P_dist` is the (non-unique) eigenprojector of the double eigenvalue (rank 2).
///
/// In the spherical case, the eigenprojectors are not defined and the tensor is simply `B = d[0] I`.
///
/// # Input
///
/// * `spc` -- The spectral decomposition (with the eigenvalues and eigenprojectors)
/// * `d` -- The diagonal values (e.g. the eigenvalues or the values of an isotropic function)
///
/// # Output
///
/// * `bb` -- The resulting tensor
pub fn spectral2_compose(bb: &mut Tensor2<6>, spc: &Spectral2, d: &[f64; 3]) {
    match spc.status {
        EigStatus::Spherical => {
            bb.vec[0] = d[0];
            bb.vec[1] = d[0];
            bb.vec[2] = d[0];
            bb.vec[3] = 0.0;
            bb.vec[4] = 0.0;
            bb.vec[5] = 0.0;
        }
        EigStatus::Coalesce01 | EigStatus::Coalesce12 => {
            bb.vec[0] = d[0] * spc.proj[0].vec[0] + d[1] * spc.proj[1].vec[0];
            bb.vec[1] = d[0] * spc.proj[0].vec[1] + d[1] * spc.proj[1].vec[1];
            bb.vec[2] = d[0] * spc.proj[0].vec[2] + d[1] * spc.proj[1].vec[2];
            bb.vec[3] = d[0] * spc.proj[0].vec[3] + d[1] * spc.proj[1].vec[3];
            bb.vec[4] = d[0] * spc.proj[0].vec[4] + d[1] * spc.proj[1].vec[4];
            bb.vec[5] = d[0] * spc.proj[0].vec[5] + d[1] * spc.proj[1].vec[5];
        }
        _ => {
            bb.vec[0] = d[0] * spc.proj[0].vec[0] + d[1] * spc.proj[1].vec[0] + d[2] * spc.proj[2].vec[0];
            bb.vec[1] = d[0] * spc.proj[0].vec[1] + d[1] * spc.proj[1].vec[1] + d[2] * spc.proj[2].vec[1];
            bb.vec[2] = d[0] * spc.proj[0].vec[2] + d[1] * spc.proj[1].vec[2] + d[2] * spc.proj[2].vec[2];
            bb.vec[3] = d[0] * spc.proj[0].vec[3] + d[1] * spc.proj[1].vec[3] + d[2] * spc.proj[2].vec[3];
            bb.vec[4] = d[0] * spc.proj[0].vec[4] + d[1] * spc.proj[1].vec[4] + d[2] * spc.proj[2].vec[4];
            bb.vec[5] = d[0] * spc.proj[0].vec[5] + d[1] * spc.proj[1].vec[5] + d[2] * spc.proj[2].vec[5];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{spectral2_compose, spectral2_octahedral};
    use crate::{EigDerivStatus, EigMethod, EigStatus, SQRT_3, SQRT_3_BY_2, SamplesTensor2, Spectral2, Tensor2};
    use russell_lab::{approx_eq, mat_approx_eq};

    #[test]
    fn spectral_octahedral_works() {
        // the following data corresponds to p = 1 and q = 3
        #[rustfmt::skip]
        let principal_stresses_and_lode = [
            ( 3.0          ,  0.0          ,  0.0          ,  1.0 ),
            ( 0.0          ,  3.0          ,  0.0          ,  1.0 ),
            ( 0.0          ,  0.0          ,  3.0          ,  1.0 ),
            ( 1.0 + SQRT_3 ,  1.0 - SQRT_3 ,  1.0          ,  0.0 ),
            ( 1.0 + SQRT_3 ,  1.0          ,  1.0 - SQRT_3 ,  0.0 ),
            ( 1.0          ,  1.0 + SQRT_3 ,  1.0 - SQRT_3 ,  0.0 ),
            ( 1.0 - SQRT_3 ,  1.0 + SQRT_3 ,  1.0          ,  0.0 ),
            ( 1.0          ,  1.0 - SQRT_3 ,  1.0 + SQRT_3 ,  0.0 ),
            ( 1.0 - SQRT_3 ,  1.0          ,  1.0 + SQRT_3 ,  0.0 ),
            ( 2.0          , -1.0          ,  2.0          , -1.0 ),
            ( 2.0          ,  2.0          , -1.0          , -1.0 ),
            (-1.0          ,  2.0          ,  2.0          , -1.0 ),
        ];
        let mut spec = Spectral2::new();
        let mut tt = Tensor2::<6>::new();
        for (sigma_1, sigma_2, sigma_3, lode_correct) in &principal_stresses_and_lode {
            tt.set(0, *sigma_1);
            tt.set(1, *sigma_2);
            tt.set(2, *sigma_3);
            spec.decompose(&tt).unwrap();
            let (ls1, ls2, ls3) = spectral2_octahedral(&spec);
            let radius = f64::sqrt(ls3 * ls3 + ls1 * ls1);
            let distance = ls2;
            approx_eq(distance / SQRT_3, 1.0, 1e-15);
            approx_eq(radius * SQRT_3_BY_2, 3.0, 1e-15);
            if radius > 0.0 {
                let cos_theta = ls1 / radius;
                let lode = 4.0 * f64::powf(cos_theta, 3.0) - 3.0 * cos_theta;
                approx_eq(lode, *lode_correct, 1e-15);
            }
        }
    }

    #[test]
    fn spectral_compose_distinct_works() {
        #[rustfmt::skip]
        let aa = Tensor2::<6>::from_std_matrix(&[
            [ 4.0, -1.0,  0.5],
            [-1.0,  2.0,  0.2],
            [ 0.5,  0.2,  1.0],
        ]).unwrap();
        let mut spec = Spectral2::new();
        spec.decompose(&aa).unwrap();
        assert_eq!(spec.status, EigStatus::Distinct);

        // reconstruction: B = Σ λ[k] P[k] = A
        let mut bb = Tensor2::<6>::new();
        spectral2_compose(&mut bb, &spec, &spec.lam);
        mat_approx_eq(&bb.as_std_matrix(), &aa.as_std_matrix(), 1e-14);

        // a generic set of diagonal values: B = Σ d[k] P[k]
        let d = [10.0, 20.0, 30.0];
        spectral2_compose(&mut bb, &spec, &d);
        let mut cc = Tensor2::<6>::new();
        for k in 0..3 {
            for m in 0..6 {
                cc.vec[m] += d[k] * spec.proj[k].vec[m];
            }
        }
        mat_approx_eq(&bb.as_std_matrix(), &cc.as_std_matrix(), 1e-14);
    }

    #[test]
    fn spectral_compose_coalescent_works() {
        // The coalescent case requires the layout set by `deriv_eigenproj`: P[0] is the
        // eigenprojector of the distinct eigenvalue and P[1] = ½ (I - P[0]), so that
        // B = d[0] P[0] + 2 d[1] P[1].
        for sample in [&SamplesTensor2::COAL_01, &SamplesTensor2::COAL_12] {
            let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
            let mut spec = Spectral2::new();
            let status = spec.deriv_eigenproj(&aa, EigMethod::AnalyticalHZ).unwrap();
            assert_eq!(status, EigDerivStatus::Success);

            // reconstruction: B = A (with d = λ)
            let mut bb = Tensor2::<6>::new();
            spectral2_compose(&mut bb, &spec, &spec.lam);
            mat_approx_eq(&bb.as_std_matrix(), &aa.as_std_matrix(), 1e-12);

            // isotropic function η: B = η(L) P[0] + η(λ) P_coal
            let d = [
                spec.lam[0] * spec.lam[0],
                spec.lam[1] * spec.lam[1],
                spec.lam[2] * spec.lam[2],
            ];
            spectral2_compose(&mut bb, &spec, &d);
            let mut cc = Tensor2::<6>::new();
            for m in 0..6 {
                cc.vec[m] = d[0] * spec.proj[0].vec[m] + d[1] * spec.proj[1].vec[m];
            }
            mat_approx_eq(&bb.as_std_matrix(), &cc.as_std_matrix(), 1e-12);
        }
    }

    #[test]
    fn spectral_compose_spherical_works() {
        // The spherical case has undefined eigenprojectors; the result is B = d[0] I.
        #[rustfmt::skip]
        let aa = Tensor2::<6>::from_std_matrix(&[
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]).unwrap();
        let mut spec = Spectral2::new();
        spec.decompose(&aa).unwrap();
        assert_eq!(spec.status, EigStatus::Spherical);

        let d = [7.0, -1.0, 99.0];
        let mut bb = Tensor2::<6>::new();
        spectral2_compose(&mut bb, &spec, &d);
        let mut expected = Tensor2::<6>::new();
        expected.set(0, 7.0);
        expected.set(1, 7.0);
        expected.set(2, 7.0);
        mat_approx_eq(&bb.as_std_matrix(), &expected.as_std_matrix(), 1e-14);
    }
}
