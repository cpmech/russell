//! Auxiliary methods related to [Spectral2]

use crate::{SQRT_2, SQRT_3, SQRT_6, Spectral2, Tensor2};

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

/// Composes a new tensor from the eigenprojectors and diagonal values (lambda)
///
/// ```text
///      3
/// B =  Σ  d[k] * P[k]
///     k=1
/// ```
pub fn spectral2_compose(bb: &mut Tensor2<6>, spc: &Spectral2, d: &[f64; 3]) {
    bb.vec[0] = d[0] * spc.proj[0].vec[0] + d[1] * spc.proj[1].vec[0] + d[2] * spc.proj[2].vec[0];
    bb.vec[1] = d[0] * spc.proj[0].vec[1] + d[1] * spc.proj[1].vec[1] + d[2] * spc.proj[2].vec[1];
    bb.vec[2] = d[0] * spc.proj[0].vec[2] + d[1] * spc.proj[1].vec[2] + d[2] * spc.proj[2].vec[2];
    bb.vec[3] = d[0] * spc.proj[0].vec[3] + d[1] * spc.proj[1].vec[3] + d[2] * spc.proj[2].vec[3];
    bb.vec[4] = d[0] * spc.proj[0].vec[4] + d[1] * spc.proj[1].vec[4] + d[2] * spc.proj[2].vec[4];
    bb.vec[5] = d[0] * spc.proj[0].vec[5] + d[1] * spc.proj[1].vec[5] + d[2] * spc.proj[2].vec[5];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{spectral2_compose, spectral2_octahedral};
    use crate::{SQRT_3, SQRT_3_BY_2, Spectral2, Tensor2};
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
    fn spectral_compose_works() {
        #[rustfmt::skip]
        let aa = Tensor2::<6>::from_std_matrix(&[
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]).unwrap();
        let mut spec = Spectral2::new();
        spec.decompose(&aa).unwrap();
        // B = Σ λ[k] P[k] = A
        let mut bb = Tensor2::<6>::new();
        spectral2_compose(&mut bb, &spec, &spec.lam);
        mat_approx_eq(&bb.as_std_matrix(), &aa.as_std_matrix(), 1e-14);
    }
}
