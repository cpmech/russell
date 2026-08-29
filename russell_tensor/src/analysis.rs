use crate::{Rep, SQRT_2, StrError, Tensor2, Tensor4};
use russell_lab::{Matrix, mat_inverse};
use std::fmt;

/// Holds the Voigt-Reuss-Hill averages and the universal anisotropy index (Au)
///
/// # Note
///
/// This structure is only meaningful for **super-symmetric** tensors
/// (i.e., tensors with both minor and major symmetries).
///
/// # Reference
///
/// R. Hill (1952) The elastic behavior of a crystalline aggregate,
/// Proceedings of the Physical Society. Section A, 65(5), 349–354,
/// <https://doi.org/10.1088/0370-1298/65/5/307>
#[derive(Clone, Copy, Debug)]
pub struct VoigtReussHill {
    /// `Kv`: the Voigt average of the bulk modulus
    pub kk_v: f64,

    /// `Gv`: the Voigt average of the shear modulus
    pub gg_v: f64,

    /// `Kr`: the Reuss average of the bulk modulus
    pub kk_r: f64,

    /// `Gr`: the Reuss average of the shear modulus
    pub gg_r: f64,

    /// `Kh`: the Hill (arithmetic) average of the bulk modulus
    pub kk_h: f64,

    /// `Gh`: the Hill (arithmetic) average of the shear modulus
    pub gg_h: f64,

    /// `Au`: the universal anisotropy index
    pub aa_u: f64,
}

impl fmt::Display for VoigtReussHill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match f.precision() {
            Some(p) => {
                write!(
                    f,
                    "Kv = {:.p$}\nGv = {:.p$}\nKr = {:.p$}\nGr = {:.p$}\nKh = {:.p$}\nGh = {:.p$}\nAu = {:.p$}",
                    self.kk_v,
                    self.gg_v,
                    self.kk_r,
                    self.gg_r,
                    self.kk_h,
                    self.gg_h,
                    self.aa_u,
                    p = p,
                )
            }
            None => {
                write!(
                    f,
                    "Kv = {}\nGv = {}\nKr = {}\nGr = {}\nKh = {}\nGh = {}\nAu = {}",
                    self.kk_v, self.gg_v, self.kk_r, self.gg_r, self.kk_h, self.gg_h, self.aa_u
                )
            }
        }
    }
}

/// Computes the Voigt-Reuss-Hill averages and the universal anisotropy index (Au)
///
/// # Warning
///
/// The Kelvin-Mandel matrix must be **super-symmetric** (i.e., the tensor has both
/// minor and major symmetry). Otherwise, the results are wrong and this function
/// will **not** raise any error.
///
/// # Output
///
/// * `ss` -- the compliance tensor `S` (the inverse of `cc`); must be [Rep::Symmetric]
///
/// # Input
///
/// * `cc` -- the elasticity (stiffness) tensor `C`; must be [Rep::Symmetric]
///
/// # Returns
///
/// Returns a [VoigtReussHill] with the values `Kv`, `Gv`, `Kr`, `Gr`, `Kh`, `Gh`, and `Au`.
///
/// # Formulas
///
/// ```text
/// K_V = (C11 + C22 + C33 + 2 (C12 + C13 + C23)) / 9
/// G_V = (C11 + C22 + C33 − (C12 + C13 + C23) + 3 (C44 + C55 + C66)) / 15
/// K_R = 1 / (S11 + S22 + S33 + 2 (S12 + S13 + S23))
/// G_R = 15 / (4 (S11 + S22 + S33) − 4 (S12 + S13 + S23) + 3 (S44 + S55 + S66))
/// K_H = (K_V + K_R) / 2
/// G_H = (G_V + G_R) / 2
/// Au = 5 G_V / G_R + K_V / K_R − 6
/// ```
///
/// where `C` and `S` are the 6×6 stiffness and compliance matrices in Voigt notation.
/// Internally, the tensors are stored in Kelvin-Mandel notation; the shear-shear
/// components are twice the Voigt ones (hence the `1.5` and `6.0` factors in the code).
///
/// # Errors
///
/// Returns an error if `ss` or `cc` is not [Rep::Symmetric], or if `cc` cannot be inverted.
///
/// # Reference
///
/// R. Hill (1952) The elastic behavior of a crystalline aggregate,
/// Proceedings of the Physical Society. Section A, 65(5), 349–354,
/// <https://doi.org/10.1088/0370-1298/65/5/307>
pub fn calc_voigt_reuss_hill(ss: &mut Tensor4, cc: &Tensor4) -> Result<VoigtReussHill, StrError> {
    // check
    if ss.rep() != Rep::Symmetric {
        return Err("ss must be Rep::Symmetric");
    }
    if cc.rep() != Rep::Symmetric {
        return Err("cc must be Rep::Symmetric");
    }

    // stiffness matrix (Kelvin-Mandel)
    let mut c_mat = Matrix::new(6, 6);
    for m in 0..6 {
        for n in 0..6 {
            c_mat.set(m, n, cc.get(m, n));
        }
    }

    // compliance matrix (Kelvin-Mandel)
    let mut s_mat = Matrix::new(6, 6);
    mat_inverse(&mut s_mat, &c_mat)?;

    // set the compliance tensor
    for m in 0..6 {
        for n in 0..6 {
            ss.set(m, n, s_mat.get(m, n));
        }
    }

    // stiffness components
    let sum_c_diag = c_mat.get(0, 0) + c_mat.get(1, 1) + c_mat.get(2, 2);
    let sum_c_off = c_mat.get(0, 1) + c_mat.get(0, 2) + c_mat.get(1, 2);
    let sum_c_shear = c_mat.get(3, 3) + c_mat.get(4, 4) + c_mat.get(5, 5);

    // compliance components
    let sum_s_diag = s_mat.get(0, 0) + s_mat.get(1, 1) + s_mat.get(2, 2);
    let sum_s_off = s_mat.get(0, 1) + s_mat.get(0, 2) + s_mat.get(1, 2);
    let sum_s_shear = s_mat.get(3, 3) + s_mat.get(4, 4) + s_mat.get(5, 5);

    // Voigt averages (using stiffness)
    let k_v = (sum_c_diag + 2.0 * sum_c_off) / 9.0;
    let g_v = (sum_c_diag - sum_c_off + 1.5 * sum_c_shear) / 15.0;

    // Reuss averages (using compliance)
    let k_r = 1.0 / (sum_s_diag + 2.0 * sum_s_off);
    let g_r = 15.0 / (4.0 * sum_s_diag - 4.0 * sum_s_off + 6.0 * sum_s_shear);

    // Hill averages (arithmetic mean)
    let k_h = (k_v + k_r) / 2.0;
    let g_h = (g_v + g_r) / 2.0;

    // universal anisotropy index
    let a_u = 5.0 * (g_v / g_r) + (k_v / k_r) - 6.0;

    Ok(VoigtReussHill {
        kk_v: k_v,
        gg_v: g_v,
        kk_r: k_r,
        gg_r: g_r,
        kk_h: k_h,
        gg_h: g_h,
        aa_u: a_u,
    })
}

/// Calculates the internal stability tensor for the analysis of symmetric tensors
///
/// The output of this function corresponds to equation (27) of Reference 1
/// and the H tensor in Equation (4.1) of Reference 2.
///
/// # Output
///
/// * `hh` -- the internal stability tensor; must be [Rep::Symmetric]
///
/// # Input
///
/// * `sigma` -- the Cauchy stress tensor; must be [Rep::Symmetric]
///
/// # Errors
///
/// Returns an error if `hh` or `sigma` is not [Rep::Symmetric].
///
/// # References
///
/// 1. J. W. Morris Jr. & C. R. Krenn (2000) The internal stability of an elastic solid,
///    Philosophical Magazine A, 80:12, 2827-2840, <https://doi.org/10.1080/01418610008223897>
/// 2. M. Maździarz (2025) Mechanical stability conditions for 3D and 2D crystals under arbitrary load,
///    Archives of Mechanics, 77(4), 379–399, 2025, <https://doi.org/10.24423/aom.4679>
pub fn calc_internal_stability_tensor(hh: &mut Tensor4, sigma: &Tensor2) -> Result<(), StrError> {
    // check
    if hh.rep() != Rep::Symmetric {
        return Err("hh must be Rep::Symmetric");
    }
    if sigma.rep() != Rep::Symmetric {
        return Err("sigma must be Rep::Symmetric");
    }
    let sig = sigma.as_data();

    // row 0
    hh.set(0, 0, sig[0]);
    hh.set(0, 1, (-sig[0] - sig[1]) / 2.0);
    hh.set(0, 2, (-sig[0] - sig[2]) / 2.0);
    hh.set(0, 3, sig[3] / 2.0);
    hh.set(0, 4, -sig[4] / 2.0);
    hh.set(0, 5, sig[5] / 2.0);

    // row 1
    hh.set(1, 0, (-sig[0] - sig[1]) / 2.0);
    hh.set(1, 1, sig[1]);
    hh.set(1, 2, (-sig[1] - sig[2]) / 2.0);
    hh.set(1, 3, sig[3] / 2.0);
    hh.set(1, 4, sig[4] / 2.0);
    hh.set(1, 5, -sig[5] / 2.0);

    // row 2
    hh.set(2, 0, (-sig[0] - sig[2]) / 2.0);
    hh.set(2, 1, (-sig[1] - sig[2]) / 2.0);
    hh.set(2, 2, sig[2]);
    hh.set(2, 3, -sig[3] / 2.0);
    hh.set(2, 4, sig[4] / 2.0);
    hh.set(2, 5, sig[5] / 2.0);

    // row 3
    hh.set(3, 0, sig[3] / 2.0);
    hh.set(3, 1, sig[3] / 2.0);
    hh.set(3, 2, -sig[3] / 2.0);
    hh.set(3, 3, sig[0] + sig[1]);
    hh.set(3, 4, sig[5] / SQRT_2);
    hh.set(3, 5, sig[4] / SQRT_2);

    // row 4
    hh.set(4, 0, -sig[4] / 2.0);
    hh.set(4, 1, sig[4] / 2.0);
    hh.set(4, 2, sig[4] / 2.0);
    hh.set(4, 3, sig[5] / SQRT_2);
    hh.set(4, 4, sig[1] + sig[2]);
    hh.set(4, 5, sig[3] / SQRT_2);

    // row 5
    hh.set(5, 0, sig[5] / 2.0);
    hh.set(5, 1, -sig[5] / 2.0);
    hh.set(5, 2, sig[5] / 2.0);
    hh.set(5, 3, sig[4] / SQRT_2);
    hh.set(5, 4, sig[3] / SQRT_2);
    hh.set(5, 5, sig[0] + sig[2]);

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{calc_internal_stability_tensor, calc_voigt_reuss_hill};
    use crate::{Rep, SQRT_2, Tensor2, Tensor4};
    use russell_lab::approx_eq;

    #[test]
    fn calc_internal_stability_tensor_works() {
        // reference: Maździarz (2025), for sigma = diag(27.06, 27.06, 20.585)
        let sigma = Tensor2::from_std_matrix(
            &[[27.06, 0.0, 0.0], [0.0, 27.06, 0.0], [0.0, 0.0, 20.585]],
            Rep::Symmetric,
        )
        .unwrap();
        let mut hh = Tensor4::new(Rep::Symmetric);
        calc_internal_stability_tensor(&mut hh, &sigma).unwrap();
        #[rustfmt::skip]
        let correct = [
            [27.06, -27.06, -23.8225, 0.0, 0.0, 0.0],
            [-27.06, 27.06, -23.8225, 0.0, 0.0, 0.0],
            [-23.8225, -23.8225, 20.585, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 54.12, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 47.645, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 47.645],
        ];
        for m in 0..6 {
            for n in 0..6 {
                approx_eq(hh.get(m, n), correct[m][n], 1e-12);
            }
        }
    }

    #[test]
    fn calc_voigt_reuss_hill_works() {
        // Eq19 from Maździarz (2025), in Voigt notation
        #[rustfmt::skip]
        let voigt = [
            [296.57, 144.76, 125.5, -35.27, -2.5, 3.45],
            [144.76, 273.54, 74.42, 17.96, -4.93, 1.37],
            [125.5, 74.42, 169.18, -39.37, -18.81, 9.45],
            [-35.27, 17.96, -39.37, 110.56, 0.02, 0.17],
            [-2.5, -4.93, -18.81, 0.02, 113.03, -31.15],
            [3.45, 1.37, 9.45, 0.17, -31.15, 112.41],
        ];

        // convert Voigt -> Kelvin-Mandel
        let mut cc = Tensor4::new(Rep::Symmetric);
        for m in 0..6 {
            for n in 0..6 {
                let factor = if m < 3 && n < 3 {
                    1.0
                } else if (m < 3) != (n < 3) {
                    SQRT_2
                } else {
                    2.0
                };
                cc.set(m, n, factor * voigt[m][n]);
            }
        }

        // calculate the averages
        let mut ss = Tensor4::new(Rep::Symmetric);
        let vrh = calc_voigt_reuss_hill(&mut ss, &cc).unwrap();

        // check
        approx_eq(vrh.kk_v, 158.7388888888889, 1e-12);
        approx_eq(vrh.gg_v, 93.5073333333333, 1e-12);
        approx_eq(vrh.kk_r, 131.6385407574474, 1e-12);
        approx_eq(vrh.gg_r, 74.87683938076444, 1e-12);
        approx_eq(vrh.kk_h, 145.1887148231681, 1e-12);
        approx_eq(vrh.gg_h, 84.1920863570489, 1e-12);
        approx_eq(vrh.aa_u, 1.449945284449501, 1e-12);

        // check the Display implementation (no precision specified)
        assert_eq!(
            format!("{}", vrh),
            "Kv = 158.7388888888889\n\
             Gv = 93.50733333333332\n\
             Kr = 131.6385407574474\n\
             Gr = 74.87683938076444\n\
             Kh = 145.18871482316814\n\
             Gh = 84.19208635704888\n\
             Au = 1.4499452844495009"
        );

        // check the Display implementation (precision = 3)
        assert_eq!(
            format!("{:.3}", vrh),
            "Kv = 158.739\n\
             Gv = 93.507\n\
             Kr = 131.639\n\
             Gr = 74.877\n\
             Kh = 145.189\n\
             Gh = 84.192\n\
             Au = 1.450"
        );
    }
}
