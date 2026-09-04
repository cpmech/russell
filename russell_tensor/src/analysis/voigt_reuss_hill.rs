use crate::{StrError, Tensor4};
use russell_lab::{Matrix, mat_inverse};
use serde::{Deserialize, Serialize};
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
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
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
    #[rustfmt::skip]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match f.precision() {
            Some(p) => {
                write!(
                    f, "Kv = {:.p$}\nGv = {:.p$}\nKr = {:.p$}\nGr = {:.p$}\nKh = {:.p$}\nGh = {:.p$}\nAu = {:.p$}",
                    self.kk_v, self.gg_v, self.kk_r, self.gg_r, self.kk_h, self.gg_h, self.aa_u, p = p,
                )
            }
            None => {
                write!(
                    f, "Kv = {}\nGv = {}\nKr = {}\nGr = {}\nKh = {}\nGh = {}\nAu = {}",
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
/// * `ss` -- the compliance tensor `S` (the inverse of `cc`); must be symmetric
///
/// # Input
///
/// * `cc` -- the elasticity (stiffness) tensor `C`; must be symmetric
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
/// Returns an error if `ss` or `cc` is not symmetric, or if `cc` cannot be inverted.
///
/// # Reference
///
/// R. Hill (1952) The elastic behavior of a crystalline aggregate,
/// Proceedings of the Physical Society. Section A, 65(5), 349–354,
/// <https://doi.org/10.1088/0370-1298/65/5/307>
pub fn voigt_reuss_hill(ss: &mut Tensor4<6>, cc: &Tensor4<6>) -> Result<VoigtReussHill, StrError> {
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::voigt_reuss_hill;
    use crate::Tensor4;
    use russell_lab::approx_eq;

    #[test]
    fn voigt_reuss_hill_works() {
        // Eq19 from Maździarz (2025)
        let cc = Tensor4::<6>::from_std_array(&[
            [
                [[296.57, -35.27, 3.45], [-35.27, 144.76, -2.5], [3.45, -2.5, 125.5]],
                [[-35.27, 110.56, 0.17], [110.56, 17.96, 0.02], [0.17, 0.02, -39.37]],
                [[3.45, 0.17, 112.41], [0.17, 1.37, -31.15], [112.41, -31.15, 9.45]],
            ],
            [
                [[-35.27, 110.56, 0.17], [110.56, 17.96, 0.02], [0.17, 0.02, -39.37]],
                [[144.76, 17.96, 1.37], [17.96, 273.54, -4.93], [1.37, -4.93, 74.42]],
                [[-2.5, 0.02, -31.15], [0.02, -4.93, 113.03], [-31.15, 113.03, -18.81]],
            ],
            [
                [[3.45, 0.17, 112.41], [0.17, 1.37, -31.15], [112.41, -31.15, 9.45]],
                [[-2.5, 0.02, -31.15], [0.02, -4.93, 113.03], [-31.15, 113.03, -18.81]],
                [[125.5, -39.37, 9.45], [-39.37, 74.42, -18.81], [9.45, -18.81, 169.18]],
            ],
        ])
        .unwrap();

        // calculate the averages
        let mut ss = Tensor4::<6>::new();
        let vrh = voigt_reuss_hill(&mut ss, &cc).unwrap();

        // check
        approx_eq(vrh.kk_v, 158.7388888888889, 1e-13);
        approx_eq(vrh.gg_v, 93.5073333333333, 1e-13);
        approx_eq(vrh.kk_r, 131.6385407574474, 1e-13);
        approx_eq(vrh.gg_r, 74.87683938076444, 1e-13);
        approx_eq(vrh.kk_h, 145.1887148231681, 1e-13);
        approx_eq(vrh.gg_h, 84.1920863570489, 1e-13);
        approx_eq(vrh.aa_u, 1.449945284449501, 1e-13);

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
