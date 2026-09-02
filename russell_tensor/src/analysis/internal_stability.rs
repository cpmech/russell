use crate::{SQRT_2, StrError, Tensor2, Tensor4};

/// Calculates the internal stability tensor for the analysis of symmetric tensors
///
/// The output of this function corresponds to equation (27) of Reference 1
/// and the H tensor in Equation (4.1) of Reference 2.
///
/// # Output
///
/// * `hh` -- the internal stability tensor; must be symmetric
///
/// # Input
///
/// * `sigma` -- the Cauchy stress tensor; must be symmetric
///
/// # Errors
///
/// Returns an error if `hh` or `sigma` is not symmetric.
///
/// # References
///
/// 1. J. W. Morris Jr. & C. R. Krenn (2000) The internal stability of an elastic solid,
///    Philosophical Magazine A, 80:12, 2827-2840, <https://doi.org/10.1080/01418610008223897>
/// 2. M. Maździarz (2025) Mechanical stability conditions for 3D and 2D crystals under arbitrary load,
///    Archives of Mechanics, 77(4), 379–399, 2025, <https://doi.org/10.24423/aom.4679>
pub fn internal_stability_tensor(hh: &mut Tensor4<6>, sigma: &Tensor2<6>) -> Result<(), StrError> {
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
    use super::internal_stability_tensor;
    use crate::{Tensor2, Tensor4};
    use russell_lab::approx_eq;

    #[test]
    fn internal_stability_tensor_works() {
        // reference: Maździarz (2025), for sigma = diag(27.06, 27.06, 20.585)
        let sigma = Tensor2::<6>::from_std_matrix(&[
            [27.06, 0.0, 0.0],  // 1
            [0.0, 27.06, 0.0],  // 2
            [0.0, 0.0, 20.585], // 3
        ])
        .unwrap();
        let mut hh = Tensor4::<6>::new();
        internal_stability_tensor(&mut hh, &sigma).unwrap();
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
}
