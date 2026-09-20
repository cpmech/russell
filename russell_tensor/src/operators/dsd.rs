use crate::ADD;
use crate::{Tensor2, Tensor4};

/// Performs the duo-sum-dyadic (dsd) operation with two Tensor2 yielding a minor-symmetric Tensor4
///
/// Computes:
///
/// ```text
/// ADD: D += s (A ⊗ B + B ⊗ A)  or  SET: D = s (A ⊗ B + B ⊗ A)
/// ```
///
/// With Cartesian components (example with SET):
///
/// ```text
/// Dᵢⱼₖₗ = s (Aᵢⱼ Bₖₗ + Bᵢⱼ Aₖₗ)
/// ```
///
/// **Note:** For `N = 4` (symmetric generalized plane), the `{00,11,22,01}` block
/// is analytically zero.
///
/// # Output
///
/// * `dd` -- The resulting tensor (minor-symmetric)
///
/// # Input
///
/// * `op` -- operation: ADD or SET
/// * `aa` -- Second-order tensor, symmetric or not.
pub fn dsd_fn<const N: usize>(dd: &mut Tensor4<N>, op: u8, s: f64, aa: &Tensor2<N>, bb: &Tensor2<N>) {
    dsd_fn_slice::<N>(dd, op, s, aa.as_data(), bb.as_data());
}

/// Internal (unrolled) duo-sum-dyadic operation on raw Kelvin-Mandel vectors.
#[rustfmt::skip]
#[inline]
pub(crate) fn dsd_fn_slice<const N: usize>(dd: &mut Tensor4<N>, op: u8, s: f64, a: &[f64], b: &[f64]) {
    if op == ADD {
        if N == 4 {
            dd.add(0, 0, s*(2.0*a[0]*b[0]));
            dd.add(0, 1, s*(a[0]*b[1] + a[1]*b[0]));
            dd.add(0, 2, s*(a[0]*b[2] + a[2]*b[0]));
            dd.add(0, 3, s*(a[0]*b[3] + a[3]*b[0]));

            dd.add(1, 0, s*(a[1]*b[0] + a[0]*b[1]));
            dd.add(1, 1, s*(2.0*a[1]*b[1]));
            dd.add(1, 2, s*(a[1]*b[2] + a[2]*b[1]));
            dd.add(1, 3, s*(a[1]*b[3] + a[3]*b[1]));

            dd.add(2, 0, s*(a[2]*b[0] + a[0]*b[2]));
            dd.add(2, 1, s*(a[2]*b[1] + a[1]*b[2]));
            dd.add(2, 2, s*(2.0*a[2]*b[2]));
            dd.add(2, 3, s*(a[2]*b[3] + a[3]*b[2]));

            dd.add(3, 0, s*(a[3]*b[0] + a[0]*b[3]));
            dd.add(3, 1, s*(a[3]*b[1] + a[1]*b[3]));
            dd.add(3, 2, s*(a[3]*b[2] + a[2]*b[3]));
            dd.add(3, 3, s*(2.0*a[3]*b[3]));
        } else if N == 6 {
            dd.add(0, 0, s*(2.0*a[0]*b[0]));
            dd.add(0, 1, s*(a[0]*b[1] + a[1]*b[0]));
            dd.add(0, 2, s*(a[0]*b[2] + a[2]*b[0]));
            dd.add(0, 3, s*(a[0]*b[3] + a[3]*b[0]));
            dd.add(0, 4, s*(a[0]*b[4] + a[4]*b[0]));
            dd.add(0, 5, s*(a[0]*b[5] + a[5]*b[0]));

            dd.add(1, 0, s*(a[1]*b[0] + a[0]*b[1]));
            dd.add(1, 1, s*(2.0*a[1]*b[1]));
            dd.add(1, 2, s*(a[1]*b[2] + a[2]*b[1]));
            dd.add(1, 3, s*(a[1]*b[3] + a[3]*b[1]));
            dd.add(1, 4, s*(a[1]*b[4] + a[4]*b[1]));
            dd.add(1, 5, s*(a[1]*b[5] + a[5]*b[1]));

            dd.add(2, 0, s*(a[2]*b[0] + a[0]*b[2]));
            dd.add(2, 1, s*(a[2]*b[1] + a[1]*b[2]));
            dd.add(2, 2, s*(2.0*a[2]*b[2]));
            dd.add(2, 3, s*(a[2]*b[3] + a[3]*b[2]));
            dd.add(2, 4, s*(a[2]*b[4] + a[4]*b[2]));
            dd.add(2, 5, s*(a[2]*b[5] + a[5]*b[2]));

            dd.add(3, 0, s*(a[3]*b[0] + a[0]*b[3]));
            dd.add(3, 1, s*(a[3]*b[1] + a[1]*b[3]));
            dd.add(3, 2, s*(a[3]*b[2] + a[2]*b[3]));
            dd.add(3, 3, s*(2.0*a[3]*b[3]));
            dd.add(3, 4, s*(a[3]*b[4] + a[4]*b[3]));
            dd.add(3, 5, s*(a[3]*b[5] + a[5]*b[3]));

            dd.add(4, 0, s*(a[4]*b[0] + a[0]*b[4]));
            dd.add(4, 1, s*(a[4]*b[1] + a[1]*b[4]));
            dd.add(4, 2, s*(a[4]*b[2] + a[2]*b[4]));
            dd.add(4, 3, s*(a[4]*b[3] + a[3]*b[4]));
            dd.add(4, 4, s*(2.0*a[4]*b[4]));
            dd.add(4, 5, s*(a[4]*b[5] + a[5]*b[4]));

            dd.add(5, 0, s*(a[5]*b[0] + a[0]*b[5]));
            dd.add(5, 1, s*(a[5]*b[1] + a[1]*b[5]));
            dd.add(5, 2, s*(a[5]*b[2] + a[2]*b[5]));
            dd.add(5, 3, s*(a[5]*b[3] + a[3]*b[5]));
            dd.add(5, 4, s*(a[5]*b[4] + a[4]*b[5]));
            dd.add(5, 5, s*(2.0*a[5]*b[5]));
        } else {
            debug_assert!(N == 9);

            dd.add(0, 0, s*(2.0*a[0]*b[0]));
            dd.add(0, 1, s*(a[0]*b[1] + a[1]*b[0]));
            dd.add(0, 2, s*(a[0]*b[2] + a[2]*b[0]));
            dd.add(0, 3, s*(a[0]*b[3] + a[3]*b[0]));
            dd.add(0, 4, s*(a[0]*b[4] + a[4]*b[0]));
            dd.add(0, 5, s*(a[0]*b[5] + a[5]*b[0]));
            dd.add(0, 6, s*(a[0]*b[6] + a[6]*b[0]));
            dd.add(0, 7, s*(a[0]*b[7] + a[7]*b[0]));
            dd.add(0, 8, s*(a[0]*b[8] + a[8]*b[0]));

            dd.add(1, 0, s*(a[1]*b[0] + a[0]*b[1]));
            dd.add(1, 1, s*(2.0*a[1]*b[1]));
            dd.add(1, 2, s*(a[1]*b[2] + a[2]*b[1]));
            dd.add(1, 3, s*(a[1]*b[3] + a[3]*b[1]));
            dd.add(1, 4, s*(a[1]*b[4] + a[4]*b[1]));
            dd.add(1, 5, s*(a[1]*b[5] + a[5]*b[1]));
            dd.add(1, 6, s*(a[1]*b[6] + a[6]*b[1]));
            dd.add(1, 7, s*(a[1]*b[7] + a[7]*b[1]));
            dd.add(1, 8, s*(a[1]*b[8] + a[8]*b[1]));

            dd.add(2, 0, s*(a[2]*b[0] + a[0]*b[2]));
            dd.add(2, 1, s*(a[2]*b[1] + a[1]*b[2]));
            dd.add(2, 2, s*(2.0*a[2]*b[2]));
            dd.add(2, 3, s*(a[2]*b[3] + a[3]*b[2]));
            dd.add(2, 4, s*(a[2]*b[4] + a[4]*b[2]));
            dd.add(2, 5, s*(a[2]*b[5] + a[5]*b[2]));
            dd.add(2, 6, s*(a[2]*b[6] + a[6]*b[2]));
            dd.add(2, 7, s*(a[2]*b[7] + a[7]*b[2]));
            dd.add(2, 8, s*(a[2]*b[8] + a[8]*b[2]));

            dd.add(3, 0, s*(a[3]*b[0] + a[0]*b[3]));
            dd.add(3, 1, s*(a[3]*b[1] + a[1]*b[3]));
            dd.add(3, 2, s*(a[3]*b[2] + a[2]*b[3]));
            dd.add(3, 3, s*(2.0*a[3]*b[3]));
            dd.add(3, 4, s*(a[3]*b[4] + a[4]*b[3]));
            dd.add(3, 5, s*(a[3]*b[5] + a[5]*b[3]));
            dd.add(3, 6, s*(a[3]*b[6] + a[6]*b[3]));
            dd.add(3, 7, s*(a[3]*b[7] + a[7]*b[3]));
            dd.add(3, 8, s*(a[3]*b[8] + a[8]*b[3]));

            dd.add(4, 0, s*(a[4]*b[0] + a[0]*b[4]));
            dd.add(4, 1, s*(a[4]*b[1] + a[1]*b[4]));
            dd.add(4, 2, s*(a[4]*b[2] + a[2]*b[4]));
            dd.add(4, 3, s*(a[4]*b[3] + a[3]*b[4]));
            dd.add(4, 4, s*(2.0*a[4]*b[4]));
            dd.add(4, 5, s*(a[4]*b[5] + a[5]*b[4]));
            dd.add(4, 6, s*(a[4]*b[6] + a[6]*b[4]));
            dd.add(4, 7, s*(a[4]*b[7] + a[7]*b[4]));
            dd.add(4, 8, s*(a[4]*b[8] + a[8]*b[4]));

            dd.add(5, 0, s*(a[5]*b[0] + a[0]*b[5]));
            dd.add(5, 1, s*(a[5]*b[1] + a[1]*b[5]));
            dd.add(5, 2, s*(a[5]*b[2] + a[2]*b[5]));
            dd.add(5, 3, s*(a[5]*b[3] + a[3]*b[5]));
            dd.add(5, 4, s*(a[5]*b[4] + a[4]*b[5]));
            dd.add(5, 5, s*(2.0*a[5]*b[5]));
            dd.add(5, 6, s*(a[5]*b[6] + a[6]*b[5]));
            dd.add(5, 7, s*(a[5]*b[7] + a[7]*b[5]));
            dd.add(5, 8, s*(a[5]*b[8] + a[8]*b[5]));

            dd.add(6, 0, s*(a[6]*b[0] + a[0]*b[6]));
            dd.add(6, 1, s*(a[6]*b[1] + a[1]*b[6]));
            dd.add(6, 2, s*(a[6]*b[2] + a[2]*b[6]));
            dd.add(6, 3, s*(a[6]*b[3] + a[3]*b[6]));
            dd.add(6, 4, s*(a[6]*b[4] + a[4]*b[6]));
            dd.add(6, 5, s*(a[6]*b[5] + a[5]*b[6]));
            dd.add(6, 6, s*(2.0*a[6]*b[6]));
            dd.add(6, 7, s*(a[6]*b[7] + a[7]*b[6]));
            dd.add(6, 8, s*(a[6]*b[8] + a[8]*b[6]));

            dd.add(7, 0, s*(a[7]*b[0] + a[0]*b[7]));
            dd.add(7, 1, s*(a[7]*b[1] + a[1]*b[7]));
            dd.add(7, 2, s*(a[7]*b[2] + a[2]*b[7]));
            dd.add(7, 3, s*(a[7]*b[3] + a[3]*b[7]));
            dd.add(7, 4, s*(a[7]*b[4] + a[4]*b[7]));
            dd.add(7, 5, s*(a[7]*b[5] + a[5]*b[7]));
            dd.add(7, 6, s*(a[7]*b[6] + a[6]*b[7]));
            dd.add(7, 7, s*(2.0*a[7]*b[7]));
            dd.add(7, 8, s*(a[7]*b[8] + a[8]*b[7]));

            dd.add(8, 0, s*(a[8]*b[0] + a[0]*b[8]));
            dd.add(8, 1, s*(a[8]*b[1] + a[1]*b[8]));
            dd.add(8, 2, s*(a[8]*b[2] + a[2]*b[8]));
            dd.add(8, 3, s*(a[8]*b[3] + a[3]*b[8]));
            dd.add(8, 4, s*(a[8]*b[4] + a[4]*b[8]));
            dd.add(8, 5, s*(a[8]*b[5] + a[5]*b[8]));
            dd.add(8, 6, s*(a[8]*b[6] + a[6]*b[8]));
            dd.add(8, 7, s*(a[8]*b[7] + a[7]*b[8]));
            dd.add(8, 8, s*(2.0*a[8]*b[8]));
        }
    } else {
        if N == 4 {
            dd.set(0, 0, s*(2.0*a[0]*b[0]));
            dd.set(0, 1, s*(a[0]*b[1] + a[1]*b[0]));
            dd.set(0, 2, s*(a[0]*b[2] + a[2]*b[0]));
            dd.set(0, 3, s*(a[0]*b[3] + a[3]*b[0]));

            dd.set(1, 0, s*(a[1]*b[0] + a[0]*b[1]));
            dd.set(1, 1, s*(2.0*a[1]*b[1]));
            dd.set(1, 2, s*(a[1]*b[2] + a[2]*b[1]));
            dd.set(1, 3, s*(a[1]*b[3] + a[3]*b[1]));

            dd.set(2, 0, s*(a[2]*b[0] + a[0]*b[2]));
            dd.set(2, 1, s*(a[2]*b[1] + a[1]*b[2]));
            dd.set(2, 2, s*(2.0*a[2]*b[2]));
            dd.set(2, 3, s*(a[2]*b[3] + a[3]*b[2]));

            dd.set(3, 0, s*(a[3]*b[0] + a[0]*b[3]));
            dd.set(3, 1, s*(a[3]*b[1] + a[1]*b[3]));
            dd.set(3, 2, s*(a[3]*b[2] + a[2]*b[3]));
            dd.set(3, 3, s*(2.0*a[3]*b[3]));
        } else if N == 6 {
            dd.set(0, 0, s*(2.0*a[0]*b[0]));
            dd.set(0, 1, s*(a[0]*b[1] + a[1]*b[0]));
            dd.set(0, 2, s*(a[0]*b[2] + a[2]*b[0]));
            dd.set(0, 3, s*(a[0]*b[3] + a[3]*b[0]));
            dd.set(0, 4, s*(a[0]*b[4] + a[4]*b[0]));
            dd.set(0, 5, s*(a[0]*b[5] + a[5]*b[0]));

            dd.set(1, 0, s*(a[1]*b[0] + a[0]*b[1]));
            dd.set(1, 1, s*(2.0*a[1]*b[1]));
            dd.set(1, 2, s*(a[1]*b[2] + a[2]*b[1]));
            dd.set(1, 3, s*(a[1]*b[3] + a[3]*b[1]));
            dd.set(1, 4, s*(a[1]*b[4] + a[4]*b[1]));
            dd.set(1, 5, s*(a[1]*b[5] + a[5]*b[1]));

            dd.set(2, 0, s*(a[2]*b[0] + a[0]*b[2]));
            dd.set(2, 1, s*(a[2]*b[1] + a[1]*b[2]));
            dd.set(2, 2, s*(2.0*a[2]*b[2]));
            dd.set(2, 3, s*(a[2]*b[3] + a[3]*b[2]));
            dd.set(2, 4, s*(a[2]*b[4] + a[4]*b[2]));
            dd.set(2, 5, s*(a[2]*b[5] + a[5]*b[2]));

            dd.set(3, 0, s*(a[3]*b[0] + a[0]*b[3]));
            dd.set(3, 1, s*(a[3]*b[1] + a[1]*b[3]));
            dd.set(3, 2, s*(a[3]*b[2] + a[2]*b[3]));
            dd.set(3, 3, s*(2.0*a[3]*b[3]));
            dd.set(3, 4, s*(a[3]*b[4] + a[4]*b[3]));
            dd.set(3, 5, s*(a[3]*b[5] + a[5]*b[3]));

            dd.set(4, 0, s*(a[4]*b[0] + a[0]*b[4]));
            dd.set(4, 1, s*(a[4]*b[1] + a[1]*b[4]));
            dd.set(4, 2, s*(a[4]*b[2] + a[2]*b[4]));
            dd.set(4, 3, s*(a[4]*b[3] + a[3]*b[4]));
            dd.set(4, 4, s*(2.0*a[4]*b[4]));
            dd.set(4, 5, s*(a[4]*b[5] + a[5]*b[4]));

            dd.set(5, 0, s*(a[5]*b[0] + a[0]*b[5]));
            dd.set(5, 1, s*(a[5]*b[1] + a[1]*b[5]));
            dd.set(5, 2, s*(a[5]*b[2] + a[2]*b[5]));
            dd.set(5, 3, s*(a[5]*b[3] + a[3]*b[5]));
            dd.set(5, 4, s*(a[5]*b[4] + a[4]*b[5]));
            dd.set(5, 5, s*(2.0*a[5]*b[5]));
        } else {
            debug_assert!(N == 9);

            dd.set(0, 0, s*(2.0*a[0]*b[0]));
            dd.set(0, 1, s*(a[0]*b[1] + a[1]*b[0]));
            dd.set(0, 2, s*(a[0]*b[2] + a[2]*b[0]));
            dd.set(0, 3, s*(a[0]*b[3] + a[3]*b[0]));
            dd.set(0, 4, s*(a[0]*b[4] + a[4]*b[0]));
            dd.set(0, 5, s*(a[0]*b[5] + a[5]*b[0]));
            dd.set(0, 6, s*(a[0]*b[6] + a[6]*b[0]));
            dd.set(0, 7, s*(a[0]*b[7] + a[7]*b[0]));
            dd.set(0, 8, s*(a[0]*b[8] + a[8]*b[0]));

            dd.set(1, 0, s*(a[1]*b[0] + a[0]*b[1]));
            dd.set(1, 1, s*(2.0*a[1]*b[1]));
            dd.set(1, 2, s*(a[1]*b[2] + a[2]*b[1]));
            dd.set(1, 3, s*(a[1]*b[3] + a[3]*b[1]));
            dd.set(1, 4, s*(a[1]*b[4] + a[4]*b[1]));
            dd.set(1, 5, s*(a[1]*b[5] + a[5]*b[1]));
            dd.set(1, 6, s*(a[1]*b[6] + a[6]*b[1]));
            dd.set(1, 7, s*(a[1]*b[7] + a[7]*b[1]));
            dd.set(1, 8, s*(a[1]*b[8] + a[8]*b[1]));

            dd.set(2, 0, s*(a[2]*b[0] + a[0]*b[2]));
            dd.set(2, 1, s*(a[2]*b[1] + a[1]*b[2]));
            dd.set(2, 2, s*(2.0*a[2]*b[2]));
            dd.set(2, 3, s*(a[2]*b[3] + a[3]*b[2]));
            dd.set(2, 4, s*(a[2]*b[4] + a[4]*b[2]));
            dd.set(2, 5, s*(a[2]*b[5] + a[5]*b[2]));
            dd.set(2, 6, s*(a[2]*b[6] + a[6]*b[2]));
            dd.set(2, 7, s*(a[2]*b[7] + a[7]*b[2]));
            dd.set(2, 8, s*(a[2]*b[8] + a[8]*b[2]));

            dd.set(3, 0, s*(a[3]*b[0] + a[0]*b[3]));
            dd.set(3, 1, s*(a[3]*b[1] + a[1]*b[3]));
            dd.set(3, 2, s*(a[3]*b[2] + a[2]*b[3]));
            dd.set(3, 3, s*(2.0*a[3]*b[3]));
            dd.set(3, 4, s*(a[3]*b[4] + a[4]*b[3]));
            dd.set(3, 5, s*(a[3]*b[5] + a[5]*b[3]));
            dd.set(3, 6, s*(a[3]*b[6] + a[6]*b[3]));
            dd.set(3, 7, s*(a[3]*b[7] + a[7]*b[3]));
            dd.set(3, 8, s*(a[3]*b[8] + a[8]*b[3]));

            dd.set(4, 0, s*(a[4]*b[0] + a[0]*b[4]));
            dd.set(4, 1, s*(a[4]*b[1] + a[1]*b[4]));
            dd.set(4, 2, s*(a[4]*b[2] + a[2]*b[4]));
            dd.set(4, 3, s*(a[4]*b[3] + a[3]*b[4]));
            dd.set(4, 4, s*(2.0*a[4]*b[4]));
            dd.set(4, 5, s*(a[4]*b[5] + a[5]*b[4]));
            dd.set(4, 6, s*(a[4]*b[6] + a[6]*b[4]));
            dd.set(4, 7, s*(a[4]*b[7] + a[7]*b[4]));
            dd.set(4, 8, s*(a[4]*b[8] + a[8]*b[4]));

            dd.set(5, 0, s*(a[5]*b[0] + a[0]*b[5]));
            dd.set(5, 1, s*(a[5]*b[1] + a[1]*b[5]));
            dd.set(5, 2, s*(a[5]*b[2] + a[2]*b[5]));
            dd.set(5, 3, s*(a[5]*b[3] + a[3]*b[5]));
            dd.set(5, 4, s*(a[5]*b[4] + a[4]*b[5]));
            dd.set(5, 5, s*(2.0*a[5]*b[5]));
            dd.set(5, 6, s*(a[5]*b[6] + a[6]*b[5]));
            dd.set(5, 7, s*(a[5]*b[7] + a[7]*b[5]));
            dd.set(5, 8, s*(a[5]*b[8] + a[8]*b[5]));

            dd.set(6, 0, s*(a[6]*b[0] + a[0]*b[6]));
            dd.set(6, 1, s*(a[6]*b[1] + a[1]*b[6]));
            dd.set(6, 2, s*(a[6]*b[2] + a[2]*b[6]));
            dd.set(6, 3, s*(a[6]*b[3] + a[3]*b[6]));
            dd.set(6, 4, s*(a[6]*b[4] + a[4]*b[6]));
            dd.set(6, 5, s*(a[6]*b[5] + a[5]*b[6]));
            dd.set(6, 6, s*(2.0*a[6]*b[6]));
            dd.set(6, 7, s*(a[6]*b[7] + a[7]*b[6]));
            dd.set(6, 8, s*(a[6]*b[8] + a[8]*b[6]));

            dd.set(7, 0, s*(a[7]*b[0] + a[0]*b[7]));
            dd.set(7, 1, s*(a[7]*b[1] + a[1]*b[7]));
            dd.set(7, 2, s*(a[7]*b[2] + a[2]*b[7]));
            dd.set(7, 3, s*(a[7]*b[3] + a[3]*b[7]));
            dd.set(7, 4, s*(a[7]*b[4] + a[4]*b[7]));
            dd.set(7, 5, s*(a[7]*b[5] + a[5]*b[7]));
            dd.set(7, 6, s*(a[7]*b[6] + a[6]*b[7]));
            dd.set(7, 7, s*(2.0*a[7]*b[7]));
            dd.set(7, 8, s*(a[7]*b[8] + a[8]*b[7]));

            dd.set(8, 0, s*(a[8]*b[0] + a[0]*b[8]));
            dd.set(8, 1, s*(a[8]*b[1] + a[1]*b[8]));
            dd.set(8, 2, s*(a[8]*b[2] + a[2]*b[8]));
            dd.set(8, 3, s*(a[8]*b[3] + a[3]*b[8]));
            dd.set(8, 4, s*(a[8]*b[4] + a[4]*b[8]));
            dd.set(8, 5, s*(a[8]*b[5] + a[5]*b[8]));
            dd.set(8, 6, s*(a[8]*b[6] + a[6]*b[8]));
            dd.set(8, 7, s*(a[8]*b[7] + a[7]*b[8]));
            dd.set(8, 8, s*(2.0*a[8]*b[8]));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::dsd_fn;
    use crate::{ADD, MN_TO_IJKL, SET};
    use crate::{Tensor2, Tensor4};
    use russell_lab::{Matrix, mat_approx_eq};

    fn check_dsd<const N: usize>(s: f64, a_ten: &Tensor2<N>, b_ten: &Tensor2<N>, dd_ten: &Tensor4<N>, tol: f64) {
        let a = a_ten.as_std_matrix();
        let b = b_ten.as_std_matrix();
        let dd = dd_ten.as_std_matrix();
        let mut correct = Matrix::new(9, 9); // Use 9 here due to the conversion to "STD"
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * (a.get(i, j) * b.get(k, l) + b.get(i, j) * a.get(k, l)));
            }
        }
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn dsd_fn_works() {
        // general
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ]).unwrap();
        let mut dd = Tensor4::<9>::new();
        dsd_fn(&mut dd, SET, 2.0, &a, &b);
        check_dsd(2.0, &a, &b, &dd, 1e-13);

        // symmetric
        #[rustfmt::skip]
        let a = Tensor2::<6>::from_std_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<6>::from_std_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ]).unwrap();
        let mut dd = Tensor4::<6>::new();
        dsd_fn(&mut dd, SET, 2.0, &a, &b);
        check_dsd(2.0, &a, &b, &dd, 1e-13);

        // symmetric generalized plane
        #[rustfmt::skip]
        let a = Tensor2::<4>::from_std_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<4>::from_std_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]).unwrap();
        let mut dd = Tensor4::<4>::new();
        dsd_fn(&mut dd, SET, 2.0, &a, &b);
        check_dsd(2.0, &a, &b, &dd, 1e-14);
    }

    #[test]
    fn dsd_fn_add_works() {
        // general
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ]).unwrap();
        let mut dd = Tensor4::<9>::new();
        dsd_fn(&mut dd, SET, 2.0, &a, &b);
        dsd_fn(&mut dd, ADD, 3.0, &a, &b);
        check_dsd(5.0, &a, &b, &dd, 1e-12);

        // symmetric
        #[rustfmt::skip]
        let a = Tensor2::<6>::from_std_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<6>::from_std_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ]).unwrap();
        let mut dd = Tensor4::<6>::new();
        dsd_fn(&mut dd, SET, 2.0, &a, &b);
        dsd_fn(&mut dd, ADD, 3.0, &a, &b);
        check_dsd(5.0, &a, &b, &dd, 1e-12);

        // symmetric generalized plane
        #[rustfmt::skip]
        let a = Tensor2::<4>::from_std_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<4>::from_std_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]).unwrap();
        let mut dd = Tensor4::<4>::new();
        dsd_fn(&mut dd, SET, 2.0, &a, &b);
        dsd_fn(&mut dd, ADD, 3.0, &a, &b);
        check_dsd(5.0, &a, &b, &dd, 1e-12);
    }
}
