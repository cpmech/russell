use crate::{ADD, SQRT_2};
use crate::{Tensor2, Tensor4};

/// Performs the quad-sum-dyadic (qsd) operation with two Tensor2 yielding a minor-symmetric Tensor4
///
/// Computes:
///
/// ```text
///                _               _                               _               _
/// ADD: D += s (A ⊗ B + A ⊗ B + B ⊗ A + B ⊗ A)  or  SET: D = s (A ⊗ B + A ⊗ B + B ⊗ A + B ⊗ A)
///                        ‾               ‾                               ‾               ‾
/// ```
///
/// With Cartesian components (example with SET):
///
/// ```text
/// Dᵢⱼₖₗ = s (Aᵢₖ Bⱼₗ + Aᵢₗ Bⱼₖ + Bᵢₖ Aⱼₗ + Bᵢₗ Aⱼₖ)
/// ```
///
/// **Note:** For `N = 4` (symmetric generalized plane), only the `{00,11,22,01}` block of the
/// minor-symmetric result is computed; the unrepresented out-of-plane shear components are
/// omitted. See the reduced dimension and truncation (chop) strategy in the crate documentation.
///
/// # Output
///
/// * `dd` -- The resulting tensor (minor-symmetric)
///
/// # Input
///
/// * `op` -- operation: ADD or SET
/// * `aa` -- Second-order tensor, symmetric or not
/// * `bb` -- Second-order tensor, symmetric or not
pub fn qsd_fn<const N: usize>(dd: &mut Tensor4<N>, op: u8, s: f64, aa: &Tensor2<N>, bb: &Tensor2<N>) {
    qsd_fn_slice::<N>(dd, op, s, aa.as_data(), bb.as_data());
}

/// Internal (unrolled) quad-sum-dyadic operation on raw Kelvin-Mandel vectors.
#[rustfmt::skip]
#[inline]
pub(crate) fn qsd_fn_slice<const N: usize>(dd: &mut Tensor4<N>, op: u8, s: f64, a: &[f64], b: &[f64]) {
    if op == ADD {
        if N == 4 {
            dd.add(0, 0, s*(4.0*a[0]*b[0]));
            dd.add(0, 1, s*(2.0*a[3]*b[3]));
            dd.add(0, 2, 0.0);
            dd.add(0, 3, s*(2.0*(a[3]*b[0] + a[0]*b[3])));

            dd.add(1, 0, s*(2.0*a[3]*b[3]));
            dd.add(1, 1, s*(4.0*a[1]*b[1]));
            dd.add(1, 2, 0.0);
            dd.add(1, 3, s*(2.0*(a[3]*b[1] + a[1]*b[3])));

            dd.add(2, 0, 0.0);
            dd.add(2, 1, 0.0);
            dd.add(2, 2, s*(4.0*a[2]*b[2]));
            dd.add(2, 3, 0.0);

            dd.add(3, 0, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
            dd.add(3, 1, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
            dd.add(3, 2, 0.0);
            dd.add(3, 3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])));
        } else if N == 6 {
            dd.add(0, 0, s*(4.0*a[0]*b[0]));
            dd.add(0, 1, s*(2.0*a[3]*b[3]));
            dd.add(0, 2, s*(2.0*a[5]*b[5]));
            dd.add(0, 3, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
            dd.add(0, 4, s*(SQRT_2*(a[5]*b[3] + a[3]*b[5])));
            dd.add(0, 5, s*(2.0*(a[5]*b[0] + a[0]*b[5])));

            dd.add(1, 0, s*(2.0*a[3]*b[3]));
            dd.add(1, 1, s*(4.0*a[1]*b[1]));
            dd.add(1, 2, s*(2.0*a[4]*b[4]));
            dd.add(1, 3, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
            dd.add(1, 4, s*(2.0*(a[4]*b[1] + a[1]*b[4])));
            dd.add(1, 5, s*(SQRT_2*(a[4]*b[3] + a[3]*b[4])));

            dd.add(2, 0, s*(2.0*a[5]*b[5]));
            dd.add(2, 1, s*(2.0*a[4]*b[4]));
            dd.add(2, 2, s*(4.0*a[2]*b[2]));
            dd.add(2, 3, s*(SQRT_2*(a[5]*b[4] + a[4]*b[5])));
            dd.add(2, 4, s*(2.0*(a[4]*b[2] + a[2]*b[4])));
            dd.add(2, 5, s*(2.0*(a[5]*b[2] + a[2]*b[5])));

            dd.add(3, 0, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
            dd.add(3, 1, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
            dd.add(3, 2, s*(SQRT_2*(a[5]*b[4] + a[4]*b[5])));
            dd.add(3, 3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])));
            dd.add(3, 4, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5]));
            dd.add(3, 5, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5]));

            dd.add(4, 0, s*(SQRT_2*(a[5]*b[3] + a[3]*b[5])));
            dd.add(4, 1, s*(2.0*(a[4]*b[1] + a[1]*b[4])));
            dd.add(4, 2, s*(2.0*(a[4]*b[2] + a[2]*b[4])));
            dd.add(4, 3, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5]));
            dd.add(4, 4, s*(2.0*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])));
            dd.add(4, 5, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5]));

            dd.add(5, 0, s*(2.0*(a[5]*b[0] + a[0]*b[5])));
            dd.add(5, 1, s*(SQRT_2*(a[4]*b[3] + a[3]*b[4])));
            dd.add(5, 2, s*(2.0*(a[5]*b[2] + a[2]*b[5])));
            dd.add(5, 3, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5]));
            dd.add(5, 4, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5]));
            dd.add(5, 5, s*(2.0*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])));
        } else {
            debug_assert!(N == 9);

            dd.add(0, 0, s*(4.0*a[0]*b[0]));
            dd.add(0, 1, s*(2.0*(a[3] + a[6])*(b[3] + b[6])));
            dd.add(0, 2, s*(2.0*(a[5] + a[8])*(b[5] + b[8])));
            dd.add(0, 3, s*(2.0*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))));
            dd.add(0, 4, s*(SQRT_2*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))));
            dd.add(0, 5, s*(2.0*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))));

            dd.add(1, 0, s*(2.0*(a[3] - a[6])*(b[3] - b[6])));
            dd.add(1, 1, s*(4.0*a[1]*b[1]));
            dd.add(1, 2, s*(2.0*(a[4] + a[7])*(b[4] + b[7])));
            dd.add(1, 3, s*(2.0*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))));
            dd.add(1, 4, s*(2.0*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))));
            dd.add(1, 5, s*(SQRT_2*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))));

            dd.add(2, 0, s*(2.0*(a[5] - a[8])*(b[5] - b[8])));
            dd.add(2, 1, s*(2.0*(a[4] - a[7])*(b[4] - b[7])));
            dd.add(2, 2, s*(4.0*a[2]*b[2]));
            dd.add(2, 3, s*(SQRT_2*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))));
            dd.add(2, 4, s*(2.0*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))));
            dd.add(2, 5, s*(2.0*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))));

            dd.add(3, 0, s*(2.0*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))));
            dd.add(3, 1, s*(2.0*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))));
            dd.add(3, 2, s*(SQRT_2*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))));
            dd.add(3, 3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])));
            dd.add(3, 4, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8])));
            dd.add(3, 5, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8])));

            dd.add(4, 0, s*(SQRT_2*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))));
            dd.add(4, 1, s*(2.0*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))));
            dd.add(4, 2, s*(2.0*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))));
            dd.add(4, 3, s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8])));
            dd.add(4, 4, s*(2.0*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])));
            dd.add(4, 5, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8])));

            dd.add(5, 0, s*(2.0*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))));
            dd.add(5, 1, s*(SQRT_2*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))));
            dd.add(5, 2, s*(2.0*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))));
            dd.add(5, 3, s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8])));
            dd.add(5, 4, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8])));
            dd.add(5, 5, s*(2.0*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])));
        }
    } else {
        if N == 4 {
            dd.set(0, 0, s*(4.0*a[0]*b[0]));
            dd.set(0, 1, s*(2.0*a[3]*b[3]));
            dd.set(0, 2, 0.0);
            dd.set(0, 3, s*(2.0*(a[3]*b[0] + a[0]*b[3])));

            dd.set(1, 0, s*(2.0*a[3]*b[3]));
            dd.set(1, 1, s*(4.0*a[1]*b[1]));
            dd.set(1, 2, 0.0);
            dd.set(1, 3, s*(2.0*(a[3]*b[1] + a[1]*b[3])));

            dd.set(2, 0, 0.0);
            dd.set(2, 1, 0.0);
            dd.set(2, 2, s*(4.0*a[2]*b[2]));
            dd.set(2, 3, 0.0);

            dd.set(3, 0, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
            dd.set(3, 1, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
            dd.set(3, 2, 0.0);
            dd.set(3, 3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])));
        } else if N == 6 {
            dd.set(0, 0, s*(4.0*a[0]*b[0]));
            dd.set(0, 1, s*(2.0*a[3]*b[3]));
            dd.set(0, 2, s*(2.0*a[5]*b[5]));
            dd.set(0, 3, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
            dd.set(0, 4, s*(SQRT_2*(a[5]*b[3] + a[3]*b[5])));
            dd.set(0, 5, s*(2.0*(a[5]*b[0] + a[0]*b[5])));

            dd.set(1, 0, s*(2.0*a[3]*b[3]));
            dd.set(1, 1, s*(4.0*a[1]*b[1]));
            dd.set(1, 2, s*(2.0*a[4]*b[4]));
            dd.set(1, 3, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
            dd.set(1, 4, s*(2.0*(a[4]*b[1] + a[1]*b[4])));
            dd.set(1, 5, s*(SQRT_2*(a[4]*b[3] + a[3]*b[4])));

            dd.set(2, 0, s*(2.0*a[5]*b[5]));
            dd.set(2, 1, s*(2.0*a[4]*b[4]));
            dd.set(2, 2, s*(4.0*a[2]*b[2]));
            dd.set(2, 3, s*(SQRT_2*(a[5]*b[4] + a[4]*b[5])));
            dd.set(2, 4, s*(2.0*(a[4]*b[2] + a[2]*b[4])));
            dd.set(2, 5, s*(2.0*(a[5]*b[2] + a[2]*b[5])));

            dd.set(3, 0, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
            dd.set(3, 1, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
            dd.set(3, 2, s*(SQRT_2*(a[5]*b[4] + a[4]*b[5])));
            dd.set(3, 3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])));
            dd.set(3, 4, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5]));
            dd.set(3, 5, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5]));

            dd.set(4, 0, s*(SQRT_2*(a[5]*b[3] + a[3]*b[5])));
            dd.set(4, 1, s*(2.0*(a[4]*b[1] + a[1]*b[4])));
            dd.set(4, 2, s*(2.0*(a[4]*b[2] + a[2]*b[4])));
            dd.set(4, 3, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5]));
            dd.set(4, 4, s*(2.0*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])));
            dd.set(4, 5, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5]));

            dd.set(5, 0, s*(2.0*(a[5]*b[0] + a[0]*b[5])));
            dd.set(5, 1, s*(SQRT_2*(a[4]*b[3] + a[3]*b[4])));
            dd.set(5, 2, s*(2.0*(a[5]*b[2] + a[2]*b[5])));
            dd.set(5, 3, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5]));
            dd.set(5, 4, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5]));
            dd.set(5, 5, s*(2.0*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])));
        } else {
            debug_assert!(N == 9);

            dd.set(0, 0, s*(4.0*a[0]*b[0]));
            dd.set(0, 1, s*(2.0*(a[3] + a[6])*(b[3] + b[6])));
            dd.set(0, 2, s*(2.0*(a[5] + a[8])*(b[5] + b[8])));
            dd.set(0, 3, s*(2.0*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))));
            dd.set(0, 4, s*(SQRT_2*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))));
            dd.set(0, 5, s*(2.0*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))));

            dd.set(1, 0, s*(2.0*(a[3] - a[6])*(b[3] - b[6])));
            dd.set(1, 1, s*(4.0*a[1]*b[1]));
            dd.set(1, 2, s*(2.0*(a[4] + a[7])*(b[4] + b[7])));
            dd.set(1, 3, s*(2.0*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))));
            dd.set(1, 4, s*(2.0*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))));
            dd.set(1, 5, s*(SQRT_2*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))));

            dd.set(2, 0, s*(2.0*(a[5] - a[8])*(b[5] - b[8])));
            dd.set(2, 1, s*(2.0*(a[4] - a[7])*(b[4] - b[7])));
            dd.set(2, 2, s*(4.0*a[2]*b[2]));
            dd.set(2, 3, s*(SQRT_2*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))));
            dd.set(2, 4, s*(2.0*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))));
            dd.set(2, 5, s*(2.0*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))));

            dd.set(3, 0, s*(2.0*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))));
            dd.set(3, 1, s*(2.0*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))));
            dd.set(3, 2, s*(SQRT_2*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))));
            dd.set(3, 3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])));
            dd.set(3, 4, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8])));
            dd.set(3, 5, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8])));

            dd.set(4, 0, s*(SQRT_2*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))));
            dd.set(4, 1, s*(2.0*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))));
            dd.set(4, 2, s*(2.0*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))));
            dd.set(4, 3, s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8])));
            dd.set(4, 4, s*(2.0*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])));
            dd.set(4, 5, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8])));

            dd.set(5, 0, s*(2.0*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))));
            dd.set(5, 1, s*(SQRT_2*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))));
            dd.set(5, 2, s*(2.0*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))));
            dd.set(5, 3, s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8])));
            dd.set(5, 4, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8])));
            dd.set(5, 5, s*(2.0*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::qsd_fn;
    use crate::{ADD, IJ_TO_M_SYM, MN_TO_IJKL, SET};
    use crate::{Tensor2, Tensor4};
    use russell_lab::{Matrix, mat_approx_eq};

    // Zeroes the entries of a 9x9 standard matrix that are not represented by a Tensor4<4>
    fn zero_unrepresented_shears(mat: &mut Matrix) {
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                if IJ_TO_M_SYM[i][j] >= 4 || IJ_TO_M_SYM[k][l] >= 4 {
                    mat.set(m, n, 0.0);
                }
            }
        }
    }

    fn check_qsd<const N: usize>(s: f64, a_ten: &Tensor2<N>, b_ten: &Tensor2<N>, dd_ten: &Tensor4<N>, tol: f64) {
        let a = a_ten.as_std_matrix();
        let b = b_ten.as_std_matrix();
        let dd = dd_ten.as_std_matrix();
        let mut correct = Matrix::new(9, 9); // Use 9 here due to the conversion to "STD"
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(
                    m,
                    n,
                    s * (a.get(i, k) * b.get(j, l)
                        + a.get(i, l) * b.get(j, k)
                        + b.get(i, k) * a.get(j, l)
                        + b.get(i, l) * a.get(j, k)),
                );
            }
        }
        if N == 4 {
            zero_unrepresented_shears(&mut correct);
        }
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn qsd_fn_works() {
        // general qsd general
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
        qsd_fn(&mut dd, SET, 2.0, &a, &b);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
            [72.0, 128.0, 168.0, 104.0, 152.0, 136.0, 104.0, 152.0, 136.0],
            [192.0, 200.0, 192.0, 200.0, 200.0, 208.0, 200.0, 200.0, 208.0],
            [168.0, 128.0, 72.0, 152.0, 104.0, 136.0, 152.0, 104.0, 136.0],
            [168.0, 200.0, 216.0, 188.0, 212.0, 208.0, 188.0, 212.0, 208.0],
            [216.0, 200.0, 168.0, 212.0, 188.0, 208.0, 212.0, 188.0, 208.0],
            [264.0, 272.0, 264.0, 272.0, 272.0, 280.0, 272.0, 272.0, 280.0],
            [168.0, 200.0, 216.0, 188.0, 212.0, 208.0, 188.0, 212.0, 208.0],
            [216.0, 200.0, 168.0, 212.0, 188.0, 208.0, 212.0, 188.0, 208.0],
            [264.0, 272.0, 264.0, 272.0, 272.0, 280.0, 272.0, 272.0, 280.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_qsd(2.0, &a, &b, &dd, 1e-13);

        // symmetric qsd symmetric
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
        qsd_fn(&mut dd, SET, 2.0, &a, &b);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
            [24.0, 160.0, 288.0, 68.0, 216.0, 96.0, 68.0, 216.0, 96.0],
            [160.0, 32.0, 160.0, 72.0, 72.0, 164.0, 72.0, 72.0, 164.0],
            [288.0, 160.0, 24.0, 216.0, 68.0, 96.0, 216.0, 68.0, 96.0],
            [68.0, 72.0, 216.0, 96.0, 130.0, 146.0, 96.0, 130.0, 146.0],
            [216.0, 72.0, 68.0, 130.0, 96.0, 146.0, 130.0, 96.0, 146.0],
            [96.0, 164.0, 96.0, 146.0, 146.0, 164.0, 146.0, 146.0, 164.0],
            [68.0, 72.0, 216.0, 96.0, 130.0, 146.0, 96.0, 130.0, 146.0],
            [216.0, 72.0, 68.0, 130.0, 96.0, 146.0, 130.0, 96.0, 146.0],
            [96.0, 164.0, 96.0, 146.0, 146.0, 164.0, 146.0, 146.0, 164.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_qsd(2.0, &a, &b, &dd, 1e-13);

        // symmetric generalized plane qsd symmetric generalized plane
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
        qsd_fn(&mut dd, SET, 2.0, &a, &b);
        let mat = dd.as_std_matrix();
        let mut correct = Matrix::from(&[
            [24.0, 128.0, 0.0, 64.0, 0.0, 0.0, 64.0, 0.0, 0.0],
            [128.0, 32.0, 0.0, 64.0, 0.0, 0.0, 64.0, 0.0, 0.0],
            [0.0, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [64.0, 64.0, 0.0, 80.0, 0.0, 0.0, 80.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 16.0, 32.0, 0.0, 16.0, 32.0],
            [0.0, 0.0, 0.0, 0.0, 32.0, 20.0, 0.0, 32.0, 20.0],
            [64.0, 64.0, 0.0, 80.0, 0.0, 0.0, 80.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 16.0, 32.0, 0.0, 16.0, 32.0],
            [0.0, 0.0, 0.0, 0.0, 32.0, 20.0, 0.0, 32.0, 20.0],
        ]);
        zero_unrepresented_shears(&mut correct);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_qsd(2.0, &a, &b, &dd, 1e-14);
    }

    #[test]
    fn qsd_fn_add_works() {
        // general qsd general
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
        qsd_fn(&mut dd, SET, 2.0, &a, &b);
        qsd_fn(&mut dd, ADD, 3.0, &a, &b);
        check_qsd(5.0, &a, &b, &dd, 1e-12);

        // symmetric qsd symmetric
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
        qsd_fn(&mut dd, SET, 2.0, &a, &b);
        qsd_fn(&mut dd, ADD, 3.0, &a, &b);
        check_qsd(5.0, &a, &b, &dd, 1e-12);

        // symmetric generalized plane qsd symmetric generalized plane
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
        qsd_fn(&mut dd, SET, 2.0, &a, &b);
        qsd_fn(&mut dd, ADD, 3.0, &a, &b);
        check_qsd(5.0, &a, &b, &dd, 1e-12);
    }
}
