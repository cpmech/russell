use super::{Tensor2, Tensor4};
use crate::{Rep, SQRT_2};

/// Performs the overbar dyadic product between two Tensor2 resulting in a (general) Tensor4
///
/// Computes:
///
/// ```text
///         _
/// D = s A ⊗ B
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ = s Aᵢₖ Bⱼₗ
/// ```
///
/// **Important:** The result is **not** necessarily minor-symmetric; therefore `D` must be General.
///
/// # Output
///
/// * `dd` -- the tensor `D`; it must be [Rep::General]
///
/// # Input
///
/// * `a` -- first tensor; with the same [Rep] as `b`
/// * `b` -- second tensor; with the same [Rep] as `a`
///
/// # Panics
///
/// 1. A panic will occur if `dd` is not [Rep::General]
/// 2. A panic will occur if `a` and `b` have different [Rep]
#[inline]
pub fn t2_odyad_t2(dd: &mut Tensor4, s: f64, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(dd.rep(), Rep::General);
    assert_eq!(bb.rep(), aa.rep());
    t2_odyad_t2_slice(dd, s, aa.as_data(), bb.as_data(), aa.dim());
}

/// Internal (unrolled) overbar dyadic product on raw Kelvin-Mandel vectors.
#[rustfmt::skip]
#[inline]
pub(crate) fn t2_odyad_t2_slice(dd: &mut Tensor4, s: f64, a: &[f64], b: &[f64], dim: usize) {
    let tsq2 = 2.0 * SQRT_2;
    if dim == 4 {
        let a = &a[..4];
        let b = &b[..4];
        dd.set(0, 0, s*a[0]*b[0]);
        dd.set(0, 1, s*(a[3]*b[3])/2.0);
        dd.set(0, 2, 0.0);
        dd.set(0, 3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(0, 4, 0.0);
        dd.set(0, 5, 0.0);
        dd.set(0, 6, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.set(0, 7, 0.0);
        dd.set(0, 8, 0.0);

        dd.set(1, 0, s*(a[3]*b[3])/2.0);
        dd.set(1, 1, s*a[1]*b[1]);
        dd.set(1, 2, 0.0);
        dd.set(1, 3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(1, 4, 0.0);
        dd.set(1, 5, 0.0);
        dd.set(1, 6, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.set(1, 7, 0.0);
        dd.set(1, 8, 0.0);

        dd.set(2, 0, 0.0);
        dd.set(2, 1, 0.0);
        dd.set(2, 2, s*a[2]*b[2]);
        dd.set(2, 3, 0.0);
        dd.set(2, 4, 0.0);
        dd.set(2, 5, 0.0);
        dd.set(2, 6, 0.0);
        dd.set(2, 7, 0.0);
        dd.set(2, 8, 0.0);

        dd.set(3, 0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(3, 1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(3, 2, 0.0);
        dd.set(3, 3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.set(3, 4, 0.0);
        dd.set(3, 5, 0.0);
        dd.set(3, 6, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.set(3, 7, 0.0);
        dd.set(3, 8, 0.0);

        dd.set(4, 0, 0.0);
        dd.set(4, 1, 0.0);
        dd.set(4, 2, 0.0);
        dd.set(4, 3, 0.0);
        dd.set(4, 4, s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.set(4, 5, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.set(4, 6, 0.0);
        dd.set(4, 7, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.set(4, 8, s*(a[3]*b[2] - a[2]*b[3])/tsq2);

        dd.set(5, 0, 0.0);
        dd.set(5, 1, 0.0);
        dd.set(5, 2, 0.0);
        dd.set(5, 3, 0.0);
        dd.set(5, 4, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.set(5, 5, s*(a[2]*b[0] + a[0]*b[2])/2.0);
        dd.set(5, 6, 0.0);
        dd.set(5, 7, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.set(5, 8, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);

        dd.set(6, 0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.set(6, 1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.set(6, 2, 0.0);
        dd.set(6, 3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.set(6, 4, 0.0);
        dd.set(6, 5, 0.0);
        dd.set(6, 6, s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3])/2.0);
        dd.set(6, 7, 0.0);
        dd.set(6, 8, 0.0);

        dd.set(7, 0, 0.0);
        dd.set(7, 1, 0.0);
        dd.set(7, 2, 0.0);
        dd.set(7, 3, 0.0);
        dd.set(7, 4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.set(7, 5, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.set(7, 6, 0.0);
        dd.set(7, 7, s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.set(7, 8, s*(a[3]*b[2] + a[2]*b[3])/tsq2);

        dd.set(8, 0, 0.0);
        dd.set(8, 1, 0.0);
        dd.set(8, 2, 0.0);
        dd.set(8, 3, 0.0);
        dd.set(8, 4, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.set(8, 5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.set(8, 6, 0.0);
        dd.set(8, 7, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.set(8, 8, s*(a[2]*b[0] + a[0]*b[2])/2.0);
    } else if dim == 6 {
        let a = &a[..6];
        let b = &b[..6];
        dd.set(0, 0, s*a[0]*b[0]);
        dd.set(0, 1, s*(a[3]*b[3])/2.0);
        dd.set(0, 2, s*(a[5]*b[5])/2.0);
        dd.set(0, 3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(0, 4, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.set(0, 5, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.set(0, 6, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.set(0, 7, s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.set(0, 8, s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);

        dd.set(1, 0, s*(a[3]*b[3])/2.0);
        dd.set(1, 1, s*a[1]*b[1]);
        dd.set(1, 2, s*(a[4]*b[4])/2.0);
        dd.set(1, 3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(1, 4, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.set(1, 5, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.set(1, 6, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.set(1, 7, s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.set(1, 8, s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);

        dd.set(2, 0, s*(a[5]*b[5])/2.0);
        dd.set(2, 1, s*(a[4]*b[4])/2.0);
        dd.set(2, 2, s*a[2]*b[2]);
        dd.set(2, 3, s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.set(2, 4, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.set(2, 5, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.set(2, 6, s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.set(2, 7, s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.set(2, 8, s*(a[5]*b[2] - a[2]*b[5])/2.0);

        dd.set(3, 0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(3, 1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(3, 2, s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.set(3, 3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.set(3, 4, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(3, 5, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(3, 6, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.set(3, 7, s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(3, 8, s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);

        dd.set(4, 0, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.set(4, 1, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.set(4, 2, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.set(4, 3, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(4, 4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])/2.0);
        dd.set(4, 5, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(4, 6, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.set(4, 7, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.set(4, 8, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);

        dd.set(5, 0, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.set(5, 1, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.set(5, 2, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.set(5, 3, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(5, 4, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(5, 5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])/2.0);
        dd.set(5, 6, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.set(5, 7, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(5, 8, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);

        dd.set(6, 0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.set(6, 1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.set(6, 2, s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.set(6, 3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.set(6, 4, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.set(6, 5, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.set(6, 6, s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3])/2.0);
        dd.set(6, 7, s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.set(6, 8, s*(SQRT_2*a[4]*b[0] - a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);

        dd.set(7, 0, s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.set(7, 1, s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.set(7, 2, s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.set(7, 3, s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(7, 4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.set(7, 5, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(7, 6, s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.set(7, 7, s*(a[2]*b[1] + a[1]*b[2] - a[4]*b[4])/2.0);
        dd.set(7, 8, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] - a[5]*b[4] - a[4]*b[5])/4.0);

        dd.set(8, 0, s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);
        dd.set(8, 1, s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);
        dd.set(8, 2, s*(a[5]*b[2] - a[2]*b[5])/2.0);
        dd.set(8, 3, s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(8, 4, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
        dd.set(8, 5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.set(8, 6, s*(SQRT_2*a[4]*b[0] - a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.set(8, 7, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] - a[5]*b[4] - a[4]*b[5])/4.0);
        dd.set(8, 8, s*(a[2]*b[0] + a[0]*b[2] - a[5]*b[5])/2.0);
    } else {
        let a = &a[..9];
        let b = &b[..9];
        dd.set(0, 0, s*a[0]*b[0]);
        dd.set(0, 1, s*((a[3] + a[6])*(b[3] + b[6]))/2.0);
        dd.set(0, 2, s*((a[5] + a[8])*(b[5] + b[8]))/2.0);
        dd.set(0, 3, s*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.set(0, 4, s*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.set(0, 5, s*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);
        dd.set(0, 6, s*(-(a[3]*b[0]) - a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.set(0, 7, s*(-((a[5] + a[8])*(b[3] + b[6])) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.set(0, 8, s*(-(a[5]*b[0]) - a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);

        dd.set(1, 0, s*((a[3] - a[6])*(b[3] - b[6]))/2.0);
        dd.set(1, 1, s*a[1]*b[1]);
        dd.set(1, 2, s*((a[4] + a[7])*(b[4] + b[7]))/2.0);
        dd.set(1, 3, s*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
        dd.set(1, 4, s*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.set(1, 5, s*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);
        dd.set(1, 6, s*(a[3]*b[1] - a[6]*b[1] + a[1]*(-b[3] + b[6]))/2.0);
        dd.set(1, 7, s*(-(a[4]*b[1]) - a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.set(1, 8, s*(-((a[4] + a[7])*(b[3] - b[6])) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);

        dd.set(2, 0, s*((a[5] - a[8])*(b[5] - b[8]))/2.0);
        dd.set(2, 1, s*((a[4] - a[7])*(b[4] - b[7]))/2.0);
        dd.set(2, 2, s*a[2]*b[2]);
        dd.set(2, 3, s*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.set(2, 4, s*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
        dd.set(2, 5, s*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);
        dd.set(2, 6, s*((a[5] - a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.set(2, 7, s*(a[4]*b[2] - a[7]*b[2] + a[2]*(-b[4] + b[7]))/2.0);
        dd.set(2, 8, s*(a[5]*b[2] - a[8]*b[2] + a[2]*(-b[5] + b[8]))/2.0);

        dd.set(3, 0, s*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.set(3, 1, s*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))/2.0);
        dd.set(3, 2, s*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.set(3, 3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
        dd.set(3, 4, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(3, 5, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.set(3, 6, s*(-(a[1]*b[0]) + a[0]*b[1] - a[6]*b[3] + a[3]*b[6])/2.0);
        dd.set(3, 7, s*(-(SQRT_2*(a[5] + a[8])*b[1]) - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(3, 8, s*(-(SQRT_2*(a[4] + a[7])*b[0]) - (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.set(4, 0, s*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.set(4, 1, s*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.set(4, 2, s*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))/2.0);
        dd.set(4, 3, s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(4, 4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
        dd.set(4, 5, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.set(4, 6, s*(SQRT_2*(a[5] - a[8])*b[1] - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) - SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(4, 7, s*(-(a[2]*b[1]) + a[1]*b[2] - a[7]*b[4] + a[4]*b[7])/2.0);
        dd.set(4, 8, s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.set(5, 0, s*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.set(5, 1, s*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.set(5, 2, s*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))/2.0);
        dd.set(5, 3, s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(5, 4, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(5, 5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
        dd.set(5, 6, s*(-(SQRT_2*(a[4] - a[7])*b[0]) + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) - (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(5, 7, s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) - (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(5, 8, s*(-(a[2]*b[0]) + a[0]*b[2] - a[8]*b[5] + a[5]*b[8])/2.0);

        dd.set(6, 0, s*(-(a[3]*b[0]) + a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.set(6, 1, s*(a[3]*b[1] + a[6]*b[1] - a[1]*(b[3] + b[6]))/2.0);
        dd.set(6, 2, s*((a[5] + a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.set(6, 3, s*(-(a[1]*b[0]) + a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
        dd.set(6, 4, s*(SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(6, 5, s*(-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.set(6, 6, s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3] + a[6]*b[6])/2.0);
        dd.set(6, 7, s*(-(SQRT_2*(a[5] + a[8])*b[1]) + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(6, 8, s*(SQRT_2*(a[4] + a[7])*b[0] - (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.set(7, 0, s*(-((a[5] - a[8])*(b[3] - b[6])) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.set(7, 1, s*(-(a[4]*b[1]) + a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.set(7, 2, s*(a[4]*b[2] + a[7]*b[2] - a[2]*(b[4] + b[7]))/2.0);
        dd.set(7, 3, s*(-(SQRT_2*(a[5] - a[8])*b[1]) - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(7, 4, s*(-(a[2]*b[1]) + a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
        dd.set(7, 5, s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.set(7, 6, s*(-(SQRT_2*(a[5] - a[8])*b[1]) + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) - SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(7, 7, s*(a[2]*b[1] + a[1]*b[2] - a[4]*b[4] + a[7]*b[7])/2.0);
        dd.set(7, 8, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.set(8, 0, s*(-(a[5]*b[0]) + a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.set(8, 1, s*(-((a[4] - a[7])*(b[3] + b[6])) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.set(8, 2, s*(a[5]*b[2] + a[8]*b[2] - a[2]*(b[5] + b[8]))/2.0);
        dd.set(8, 3, s*(-(SQRT_2*(a[4] - a[7])*b[0]) - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(8, 4, s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(8, 5, s*(-(a[2]*b[0]) + a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);
        dd.set(8, 6, s*(SQRT_2*(a[4] - a[7])*b[0] - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) - (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(8, 7, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) - (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(8, 8, s*(a[2]*b[0] + a[0]*b[2] - a[5]*b[5] + a[8]*b[8])/2.0);
    }
}

/// Internal (unrolled) overbar dyadic product (accumulate) on raw Kelvin-Mandel vectors.
///
/// Computes `dd += s (A ⊗ B)`.
#[rustfmt::skip]
#[inline]
pub(crate) fn t2_odyad_t2_update_slice(dd: &mut Tensor4, s: f64, a: &[f64], b: &[f64], dim: usize) {
    let tsq2 = 2.0 * SQRT_2;
    if dim == 4 {
        let a = &a[..4];
        let b = &b[..4];
        dd.set(0, 0, dd.get(0, 0) + s*a[0]*b[0]);
        dd.set(0, 1, dd.get(0, 1) + s*(a[3]*b[3])/2.0);
        dd.set(0, 3, dd.get(0, 3) + s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(0, 6, dd.get(0, 6) + s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);

        dd.set(1, 0, dd.get(1, 0) + s*(a[3]*b[3])/2.0);
        dd.set(1, 1, dd.get(1, 1) + s*a[1]*b[1]);
        dd.set(1, 3, dd.get(1, 3) + s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(1, 6, dd.get(1, 6) + s*(a[3]*b[1] - a[1]*b[3])/2.0);

        dd.set(2, 2, dd.get(2, 2) + s*a[2]*b[2]);

        dd.set(3, 0, dd.get(3, 0) + s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(3, 1, dd.get(3, 1) + s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(3, 3, dd.get(3, 3) + s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.set(3, 6, dd.get(3, 6) + s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);

        dd.set(4, 4, dd.get(4, 4) + s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.set(4, 5, dd.get(4, 5) + s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.set(4, 7, dd.get(4, 7) + s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.set(4, 8, dd.get(4, 8) + s*(a[3]*b[2] - a[2]*b[3])/tsq2);

        dd.set(5, 4, dd.get(5, 4) + s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.set(5, 5, dd.get(5, 5) + s*(a[2]*b[0] + a[0]*b[2])/2.0);
        dd.set(5, 7, dd.get(5, 7) + s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.set(5, 8, dd.get(5, 8) + s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);

        dd.set(6, 0, dd.get(6, 0) + s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.set(6, 1, dd.get(6, 1) + s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.set(6, 3, dd.get(6, 3) + s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.set(6, 6, dd.get(6, 6) + s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3])/2.0);

        dd.set(7, 4, dd.get(7, 4) + s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.set(7, 5, dd.get(7, 5) + s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.set(7, 7, dd.get(7, 7) + s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.set(7, 8, dd.get(7, 8) + s*(a[3]*b[2] + a[2]*b[3])/tsq2);

        dd.set(8, 4, dd.get(8, 4) + s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.set(8, 5, dd.get(8, 5) + s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.set(8, 7, dd.get(8, 7) + s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.set(8, 8, dd.get(8, 8) + s*(a[2]*b[0] + a[0]*b[2])/2.0);
    } else if dim == 6 {
        let a = &a[..6];
        let b = &b[..6];
        dd.set(0, 0, dd.get(0, 0) + s*a[0]*b[0]);
        dd.set(0, 1, dd.get(0, 1) + s*(a[3]*b[3])/2.0);
        dd.set(0, 2, dd.get(0, 2) + s*(a[5]*b[5])/2.0);
        dd.set(0, 3, dd.get(0, 3) + s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(0, 4, dd.get(0, 4) + s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.set(0, 5, dd.get(0, 5) + s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.set(0, 6, dd.get(0, 6) + s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.set(0, 7, dd.get(0, 7) + s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.set(0, 8, dd.get(0, 8) + s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);

        dd.set(1, 0, dd.get(1, 0) + s*(a[3]*b[3])/2.0);
        dd.set(1, 1, dd.get(1, 1) + s*a[1]*b[1]);
        dd.set(1, 2, dd.get(1, 2) + s*(a[4]*b[4])/2.0);
        dd.set(1, 3, dd.get(1, 3) + s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(1, 4, dd.get(1, 4) + s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.set(1, 5, dd.get(1, 5) + s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.set(1, 6, dd.get(1, 6) + s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.set(1, 7, dd.get(1, 7) + s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.set(1, 8, dd.get(1, 8) + s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);

        dd.set(2, 0, dd.get(2, 0) + s*(a[5]*b[5])/2.0);
        dd.set(2, 1, dd.get(2, 1) + s*(a[4]*b[4])/2.0);
        dd.set(2, 2, dd.get(2, 2) + s*a[2]*b[2]);
        dd.set(2, 3, dd.get(2, 3) + s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.set(2, 4, dd.get(2, 4) + s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.set(2, 5, dd.get(2, 5) + s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.set(2, 6, dd.get(2, 6) + s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.set(2, 7, dd.get(2, 7) + s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.set(2, 8, dd.get(2, 8) + s*(a[5]*b[2] - a[2]*b[5])/2.0);

        dd.set(3, 0, dd.get(3, 0) + s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(3, 1, dd.get(3, 1) + s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(3, 2, dd.get(3, 2) + s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.set(3, 3, dd.get(3, 3) + s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.set(3, 4, dd.get(3, 4) + s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(3, 5, dd.get(3, 5) + s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(3, 6, dd.get(3, 6) + s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.set(3, 7, dd.get(3, 7) + s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(3, 8, dd.get(3, 8) + s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);

        dd.set(4, 0, dd.get(4, 0) + s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.set(4, 1, dd.get(4, 1) + s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.set(4, 2, dd.get(4, 2) + s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.set(4, 3, dd.get(4, 3) + s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(4, 4, dd.get(4, 4) + s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])/2.0);
        dd.set(4, 5, dd.get(4, 5) + s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(4, 6, dd.get(4, 6) + s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.set(4, 7, dd.get(4, 7) + s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.set(4, 8, dd.get(4, 8) + s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);

        dd.set(5, 0, dd.get(5, 0) + s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.set(5, 1, dd.get(5, 1) + s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.set(5, 2, dd.get(5, 2) + s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.set(5, 3, dd.get(5, 3) + s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(5, 4, dd.get(5, 4) + s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(5, 5, dd.get(5, 5) + s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])/2.0);
        dd.set(5, 6, dd.get(5, 6) + s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.set(5, 7, dd.get(5, 7) + s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(5, 8, dd.get(5, 8) + s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);

        dd.set(6, 0, dd.get(6, 0) + s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.set(6, 1, dd.get(6, 1) + s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.set(6, 2, dd.get(6, 2) + s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.set(6, 3, dd.get(6, 3) + s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.set(6, 4, dd.get(6, 4) + s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.set(6, 5, dd.get(6, 5) + s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.set(6, 6, dd.get(6, 6) + s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3])/2.0);
        dd.set(6, 7, dd.get(6, 7) + s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.set(6, 8, dd.get(6, 8) + s*(SQRT_2*a[4]*b[0] - a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);

        dd.set(7, 0, dd.get(7, 0) + s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.set(7, 1, dd.get(7, 1) + s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.set(7, 2, dd.get(7, 2) + s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.set(7, 3, dd.get(7, 3) + s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(7, 4, dd.get(7, 4) + s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.set(7, 5, dd.get(7, 5) + s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(7, 6, dd.get(7, 6) + s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.set(7, 7, dd.get(7, 7) + s*(a[2]*b[1] + a[1]*b[2] - a[4]*b[4])/2.0);
        dd.set(7, 8, dd.get(7, 8) + s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] - a[5]*b[4] - a[4]*b[5])/4.0);

        dd.set(8, 0, dd.get(8, 0) + s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);
        dd.set(8, 1, dd.get(8, 1) + s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);
        dd.set(8, 2, dd.get(8, 2) + s*(a[5]*b[2] - a[2]*b[5])/2.0);
        dd.set(8, 3, dd.get(8, 3) + s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(8, 4, dd.get(8, 4) + s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
        dd.set(8, 5, dd.get(8, 5) + s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.set(8, 6, dd.get(8, 6) + s*(SQRT_2*a[4]*b[0] - a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.set(8, 7, dd.get(8, 7) + s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] - a[5]*b[4] - a[4]*b[5])/4.0);
        dd.set(8, 8, dd.get(8, 8) + s*(a[2]*b[0] + a[0]*b[2] - a[5]*b[5])/2.0);
    } else {
        let a = &a[..9];
        let b = &b[..9];
        dd.set(0, 0, dd.get(0, 0) + s*a[0]*b[0]);
        dd.set(0, 1, dd.get(0, 1) + s*((a[3] + a[6])*(b[3] + b[6]))/2.0);
        dd.set(0, 2, dd.get(0, 2) + s*((a[5] + a[8])*(b[5] + b[8]))/2.0);
        dd.set(0, 3, dd.get(0, 3) + s*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.set(0, 4, dd.get(0, 4) + s*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.set(0, 5, dd.get(0, 5) + s*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);
        dd.set(0, 6, dd.get(0, 6) + s*(-(a[3]*b[0]) - a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.set(0, 7, dd.get(0, 7) + s*(-((a[5] + a[8])*(b[3] + b[6])) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.set(0, 8, dd.get(0, 8) + s*(-(a[5]*b[0]) - a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);

        dd.set(1, 0, dd.get(1, 0) + s*((a[3] - a[6])*(b[3] - b[6]))/2.0);
        dd.set(1, 1, dd.get(1, 1) + s*a[1]*b[1]);
        dd.set(1, 2, dd.get(1, 2) + s*((a[4] + a[7])*(b[4] + b[7]))/2.0);
        dd.set(1, 3, dd.get(1, 3) + s*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
        dd.set(1, 4, dd.get(1, 4) + s*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.set(1, 5, dd.get(1, 5) + s*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);
        dd.set(1, 6, dd.get(1, 6) + s*(a[3]*b[1] - a[6]*b[1] + a[1]*(-b[3] + b[6]))/2.0);
        dd.set(1, 7, dd.get(1, 7) + s*(-(a[4]*b[1]) - a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.set(1, 8, dd.get(1, 8) + s*(-((a[4] + a[7])*(b[3] - b[6])) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);

        dd.set(2, 0, dd.get(2, 0) + s*((a[5] - a[8])*(b[5] - b[8]))/2.0);
        dd.set(2, 1, dd.get(2, 1) + s*((a[4] - a[7])*(b[4] - b[7]))/2.0);
        dd.set(2, 2, dd.get(2, 2) + s*a[2]*b[2]);
        dd.set(2, 3, dd.get(2, 3) + s*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.set(2, 4, dd.get(2, 4) + s*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
        dd.set(2, 5, dd.get(2, 5) + s*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);
        dd.set(2, 6, dd.get(2, 6) + s*((a[5] - a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.set(2, 7, dd.get(2, 7) + s*(a[4]*b[2] - a[7]*b[2] + a[2]*(-b[4] + b[7]))/2.0);
        dd.set(2, 8, dd.get(2, 8) + s*(a[5]*b[2] - a[8]*b[2] + a[2]*(-b[5] + b[8]))/2.0);

        dd.set(3, 0, dd.get(3, 0) + s*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.set(3, 1, dd.get(3, 1) + s*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))/2.0);
        dd.set(3, 2, dd.get(3, 2) + s*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.set(3, 3, dd.get(3, 3) + s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
        dd.set(3, 4, dd.get(3, 4) + s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(3, 5, dd.get(3, 5) + s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.set(3, 6, dd.get(3, 6) + s*(-(a[1]*b[0]) + a[0]*b[1] - a[6]*b[3] + a[3]*b[6])/2.0);
        dd.set(3, 7, dd.get(3, 7) + s*(-(SQRT_2*(a[5] + a[8])*b[1]) - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(3, 8, dd.get(3, 8) + s*(-(SQRT_2*(a[4] + a[7])*b[0]) - (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.set(4, 0, dd.get(4, 0) + s*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.set(4, 1, dd.get(4, 1) + s*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.set(4, 2, dd.get(4, 2) + s*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))/2.0);
        dd.set(4, 3, dd.get(4, 3) + s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(4, 4, dd.get(4, 4) + s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
        dd.set(4, 5, dd.get(4, 5) + s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.set(4, 6, dd.get(4, 6) + s*(SQRT_2*(a[5] - a[8])*b[1] - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) - SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(4, 7, dd.get(4, 7) + s*(-(a[2]*b[1]) + a[1]*b[2] - a[7]*b[4] + a[4]*b[7])/2.0);
        dd.set(4, 8, dd.get(4, 8) + s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.set(5, 0, dd.get(5, 0) + s*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.set(5, 1, dd.get(5, 1) + s*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.set(5, 2, dd.get(5, 2) + s*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))/2.0);
        dd.set(5, 3, dd.get(5, 3) + s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(5, 4, dd.get(5, 4) + s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(5, 5, dd.get(5, 5) + s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
        dd.set(5, 6, dd.get(5, 6) + s*(-(SQRT_2*(a[4] - a[7])*b[0]) + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) - (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(5, 7, dd.get(5, 7) + s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) - (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(5, 8, dd.get(5, 8) + s*(-(a[2]*b[0]) + a[0]*b[2] - a[8]*b[5] + a[5]*b[8])/2.0);

        dd.set(6, 0, dd.get(6, 0) + s*(-(a[3]*b[0]) + a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.set(6, 1, dd.get(6, 1) + s*(a[3]*b[1] + a[6]*b[1] - a[1]*(b[3] + b[6]))/2.0);
        dd.set(6, 2, dd.get(6, 2) + s*((a[5] + a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.set(6, 3, dd.get(6, 3) + s*(-(a[1]*b[0]) + a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
        dd.set(6, 4, dd.get(6, 4) + s*(SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(6, 5, dd.get(6, 5) + s*(-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.set(6, 6, dd.get(6, 6) + s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3] + a[6]*b[6])/2.0);
        dd.set(6, 7, dd.get(6, 7) + s*(-(SQRT_2*(a[5] + a[8])*b[1]) + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(6, 8, dd.get(6, 8) + s*(SQRT_2*(a[4] + a[7])*b[0] - (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.set(7, 0, dd.get(7, 0) + s*(-((a[5] - a[8])*(b[3] - b[6])) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.set(7, 1, dd.get(7, 1) + s*(-(a[4]*b[1]) + a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.set(7, 2, dd.get(7, 2) + s*(a[4]*b[2] + a[7]*b[2] - a[2]*(b[4] + b[7]))/2.0);
        dd.set(7, 3, dd.get(7, 3) + s*(-(SQRT_2*(a[5] - a[8])*b[1]) - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(7, 4, dd.get(7, 4) + s*(-(a[2]*b[1]) + a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
        dd.set(7, 5, dd.get(7, 5) + s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.set(7, 6, dd.get(7, 6) + s*(-(SQRT_2*(a[5] - a[8])*b[1]) + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) - SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(7, 7, dd.get(7, 7) + s*(a[2]*b[1] + a[1]*b[2] - a[4]*b[4] + a[7]*b[7])/2.0);
        dd.set(7, 8, dd.get(7, 8) + s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.set(8, 0, dd.get(8, 0) + s*(-(a[5]*b[0]) + a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.set(8, 1, dd.get(8, 1) + s*(-((a[4] - a[7])*(b[3] + b[6])) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.set(8, 2, dd.get(8, 2) + s*(a[5]*b[2] + a[8]*b[2] - a[2]*(b[5] + b[8]))/2.0);
        dd.set(8, 3, dd.get(8, 3) + s*(-(SQRT_2*(a[4] - a[7])*b[0]) - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(8, 4, dd.get(8, 4) + s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(8, 5, dd.get(8, 5) + s*(-(a[2]*b[0]) + a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);
        dd.set(8, 6, dd.get(8, 6) + s*(SQRT_2*(a[4] - a[7])*b[0] - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) - (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(8, 7, dd.get(8, 7) + s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) - (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(8, 8, dd.get(8, 8) + s*(a[2]*b[0] + a[0]*b[2] - a[5]*b[5] + a[8]*b[8])/2.0);
    }
}

/// Performs the underbar dyadic product between two Tensor2 resulting in a (general) Tensor4
///
/// Computes:
///
/// ```text
/// D = s A ⊗ B
///         ‾
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ = s Aᵢₗ Bⱼₖ
/// ```
///
/// **Important:** The result is **not** necessarily minor-symmetric; therefore `D` must be General.
///
/// # Output
///
/// * `dd` -- the tensor `D`; it must be [Rep::General]
///
/// # Input
///
/// * `a` -- first tensor; with the same [Rep] as `b`
/// * `b` -- second tensor; with the same [Rep] as `a`
///
/// # Panics
///
/// 1. A panic will occur if `dd` is not [Rep::General]
/// 2. A panic will occur if the `a` and `b` have different [Rep]
#[inline]
pub fn t2_udyad_t2(dd: &mut Tensor4, s: f64, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(dd.rep(), Rep::General);
    assert_eq!(bb.rep(), aa.rep());
    t2_udyad_t2_slice(dd, s, aa.as_data(), bb.as_data(), aa.dim());
}

/// Internal (unrolled) underbar dyadic product on raw Kelvin-Mandel vectors.
#[rustfmt::skip]
#[inline]
pub(crate) fn t2_udyad_t2_slice(dd: &mut Tensor4, s: f64, a: &[f64], b: &[f64], dim: usize) {
    let tsq2 = 2.0 * SQRT_2;
    if dim == 4 {
        let a = &a[..4];
        let b = &b[..4];
        dd.set(0, 0, s*a[0]*b[0]);
        dd.set(0, 1, s*(a[3]*b[3])/2.0);
        dd.set(0, 2, 0.0);
        dd.set(0, 3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(0, 4, 0.0);
        dd.set(0, 5, 0.0);
        dd.set(0, 6, s*(a[3]*b[0] - a[0]*b[3])/2.0);
        dd.set(0, 7, 0.0);
        dd.set(0, 8, 0.0);

        dd.set(1, 0, s*(a[3]*b[3])/2.0);
        dd.set(1, 1, s*a[1]*b[1]);
        dd.set(1, 2, 0.0);
        dd.set(1, 3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(1, 4, 0.0);
        dd.set(1, 5, 0.0);
        dd.set(1, 6, s*(-(a[3]*b[1]) + a[1]*b[3])/2.0);
        dd.set(1, 7, 0.0);
        dd.set(1, 8, 0.0);

        dd.set(2, 0, 0.0);
        dd.set(2, 1, 0.0);
        dd.set(2, 2, s*a[2]*b[2]);
        dd.set(2, 3, 0.0);
        dd.set(2, 4, 0.0);
        dd.set(2, 5, 0.0);
        dd.set(2, 6, 0.0);
        dd.set(2, 7, 0.0);
        dd.set(2, 8, 0.0);

        dd.set(3, 0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(3, 1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(3, 2, 0.0);
        dd.set(3, 3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.set(3, 4, 0.0);
        dd.set(3, 5, 0.0);
        dd.set(3, 6, s*(a[1]*b[0] - a[0]*b[1])/2.0);
        dd.set(3, 7, 0.0);
        dd.set(3, 8, 0.0);

        dd.set(4, 0, 0.0);
        dd.set(4, 1, 0.0);
        dd.set(4, 2, 0.0);
        dd.set(4, 3, 0.0);
        dd.set(4, 4, s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.set(4, 5, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.set(4, 6, 0.0);
        dd.set(4, 7, s*(a[2]*b[1] - a[1]*b[2])/2.0);
        dd.set(4, 8, s*(-(a[3]*b[2]) + a[2]*b[3])/tsq2);

        dd.set(5, 0, 0.0);
        dd.set(5, 1, 0.0);
        dd.set(5, 2, 0.0);
        dd.set(5, 3, 0.0);
        dd.set(5, 4, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.set(5, 5, s*(a[2]*b[0] + a[0]*b[2])/2.0);
        dd.set(5, 6, 0.0);
        dd.set(5, 7, s*(-(a[3]*b[2]) + a[2]*b[3])/tsq2);
        dd.set(5, 8, s*(a[2]*b[0] - a[0]*b[2])/2.0);

        dd.set(6, 0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.set(6, 1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.set(6, 2, 0.0);
        dd.set(6, 3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.set(6, 4, 0.0);
        dd.set(6, 5, 0.0);
        dd.set(6, 6, s*(-(a[1]*b[0]) - a[0]*b[1] + a[3]*b[3])/2.0);
        dd.set(6, 7, 0.0);
        dd.set(6, 8, 0.0);

        dd.set(7, 0, 0.0);
        dd.set(7, 1, 0.0);
        dd.set(7, 2, 0.0);
        dd.set(7, 3, 0.0);
        dd.set(7, 4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.set(7, 5, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.set(7, 6, 0.0);
        dd.set(7, 7, s*(-(a[2]*b[1]) - a[1]*b[2])/2.0);
        dd.set(7, 8, s*(-(a[3]*b[2] + a[2]*b[3])/tsq2));

        dd.set(8, 0, 0.0);
        dd.set(8, 1, 0.0);
        dd.set(8, 2, 0.0);
        dd.set(8, 3, 0.0);
        dd.set(8, 4, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.set(8, 5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.set(8, 6, 0.0);
        dd.set(8, 7, s*(-(a[3]*b[2] + a[2]*b[3])/tsq2));
        dd.set(8, 8, s*(-(a[2]*b[0]) - a[0]*b[2])/2.0);
    } else if dim == 6 {
        let a = &a[..6];
        let b = &b[..6];
        dd.set(0, 0, s*a[0]*b[0]);
        dd.set(0, 1, s*(a[3]*b[3])/2.0);
        dd.set(0, 2, s*(a[5]*b[5])/2.0);
        dd.set(0, 3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(0, 4, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.set(0, 5, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.set(0, 6, s*(a[3]*b[0] - a[0]*b[3])/2.0);
        dd.set(0, 7, s*(a[5]*b[3] - a[3]*b[5])/tsq2);
        dd.set(0, 8, s*(a[5]*b[0] - a[0]*b[5])/2.0);

        dd.set(1, 0, s*(a[3]*b[3])/2.0);
        dd.set(1, 1, s*a[1]*b[1]);
        dd.set(1, 2, s*(a[4]*b[4])/2.0);
        dd.set(1, 3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(1, 4, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.set(1, 5, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.set(1, 6, s*(-(a[3]*b[1]) + a[1]*b[3])/2.0);
        dd.set(1, 7, s*(a[4]*b[1] - a[1]*b[4])/2.0);
        dd.set(1, 8, s*(a[4]*b[3] - a[3]*b[4])/tsq2);

        dd.set(2, 0, s*(a[5]*b[5])/2.0);
        dd.set(2, 1, s*(a[4]*b[4])/2.0);
        dd.set(2, 2, s*a[2]*b[2]);
        dd.set(2, 3, s*(a[ 5]*b[4] + a[4]*b[5])/tsq2);
        dd.set(2, 4, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.set(2, 5, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.set(2, 6, s*(-(a[5]*b[4]) + a[4]*b[5])/tsq2);
        dd.set(2, 7, s*(-(a[4]*b[2]) + a[2]*b[4])/2.0);
        dd.set(2, 8, s*(-(a[5]*b[2]) + a[2]*b[5])/2.0);

        dd.set(3, 0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.set(3, 1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.set(3, 2, s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.set(3, 3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.set(3, 4, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(3, 5, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(3, 6, s*(a[1]*b[0] - a[0]*b[1])/2.0);
        dd.set(3, 7, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] - a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.set(3, 8, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] - SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);

        dd.set(4, 0, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.set(4, 1, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.set(4, 2, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.set(4, 3, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(4, 4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])/2.0);
        dd.set(4, 5, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(4, 6, s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] - a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(4, 7, s*(a[2]*b[1] - a[1]*b[2])/2.0);
        dd.set(4, 8, s*(-(SQRT_2*a[3]*b[2]) + SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);

        dd.set(5, 0, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.set(5, 1, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.set(5, 2, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.set(5, 3, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(5, 4, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(5, 5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])/2.0);
        dd.set(5, 6, s*(SQRT_2*a[4]*b[0] - a[5]*b[3] - SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(5, 7, s*(-(SQRT_2*a[3]*b[2]) + SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
        dd.set(5, 8, s*(a[2]*b[0] - a[0]*b[2])/2.0);

        dd.set(6, 0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.set(6, 1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.set(6, 2, s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.set(6, 3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.set(6, 4, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.set(6, 5, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.set(6, 6, s*(-(a[1]*b[0]) - a[0]*b[1] + a[3]*b[3])/2.0);
        dd.set(6, 7, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] - a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(6, 8, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] - SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);

        dd.set(7, 0, s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.set(7, 1, s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.set(7, 2, s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.set(7, 3, s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(7, 4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.set(7, 5, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(7, 6, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] - a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.set(7, 7, s*(-(a[2]*b[1]) - a[1]*b[2] + a[4]*b[4])/2.0);
        dd.set(7, 8, s*(-(SQRT_2*a[3]*b[2]) - SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);

        dd.set(8, 0, s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);
        dd.set(8, 1, s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);
        dd.set(8, 2, s*(a[5]*b[2] - a[2]*b[5])/2.0);
        dd.set(8, 3, s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(8, 4, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
        dd.set(8, 5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.set(8, 6, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] - SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.set(8, 7, s*(-(SQRT_2*a[3]*b[2]) - SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.set(8, 8, s*(-(a[2]*b[0]) - a[0]*b[2] + a[5]*b[5])/2.0);
    } else {
        let a = &a[..9];
        let b = &b[..9];
        dd.set(0, 0, s*a[0]*b[0]);
        dd.set(0, 1, s*((a[3] + a[6])*(b[3] + b[6]))/2.0);
        dd.set(0, 2, s*((a[5] + a[8])*(b[5] + b[8]))/2.0);
        dd.set(0, 3, s*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.set(0, 4, s*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.set(0, 5, s*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);
        dd.set(0, 6, s*(a[3]*b[0] + a[6]*b[0] - a[0]*(b[3] + b[6]))/2.0);
        dd.set(0, 7, s*((a[5] + a[8])*(b[3] + b[6]) - (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.set(0, 8, s*(a[5]*b[0] + a[8]*b[0] - a[0]*(b[5] + b[8]))/2.0);

        dd.set(1, 0, s*((a[3] - a[6])*(b[3] - b[6]))/2.0);
        dd.set(1, 1, s*a[1]*b[1]);
        dd.set(1, 2, s*((a[4] + a[7])*(b[4] + b[7]))/2.0);
        dd.set(1, 3, s*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
        dd.set(1, 4, s*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.set(1, 5, s*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);
        dd.set(1, 6, s*(-(a[3]*b[1]) + a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
        dd.set(1, 7, s*(a[4]*b[1] + a[7]*b[1] - a[1]*(b[4] + b[7]))/2.0);
        dd.set(1, 8, s*((a[4] + a[7])*(b[3] - b[6]) - (a[3] - a[6])*(b[4] + b[7]))/tsq2);

        dd.set(2, 0, s*((a[5] - a[8])*(b[5] - b[8]))/2.0);
        dd.set(2, 1, s*((a[4] - a[7])*(b[4] - b[7]))/2.0);
        dd.set(2, 2, s*a[2]*b[2]);
        dd.set(2, 3, s*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.set(2, 4, s*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
        dd.set(2, 5, s*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);
        dd.set(2, 6, s*(-((a[5] - a[8])*(b[4] - b[7])) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.set(2, 7, s*(-(a[4]*b[2]) + a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
        dd.set(2, 8, s*(-(a[5]*b[2]) + a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);

        dd.set(3, 0, s*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.set(3, 1, s*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))/2.0);
        dd.set(3, 2, s*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.set(3, 3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
        dd.set(3, 4, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(3, 5, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.set(3, 6, s*(a[1]*b[0] - a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
        dd.set(3, 7, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) - (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(3, 8, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) - SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.set(4, 0, s*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.set(4, 1, s*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.set(4, 2, s*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))/2.0);
        dd.set(4, 3, s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(4, 4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
        dd.set(4, 5, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.set(4, 6, s*(-(SQRT_2*(a[5] - a[8])*b[1]) + (a[4] - a[7])*(b[3] - b[6]) - (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(4, 7, s*(a[2]*b[1] - a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
        dd.set(4, 8, s*(-(SQRT_2*(a[3] - a[6])*b[2]) + SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.set(5, 0, s*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.set(5, 1, s*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.set(5, 2, s*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))/2.0);
        dd.set(5, 3, s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(5, 4, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(5, 5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
        dd.set(5, 6, s*(SQRT_2*(a[4] - a[7])*b[0] - (a[5] - a[8])*(b[3] + b[6]) - SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(5, 7, s*(-(SQRT_2*(a[3] + a[6])*b[2]) + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(5, 8, s*(a[2]*b[0] - a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);

        dd.set(6, 0, s*(-(a[3]*b[0]) + a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.set(6, 1, s*(a[3]*b[1] + a[6]*b[1] - a[1]*(b[3] + b[6]))/2.0);
        dd.set(6, 2, s*((a[5] + a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.set(6, 3, s*(-(a[1]*b[0]) + a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
        dd.set(6, 4, s*(SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(6, 5, s*(-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.set(6, 6, s*(-(a[1]*b[0]) - a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
        dd.set(6, 7, s*(SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) - (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.set(6, 8, s*(-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) - SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.set(7, 0, s*(-((a[5] - a[8])*(b[3] - b[6])) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.set(7, 1, s*(-(a[4]*b[1]) + a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.set(7, 2, s*(a[4]*b[2] + a[7]*b[2] - a[2]*(b[4] + b[7]))/2.0);
        dd.set(7, 3, s*(-(SQRT_2*(a[5] - a[8])*b[1]) - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(7, 4, s*(-(a[2]*b[1]) + a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
        dd.set(7, 5, s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.set(7, 6, s*(SQRT_2*(a[5] - a[8])*b[1] - (a[4] - a[7])*(b[3] - b[6]) - (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.set(7, 7, s*(-(a[2]*b[1]) - a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
        dd.set(7, 8, s*(-(SQRT_2*(a[3] - a[6])*b[2]) - SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.set(8, 0, s*(-(a[5]*b[0]) + a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.set(8, 1, s*(-((a[4] - a[7])*(b[3] + b[6])) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.set(8, 2, s*(a[5]*b[2] + a[8]*b[2] - a[2]*(b[5] + b[8]))/2.0);
        dd.set(8, 3, s*(-(SQRT_2*(a[4] - a[7])*b[0]) - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(8, 4, s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(8, 5, s*(-(a[2]*b[0]) + a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);
        dd.set(8, 6, s*(-(SQRT_2*(a[4] - a[7])*b[0]) + (a[5] - a[8])*(b[3] + b[6]) - SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.set(8, 7, s*(-(SQRT_2*(a[3] + a[6])*b[2]) - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.set(8, 8, s*(-(a[2]*b[0]) - a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
    }
}

/// Performs the self-sum-dyadic (ssd) operation with a Tensor2 yielding a minor-symmetric Tensor4
///
/// Computes:
///
/// ```text
///          _
/// D = s (A ⊗ A + A ⊗ A)
///                  ‾
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ = s (Aᵢₖ Aⱼₗ + Aᵢₗ Aⱼₖ)
/// ```
///
/// **Important:** Even if `A` is Symmetric 2D, the result may not be expressed by a Symmetric 2D Tensor4.
///
/// # Output
///
/// * `dd` -- The resulting tensor (minor-symmetric); it must be [Rep::Symmetric]
///
/// # Input
///
/// * `aa` -- Second-order tensor, symmetric or not.
///
/// # Panics
///
/// A panic will occur if `dd` is not [Rep::Symmetric]
#[inline]
pub fn ssd_fn(dd: &mut Tensor4, s: f64, aa: &Tensor2) {
    assert_eq!(dd.rep(), Rep::Symmetric);
    ssd_fn_slice(dd, s, aa.as_data(), aa.dim());
}

/// Internal (unrolled) self-sum-dyadic operation on raw Kelvin-Mandel vectors.
#[rustfmt::skip]
#[inline]
pub(crate) fn ssd_fn_slice(dd: &mut Tensor4, s: f64, a: &[f64], dim: usize) {
    if dim == 4 {
        let a = &a[..4];
        dd.set(0, 0, s*(2.0*a[0]*a[0]));
        dd.set(0, 1, s*(a[3]*a[3]));
        dd.set(0, 2, 0.0);
        dd.set(0, 3, s*(2.0*a[0]*a[3]));
        dd.set(0, 4, 0.0);
        dd.set(0, 5, 0.0);

        dd.set(1, 0, s*(a[3]*a[3]));
        dd.set(1, 1, s*(2.0*a[1]*a[1]));
        dd.set(1, 2, 0.0);
        dd.set(1, 3, s*(2.0*a[1]*a[3]));
        dd.set(1, 4, 0.0);
        dd.set(1, 5, 0.0);

        dd.set(2, 0, 0.0);
        dd.set(2, 1, 0.0);
        dd.set(2, 2, s*(2.0*a[2]*a[2]));
        dd.set(2, 3, 0.0);
        dd.set(2, 4, 0.0);
        dd.set(2, 5, 0.0);

        dd.set(3, 0, s*(2.0*a[0]*a[3]));
        dd.set(3, 1, s*(2.0*a[1]*a[3]));
        dd.set(3, 2, 0.0);
        dd.set(3, 3, s*(2.0*a[0]*a[1] + a[3]*a[3]));
        dd.set(3, 4, 0.0);
        dd.set(3, 5, 0.0);

        dd.set(4, 0, 0.0);
        dd.set(4, 1, 0.0);
        dd.set(4, 2, 0.0);
        dd.set(4, 3, 0.0);
        dd.set(4, 4, s*(2.0*a[1]*a[2]));
        dd.set(4, 5, s*(SQRT_2*a[2]*a[3]));

        dd.set(5, 0, 0.0);
        dd.set(5, 1, 0.0);
        dd.set(5, 2, 0.0);
        dd.set(5, 3, 0.0);
        dd.set(5, 4, s*(SQRT_2*a[2]*a[3]));
        dd.set(5, 5, s*(2.0*a[0]*a[2]));
    } else if dim == 6 {
        let a = &a[..6];
        dd.set(0, 0, s*(2.0*a[0]*a[0]));
        dd.set(0, 1, s*(a[3]*a[3]));
        dd.set(0, 2, s*(a[5]*a[5]));
        dd.set(0, 3, s*(2.0*a[0]*a[3]));
        dd.set(0, 4, s*(SQRT_2*a[3]*a[5]));
        dd.set(0, 5, s*(2.0*a[ 0]*a[5]));

        dd.set(1, 0, s*(a[3]*a[3]));
        dd.set(1, 1, s*(2.0*a[1]*a[1]));
        dd.set(1, 2, s*(a[4]*a[4]));
        dd.set(1, 3, s*(2.0*a[1]*a[3]));
        dd.set(1, 4, s*(2.0*a[1]*a[4]));
        dd.set(1, 5, s*(SQRT_2*a[3]*a[4]));

        dd.set(2, 0, s*(a[5]*a[5]));
        dd.set(2, 1, s*(a[4]*a[4]));
        dd.set(2, 2, s*(2.0*a[2]*a[2]));
        dd.set(2, 3, s*(SQRT_2*a[4]*a[ 5]));
        dd.set(2, 4, s*(2.0*a[2]*a[4]));
        dd.set(2, 5, s*(2.0*a[2]*a[5]));

        dd.set(3, 0, s*(2.0*a[0]*a[3]));
        dd.set(3, 1, s*(2.0*a[1]*a[3]));
        dd.set(3, 2, s*(SQRT_2*a[4]* a[5]));
        dd.set(3, 3, s*(2.0*a[0]*a[1] + a[3]*a[3]));
        dd.set(3, 4, s*(a[3]*a[4] + SQRT_2*a[1]*a[5]));
        dd.set(3, 5, s*(SQRT_2*a[0]*a[4] + a[3]*a[5]));

        dd.set(4, 0, s*(SQRT_2*a[3]*a[5]));
        dd.set(4, 1, s*(2.0*a[1]*a[4]));
        dd.set(4, 2, s*(2.0*a[2]*a[4]));
        dd.set(4, 3, s*(a[3]*a[4] + SQRT_2*a[1]*a[5]));
        dd.set(4, 4, s*(2.0*a[1]*a[2] + a[4]*a[4]));
        dd.set(4, 5, s*(SQRT_2*a[2]*a[3] + a[4]*a[5]));

        dd.set(5, 0, s*(2.0*a[0]*a[5]));
        dd.set(5, 1, s*(SQRT_2*a[3]*a[4]));
        dd.set(5, 2, s*(2.0*a[2]*a[5]));
        dd.set(5, 3, s*(SQRT_2*a[0]* a[4] + a[3]*a[5]));
        dd.set(5, 4, s*(SQRT_2*a[2]*a[3] + a[4]*a[5]));
        dd.set(5, 5, s*(2.0*a[0]*a[2] + a[5]*a[5]));
    } else {
        let a = &a[..9];
        dd.set(0, 0, s*(2.0*a[0]*a[0]));
        dd.set(0, 1, s*((a[3] + a[6])*(a[3] + a[6])));
        dd.set(0, 2, s*((a[5] + a[8])*(a[5] + a[8])));
        dd.set(0, 3, s*(2.0*a[0]*(a[3] + a[6])));
        dd.set(0, 4, s*(SQRT_2*(a[3] + a[6])*(a[5] + a[8])));
        dd.set(0, 5, s*(2.0*a[0]*(a[5] + a[8])));

        dd.set(1, 0, s*((a[3] - a[6])*(a[3] - a[6])));
        dd.set(1, 1, s*(2.0*a[1]*a[1]));
        dd.set(1, 2, s*((a[4] + a[7])*(a[4] + a[7])));
        dd.set(1, 3, s*(2.0*a[1]*(a[3] - a[6])));
        dd.set(1, 4, s*(2.0*a[1]*(a[4] + a[7])));
        dd.set(1, 5, s*(SQRT_2*(a[3] - a[6])*(a[4] + a[7])));

        dd.set(2, 0, s*((a[5] - a[8])*(a[5] - a[8])));
        dd.set(2, 1, s*((a[4] - a[7])*(a[4] - a[7])));
        dd.set(2, 2, s*(2.0*a[2]*a[2]));
        dd.set(2, 3, s*(SQRT_2*(a[4] - a[7])*(a[5] - a[8])));
        dd.set(2, 4, s*(2.0*a[2]*(a[4] - a[7])));
        dd.set(2, 5, s*(2.0*a[2]*(a[5] - a[8])));

        dd.set(3, 0, s*(2.0*a[0]*(a[3] - a[6])));
        dd.set(3, 1, s*(2.0*a[1]*(a[3] + a[6])));
        dd.set(3, 2, s*(SQRT_2*(a[4] + a[7])*(a[5] + a[8])));
        dd.set(3, 3, s*(2.0*a[0]*a[1] + a[3]*a[3] - a[6]*a[6]));
        dd.set(3, 4, s*((a[3] + a[6])*(a[4] + a[7]) + SQRT_2*a[1]*(a[5] + a[8])));
        dd.set(3, 5, s*(SQRT_2*a[0]*(a[4] + a[7]) + (a[3] - a[6])*(a[5] + a[8])));

        dd.set(4, 0, s*(SQRT_2*(a[3] - a[6])*(a[5] - a[8])));
        dd.set(4, 1, s*(2.0*a[1]*(a[4] - a[7])));
        dd.set(4, 2, s*(2.0*a[2]*(a[4] + a[7])));
        dd.set(4, 3, s*((a[3] - a[6])*(a[4] - a[7]) + SQRT_2*a[1]*(a[5] - a[8])));
        dd.set(4, 4, s*(2.0*a[1]*a[2] + a[4]*a[4] - a[7]*a[7]));
        dd.set(4, 5, s*(SQRT_2*a[2]*(a[3] - a[6]) + (a[4] + a[7])*(a[5] - a[8])));

        dd.set(5, 0, s*(2.0*a[0]*(a[5] - a[8])));
        dd.set(5, 1, s*(SQRT_2*(a[3] + a[6])*(a[4] - a[7])));
        dd.set(5, 2, s*(2.0*a[2]*(a[5] + a[8])));
        dd.set(5, 3, s*(SQRT_2*a[0]*(a[4] - a[7]) + (a[3] + a[6])*(a[5] - a[8])));
        dd.set(5, 4, s*(SQRT_2*a[2]*(a[3] + a[6]) + (a[4] - a[7])*(a[5] + a[8])));
        dd.set(5, 5, s*(2.0*a[0]*a[2] + a[5]*a[5] - a[8]*a[8]));
    }
}

/// Performs the quad-sum-dyadic (qsd) operation with two Tensor2 yielding a minor-symmetric Tensor4
///
/// Computes:
///
/// ```text
///          _               _
/// D = s (A ⊗ B + A ⊗ B + B ⊗ A + B ⊗ A)
///                  ‾               ‾
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ = s (Aᵢₖ Bⱼₗ + Aᵢₗ Bⱼₖ + Bᵢₖ Aⱼₗ + Bᵢₗ Aⱼₖ)
/// ```
///
/// **Important:** Even if `A` and `B` are Symmetric 2D, the result may not be expressed by a Symmetric 2D Tensor4.
///
/// # Output
///
/// * `dd` -- The resulting tensor (minor-symmetric); it must be [Rep::Symmetric]
///
/// # Input
///
/// * `aa` -- Second-order tensor, symmetric or not; with the same [Rep] as `bb`
/// * `bb` -- Second-order tensor, symmetric or not; with the same [Rep] as `aa`
///
/// # Panics
///
/// 1. A panic will occur if `dd` is not [Rep::Symmetric]
/// 2. A panic will occur if `aa` and `bb` have different [Rep]
#[inline]
pub fn qsd_fn(dd: &mut Tensor4, s: f64, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(dd.rep(), Rep::Symmetric);
    assert_eq!(bb.rep(), aa.rep());
    qsd_fn_slice(dd, s, aa.as_data(), bb.as_data(), aa.dim());
}

/// Internal (unrolled) quad-sum-dyadic operation on raw Kelvin-Mandel vectors.
#[rustfmt::skip]
#[inline]
pub(crate) fn qsd_fn_slice(dd: &mut Tensor4, s: f64, a: &[f64], b: &[f64], dim: usize) {
    if dim == 4 {
        dd.set(0, 0, s*(4.0*a[0]*b[0]));
        dd.set(0, 1, s*(2.0*a[3]*b[3]));
        dd.set(0, 2, 0.0);
        dd.set(0, 3, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
        dd.set(0, 4, 0.0);
        dd.set(0, 5, 0.0);

        dd.set(1, 0, s*(2.0*a[3]*b[3]));
        dd.set(1, 1, s*(4.0*a[1]*b[1]));
        dd.set(1, 2, 0.0);
        dd.set(1, 3, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
        dd.set(1, 4, 0.0);
        dd.set(1, 5, 0.0);

        dd.set(2, 0, 0.0);
        dd.set(2, 1, 0.0);
        dd.set(2, 2, s*(4.0*a[2]*b[2]));
        dd.set(2, 3, 0.0);
        dd.set(2, 4, 0.0);
        dd.set(2, 5, 0.0);

        dd.set(3, 0, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
        dd.set(3, 1, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
        dd.set(3, 2, 0.0);
        dd.set(3, 3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])));
        dd.set(3, 4, 0.0);
        dd.set(3, 5, 0.0);

        dd.set(4, 0, 0.0);
        dd.set(4, 1, 0.0);
        dd.set(4, 2, 0.0);
        dd.set(4, 3, 0.0);
        dd.set(4, 4, s*(2.0*(a[2]*b[1] + a[1]*b[2])));
        dd.set(4, 5, s*(SQRT_2*(a[3]*b[2] + a[2]*b[3])));

        dd.set(5, 0, 0.0);
        dd.set(5, 1, 0.0);
        dd.set(5, 2, 0.0);
        dd.set(5, 3, 0.0);
        dd.set(5, 4, s*(SQRT_2*(a[3]*b[2] + a[2]*b[3])));
        dd.set(5, 5, s*(2.0*(a[2]*b[0] + a[0]*b[2])));
    } else if dim == 6 {
        let a = &a[..6];
        let b = &b[..6];
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
        let a = &a[..9];
        let b = &b[..9];
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MN_TO_IJKL, Rep};
    use russell_lab::{Matrix, mat_approx_eq};

    fn kelvin_matrix(dd: &Tensor4) -> Matrix {
        let dim = dd.dim();
        let mut m = Matrix::new(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                m.set(i, j, dd.get(i, j));
            }
        }
        m
    }

    #[test]
    #[should_panic]
    fn t2_odyad_t2_panics_on_non_general() {
        let a = Tensor2::new(Rep::Symmetric2D);
        let b = Tensor2::new(Rep::Symmetric2D);
        let mut dd = Tensor4::new(Rep::Symmetric2D); // wrong; it must be General
        t2_odyad_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_odyad_t2_panics_on_different_rep() {
        let a = Tensor2::new(Rep::Symmetric2D);
        let b = Tensor2::new(Rep::Symmetric); // wrong; it must be the same as `a`
        let mut dd = Tensor4::new(Rep::General);
        t2_odyad_t2(&mut dd, 1.0, &a, &b);
    }

    fn check_odyad(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_std_matrix();
        let b = b_ten.as_std_matrix();
        let dd = dd_ten.as_std_matrix();
        let mut correct = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * a.get(i, k) * b.get(j, l));
            }
        }
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn t2_odyad_t2_works() {
        // general odyad general
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Rep::General).unwrap();
        let mut dd = Tensor4::new(Rep::General);
        t2_odyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
            [18.0, 32.0, 42.0, 16.0, 28.0, 14.0, 36.0, 48.0, 54.0],
            [48.0, 50.0, 48.0, 40.0, 40.0, 32.0, 60.0, 60.0, 72.0],
            [42.0, 32.0, 18.0, 28.0, 16.0, 14.0, 48.0, 36.0, 54.0],
            [12.0, 20.0, 24.0, 10.0, 16.0, 8.0, 24.0, 30.0, 36.0],
            [24.0, 20.0, 12.0, 16.0, 10.0, 8.0, 30.0, 24.0, 36.0],
            [6.0, 8.0, 6.0, 4.0, 4.0, 2.0, 12.0, 12.0, 18.0],
            [72.0, 80.0, 84.0, 64.0, 70.0, 56.0, 90.0, 96.0, 108.0],
            [84.0, 80.0, 72.0, 70.0, 64.0, 56.0, 96.0, 90.0, 108.0],
            [126.0, 128.0, 126.0, 112.0, 112.0, 98.0, 144.0, 144.0, 162.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_odyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric odyad symmetric
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Rep::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::General);
        t2_odyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
            [6.0, 40.0, 72.0, 10.0, 48.0, 12.0, 24.0, 60.0, 36.0],
            [40.0, 8.0, 40.0, 16.0, 16.0, 32.0, 20.0, 20.0, 50.0],
            [72.0, 40.0, 6.0, 48.0, 10.0, 12.0, 60.0, 24.0, 36.0],
            [10.0, 16.0, 48.0, 4.0, 32.0, 8.0, 40.0, 24.0, 60.0],
            [48.0, 16.0, 10.0, 32.0, 4.0, 8.0, 24.0, 40.0, 60.0],
            [12.0, 32.0, 12.0, 8.0, 8.0, 2.0, 48.0, 48.0, 72.0],
            [24.0, 20.0, 60.0, 40.0, 24.0, 48.0, 12.0, 50.0, 30.0],
            [60.0, 20.0, 24.0, 24.0, 40.0, 48.0, 50.0, 12.0, 30.0],
            [36.0, 50.0, 36.0, 60.0, 60.0, 72.0, 30.0, 30.0, 18.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_odyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric 2D odyad symmetric 2D
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Rep::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Rep::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Rep::General);
        t2_odyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_std_matrix();
        // println!("{:.1}", mat);
        let correct = Matrix::from(&[
            [6.0, 32.0, 0.0, 8.0, 0.0, 0.0, 24.0, 0.0, 0.0],
            [32.0, 8.0, 0.0, 16.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8.0, 16.0, 0.0, 4.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.0, 8.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 8.0, 2.0, 0.0, 0.0, 0.0],
            [24.0, 16.0, 0.0, 32.0, 0.0, 0.0, 12.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 24.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0, 18.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-14);
        check_odyad(2.0, &a, &b, &dd, 1e-15);
    }

    #[test]
    fn t2_odyad_t2_update_slice_works() {
        // dd += s (A ⊗̄ B) for each representation
        for (mat_a, mat_b, rep) in [
            (
                &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                &[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
                Rep::General,
            ),
            (
                &[[1.0, 4.0, 6.0], [4.0, 2.0, 5.0], [6.0, 5.0, 3.0]],
                &[[3.0, 5.0, 6.0], [5.0, 2.0, 4.0], [6.0, 4.0, 1.0]],
                Rep::Symmetric,
            ),
            (
                &[[1.0, 4.0, 0.0], [4.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                &[[3.0, 4.0, 0.0], [4.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                Rep::Symmetric2D,
            ),
        ] {
            let a = Tensor2::from_std_matrix(mat_a, rep).unwrap();
            let b = Tensor2::from_std_matrix(mat_b, rep).unwrap();

            // dd := 2.0 (A ⊗̄ B)
            let mut dd = Tensor4::new(Rep::General);
            t2_odyad_t2(&mut dd, 2.0, &a, &b);

            // dd += 3.0 (A ⊗̄ B)  =>  dd == 5.0 (A ⊗̄ B)
            t2_odyad_t2_update_slice(&mut dd, 3.0, a.as_data(), b.as_data(), a.dim());

            // reference
            let mut dd_ref = Tensor4::new(Rep::General);
            t2_odyad_t2(&mut dd_ref, 5.0, &a, &b);
            mat_approx_eq(&dd.as_std_matrix(), &dd_ref.as_std_matrix(), 1e-13);
        }
    }

    #[test]
    #[should_panic]
    fn t2_udyad_t2_panics_on_non_general() {
        let a = Tensor2::new(Rep::Symmetric2D);
        let b = Tensor2::new(Rep::Symmetric2D);
        let mut dd = Tensor4::new(Rep::Symmetric2D); // wrong; it must be General
        t2_udyad_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_udyad_t2_panics_on_different_rep() {
        let a = Tensor2::new(Rep::Symmetric2D);
        let b = Tensor2::new(Rep::Symmetric); // wrong; it must be the same as `a`
        let mut dd = Tensor4::new(Rep::General);
        t2_udyad_t2(&mut dd, 1.0, &a, &b);
    }

    fn check_udyad(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_std_matrix();
        let b = b_ten.as_std_matrix();
        let dd = dd_ten.as_std_matrix();
        let mut correct = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * a.get(i, l) * b.get(j, k));
            }
        }
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn t2_udyad_t2_works() {
        // general udyad general
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Rep::General).unwrap();
        let mut dd = Tensor4::new(Rep::General);
        t2_udyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
            [18.0, 32.0, 42.0, 36.0, 48.0, 54.0, 16.0, 28.0, 14.0],
            [48.0, 50.0, 48.0, 60.0, 60.0, 72.0, 40.0, 40.0, 32.0],
            [42.0, 32.0, 18.0, 48.0, 36.0, 54.0, 28.0, 16.0, 14.0],
            [12.0, 20.0, 24.0, 24.0, 30.0, 36.0, 10.0, 16.0, 8.0],
            [24.0, 20.0, 12.0, 30.0, 24.0, 36.0, 16.0, 10.0, 8.0],
            [6.0, 8.0, 6.0, 12.0, 12.0, 18.0, 4.0, 4.0, 2.0],
            [72.0, 80.0, 84.0, 90.0, 96.0, 108.0, 64.0, 70.0, 56.0],
            [84.0, 80.0, 72.0, 96.0, 90.0, 108.0, 70.0, 64.0, 56.0],
            [126.0, 128.0, 126.0, 144.0, 144.0, 162.0, 112.0, 112.0, 98.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_udyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric udyad symmetric
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Rep::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::General);
        t2_udyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
            [6.0, 40.0, 72.0, 24.0, 60.0, 36.0, 10.0, 48.0, 12.0],
            [40.0, 8.0, 40.0, 20.0, 20.0, 50.0, 16.0, 16.0, 32.0],
            [72.0, 40.0, 6.0, 60.0, 24.0, 36.0, 48.0, 10.0, 12.0],
            [10.0, 16.0, 48.0, 40.0, 24.0, 60.0, 4.0, 32.0, 8.0],
            [48.0, 16.0, 10.0, 24.0, 40.0, 60.0, 32.0, 4.0, 8.0],
            [12.0, 32.0, 12.0, 48.0, 48.0, 72.0, 8.0, 8.0, 2.0],
            [24.0, 20.0, 60.0, 12.0, 50.0, 30.0, 40.0, 24.0, 48.0],
            [60.0, 20.0, 24.0, 50.0, 12.0, 30.0, 24.0, 40.0, 48.0],
            [36.0, 50.0, 36.0, 30.0, 30.0, 18.0, 60.0, 60.0, 72.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_udyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric 2D udyad symmetric 2D
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Rep::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Rep::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Rep::General);
        t2_udyad_t2(&mut dd, 2.0, &a, &b);
        let kelvin_mat = Matrix::from(&[
            [6.0, 32.0, 0.0, 16.0 * SQRT_2, 0.0, 0.0, 8.0 * SQRT_2, 0.0, 0.0],
            [32.0, 8.0, 0.0, 16.0 * SQRT_2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [16.0 * SQRT_2, 16.0 * SQRT_2, 0.0, 40.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 8.0, 16.0, 0.0, 4.0, 8.0],
            [0.0, 0.0, 0.0, 0.0, 16.0, 10.0, 0.0, 8.0, 8.0],
            [-8.0 * SQRT_2, 0.0, 0.0, -4.0, 0.0, 0.0, 24.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -4.0, -8.0, 0.0, -8.0, -16.0],
            [0.0, 0.0, 0.0, 0.0, -8.0, -8.0, 0.0, -16.0, -10.0],
        ]);
        mat_approx_eq(&kelvin_matrix(&dd), &kelvin_mat, 1e-14);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
            [6.0, 32.0, 0.0, 24.0, 0.0, 0.0, 8.0, 0.0, 0.0],
            [32.0, 8.0, 0.0, 16.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8.0, 16.0, 0.0, 32.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 8.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 2.0],
            [24.0, 16.0, 0.0, 12.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 12.0, 24.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 24.0, 18.0, 0.0, 0.0, 0.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-14);
        check_udyad(2.0, &a, &b, &dd, 1e-15);
    }

    #[test]
    #[should_panic]
    fn ssd_fn_panics_on_non_sym() {
        let a = Tensor2::new(Rep::Symmetric2D);
        let mut dd = Tensor4::new(Rep::Symmetric2D); // wrong; it must be Symmetric
        ssd_fn(&mut dd, 1.0, &a);
    }

    fn check_ssd(s: f64, a_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_std_matrix();
        let dd = dd_ten.as_std_matrix();
        let mut correct = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * (a.get(i, k) * a.get(j, l) + a.get(i, l) * a.get(j, k)));
            }
        }
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn ssd_fn_works() {
        // general
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Rep::General).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        ssd_fn(&mut dd, 2.0, &a);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
            [4.0, 16.0, 36.0, 8.0, 24.0, 12.0, 8.0, 24.0, 12.0],
            [64.0, 100.0, 144.0, 80.0, 120.0, 96.0, 80.0, 120.0, 96.0],
            [196.0, 256.0, 324.0, 224.0, 288.0, 252.0, 224.0, 288.0, 252.0],
            [16.0, 40.0, 72.0, 26.0, 54.0, 36.0, 26.0, 54.0, 36.0],
            [112.0, 160.0, 216.0, 134.0, 186.0, 156.0, 134.0, 186.0, 156.0],
            [28.0, 64.0, 108.0, 44.0, 84.0, 60.0, 44.0, 84.0, 60.0],
            [16.0, 40.0, 72.0, 26.0, 54.0, 36.0, 26.0, 54.0, 36.0],
            [112.0, 160.0, 216.0, 134.0, 186.0, 156.0, 134.0, 186.0, 156.0],
            [28.0, 64.0, 108.0, 44.0, 84.0, 60.0, 44.0, 84.0, 60.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_ssd(2.0, &a, &dd, 1e-13);

        // symmetric
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        ssd_fn(&mut dd, 2.0, &a);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
            [4.0, 64.0, 144.0, 16.0, 96.0, 24.0, 16.0, 96.0, 24.0],
            [64.0, 16.0, 100.0, 32.0, 40.0, 80.0, 32.0, 40.0, 80.0],
            [144.0, 100.0, 36.0, 120.0, 60.0, 72.0, 120.0, 60.0, 72.0],
            [16.0, 32.0, 120.0, 36.0, 64.0, 58.0, 36.0, 64.0, 58.0],
            [96.0, 40.0, 60.0, 64.0, 62.0, 84.0, 64.0, 62.0, 84.0],
            [24.0, 80.0, 72.0, 58.0, 84.0, 78.0, 58.0, 84.0, 78.0],
            [16.0, 32.0, 120.0, 36.0, 64.0, 58.0, 36.0, 64.0, 58.0],
            [96.0, 40.0, 60.0, 64.0, 62.0, 84.0, 64.0, 62.0, 84.0],
            [24.0, 80.0, 72.0, 58.0, 84.0, 78.0, 58.0, 84.0, 78.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_ssd(2.0, &a, &dd, 1e-13);

        // symmetric 2D
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Rep::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        ssd_fn(&mut dd, 2.0, &a);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
            [4.0, 64.0, 0.0, 16.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [64.0, 16.0, 0.0, 32.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 36.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [16.0, 32.0, 0.0, 36.0, 0.0, 0.0, 36.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 12.0, 24.0, 0.0, 12.0, 24.0],
            [0.0, 0.0, 0.0, 0.0, 24.0, 6.0, 0.0, 24.0, 6.0],
            [16.0, 32.0, 0.0, 36.0, 0.0, 0.0, 36.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 12.0, 24.0, 0.0, 12.0, 24.0],
            [0.0, 0.0, 0.0, 0.0, 24.0, 6.0, 0.0, 24.0, 6.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_ssd(2.0, &a, &dd, 1e-14);
    }

    #[test]
    #[should_panic]
    fn qsd_fn_panics_on_non_sym() {
        let a = Tensor2::new(Rep::Symmetric2D);
        let b = Tensor2::new(Rep::Symmetric2D);
        let mut dd = Tensor4::new(Rep::Symmetric2D); // wrong; it must be Symmetric
        qsd_fn(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn qsd_fn_panics_on_different_rep() {
        let a = Tensor2::new(Rep::Symmetric2D);
        let b = Tensor2::new(Rep::Symmetric); // wrong; it must be the same as `a`
        let mut dd = Tensor4::new(Rep::Symmetric);
        qsd_fn(&mut dd, 1.0, &a, &b);
    }

    fn check_qsd(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_std_matrix();
        let b = b_ten.as_std_matrix();
        let dd = dd_ten.as_std_matrix();
        let mut correct = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * a.get(i, l) * b.get(j, k));
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
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn qsd_fn_works() {
        // general qsd general
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Rep::General).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        qsd_fn(&mut dd, 2.0, &a, &b);
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
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Rep::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        qsd_fn(&mut dd, 2.0, &a, &b);
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

        // symmetric 2D qsd symmetric 2D
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Rep::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Rep::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        qsd_fn(&mut dd, 2.0, &a, &b);
        let mat = dd.as_std_matrix();
        let correct = Matrix::from(&[
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
        mat_approx_eq(&mat, &correct, 1e-13);
        check_qsd(2.0, &a, &b, &dd, 1e-14);
    }
}
