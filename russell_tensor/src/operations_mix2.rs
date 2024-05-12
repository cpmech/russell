use super::{Tensor2, Tensor4};
use crate::{Mandel, SQRT_2};

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
/// * `dd` -- the tensor `D`; it must be [Mandel::General]
///
/// # Input
///
/// * `a` -- first tensor; with the same [Mandel] as `b`
/// * `b` -- second tensor; with the same [Mandel] as `a`
///
/// # Panics
///
/// 1. A panic will occur if `dd` is not [Mandel::General]
/// 2. A panic will occur the `a` and `b` have different [Mandel]
#[rustfmt::skip]
pub fn t2_odyad_t2(dd: &mut Tensor4, s: f64, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(dd.mandel, Mandel::General);
    assert_eq!(bb.mandel, aa.mandel);
    let dim = aa.vec.dim();
    let a = &aa.vec;
    let b = &bb.vec;
    let tsq2 = 2.0 * SQRT_2;
    if dim == 4 {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*(a[3]*b[3])/2.0);
        dd.mat.set(0,2, 0.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(0,4, 0.0);
        dd.mat.set(0,5, 0.0);
        dd.mat.set(0,6, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(0,7, 0.0);
        dd.mat.set(0,8, 0.0);

        dd.mat.set(1,0, s*(a[3]*b[3])/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, 0.0);
        dd.mat.set(1,3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(1,4, 0.0);
        dd.mat.set(1,5, 0.0);
        dd.mat.set(1,6, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(1,7, 0.0);
        dd.mat.set(1,8, 0.0);

        dd.mat.set(2,0, 0.0);
        dd.mat.set(2,1, 0.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, 0.0);
        dd.mat.set(2,4, 0.0);
        dd.mat.set(2,5, 0.0);
        dd.mat.set(2,6, 0.0);
        dd.mat.set(2,7, 0.0);
        dd.mat.set(2,8, 0.0);

        dd.mat.set(3,0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(3,2, 0.0);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(3,4, 0.0);
        dd.mat.set(3,5, 0.0);
        dd.mat.set(3,6, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(3,7, 0.0);
        dd.mat.set(3,8, 0.0);

        dd.mat.set(4,0, 0.0);
        dd.mat.set(4,1, 0.0);
        dd.mat.set(4,2, 0.0);
        dd.mat.set(4,3, 0.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.mat.set(4,5, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.mat.set(4,6, 0.0);
        dd.mat.set(4,7, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(4,8, s*(a[3]*b[2] - a[2]*b[3])/tsq2);

        dd.mat.set(5,0, 0.0);
        dd.mat.set(5,1, 0.0);
        dd.mat.set(5,2, 0.0);
        dd.mat.set(5,3, 0.0);
        dd.mat.set(5,4, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2])/2.0);
        dd.mat.set(5,6, 0.0);
        dd.mat.set(5,7, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.mat.set(5,8, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);

        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(6,2, 0.0);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(6,4, 0.0);
        dd.mat.set(6,5, 0.0);
        dd.mat.set(6,6, s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3])/2.0);
        dd.mat.set(6,7, 0.0);
        dd.mat.set(6,8, 0.0);

        dd.mat.set(7,0, 0.0);
        dd.mat.set(7,1, 0.0);
        dd.mat.set(7,2, 0.0);
        dd.mat.set(7,3, 0.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(7,5, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.mat.set(7,6, 0.0);
        dd.mat.set(7,7, s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.mat.set(7,8, s*(a[3]*b[2] + a[2]*b[3])/tsq2);

        dd.mat.set(8,0, 0.0);
        dd.mat.set(8,1, 0.0);
        dd.mat.set(8,2, 0.0);
        dd.mat.set(8,3, 0.0);
        dd.mat.set(8,4, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.mat.set(8,6, 0.0);
        dd.mat.set(8,7, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.mat.set(8,8, s*(a[2]*b[0] + a[0]*b[2])/2.0);
    } else if dim == 6 {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*(a[3]*b[3])/2.0);
        dd.mat.set(0,2, s*(a[5]*b[5])/2.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(0,4, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.mat.set(0,5, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.mat.set(0,6, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(0,7, s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.mat.set(0,8, s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);
                        
        dd.mat.set(1,0, s*(a[3]*b[3])/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, s*(a[4]*b[4])/2.0);
        dd.mat.set(1,3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(1,4, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.mat.set(1,5, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.mat.set(1,6, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(1,7, s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.mat.set(1,8, s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);
                        
        dd.mat.set(2,0, s*(a[5]*b[5])/2.0);
        dd.mat.set(2,1, s*(a[4]*b[4])/2.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.mat.set(2,4, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.mat.set(2,5, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.mat.set(2,6, s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.mat.set(2,7, s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.mat.set(2,8, s*(a[5]*b[2] - a[2]*b[5])/2.0);
                        
        dd.mat.set(3,0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(3,2, s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(3,4, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(3,5, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(3,6, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(3,7, s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(3,8, s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
                        
        dd.mat.set(4,0, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.mat.set(4,1, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.mat.set(4,2, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.mat.set(4,3, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])/2.0);
        dd.mat.set(4,5, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(4,6, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(4,7, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(4,8, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
                        
        dd.mat.set(5,0, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.mat.set(5,1, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.mat.set(5,2, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.mat.set(5,3, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(5,4, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])/2.0);
        dd.mat.set(5,6, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.mat.set(5,7, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(5,8, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
                        
        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(6,2, s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(6,4, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(6,5, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.mat.set(6,6, s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3])/2.0);
        dd.mat.set(6,7, s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(6,8, s*(SQRT_2*a[4]*b[0] - a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
                        
        dd.mat.set(7,0, s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.mat.set(7,1, s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.mat.set(7,2, s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.mat.set(7,3, s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(7,5, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(7,6, s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(7,7, s*(a[2]*b[1] + a[1]*b[2] - a[4]*b[4])/2.0);
        dd.mat.set(7,8, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] - a[5]*b[4] - a[4]*b[5])/4.0);
                        
        dd.mat.set(8,0, s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);
        dd.mat.set(8,1, s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);
        dd.mat.set(8,2, s*(a[5]*b[2] - a[2]*b[5])/2.0);
        dd.mat.set(8,3, s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(8,4, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.mat.set(8,6, s*(SQRT_2*a[4]*b[0] - a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.mat.set(8,7, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] - a[5]*b[4] - a[4]*b[5])/4.0);
        dd.mat.set(8,8, s*(a[2]*b[0] + a[0]*b[2] - a[5]*b[5])/2.0);
    } else {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*((a[3] + a[6])*(b[3] + b[6]))/2.0);
        dd.mat.set(0,2, s*((a[5] + a[8])*(b[5] + b[8]))/2.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.mat.set(0,4, s*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.mat.set(0,5, s*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);
        dd.mat.set(0,6, s*(-(a[3]*b[0]) - a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.mat.set(0,7, s*(-((a[5] + a[8])*(b[3] + b[6])) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.mat.set(0,8, s*(-(a[5]*b[0]) - a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);

        dd.mat.set(1,0, s*((a[3] - a[6])*(b[3] - b[6]))/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, s*((a[4] + a[7])*(b[4] + b[7]))/2.0);
        dd.mat.set(1,3, s*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
        dd.mat.set(1,4, s*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.mat.set(1,5, s*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);
        dd.mat.set(1,6, s*(a[3]*b[1] - a[6]*b[1] + a[1]*(-b[3] + b[6]))/2.0);
        dd.mat.set(1,7, s*(-(a[4]*b[1]) - a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.mat.set(1,8, s*(-((a[4] + a[7])*(b[3] - b[6])) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);

        dd.mat.set(2,0, s*((a[5] - a[8])*(b[5] - b[8]))/2.0);
        dd.mat.set(2,1, s*((a[4] - a[7])*(b[4] - b[7]))/2.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, s*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.mat.set(2,4, s*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
        dd.mat.set(2,5, s*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);
        dd.mat.set(2,6, s*((a[5] - a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.mat.set(2,7, s*(a[4]*b[2] - a[7]*b[2] + a[2]*(-b[4] + b[7]))/2.0);
        dd.mat.set(2,8, s*(a[5]*b[2] - a[8]*b[2] + a[2]*(-b[5] + b[8]))/2.0);

        dd.mat.set(3,0, s*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))/2.0);
        dd.mat.set(3,2, s*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
        dd.mat.set(3,4, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(3,5, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.mat.set(3,6, s*(-(a[1]*b[0]) + a[0]*b[1] - a[6]*b[3] + a[3]*b[6])/2.0);
        dd.mat.set(3,7, s*(-(SQRT_2*(a[5] + a[8])*b[1]) - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(3,8, s*(-(SQRT_2*(a[4] + a[7])*b[0]) - (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.mat.set(4,0, s*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.mat.set(4,1, s*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.mat.set(4,2, s*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))/2.0);
        dd.mat.set(4,3, s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
        dd.mat.set(4,5, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.mat.set(4,6, s*(SQRT_2*(a[5] - a[8])*b[1] - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) - SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(4,7, s*(-(a[2]*b[1]) + a[1]*b[2] - a[7]*b[4] + a[4]*b[7])/2.0);
        dd.mat.set(4,8, s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.mat.set(5,0, s*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.mat.set(5,1, s*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.mat.set(5,2, s*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))/2.0);
        dd.mat.set(5,3, s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(5,4, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
        dd.mat.set(5,6, s*(-(SQRT_2*(a[4] - a[7])*b[0]) + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) - (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(5,7, s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) - (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(5,8, s*(-(a[2]*b[0]) + a[0]*b[2] - a[8]*b[5] + a[5]*b[8])/2.0);

        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] + a[6]*b[1] - a[1]*(b[3] + b[6]))/2.0);
        dd.mat.set(6,2, s*((a[5] + a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
        dd.mat.set(6,4, s*(SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(6,5, s*(-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.mat.set(6,6, s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3] + a[6]*b[6])/2.0);
        dd.mat.set(6,7, s*(-(SQRT_2*(a[5] + a[8])*b[1]) + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(6,8, s*(SQRT_2*(a[4] + a[7])*b[0] - (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.mat.set(7,0, s*(-((a[5] - a[8])*(b[3] - b[6])) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.mat.set(7,1, s*(-(a[4]*b[1]) + a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.mat.set(7,2, s*(a[4]*b[2] + a[7]*b[2] - a[2]*(b[4] + b[7]))/2.0);
        dd.mat.set(7,3, s*(-(SQRT_2*(a[5] - a[8])*b[1]) - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
        dd.mat.set(7,5, s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.mat.set(7,6, s*(-(SQRT_2*(a[5] - a[8])*b[1]) + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) - SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(7,7, s*(a[2]*b[1] + a[1]*b[2] - a[4]*b[4] + a[7]*b[7])/2.0);
        dd.mat.set(7,8, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.mat.set(8,0, s*(-(a[5]*b[0]) + a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.mat.set(8,1, s*(-((a[4] - a[7])*(b[3] + b[6])) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.mat.set(8,2, s*(a[5]*b[2] + a[8]*b[2] - a[2]*(b[5] + b[8]))/2.0);
        dd.mat.set(8,3, s*(-(SQRT_2*(a[4] - a[7])*b[0]) - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(8,4, s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);
        dd.mat.set(8,6, s*(SQRT_2*(a[4] - a[7])*b[0] - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) - (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(8,7, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) - (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(8,8, s*(a[2]*b[0] + a[0]*b[2] - a[5]*b[5] + a[8]*b[8])/2.0);
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
/// * `dd` -- the tensor `D`; it must be [Mandel::General]
///
/// # Input
///
/// * `a` -- first tensor; with the same [Mandel] as `b`
/// * `b` -- second tensor; with the same [Mandel] as `a`
///
/// # Panics
///
/// 1. A panic will occur if `dd` is not [Mandel::General]
/// 2. A panic will occur the `a` and `b` have different [Mandel]
#[rustfmt::skip]
pub fn t2_udyad_t2(dd: &mut Tensor4, s: f64, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(dd.mandel, Mandel::General);
    assert_eq!(bb.mandel, aa.mandel);
    let dim = aa.vec.dim();
    let a = &aa.vec;
    let b = &bb.vec;
    let tsq2 = 2.0 * SQRT_2;
    if dim == 4 {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*(a[3]*b[3])/2.0);
        dd.mat.set(0,2, 0.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(0,4, 0.0);
        dd.mat.set(0,5, 0.0);
        dd.mat.set(0,6, s*(a[3]*b[0] - a[0]*b[3])/2.0);
        dd.mat.set(0,7, 0.0);
        dd.mat.set(0,8, 0.0);
                        
        dd.mat.set(1,0, s*(a[3]*b[3])/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, 0.0);
        dd.mat.set(1,3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(1,4, 0.0);
        dd.mat.set(1,5, 0.0);
        dd.mat.set(1,6, s*(-(a[3]*b[1]) + a[1]*b[3])/2.0);
        dd.mat.set(1,7, 0.0);
        dd.mat.set(1,8, 0.0);
                        
        dd.mat.set(2,0, 0.0);
        dd.mat.set(2,1, 0.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, 0.0);
        dd.mat.set(2,4, 0.0);
        dd.mat.set(2,5, 0.0);
        dd.mat.set(2,6, 0.0);
        dd.mat.set(2,7, 0.0);
        dd.mat.set(2,8, 0.0);
                        
        dd.mat.set(3,0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(3,2, 0.0);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(3,4, 0.0);
        dd.mat.set(3,5, 0.0);
        dd.mat.set(3,6, s*(a[1]*b[0] - a[0]*b[1])/2.0);
        dd.mat.set(3,7, 0.0);
        dd.mat.set(3,8, 0.0);
                        
        dd.mat.set(4,0, 0.0);
        dd.mat.set(4,1, 0.0);
        dd.mat.set(4,2, 0.0);
        dd.mat.set(4,3, 0.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.mat.set(4,5, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.mat.set(4,6, 0.0);
        dd.mat.set(4,7, s*(a[2]*b[1] - a[1]*b[2])/2.0);
        dd.mat.set(4,8, s*(-(a[3]*b[2]) + a[2]*b[3])/tsq2);
                        
        dd.mat.set(5,0, 0.0);
        dd.mat.set(5,1, 0.0);
        dd.mat.set(5,2, 0.0);
        dd.mat.set(5,3, 0.0);
        dd.mat.set(5,4, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2])/2.0);
        dd.mat.set(5,6, 0.0);
        dd.mat.set(5,7, s*(-(a[3]*b[2]) + a[2]*b[3])/tsq2);
        dd.mat.set(5,8, s*(a[2]*b[0] - a[0]*b[2])/2.0);
                        
        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(6,2, 0.0);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(6,4, 0.0);
        dd.mat.set(6,5, 0.0);
        dd.mat.set(6,6, s*(-(a[1]*b[0]) - a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(6,7, 0.0);
        dd.mat.set(6,8, 0.0);
                        
        dd.mat.set(7,0, 0.0);
        dd.mat.set(7,1, 0.0);
        dd.mat.set(7,2, 0.0);
        dd.mat.set(7,3, 0.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(7,5, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.mat.set(7,6, 0.0);
        dd.mat.set(7,7, s*(-(a[2]*b[1]) - a[1]*b[2])/2.0);
        dd.mat.set(7,8, s*(-(a[3]*b[2] + a[2]*b[3])/tsq2));
                        
        dd.mat.set(8,0, 0.0);
        dd.mat.set(8,1, 0.0);
        dd.mat.set(8,2, 0.0);
        dd.mat.set(8,3, 0.0);
        dd.mat.set(8,4, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.mat.set(8,6, 0.0);
        dd.mat.set(8,7, s*(-(a[3]*b[2] + a[2]*b[3])/tsq2));
        dd.mat.set(8,8, s*(-(a[2]*b[0]) - a[0]*b[2])/2.0);
    } else if dim == 6 {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*(a[3]*b[3])/2.0);
        dd.mat.set(0,2, s*(a[5]*b[5])/2.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(0,4, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.mat.set(0,5, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.mat.set(0,6, s*(a[3]*b[0] - a[0]*b[3])/2.0);
        dd.mat.set(0,7, s*(a[5]*b[3] - a[3]*b[5])/tsq2);
        dd.mat.set(0,8, s*(a[5]*b[0] - a[0]*b[5])/2.0);
                        
        dd.mat.set(1,0, s*(a[3]*b[3])/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, s*(a[4]*b[4])/2.0);
        dd.mat.set(1,3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(1,4, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.mat.set(1,5, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.mat.set(1,6, s*(-(a[3]*b[1]) + a[1]*b[3])/2.0);
        dd.mat.set(1,7, s*(a[4]*b[1] - a[1]*b[4])/2.0);
        dd.mat.set(1,8, s*(a[4]*b[3] - a[3]*b[4])/tsq2);
                        
        dd.mat.set(2,0, s*(a[5]*b[5])/2.0);
        dd.mat.set(2,1, s*(a[4]*b[4])/2.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, s*(a[ 5]*b[4] + a[4]*b[5])/tsq2);
        dd.mat.set(2,4, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.mat.set(2,5, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.mat.set(2,6, s*(-(a[5]*b[4]) + a[4]*b[5])/tsq2);
        dd.mat.set(2,7, s*(-(a[4]*b[2]) + a[2]*b[4])/2.0);
        dd.mat.set(2,8, s*(-(a[5]*b[2]) + a[2]*b[5])/2.0);
                        
        dd.mat.set(3,0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(3,2, s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(3,4, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(3,5, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(3,6, s*(a[1]*b[0] - a[0]*b[1])/2.0);
        dd.mat.set(3,7, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] - a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(3,8, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] - SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
                        
        dd.mat.set(4,0, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.mat.set(4,1, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.mat.set(4,2, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.mat.set(4,3, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])/2.0);
        dd.mat.set(4,5, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(4,6, s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] - a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(4,7, s*(a[2]*b[1] - a[1]*b[2])/2.0);
        dd.mat.set(4,8, s*(-(SQRT_2*a[3]*b[2]) + SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
                        
        dd.mat.set(5,0, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.mat.set(5,1, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.mat.set(5,2, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.mat.set(5,3, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(5,4, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])/2.0);
        dd.mat.set(5,6, s*(SQRT_2*a[4]*b[0] - a[5]*b[3] - SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(5,7, s*(-(SQRT_2*a[3]*b[2]) + SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
        dd.mat.set(5,8, s*(a[2]*b[0] - a[0]*b[2])/2.0);
                        
        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(6,2, s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(6,4, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(6,5, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.mat.set(6,6, s*(-(a[1]*b[0]) - a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(6,7, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] - a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(6,8, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] - SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
                        
        dd.mat.set(7,0, s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.mat.set(7,1, s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.mat.set(7,2, s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.mat.set(7,3, s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(7,5, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(7,6, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] - a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(7,7, s*(-(a[2]*b[1]) - a[1]*b[2] + a[4]*b[4])/2.0);
        dd.mat.set(7,8, s*(-(SQRT_2*a[3]*b[2]) - SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
                        
        dd.mat.set(8,0, s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);
        dd.mat.set(8,1, s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);
        dd.mat.set(8,2, s*(a[5]*b[2] - a[2]*b[5])/2.0);
        dd.mat.set(8,3, s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(8,4, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.mat.set(8,6, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] - SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(8,7, s*(-(SQRT_2*a[3]*b[2]) - SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(8,8, s*(-(a[2]*b[0]) - a[0]*b[2] + a[5]*b[5])/2.0);
    } else {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*((a[3] + a[6])*(b[3] + b[6]))/2.0);
        dd.mat.set(0,2, s*((a[5] + a[8])*(b[5] + b[8]))/2.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.mat.set(0,4, s*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.mat.set(0,5, s*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);
        dd.mat.set(0,6, s*(a[3]*b[0] + a[6]*b[0] - a[0]*(b[3] + b[6]))/2.0);
        dd.mat.set(0,7, s*((a[5] + a[8])*(b[3] + b[6]) - (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.mat.set(0,8, s*(a[5]*b[0] + a[8]*b[0] - a[0]*(b[5] + b[8]))/2.0);

        dd.mat.set(1,0, s*((a[3] - a[6])*(b[3] - b[6]))/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, s*((a[4] + a[7])*(b[4] + b[7]))/2.0);
        dd.mat.set(1,3, s*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
        dd.mat.set(1,4, s*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.mat.set(1,5, s*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);
        dd.mat.set(1,6, s*(-(a[3]*b[1]) + a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
        dd.mat.set(1,7, s*(a[4]*b[1] + a[7]*b[1] - a[1]*(b[4] + b[7]))/2.0);
        dd.mat.set(1,8, s*((a[4] + a[7])*(b[3] - b[6]) - (a[3] - a[6])*(b[4] + b[7]))/tsq2);

        dd.mat.set(2,0, s*((a[5] - a[8])*(b[5] - b[8]))/2.0);
        dd.mat.set(2,1, s*((a[4] - a[7])*(b[4] - b[7]))/2.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, s*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.mat.set(2,4, s*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
        dd.mat.set(2,5, s*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);
        dd.mat.set(2,6, s*(-((a[5] - a[8])*(b[4] - b[7])) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.mat.set(2,7, s*(-(a[4]*b[2]) + a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
        dd.mat.set(2,8, s*(-(a[5]*b[2]) + a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);

        dd.mat.set(3,0, s*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))/2.0);
        dd.mat.set(3,2, s*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
        dd.mat.set(3,4, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(3,5, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.mat.set(3,6, s*(a[1]*b[0] - a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
        dd.mat.set(3,7, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) - (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(3,8, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) - SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.mat.set(4,0, s*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.mat.set(4,1, s*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.mat.set(4,2, s*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))/2.0);
        dd.mat.set(4,3, s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
        dd.mat.set(4,5, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.mat.set(4,6, s*(-(SQRT_2*(a[5] - a[8])*b[1]) + (a[4] - a[7])*(b[3] - b[6]) - (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(4,7, s*(a[2]*b[1] - a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
        dd.mat.set(4,8, s*(-(SQRT_2*(a[3] - a[6])*b[2]) + SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.mat.set(5,0, s*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.mat.set(5,1, s*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.mat.set(5,2, s*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))/2.0);
        dd.mat.set(5,3, s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(5,4, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
        dd.mat.set(5,6, s*(SQRT_2*(a[4] - a[7])*b[0] - (a[5] - a[8])*(b[3] + b[6]) - SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(5,7, s*(-(SQRT_2*(a[3] + a[6])*b[2]) + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(5,8, s*(a[2]*b[0] - a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);

        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] + a[6]*b[1] - a[1]*(b[3] + b[6]))/2.0);
        dd.mat.set(6,2, s*((a[5] + a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
        dd.mat.set(6,4, s*(SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(6,5, s*(-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.mat.set(6,6, s*(-(a[1]*b[0]) - a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
        dd.mat.set(6,7, s*(SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) - (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(6,8, s*(-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) - SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.mat.set(7,0, s*(-((a[5] - a[8])*(b[3] - b[6])) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.mat.set(7,1, s*(-(a[4]*b[1]) + a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.mat.set(7,2, s*(a[4]*b[2] + a[7]*b[2] - a[2]*(b[4] + b[7]))/2.0);
        dd.mat.set(7,3, s*(-(SQRT_2*(a[5] - a[8])*b[1]) - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
        dd.mat.set(7,5, s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.mat.set(7,6, s*(SQRT_2*(a[5] - a[8])*b[1] - (a[4] - a[7])*(b[3] - b[6]) - (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(7,7, s*(-(a[2]*b[1]) - a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
        dd.mat.set(7,8, s*(-(SQRT_2*(a[3] - a[6])*b[2]) - SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.mat.set(8,0, s*(-(a[5]*b[0]) + a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.mat.set(8,1, s*(-((a[4] - a[7])*(b[3] + b[6])) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.mat.set(8,2, s*(a[5]*b[2] + a[8]*b[2] - a[2]*(b[5] + b[8]))/2.0);
        dd.mat.set(8,3, s*(-(SQRT_2*(a[4] - a[7])*b[0]) - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(8,4, s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);
        dd.mat.set(8,6, s*(-(SQRT_2*(a[4] - a[7])*b[0]) + (a[5] - a[8])*(b[3] + b[6]) - SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(8,7, s*(-(SQRT_2*(a[3] + a[6])*b[2]) - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(8,8, s*(-(a[2]*b[0]) - a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
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
/// * `dd` -- The resulting tensor (minor-symmetric); it must be [Mandel::Symmetric]
///
/// # Input
///
/// * `aa` -- Second-order tensor, symmetric or not.
/// 
/// # Panics
/// 
/// A panic will occur if `dd` is not [Mandel::Symmetric]
#[rustfmt::skip]
pub fn t2_ssd(dd: &mut Tensor4, s: f64, aa: &Tensor2) {
    assert_eq!(dd.mandel, Mandel::Symmetric);
    let dim = aa.vec.dim();
    let a = &aa.vec;
    if dim == 4 {
        dd.mat.set(0,0, s*(2.0*a[0]*a[0]));
        dd.mat.set(0,1, s*(a[3]*a[3]));
        dd.mat.set(0,2, 0.0);
        dd.mat.set(0,3, s*(2.0*a[0]*a[3]));
        dd.mat.set(0,4, 0.0);
        dd.mat.set(0,5, 0.0);

        dd.mat.set(1,0, s*(a[3]*a[3]));
        dd.mat.set(1,1, s*(2.0*a[1]*a[1]));
        dd.mat.set(1,2, 0.0);
        dd.mat.set(1,3, s*(2.0*a[1]*a[3]));
        dd.mat.set(1,4, 0.0);
        dd.mat.set(1,5, 0.0);

        dd.mat.set(2,0, 0.0);
        dd.mat.set(2,1, 0.0);
        dd.mat.set(2,2, s*(2.0*a[2]*a[2]));
        dd.mat.set(2,3, 0.0);
        dd.mat.set(2,4, 0.0);
        dd.mat.set(2,5, 0.0);

        dd.mat.set(3,0, s*(2.0*a[0]*a[3]));
        dd.mat.set(3,1, s*(2.0*a[1]*a[3]));
        dd.mat.set(3,2, 0.0);
        dd.mat.set(3,3, s*(2.0*a[0]*a[1] + a[3]*a[3]));
        dd.mat.set(3,4, 0.0);
        dd.mat.set(3,5, 0.0);

        dd.mat.set(4,0, 0.0);
        dd.mat.set(4,1, 0.0);
        dd.mat.set(4,2, 0.0);
        dd.mat.set(4,3, 0.0);
        dd.mat.set(4,4, s*(2.0*a[1]*a[2]));
        dd.mat.set(4,5, s*(SQRT_2*a[2]*a[3]));

        dd.mat.set(5,0, 0.0);
        dd.mat.set(5,1, 0.0);
        dd.mat.set(5,2, 0.0);
        dd.mat.set(5,3, 0.0);
        dd.mat.set(5,4, s*(SQRT_2*a[2]*a[3]));
        dd.mat.set(5,5, s*(2.0*a[0]*a[2]));
    } else if dim == 6 {
        dd.mat.set(0,0, s*(2.0*a[0]*a[0]));
        dd.mat.set(0,1, s*(a[3]*a[3]));
        dd.mat.set(0,2, s*(a[5]*a[5]));
        dd.mat.set(0,3, s*(2.0*a[0]*a[3]));
        dd.mat.set(0,4, s*(SQRT_2*a[3]*a[5]));
        dd.mat.set(0,5, s*(2.0*a[ 0]*a[5]));

        dd.mat.set(1,0, s*(a[3]*a[3]));
        dd.mat.set(1,1, s*(2.0*a[1]*a[1]));
        dd.mat.set(1,2, s*(a[4]*a[4]));
        dd.mat.set(1,3, s*(2.0*a[1]*a[3]));
        dd.mat.set(1,4, s*(2.0*a[1]*a[4]));
        dd.mat.set(1,5, s*(SQRT_2*a[3]*a[4]));

        dd.mat.set(2,0, s*(a[5]*a[5]));
        dd.mat.set(2,1, s*(a[4]*a[4]));
        dd.mat.set(2,2, s*(2.0*a[2]*a[2]));
        dd.mat.set(2,3, s*(SQRT_2*a[4]*a[ 5]));
        dd.mat.set(2,4, s*(2.0*a[2]*a[4]));
        dd.mat.set(2,5, s*(2.0*a[2]*a[5]));

        dd.mat.set(3,0, s*(2.0*a[0]*a[3]));
        dd.mat.set(3,1, s*(2.0*a[1]*a[3]));
        dd.mat.set(3,2, s*(SQRT_2*a[4]* a[5]));
        dd.mat.set(3,3, s*(2.0*a[0]*a[1] + a[3]*a[3]));
        dd.mat.set(3,4, s*(a[3]*a[4] + SQRT_2*a[1]*a[5]));
        dd.mat.set(3,5, s*(SQRT_2*a[0]*a[4] + a[3]*a[5]));

        dd.mat.set(4,0, s*(SQRT_2*a[3]*a[5]));
        dd.mat.set(4,1, s*(2.0*a[1]*a[4]));
        dd.mat.set(4,2, s*(2.0*a[2]*a[4]));
        dd.mat.set(4,3, s*(a[3]*a[4] + SQRT_2*a[1]*a[5]));
        dd.mat.set(4,4, s*(2.0*a[1]*a[2] + a[4]*a[4]));
        dd.mat.set(4,5, s*(SQRT_2*a[2]*a[3] + a[4]*a[5]));

        dd.mat.set(5,0, s*(2.0*a[0]*a[5]));
        dd.mat.set(5,1, s*(SQRT_2*a[3]*a[4]));
        dd.mat.set(5,2, s*(2.0*a[2]*a[5]));
        dd.mat.set(5,3, s*(SQRT_2*a[0]* a[4] + a[3]*a[5]));
        dd.mat.set(5,4, s*(SQRT_2*a[2]*a[3] + a[4]*a[5]));
        dd.mat.set(5,5, s*(2.0*a[0]*a[2] + a[5]*a[5]));
    } else {
        dd.mat.set(0,0, s*(2.0*a[0]*a[0]));
        dd.mat.set(0,1, s*((a[3] + a[6])*(a[3] + a[6])));
        dd.mat.set(0,2, s*((a[5] + a[8])*(a[5] + a[8])));
        dd.mat.set(0,3, s*(2.0*a[0]*(a[3] + a[6])));
        dd.mat.set(0,4, s*(SQRT_2*(a[3] + a[6])*(a[5] + a[8])));
        dd.mat.set(0,5, s*(2.0*a[0]*(a[5] + a[8])));

        dd.mat.set(1,0, s*((a[3] - a[6])*(a[3] - a[6])));
        dd.mat.set(1,1, s*(2.0*a[1]*a[1]));
        dd.mat.set(1,2, s*((a[4] + a[7])*(a[4] + a[7])));
        dd.mat.set(1,3, s*(2.0*a[1]*(a[3] - a[6])));
        dd.mat.set(1,4, s*(2.0*a[1]*(a[4] + a[7])));
        dd.mat.set(1,5, s*(SQRT_2*(a[3] - a[6])*(a[4] + a[7])));

        dd.mat.set(2,0, s*((a[5] - a[8])*(a[5] - a[8])));
        dd.mat.set(2,1, s*((a[4] - a[7])*(a[4] - a[7])));
        dd.mat.set(2,2, s*(2.0*a[2]*a[2]));
        dd.mat.set(2,3, s*(SQRT_2*(a[4] - a[7])*(a[5] - a[8])));
        dd.mat.set(2,4, s*(2.0*a[2]*(a[4] - a[7])));
        dd.mat.set(2,5, s*(2.0*a[2]*(a[5] - a[8])));

        dd.mat.set(3,0, s*(2.0*a[0]*(a[3] - a[6])));
        dd.mat.set(3,1, s*(2.0*a[1]*(a[3] + a[6])));
        dd.mat.set(3,2, s*(SQRT_2*(a[4] + a[7])*(a[5] + a[8])));
        dd.mat.set(3,3, s*(2.0*a[0]*a[1] + a[3]*a[3] - a[6]*a[6]));
        dd.mat.set(3,4, s*((a[3] + a[6])*(a[4] + a[7]) + SQRT_2*a[1]*(a[5] + a[8])));
        dd.mat.set(3,5, s*(SQRT_2*a[0]*(a[4] + a[7]) + (a[3] - a[6])*(a[5] + a[8])));

        dd.mat.set(4,0, s*(SQRT_2*(a[3] - a[6])*(a[5] - a[8])));
        dd.mat.set(4,1, s*(2.0*a[1]*(a[4] - a[7])));
        dd.mat.set(4,2, s*(2.0*a[2]*(a[4] + a[7])));
        dd.mat.set(4,3, s*((a[3] - a[6])*(a[4] - a[7]) + SQRT_2*a[1]*(a[5] - a[8])));
        dd.mat.set(4,4, s*(2.0*a[1]*a[2] + a[4]*a[4] - a[7]*a[7]));
        dd.mat.set(4,5, s*(SQRT_2*a[2]*(a[3] - a[6]) + (a[4] + a[7])*(a[5] - a[8])));

        dd.mat.set(5,0, s*(2.0*a[0]*(a[5] - a[8])));
        dd.mat.set(5,1, s*(SQRT_2*(a[3] + a[6])*(a[4] - a[7])));
        dd.mat.set(5,2, s*(2.0*a[2]*(a[5] + a[8])));
        dd.mat.set(5,3, s*(SQRT_2*a[0]*(a[4] - a[7]) + (a[3] + a[6])*(a[5] - a[8])));
        dd.mat.set(5,4, s*(SQRT_2*a[2]*(a[3] + a[6]) + (a[4] - a[7])*(a[5] + a[8])));
        dd.mat.set(5,5, s*(2.0*a[0]*a[2] + a[5]*a[5] - a[8]*a[8]));
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
/// * `dd` -- The resulting tensor (minor-symmetric); it must be [Mandel::Symmetric]
///
/// # Input
///
/// * `aa` -- Second-order tensor, symmetric or not; with the same [Mandel] as `bb`
/// * `bb` -- Second-order tensor, symmetric or not; with the same [Mandel] as `aa`
/// 
/// # Panics
/// 
/// 1. A panic will occur if `dd` is not [Mandel::Symmetric]
/// 2. A panic will occur `aa` and `bb` have different [Mandel]
#[rustfmt::skip]
pub fn t2_qsd_t2(dd: &mut Tensor4, s: f64, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(dd.mandel, Mandel::Symmetric);
    assert_eq!(bb.mandel, aa.mandel);
    let dim = aa.vec.dim();
    let a = &aa.vec;
    let b = &bb.vec;
    if dim == 4 {
        dd.mat.set(0,0, s*(4.0*a[0]*b[0]));
        dd.mat.set(0,1, s*(2.0*a[3]*b[3]));
        dd.mat.set(0,2, 0.0);
        dd.mat.set(0,3, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
        dd.mat.set(0,4, 0.0);
        dd.mat.set(0,5, 0.0);

        dd.mat.set(1,0, s*(2.0*a[3]*b[3]));
        dd.mat.set(1,1, s*(4.0*a[1]*b[1]));
        dd.mat.set(1,2, 0.0);
        dd.mat.set(1,3, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
        dd.mat.set(1,4, 0.0);
        dd.mat.set(1,5, 0.0);

        dd.mat.set(2,0, 0.0);
        dd.mat.set(2,1, 0.0);
        dd.mat.set(2,2, s*(4.0*a[2]*b[2]));
        dd.mat.set(2,3, 0.0);
        dd.mat.set(2,4, 0.0);
        dd.mat.set(2,5, 0.0);

        dd.mat.set(3,0, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
        dd.mat.set(3,1, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
        dd.mat.set(3,2, 0.0);
        dd.mat.set(3,3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])));
        dd.mat.set(3,4, 0.0);
        dd.mat.set(3,5, 0.0);

        dd.mat.set(4,0, 0.0);
        dd.mat.set(4,1, 0.0);
        dd.mat.set(4,2, 0.0);
        dd.mat.set(4,3, 0.0);
        dd.mat.set(4,4, s*(2.0*(a[2]*b[1] + a[1]*b[2])));
        dd.mat.set(4,5, s*(SQRT_2*(a[3]*b[2] + a[2]*b[3])));

        dd.mat.set(5,0, 0.0);
        dd.mat.set(5,1, 0.0);
        dd.mat.set(5,2, 0.0);
        dd.mat.set(5,3, 0.0);
        dd.mat.set(5,4, s*(SQRT_2*(a[3]*b[2] + a[2]*b[3])));
        dd.mat.set(5,5, s*(2.0*(a[2]*b[0] + a[0]*b[2])));
    } else if dim == 6 {
        dd.mat.set(0,0, s*(4.0*a[0]*b[0]));
        dd.mat.set(0,1, s*(2.0*a[3]*b[3]));
        dd.mat.set(0,2, s*(2.0*a[5]*b[5]));
        dd.mat.set(0,3, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
        dd.mat.set(0,4, s*(SQRT_2*(a[5]*b[3] + a[3]*b[5])));
        dd.mat.set(0,5, s*(2.0*(a[5]*b[0] + a[0]*b[5])));

        dd.mat.set(1,0, s*(2.0*a[3]*b[3]));
        dd.mat.set(1,1, s*(4.0*a[1]*b[1]));
        dd.mat.set(1,2, s*(2.0*a[4]*b[4]));
        dd.mat.set(1,3, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
        dd.mat.set(1,4, s*(2.0*(a[4]*b[1] + a[1]*b[4])));
        dd.mat.set(1,5, s*(SQRT_2*(a[4]*b[3] + a[3]*b[4])));

        dd.mat.set(2,0, s*(2.0*a[5]*b[5]));
        dd.mat.set(2,1, s*(2.0*a[4]*b[4]));
        dd.mat.set(2,2, s*(4.0*a[2]*b[2]));
        dd.mat.set(2,3, s*(SQRT_2*(a[5]*b[4] + a[4]*b[5])));
        dd.mat.set(2,4, s*(2.0*(a[4]*b[2] + a[2]*b[4])));
        dd.mat.set(2,5, s*(2.0*(a[5]*b[2] + a[2]*b[5])));

        dd.mat.set(3,0, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
        dd.mat.set(3,1, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
        dd.mat.set(3,2, s*(SQRT_2*(a[5]*b[4] + a[4]*b[5])));
        dd.mat.set(3,3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])));
        dd.mat.set(3,4, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5]));
        dd.mat.set(3,5, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5]));

        dd.mat.set(4,0, s*(SQRT_2*(a[5]*b[3] + a[3]*b[5])));
        dd.mat.set(4,1, s*(2.0*(a[4]*b[1] + a[1]*b[4])));
        dd.mat.set(4,2, s*(2.0*(a[4]*b[2] + a[2]*b[4])));
        dd.mat.set(4,3, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5]));
        dd.mat.set(4,4, s*(2.0*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])));
        dd.mat.set(4,5, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5]));

        dd.mat.set(5,0, s*(2.0*(a[5]*b[0] + a[0]*b[5])));
        dd.mat.set(5,1, s*(SQRT_2*(a[4]*b[3] + a[3]*b[4])));
        dd.mat.set(5,2, s*(2.0*(a[5]*b[2] + a[2]*b[5])));
        dd.mat.set(5,3, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5]));
        dd.mat.set(5,4, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5]));
        dd.mat.set(5,5, s*(2.0*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])));
    } else {
        dd.mat.set(0,0, s*(4.0*a[0]*b[0]));
        dd.mat.set(0,1, s*(2.0*(a[3] + a[6])*(b[3] + b[6])));
        dd.mat.set(0,2, s*(2.0*(a[5] + a[8])*(b[5] + b[8])));
        dd.mat.set(0,3, s*(2.0*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))));
        dd.mat.set(0,4, s*(SQRT_2*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))));
        dd.mat.set(0,5, s*(2.0*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))));

        dd.mat.set(1,0, s*(2.0*(a[3] - a[6])*(b[3] - b[6])));
        dd.mat.set(1,1, s*(4.0*a[1]*b[1]));
        dd.mat.set(1,2, s*(2.0*(a[4] + a[7])*(b[4] + b[7])));
        dd.mat.set(1,3, s*(2.0*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))));
        dd.mat.set(1,4, s*(2.0*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))));
        dd.mat.set(1,5, s*(SQRT_2*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))));

        dd.mat.set(2,0, s*(2.0*(a[5] - a[8])*(b[5] - b[8])));
        dd.mat.set(2,1, s*(2.0*(a[4] - a[7])*(b[4] - b[7])));
        dd.mat.set(2,2, s*(4.0*a[2]*b[2]));
        dd.mat.set(2,3, s*(SQRT_2*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))));
        dd.mat.set(2,4, s*(2.0*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))));
        dd.mat.set(2,5, s*(2.0*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))));

        dd.mat.set(3,0, s*(2.0*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))));
        dd.mat.set(3,1, s*(2.0*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))));
        dd.mat.set(3,2, s*(SQRT_2*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))));
        dd.mat.set(3,3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])));
        dd.mat.set(3,4, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8])));
        dd.mat.set(3,5, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8])));

        dd.mat.set(4,0, s*(SQRT_2*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))));
        dd.mat.set(4,1, s*(2.0*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))));
        dd.mat.set(4,2, s*(2.0*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))));
        dd.mat.set(4,3, s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8])));
        dd.mat.set(4,4, s*(2.0*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])));
        dd.mat.set(4,5, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8])));

        dd.mat.set(5,0, s*(2.0*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))));
        dd.mat.set(5,1, s*(SQRT_2*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))));
        dd.mat.set(5,2, s*(2.0*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))));
        dd.mat.set(5,3, s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8])));
        dd.mat.set(5,4, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8])));
        dd.mat.set(5,5, s*(2.0*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mandel, MN_TO_IJKL};
    use russell_lab::{mat_approx_eq, Matrix};

    #[test]
    #[should_panic]
    fn t2_odyad_t2_panics_on_non_general() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric2D); // wrong; it must be General
        t2_odyad_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_odyad_t2_panics_on_different_mandel() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `a`
        let mut dd = Tensor4::new(Mandel::General);
        t2_odyad_t2(&mut dd, 1.0, &a, &b);
    }

    fn check_odyad(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_matrix();
        let b = b_ten.as_matrix();
        let dd = dd_ten.as_matrix();
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
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_odyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Mandel::Symmetric).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_odyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_odyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
    #[should_panic]
    fn t2_udyad_t2_panics_on_non_general() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric2D); // wrong; it must be General
        t2_udyad_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_udyad_t2_panics_on_different_mandel() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `a`
        let mut dd = Tensor4::new(Mandel::General);
        t2_udyad_t2(&mut dd, 1.0, &a, &b);
    }

    fn check_udyad(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_matrix();
        let b = b_ten.as_matrix();
        let dd = dd_ten.as_matrix();
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
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_udyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Mandel::Symmetric).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_udyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_udyad_t2(&mut dd, 2.0, &a, &b);
        let mandel_mat = Matrix::from(&[
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
        mat_approx_eq(&dd.mat, &mandel_mat, 1e-14);
        let mat = dd.as_matrix();
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
    fn t2_ssd_panics_on_non_sym() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric2D); // wrong; it must be Symmetric
        t2_ssd(&mut dd, 1.0, &a);
    }

    fn check_ssd(s: f64, a_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_matrix();
        let dd = dd_ten.as_matrix();
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
    fn t2_ssd_works() {
        // general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_ssd(&mut dd, 2.0, &a);
        let mat = dd.as_matrix();
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
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Mandel::Symmetric).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_ssd(&mut dd, 2.0, &a);
        let mat = dd.as_matrix();
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
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_ssd(&mut dd, 2.0, &a);
        let mat = dd.as_matrix();
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
    fn t2_qsd_t2_panics_on_non_sym() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric2D); // wrong; it must be Symmetric
        t2_qsd_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_qsd_t2_panics_on_different_mandel() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `a`
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_qsd_t2(&mut dd, 1.0, &a, &b);
    }

    fn check_qsd(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_matrix();
        let b = b_ten.as_matrix();
        let dd = dd_ten.as_matrix();
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
    fn t2_qsd_t2_works() {
        // general qsd general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_qsd_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Mandel::Symmetric).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_qsd_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_qsd_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
