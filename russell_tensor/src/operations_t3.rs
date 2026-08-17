use super::{Tensor2, Tensor3};
use russell_lab::Vector;

#[allow(unused)]
use crate::Rep; // for documentation

/// Adds two third-order tensors
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// # Panics
///
/// A panic will occur if the tensors have different [Rep] or different case flag
pub fn t3_add(c: &mut Tensor3, alpha: f64, a: &Tensor3, beta: f64, b: &Tensor3) {
    assert_eq!(b.rep, a.rep);
    assert_eq!(c.rep, a.rep);
    assert_eq!(b.case_a, a.case_a);
    assert_eq!(c.case_a, a.case_a);
    for i in 0..a.nrow {
        for j in 0..a.ncol {
            c.mat[i][j] = alpha * a.mat[i][j] + beta * b.mat[i][j];
        }
    }
}

/// Performs the single-dot operation between a Tensor3 and a vector resulting in a Tensor2 (Case A)
///
/// Computes:
///
/// ```text
/// T = α H · u
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Tᵢⱼ = α Σ Hᵢⱼₖ uₖ
///        k
/// ```
///
/// Or, in Kelvin basis:
///
/// ```text
/// Tₘ = α Σ Hₘₖ uₖ
///        k
/// ```
///
/// # Output
///
/// * `T` -- the resulting second-order tensor; with the same [Rep] as `H`
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `hh` -- the third-order tensor
/// * `u` -- the 3D vector; must have 3 components (w.r.t. standard Cartesian basis)
///
/// # Panics
///
/// 1. If `H` was not allocated for Case A
/// 2. If `T` and `H` have different [Rep]
/// 3. If `u` does not have 3 components
pub fn t3_dot_vec(tt: &mut Tensor2, alpha: f64, hh: &Tensor3, u: &Vector) {
    assert!(hh.case_a);
    assert_eq!(tt.rep, hh.rep);
    assert_eq!(u.dim(), 3);
    for m in 0..hh.nrow {
        tt.vec[m] = alpha * (hh.mat[m][0] * u[0] + hh.mat[m][1] * u[1] + hh.mat[m][2] * u[2]);
    }
}

/// Performs the double-dot operation between a Tensor3 and a Tensor2 resulting in a vector (Case B)
///
/// Computes:
///
/// ```text
/// u = α H : T
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// uᵢ = α Σ Σ Hᵢⱼₖ Tⱼₖ
///       j k
/// ```
///
/// Or, in Kelvin basis:
///
/// ```text
/// uᵢ = α Σ Hᵢₙ Tₙ
///       n
/// ```
///
/// # Output
///
/// * `u` -- the resulting vector (with 3 standard components)
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `hh` -- the third-order tensor
/// * `T` -- the second-order tensor with the same [Rep] as `H`
///
/// # Panics
///
/// 1. If `H` was not allocated for Case B
/// 2. If `T` and `H` have different [Rep]
/// 3. If `u` does not have 3 components
pub fn t3_dot_t2(u: &mut Vector, alpha: f64, hh: &Tensor3, tt: &Tensor2) {
    assert!(!hh.case_a);
    assert_eq!(tt.rep, hh.rep);
    assert_eq!(u.dim(), 3);
    u[0] = 0.0;
    u[1] = 0.0;
    u[2] = 0.0;
    for n in 0..hh.ncol {
        u[0] += alpha * hh.mat[0][n] * tt.vec[n];
        u[1] += alpha * hh.mat[1][n] * tt.vec[n];
        u[2] += alpha * hh.mat[2][n] * tt.vec[n];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {}
