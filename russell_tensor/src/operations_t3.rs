use super::{Tensor2, Tensor3};
use russell_lab::{StrError, Vector};

/// Adds two third-order tensors
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// # Panics
///
/// A panic will occur if the tensors have different [Rep]
pub fn t3_add(c: &mut Tensor3, alpha: f64, a: &Tensor3, beta: f64, b: &Tensor3) {
    assert_eq!(b.rep, a.rep);
    assert_eq!(c.rep, a.rep);
    for i in 0..c.nrow {
        for j in 0..3 {
            c.mat[i][j] = alpha * a.mat[i][j] + beta * b.mat[i][j];
        }
    }
}

/// Performs the single dot operation between a Tensor3 and a vector resulting in a Tensor2
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
/// 1. If `T` and `H` have different [Rep]
/// 2. If `u` does not have 3 components
pub fn t3_dot_vec(tt: &mut Tensor2, alpha: f64, hh: &Tensor3, u: &Vector) {
    assert_eq!(tt.rep, hh.rep);
    assert_eq!(u.dim(), 3);
    for m in 0..tt.dim {
        tt.vec[m] = alpha * (hh.mat[m][0] * u[0] + hh.mat[m][1] * u[1] + hh.mat[m][2] * u[2]);
    }
}

/// Performs the single dot operation between a vector and a Tensor3 resulting in a Tensor2
///
/// Computes:
///
/// ```text
/// T = α u · H
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Tⱼₖ = α Σ uᵢ Hᵢⱼₖ
///        i
/// ```
///
/// Or, in Kelvin basis:
///
/// ```text
/// Tₘ = α Σ uₖ Hₘₖ (WRONG)
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
/// * `u` -- the 3D vector; must have 3 components (w.r.t. standard Cartesian basis)
/// * `hh` -- the third-order tensor
///
/// # Panics
///
/// 1. If `T` and `H` have different [Rep]
/// 2. If `u` does not have 3 components
pub fn vec_dot_t3(tt: &mut Tensor2, alpha: f64, u: &Vector, hh: &Tensor3) {
    // TODO
}

/// Performs the dyadic product between a Tensor2 and a vector resulting in a Tensor3
///
/// Computes:
///
/// ```text
/// H = α T ⊗ u
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Hᵢⱼₖ = α Tᵢⱼ uₖ
/// ```
///
/// # Output
///
/// * `H` -- the resulting third-order tensor
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `T` -- the second-order tensor with the same [Rep] as `H`
/// * `u` -- the 3D vector; must have 3 components (w.r.t. standard Cartesian basis)
///
/// # Panics
///
/// 1. If `T` and `H` have different [Rep]
/// 2. If `u` does not have 3 components
pub fn t2_dyad_vec(hh: &mut Tensor3, alpha: f64, tt: &Tensor2, u: &Vector) -> Result<(), StrError> {
    // TODO
    Ok(())
}

/// Performs the dyadic product between a vector and a Tensor2 resulting in a Tensor3
///
/// Computes:
///
/// ```text
/// H = α u ⊗ T
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Hᵢⱼₖ = α uᵢ Tⱼₖ
/// ```
///
/// # Output
///
/// * `H` -- the resulting third-order tensor
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `T` -- the second-order tensor with the same [Rep] as `H`
/// * `u` -- the 3D vector; must have 3 components (w.r.t. standard Cartesian basis)
///
/// # Panics
///
/// 1. If `T` and `H` have different [Rep]
/// 2. If `u` does not have 3 components
pub fn vec_dyad_t2(hh: &mut Tensor3, alpha: f64, u: &Vector, tt: &Tensor2) -> Result<(), StrError> {
    // TODO
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {}
