use crate::internal::{NORM_EUC, NORM_FRO, NORM_INF, NORM_MAX, NORM_ONE};

/// Options to compute vector and matrix norms
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Norm {
    /// Specifies the Euclidean-norm
    ///
    /// **matrix**
    ///
    /// Will compute the same as [Norm::Fro]
    ///
    /// **vector**
    ///
    /// ```text
    /// ‖u‖₂ = sqrt(Σ_i |uᵢ|⋅|uᵢ|)
    /// ```
    Euc = NORM_EUC,

    /// Specifies the Frobenius-norm (2-norm)
    ///
    /// **matrix**
    ///
    /// ```text
    /// ‖a‖F = sqrt(Σ_i Σ_j |aᵢⱼ|⋅|aᵢⱼ|) == ‖a‖₂
    /// ```
    ///
    /// **vector**
    ///
    /// Will compute the same as [Norm::Euc]
    Fro = NORM_FRO,

    /// Specifies the Inf-norm
    ///
    /// **matrix**
    ///
    /// ```text
    /// ‖a‖∞ = max_i ( Σ_j |aᵢⱼ| )
    /// ```
    ///
    /// **vector**
    ///
    /// Will compute that same as [Norm::Max]
    Inf = NORM_INF,

    /// Specifies the Max-norm
    ///
    /// **matrix**
    ///
    /// ```text
    /// ‖a‖ₘₐₓ = max_ij ( |aᵢⱼ| )
    /// ```
    ///
    /// **vector**
    ///
    /// ```text
    /// ‖u‖ₘₐₓ = max_i ( |uᵢ| ) == ‖u‖_∞
    /// ```
    Max = NORM_MAX,

    /// Specifies the 1-norm
    ///
    /// **matrix**
    ///
    /// ```text
    /// ‖a‖₁ = max_j ( Σ_i |aᵢⱼ| )
    /// ```
    ///
    /// **vector** (taxicab or sum of abs values)
    ///
    /// ```text
    /// ‖u‖₁ := sum_i |uᵢ|
    /// ```
    One = NORM_ONE,
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Norm;

    #[test]
    fn clone_copy_and_debug_work() {
        let norm = Norm::Inf;
        let copy = norm;
        let clone = norm.clone();
        assert_eq!(format!("{:?}", norm), "Inf");
        assert_eq!(copy, Norm::Inf);
        assert_eq!(clone, Norm::Inf);
    }
}
