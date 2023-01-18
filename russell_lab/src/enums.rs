/// Options to compute vector and matrix norms
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
    /// ‖u‖_2 = sqrt(Σ_i |uᵢ|⋅|uᵢ|)
    /// ```
    Euc,

    /// Specifies the Frobenius-norm (2-norm)
    ///
    /// **matrix**
    ///
    /// ```text
    /// ‖a‖_F = sqrt(Σ_i Σ_j |aᵢⱼ|⋅|aᵢⱼ|) == ‖a‖_2
    /// ```
    ///
    /// **vector**
    ///
    /// Will compute the same as [Norm::Euc]
    Fro,

    /// Specifies the Inf-norm
    ///
    /// **matrix**
    ///
    /// ```text
    /// ‖a‖_∞ = max_i ( Σ_j |aᵢⱼ| )
    /// ```
    ///
    /// **vector**
    ///
    /// Will compute that same as [Norm::Max]
    Inf,

    /// Specifies the Max-norm
    ///
    /// **matrix**
    ///
    /// ```text
    /// ‖a‖_max = max_ij ( |aᵢⱼ| )
    /// ```
    ///
    /// **vector**
    ///
    /// ```text
    /// ‖u‖_max = max_i ( |uᵢ| ) == ‖u‖_∞
    /// ```
    Max,

    /// Specifies the 1-norm
    ///
    /// **matrix**
    ///
    /// ```text
    /// ‖a‖_1 = max_j ( Σ_i |aᵢⱼ| )
    /// ```
    ///
    /// **vector** (taxicab or sum of abs values)
    ///
    /// ```text
    /// ‖u‖_1 := sum_i |uᵢ|
    /// ```
    One,
}
