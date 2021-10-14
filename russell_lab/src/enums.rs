/// Options to compute matrix norm
pub enum NormMat {
    /// 1-norm
    ///
    /// ‖a‖_1 = max_j ( Σ_i |aᵢⱼ| )
    One,

    /// inf-norm
    ///
    /// ‖a‖_∞ = max_i ( Σ_j |aᵢⱼ| )
    Inf,

    /// Frobenius-norm (2-norm)
    ///
    /// ‖a‖_F = sqrt(Σ_i Σ_j aᵢⱼ⋅aᵢⱼ) == ‖a‖_2
    Fro,

    /// max-norm
    ///
    /// ‖a‖_max = max_ij ( |aᵢⱼ| )
    Max,
}

/// Options to compute vector norm
pub enum NormVec {
    /// 1-norm (taxicab or sum of abs values)
    ///
    /// ‖u‖_1 := sum_i |uᵢ|
    One,

    /// Euclidean-norm
    ///
    /// ‖u‖_2 = sqrt(Σ_i uᵢ⋅uᵢ)
    Euc,

    /// max-norm (inf-norm)
    ///
    /// ‖u‖_max = max_i ( |uᵢ| ) == ‖u‖_∞
    Max,
}
