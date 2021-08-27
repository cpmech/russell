pub enum EnumMatrixNorm {
    /// 1-norm
    ///
    /// ‖a‖_1 = max_j ( Σ_i |aij| )
    One,

    /// inf-norm
    ///
    /// ‖a‖_∞ = max_i ( Σ_j |aij| )
    Inf,

    /// Frobenius-norm
    ///
    /// ‖a‖_F = sqrt(Σ_i Σ_j aij⋅aij) == ‖a‖_2
    Fro,

    /// max-norm
    ///
    /// ‖a‖_max = max_ij ( |aij| )
    Max,
}
