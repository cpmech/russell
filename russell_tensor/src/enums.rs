pub enum EigenVals3 {
    /// Holds the fully coalescent eigenvalue λ = λ0 = λ1 = λ2
    Spherical(f64),

    /// Holds the non-repeated and the coalescent eigenvalue. λ0 = λ1* > λ2*
    ///
    /// Holds (λ1, λ2)
    Repeat01(f64, f64),

    /// Holds the non-repeated and the coalescent eigenvalue. λ0* > λ1* = λ2
    ///
    /// Holds (λ0, λ1)
    Repeat12(f64, f64),

    /// Holds all distinct eigenvalues. λ0 > λ1 > λ2
    Distinct(f64, f64, f64),
}

impl EigenVals3 {
    // pub fn index_
}
