use super::Vector;
use crate::NormVec;
use russell_openblas::{dasum, dnrm2, idamax, to_i32};

/// Returns the vector norm
///
/// Computes one of:
///
/// ```text
/// One:  1-norm (taxicab or sum of abs values)
///
///       ‖u‖_1 := sum_i |uᵢ|
/// ```
///
/// ```text
/// Euc:  Euclidean-norm
///
///       ‖u‖_2 = sqrt(Σ_i uᵢ⋅uᵢ)
/// ```
///
/// ```text
/// Max:  max-norm (inf-norm)
///
///       ‖u‖_max = max_i ( |uᵢ| ) == ‖u‖_∞
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{vector_norm, NormVec, Vector};
///
/// fn main() {
///     let u = Vector::from(&[2.0, -2.0, 2.0, -2.0, -3.0]);
///     assert_eq!(vector_norm(&u, NormVec::One), 11.0);
///     assert_eq!(vector_norm(&u, NormVec::Euc), 5.0);
///     assert_eq!(vector_norm(&u, NormVec::Max), 3.0);
/// }
/// ```
pub fn vector_norm(v: &Vector, kind: NormVec) -> f64 {
    let n = to_i32(v.dim());
    if n == 0 {
        return 0.0;
    }
    match kind {
        NormVec::One => dasum(n, &v.as_data(), 1),
        NormVec::Euc => dnrm2(n, &v.as_data(), 1),
        NormVec::Max => {
            let idx = idamax(n, &v.as_data(), 1);
            f64::abs(v.get(idx as usize))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vector_norm, Vector};
    use crate::NormVec;

    #[test]
    fn vector_norm_works() {
        let u0 = Vector::new(0);
        assert_eq!(vector_norm(&u0, NormVec::One), 0.0);
        assert_eq!(vector_norm(&u0, NormVec::Euc), 0.0);
        assert_eq!(vector_norm(&u0, NormVec::Max), 0.0);
        let u = Vector::from(&[-3.0, 2.0, 1.0, 1.0, 1.0]);
        assert_eq!(vector_norm(&u, NormVec::One), 8.0);
        assert_eq!(vector_norm(&u, NormVec::Euc), 4.0);
        assert_eq!(vector_norm(&u, NormVec::Max), 3.0);
    }
}
