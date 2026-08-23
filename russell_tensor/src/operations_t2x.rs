use super::Tensor2;
use crate::{Rep, StrError, SQRT_2};

/// Performs the matrix multiplication between two Tensor2
///
/// # Supported Combinations
///
/// | Formula | `a` Rep | `b` Rep | `c` Rep | `tra_a` | `tra_b` | Equivalent/Notes |
/// | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
/// | C = α A · B | `General` | `General` | `General` | `false` | `false` | Standard dot product |
/// | C = α Aᵀ · B | `General` | `General` | `General` | `true` | `false` | Left transpose |
/// | C = α A · Bᵀ | `General` | `General` | `General` | `false` | `true` | Right transpose |
/// | C = α Aᵀ · A | `General` | (same as `a`) | `Symmetric` | `true` | `false` | Tensor multiplied by itself |
/// | C = α sym(Aᵀ · B) | `General` | `General` | `Symmetric` | `true` | `false` | Skew components chopped |
/// | C = α sym(A · Bᵀ) | `General` | `General` | `Symmetric` | `false` | `true` | Skew components chopped |
/// | C = α A · B | `General` | `Symmetric` | `General` | `false` | `any` | Transposing B is a no-op |
/// | C = α Aᵀ · B | `General` | `Symmetric` | `General` | `true` | `any` | |
/// | C = α A · B | `Symmetric` | `General` | `General` | `any` | `false` | Transposing A is a no-op |
/// | C = α A · Bᵀ | `Symmetric` | `General` | `General` | `any` | `true` | |
/// | C = α A · B | `Symmetric` | `Symmetric` | `General` | `any` | `any` | Product is NOT symmetric in general |
///
/// # Output
///
/// * `c` -- the resulting tensor
///
/// # Input
///
/// * `alpha` -- the multiplier α
/// * `a` -- the first tensor A
/// * `tra_a` -- whether to use Aᵀ instead of A
/// * `b` -- the second tensor B
/// * `tra_b` -- whether to use Bᵀ instead of B
///
/// # Returns
///
/// Returns an error if the combination of representations and transpositions is unavailable or impossible.
pub fn t2_matmul(
    c: &mut Tensor2,
    alpha: f64,
    a: &Tensor2,
    tra_a: bool,
    b: &Tensor2,
    tra_b: bool,
) -> Result<(), StrError> {
    match (a.rep(), b.rep(), tra_a, tra_b) {
        (Rep::General, Rep::General, false, false) => {
            if c.rep() != Rep::General {
                return Err("c must be General for this combination");
            }
            t2_gen_dot_gen(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (Rep::General, Rep::Symmetric, false, true) |
        (Rep::General, Rep::Symmetric, false, false) => {
            if c.rep() != Rep::General {
                return Err("c must be General for this combination");
            }
            t2_gen_dot_sym(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (Rep::Symmetric, Rep::General, true, false) |
        (Rep::Symmetric, Rep::General, false, false) => {
            if c.rep() != Rep::General {
                return Err("c must be General for this combination");
            }
            t2_sym_dot_gen(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (Rep::Symmetric, Rep::Symmetric, true, true) |
        (Rep::Symmetric, Rep::Symmetric, true, false) |
        (Rep::Symmetric, Rep::Symmetric, false, true) |
        (Rep::Symmetric, Rep::Symmetric, false, false) => {
            if c.rep() != Rep::General {
                return Err("c must be General for this combination");
            }
            t2_sym_dot_sym(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (Rep::General, Rep::General, true, false) => {
            if std::ptr::eq(a, b) {
                if c.rep() != Rep::Symmetric {
                    return Err("c must be Symmetric");
                }
                t2_gen_tra_dot_self(c.as_mut_data(), alpha, a.as_data());
            } else {
                if c.rep() == Rep::Symmetric {
                    t2_gen_tra_dot_gen_chop(c.as_mut_data(), alpha, a.as_data(), b.as_data(), true);
                } else if c.rep() == Rep::General {
                    t2_gen_tra_dot_gen_chop(c.as_mut_data(), alpha, a.as_data(), b.as_data(), false);
                } else {
                    return Err("c must be Symmetric or General");
                }
            }
        }
        (Rep::General, Rep::General, false, true) => {
            if c.rep() == Rep::Symmetric {
                t2_gen_dot_gen_tra_chop(c.as_mut_data(), alpha, a.as_data(), b.as_data(), true);
            } else if c.rep() == Rep::General {
                t2_gen_dot_gen_tra_chop(c.as_mut_data(), alpha, a.as_data(), b.as_data(), false);
            } else {
                return Err("c must be Symmetric or General");
            }
        }
        _ => return Err("t2_matmul: combination of representations and transpositions is unavailable"),
    }
    Ok(())
}

/// Performs a triple matrix multiplication: C = α A · B · Aᵀ or C = α Aᵀ · B · A
///
/// # Supported Combinations
///
/// | Formula | `a` Rep | `b` Rep | `c` Rep | `forward` | Notes |
/// | :--- | :--- | :--- | :--- | :--- | :--- |
/// | C = α A · B · Aᵀ | `General` | `Symmetric` | `Symmetric` | `true` | e.g., Push-forward of Piola-Kirchhoff stress to Cauchy stress |
/// | C = α Aᵀ · B · A | `General` | `Symmetric` | `Symmetric` | `false` | e.g., Pull-back of Cauchy stress to Piola-Kirchhoff stress |
///
/// # Output
///
/// * `c` -- the resulting tensor
///
/// # Input
///
/// * `alpha` -- the multiplier α
/// * `a` -- the outer tensor A
/// * `forward` -- if true, computes A · B · Aᵀ. If false, computes Aᵀ · B · A
/// * `b` -- the inner tensor B
///
/// # Returns
///
/// Returns an error if the combination is unsupported.
pub fn t2_matmulx(
    c: &mut Tensor2,
    alpha: f64,
    a: &Tensor2,
    forward: bool,
    b: &Tensor2,
) -> Result<(), StrError> {
    match (a.rep(), b.rep(), forward) {
        (Rep::General, Rep::Symmetric, true) => {
            if c.rep() != Rep::Symmetric {
                return Err("c must be Symmetric for this combination");
            }
            t2_gen_dot_sym_dot_self_tra(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (Rep::General, Rep::Symmetric, false) => {
            if c.rep() != Rep::Symmetric {
                return Err("c must be Symmetric for this combination");
            }
            t2_gen_tra_dot_sym_dot_self(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        _ => return Err("t2_matmulx: unsupported combination"),
    }
    Ok(())
}

/// Performs the general tensor dot general tensor operation: C = A · B
/// 
/// Computes:
///
/// ```text
/// C = A · B
/// ```
///
/// # Output
///
/// * `c` -- the resulting tensor C (slice)
///
/// # Input
///
/// * `a` -- the first tensor A (slice)
/// * `b` -- the second tensor B (slice)
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect, `a.len()` is incorrect, or `b.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_dot_gen(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 9);
    debug_assert!(c.len() >= 9);
    c[0] = alpha * ((2.0 * a[0] * b[0] + (a[3] + a[6]) * (b[3] - b[6]) + (a[5] + a[8]) * (b[5] - b[8])) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + (a[3] - a[6]) * (b[3] + b[6]) + (a[4] + a[7]) * (b[4] - b[7])) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + (a[4] - a[7]) * (b[4] + b[7]) + (a[5] - a[8]) * (b[5] + b[8])) / 2.0);
    c[3] = alpha * ((2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] - b[7]) + SQRT_2 * (a[4] + a[7]) * (b[5] - b[8])) / 4.0);
    c[4] = alpha * ((2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] + SQRT_2 * (a[5] - a[8]) * (b[3] + b[6]) + 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] + b[8])) / 4.0);
    c[5] = alpha * ((2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] + SQRT_2 * (a[4] - a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] + b[7]) + 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0);
    c[6] = alpha * ((-2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] - 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] - b[7]) - SQRT_2 * (a[4] + a[7]) * (b[5] - b[8])) / 4.0);
    c[7] = alpha * ((-2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] - SQRT_2 * (a[5] - a[8]) * (b[3] + b[6]) - 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] + b[8])) / 4.0);
    c[8] = alpha * ((-2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] - SQRT_2 * (a[4] - a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] + b[7]) - 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0);
}

/// Performs the general tensor dot symmetric tensor operation: C = A · B
/// 
/// Computes:
///
/// ```text
/// C = A · B
/// ```
///
/// # Output
///
/// * `c` -- the resulting tensor C (slice)
///
/// # Input
///
/// * `a` -- the first tensor A (slice)
/// * `b` -- the second tensor B (slice)
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect, `a.len()` is incorrect, or `b.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_dot_sym(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 6);
    debug_assert!(c.len() >= 9);
    c[0] = alpha * ((2.0 * a[0] * b[0] + (a[3] + a[6]) * b[3] + (a[5] + a[8]) * b[5]) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + (a[3] - a[6]) * b[3] + (a[4] + a[7]) * b[4]) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + (a[4] - a[7]) * b[4] + (a[5] - a[8]) * b[5]) / 2.0);
    c[3] = alpha * ((SQRT_2 * a[6] * (-b[0] + b[1]) + SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[8] * b[4] + a[4] * b[5] + a[7] * b[5]) / (2.0 * SQRT_2));
    c[4] = alpha * ((SQRT_2 * a[7] * (-b[1] + b[2]) + SQRT_2 * a[4] * (b[1] + b[2]) + a[5] * b[3] - a[8] * b[3] + SQRT_2 * a[1] * b[4] + SQRT_2 * a[2] * b[4] + a[3] * b[5] - a[6] * b[5]) / (2.0 * SQRT_2));
    c[5] = alpha * ((SQRT_2 * a[8] * (-b[0] + b[2]) + SQRT_2 * a[5] * (b[0] + b[2]) + a[4] * b[3] - a[7] * b[3] + a[3] * b[4] + a[6] * b[4] + SQRT_2 * a[0] * b[5] + SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2));
    c[6] = alpha * ((-2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[0] * b[3] - 2.0 * a[1] * b[3] + SQRT_2 * (a[5] + a[8]) * b[4] - SQRT_2 * (a[4] + a[7]) * b[5]) / 4.0);
    c[7] = alpha * ((SQRT_2 * a[4] * (-b[1] + b[2]) + SQRT_2 * a[7] * (b[1] + b[2]) - a[5] * b[3] + a[8] * b[3] + SQRT_2 * a[1] * b[4] - SQRT_2 * a[2] * b[4] + a[3] * b[5] - a[6] * b[5]) / (2.0 * SQRT_2));
    c[8] = alpha * ((SQRT_2 * a[5] * (-b[0] + b[2]) + SQRT_2 * a[8] * (b[0] + b[2]) - a[4] * b[3] + a[7] * b[3] + a[3] * b[4] + a[6] * b[4] + SQRT_2 * a[0] * b[5] - SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2));
}

/// Performs the symmetric tensor dot general tensor operation: C = A · B
/// 
/// Computes:
///
/// ```text
/// C = A · B
/// ```
///
/// # Output
///
/// * `c` -- the resulting tensor C (slice)
///
/// # Input
///
/// * `a` -- the first tensor A (slice)
/// * `b` -- the second tensor B (slice)
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect, `a.len()` is incorrect, or `b.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_sym_dot_gen(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 6);
    debug_assert!(b.len() >= 9);
    debug_assert!(c.len() >= 9);
    c[0] = alpha * ((2.0 * a[0] * b[0] + a[3] * (b[3] - b[6]) + a[5] * (b[5] - b[8])) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + a[3] * (b[3] + b[6]) + a[4] * (b[4] - b[7])) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + a[4] * (b[4] + b[7]) + a[5] * (b[5] + b[8])) / 2.0);
    c[3] = alpha * ((SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[4] * b[5] - SQRT_2 * a[1] * b[6] + SQRT_2 * a[0] * (b[3] + b[6]) - a[5] * b[7] - a[4] * b[8]) / (2.0 * SQRT_2));
    c[4] = alpha * ((2.0 * a[4] * b[1] + 2.0 * a[4] * b[2] + SQRT_2 * a[5] * (b[3] + b[6]) + 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * a[3] * (b[5] + b[8])) / 4.0);
    c[5] = alpha * ((2.0 * a[5] * b[0] + 2.0 * a[5] * b[2] + SQRT_2 * a[4] * (b[3] - b[6]) + SQRT_2 * a[3] * (b[4] + b[7]) + 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0);
    c[6] = alpha * ((SQRT_2 * a[3] * (-b[0] + b[1]) - SQRT_2 * a[1] * b[3] + a[5] * b[4] - a[4] * b[5] + SQRT_2 * a[1] * b[6] + SQRT_2 * a[0] * (b[3] + b[6]) - a[5] * b[7] + a[4] * b[8]) / (2.0 * SQRT_2));
    c[7] = alpha * ((-2.0 * a[4] * b[1] + 2.0 * a[4] * b[2] - SQRT_2 * a[5] * (b[3] + b[6]) - 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * a[3] * (b[5] + b[8])) / 4.0);
    c[8] = alpha * ((-2.0 * a[5] * b[0] + 2.0 * a[5] * b[2] - SQRT_2 * a[4] * (b[3] - b[6]) + SQRT_2 * a[3] * (b[4] + b[7]) - 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0);
}

/// Performs the symmetric tensor dot symmetric tensor operation: C = A · B
/// 
/// Computes:
///
/// ```text
/// C = A · B
/// ```
///
/// # Output
///
/// * `c` -- the resulting tensor C (slice)
///
/// # Input
///
/// * `a` -- the first tensor A (slice)
/// * `b` -- the second tensor B (slice)
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect, `a.len()` is incorrect, or `b.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_sym_dot_sym(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 6);
    debug_assert!(b.len() >= 6);
    debug_assert!(c.len() >= 9);
    c[0] = alpha * ((2.0 * a[0] * b[0] + a[3] * b[3] + a[5] * b[5]) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + a[3] * b[3] + a[4] * b[4]) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + a[4] * b[4] + a[5] * b[5]) / 2.0);
    c[3] = alpha * ((SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[4] * b[5]) / (2.0 * SQRT_2));
    c[4] = alpha * ((SQRT_2 * a[4] * (b[1] + b[2]) + a[5] * b[3] + SQRT_2 * a[1] * b[4] + SQRT_2 * a[2] * b[4] + a[3] * b[5]) / (2.0 * SQRT_2));
    c[5] = alpha * ((SQRT_2 * a[5] * (b[0] + b[2]) + a[4] * b[3] + a[3] * b[4] + SQRT_2 * a[0] * b[5] + SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2));
    c[6] = alpha * ((SQRT_2 * a[3] * (-b[0] + b[1]) + SQRT_2 * a[0] * b[3] - SQRT_2 * a[1] * b[3] + a[5] * b[4] - a[4] * b[5]) / (2.0 * SQRT_2));
    c[7] = alpha * ((SQRT_2 * a[4] * (-b[1] + b[2]) - a[5] * b[3] + SQRT_2 * a[1] * b[4] - SQRT_2 * a[2] * b[4] + a[3] * b[5]) / (2.0 * SQRT_2));
    c[8] = alpha * ((SQRT_2 * a[5] * (-b[0] + b[2]) - a[4] * b[3] + a[3] * b[4] + SQRT_2 * a[0] * b[5] - SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2));
}

/// Performs the general transposed tensor dot itself operation: C = Aᵀ · A
/// 
/// Computes:
///
/// ```text
/// C = Aᵀ · A
/// ```
///
/// # Output
///
/// * `c` -- the resulting tensor C (slice)
///
/// # Input
///
/// * `a` -- the first tensor A (slice)
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect or `a.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_tra_dot_self(c: &mut [f64], alpha: f64, a: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(c.len() >= 6);
    c[0] = alpha * ((2.0 * (a[0] * a[0]) + (a[3] - a[6]) * (a[3] - a[6]) + (a[5] - a[8]) * (a[5] - a[8])) / 2.0);
    c[1] = alpha * ((2.0 * (a[1] * a[1]) + (a[3] + a[6]) * (a[3] + a[6]) + (a[4] - a[7]) * (a[4] - a[7])) / 2.0);
    c[2] = alpha * ((2.0 * (a[2] * a[2]) + (a[4] + a[7]) * (a[4] + a[7]) + (a[5] + a[8]) * (a[5] + a[8])) / 2.0);
    c[3] = alpha * (a[1] * (a[3] - a[6]) + a[0] * (a[3] + a[6]) + (a[4] - a[7]) * (a[5] - a[8]) / SQRT_2);
    c[4] = alpha * (a[2] * (a[4] - a[7]) + a[1] * (a[4] + a[7]) + (a[3] + a[6]) * (a[5] + a[8]) / SQRT_2);
    c[5] = alpha * ((a[3] - a[6]) * (a[4] + a[7]) / SQRT_2 + a[2] * (a[5] - a[8]) + a[0] * (a[5] + a[8]));
}

/// Performs the general transposed tensor dot general tensor operation: C = Aᵀ · B
/// 
/// Computes:
///
/// ```text
/// C = Aᵀ · B
/// ```
///
/// # Output
///
/// * `c` -- the resulting tensor C (slice)
///
/// # Input
///
/// * `a` -- the first tensor A (slice)
/// * `b` -- the second tensor B (slice)
/// * `chop` -- if true, ignores the skew components and treats C as symmetric
///
/// # Panics
/// 
/// A panic will occur if `a.len()` is incorrect, `b.len()` is incorrect, or `c.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_tra_dot_gen_chop(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64], chop: bool) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 9);
    if chop {
        debug_assert!(c.len() >= 6);
    } else {
        debug_assert!(c.len() >= 9);
    }
    c[0] = alpha * ((2.0 * a[0] * b[0] + (a[3] - a[6]) * (b[3] - b[6]) + (a[5] - a[8]) * (b[5] - b[8])) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + (a[3] + a[6]) * (b[3] + b[6]) + (a[4] - a[7]) * (b[4] - b[7])) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + (a[4] + a[7]) * (b[4] + b[7]) + (a[5] + a[8]) * (b[5] + b[8])) / 2.0);
    c[3] = alpha * ((2.0 * (a[3] + a[6]) * b[0] + 2.0 * (a[3] - a[6]) * b[1] + 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] - a[8]) * (b[4] - b[7]) + SQRT_2 * (a[4] - a[7]) * (b[5] - b[8])) / 4.0);
    c[4] = alpha * ((2.0 * (a[4] + a[7]) * b[1] + 2.0 * (a[4] - a[7]) * b[2] + SQRT_2 * (a[5] + a[8]) * (b[3] + b[6]) + 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] + a[6]) * (b[5] + b[8])) / 4.0);
    c[5] = alpha * ((2.0 * (a[5] + a[8]) * b[0] + 2.0 * (a[5] - a[8]) * b[2] + SQRT_2 * (a[4] + a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] - a[6]) * (b[4] + b[7]) + 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0);
    if !chop {
        c[6] = alpha * ((-2.0 * (a[3] + a[6]) * b[0] + 2.0 * (a[3] - a[6]) * b[1] - 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] - a[8]) * (b[4] - b[7]) - SQRT_2 * (a[4] - a[7]) * (b[5] - b[8])) / 4.0);
        c[7] = alpha * ((-2.0 * (a[4] + a[7]) * b[1] + 2.0 * (a[4] - a[7]) * b[2] - SQRT_2 * (a[5] + a[8]) * (b[3] + b[6]) - 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] + a[6]) * (b[5] + b[8])) / 4.0);
        c[8] = alpha * ((-2.0 * (a[5] + a[8]) * b[0] + 2.0 * (a[5] - a[8]) * b[2] - SQRT_2 * (a[4] + a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] - a[6]) * (b[4] + b[7]) - 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0);
    }
}

/// Performs the general tensor dot general transposed tensor operation: C = A · Bᵀ
/// 
/// Computes:
///
/// ```text
/// C = A · Bᵀ
/// ```
///
/// # Output
///
/// * `c` -- the resulting tensor C (slice)
///
/// # Input
///
/// * `a` -- the first tensor A (slice)
/// * `b` -- the second tensor B (slice)
/// * `chop` -- if true, ignores the skew components and treats C as symmetric
///
/// # Panics
/// 
/// A panic will occur if `a.len()` is incorrect, `b.len()` is incorrect, or `c.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_dot_gen_tra_chop(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64], chop: bool) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 9);
    if chop {
        debug_assert!(c.len() >= 6);
    } else {
        debug_assert!(c.len() >= 9);
    }
    c[0] = alpha * ((2.0 * a[0] * b[0] + (a[3] + a[6]) * (b[3] + b[6]) + (a[5] + a[8]) * (b[5] + b[8])) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + (a[3] - a[6]) * (b[3] - b[6]) + (a[4] + a[7]) * (b[4] + b[7])) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + (a[4] - a[7]) * (b[4] - b[7]) + (a[5] - a[8]) * (b[5] - b[8])) / 2.0);
    c[3] = alpha * ((2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[0] * (b[3] - b[6]) + 2.0 * a[1] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] + b[7]) + SQRT_2 * (a[4] + a[7]) * (b[5] + b[8])) / 4.0);
    c[4] = alpha * ((2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] + SQRT_2 * (a[5] - a[8]) * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) + 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] - b[8])) / 4.0);
    c[5] = alpha * ((2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] + SQRT_2 * (a[4] - a[7]) * (b[3] + b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) + 2.0 * a[2] * (b[5] + b[8])) / 4.0);
    if !chop {
        c[6] = alpha * ((-2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[0] * (b[3] - b[6]) - 2.0 * a[1] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] + b[7]) - SQRT_2 * (a[4] + a[7]) * (b[5] + b[8])) / 4.0);
        c[7] = alpha * ((-2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] - SQRT_2 * (a[5] - a[8]) * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) - 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] - b[8])) / 4.0);
        c[8] = alpha * ((-2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] - SQRT_2 * (a[4] - a[7]) * (b[3] + b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) - 2.0 * a[2] * (b[5] + b[8])) / 4.0);
    }
}

/// Performs the general tensor dot symmetric tensor dot its transposed self operation: C = A · B · Aᵀ
/// 
/// Computes:
///
/// ```text
/// C = A · B · Aᵀ
/// ```
///
/// # Output
///
/// * `c` -- the resulting tensor C (slice)
///
/// # Input
///
/// * `a` -- the first tensor A (slice)
/// * `b` -- the second tensor B (slice)
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect, `a.len()` is incorrect, or `b.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_dot_sym_dot_self_tra(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 6);
    debug_assert!(c.len() >= 6);
    c[0] = alpha * ((SQRT_2 * (a[3] + a[6]) * (SQRT_2 * (a[3] + a[6]) * b[1] + SQRT_2 * a[0] * b[3] + (a[5] + a[8]) * b[4]) + SQRT_2 * (a[5] + a[8]) * (SQRT_2 * (a[5] + a[8]) * b[2] + (a[3] + a[6]) * b[4] + SQRT_2 * a[0] * b[5]) + 2.0 * a[0] * (2.0 * a[0] * b[0] + (a[3] + a[6]) * b[3] + (a[5] + a[8]) * b[5])) / 4.0);
    c[1] = alpha * ((a[3] * a[3] * b[0] + a[6] * a[6] * b[0] + 2.0 * (a[1] * a[1]) * b[1] + a[4] * a[4] * b[2] + 2.0 * a[4] * a[7] * b[2] + a[7] * a[7] * b[2] + 2.0 * a[1] * a[4] * b[4] + 2.0 * a[1] * a[7] * b[4] - a[6] * (2.0 * a[1] * b[3] + SQRT_2 * (a[4] + a[7]) * b[5]) + a[3] * (-2.0 * a[6] * b[0] + 2.0 * a[1] * b[3] + SQRT_2 * (a[4] + a[7]) * b[5])) / 2.0);
    c[2] = alpha * ((a[5] * a[5] * b[0] + a[8] * a[8] * b[0] + a[4] * a[4] * b[1] - 2.0 * a[4] * a[7] * b[1] + a[7] * a[7] * b[1] + 2.0 * (a[2] * a[2]) * b[2] + 2.0 * a[2] * a[4] * b[4] - 2.0 * a[2] * a[7] * b[4] + a[8] * (-(SQRT_2 * a[4] * b[3]) + SQRT_2 * a[7] * b[3] - 2.0 * a[2] * b[5]) + a[5] * (-2.0 * a[8] * b[0] + SQRT_2 * a[4] * b[3] - SQRT_2 * a[7] * b[3] + 2.0 * a[2] * b[5])) / 2.0);
    c[3] = alpha * ((SQRT_2 * a[4] * a[5] * b[2] + SQRT_2 * a[5] * a[7] * b[2] + SQRT_2 * a[4] * a[8] * b[2] + SQRT_2 * a[7] * a[8] * b[2] + a[3] * a[3] * b[3] - a[6] * a[6] * b[3] + a[3] * a[4] * b[4] + a[4] * a[6] * b[4] + a[3] * a[7] * b[4] + a[6] * a[7] * b[4] + a[1] * (2.0 * a[3] * b[1] + 2.0 * a[6] * b[1] + SQRT_2 * (a[5] + a[8]) * b[4]) + a[3] * a[5] * b[5] - a[5] * a[6] * b[5] + a[3] * a[8] * b[5] - a[6] * a[8] * b[5] + a[0] * (2.0 * a[3] * b[0] - 2.0 * a[6] * b[0] + 2.0 * a[1] * b[3] + SQRT_2 * a[4] * b[5] + SQRT_2 * a[7] * b[5])) / 2.0);
    c[4] = alpha * ((SQRT_2 * a[6] * a[8] * b[0] + 2.0 * a[1] * a[4] * b[1] - 2.0 * a[1] * a[7] * b[1] + 2.0 * a[2] * a[4] * b[2] + 2.0 * a[2] * a[7] * b[2] - a[4] * a[6] * b[3] + a[6] * a[7] * b[3] - SQRT_2 * a[1] * a[8] * b[3] + 2.0 * a[1] * a[2] * b[4] + a[4] * a[4] * b[4] - a[7] * a[7] * b[4] - SQRT_2 * a[2] * a[6] * b[5] - a[4] * a[8] * b[5] - a[7] * a[8] * b[5] + a[3] * (SQRT_2 * a[5] * b[0] - SQRT_2 * a[8] * b[0] + a[4] * b[3] - a[7] * b[3] + SQRT_2 * a[2] * b[5]) + a[5] * (-(SQRT_2 * a[6] * b[0]) + SQRT_2 * a[1] * b[3] + (a[4] + a[7]) * b[5])) / 2.0);
    c[5] = alpha * ((SQRT_2 * a[4] * a[6] * b[1] - SQRT_2 * a[6] * a[7] * b[1] + 2.0 * a[2] * a[5] * b[2] + 2.0 * a[2] * a[8] * b[2] + a[5] * a[6] * b[3] - a[6] * a[8] * b[3] + a[4] * a[5] * b[4] + SQRT_2 * a[2] * a[6] * b[4] - a[5] * a[7] * b[4] + a[4] * a[8] * b[4] - a[7] * a[8] * b[4] + a[3] * (SQRT_2 * a[4] * b[1] - SQRT_2 * a[7] * b[1] + a[5] * b[3] - a[8] * b[3] + SQRT_2 * a[2] * b[4]) + a[5] * a[5] * b[5] - a[8] * a[8] * b[5] + a[0] * (2.0 * a[5] * b[0] - 2.0 * a[8] * b[0] + SQRT_2 * a[4] * b[3] - SQRT_2 * a[7] * b[3] + 2.0 * a[2] * b[5])) / 2.0);
}

/// Performs the general transposed tensor dot symmetric tensor dot itself operation: C = Aᵀ · B · A
/// 
/// Computes:
///
/// ```text
/// C = Aᵀ · B · A
/// ```
///
/// # Output
///
/// * `c` -- the resulting tensor C (slice)
///
/// # Input
///
/// * `a` -- the first tensor A (slice)
/// * `b` -- the second tensor B (slice)
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect, `a.len()` is incorrect, or `b.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_tra_dot_sym_dot_self(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 6);
    debug_assert!(c.len() >= 6);
    c[0] = alpha * ((2.0 * (a[0] * a[0]) * b[0] + a[3] * a[3] * b[1] + a[6] * a[6] * b[1] + a[5] * a[5] * b[2] - 2.0 * a[5] * a[8] * b[2] + a[8] * a[8] * b[2] - SQRT_2 * a[5] * a[6] * b[4] + SQRT_2 * a[6] * a[8] * b[4] + a[3] * (-2.0 * a[6] * b[1] + SQRT_2 * (a[5] - a[8]) * b[4]) + 2.0 * a[0] * (a[3] * b[3] - a[6] * b[3] + (a[5] - a[8]) * b[5])) / 2.0);
    c[1] = alpha * ((a[3] * a[3] * b[0] + a[6] * a[6] * b[0] + 2.0 * (a[1] * a[1]) * b[1] + a[4] * a[4] * b[2] - 2.0 * a[4] * a[7] * b[2] + a[7] * a[7] * b[2] + 2.0 * a[1] * a[4] * b[4] - 2.0 * a[1] * a[7] * b[4] + a[6] * (2.0 * a[1] * b[3] + SQRT_2 * (a[4] - a[7]) * b[5]) + a[3] * (2.0 * a[6] * b[0] + 2.0 * a[1] * b[3] + SQRT_2 * (a[4] - a[7]) * b[5])) / 2.0);
    c[2] = alpha * ((SQRT_2 * (a[4] + a[7]) * (SQRT_2 * (a[4] + a[7]) * b[1] + (a[5] + a[8]) * b[3] + SQRT_2 * a[2] * b[4]) + SQRT_2 * (a[5] + a[8]) * (SQRT_2 * (a[5] + a[8]) * b[0] + (a[4] + a[7]) * b[3] + SQRT_2 * a[2] * b[5]) + 2.0 * a[2] * (2.0 * a[2] * b[2] + (a[4] + a[7]) * b[4] + (a[5] + a[8]) * b[5])) / 4.0);
    c[3] = alpha * ((SQRT_2 * a[4] * a[5] * b[2] - SQRT_2 * a[5] * a[7] * b[2] - SQRT_2 * a[4] * a[8] * b[2] + SQRT_2 * a[7] * a[8] * b[2] + a[3] * a[3] * b[3] - a[6] * a[6] * b[3] + a[3] * a[4] * b[4] - a[4] * a[6] * b[4] - a[3] * a[7] * b[4] + a[6] * a[7] * b[4] + a[1] * (2.0 * a[3] * b[1] - 2.0 * a[6] * b[1] + SQRT_2 * (a[5] - a[8]) * b[4]) + a[3] * a[5] * b[5] + a[5] * a[6] * b[5] - a[3] * a[8] * b[5] - a[6] * a[8] * b[5] + a[0] * (2.0 * a[3] * b[0] + 2.0 * a[6] * b[0] + 2.0 * a[1] * b[3] + SQRT_2 * a[4] * b[5] - SQRT_2 * a[7] * b[5])) / 2.0);
    c[4] = alpha * ((SQRT_2 * a[6] * a[8] * b[0] + 2.0 * a[1] * a[4] * b[1] + 2.0 * a[1] * a[7] * b[1] + 2.0 * a[2] * a[4] * b[2] - 2.0 * a[2] * a[7] * b[2] + a[4] * a[6] * b[3] + a[6] * a[7] * b[3] + SQRT_2 * a[1] * a[8] * b[3] + 2.0 * a[1] * a[2] * b[4] + a[4] * a[4] * b[4] - a[7] * a[7] * b[4] + SQRT_2 * a[2] * a[6] * b[5] + a[4] * a[8] * b[5] - a[7] * a[8] * b[5] + a[3] * (SQRT_2 * a[5] * b[0] + SQRT_2 * a[8] * b[0] + a[4] * b[3] + a[7] * b[3] + SQRT_2 * a[2] * b[5]) + a[5] * (SQRT_2 * a[6] * b[0] + SQRT_2 * a[1] * b[3] + (a[4] - a[7]) * b[5])) / 2.0);
    c[5] = alpha * ((-(SQRT_2 * a[4] * a[6] * b[1]) - SQRT_2 * a[6] * a[7] * b[1] + 2.0 * a[2] * a[5] * b[2] - 2.0 * a[2] * a[8] * b[2] - a[5] * a[6] * b[3] - a[6] * a[8] * b[3] + a[4] * a[5] * b[4] - SQRT_2 * a[2] * a[6] * b[4] + a[5] * a[7] * b[4] - a[4] * a[8] * b[4] - a[7] * a[8] * b[4] + a[3] * (SQRT_2 * a[4] * b[1] + SQRT_2 * a[7] * b[1] + a[5] * b[3] + a[8] * b[3] + SQRT_2 * a[2] * b[4]) + a[5] * a[5] * b[5] - a[8] * a[8] * b[5] + a[0] * (2.0 * a[5] * b[0] + 2.0 * a[8] * b[0] + SQRT_2 * a[4] * b[3] + SQRT_2 * a[7] * b[3] + 2.0 * a[2] * b[5])) / 2.0);
}


// --- Manual Stack Functions ---

#[rustfmt::skip]
pub(crate) fn t2_gen_dot_sym_stack(c: &mut [f64], a: &[f64], b: &[f64]) {
    c[0] = (2.0 * a[0] * b[0] + (a[3] + a[6]) * b[3] + (a[5] + a[8]) * b[5]) / 2.0;
    c[1] = (2.0 * a[1] * b[1] + (a[3] - a[6]) * b[3] + (a[4] + a[7]) * b[4]) / 2.0;
    c[2] = (2.0 * a[2] * b[2] + (a[4] - a[7]) * b[4] + (a[5] - a[8]) * b[5]) / 2.0;
    c[3] = (SQRT_2 * a[6] * (-b[0] + b[1]) + SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[8] * b[4] + a[4] * b[5] + a[7] * b[5]) / (2.0 * SQRT_2);
    c[4] = (SQRT_2 * a[7] * (-b[1] + b[2]) + SQRT_2 * a[4] * (b[1] + b[2]) + a[5] * b[3] - a[8] * b[3] + SQRT_2 * a[1] * b[4] + SQRT_2 * a[2] * b[4] + a[3] * b[5] - a[6] * b[5]) / (2.0 * SQRT_2);
    c[5] = (SQRT_2 * a[8] * (-b[0] + b[2]) + SQRT_2 * a[5] * (b[0] + b[2]) + a[4] * b[3] - a[7] * b[3] + a[3] * b[4] + a[6] * b[4] + SQRT_2 * a[0] * b[5] + SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2);
    c[6] = (-2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[0] * b[3] - 2.0 * a[1] * b[3] + SQRT_2 * (a[5] + a[8]) * b[4] - SQRT_2 * (a[4] + a[7]) * b[5]) / 4.0;
    c[7] = (SQRT_2 * a[4] * (-b[1] + b[2]) + SQRT_2 * a[7] * (b[1] + b[2]) - a[5] * b[3] + a[8] * b[3] + SQRT_2 * a[1] * b[4] - SQRT_2 * a[2] * b[4] + a[3] * b[5] - a[6] * b[5]) / (2.0 * SQRT_2);
    c[8] = (SQRT_2 * a[5] * (-b[0] + b[2]) + SQRT_2 * a[8] * (b[0] + b[2]) - a[4] * b[3] + a[7] * b[3] + a[3] * b[4] + a[6] * b[4] + SQRT_2 * a[0] * b[5] - SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2);
}

#[rustfmt::skip]
pub(crate) fn t2_gen_tra_dot_self_stack(c: &mut [f64], a: &[f64]) {
    c[0] = (2.0 * (a[0] * a[0]) + (a[3] - a[6]) * (a[3] - a[6]) + (a[5] - a[8]) * (a[5] - a[8])) / 2.0;
    c[1] = (2.0 * (a[1] * a[1]) + (a[3] + a[6]) * (a[3] + a[6]) + (a[4] - a[7]) * (a[4] - a[7])) / 2.0;
    c[2] = (2.0 * (a[2] * a[2]) + (a[4] + a[7]) * (a[4] + a[7]) + (a[5] + a[8]) * (a[5] + a[8])) / 2.0;
    c[3] = a[1] * (a[3] - a[6]) + a[0] * (a[3] + a[6]) + (a[4] - a[7]) * (a[5] - a[8]) / SQRT_2;
    c[4] = a[2] * (a[4] - a[7]) + a[1] * (a[4] + a[7]) + (a[3] + a[6]) * (a[5] + a[8]) / SQRT_2;
    c[5] = (a[3] - a[6]) * (a[4] + a[7]) / SQRT_2 + a[2] * (a[5] - a[8]) + a[0] * (a[5] + a[8]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rep, Tensor2};
    use russell_lab::mat_approx_eq;

    #[test]
    fn t2_matmul_gen_gen_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ], Rep::General).unwrap();
        let mut c = Tensor2::new(Rep::General);
        t2_matmul(&mut c, 2.0, &a, false, &b, false).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [  60.00000,   48.00000,   36.00000],
            [ 168.00000,  138.00000,  108.00000],
            [ 276.00000,  228.00000,  180.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_gen_sym_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [   3.00000,    5.00000,    6.00000],
            [   5.00000,    2.00000,    4.00000],
            [   6.00000,    4.00000,    1.00000],
        ], Rep::Symmetric).unwrap();
        let mut c = Tensor2::new(Rep::General);
        t2_matmul(&mut c, 3.0, &a, false, &b, false).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [  93.00000,   63.00000,   51.00000],
            [ 219.00000,  162.00000,  150.00000],
            [ 345.00000,  261.00000,  249.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_sym_gen_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    4.00000,    6.00000],
            [   4.00000,    2.00000,    5.00000],
            [   6.00000,    5.00000,    3.00000],
        ], Rep::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ], Rep::General).unwrap();
        let mut c = Tensor2::new(Rep::General);
        t2_matmul(&mut c, 1.5, &a, false, &b, false).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [  76.50000,   60.00000,   43.50000],
            [  94.50000,   78.00000,   61.50000],
            [ 139.50000,  118.50000,   97.50000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_sym_sym_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    4.00000,    6.00000],
            [   4.00000,    2.00000,    5.00000],
            [   6.00000,    5.00000,    3.00000],
        ], Rep::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [   3.00000,    5.00000,    6.00000],
            [   5.00000,    2.00000,    4.00000],
            [   6.00000,    4.00000,    1.00000],
        ], Rep::Symmetric).unwrap();
        let mut c = Tensor2::new(Rep::General);
        t2_matmul(&mut c, 0.5, &a, false, &b, false).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [  29.50000,   18.50000,   14.00000],
            [  26.00000,   22.00000,   18.50000],
            [  30.50000,   26.00000,   29.50000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_tra_gen_gen_chop_false_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ], Rep::General).unwrap();
        let mut c = Tensor2::new(Rep::General);
        t2_matmul(&mut c, 2.0, &a, true, &b, false).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 108.00000,   84.00000,   60.00000],
            [ 144.00000,  114.00000,   84.00000],
            [ 180.00000,  144.00000,  108.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_tra_gen_gen_chop_true_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ], Rep::General).unwrap();
        let mut c = Tensor2::new(Rep::Symmetric);
        t2_matmul(&mut c, 2.0, &a, true, &b, false).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 108.00000,  114.00000,  120.00000],
            [ 114.00000,  114.00000,  114.00000],
            [ 120.00000,  114.00000,  108.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_gen_tra_gen_chop_false_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ], Rep::General).unwrap();
        let mut c = Tensor2::new(Rep::General);
        t2_matmul(&mut c, 2.0, &a, false, &b, true).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [  92.00000,   56.00000,   20.00000],
            [ 236.00000,  146.00000,   56.00000],
            [ 380.00000,  236.00000,   92.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_gen_tra_gen_chop_true_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ], Rep::General).unwrap();
        let mut c = Tensor2::new(Rep::Symmetric);
        t2_matmul(&mut c, 2.0, &a, false, &b, true).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [  92.00000,  146.00000,  200.00000],
            [ 146.00000,  146.00000,  146.00000],
            [ 200.00000,  146.00000,   92.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmulx_sym_dot_self_tra_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [   3.00000,    5.00000,    6.00000],
            [   5.00000,    2.00000,    4.00000],
            [   6.00000,    4.00000,    1.00000],
        ], Rep::Symmetric).unwrap();
        let mut c = Tensor2::new(Rep::Symmetric);
        t2_matmulx(&mut c, 1.5, &a, true, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 186.00000,  496.50000,  807.00000],
            [ 496.50000, 1293.00000, 2089.50000],
            [ 807.00000, 2089.50000, 3372.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmulx_tra_sym_dot_self_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [   3.00000,    5.00000,    6.00000],
            [   5.00000,    2.00000,    4.00000],
            [   6.00000,    4.00000,    1.00000],
        ], Rep::Symmetric).unwrap();
        let mut c = Tensor2::new(Rep::Symmetric);
        t2_matmulx(&mut c, 1.5, &a, false, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 648.00000,  850.50000, 1053.00000],
            [ 850.50000, 1107.00000, 1363.50000],
            [1053.00000, 1363.50000, 1674.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

}