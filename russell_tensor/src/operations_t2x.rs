use super::Tensor2;
use crate::{SQRT_2, StrError};

/// Performs the matrix multiplication between two Tensor2
///
/// # Supported Combinations
///
/// | Formula | `a` | `b` | `c` | `tra_a` | `tra_b` |
/// | :--- | :--- | :--- | :--- | :--- | :--- |
/// | C = α A · B | `General` | `General` | `General` | `false` | `false` |
/// | C = α A · A | `General` | (same as a) | `General` | `false` | `false` |
/// | C = α A · Bᵀ | `General` | `Symmetric` | `General` | `false` | `true` |
/// | C = α A · B | `General` | `Symmetric` | `General` | `false` | `false` |
/// | C = α Aᵀ · B | `Symmetric` | `General` | `General` | `true` | `false` |
/// | C = α A · B | `Symmetric` | `General` | `General` | `false` | `false` |
/// | C = α Aᵀ · Bᵀ | `Symmetric` | `Symmetric` | `General` | `true` | `true` |
/// | C = α Aᵀ · A | `Symmetric` | (same as a) | `Symmetric` | `true` | `true` |
/// | C = α Aᵀ · B | `Symmetric` | `Symmetric` | `General` | `true` | `false` |
/// | C = α Aᵀ · A | `Symmetric` | (same as a) | `Symmetric` | `true` | `false` |
/// | C = α A · Bᵀ | `Symmetric` | `Symmetric` | `General` | `false` | `true` |
/// | C = α A · A | `Symmetric` | (same as a) | `Symmetric` | `false` | `true` |
/// | C = α A · B | `Symmetric` | `Symmetric` | `General` | `false` | `false` |
/// | C = α A · A | `Symmetric` | (same as a) | `Symmetric` | `false` | `false` |
/// | C = α Aᵀ · A | `General` | (same as a) | `Symmetric` | `true` | `false` |
/// | C = α chop(Aᵀ · B) | `General` | `General` | `Symmetric or General` | `true` | `false` |
/// | C = α Aᵀ · B | `General` | `General` | `General` | `true` | `false` |
/// | C = α A · A | `General` | (same as a) | `Symmetric` | `false` | `true` |
/// | C = α chop(A · Bᵀ) | `General` | `General` | `Symmetric or General` | `false` | `true` |
/// | C = α A · Bᵀ | `General` | `General` | `General` | `false` | `true` |
/// | C = α Aᵀ · Bᵀ | `General` | `General` | `General` | `true` | `true` |
/// | C = α Aᵀ · Bᵀ | `Symmetric` | `General` | `General` | `true` | `true` |
/// | C = α A · Bᵀ | `Symmetric` | `General` | `General` | `false` | `true` |
/// | C = α Aᵀ · Bᵀ | `General` | `Symmetric` | `General` | `true` | `true` |
/// | C = α Aᵀ · B | `General` | `Symmetric` | `General` | `true` | `false` |
///
/// **Note:** The use of `chop` is decided by the N of the output tensor `c`.
/// Also, `chop` doesn't actually check for symmetry; it just ignores (chops) the last 3 rows of the Kelvin-Mandel vector.
///
/// **Note:** `Symmetric2D` is not supported (use `Symmetric` instead).
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
/// Returns an error if the combination of representations and transpositions is unavailable or impossible, or if `Symmetric2D` is used.
pub fn t2_matmul<const L: usize, const M: usize, const N: usize>(
    c: &mut Tensor2<L>,
    alpha: f64,
    a: &Tensor2<M>,
    tra_a: bool,
    b: &Tensor2<N>,
    tra_b: bool,
) -> Result<(), StrError> {
    match (M, N, tra_a, tra_b) {
        (9, 9, false, false) => {
            if a as *const _ as *const () == b as *const _ as *const () {
                if L != 9 {
                    return Err("c must be General for this combination");
                }
                t2_gen_dot_self(c.as_mut_data(), alpha, a.as_data());
            } else {
                if L != 9 {
                    return Err("c must be General for this combination");
                }
                t2_gen_dot_gen(c.as_mut_data(), alpha, a.as_data(), b.as_data());
            }
        }
        (9, 6, false, true) | (9, 6, false, false) => {
            if L != 9 {
                return Err("c must be General for this combination");
            }
            t2_gen_dot_sym(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (6, 9, true, false) | (6, 9, false, false) => {
            if L != 9 {
                return Err("c must be General for this combination");
            }
            t2_sym_dot_gen(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (6, 6, true, true) | (6, 6, true, false) | (6, 6, false, true) | (6, 6, false, false) => {
            if a as *const _ as *const () == b as *const _ as *const () {
                if L != 6 {
                    return Err("c must be Symmetric for this combination");
                }
                t2_sym_dot_self(c.as_mut_data(), alpha, a.as_data());
            } else {
                if L != 9 {
                    return Err("c must be General for this combination");
                }
                t2_sym_dot_sym(c.as_mut_data(), alpha, a.as_data(), b.as_data());
            }
        }
        (9, 9, true, false) => {
            if a as *const _ as *const () == b as *const _ as *const () {
                if L != 6 {
                    return Err("c must be Symmetric for this combination");
                }
                t2_gen_tra_dot_self(c.as_mut_data(), alpha, a.as_data());
            } else {
                if L == 6 {
                    t2_gen_tra_dot_gen_chop(c.as_mut_data(), alpha, a.as_data(), b.as_data());
                } else if L == 9 {
                    t2_gen_tra_dot_gen(c.as_mut_data(), alpha, a.as_data(), b.as_data());
                } else {
                    return Err("c must be Symmetric or General for this combination");
                }
            }
        }
        (9, 9, false, true) => {
            if a as *const _ as *const () == b as *const _ as *const () {
                if L != 6 {
                    return Err("c must be Symmetric for this combination");
                }
                t2_gen_dot_self_tra(c.as_mut_data(), alpha, a.as_data());
            } else {
                if L == 6 {
                    t2_gen_dot_gen_tra_chop(c.as_mut_data(), alpha, a.as_data(), b.as_data());
                } else if L == 9 {
                    t2_gen_dot_gen_tra(c.as_mut_data(), alpha, a.as_data(), b.as_data());
                } else {
                    return Err("c must be Symmetric or General for this combination");
                }
            }
        }
        (9, 9, true, true) => {
            if L != 9 {
                return Err("c must be General for this combination");
            }
            t2_gen_tra_dot_gen_tra(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (6, 9, true, true) | (6, 9, false, true) => {
            if L != 9 {
                return Err("c must be General for this combination");
            }
            t2_sym_dot_gen_tra(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (9, 6, true, true) | (9, 6, true, false) => {
            if L != 9 {
                return Err("c must be General for this combination");
            }
            t2_gen_tra_dot_sym(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (4, _, _, _) | (_, 4, _, _) => {
            return Err("t2_matmul: Symmetric2D is not supported; use Symmetric instead");
        }
        _ => return Err("t2_matmul: unsupported combination"),
    }
    Ok(())
}

/// Performs a triple matrix multiplication: C = α A · B · Aᵀ or C = α Aᵀ · B · A
///
/// # Supported Combinations
///
/// | Formula | `a` | `b` | `c` | `forward` |
/// | :--- | :--- | :--- | :--- | :--- |
/// | C = α A · B · Aᵀ | `General` | `Symmetric` | `Symmetric` | `true` |
/// | C = α Aᵀ · B · A | `General` | `Symmetric` | `Symmetric` | `false` |
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
pub fn t2_matmulx<const L: usize, const M: usize, const N: usize>(
    c: &mut Tensor2<L>,
    alpha: f64,
    a: &Tensor2<M>,
    forward: bool,
    b: &Tensor2<N>,
) -> Result<(), StrError> {
    match (M, N, forward) {
        (9, 6, true) => {
            if L != 6 {
                return Err("c must be symmetric for this combination");
            }
            t2_gen_dot_sym_dot_self_tra(c.as_mut_data(), alpha, a.as_data(), b.as_data());
        }
        (9, 6, false) => {
            if L != 6 {
                return Err("c must be symmetric for this combination");
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

/// Performs the general tensor dot its transposed self operation: C = A · Aᵀ
/// 
/// Computes:
///
/// ```text
/// C = A · Aᵀ
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
pub(crate) fn t2_gen_dot_self_tra(c: &mut [f64], alpha: f64, a: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(c.len() >= 6);
    c[0] = alpha * ((2.0 * (a[0] * a[0]) + (a[3] + a[6]) * (a[3] + a[6]) + (a[5] + a[8]) * (a[5] + a[8])) / 2.0);
    c[1] = alpha * ((2.0 * (a[1] * a[1]) + (a[3] - a[6]) * (a[3] - a[6]) + (a[4] + a[7]) * (a[4] + a[7])) / 2.0);
    c[2] = alpha * ((2.0 * (a[2] * a[2]) + (a[4] - a[7]) * (a[4] - a[7]) + (a[5] - a[8]) * (a[5] - a[8])) / 2.0);
    c[3] = alpha * (a[0] * (a[3] - a[6]) + a[1] * (a[3] + a[6]) + (a[4] + a[7]) * (a[5] + a[8]) / SQRT_2);
    c[4] = alpha * (a[1] * (a[4] - a[7]) + a[2] * (a[4] + a[7]) + (a[3] - a[6]) * (a[5] - a[8]) / SQRT_2);
    c[5] = alpha * ((a[3] + a[6]) * (a[4] - a[7]) / SQRT_2 + a[0] * (a[5] - a[8]) + a[2] * (a[5] + a[8]));
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
///
/// # Panics
/// 
/// A panic will occur if `a.len()` is incorrect, `b.len()` is incorrect, or `c.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_tra_dot_gen_chop(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 9);
    debug_assert!(c.len() >= 6);
    c[0] = alpha * ((2.0 * a[0] * b[0] + (a[3] - a[6]) * (b[3] - b[6]) + (a[5] - a[8]) * (b[5] - b[8])) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + (a[3] + a[6]) * (b[3] + b[6]) + (a[4] - a[7]) * (b[4] - b[7])) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + (a[4] + a[7]) * (b[4] + b[7]) + (a[5] + a[8]) * (b[5] + b[8])) / 2.0);
    c[3] = alpha * ((2.0 * (a[3] + a[6]) * b[0] + 2.0 * (a[3] - a[6]) * b[1] + 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] - a[8]) * (b[4] - b[7]) + SQRT_2 * (a[4] - a[7]) * (b[5] - b[8])) / 4.0);
    c[4] = alpha * ((2.0 * (a[4] + a[7]) * b[1] + 2.0 * (a[4] - a[7]) * b[2] + SQRT_2 * (a[5] + a[8]) * (b[3] + b[6]) + 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] + a[6]) * (b[5] + b[8])) / 4.0);
    c[5] = alpha * ((2.0 * (a[5] + a[8]) * b[0] + 2.0 * (a[5] - a[8]) * b[2] + SQRT_2 * (a[4] + a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] - a[6]) * (b[4] + b[7]) + 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0);
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
///
/// # Panics
/// 
/// A panic will occur if `a.len()` is incorrect, `b.len()` is incorrect, or `c.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_dot_gen_tra_chop(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 9);
    debug_assert!(c.len() >= 6);
    c[0] = alpha * ((2.0 * a[0] * b[0] + (a[3] + a[6]) * (b[3] + b[6]) + (a[5] + a[8]) * (b[5] + b[8])) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + (a[3] - a[6]) * (b[3] - b[6]) + (a[4] + a[7]) * (b[4] + b[7])) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + (a[4] - a[7]) * (b[4] - b[7]) + (a[5] - a[8]) * (b[5] - b[8])) / 2.0);
    c[3] = alpha * ((2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[0] * (b[3] - b[6]) + 2.0 * a[1] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] + b[7]) + SQRT_2 * (a[4] + a[7]) * (b[5] + b[8])) / 4.0);
    c[4] = alpha * ((2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] + SQRT_2 * (a[5] - a[8]) * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) + 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] - b[8])) / 4.0);
    c[5] = alpha * ((2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] + SQRT_2 * (a[4] - a[7]) * (b[3] + b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) + 2.0 * a[2] * (b[5] + b[8])) / 4.0);
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

/// Performs the general tensor dot itself operation: C = A · A
/// 
/// Computes:
///
/// ```text
/// C = A · A
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
pub(crate) fn t2_gen_dot_self(c: &mut [f64], alpha: f64, a: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(c.len() >= 9);
    c[0] = alpha * ((2.0 * (a[0] * a[0]) + a[3] * a[3] + a[5] * a[5] - a[6] * a[6] - a[8] * a[8]) / 2.0);
    c[1] = alpha * ((2.0 * (a[1] * a[1]) + a[3] * a[3] + a[4] * a[4] - a[6] * a[6] - a[7] * a[7]) / 2.0);
    c[2] = alpha * ((2.0 * (a[2] * a[2]) + a[4] * a[4] + a[5] * a[5] - a[7] * a[7] - a[8] * a[8]) / 2.0);
    c[3] = alpha * (a[0] * a[3] + a[1] * a[3] + (a[4] * a[5] - a[7] * a[8]) / SQRT_2);
    c[4] = alpha * (a[1] * a[4] + a[2] * a[4] + (a[3] * a[5] - a[6] * a[8]) / SQRT_2);
    c[5] = alpha * (a[3] * a[4] / SQRT_2 + a[0] * a[5] + a[2] * a[5] + a[6] * a[7] / SQRT_2);
    c[6] = alpha * (a[0] * a[6] + a[1] * a[6] + (-(a[5] * a[7]) + a[4] * a[8]) / SQRT_2);
    c[7] = alpha * (-(a[5] * a[6] / SQRT_2) + a[1] * a[7] + a[2] * a[7] + a[3] * a[8] / SQRT_2);
    c[8] = alpha * (a[4] * a[6] / SQRT_2 + a[3] * a[7] / SQRT_2 + (a[0] + a[2]) * a[8]);
}

/// Performs the symmetric tensor dot itself operation: C = A · A
/// 
/// Computes:
///
/// ```text
/// C = A · A
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
pub(crate) fn t2_sym_dot_self(c: &mut [f64], alpha: f64, a: &[f64]) {
    debug_assert!(a.len() >= 6);
    debug_assert!(c.len() >= 6);
    c[0] = alpha * ((2.0 * (a[0] * a[0]) + a[3] * a[3] + a[5] * a[5]) / 2.0);
    c[1] = alpha * ((2.0 * (a[1] * a[1]) + a[3] * a[3] + a[4] * a[4]) / 2.0);
    c[2] = alpha * ((2.0 * (a[2] * a[2]) + a[4] * a[4] + a[5] * a[5]) / 2.0);
    c[3] = alpha * (a[0] * a[3] + a[1] * a[3] + a[4] * a[5] / SQRT_2);
    c[4] = alpha * (a[1] * a[4] + a[2] * a[4] + a[3] * a[5] / SQRT_2);
    c[5] = alpha * (a[3] * a[4] / SQRT_2 + (a[0] + a[2]) * a[5]);
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
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect, `a.len()` is incorrect, or `b.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_tra_dot_gen(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 9);
    debug_assert!(c.len() >= 9);
    c[0] = alpha * ((2.0 * a[0] * b[0] + (a[3] - a[6]) * (b[3] - b[6]) + (a[5] - a[8]) * (b[5] - b[8])) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + (a[3] + a[6]) * (b[3] + b[6]) + (a[4] - a[7]) * (b[4] - b[7])) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + (a[4] + a[7]) * (b[4] + b[7]) + (a[5] + a[8]) * (b[5] + b[8])) / 2.0);
    c[3] = alpha * ((2.0 * (a[3] + a[6]) * b[0] + 2.0 * (a[3] - a[6]) * b[1] + 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] - a[8]) * (b[4] - b[7]) + SQRT_2 * (a[4] - a[7]) * (b[5] - b[8])) / 4.0);
    c[4] = alpha * ((2.0 * (a[4] + a[7]) * b[1] + 2.0 * (a[4] - a[7]) * b[2] + SQRT_2 * (a[5] + a[8]) * (b[3] + b[6]) + 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] + a[6]) * (b[5] + b[8])) / 4.0);
    c[5] = alpha * ((2.0 * (a[5] + a[8]) * b[0] + 2.0 * (a[5] - a[8]) * b[2] + SQRT_2 * (a[4] + a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] - a[6]) * (b[4] + b[7]) + 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0);
    c[6] = alpha * ((-2.0 * (a[3] + a[6]) * b[0] + 2.0 * (a[3] - a[6]) * b[1] - 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] - a[8]) * (b[4] - b[7]) - SQRT_2 * (a[4] - a[7]) * (b[5] - b[8])) / 4.0);
    c[7] = alpha * ((-2.0 * (a[4] + a[7]) * b[1] + 2.0 * (a[4] - a[7]) * b[2] - SQRT_2 * (a[5] + a[8]) * (b[3] + b[6]) - 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] + a[6]) * (b[5] + b[8])) / 4.0);
    c[8] = alpha * ((-2.0 * (a[5] + a[8]) * b[0] + 2.0 * (a[5] - a[8]) * b[2] - SQRT_2 * (a[4] + a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] - a[6]) * (b[4] + b[7]) - 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0);
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
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect, `a.len()` is incorrect, or `b.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_dot_gen_tra(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 9);
    debug_assert!(c.len() >= 9);
    c[0] = alpha * ((2.0 * a[0] * b[0] + (a[3] + a[6]) * (b[3] + b[6]) + (a[5] + a[8]) * (b[5] + b[8])) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + (a[3] - a[6]) * (b[3] - b[6]) + (a[4] + a[7]) * (b[4] + b[7])) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + (a[4] - a[7]) * (b[4] - b[7]) + (a[5] - a[8]) * (b[5] - b[8])) / 2.0);
    c[3] = alpha * ((2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[0] * (b[3] - b[6]) + 2.0 * a[1] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] + b[7]) + SQRT_2 * (a[4] + a[7]) * (b[5] + b[8])) / 4.0);
    c[4] = alpha * ((2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] + SQRT_2 * (a[5] - a[8]) * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) + 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] - b[8])) / 4.0);
    c[5] = alpha * ((2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] + SQRT_2 * (a[4] - a[7]) * (b[3] + b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) + 2.0 * a[2] * (b[5] + b[8])) / 4.0);
    c[6] = alpha * ((-2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[0] * (b[3] - b[6]) - 2.0 * a[1] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] + b[7]) - SQRT_2 * (a[4] + a[7]) * (b[5] + b[8])) / 4.0);
    c[7] = alpha * ((-2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] - SQRT_2 * (a[5] - a[8]) * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) - 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] - b[8])) / 4.0);
    c[8] = alpha * ((-2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] - SQRT_2 * (a[4] - a[7]) * (b[3] + b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) - 2.0 * a[2] * (b[5] + b[8])) / 4.0);
}

/// Performs the general transposed tensor dot general transposed tensor operation: C = Aᵀ · Bᵀ
/// 
/// Computes:
///
/// ```text
/// C = Aᵀ · Bᵀ
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
pub(crate) fn t2_gen_tra_dot_gen_tra(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 9);
    debug_assert!(c.len() >= 9);
    c[0] = alpha * ((2.0 * a[0] * b[0] + (a[3] - a[6]) * (b[3] + b[6]) + (a[5] - a[8]) * (b[5] + b[8])) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + (a[3] + a[6]) * (b[3] - b[6]) + (a[4] - a[7]) * (b[4] + b[7])) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + (a[4] + a[7]) * (b[4] - b[7]) + (a[5] + a[8]) * (b[5] - b[8])) / 2.0);
    c[3] = alpha * ((2.0 * (a[3] + a[6]) * b[0] + 2.0 * (a[3] - a[6]) * b[1] + 2.0 * a[0] * (b[3] - b[6]) + 2.0 * a[1] * (b[3] + b[6]) + SQRT_2 * (a[5] - a[8]) * (b[4] + b[7]) + SQRT_2 * (a[4] - a[7]) * (b[5] + b[8])) / 4.0);
    c[4] = alpha * ((2.0 * (a[4] + a[7]) * b[1] + 2.0 * (a[4] - a[7]) * b[2] + SQRT_2 * (a[5] + a[8]) * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) + 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * (a[3] + a[6]) * (b[5] - b[8])) / 4.0);
    c[5] = alpha * ((2.0 * (a[5] + a[8]) * b[0] + 2.0 * (a[5] - a[8]) * b[2] + SQRT_2 * (a[4] + a[7]) * (b[3] + b[6]) + SQRT_2 * (a[3] - a[6]) * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) + 2.0 * a[2] * (b[5] + b[8])) / 4.0);
    c[6] = alpha * ((-2.0 * (a[3] + a[6]) * b[0] + 2.0 * (a[3] - a[6]) * b[1] + 2.0 * a[0] * (b[3] - b[6]) - 2.0 * a[1] * (b[3] + b[6]) + SQRT_2 * (a[5] - a[8]) * (b[4] + b[7]) - SQRT_2 * (a[4] - a[7]) * (b[5] + b[8])) / 4.0);
    c[7] = alpha * ((-2.0 * (a[4] + a[7]) * b[1] + 2.0 * (a[4] - a[7]) * b[2] - SQRT_2 * (a[5] + a[8]) * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) - 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * (a[3] + a[6]) * (b[5] - b[8])) / 4.0);
    c[8] = alpha * ((-2.0 * (a[5] + a[8]) * b[0] + 2.0 * (a[5] - a[8]) * b[2] - SQRT_2 * (a[4] + a[7]) * (b[3] + b[6]) + SQRT_2 * (a[3] - a[6]) * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) - 2.0 * a[2] * (b[5] + b[8])) / 4.0);
}

/// Performs the symmetric tensor dot general transposed tensor operation: C = A · Bᵀ
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
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect, `a.len()` is incorrect, or `b.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_sym_dot_gen_tra(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 6);
    debug_assert!(b.len() >= 9);
    debug_assert!(c.len() >= 9);
    c[0] = alpha * ((2.0 * a[0] * b[0] + a[3] * (b[3] + b[6]) + a[5] * (b[5] + b[8])) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + a[3] * (b[3] - b[6]) + a[4] * (b[4] + b[7])) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + a[4] * (b[4] - b[7]) + a[5] * (b[5] - b[8])) / 2.0);
    c[3] = alpha * ((SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[4] * b[5] + SQRT_2 * a[0] * (b[3] - b[6]) + SQRT_2 * a[1] * b[6] + a[5] * b[7] + a[4] * b[8]) / (2.0 * SQRT_2));
    c[4] = alpha * ((2.0 * a[4] * b[1] + 2.0 * a[4] * b[2] + SQRT_2 * a[5] * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) + 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * a[3] * (b[5] - b[8])) / 4.0);
    c[5] = alpha * ((2.0 * a[5] * b[0] + 2.0 * a[5] * b[2] + SQRT_2 * a[4] * (b[3] + b[6]) + SQRT_2 * a[3] * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) + 2.0 * a[2] * (b[5] + b[8])) / 4.0);
    c[6] = alpha * ((-2.0 * a[3] * b[0] + 2.0 * a[3] * b[1] + 2.0 * a[0] * (b[3] - b[6]) - 2.0 * a[1] * (b[3] + b[6]) + SQRT_2 * a[5] * (b[4] + b[7]) - SQRT_2 * a[4] * (b[5] + b[8])) / 4.0);
    c[7] = alpha * ((-2.0 * a[4] * b[1] + 2.0 * a[4] * b[2] - SQRT_2 * a[5] * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) - 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * a[3] * (b[5] - b[8])) / 4.0);
    c[8] = alpha * ((-2.0 * a[5] * b[0] + 2.0 * a[5] * b[2] - SQRT_2 * a[4] * (b[3] + b[6]) + SQRT_2 * a[3] * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) - 2.0 * a[2] * (b[5] + b[8])) / 4.0);
}

/// Performs the general transposed tensor dot symmetric tensor operation: C = Aᵀ · B
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
///
/// # Panics
/// 
/// A panic will occur if `c.len()` is incorrect, `a.len()` is incorrect, or `b.len()` is incorrect.
#[rustfmt::skip]
pub(crate) fn t2_gen_tra_dot_sym(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64]) {
    debug_assert!(a.len() >= 9);
    debug_assert!(b.len() >= 6);
    debug_assert!(c.len() >= 9);
    c[0] = alpha * ((2.0 * a[0] * b[0] + (a[3] - a[6]) * b[3] + (a[5] - a[8]) * b[5]) / 2.0);
    c[1] = alpha * ((2.0 * a[1] * b[1] + (a[3] + a[6]) * b[3] + (a[4] - a[7]) * b[4]) / 2.0);
    c[2] = alpha * ((2.0 * a[2] * b[2] + (a[4] + a[7]) * b[4] + (a[5] + a[8]) * b[5]) / 2.0);
    c[3] = alpha * ((SQRT_2 * a[6] * (b[0] - b[1]) + SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] + SQRT_2 * a[1] * b[3] + a[5] * b[4] - a[8] * b[4] + a[4] * b[5] - a[7] * b[5]) / (2.0 * SQRT_2));
    c[4] = alpha * ((SQRT_2 * a[7] * (b[1] - b[2]) + SQRT_2 * a[4] * (b[1] + b[2]) + a[5] * b[3] + a[8] * b[3] + SQRT_2 * a[1] * b[4] + SQRT_2 * a[2] * b[4] + a[3] * b[5] + a[6] * b[5]) / (2.0 * SQRT_2));
    c[5] = alpha * ((SQRT_2 * a[8] * (b[0] - b[2]) + SQRT_2 * a[5] * (b[0] + b[2]) + a[4] * b[3] + a[7] * b[3] + a[3] * b[4] - a[6] * b[4] + SQRT_2 * a[0] * b[5] + SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2));
    c[6] = alpha * ((SQRT_2 * a[3] * (-b[0] + b[1]) - SQRT_2 * a[6] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] - SQRT_2 * a[1] * b[3] + a[5] * b[4] - a[8] * b[4] - a[4] * b[5] + a[7] * b[5]) / (2.0 * SQRT_2));
    c[7] = alpha * ((-2.0 * (a[4] + a[7]) * b[1] + 2.0 * (a[4] - a[7]) * b[2] - SQRT_2 * (a[5] + a[8]) * b[3] + 2.0 * a[1] * b[4] - 2.0 * a[2] * b[4] + SQRT_2 * (a[3] + a[6]) * b[5]) / 4.0);
    c[8] = alpha * ((-2.0 * (a[5] + a[8]) * b[0] + 2.0 * (a[5] - a[8]) * b[2] - SQRT_2 * (a[4] + a[7]) * b[3] + SQRT_2 * (a[3] - a[6]) * b[4] + 2.0 * a[0] * b[5] - 2.0 * a[2] * b[5]) / 4.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor2;
    use russell_lab::mat_approx_eq;

    #[test]
    fn t2_matmul_gen_self_works() {
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
        t2_matmul(&mut c, 2.0, &a, false, &a, false).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [  60.00000,   72.00000,   84.00000],
            [ 132.00000,  162.00000,  192.00000],
            [ 204.00000,  252.00000,  300.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_tra_self_works() {
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        let mut c = Tensor2::<6>::new();
        t2_matmul(&mut c, 2.0, &a, true, &a, false).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 132.00000,  156.00000,  180.00000],
            [ 156.00000,  186.00000,  216.00000],
            [ 180.00000,  216.00000,  252.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_self_tra_works() {
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        let mut c = Tensor2::<6>::new();
        t2_matmul(&mut c, 2.0, &a, false, &a, true).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [  28.00000,   64.00000,  100.00000],
            [  64.00000,  154.00000,  244.00000],
            [ 100.00000,  244.00000,  388.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_sym_self_works() {
        #[rustfmt::skip]
        let a = Tensor2::<6>::from_std_matrix(&[
            [   1.00000,    4.00000,    6.00000],
            [   4.00000,    2.00000,    5.00000],
            [   6.00000,    5.00000,    3.00000],
        ]).unwrap();
        let mut c = Tensor2::<6>::new();
        t2_matmul(&mut c, 2.0, &a, false, &a, false).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 106.00000,   84.00000,   88.00000],
            [  84.00000,   90.00000,   98.00000],
            [  88.00000,   98.00000,  140.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_tra_gen_gen_works() {
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
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
    fn t2_matmul_gen_tra_gen_works() {
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
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
    fn t2_matmul_tra_gen_tra_gen_works() {
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
        t2_matmul(&mut c, 2.0, &a, true, &b, true).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 180.00000,  108.00000,   36.00000],
            [ 228.00000,  138.00000,   48.00000],
            [ 276.00000,  168.00000,   60.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_sym_tra_gen_works() {
        #[rustfmt::skip]
        let a = Tensor2::<6>::from_std_matrix(&[
            [   1.00000,    4.00000,    6.00000],
            [   4.00000,    2.00000,    5.00000],
            [   6.00000,    5.00000,    3.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
        t2_matmul(&mut c, 2.0, &a, false, &b, true).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 166.00000,  100.00000,   34.00000],
            [ 174.00000,  108.00000,   42.00000],
            [ 230.00000,  146.00000,   62.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_gen_tra_sym_works() {
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<6>::from_std_matrix(&[
            [   3.00000,    5.00000,    6.00000],
            [   5.00000,    2.00000,    4.00000],
            [   6.00000,    4.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
        t2_matmul(&mut c, 2.0, &a, true, &b, false).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 130.00000,   82.00000,   58.00000],
            [ 158.00000,  104.00000,   80.00000],
            [ 186.00000,  126.00000,  102.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_matmul_gen_gen_works() {
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
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
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<6>::from_std_matrix(&[
            [   3.00000,    5.00000,    6.00000],
            [   5.00000,    2.00000,    4.00000],
            [   6.00000,    4.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
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
        let a = Tensor2::<6>::from_std_matrix(&[
            [   1.00000,    4.00000,    6.00000],
            [   4.00000,    2.00000,    5.00000],
            [   6.00000,    5.00000,    3.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
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
        let a = Tensor2::<6>::from_std_matrix(&[
            [   1.00000,    4.00000,    6.00000],
            [   4.00000,    2.00000,    5.00000],
            [   6.00000,    5.00000,    3.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<6>::from_std_matrix(&[
            [   3.00000,    5.00000,    6.00000],
            [   5.00000,    2.00000,    4.00000],
            [   6.00000,    4.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
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
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
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
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<6>::new();
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
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<9>::new();
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
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [   9.00000,    8.00000,    7.00000],
            [   6.00000,    5.00000,    4.00000],
            [   3.00000,    2.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<6>::new();
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
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<6>::from_std_matrix(&[
            [   3.00000,    5.00000,    6.00000],
            [   5.00000,    2.00000,    4.00000],
            [   6.00000,    4.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<6>::new();
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
        let a = Tensor2::<9>::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ]).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::<6>::from_std_matrix(&[
            [   3.00000,    5.00000,    6.00000],
            [   5.00000,    2.00000,    4.00000],
            [   6.00000,    4.00000,    1.00000],
        ]).unwrap();
        let mut c = Tensor2::<6>::new();
        t2_matmulx(&mut c, 1.5, &a, false, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 648.00000,  850.50000, 1053.00000],
            [ 850.50000, 1107.00000, 1363.50000],
            [1053.00000, 1363.50000, 1674.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    //
    // --- dispatcher tests ---
    //

    #[test]
    fn test_t2_matmul_dispatcher() {
        let a_gen = Tensor2::<9>::new();
        let b_gen = Tensor2::<9>::new();
        let a_sym = Tensor2::<6>::new();
        let b_sym = Tensor2::<6>::new();
        let a_2d = Tensor2::<4>::new();
        let mut c_gen = Tensor2::<9>::new();
        let mut c_sym = Tensor2::<6>::new();
        let mut c_2d = Tensor2::<4>::new();

        assert!(t2_matmul(&mut c_gen, 1.0, &a_gen, true, &b_gen, true).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_gen, true, &b_gen, true).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_gen, true, &b_gen, false).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_gen, true, &b_gen, false).is_ok());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_gen, false, &b_gen, true).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_gen, false, &b_gen, true).is_ok());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_gen, false, &b_gen, false).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_gen, false, &b_gen, false).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_gen, true, &b_sym, true).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_gen, true, &b_sym, true).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_gen, true, &b_sym, false).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_gen, true, &b_sym, false).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_gen, false, &b_sym, true).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_gen, false, &b_sym, true).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_gen, false, &b_sym, false).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_gen, false, &b_sym, false).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_sym, true, &b_gen, true).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_sym, true, &b_gen, true).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_sym, true, &b_gen, false).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_sym, true, &b_gen, false).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_sym, false, &b_gen, true).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_sym, false, &b_gen, true).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_sym, false, &b_gen, false).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_sym, false, &b_gen, false).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_sym, true, &b_sym, true).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_sym, true, &b_sym, true).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_sym, true, &b_sym, false).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_sym, true, &b_sym, false).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_sym, false, &b_sym, true).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_sym, false, &b_sym, true).is_err());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_sym, false, &b_sym, false).is_ok());
        assert!(t2_matmul(&mut c_sym, 1.0, &a_sym, false, &b_sym, false).is_err());

        // Test ptr::eq(a, b) cases
        assert!(t2_matmul(&mut c_sym, 1.0, &a_gen, true, &a_gen, false).is_ok());
        assert!(t2_matmul(&mut c_gen, 1.0, &a_gen, true, &a_gen, false).is_err()); // Must fail because c must be Symmetric for A^T * A

        // c must be General for A·A (same tensor, no transpose)
        assert!(t2_matmul(&mut c_sym, 1.0, &a_gen, false, &a_gen, false).is_err());
        // c must be Symmetric for A·A (same symmetric tensor)
        assert!(t2_matmul(&mut c_gen, 1.0, &a_sym, false, &a_sym, false).is_err());
        // c must be Symmetric for A·Aᵀ (same tensor)
        assert!(t2_matmul(&mut c_gen, 1.0, &a_gen, false, &a_gen, true).is_err());
        // c must be Symmetric or General (c is Symmetric2D)
        assert!(t2_matmul(&mut c_2d, 1.0, &a_gen, true, &b_gen, false).is_err());
        assert!(t2_matmul(&mut c_2d, 1.0, &a_gen, false, &b_gen, true).is_err());
        // unsupported combination (Symmetric2D input)
        assert!(t2_matmul(&mut c_gen, 1.0, &a_2d, false, &b_gen, false).is_err());
    }

    #[test]
    fn test_t2_matmulx_dispatcher() {
        let a_gen = Tensor2::<9>::new();
        let b_gen = Tensor2::<9>::new();
        let a_sym = Tensor2::<6>::new();
        let b_sym = Tensor2::<6>::new();
        let mut c_gen = Tensor2::<9>::new();
        let mut c_sym = Tensor2::<6>::new();

        assert!(t2_matmulx(&mut c_gen, 1.0, &a_gen, true, &b_gen).is_err());
        assert!(t2_matmulx(&mut c_sym, 1.0, &a_gen, true, &b_gen).is_err());
        assert!(t2_matmulx(&mut c_gen, 1.0, &a_gen, false, &b_gen).is_err());
        assert!(t2_matmulx(&mut c_sym, 1.0, &a_gen, false, &b_gen).is_err());
        assert!(t2_matmulx(&mut c_gen, 1.0, &a_gen, true, &b_sym).is_err());
        assert!(t2_matmulx(&mut c_sym, 1.0, &a_gen, true, &b_sym).is_ok());
        assert!(t2_matmulx(&mut c_gen, 1.0, &a_gen, false, &b_sym).is_err());
        assert!(t2_matmulx(&mut c_sym, 1.0, &a_gen, false, &b_sym).is_ok());
        assert!(t2_matmulx(&mut c_gen, 1.0, &a_sym, true, &b_gen).is_err());
        assert!(t2_matmulx(&mut c_sym, 1.0, &a_sym, true, &b_gen).is_err());
        assert!(t2_matmulx(&mut c_gen, 1.0, &a_sym, false, &b_gen).is_err());
        assert!(t2_matmulx(&mut c_sym, 1.0, &a_sym, false, &b_gen).is_err());
        assert!(t2_matmulx(&mut c_gen, 1.0, &a_sym, true, &b_sym).is_err());
        assert!(t2_matmulx(&mut c_sym, 1.0, &a_sym, true, &b_sym).is_err());
        assert!(t2_matmulx(&mut c_gen, 1.0, &a_sym, false, &b_sym).is_err());
        assert!(t2_matmulx(&mut c_sym, 1.0, &a_sym, false, &b_sym).is_err());
    }
}
