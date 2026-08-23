use super::Tensor2;
use crate::{Rep, SQRT_2};

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
/// * `cc` -- the resulting tensor C; must be [Rep::General]
///
/// # Input
///
/// * `aa` -- the first tensor A; must be [Rep::General]
/// * `bb` -- the second tensor B; must be [Rep::General]
///
/// # Panics
/// 
/// A panic will occur if `cc` is not [Rep::General], `aa` is not [Rep::General], or `bb` is not [Rep::General].
#[rustfmt::skip]
pub fn t2_gen_dot_gen(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(aa.rep(), Rep::General);
    assert_eq!(bb.rep(), Rep::General);
    assert_eq!(cc.rep(), Rep::General);
    let c = &mut cc.vec;
    let a = &aa.vec;
    let b = &bb.vec;
    c[0] = (2.0 * a[0] * b[0] + (a[3] + a[6]) * (b[3] - b[6]) + (a[5] + a[8]) * (b[5] - b[8])) / 2.0;
    c[1] = (2.0 * a[1] * b[1] + (a[3] - a[6]) * (b[3] + b[6]) + (a[4] + a[7]) * (b[4] - b[7])) / 2.0;
    c[2] = (2.0 * a[2] * b[2] + (a[4] - a[7]) * (b[4] + b[7]) + (a[5] - a[8]) * (b[5] + b[8])) / 2.0;
    c[3] = (2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] - b[7]) + SQRT_2 * (a[4] + a[7]) * (b[5] - b[8])) / 4.0;
    c[4] = (2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] + SQRT_2 * (a[5] - a[8]) * (b[3] + b[6]) + 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] + b[8])) / 4.0;
    c[5] = (2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] + SQRT_2 * (a[4] - a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] + b[7]) + 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0;
    c[6] = (-2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] - 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] - b[7]) - SQRT_2 * (a[4] + a[7]) * (b[5] - b[8])) / 4.0;
    c[7] = (-2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] - SQRT_2 * (a[5] - a[8]) * (b[3] + b[6]) - 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] + b[8])) / 4.0;
    c[8] = (-2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] - SQRT_2 * (a[4] - a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] + b[7]) - 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0;
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
/// * `cc` -- the resulting tensor C; must be [Rep::General]
///
/// # Input
///
/// * `aa` -- the first tensor A; must be [Rep::General]
/// * `bb` -- the second tensor B; must be [Rep::Symmetric]
///
/// # Panics
/// 
/// A panic will occur if `cc` is not [Rep::General], `aa` is not [Rep::General], or `bb` is not [Rep::Symmetric].
#[rustfmt::skip]
pub fn t2_gen_dot_sym(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(aa.rep(), Rep::General);
    assert_eq!(bb.rep(), Rep::Symmetric);
    assert_eq!(cc.rep(), Rep::General);
    t2_gen_dot_sym_stack(cc.as_mut_data(), aa.as_data(), bb.as_data());
}

/// Performs the general tensor dot symmetric tensor operation (stack version): C = A · B
///
/// Note: B must be symmetric (components `b[6]`, `b[7]`, `b[8]` are zero).
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
/// * `cc` -- the resulting tensor C; must be [Rep::General]
///
/// # Input
///
/// * `aa` -- the first tensor A; must be [Rep::Symmetric]
/// * `bb` -- the second tensor B; must be [Rep::General]
///
/// # Panics
/// 
/// A panic will occur if `cc` is not [Rep::General], `aa` is not [Rep::Symmetric], or `bb` is not [Rep::General].
#[rustfmt::skip]
pub fn t2_sym_dot_gen(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(aa.rep(), Rep::Symmetric);
    assert_eq!(bb.rep(), Rep::General);
    assert_eq!(cc.rep(), Rep::General);
    let c = &mut cc.vec;
    let a = &aa.vec;
    let b = &bb.vec;
    c[0] = (2.0 * a[0] * b[0] + a[3] * (b[3] - b[6]) + a[5] * (b[5] - b[8])) / 2.0;
    c[1] = (2.0 * a[1] * b[1] + a[3] * (b[3] + b[6]) + a[4] * (b[4] - b[7])) / 2.0;
    c[2] = (2.0 * a[2] * b[2] + a[4] * (b[4] + b[7]) + a[5] * (b[5] + b[8])) / 2.0;
    c[3] = (SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[4] * b[5] - SQRT_2 * a[1] * b[6] + SQRT_2 * a[0] * (b[3] + b[6]) - a[5] * b[7] - a[4] * b[8]) / (2.0 * SQRT_2);
    c[4] = (2.0 * a[4] * b[1] + 2.0 * a[4] * b[2] + SQRT_2 * a[5] * (b[3] + b[6]) + 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * a[3] * (b[5] + b[8])) / 4.0;
    c[5] = (2.0 * a[5] * b[0] + 2.0 * a[5] * b[2] + SQRT_2 * a[4] * (b[3] - b[6]) + SQRT_2 * a[3] * (b[4] + b[7]) + 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0;
    c[6] = (SQRT_2 * a[3] * (-b[0] + b[1]) - SQRT_2 * a[1] * b[3] + a[5] * b[4] - a[4] * b[5] + SQRT_2 * a[1] * b[6] + SQRT_2 * a[0] * (b[3] + b[6]) - a[5] * b[7] + a[4] * b[8]) / (2.0 * SQRT_2);
    c[7] = (-2.0 * a[4] * b[1] + 2.0 * a[4] * b[2] - SQRT_2 * a[5] * (b[3] + b[6]) - 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * a[3] * (b[5] + b[8])) / 4.0;
    c[8] = (-2.0 * a[5] * b[0] + 2.0 * a[5] * b[2] - SQRT_2 * a[4] * (b[3] - b[6]) + SQRT_2 * a[3] * (b[4] + b[7]) - 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0;
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
/// * `cc` -- the resulting tensor C; must be [Rep::General]
///
/// # Input
///
/// * `aa` -- the first tensor A; must be [Rep::Symmetric]
/// * `bb` -- the second tensor B; must be [Rep::Symmetric]
///
/// # Panics
/// 
/// A panic will occur if `cc` is not [Rep::General], `aa` is not [Rep::Symmetric], or `bb` is not [Rep::Symmetric].
#[rustfmt::skip]
pub fn t2_sym_dot_sym(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(aa.rep(), Rep::Symmetric);
    assert_eq!(bb.rep(), Rep::Symmetric);
    assert_eq!(cc.rep(), Rep::General);
    let c = &mut cc.vec;
    let a = &aa.vec;
    let b = &bb.vec;
    c[0] = (2.0 * a[0] * b[0] + a[3] * b[3] + a[5] * b[5]) / 2.0;
    c[1] = (2.0 * a[1] * b[1] + a[3] * b[3] + a[4] * b[4]) / 2.0;
    c[2] = (2.0 * a[2] * b[2] + a[4] * b[4] + a[5] * b[5]) / 2.0;
    c[3] = (SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[4] * b[5]) / (2.0 * SQRT_2);
    c[4] = (SQRT_2 * a[4] * (b[1] + b[2]) + a[5] * b[3] + SQRT_2 * a[1] * b[4] + SQRT_2 * a[2] * b[4] + a[3] * b[5]) / (2.0 * SQRT_2);
    c[5] = (SQRT_2 * a[5] * (b[0] + b[2]) + a[4] * b[3] + a[3] * b[4] + SQRT_2 * a[0] * b[5] + SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2);
    c[6] = (SQRT_2 * a[3] * (-b[0] + b[1]) + SQRT_2 * a[0] * b[3] - SQRT_2 * a[1] * b[3] + a[5] * b[4] - a[4] * b[5]) / (2.0 * SQRT_2);
    c[7] = (SQRT_2 * a[4] * (-b[1] + b[2]) - a[5] * b[3] + SQRT_2 * a[1] * b[4] - SQRT_2 * a[2] * b[4] + a[3] * b[5]) / (2.0 * SQRT_2);
    c[8] = (SQRT_2 * a[5] * (-b[0] + b[2]) - a[4] * b[3] + a[3] * b[4] + SQRT_2 * a[0] * b[5] - SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2);
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
/// * `cc` -- the resulting tensor C; must be [Rep::Symmetric]
///
/// # Input
///
/// * `aa` -- the first tensor A; must be [Rep::General]
///
/// # Panics
/// 
/// A panic will occur if `cc` is not [Rep::Symmetric] or `aa` is not [Rep::General].
#[rustfmt::skip]
pub fn t2_gen_tra_dot_self(cc: &mut Tensor2, aa: &Tensor2) {
    assert_eq!(aa.rep(), Rep::General);
    assert_eq!(cc.rep(), Rep::Symmetric);
    t2_gen_tra_dot_self_stack(cc.as_mut_data(), aa.as_data());
}

/// Performs the general transposed tensor dot itself operation (stack version): C = Aᵀ · A
///
/// Note: only the symmetric components `c[0..6]` are written; the skew
/// components `c[6]`, `c[7]`, `c[8]` are left untouched.
#[rustfmt::skip]
pub(crate) fn t2_gen_tra_dot_self_stack(c: &mut [f64], a: &[f64]) {
    c[0] = (2.0 * (a[0] * a[0]) + (a[3] - a[6]) * (a[3] - a[6]) + (a[5] - a[8]) * (a[5] - a[8])) / 2.0;
    c[1] = (2.0 * (a[1] * a[1]) + (a[3] + a[6]) * (a[3] + a[6]) + (a[4] - a[7]) * (a[4] - a[7])) / 2.0;
    c[2] = (2.0 * (a[2] * a[2]) + (a[4] + a[7]) * (a[4] + a[7]) + (a[5] + a[8]) * (a[5] + a[8])) / 2.0;
    c[3] = a[1] * (a[3] - a[6]) + a[0] * (a[3] + a[6]) + (a[4] - a[7]) * (a[5] - a[8]) / SQRT_2;
    c[4] = a[2] * (a[4] - a[7]) + a[1] * (a[4] + a[7]) + (a[3] + a[6]) * (a[5] + a[8]) / SQRT_2;
    c[5] = (a[3] - a[6]) * (a[4] + a[7]) / SQRT_2 + a[2] * (a[5] - a[8]) + a[0] * (a[5] + a[8]);
}

/// Performs the general transposed tensor dot general tensor operation: C = Aᵀ · B
/// 
/// Computes:
///
/// ```text
/// C = Aᵀ · B
/// ```
///
/// When `chop` is `true`, only the symmetric part of C is computed and the
/// skew components are chopped off (discarded), so `cc` must be
/// [Rep::Symmetric]. This is used, for example, to compute the right
/// stretch `U = Rᵀ · F`, which is symmetric.
///
/// When `chop` is `false`, the full (general) result is computed, including
/// the skew components `c[6]`, `c[7]`, `c[8]`.
///
/// # Output
///
/// * `cc` -- the resulting tensor C; must be [Rep::Symmetric] if chop is true, or [Rep::General] otherwise
///
/// # Input
///
/// * `aa` -- the first tensor A; must be [Rep::General]
/// * `bb` -- the second tensor B; must be [Rep::General]
/// * `chop` -- if true, ignores the skew components and treats C as symmetric
///
/// # Panics
/// 
/// A panic will occur if `aa` is not [Rep::General], `bb` is not [Rep::General], or `cc` representation does not match the chop flag.
#[rustfmt::skip]
pub fn t2_gen_tra_dot_gen_chop(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2, chop: bool) {
    assert_eq!(aa.rep(), Rep::General);
    assert_eq!(bb.rep(), Rep::General);
    if chop {
        assert_eq!(cc.rep(), Rep::Symmetric);
    } else {
        assert_eq!(cc.rep(), Rep::General);
    }
    let c = &mut cc.vec;
    let a = &aa.vec;
    let b = &bb.vec;
    c[0] = (2.0 * a[0] * b[0] + (a[3] - a[6]) * (b[3] - b[6]) + (a[5] - a[8]) * (b[5] - b[8])) / 2.0;
    c[1] = (2.0 * a[1] * b[1] + (a[3] + a[6]) * (b[3] + b[6]) + (a[4] - a[7]) * (b[4] - b[7])) / 2.0;
    c[2] = (2.0 * a[2] * b[2] + (a[4] + a[7]) * (b[4] + b[7]) + (a[5] + a[8]) * (b[5] + b[8])) / 2.0;
    c[3] = (2.0 * (a[3] + a[6]) * b[0] + 2.0 * (a[3] - a[6]) * b[1] + 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] - a[8]) * (b[4] - b[7]) + SQRT_2 * (a[4] - a[7]) * (b[5] - b[8])) / 4.0;
    c[4] = (2.0 * (a[4] + a[7]) * b[1] + 2.0 * (a[4] - a[7]) * b[2] + SQRT_2 * (a[5] + a[8]) * (b[3] + b[6]) + 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] + a[6]) * (b[5] + b[8])) / 4.0;
    c[5] = (2.0 * (a[5] + a[8]) * b[0] + 2.0 * (a[5] - a[8]) * b[2] + SQRT_2 * (a[4] + a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] - a[6]) * (b[4] + b[7]) + 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0;
    if !chop {
        c[6] = (-2.0 * (a[3] + a[6]) * b[0] + 2.0 * (a[3] - a[6]) * b[1] - 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] - a[8]) * (b[4] - b[7]) - SQRT_2 * (a[4] - a[7]) * (b[5] - b[8])) / 4.0;
        c[7] = (-2.0 * (a[4] + a[7]) * b[1] + 2.0 * (a[4] - a[7]) * b[2] - SQRT_2 * (a[5] + a[8]) * (b[3] + b[6]) - 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] + a[6]) * (b[5] + b[8])) / 4.0;
        c[8] = (-2.0 * (a[5] + a[8]) * b[0] + 2.0 * (a[5] - a[8]) * b[2] - SQRT_2 * (a[4] + a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] - a[6]) * (b[4] + b[7]) - 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0;
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
/// When `chop` is `true`, only the symmetric part of C is computed and the
/// skew components are chopped off (discarded), so `cc` must be
/// [Rep::Symmetric]. This is used, for example, to compute the left
/// stretch `V = F · Rᵀ`, which is symmetric.
///
/// When `chop` is `false`, the full (general) result is computed, including
/// the skew components `c[6]`, `c[7]`, `c[8]`.
///
/// # Output
///
/// * `cc` -- the resulting tensor C; must be [Rep::Symmetric] if chop is true, or [Rep::General] otherwise
///
/// # Input
///
/// * `aa` -- the first tensor A; must be [Rep::General]
/// * `bb` -- the second tensor B; must be [Rep::General]
/// * `chop` -- if true, ignores the skew components and treats C as symmetric
///
/// # Panics
/// 
/// A panic will occur if `aa` is not [Rep::General], `bb` is not [Rep::General], or `cc` representation does not match the chop flag.
#[rustfmt::skip]
pub fn t2_gen_dot_gen_tra_chop(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2, chop: bool) {
    assert_eq!(aa.rep(), Rep::General);
    assert_eq!(bb.rep(), Rep::General);
    if chop {
        assert_eq!(cc.rep(), Rep::Symmetric);
    } else {
        assert_eq!(cc.rep(), Rep::General);
    }
    let c = &mut cc.vec;
    let a = &aa.vec;
    let b = &bb.vec;
    c[0] = (2.0 * a[0] * b[0] + (a[3] + a[6]) * (b[3] + b[6]) + (a[5] + a[8]) * (b[5] + b[8])) / 2.0;
    c[1] = (2.0 * a[1] * b[1] + (a[3] - a[6]) * (b[3] - b[6]) + (a[4] + a[7]) * (b[4] + b[7])) / 2.0;
    c[2] = (2.0 * a[2] * b[2] + (a[4] - a[7]) * (b[4] - b[7]) + (a[5] - a[8]) * (b[5] - b[8])) / 2.0;
    c[3] = (2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[0] * (b[3] - b[6]) + 2.0 * a[1] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] + b[7]) + SQRT_2 * (a[4] + a[7]) * (b[5] + b[8])) / 4.0;
    c[4] = (2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] + SQRT_2 * (a[5] - a[8]) * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) + 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] - b[8])) / 4.0;
    c[5] = (2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] + SQRT_2 * (a[4] - a[7]) * (b[3] + b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) + 2.0 * a[2] * (b[5] + b[8])) / 4.0;
    if !chop {
        c[6] = (-2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[0] * (b[3] - b[6]) - 2.0 * a[1] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] + b[7]) - SQRT_2 * (a[4] + a[7]) * (b[5] + b[8])) / 4.0;
        c[7] = (-2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] - SQRT_2 * (a[5] - a[8]) * (b[3] - b[6]) + 2.0 * a[1] * (b[4] - b[7]) - 2.0 * a[2] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] - b[8])) / 4.0;
        c[8] = (-2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] - SQRT_2 * (a[4] - a[7]) * (b[3] + b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] - b[7]) + 2.0 * a[0] * (b[5] - b[8]) - 2.0 * a[2] * (b[5] + b[8])) / 4.0;
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
/// * `cc` -- the resulting tensor C; must be [Rep::Symmetric]
///
/// # Input
///
/// * `aa` -- the first tensor A; must be [Rep::General]
/// * `bb` -- the second tensor B; must be [Rep::Symmetric]
///
/// # Panics
/// 
/// A panic will occur if `cc` is not [Rep::Symmetric], `aa` is not [Rep::General], or `bb` is not [Rep::Symmetric].
#[rustfmt::skip]
pub fn t2_gen_dot_sym_dot_self_tra(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(aa.rep(), Rep::General);
    assert_eq!(bb.rep(), Rep::Symmetric);
    assert_eq!(cc.rep(), Rep::Symmetric);
    let c = &mut cc.vec;
    let a = &aa.vec;
    let b = &bb.vec;
    c[0] = (SQRT_2 * (a[3] + a[6]) * (SQRT_2 * (a[3] + a[6]) * b[1] + SQRT_2 * a[0] * b[3] + (a[5] + a[8]) * b[4]) + SQRT_2 * (a[5] + a[8]) * (SQRT_2 * (a[5] + a[8]) * b[2] + (a[3] + a[6]) * b[4] + SQRT_2 * a[0] * b[5]) + 2.0 * a[0] * (2.0 * a[0] * b[0] + (a[3] + a[6]) * b[3] + (a[5] + a[8]) * b[5])) / 4.0;
    c[1] = (a[3] * a[3] * b[0] + a[6] * a[6] * b[0] + 2.0 * (a[1] * a[1]) * b[1] + a[4] * a[4] * b[2] + 2.0 * a[4] * a[7] * b[2] + a[7] * a[7] * b[2] + 2.0 * a[1] * a[4] * b[4] + 2.0 * a[1] * a[7] * b[4] - a[6] * (2.0 * a[1] * b[3] + SQRT_2 * (a[4] + a[7]) * b[5]) + a[3] * (-2.0 * a[6] * b[0] + 2.0 * a[1] * b[3] + SQRT_2 * (a[4] + a[7]) * b[5])) / 2.0;
    c[2] = (a[5] * a[5] * b[0] + a[8] * a[8] * b[0] + a[4] * a[4] * b[1] - 2.0 * a[4] * a[7] * b[1] + a[7] * a[7] * b[1] + 2.0 * (a[2] * a[2]) * b[2] + 2.0 * a[2] * a[4] * b[4] - 2.0 * a[2] * a[7] * b[4] + a[8] * (-(SQRT_2 * a[4] * b[3]) + SQRT_2 * a[7] * b[3] - 2.0 * a[2] * b[5]) + a[5] * (-2.0 * a[8] * b[0] + SQRT_2 * a[4] * b[3] - SQRT_2 * a[7] * b[3] + 2.0 * a[2] * b[5])) / 2.0;
    c[3] = (SQRT_2 * a[4] * a[5] * b[2] + SQRT_2 * a[5] * a[7] * b[2] + SQRT_2 * a[4] * a[8] * b[2] + SQRT_2 * a[7] * a[8] * b[2] + a[3] * a[3] * b[3] - a[6] * a[6] * b[3] + a[3] * a[4] * b[4] + a[4] * a[6] * b[4] + a[3] * a[7] * b[4] + a[6] * a[7] * b[4] + a[1] * (2.0 * a[3] * b[1] + 2.0 * a[6] * b[1] + SQRT_2 * (a[5] + a[8]) * b[4]) + a[3] * a[5] * b[5] - a[5] * a[6] * b[5] + a[3] * a[8] * b[5] - a[6] * a[8] * b[5] + a[0] * (2.0 * a[3] * b[0] - 2.0 * a[6] * b[0] + 2.0 * a[1] * b[3] + SQRT_2 * a[4] * b[5] + SQRT_2 * a[7] * b[5])) / 2.0;
    c[4] = (SQRT_2 * a[6] * a[8] * b[0] + 2.0 * a[1] * a[4] * b[1] - 2.0 * a[1] * a[7] * b[1] + 2.0 * a[2] * a[4] * b[2] + 2.0 * a[2] * a[7] * b[2] - a[4] * a[6] * b[3] + a[6] * a[7] * b[3] - SQRT_2 * a[1] * a[8] * b[3] + 2.0 * a[1] * a[2] * b[4] + a[4] * a[4] * b[4] - a[7] * a[7] * b[4] - SQRT_2 * a[2] * a[6] * b[5] - a[4] * a[8] * b[5] - a[7] * a[8] * b[5] + a[3] * (SQRT_2 * a[5] * b[0] - SQRT_2 * a[8] * b[0] + a[4] * b[3] - a[7] * b[3] + SQRT_2 * a[2] * b[5]) + a[5] * (-(SQRT_2 * a[6] * b[0]) + SQRT_2 * a[1] * b[3] + (a[4] + a[7]) * b[5])) / 2.0;
    c[5] = (SQRT_2 * a[4] * a[6] * b[1] - SQRT_2 * a[6] * a[7] * b[1] + 2.0 * a[2] * a[5] * b[2] + 2.0 * a[2] * a[8] * b[2] + a[5] * a[6] * b[3] - a[6] * a[8] * b[3] + a[4] * a[5] * b[4] + SQRT_2 * a[2] * a[6] * b[4] - a[5] * a[7] * b[4] + a[4] * a[8] * b[4] - a[7] * a[8] * b[4] + a[3] * (SQRT_2 * a[4] * b[1] - SQRT_2 * a[7] * b[1] + a[5] * b[3] - a[8] * b[3] + SQRT_2 * a[2] * b[4]) + a[5] * a[5] * b[5] - a[8] * a[8] * b[5] + a[0] * (2.0 * a[5] * b[0] - 2.0 * a[8] * b[0] + SQRT_2 * a[4] * b[3] - SQRT_2 * a[7] * b[3] + 2.0 * a[2] * b[5])) / 2.0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rep, Tensor2};
    use russell_lab::mat_approx_eq;

    #[test]
    fn t2_gen_dot_gen_works() {
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
        t2_gen_dot_gen(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = &[
            [  30.00000,   24.00000,   18.00000],
            [  84.00000,   69.00000,   54.00000],
            [ 138.00000,  114.00000,   90.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_gen_dot_sym_works() {
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
        t2_gen_dot_sym(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = &[
            [  31.00000,   21.00000,   17.00000],
            [  73.00000,   54.00000,   50.00000],
            [ 115.00000,   87.00000,   83.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_sym_dot_gen_works() {
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
        t2_sym_dot_gen(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = &[
            [  51.00000,   40.00000,   29.00000],
            [  63.00000,   52.00000,   41.00000],
            [  93.00000,   79.00000,   65.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_sym_dot_sym_works() {
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
        t2_sym_dot_sym(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = &[
            [  59.00000,   37.00000,   28.00000],
            [  52.00000,   44.00000,   37.00000],
            [  61.00000,   52.00000,   59.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_gen_tra_dot_self_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [   1.00000,    2.00000,    3.00000],
            [   4.00000,    5.00000,    6.00000],
            [   7.00000,    8.00000,    9.00000],
        ], Rep::General).unwrap();
        let mut c = Tensor2::new(Rep::Symmetric);
        t2_gen_tra_dot_self(&mut c, &a);
        #[rustfmt::skip]
        let correct = &[
            [  66.00000,   78.00000,   90.00000],
            [  78.00000,   93.00000,  108.00000],
            [  90.00000,  108.00000,  126.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_gen_tra_dot_gen_chop_works() {
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
        t2_gen_tra_dot_gen_chop(&mut c, &a, &b, false);
        #[rustfmt::skip]
        let correct = &[
            [  54.00000,   42.00000,   30.00000],
            [  72.00000,   57.00000,   42.00000],
            [  90.00000,   72.00000,   54.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_gen_tra_dot_gen_chop_works_chopped() {
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
        t2_gen_tra_dot_gen_chop(&mut c, &a, &b, true);
        #[rustfmt::skip]
        let correct = &[
            [  54.00000,   57.00000,   60.00000],
            [  57.00000,   57.00000,   57.00000],
            [  60.00000,   57.00000,   54.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_gen_dot_gen_tra_chop_works() {
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
        t2_gen_dot_gen_tra_chop(&mut c, &a, &b, false);
        #[rustfmt::skip]
        let correct = &[
            [  46.00000,   28.00000,   10.00000],
            [ 118.00000,   73.00000,   28.00000],
            [ 190.00000,  118.00000,   46.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_gen_dot_gen_tra_chop_works_chopped() {
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
        t2_gen_dot_gen_tra_chop(&mut c, &a, &b, true);
        #[rustfmt::skip]
        let correct = &[
            [  46.00000,   73.00000,  100.00000],
            [  73.00000,   73.00000,   73.00000],
            [ 100.00000,   73.00000,   46.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

    #[test]
    fn t2_gen_dot_sym_dot_self_tra_works() {
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
        t2_gen_dot_sym_dot_self_tra(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = &[
            [ 124.00000,  331.00000,  538.00000],
            [ 331.00000,  862.00000, 1393.00000],
            [ 538.00000, 1393.00000, 2248.00000],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-12);
    }

}