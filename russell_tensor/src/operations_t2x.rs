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
/// A panic will occur if any of the tensors are not [Rep::General].
#[rustfmt::skip]
pub fn t2_gen_dot_gen(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(aa.rep(), Rep::General);
    assert_eq!(bb.rep(), Rep::General);
    assert_eq!(cc.rep(), Rep::General);
    let a = &aa.vec;
    let b = &bb.vec;
    let c = &mut cc.vec;
    c[0] = 0.5 * (2.0 * a[0] * b[0] + (a[3] + a[6]) * (b[3] - b[6]) + (a[5] + a[8]) * (b[5] - b[8]));
    c[1] = 0.5 * (2.0 * a[1] * b[1] + (a[3] - a[6]) * (b[3] + b[6]) + (a[4] + a[7]) * (b[4] - b[7]));
    c[2] = 0.5 * (2.0 * a[2] * b[2] + (a[4] - a[7]) * (b[4] + b[7]) + (a[5] - a[8]) * (b[5] + b[8]));
    c[3] = 0.25 * (2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] - b[7]) + SQRT_2 * (a[4] + a[7]) * (b[5] - b[8]));
    c[4] = 0.25 * (2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] + SQRT_2 * (a[5] - a[8]) * (b[3] + b[6]) + 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] + b[8]));
    c[5] = 0.25 * (2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] + SQRT_2 * (a[4] - a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] + b[7]) + 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8]));
    c[6] = 0.25 * (-2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] - 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] - b[7]) - SQRT_2 * (a[4] + a[7]) * (b[5] - b[8]));
    c[7] = 0.25 * (-2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] - SQRT_2 * (a[5] - a[8]) * (b[3] + b[6]) - 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] + b[8]));
    c[8] = 0.25 * (-2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] - SQRT_2 * (a[4] - a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] + b[7]) - 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8]));
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
/// * `bb` -- the second tensor B (symmetric); must NOT be [Rep::General]
///
/// # Panics
/// 
/// A panic will occur if `aa` and `cc` are not [Rep::General], or if `bb` is [Rep::General].
#[rustfmt::skip]
pub fn t2_gen_dot_sym(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(aa.rep(), Rep::General);
    assert!(bb.rep() != Rep::General);
    assert_eq!(cc.rep(), Rep::General);
    let a = &aa.vec;
    let b = &bb.vec;
    let c = &mut cc.vec;
    c[0] = 0.5 * (2.0 * a[0] * b[0] + (a[3] + a[6]) * b[3] + (a[5] + a[8]) * b[5]);
    c[1] = 0.5 * (2.0 * a[1] * b[1] + (a[3] - a[6]) * b[3] + (a[4] + a[7]) * b[4]);
    c[2] = 0.5 * (2.0 * a[2] * b[2] + (a[4] - a[7]) * b[4] + (a[5] - a[8]) * b[5]);
    c[3] = (SQRT_2 * a[6] * (b[1] - b[0]) + SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[8] * b[4] + a[4] * b[5] + a[7] * b[5]) / (2.0 * SQRT_2);
    c[4] = (SQRT_2 * a[7] * (b[2] - b[1]) + SQRT_2 * a[4] * (b[1] + b[2]) + a[5] * b[3] - a[8] * b[3] + SQRT_2 * a[1] * b[4] + SQRT_2 * a[2] * b[4] + a[3] * b[5] - a[6] * b[5]) / (2.0 * SQRT_2);
    c[5] = (SQRT_2 * a[8] * (b[2] - b[0]) + SQRT_2 * a[5] * (b[0] + b[2]) + a[4] * b[3] - a[7] * b[3] + a[3] * b[4] + a[6] * b[4] + SQRT_2 * a[0] * b[5] + SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2);
    c[6] = (SQRT_2 * a[3] * (b[1] - b[0]) + SQRT_2 * a[6] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] - SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[8] * b[4] - a[4] * b[5] - a[7] * b[5]) / (2.0 * SQRT_2);
    c[7] = (SQRT_2 * a[4] * (b[2] - b[1]) + SQRT_2 * a[7] * (b[1] + b[2]) - a[5] * b[3] + a[8] * b[3] + SQRT_2 * a[1] * b[4] - SQRT_2 * a[2] * b[4] + a[3] * b[5] - a[6] * b[5]) / (2.0 * SQRT_2);
    c[8] = (SQRT_2 * a[5] * (b[2] - b[0]) + SQRT_2 * a[8] * (b[0] + b[2]) - a[4] * b[3] + a[7] * b[3] + a[3] * b[4] + a[6] * b[4] + SQRT_2 * a[0] * b[5] - SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2);
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
/// * `aa` -- the first tensor A (symmetric); must NOT be [Rep::General]
/// * `bb` -- the second tensor B; must be [Rep::General]
///
/// # Panics
/// 
/// A panic will occur if `bb` and `cc` are not [Rep::General], or if `aa` is [Rep::General].
#[rustfmt::skip]
pub fn t2_sym_dot_gen(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2) {
    assert!(aa.rep() != Rep::General);
    assert_eq!(bb.rep(), Rep::General);
    assert_eq!(cc.rep(), Rep::General);
    let a = &aa.vec;
    let b = &bb.vec;
    let c = &mut cc.vec;
    c[0] = 0.5 * (2.0 * a[0] * b[0] + a[3] * (b[3] - b[6]) + a[5] * (b[5] - b[8]));
    c[1] = 0.5 * (2.0 * a[1] * b[1] + a[3] * (b[3] + b[6]) + a[4] * (b[4] - b[7]));
    c[2] = 0.5 * (2.0 * a[2] * b[2] + a[4] * (b[4] + b[7]) + a[5] * (b[5] + b[8]));
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
/// * `aa` -- the first tensor A (symmetric); must NOT be [Rep::General]
/// * `bb` -- the second tensor B (symmetric); must NOT be [Rep::General]
///
/// # Panics
/// 
/// A panic will occur if `aa` or `bb` are [Rep::General], or if `cc` is not [Rep::General].
#[rustfmt::skip]
pub fn t2_sym_dot_sym(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2) {
    assert!(aa.rep() != Rep::General);
    assert!(bb.rep() != Rep::General);
    assert_eq!(cc.rep(), Rep::General);
    let a = &aa.vec;
    let b = &bb.vec;
    let c = &mut cc.vec;
    c[0] = 0.5 * (2.0 * a[0] * b[0] + a[3] * b[3] + a[5] * b[5]);
    c[1] = 0.5 * (2.0 * a[1] * b[1] + a[3] * b[3] + a[4] * b[4]);
    c[2] = 0.5 * (2.0 * a[2] * b[2] + a[4] * b[4] + a[5] * b[5]);
    c[3] = (SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[4] * b[5]) / (2.0 * SQRT_2);
    c[4] = (SQRT_2 * a[4] * (b[1] + b[2]) + a[5] * b[3] + SQRT_2 * a[1] * b[4] + SQRT_2 * a[2] * b[4] + a[3] * b[5]) / (2.0 * SQRT_2);
    c[5] = (SQRT_2 * a[5] * (b[0] + b[2]) + a[4] * b[3] + a[3] * b[4] + SQRT_2 * a[0] * b[5] + SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2);
    c[6] = (SQRT_2 * a[3] * (-b[0] + b[1]) + SQRT_2 * a[0] * b[3] - SQRT_2 * a[1] * b[3] + a[5] * b[4] - a[4] * b[5]) / (2.0 * SQRT_2);
    c[7] = (SQRT_2 * a[4] * (-b[1] + b[2]) - a[5] * b[3] + SQRT_2 * a[1] * b[4] - SQRT_2 * a[2] * b[4] + a[3] * b[5]) / (2.0 * SQRT_2);
    c[8] = (SQRT_2 * a[5] * (-b[0] + b[2]) - a[4] * b[3] + a[3] * b[4] + SQRT_2 * a[0] * b[5] - SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2);
}

/// Performs the transpose(Tensor2) dot Tensor2 operation: C = Aᵀ · A
///
/// The result C is symmetric.
///
/// # Output
///
/// * `cc` -- the resulting symmetric tensor C = Aᵀ · A; must be [Rep::Symmetric]
///
/// # Input
///
/// * `aa` -- the tensor A; must be [Rep::General]
///
/// # Panics
///
/// A panic will occur if the required [Rep] enums are incorrect.
#[rustfmt::skip]
pub fn t2_gen_tra_dot_self(cc: &mut Tensor2, aa: &Tensor2) {
    assert_eq!(aa.rep(), Rep::General);
    assert_eq!(cc.rep(), Rep::Symmetric);
    let a = &aa.vec;
    let c = &mut cc.vec;
    c[0] = 0.5 * (2.0 * a[0] * a[0] + (a[3] - a[6]) * (a[3] - a[6]) + (a[5] - a[8]) * (a[5] - a[8]));
    c[1] = 0.5 * (2.0 * a[1] * a[1] + (a[3] + a[6]) * (a[3] + a[6]) + (a[4] - a[7]) * (a[4] - a[7]));
    c[2] = 0.5 * (2.0 * a[2] * a[2] + (a[4] + a[7]) * (a[4] + a[7]) + (a[5] + a[8]) * (a[5] + a[8]));
    c[3] = a[1] * (a[3] - a[6]) + a[0] * (a[3] + a[6]) + ((a[4] - a[7]) * (a[5] - a[8])) / SQRT_2;
    c[4] = a[2] * (a[4] - a[7]) + a[1] * (a[4] + a[7]) + ((a[3] + a[6]) * (a[5] + a[8])) / SQRT_2;
    c[5] = ((a[3] - a[6]) * (a[4] + a[7])) / SQRT_2 + a[2] * (a[5] - a[8]) + a[0] * (a[5] + a[8]);
}

/// Performs the transpose(general tensor) dot general tensor operation: C = Aᵀ · B
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
/// When `chop` is `false`, the full (general) result is expected, but the
/// skew components `c[6]`, `c[7]`, `c[8]` are not implemented yet.
///
/// # Output
///
/// * `cc` -- the resulting tensor C; must be [Rep::Symmetric] if `chop` is
///   `true`, or [Rep::General] if `chop` is `false`
///
/// # Input
///
/// * `aa` -- the first tensor A (which gets transposed); must be [Rep::General]
/// * `bb` -- the second tensor B; must be [Rep::General]
/// * `chop` -- if `true`, chop off (discard) the skew components of C
///
/// # Panics
/// 
/// A panic will occur if `aa` and `bb` are not [Rep::General], or if `cc`
/// does not have the [Rep] required by `chop`.
#[rustfmt::skip]
pub fn t2_gen_tra_dot_gen_chop(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2, chop: bool) {
    assert_eq!(aa.rep(), Rep::General);
    assert_eq!(bb.rep(), Rep::General);
    let a = &aa.vec;
    let b = &bb.vec;
    let c = &mut cc.vec;
    c[0] = a[0] * b[0] + 0.5 * (a[3] - a[6]) * (b[3] - b[6]) + 0.5 * (a[5] - a[8]) * (b[5] - b[8]);
    c[1] = a[1] * b[1] + 0.5 * (a[3] + a[6]) * (b[3] + b[6]) + 0.5 * (a[4] - a[7]) * (b[4] - b[7]);
    c[2] = a[2] * b[2] + 0.5 * (a[4] + a[7]) * (b[4] + b[7]) + 0.5 * (a[5] + a[8]) * (b[5] + b[8]);
    c[3] = 0.5 * (a[0] * (b[3] + b[6]) + b[1] * (a[3] - a[6]) + b[0] * (a[3] + a[6]) + a[1] * (b[3] - b[6]) + ((a[5] - a[8]) * (b[4] - b[7]) + (a[4] - a[7]) * (b[5] - b[8])) / SQRT_2);
    c[4] = 0.5 * (a[1] * (b[4] + b[7]) + b[2] * (a[4] - a[7]) + b[1] * (a[4] + a[7]) + a[2] * (b[4] - b[7]) + ((a[3] + a[6]) * (b[5] + b[8]) + (a[5] + a[8]) * (b[3] + b[6])) / SQRT_2);
    c[5] = 0.5 * (a[0] * (b[5] + b[8]) + b[2] * (a[5] - a[8]) + b[0] * (a[5] + a[8]) + a[2] * (b[5] - b[8]) + ((a[3] - a[6]) * (b[4] + b[7]) + (a[4] + a[7]) * (b[3] - b[6])) / SQRT_2);
    if chop {
        assert_eq!(cc.rep(), Rep::Symmetric);
    } else {
        assert_eq!(cc.rep(), Rep::General);
        // TODO: compute the skew components c[6], c[7], c[8] of C = Aᵀ · B
    }
}

/// Performs the general tensor dot transpose(general tensor) operation: C = A · Bᵀ
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
/// When `chop` is `false`, the full (general) result is expected, but the
/// skew components `c[6]`, `c[7]`, `c[8]` are not implemented yet.
///
/// # Output
///
/// * `cc` -- the resulting tensor C; must be [Rep::Symmetric] if `chop` is
///   `true`, or [Rep::General] if `chop` is `false`
///
/// # Input
///
/// * `aa` -- the first tensor A; must be [Rep::General]
/// * `bb` -- the second tensor B (which gets transposed); must be [Rep::General]
/// * `chop` -- if `true`, chop off (discard) the skew components of C
///
/// # Panics
/// 
/// A panic will occur if `aa` and `bb` are not [Rep::General], or if `cc`
/// does not have the [Rep] required by `chop`.
#[rustfmt::skip]
pub fn t2_gen_dot_gen_tra_chop(cc: &mut Tensor2, aa: &Tensor2, bb: &Tensor2, chop: bool) {
    assert_eq!(aa.rep(), Rep::General);
    assert_eq!(bb.rep(), Rep::General);
    let a = &aa.vec;
    let b = &bb.vec;
    let c = &mut cc.vec;
    c[0] = a[0] * b[0] + 0.5 * (a[3] + a[6]) * (b[3] + b[6]) + 0.5 * (a[5] + a[8]) * (b[5] + b[8]);
    c[1] = a[1] * b[1] + 0.5 * (a[3] - a[6]) * (b[3] - b[6]) + 0.5 * (a[4] + a[7]) * (b[4] + b[7]);
    c[2] = a[2] * b[2] + 0.5 * (a[4] - a[7]) * (b[4] - b[7]) + 0.5 * (a[5] - a[8]) * (b[5] - b[8]);
    c[3] = 0.5 * (a[0] * (b[3] - b[6]) + b[1] * (a[3] + a[6]) + b[0] * (a[3] - a[6]) + a[1] * (b[3] + b[6]) + ((a[5] + a[8]) * (b[4] + b[7]) + (a[4] + a[7]) * (b[5] + b[8])) / SQRT_2);
    c[4] = 0.5 * (a[1] * (b[4] - b[7]) + b[2] * (a[4] + a[7]) + b[1] * (a[4] - a[7]) + a[2] * (b[4] + b[7]) + ((a[3] - a[6]) * (b[5] - b[8]) + (a[5] - a[8]) * (b[3] - b[6])) / SQRT_2);
    c[5] = 0.5 * (a[0] * (b[5] - b[8]) + b[2] * (a[5] + a[8]) + b[0] * (a[5] - a[8]) + a[2] * (b[5] + b[8]) + ((a[3] + a[6]) * (b[4] - b[7]) + (a[4] - a[7]) * (b[3] + b[6])) / SQRT_2);
    if chop {
        assert_eq!(cc.rep(), Rep::Symmetric);
    } else {
        assert_eq!(cc.rep(), Rep::General);
        // TODO: compute the skew components c[6], c[7], c[8] of C = A · Bᵀ
    }
}
