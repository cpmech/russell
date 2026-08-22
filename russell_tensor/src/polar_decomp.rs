use super::{Tensor2, t2_left_stretch, t2_right_stretch};
use crate::{Rep, SQRT_2};
use russell_lab::StrError;

const BRANNON_TOL: f64 = 1e-15;
const BRANNON_MAX_NIT: usize = 20;

/// Performs the polar decomposition F = R U = V R
///
/// Note: This is the only function where the output arguments are not the first parameters of the function.
///
/// # Arguments
///
/// * `ff` -- (in) F: the deformation gradient; must be [Rep::General]
/// * `rr` -- (out) R: the rotation tensor; must be [Rep::General]
/// * `uu` -- (out) U: the right stretch tensor; must be [Rep::Symmetric]
/// * `vv` -- (out) V: the left stretch tensor; must be [Rep::Symmetric] -- Optional
///
/// Returns the number of iterations taken for the rotation tensor to converge
///
/// # Panics
///
/// A panic will occur if the required [Rep] enums are incorrect.
pub fn polar_decomp(
    ff: &Tensor2,
    rr: &mut Tensor2,
    uu: &mut Tensor2,
    vv: Option<&mut Tensor2>,
) -> Result<usize, StrError> {
    assert_eq!(ff.rep(), Rep::General);
    assert_eq!(rr.rep(), Rep::General);
    assert_eq!(uu.rep(), Rep::Symmetric);
    let nit = polar_rotation(rr, ff)?;
    t2_right_stretch(uu, rr, ff);
    if let Some(v) = vv {
        assert_eq!(v.rep(), Rep::Symmetric);
        t2_left_stretch(v, ff, rr);
    }
    Ok(nit)
}

/// Computes the polar rotation tensor R of a general tensor F
///
/// Uses the iterative fixed-point algorithm by Rebecca Brannon.
///
/// # Arguments
///
/// * `rr` -- (out) R: the rotation tensor; must be [Rep::General]
/// * `ff` -- (in) F: the deformation gradient; must be [Rep::General]
///
/// # Returns
///
/// Returns the number of iterations taken for convergence.
///
/// # Panics
///
/// A panic will occur if the required [Rep] enums are incorrect.
pub fn polar_rotation(rr: &mut Tensor2, ff: &Tensor2) -> Result<usize, StrError> {
    assert_eq!(ff.rep(), Rep::General);
    assert_eq!(rr.rep(), Rep::General);

    let mut e = [0.0; 9];
    let mut a = [0.0; 9];
    let mut x = [0.0; 9];
    let mut i_vec_minus_e = [0.0; 9];

    // Step 1: E = F^T F
    t2_tra_dot_t2_mix(&mut e, ff);

    // Step 2: Scale F to guarantee convergence
    let mut s = 3.0 / (e[0] + e[1] + e[2]);
    for i in 0..9 {
        if i < 3 {
            e[i] = 0.5 * (s * e[i] - 1.0);
        } else {
            e[i] = 0.5 * (s * e[i]);
        }
    }

    // Step 3: First guess A = sqrt(s) F
    s = f64::sqrt(s);
    for i in 0..9 {
        a[i] = s * ff.get(i);
    }

    // Step 4: Initial error using vector dot product
    let mut errz = 0.0;
    for i in 0..9 {
        errz += e[i] * e[i];
    }

    let mut knt = 0;
    while errz > BRANNON_TOL && knt < BRANNON_MAX_NIT {
        // Step 6: X = A(I - E)
        for i in 0..9 {
            if i < 3 {
                i_vec_minus_e[i] = 1.0 - e[i];
            } else {
                i_vec_minus_e[i] = -e[i];
            }
        }
        t2_gen_dot_sym_stack(&mut x, &a, &i_vec_minus_e);
        a.copy_from_slice(&x);

        // Step 7: E = 1/2(A^T A - I)
        t2_tra_dot_t2_stack(&mut e, &a);
        for i in 0..9 {
            if i < 3 {
                e[i] = 0.5 * (e[i] - 1.0);
            } else {
                e[i] = 0.5 * e[i];
            }
        }

        // Step 8: New error
        let mut err = 0.0;
        for i in 0..9 {
            err += e[i] * e[i];
        }

        // Step 9: Convergence check
        if err >= errz {
            break;
        }
        errz = err;
        knt += 1;
    }

    if knt == BRANNON_MAX_NIT {
        return Err("polar_rotation did not converge");
    }

    // Step 10: Copy rotation vector to output
    for i in 0..9 {
        rr.vec[i] = a[i];
    }
    Ok(knt)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PRIVATE HELPER FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Calculates the symmetric product: C = A^T A (using stack and Tensor2)
/// Enforces skew components (c[6], c[7], c[8]) to 0 to prevent floating point noise.
#[inline]
#[rustfmt::skip]
fn t2_tra_dot_t2_mix(c: &mut [f64], aa: &Tensor2) {
    let a = &aa.vec;
    c[0] = 0.5 * (2.0 * a[0] * a[0] + (a[3] - a[6]) * (a[3] - a[6]) + (a[5] - a[8]) * (a[5] - a[8]));
    c[1] = 0.5 * (2.0 * a[1] * a[1] + (a[3] + a[6]) * (a[3] + a[6]) + (a[4] - a[7]) * (a[4] - a[7]));
    c[2] = 0.5 * (2.0 * a[2] * a[2] + (a[4] + a[7]) * (a[4] + a[7]) + (a[5] + a[8]) * (a[5] + a[8]));
    c[3] = a[1] * (a[3] - a[6]) + a[0] * (a[3] + a[6]) + ((a[4] - a[7]) * (a[5] - a[8])) / SQRT_2;
    c[4] = a[2] * (a[4] - a[7]) + a[1] * (a[4] + a[7]) + ((a[3] + a[6]) * (a[5] + a[8])) / SQRT_2;
    c[5] = ((a[3] - a[6]) * (a[4] + a[7])) / SQRT_2 + a[2] * (a[5] - a[8]) + a[0] * (a[5] + a[8]);
    c[6] = 0.0;
    c[7] = 0.0;
    c[8] = 0.0;
}

/// Calculates the symmetric product: C = A^T A (using stack only)
/// Enforces skew components (c[6], c[7], c[8]) to 0 to prevent floating point noise.
#[inline]
#[rustfmt::skip]
fn t2_tra_dot_t2_stack(c: &mut [f64], a: &[f64]) {
    c[0] = 0.5 * (2.0 * a[0] * a[0] + (a[3] - a[6]) * (a[3] - a[6]) + (a[5] - a[8]) * (a[5] - a[8]));
    c[1] = 0.5 * (2.0 * a[1] * a[1] + (a[3] + a[6]) * (a[3] + a[6]) + (a[4] - a[7]) * (a[4] - a[7]));
    c[2] = 0.5 * (2.0 * a[2] * a[2] + (a[4] + a[7]) * (a[4] + a[7]) + (a[5] + a[8]) * (a[5] + a[8]));
    c[3] = a[1] * (a[3] - a[6]) + a[0] * (a[3] + a[6]) + ((a[4] - a[7]) * (a[5] - a[8])) / SQRT_2;
    c[4] = a[2] * (a[4] - a[7]) + a[1] * (a[4] + a[7]) + ((a[3] + a[6]) * (a[5] + a[8])) / SQRT_2;
    c[5] = ((a[3] - a[6]) * (a[4] + a[7])) / SQRT_2 + a[2] * (a[5] - a[8]) + a[0] * (a[5] + a[8]);
    c[6] = 0.0;
    c[7] = 0.0;
    c[8] = 0.0;
}

/// Calculates the mixed product: C = A B
/// Assumes B is fully symmetric (components b[6], b[7], b[8] are zero).
#[inline]
#[rustfmt::skip]
fn t2_gen_dot_sym_stack(c: &mut [f64], a: &[f64], b: &[f64]) {
    c[0] = 0.5 * (2.0 * a[0] * b[0] + (a[3] + a[6]) * b[3] + (a[5] + a[8]) * b[5]);
    c[1] = 0.5 * (2.0 * a[1] * b[1] + (a[3] - a[6]) * b[3] + (a[4] + a[7]) * b[4]);
    c[2] = 0.5 * (2.0 * a[2] * b[2] + (a[4] - a[7]) * b[4] + (a[5] - a[8]) * b[5]);
    c[3] = (SQRT_2 * a[6] * (b[1] - b[0]) + SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[8] * b[4] + a[4] * b[5] + a[7] * b[5]) / (2.0 * SQRT_2);
    c[4] = (SQRT_2 * a[7] * (b[2] - b[1]) + SQRT_2 * a[4] * (b[1] + b[2]) - a[5] * b[3] + a[8] * b[3] + SQRT_2 * a[1] * b[4] + SQRT_2 * a[2] * b[4] + a[3] * b[5] - a[6] * b[5]) / (2.0 * SQRT_2);
    c[5] = (SQRT_2 * a[8] * (b[2] - b[0]) + SQRT_2 * a[5] * (b[0] + b[2]) - a[4] * b[3] + a[7] * b[3] + a[3] * b[4] + a[6] * b[4] + SQRT_2 * a[0] * b[5] + SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2);
    c[6] = (SQRT_2 * a[3] * (b[1] - b[0]) + SQRT_2 * a[6] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] - SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[8] * b[4] - a[4] * b[5] - a[7] * b[5]) / (2.0 * SQRT_2);
    c[7] = (SQRT_2 * a[4] * (b[2] - b[1]) + SQRT_2 * a[7] * (b[1] + b[2]) + a[5] * b[3] - a[8] * b[3] + SQRT_2 * a[1] * b[4] - SQRT_2 * a[2] * b[4] + a[3] * b[5] - a[6] * b[5]) / (2.0 * SQRT_2);
    c[8] = (SQRT_2 * a[5] * (b[2] - b[0]) + SQRT_2 * a[8] * (b[0] + b[2]) + a[4] * b[3] - a[7] * b[3] + a[3] * b[4] + a[6] * b[4] + SQRT_2 * a[0] * b[5] - SQRT_2 * a[2] * b[5]) / (2.0 * SQRT_2);
}
