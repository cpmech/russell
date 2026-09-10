//! Numerically stable closed-form eigenvalues of 3×3 matrices
//!
//! This module is a Rust port of the reference implementation from the `eig3x3`
//! library by Michal Habera and Andreas Zilian.
//!
//! The key ideas are:
//!
//! 1. Compute the invariants `J2` and `J3` from diagonal differences and
//!    off-diagonal products, avoiding cancellation.
//! 2. Compute the discriminant as a sum of squares (symmetric case, 5 terms) or
//!    a weighted sum of products (general case, 14 terms), avoiding the
//!    catastrophic cancellation of `4 J2³ - 27 J3²`.
//! 3. Compute the triple angle with `atan2`, which is stable as `Δ → 0`.
//!
//! # References
//!
//! 1. Habera M. and Zilian A. (2025) Numerically stable evaluation of closed-form
//!    expressions for eigenvalues of 3×3 matrices. <https://arxiv.org/abs/2511.00292>
//! 2. Habera M. and Zilian A. (2021) Symbolic spectral decomposition of 3x3 matrices.
//!    <https://arxiv.org/abs/2111.02117>
//! 3. <https://github.com/michalhabera/eig3x3>

// Copyright (c) 2025 Michal Habera, Andreas Zilian
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#![allow(dead_code)] // these functions will be connected to Spectral2 later

/// Computes the second invariant J2 of a general 3×3 matrix
///
/// ```text
/// J2 = ½ [tr(A)² - tr(A²)]  (for the deviator, i.e. traceless part)
/// ```
///
/// This implementation uses diagonal differences and off-diagonal products to
/// avoid cancellation.
pub fn j2(a: &[[f64; 3]; 3]) -> f64 {
    let d0 = a[0][0] - a[1][1];
    let d1 = a[0][0] - a[2][2];
    let d2 = a[1][1] - a[2][2];
    let off_diag = a[0][1] * a[1][0] + a[0][2] * a[2][0] + a[1][2] * a[2][1];
    let diag = (d0 * d0 + d1 * d1 + d2 * d2) / 6.0;
    off_diag + diag
}

/// Computes the third invariant J3 of a general 3×3 matrix
///
/// This implementation uses diagonal differences, off-diagonal products, and
/// mixed products to avoid cancellation.
pub fn j3(a: &[[f64; 3]; 3]) -> f64 {
    let d0 = a[0][0] - a[1][1];
    let d1 = a[0][0] - a[2][2];
    let d2 = a[1][1] - a[2][2];
    let t1 = d1 + d2;
    let t2 = d0 - d2;
    let t3 = -d0 - d1;
    let off_diag = a[0][1] * a[1][2] * a[2][0] + a[0][2] * a[1][0] * a[2][1];
    let mixed = (a[0][1] * a[1][0] * t1 + a[0][2] * a[2][0] * t2 + a[1][2] * a[2][1] * t3) / 3.0;
    let diag = (t1 * t2 * t3) / 27.0;
    off_diag + mixed - diag
}

/// Computes the 14 discriminant terms of a general 3×3 matrix (internal)
fn dx(a: &[[f64; 3]; 3], result: &mut [f64; 14]) {
    let d0 = a[0][0] - a[1][1];
    let d1 = a[0][0] - a[2][2];
    let d2 = a[1][1] - a[2][2];
    result[0] = a[0][1] * a[1][2] * a[2][0] - a[0][2] * a[1][0] * a[2][1];
    result[1] = -a[0][1] * a[0][2] * d2 + a[0][1] * a[0][1] * a[1][2] - a[0][2] * a[0][2] * a[2][1];
    result[2] = a[0][1] * a[2][1] * d1 - a[0][1] * a[0][1] * a[2][0] + a[0][2] * a[2][1] * a[2][1];
    result[3] = a[0][2] * a[1][2] * d0 + a[0][1] * a[1][2] * a[1][2] - a[0][2] * a[0][2] * a[1][0];
    result[4] = a[0][1] * a[1][2] * d1 - a[0][1] * a[0][2] * a[1][0] + a[0][2] * a[1][2] * a[2][1];
    result[5] = a[0][2] * a[2][1] * d0 - a[0][1] * a[0][2] * a[2][0] + a[0][1] * a[1][2] * a[2][1];
    result[6] = -a[0][2] * a[1][0] * d2 + a[0][1] * a[1][0] * a[1][2] - a[0][2] * a[1][2] * a[2][0];
    result[7] = a[1][2] * d0 * d1 - a[0][2] * a[1][0] * d1 + a[0][1] * a[1][0] * a[1][2] - a[1][2] * a[1][2] * a[2][1];
    result[8] = a[1][2] * d0 * d1 - a[0][2] * a[1][0] * d0 + a[0][2] * a[1][2] * a[2][0] - a[1][2] * a[1][2] * a[2][1];
    result[9] = a[0][1] * d1 * d2 + a[0][2] * a[2][1] * d2 + a[0][1] * a[0][2] * a[2][0] - a[0][1] * a[0][1] * a[1][0];
    result[10] = a[0][1] * d1 * d2 + a[0][2] * a[2][1] * d1 + a[0][1] * a[1][2] * a[2][1] - a[0][1] * a[0][1] * a[1][0];
    result[11] =
        -a[0][2] * d0 * d2 + a[0][1] * a[1][2] * d0 + a[0][2] * a[1][2] * a[2][1] - a[0][2] * a[0][2] * a[2][0];
    result[12] = a[0][2] * d0 * d2 + a[0][1] * a[1][2] * d2 - a[0][1] * a[0][2] * a[1][0] + a[0][2] * a[0][2] * a[2][0];
    result[13] = d0 * d1 * d2 - a[0][1] * a[1][0] * d0 + a[0][2] * a[2][0] * d1 - a[1][2] * a[2][1] * d2;
}

/// Computes the discriminant of a general 3×3 matrix
///
/// The discriminant is computed as a weighted sum of products of the 14 terms
/// produced by [`dx`], which avoids catastrophic cancellation as the
/// discriminant approaches zero.
pub fn disc(a: &[[f64; 3]; 3]) -> f64 {
    let mut u = [0.0; 14];
    let mut v = [0.0; 14];
    dx(a, &mut u);
    let at = [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ];
    dx(&at, &mut v);
    const WEIGHTS: [f64; 14] = [9.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0];
    let mut sum = 0.0;
    for i in 0..14 {
        sum = (WEIGHTS[i] * u[i]).mul_add(v[i], sum);
    }
    sum
}

/// Computes the eigenvalues of a real, diagonalizable 3×3 matrix
///
/// The eigenvalues are returned in the order produced by the cubic formula
/// (i.e., not sorted). The input matrix need not be symmetric.
pub fn eigvals(a: &[[f64; 3]; 3]) -> [f64; 3] {
    let i1 = a[0][0] + a[1][1] + a[2][2];
    let j2 = j2(a);
    let j3 = j3(a);
    let discriminant = disc(a);
    let phi = f64::atan2(f64::sqrt(27.0 * discriminant), 27.0 * j3);
    let sqrt_3j2 = f64::sqrt(3.0 * j2);
    let two_pi = 2.0 * std::f64::consts::PI;
    let amplitude = 2.0 * sqrt_3j2;
    let mut eigenvalues = [0.0; 3];
    for k in 0..3 {
        let angle = (phi + two_pi * ((k + 1) as f64)) / 3.0;
        eigenvalues[k] = amplitude.mul_add(f64::cos(angle), i1) / 3.0;
    }
    eigenvalues
}

/// Computes the second invariant J2 of a symmetric 3×3 matrix
///
/// This is the optimized version for symmetric matrices.
pub fn j2s(a: &[[f64; 3]; 3]) -> f64 {
    let d0 = a[0][0] - a[1][1];
    let d1 = a[0][0] - a[2][2];
    let d2 = a[1][1] - a[2][2];
    let off_diag = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    let diag = (d0 * d0 + d1 * d1 + d2 * d2) / 6.0;
    off_diag + diag
}

/// Computes the third invariant J3 of a symmetric 3×3 matrix
///
/// This is the optimized version for symmetric matrices.
pub fn j3s(a: &[[f64; 3]; 3]) -> f64 {
    let d0 = a[0][0] - a[1][1];
    let d1 = a[0][0] - a[2][2];
    let d2 = a[1][1] - a[2][2];
    let t1 = d1 + d2;
    let t2 = d0 - d2;
    let t3 = -d0 - d1;
    let off_diag = 2.0 * a[0][1] * a[1][2] * a[0][2];
    let mixed = (a[0][1] * a[0][1] * t1 + a[0][2] * a[0][2] * t2 + a[1][2] * a[1][2] * t3) / 3.0;
    let diag = (t1 * t2 * t3) / 27.0;
    off_diag + mixed - diag
}

/// Computes the 5 discriminant terms of a symmetric 3×3 matrix (internal)
fn dxs(a: &[[f64; 3]; 3], result: &mut [f64; 5]) {
    let d0 = a[0][0] - a[1][1];
    let d1 = a[0][0] - a[2][2];
    let d2 = a[1][1] - a[2][2];
    let w = a[0][1];
    let v = a[0][2];
    let u = a[1][2];
    let alpha = d2;
    let beta = -d1;
    let gamma = d0;
    result[0] = 3.0 * f64::sqrt(3.0) * (v * w * alpha + u * (v * v - w * w));
    result[1] = alpha * beta * gamma + alpha * u * u + beta * v * v + gamma * w * w;
    result[2] = 2.0 * u * beta * gamma - v * w * (beta - gamma) + u * (2.0 * u * u - v * v - w * w);
    result[3] = 2.0 * (v * alpha * gamma + u * w * (beta - gamma) + v * (v * v + w * w - 2.0 * u * u));
    result[4] = 2.0 * (w * alpha * beta + u * v * (beta - gamma) + w * (v * v + w * w - 2.0 * u * u));
}

/// Computes the discriminant of a symmetric 3×3 matrix
///
/// The discriminant is computed as a sum of squares of the 5 terms produced by
/// [`dxs`], which avoids catastrophic cancellation as the discriminant
/// approaches zero.
pub fn discs(a: &[[f64; 3]; 3]) -> f64 {
    let mut u = [0.0; 5];
    dxs(a, &mut u);
    let mut sum = 0.0;
    for i in 0..5 {
        sum = u[i].mul_add(u[i], sum);
    }
    sum
}

/// Computes the eigenvalues of a real, symmetric 3×3 matrix
///
/// The eigenvalues are returned in the order produced by the cubic formula
/// (i.e., not sorted).
pub fn eigvalss(a: &[[f64; 3]; 3]) -> [f64; 3] {
    let i1 = a[0][0] + a[1][1] + a[2][2];
    let j2 = j2s(a);
    let j3 = j3s(a);
    let discriminant = discs(a);
    let phi = f64::atan2(f64::sqrt(27.0 * discriminant), 27.0 * j3);
    let sqrt_3j2 = f64::sqrt(3.0 * j2);
    let two_pi = 2.0 * std::f64::consts::PI;
    let amplitude = 2.0 * sqrt_3j2;
    let mut eigenvalues = [0.0; 3];
    for k in 0..3 {
        let angle = (phi + two_pi * ((k + 1) as f64)) / 3.0;
        eigenvalues[k] = amplitude.mul_add(f64::cos(angle), i1) / 3.0;
    }
    eigenvalues
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{disc, discs, eigvals, eigvalss, j2, j2s, j3, j3s};
    use russell_lab::approx_eq;

    #[test]
    fn invariants_match_explicit_formulas() {
        #[rustfmt::skip]
        let a = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
        ];
        // general invariants
        let i1 = a[0][0] + a[1][1] + a[2][2];
        let i2 = 0.5 * (i1 * i1 - trace2(&a));
        let i3 = det(&a);
        // J2 = ½ [tr(A)² - tr(A²)] + ... for the deviator:
        // J2(A) = ½ tr(A²) - (1/6) tr(A)²  (for the deviator)
        // Here we use the definition of the port: J2 = offdiag + diag, which equals
        // ½ (tr(A²) - tr(A)²/3) = ½ tr(A²) - I1²/6
        approx_eq(j2(&a), 0.5 * trace2(&a) - i1 * i1 / 6.0, 1e-13);
        // J3 for the deviator: det(A) - (I1/3) I2 + (2/27) I1³ ... let's just check
        // against the characteristic polynomial identity J3 = (1/3) tr(S³)
        let s = deviator(&a, i1);
        approx_eq(j3(&a), trace3(&s) / 3.0, 1e-13);
        let _ = (i2, i3);
    }

    #[test]
    fn symmetric_invariants_match_general() {
        #[rustfmt::skip]
        let a = [
            [2.0, -3.0, 4.0],
            [-3.0, -5.0, 1.0],
            [4.0, 1.0, 6.0],
        ];
        approx_eq(j2s(&a), j2(&a), 1e-14);
        approx_eq(j3s(&a), j3(&a), 1e-14);
        approx_eq(discs(&a), disc(&a), 1e-13);
    }

    #[test]
    fn eigvals_are_correct() {
        // matrix with known eigenvalues (the TFEL test tensor)
        let r2 = std::f64::consts::SQRT_2;
        #[rustfmt::skip]
        let a = [
            [1.232,        1.5634 / r2, 3.3425 / r2],
            [1.5634 / r2,  2.5198,      0.9765 / r2],
            [3.3425 / r2,  0.9765 / r2, 0.234      ],
        ];
        let mut w = eigvalss(&a);
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let correct = [-1.68923153093191, 1.50793773158270, 4.16709379934921];
        for i in 0..3 {
            approx_eq(w[i], correct[i], 1e-12);
        }
    }

    #[test]
    fn eigvals_symmetric_matches_lapack_example() {
        // simple symmetric matrix with known eigenvalues
        #[rustfmt::skip]
        let a = [
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let mut w = eigvalss(&a);
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        approx_eq(w[0], 1.0, 1e-14);
        approx_eq(w[1], 3.0, 1e-14);
        approx_eq(w[2], 3.0, 1e-14);
    }

    #[test]
    fn general_eigvals_match_symmetric_for_symmetric_input() {
        #[rustfmt::skip]
        let a = [
            [4.0, 1.0, 2.0],
            [1.0, 5.0, 3.0],
            [2.0, 3.0, 6.0],
        ];
        let mut w1 = eigvals(&a);
        let mut w2 = eigvalss(&a);
        w1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        w2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..3 {
            approx_eq(w1[i], w2[i], 1e-13);
        }
    }

    fn trace2(a: &[[f64; 3]; 3]) -> f64 {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                s += a[i][j] * a[j][i];
            }
        }
        s
    }

    fn trace3(a: &[[f64; 3]; 3]) -> f64 {
        // tr(A³) = sum_ijk A_ij A_jk A_ki
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    s += a[i][j] * a[j][k] * a[k][i];
                }
            }
        }
        s
    }

    fn deviator(a: &[[f64; 3]; 3], i1: f64) -> [[f64; 3]; 3] {
        let m = i1 / 3.0;
        let mut s = *a;
        for i in 0..3 {
            s[i][i] -= m;
        }
        s
    }

    fn det(a: &[[f64; 3]; 3]) -> f64 {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    }
}
