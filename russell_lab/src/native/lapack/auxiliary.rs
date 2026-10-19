//! LAPACK's auxiliary routines, each in one place.
//!
//! LAPACK keeps these in `SRC/` as one file per routine (`dlapy2.f`, `dlaev2.f`,
//! `zlarf.f`, ...) and every other routine calls them. The port started out with
//! a copy in each module that needed one -- `dlapy2` existed in three, `dznrm2` in
//! four -- which is a silent-divergence trap: a fix applied to one copy leaves the
//! others wrong. They live here instead, and the modules that need them import
//! them.
//!
//! `dznrm2` is deliberately not here: it is BLAS level 1, so it lives in
//! [`super::blas`] with the other vector norms.

use crate::Complex64;
use num_complex::ComplexFloat;

/// LAPACK's `SAFMIN` (`DLAMCH('S')`): the smallest positive normal number.
const SAFMIN: f64 = f64::MIN_POSITIVE;

/// The zero of the complex type, so the complex routines read like the reference.
pub(super) fn zero() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

/// `dlapy2`: `sqrt(x^2 + y^2)` without spurious overflow.
pub(super) fn dlapy2(x: f64, y: f64) -> f64 {
    let xabs = x.abs();
    let yabs = y.abs();
    let w = xabs.max(yabs);
    let z = xabs.min(yabs);
    if z == 0.0 {
        w
    } else {
        w * (1.0 + (z / w) * (z / w)).sqrt()
    }
}

/// `dlapy3`: `sqrt(x^2 + y^2 + z^2)` without spurious overflow.
pub(super) fn dlapy3(x: f64, y: f64, z: f64) -> f64 {
    let xabs = x.abs();
    let yabs = y.abs();
    let zabs = z.abs();
    let w = xabs.max(yabs).max(zabs);
    if w == 0.0 || w > f64::MAX || !w.is_finite() {
        xabs + yabs + zabs
    } else {
        w * ((xabs / w) * (xabs / w) + (yabs / w) * (yabs / w) + (zabs / w) * (zabs / w)).sqrt()
    }
}

/// `dlae2`: the eigenvalues of the symmetric 2x2 block `[[a, b], [b, c]]`.
pub(super) fn dlae2(a: f64, b: f64, c: f64) -> (f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        adf * (1.0 + (ab / adf) * (ab / adf)).sqrt()
    } else if adf < ab {
        ab * (1.0 + (adf / ab) * (adf / ab)).sqrt()
    } else {
        ab * 2.0f64.sqrt()
    };
    if sm < 0.0 {
        let rt1 = 0.5 * (sm - rt);
        // The order matters: the smaller eigenvalue is fully accurate only when
        // computed from the larger one this way.
        let rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
        (rt1, rt2)
    } else if sm > 0.0 {
        let rt1 = 0.5 * (sm + rt);
        let rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
        (rt1, rt2)
    } else {
        (0.5 * rt, -0.5 * rt)
    }
}

/// `dlaev2`: the eigenvalues and the eigenvector rotation of the symmetric 2x2
/// block `[[a, b], [b, c]]`.
///
/// Returns `(rt1, rt2, cs1, sn1)`; the eigenvector is `(cs1, sn1)` and the signs
/// are chosen exactly as the reference chooses them, which is what pins the
/// eigenvector signs of the whole decomposition.
pub(super) fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        adf * (1.0 + (ab / adf) * (ab / adf)).sqrt()
    } else if adf < ab {
        ab * (1.0 + (adf / ab) * (adf / ab)).sqrt()
    } else {
        ab * 2.0f64.sqrt()
    };

    let (rt1, rt2, sgn1);
    if sm < 0.0 {
        rt1 = 0.5 * (sm - rt);
        sgn1 = -1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if sm > 0.0 {
        rt1 = 0.5 * (sm + rt);
        sgn1 = 1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        rt1 = 0.5 * rt;
        rt2 = -0.5 * rt;
        sgn1 = 1;
    }

    let (cs, sgn2) = if df >= 0.0 { (df + rt, 1) } else { (df - rt, -1) };
    let acs = cs.abs();
    let (mut cs1, mut sn1) = if acs > ab {
        let ct = -tb / cs;
        let sn1 = 1.0 / (1.0 + ct * ct).sqrt();
        (ct * sn1, sn1)
    } else if ab == 0.0 {
        (1.0, 0.0)
    } else {
        let tn = -cs / tb;
        let cs1 = 1.0 / (1.0 + tn * tn).sqrt();
        (cs1, tn * cs1)
    };
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
}

pub(super) fn dladiv2(a: f64, b: f64, c: f64, d: f64, r: f64, t: f64) -> f64 {
    if r != 0.0 {
        let br = b * r;
        if br != 0.0 { (a + br) * t } else { a * t + (b * t) * r }
    } else {
        (a + d * (b / c)) * t
    }
}

/// `zladiv`: the complex quotient `x / y`, avoiding overflow in `y`'s products.
pub(super) fn zladiv(x: Complex64, y: Complex64) -> Complex64 {
    if y.re.abs() >= y.im.abs() {
        let r = y.im / y.re;
        let den = y.re + r * y.im;
        Complex64::new((x.re + r * x.im) / den, (x.im - r * x.re) / den)
    } else {
        let r = y.re / y.im;
        let den = y.im + r * y.re;
        Complex64::new((x.re * r + x.im) / den, (x.im * r - x.re) / den)
    }
}

/// `DLASCL`/`ZLASCL` with `TYPE = 'G'`: `a := (cto / cfrom) * a`, without
/// overflowing. Generic over the element type; `dlascl_g` and [`zlascl_g`] are
/// the named wrappers.
fn lascl_g<T: Copy + std::ops::Mul<f64, Output = T>>(
    cfrom: f64,
    cto: f64,
    a: &mut [T],
    lda: usize,
    m: usize,
    n: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let smlnum = SAFMIN;
    let bignum = 1.0 / smlnum;
    let mut cfromc = cfrom;
    let mut ctoc = cto;
    loop {
        let cfrom1 = cfromc * smlnum;
        let mul;
        let done;
        if cfrom1 == cfromc {
            // `cfromc * smlnum` underflowed, so the ratio is safe to form.
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                // `ctoc / bignum` underflowed: scale by `ctoc` alone.
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == 1.0 {
                    return;
                }
            }
        }
        for j in 0..n {
            for i in 0..m {
                a[i + j * lda] = a[i + j * lda] * mul;
            }
        }
        if done {
            return;
        }
    }
}

/// `DLASCL` with `TYPE = 'G'`: scale a real matrix by `cto / cfrom`.
pub(crate) fn dlascl_g(cfrom: f64, cto: f64, a: &mut [f64], lda: usize, m: usize, n: usize) {
    lascl_g(cfrom, cto, a, lda, m, n)
}

/// `ZLASCL` with `TYPE = 'G'`: scale a complex matrix by `cto / cfrom`.
///
/// LAPACK has one routine per type; the arithmetic is the same, so here it is one
/// generic and two named wrappers, which keeps both reference names visible.
pub(super) fn zlascl_g(cfrom: f64, cto: f64, a: &mut [Complex64], lda: usize, m: usize, n: usize) {
    lascl_g(cfrom, cto, a, lda, m, n)
}

/// `ZLARF`, left: `C := (I - tau v v^H) C`, with `v[0] == 1` supplied by the caller.
///
/// `C` is the `m x ncols` block of `a` at `(r0, c0)` with leading dimension
/// `lda`; `v` has length `m`.
pub(super) fn zlarf_left(
    a: &mut [Complex64],
    lda: usize,
    r0: usize,
    m: usize,
    c0: usize,
    ncols: usize,
    v: &[Complex64],
    tau: Complex64,
) {
    if tau == zero() || m == 0 || ncols == 0 {
        return;
    }
    let mut w = vec![zero(); ncols];
    for (j, wj) in w.iter_mut().enumerate() {
        let mut s = zero();
        for i in 0..m {
            s += a[(r0 + i) + (c0 + j) * lda] * v[i].conj();
        }
        *wj = s;
    }
    for (j, &wj) in w.iter().enumerate() {
        for i in 0..m {
            a[(r0 + i) + (c0 + j) * lda] -= tau * v[i] * wj;
        }
    }
}

/// `ZLARF` with `SIDE = 'R'`: `C := C - tau * (C v) * v^H`.
///
/// `c` holds the `m x ncols` block whose `(r0, c0)` entry is its top-left corner.
/// `v` has length `ncols`.
pub(super) fn zlarf_right(
    a: &mut [Complex64],
    lda: usize,
    r0: usize,
    m: usize,
    c0: usize,
    ncols: usize,
    v: &[Complex64],
    tau: Complex64,
) {
    if tau == zero() || m == 0 || ncols == 0 {
        return;
    }
    let mut w = vec![zero(); m];
    for (i, wi) in w.iter_mut().enumerate() {
        let mut s = zero();
        for j in 0..ncols {
            s += a[(r0 + i) + (c0 + j) * lda] * v[j];
        }
        *wi = s;
    }
    for (i, &wi) in w.iter().enumerate() {
        for j in 0..ncols {
            a[(r0 + i) + (c0 + j) * lda] -= tau * wi * v[j].conj();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmplx(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// `dlapy2` is `hypot`: exact on a Pythagorean triple, and it must not
    /// overflow where squaring would.
    #[test]
    fn dlapy2_survives_the_squares() {
        assert_eq!(dlapy2(3.0, 4.0), 5.0);
        assert_eq!(dlapy2(-3.0, -4.0), 5.0);
        assert_eq!(dlapy2(5.0, 0.0), 5.0);
        assert_eq!(dlapy2(0.0, -5.0), 5.0);
        assert_eq!(dlapy2(0.0, 0.0), 0.0);
        // Squaring 1e200 overflows f64; the scaled form does not.
        let big = dlapy2(1e200, 1e200);
        assert!(big.is_finite(), "dlapy2(1e200, 1e200) = {big}");
        assert!((big / (1e200 * 2.0f64.sqrt()) - 1.0).abs() < 1e-15);
        // Squaring 1e-200 underflows to zero; the scaled form keeps the value.
        let small = dlapy2(1e-200, 1e-200);
        assert!(small > 0.0, "dlapy2(1e-200, 1e-200) underflowed");
        assert!((small / (1e-200 * 2.0f64.sqrt()) - 1.0).abs() < 1e-15);
    }

    /// `dlapy3`: the same argument for three components.
    #[test]
    fn dlapy3_survives_the_squares() {
        assert_eq!(dlapy3(1.0, 2.0, 2.0), 3.0);
        assert_eq!(dlapy3(0.0, 0.0, 0.0), 0.0);
        assert_eq!(dlapy3(-3.0, 4.0, 0.0), 5.0);
        let big = dlapy3(1e200, 1e200, 1e200);
        assert!(big.is_finite(), "dlapy3(1e200, ...) = {big}");
        assert!((big / (1e200 * 3.0f64.sqrt()) - 1.0).abs() < 1e-15);
        let small = dlapy3(1e-200, 1e-200, 1e-200);
        assert!(small > 0.0);
    }

    /// `dlae2`: the eigenvalues of a symmetric 2x2 must satisfy the trace, the
    /// determinant and their own characteristic polynomial, with the larger one
    /// first.
    #[test]
    fn dlae2_gives_the_eigenvalues_of_a_symmetric_block() {
        let cases = [
            (2.0, 1.0, 2.0),
            (1.0, 0.0, 3.0),
            (0.0, 1.0, 0.0),   // (1, -1)
            (-2.0, 0.5, -5.0), // both negative
            (4.0, -3.0, 4.0),  // b < 0
            (1.0, 0.0, 1.0),   // repeated, b == 0
        ];
        for (a, b, c) in cases {
            let (rt1, rt2) = dlae2(a, b, c);
            let scale = a.abs().max(b.abs()).max(c.abs()).max(1.0);
            // LAPACK's `dlae2` returns the eigenvalue of *larger absolute value*
            // first, which is not the same as the larger one for a negative
            // block: (-2, 0.5, -5) gives rt1 = -5.08, rt2 = -1.92.
            assert!(
                rt1.abs() >= rt2.abs() - 1e-14 * scale,
                "({a},{b},{c}): |rt1| = {} < |rt2| = {}, but rt1 is the larger in modulus",
                rt1.abs(),
                rt2.abs()
            );
            assert!(
                ((rt1 + rt2) - (a + c)).abs() <= 1e-14 * scale,
                "({a},{b},{c}): trace {} != {}",
                rt1 + rt2,
                a + c
            );
            assert!(
                (rt1 * rt2 - (a * c - b * b)).abs() <= 1e-13 * scale * scale,
                "({a},{b},{c}): determinant {} != {}",
                rt1 * rt2,
                a * c - b * b
            );
            for rt in [rt1, rt2] {
                // (a - rt)(c - rt) - b^2 = 0
                let p = (a - rt) * (c - rt) - b * b;
                assert!(
                    p.abs() <= 1e-13 * scale * scale,
                    "({a},{b},{c}): rt = {rt} is not a root ({p})"
                );
            }
        }
    }

    /// `dlaev2`: the returned rotation must diagonalize the block, with the
    /// returned eigenvalues on the diagonal, and it must be a rotation.
    #[test]
    fn dlaev2_returns_a_diagonalising_rotation() {
        let cases = [
            (2.0, 1.0, 2.0),
            (1.0, 0.0, 3.0),
            (0.0, 1.0, 0.0),
            (-2.0, 0.5, -5.0),
            (4.0, -3.0, 4.0),
            (1.0, 2.0, 1.0), // b dominant: the `ab == 0` and `acs > ab` branches
            (0.0, 0.0, 0.0),
        ];
        for (a, b, c) in cases {
            let (rt1, rt2, cs, sn) = dlaev2(a, b, c);
            let scale = a.abs().max(b.abs()).max(c.abs()).max(1.0);
            assert!(
                (cs * cs + sn * sn - 1.0).abs() <= 1e-14,
                "({a},{b},{c}): rotation is not unit ({cs},{sn})"
            );
            // R^T A R with R = [[cs, -sn], [sn, cs]] must be diag(rt1, rt2).
            let r11 = cs;
            let r21 = sn;
            let r12 = -sn;
            let r22 = cs;
            let d11 = r11 * (a * r11 + b * r21) + r21 * (b * r11 + c * r21);
            let d22 = r12 * (a * r12 + b * r22) + r22 * (b * r12 + c * r22);
            let d12 = r11 * (a * r12 + b * r22) + r21 * (b * r12 + c * r22);
            assert!(
                (d11 - rt1).abs() <= 1e-13 * scale,
                "({a},{b},{c}): R^T A R (1,1) = {d11} != rt1 = {rt1}"
            );
            assert!(
                (d22 - rt2).abs() <= 1e-13 * scale,
                "({a},{b},{c}): R^T A R (2,2) = {d22} != rt2 = {rt2}"
            );
            assert!(
                d12.abs() <= 1e-13 * scale,
                "({a},{b},{c}): R^T A R (1,2) = {d12} is not zero"
            );
        }
    }

    /// `dladiv2` computes the real part of `(a + i b) / (c + i d)` given
    /// `r = d/c` and `t = 1/(c + d r)`; all three internal branches must agree
    /// with the closed form.
    #[test]
    fn dladiv2_covers_its_three_branches() {
        // (a + ib)/(c + id) = ((ac + bd) + i(bc - ad)) / (c^2 + d^2)
        let cases = [
            (1.0, 2.0, 3.0, 4.0), // r != 0, b*r != 0
            (1.0, 0.0, 3.0, 4.0), // b == 0: the `br == 0` branch
            (2.0, 5.0, 7.0, 0.0), // d == 0: the `r == 0` branch
            (-1.5, 0.25, 2.0, -8.0),
        ];
        for (a, b, c, d) in cases {
            let (r, t) = if d == 0.0 {
                (0.0, 1.0 / c)
            } else {
                let r = d / c;
                (r, 1.0 / (c + d * r))
            };
            let p = dladiv2(a, b, c, d, r, t);
            let want = (a * c + b * d) / (c * c + d * d);
            let scale = want.abs().max(1.0);
            assert!(
                (p - want).abs() <= 1e-13 * scale,
                "({a},{b},{c},{d}): p = {p} != {want}"
            );
        }
    }

    /// `zladiv`: exact on a small quotient, and finite where the naive formula
    /// overflows in `y`'s squares.
    #[test]
    fn zladiv_survives_a_huge_divisor() {
        let q = zladiv(cmplx(1.0, 2.0), cmplx(3.0, 4.0));
        assert!((q.re - 0.44).abs() < 1e-15, "re = {}", q.re);
        assert!((q.im - 0.08).abs() < 1e-15, "im = {}", q.im);

        // |re| < |im| takes the second branch.
        let q = zladiv(cmplx(2.0, 1.0), cmplx(4.0, 3.0));
        assert!((q.re - 0.44).abs() < 1e-15, "re = {}", q.re);
        assert!((q.im + 0.08).abs() < 1e-15, "im = {}", q.im);

        // (1e300 + 0i) / (1e300 + 1e300 i) = 0.5 - 0.5 i; the naive formula
        // squares 1e300 and gives NaN.
        let q = zladiv(cmplx(1e300, 0.0), cmplx(1e300, 1e300));
        assert!(q.re.is_finite() && q.im.is_finite(), "q = {q:?}");
        assert!((q.re - 0.5).abs() < 1e-14, "re = {}", q.re);
        assert!((q.im + 0.5).abs() < 1e-14, "im = {}", q.im);
    }

    /// `lascl_g` scales by `cto/cfrom` exactly for representable ratios, and the
    /// named wrappers are the real and complex spellings of the same thing.
    #[test]
    fn lascl_g_scales_by_the_ratio() {
        for (cfrom, cto) in [(1.0, 2.0), (2.0, 1.0), (1.0, 1.0), (1e-5, 1.0), (1.0, 1e5)] {
            let mut a = vec![1.0, -2.0, 3.0, -4.0];
            dlascl_g(cfrom, cto, &mut a, 2, 2, 2);
            let ratio = cto / cfrom;
            for (k, v) in a.iter().enumerate() {
                let want = [1.0, -2.0, 3.0, -4.0][k] * ratio;
                assert!(
                    (v - want).abs() <= 1e-13 * want.abs().max(1.0),
                    "cfrom = {cfrom}, cto = {cto}: a[{k}] = {v} != {want}"
                );
            }
        }
        // A zero-length block is a no-op, as in the reference.
        let mut empty: Vec<f64> = vec![];
        dlascl_g(1.0, 2.0, &mut empty, 0, 0, 0);
        // The complex wrapper scales both parts.
        let mut z = vec![cmplx(1.0, -1.0), cmplx(2.0, 0.0)];
        zlascl_g(1.0, 4.0, &mut z, 2, 2, 1);
        assert_eq!(z[0], cmplx(4.0, -4.0));
        assert_eq!(z[1], cmplx(8.0, 0.0));
    }

    /// `lascl_g` must survive ratios that are not representable, by scaling in
    /// steps: the loop exists so that `a := (cto/cfrom) a` never forms an
    /// overflowing or vanishing intermediate. The bench above only exercises
    /// benign ratios, where the first `ctoc / cfromc` is already the answer.
    ///
    /// The expected value is formed as `(a cto) / cfrom`, an order that stays in
    /// range for these inputs, with `a` chosen so the result is representable.
    #[test]
    fn lascl_g_survives_ratios_that_overflow_or_underflow() {
        // (a, cfrom, cto, expected) -- expected = (a * cto) / cfrom
        let cases = [
            // cto/cfrom = 1e600 overflows: the loop must reach it by scaling up in
            // steps (mul = bignum), not by forming the ratio
            (1e-300, 1e-300, 1e300, 1e300),
            // cto/cfrom = 1e-600 underflows: scale down in steps (mul = smlnum)
            (1e300, 1e300, 1e-300, 1e-300),
            // cto = 0 cannot be reached by scaling down: the loop takes the
            // ctoc-underflowed branch and multiplies by ctoc itself
            (5.0, 1.0, 0.0, 0.0),
            // both factors large, ratio 1: no scaling should be applied at all
            (1e-100, 1e300, 1e300, 1e-100),
        ];
        for (a_in, cfrom, cto, expected) in cases {
            let mut a = vec![a_in, -a_in];
            dlascl_g(cfrom, cto, &mut a, 1, 1, 2); // 1x2 block, column-major with lda = 1
            assert!(
                a[0].is_finite(),
                "cfrom = {cfrom}, cto = {cto}: a[0] = {} is not finite",
                a[0]
            );
            for (k, v) in a.iter().enumerate() {
                let sign = if k == 0 { 1.0 } else { -1.0 };
                let want = sign * expected;
                assert!(
                    (v - want).abs() <= 1e-13 * want.abs().max(f64::MIN_POSITIVE),
                    "cfrom = {cfrom}, cto = {cto}: a[{k}] = {v} != {want}"
                );
            }
        }
    }

    /// `zlarf_left` must equal `C := (I - tau v v^H) C` and `zlarf_right` must
    /// equal `C := C (I - tau v v^H)`, both applied to the block that starts at
    /// `(r0, c0)` and to nothing else.
    #[test]
    fn zlarf_matches_the_explicit_reflector() {
        let m = 3;
        let n = 2;
        let mut a = vec![
            cmplx(1.0, 1.0),
            cmplx(2.0, -1.0),
            cmplx(-3.0, 0.5),
            cmplx(0.5, 2.0),
            cmplx(1.5, -0.5),
            cmplx(2.5, 1.0),
        ];
        let a0 = a.clone();
        let v = vec![cmplx(1.0, 0.0), cmplx(0.5, -1.5), cmplx(-0.5, 0.25)];
        let tau = cmplx(1.75, 0.5);

        // P = I - tau v v^H
        let mut p = vec![cmplx(0.0, 0.0); m * m];
        for i in 0..m {
            for j in 0..m {
                let delta = if i == j { 1.0 } else { 0.0 };
                p[i + j * m] = cmplx(delta, 0.0) - tau * v[i] * v[j].conj();
            }
        }
        // want = P * A
        let mut want = vec![cmplx(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = cmplx(0.0, 0.0);
                for k in 0..m {
                    acc += p[i + k * m] * a0[k + j * m];
                }
                want[i + j * m] = acc;
            }
        }
        zlarf_left(&mut a, m, 0, m, 0, n, &v, tau);
        for k in 0..m * n {
            assert!(
                (a[k] - want[k]).norm() <= 1e-14 * want[k].norm().max(1.0),
                "zlarf_left at {k}: {:?} != {:?}",
                a[k],
                want[k]
            );
        }

        // Right variant on a fresh copy: want = A * P.
        let mut a = a0.clone();
        let mut want = vec![cmplx(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = cmplx(0.0, 0.0);
                for k in 0..n {
                    acc += a0[i + k * m] * p[k + j * m];
                }
                want[i + j * m] = acc;
            }
        }
        let vn = v[0..n].to_vec();
        zlarf_right(&mut a, m, 0, m, 0, n, &vn, tau);
        for k in 0..m * n {
            assert!(
                (a[k] - want[k]).norm() <= 1e-14 * want[k].norm().max(1.0),
                "zlarf_right at {k}: {:?} != {:?}",
                a[k],
                want[k]
            );
        }

        // tau == 0 is the identity.
        let mut a = a0.clone();
        zlarf_left(&mut a, m, 0, m, 0, n, &v, cmplx(0.0, 0.0));
        assert_eq!(a, a0);
    }

    /// The reflectors must leave the rest of the array alone: `(r0, c0)` and the
    /// leading dimension select a block, and touching a neighboring entry would
    /// corrupt a caller's trailing matrix.
    #[test]
    fn zlarf_respects_the_block_bounds() {
        let ld = 4;
        let mut a = vec![cmplx(9.0, 9.0); ld * 3];
        for i in 0..ld {
            for j in 0..3 {
                a[i + j * ld] = cmplx(i as f64 + 1.0, -(j as f64));
            }
        }
        let a0 = a.clone();
        let v = vec![cmplx(1.0, 0.0), cmplx(0.0, 1.0)];
        zlarf_left(&mut a, ld, 1, 2, 1, 2, &v, cmplx(2.0, 0.0));
        // Row 0 and column 0 are outside the block.
        for j in 0..3 {
            assert_eq!(a[0 + j * ld], a0[0 + j * ld], "row 0, column {j} changed");
        }
        for i in 0..ld {
            assert_eq!(a[i], a0[i], "column 0, row {i} changed");
        }
        // ... and at least one entry inside the block did change.
        assert!(a.iter().zip(a0.iter()).any(|(x, y)| x != y));
    }
}
