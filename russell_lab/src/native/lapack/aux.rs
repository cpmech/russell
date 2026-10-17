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
