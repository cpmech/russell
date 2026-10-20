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

/// `dlamch('E')`: the relative machine precision, `EPSILON/2`, which is what the
/// reference's `DLAMCH('E')` returns -- not `EPSILON`, which is its `'P'`. The
/// difference matters in `dladiv`'s underflow guard, where it enters squared.
pub(super) const DLAMCH_E: f64 = f64::EPSILON / 2.0;

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

fn dladiv2(a: f64, b: f64, c: f64, d: f64, r: f64, t: f64) -> f64 {
    if r != 0.0 {
        let br = b * r;
        if br != 0.0 { (a + br) * t } else { a * t + (b * t) * r }
    } else {
        (a + d * (b / c)) * t
    }
}

fn dladiv1(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let r = d / c;
    let t = 1.0 / (c + d * r);
    let p = dladiv2(a, b, c, d, r, t);
    let q = dladiv2(b, -a, c, d, r, t);
    (p, q)
}

/// `dladiv`: `(a + i b) / (c + i d)`, robust against overflow and underflow.
pub(super) fn dladiv(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    const BS: f64 = 2.0;
    let ov = f64::MAX;
    let un = SAFMIN;
    let eps = DLAMCH_E;
    let be = BS / (eps * eps);

    let mut aa = a;
    let mut bb = b;
    let mut cc = c;
    let mut dd = d;
    let ab = a.abs().max(b.abs());
    let cd = c.abs().max(d.abs());
    let mut s = 1.0;

    if ab >= 0.5 * ov {
        aa *= 0.5;
        bb *= 0.5;
        s *= 2.0;
    }
    if cd >= 0.5 * ov {
        cc *= 0.5;
        dd *= 0.5;
        s *= 0.5;
    }
    if ab <= un * BS / eps {
        aa *= be;
        bb *= be;
        s /= be;
    }
    if cd <= un * BS / eps {
        cc *= be;
        dd *= be;
        s *= be;
    }
    let (mut p, mut q) = if d.abs() <= c.abs() {
        dladiv1(aa, bb, cc, dd)
    } else {
        let (pp, qq) = dladiv1(bb, aa, dd, cc);
        (pp, -qq)
    };
    p *= s;
    q *= s;
    (p, q)
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

/// The result of [`dlaln2`]: the `2 x 2` solution, the power of two it was scaled
/// by, and the norm of the solution before that scaling.
pub(super) struct Laln2 {
    pub(super) x11: f64,
    pub(super) x21: f64,
    pub(super) x12: f64,
    pub(super) x22: f64,
    pub(super) scale: f64,
    pub(super) xnorm: f64,
}

/// `dlaln2`: solve a 1x1 or 2x2 real system `(ca A - w D) X = scale B`, where
/// `w = wr + i wi`. `ltrans` selects `A` or `A^T`. Returns the entries of `X`,
/// the scale factor the caller must apply to the right-hand side, and the norm
/// of `X`.
///
/// This is LAPACK's general signature, and it is the *only* copy in the crate:
/// `eig.rs` used to carry a second one specialized to `ltrans = .false.`,
/// `ca = 1`, `d1 = d2 = 1`, about 87% textually identical, and the two differed
/// where it mattered -- the specialized copy took `EPS = DLAMCH('P')` where the
/// reference takes `DLAMCH('E')` (`EPSILON/2`), which enters `dladiv`'s underflow
/// guard squared. On the specialized copy's own inputs the two are bit-identical
/// (multiplying by `ca = 1` and `d1 = d2 = 1` is exact), so no result the crate's
/// tests or the reference grid exercises moved, and the guard is now right for
/// subnormal inputs; `examples.rs`'s `DGET31` run reports LAPACK's numbers to the
/// last digit on all 230,400 cases.
///
/// The `INFO` flag the reference returns is not part of this API: a perturbed
/// solution is reported through `scale` and `xnorm`, and `DGET31` reconstructs
/// `INFO` from the inputs it kept (see `examples::get31_info`).
#[allow(clippy::too_many_arguments)]
pub(super) fn dlaln2(
    ltrans: bool,
    na: usize,
    nw: usize,
    smin: f64,
    ca: f64,
    a11: f64,
    a21: f64,
    a12: f64,
    a22: f64,
    d1: f64,
    d2: f64,
    b11: f64,
    b21: f64,
    b12: f64,
    b22: f64,
    wr: f64,
    wi: f64,
) -> Laln2 {
    const IPIVOT: [[usize; 4]; 4] = [[1, 2, 3, 4], [2, 1, 4, 3], [3, 4, 1, 2], [4, 3, 2, 1]];
    const RSWAP: [bool; 4] = [false, true, false, true];
    const ZSWAP: [bool; 4] = [false, false, true, true];

    let smlnum = 2.0 * SAFMIN;
    let bignum = 1.0 / smlnum;
    let smini = smin.max(smlnum);
    let mut scale = 1.0;

    let (x11, x21, x12, x22, xnorm);

    if na == 1 {
        if nw == 1 {
            // Real 1x1 system.
            let mut csr = ca * a11 - wr * d1;
            let mut cnorm = csr.abs();
            if cnorm < smini {
                csr = smini;
                cnorm = smini;
            }
            let bnorm = b11.abs();
            if cnorm < 1.0 && bnorm > 1.0 && bnorm > bignum * cnorm {
                scale = 1.0 / bnorm;
            }
            x11 = (b11 * scale) / csr;
            x21 = 0.0;
            x12 = 0.0;
            x22 = 0.0;
            xnorm = x11.abs();
        } else {
            // Complex 1x1 system.
            let mut csr = ca * a11 - wr * d1;
            let mut csi = -wi * d1;
            let mut cnorm = csr.abs() + csi.abs();
            if cnorm < smini {
                csr = smini;
                csi = 0.0;
                cnorm = smini;
            }
            let bnorm = b11.abs() + b12.abs();
            if cnorm < 1.0 && bnorm > 1.0 && bnorm > bignum * cnorm {
                scale = 1.0 / bnorm;
            }
            let (p, q) = dladiv(scale * b11, scale * b12, csr, csi);
            x11 = p;
            x21 = 0.0;
            x12 = q;
            x22 = 0.0;
            xnorm = x11.abs() + x12.abs();
        }
    } else {
        // 2x2 system. CRV = [CR(1,1), CR(2,1), CR(1,2), CR(2,2)].
        let crv = if ltrans {
            [ca * a11 - wr * d1, ca * a12, ca * a21, ca * a22 - wr * d2]
        } else {
            [ca * a11 - wr * d1, ca * a21, ca * a12, ca * a22 - wr * d2]
        };
        if nw == 1 {
            // Real 2x2 system.
            let mut cmax = 0.0;
            let mut icmax = 0usize;
            for j in 0..4 {
                if crv[j].abs() > cmax {
                    cmax = crv[j].abs();
                    icmax = j;
                }
            }
            if cmax < smini {
                let bnorm = b11.abs().max(b21.abs());
                if smini < 1.0 && bnorm > 1.0 && bnorm > bignum * smini {
                    scale = 1.0 / bnorm;
                }
                let temp = scale / smini;
                return Laln2 {
                    x11: temp * b11,
                    x21: temp * b21,
                    x12: 0.0,
                    x22: 0.0,
                    scale,
                    xnorm: temp * bnorm,
                };
            }
            let ur11 = crv[icmax];
            let cr21 = crv[IPIVOT[icmax][1] - 1];
            let ur12 = crv[IPIVOT[icmax][2] - 1];
            let cr22 = crv[IPIVOT[icmax][3] - 1];
            let ur11r = 1.0 / ur11;
            let lr21 = ur11r * cr21;
            let mut ur22 = cr22 - ur12 * lr21;
            if ur22.abs() < smini {
                ur22 = smini;
            }
            let (br1, mut br2) = if RSWAP[icmax] { (b21, b11) } else { (b11, b21) };
            br2 -= lr21 * br1;
            let bbnd = (br1 * (ur22 * ur11r)).abs().max(br2.abs());
            if bbnd > 1.0 && ur22.abs() < 1.0 && bbnd >= bignum * ur22.abs() {
                scale = 1.0 / bbnd;
            }
            let xr2 = (br2 * scale) / ur22;
            let xr1 = (scale * br1) * ur11r - xr2 * (ur11r * ur12);
            let (mut xa, mut xb) = if ZSWAP[icmax] { (xr2, xr1) } else { (xr1, xr2) };
            let mut xnorm = xr1.abs().max(xr2.abs());
            if xnorm > 1.0 && cmax > 1.0 && xnorm > bignum / cmax {
                let temp = cmax / bignum;
                xa *= temp;
                xb *= temp;
                xnorm *= temp;
                scale *= temp;
            }
            return Laln2 {
                x11: xa,
                x21: xb,
                x12: 0.0,
                x22: 0.0,
                scale,
                xnorm,
            };
        }
        // Complex 2x2 system.
        let civ = [-wi * d1, 0.0, 0.0, -wi * d2];
        let mut cmax = 0.0;
        let mut icmax = 0usize;
        for j in 0..4 {
            let m = crv[j].abs() + civ[j].abs();
            if m > cmax {
                cmax = m;
                icmax = j;
            }
        }
        if cmax < smini {
            let bnorm = (b11.abs() + b12.abs()).max(b21.abs() + b22.abs());
            if smini < 1.0 && bnorm > 1.0 && bnorm > bignum * smini {
                scale = 1.0 / bnorm;
            }
            let temp = scale / smini;
            return Laln2 {
                x11: temp * b11,
                x21: temp * b21,
                x12: temp * b12,
                x22: temp * b22,
                scale,
                xnorm: temp * bnorm,
            };
        }
        let ur11 = crv[icmax];
        let ui11 = civ[icmax];
        let cr21 = crv[IPIVOT[icmax][1] - 1];
        let ci21 = civ[IPIVOT[icmax][1] - 1];
        let ur12 = crv[IPIVOT[icmax][2] - 1];
        let ui12 = civ[IPIVOT[icmax][2] - 1];
        let cr22 = crv[IPIVOT[icmax][3] - 1];
        let ci22 = civ[IPIVOT[icmax][3] - 1];
        let (ur11r, ui11r, lr21, li21, ur12s, ui12s, mut ur22, mut ui22);
        if icmax == 0 || icmax == 3 {
            if ur11.abs() > ui11.abs() {
                let temp = ui11 / ur11;
                ur11r = 1.0 / (ur11 * (1.0 + temp * temp));
                ui11r = -temp * ur11r;
            } else {
                let temp = ur11 / ui11;
                ui11r = -1.0 / (ui11 * (1.0 + temp * temp));
                ur11r = -temp * ui11r;
            }
            lr21 = cr21 * ur11r;
            li21 = cr21 * ui11r;
            ur12s = ur12 * ur11r;
            ui12s = ur12 * ui11r;
            ur22 = cr22 - ur12 * lr21;
            ui22 = ci22 - ur12 * li21;
        } else {
            ur11r = 1.0 / ur11;
            ui11r = 0.0;
            lr21 = cr21 * ur11r;
            li21 = ci21 * ur11r;
            ur12s = ur12 * ur11r;
            ui12s = ui12 * ur11r;
            ur22 = cr22 - ur12 * lr21 + ui12 * li21;
            ui22 = -ur12 * li21 - ui12 * lr21;
        }
        let u22abs = ur22.abs() + ui22.abs();
        if u22abs < smini {
            ur22 = smini;
            ui22 = 0.0;
        }
        let (mut br1, mut br2, mut bi1, mut bi2) = if RSWAP[icmax] {
            (b21, b11, b22, b12)
        } else {
            (b11, b21, b12, b22)
        };
        br2 = br2 - lr21 * br1 + li21 * bi1;
        bi2 = bi2 - li21 * br1 - lr21 * bi1;
        let bbnd = ((br1.abs() + bi1.abs()) * (u22abs * (ur11r.abs() + ui11r.abs()))).max(br2.abs() + bi2.abs());
        if bbnd > 1.0 && u22abs < 1.0 && bbnd >= bignum * u22abs {
            scale = 1.0 / bbnd;
            br1 *= scale;
            bi1 *= scale;
            br2 *= scale;
            bi2 *= scale;
        }
        let (xr2, xi2) = dladiv(br2, bi2, ur22, ui22);
        let xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
        let xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
        let (mut xa, mut xb, mut xc, mut xd) = if ZSWAP[icmax] {
            (xr2, xr1, xi2, xi1)
        } else {
            (xr1, xr2, xi1, xi2)
        };
        let mut xnorm = (xr1.abs() + xi1.abs()).max(xr2.abs() + xi2.abs());
        if xnorm > 1.0 && cmax > 1.0 && xnorm > bignum / cmax {
            let temp = cmax / bignum;
            xa *= temp;
            xb *= temp;
            xc *= temp;
            xd *= temp;
            xnorm *= temp;
            scale *= temp;
        }
        return Laln2 {
            x11: xa,
            x21: xb,
            x12: xc,
            x22: xd,
            scale,
            xnorm,
        };
    }
    Laln2 {
        x11,
        x21,
        x12,
        x22,
        scale,
        xnorm,
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

    /// `dladiv` on the inputs that reach its rescale arms, pinned against the
    /// reference.
    ///
    /// Values captured from `SRC/dladiv.f` compiled against LAPACK 3.12.1's
    /// `liblapack`. The eight cases are chosen for the branches rather than for
    /// the arithmetic: overflowing operands (`1e308`), underflowing ones
    /// (`1e-308`), a subnormal quotient, mixed magnitudes, and one where the
    /// result underflows to zero. They also pin the `DLAMCH('E')` this routine
    /// takes: the copy that used to live in `eig.rs` took `EPSILON` instead of
    /// `EPSILON/2`, which enters `be = 2/(eps*eps)`, and these cases are where
    /// that difference would show.
    #[test]
    fn dladiv_matches_the_reference_on_the_rescale_arms() {
        // (a, b, c, d) and the reference's (p, q).
        let cases: [([f64; 4], [f64; 2]); 12] = [
            ([1e308, 1e308, 1e308, 1e308], [9.99999999999999889e-01, 0.0]),
            ([1e-308, 1e-308, 1e-308, 1e-308], [9.99999999999999889e-01, 0.0]),
            ([4e-320, 0.0, 1e-320, 0.0], [3.99999999999999956e0, 0.0]),
            ([1e300, 1e300, 1e300, 1e300], [1.0, 0.0]),
            ([3.0, 2.0, 1.0, 1.0], [2.5, -0.5]),
            (
                [1.0, 0.0, 2.0, 1.0],
                [4.00000000000000022e-01, -2.00000000000000011e-01],
            ),
            ([1e-300, 1e-300, 1e300, -1e300], [0.0, 0.0]),
            ([1e200, -1e200, 1e-200, 1e200], [-1.0, -1.0]),
            // The arms the eight above do not reach: an overflowing denominator,
            // a negligible numerator, a negligible denominator, and |d| > |c|
            // (the branch that swaps the operands before dividing).
            ([1.0, 1.0, 1e308, 1e308], [9.99999999999999909e-309, 0.0]),
            (
                [1e-300, 0.0, 1.0, 1.0],
                [5.00000000000000013e-301, -5.00000000000000013e-301],
            ),
            ([1.0, 1.0, 1e-300, 1e-300], [9.99999999999999904e299, 0.0]),
            ([2.0, 1.0, 1.0, 3.0], [4.99999999999999944e-01, -0.5]),
        ];
        for (k, ([a, b, c, d], want)) in cases.iter().enumerate() {
            let (p, q) = dladiv(*a, *b, *c, *d);
            assert_eq!([p, q], *want, "case {k}");
        }
    }
}
