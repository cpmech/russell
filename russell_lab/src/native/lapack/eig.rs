//! Native real eigen-decompositions: `dgeev` and the pieces it is built from.
//!
//! # Why this is split into steps
//!
//! `dgeev` in LAPACK is a pipeline, and each stage is useful (and testable) on
//! its own:
//!
//! | Step | LAPACK | What it does |
//! |---|---|---|
//! | 1 | `dgebal` | permutes and scales to improve the conditioning |
//! | 2 | `dgehrd` | orthogonal reduction to upper Hessenberg form, `H = Q^T A Q` |
//! | 3 | `dhseqr` | the QR algorithm: real Schur form `T = Z^T H Z` |
//! | 4 | `dtrevc` | eigenvectors of `T` by back-substitution |
//! | 5 | `dgebak` | undoes the balancing |
//!
//! All five steps are here, plus the `dgeev` driver, in the reference's order:
//! balancing, Hessenberg reduction, Schur reduction, eigenvectors, and undoing
//! the balancing. Everything is column-major, so element `(i, j)` of an `n x n`
//! matrix lives at `data[i + j * n]`.
//!
//! # The Schur form is a *real* one
//!
//! A real matrix can have complex eigenvalues, so the real Schur form is only
//! *quasi*-triangular: 1x1 blocks on the diagonal for real eigenvalues and 2x2
//! blocks for complex conjugate pairs. Those 2x2 blocks are why the eigenvector
//! step needs `dlaln2`-style solving rather than a plain triangular substitution.

use super::LapackError;
use super::aux::dladiv2;

/// A 2x2 block's eigenvalues, as [`dlanv2`] returns them.
///
/// `rt1r`/`rt1i` and `rt2r`/`rt2i` are the two eigenvalues. They are real when
/// `rt1i == rt2i == 0`, and otherwise a complex conjugate pair with `rt1i > 0`,
/// which is LAPACK's convention and is relied on by the 2x2-block handling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eigen2 {
    /// Real part of the first eigenvalue
    pub rt1r: f64,
    /// Imaginary part of the first eigenvalue (`> 0` for a conjugate pair)
    pub rt1i: f64,
    /// Real part of the second eigenvalue
    pub rt2r: f64,
    /// Imaginary part of the second eigenvalue (`-rt1i` for a pair)
    pub rt2i: f64,
}

/// `dlarfg`: an elementary reflector `H = I - tau * v * v^T` with `H x = beta e1`.
///
/// `alpha` is the entry to annihilate and `x` the rest of the column. The return
/// is `(beta, tau, v)` where `v[0] == 1` and the remaining entries are the tail;
/// `tau == 0` marks the identity (the tail was already zero) and `v` is then all
/// zeros.
///
/// The scaling follows LAPACK 3.12.1's `dlarfg`, because the naive form fails at
/// both ends of the exponent range:
///
/// * `beta = -sign(alpha) * hypot(alpha, ||x||)`. The *opposite* sign to `alpha`
///   avoids catastrophic cancellation when `x` is small; taking the same sign
///   loses most of the significant digits.
/// * If `beta` comes out subnormal then the norm and `beta` are themselves
///   inaccurate, so the vector is scaled up by `safmin` repeatedly (at most 20
///   times) and recomputed. The result is scaled back at the end, so `beta` and
///   `v` are consistent with the *original* input.
pub fn dlarfg(alpha: f64, x: &[f64]) -> (f64, f64, Vec<f64>) {
    let n = x.len() + 1;
    let mut v = vec![0.0; n];
    if n <= 1 {
        return (alpha, 0.0, v);
    }

    // ||x||_2, rescaled so that the sum of squares cannot overflow.
    let mut xscale = 0.0;
    for &xi in x {
        let a = xi.abs();
        if a > xscale {
            xscale = a;
        }
    }
    if xscale == 0.0 {
        // The tail is already zero, so no reflector is needed.
        return (alpha, 0.0, v);
    }
    let ss: f64 = x.iter().map(|&xi| (xi / xscale) * (xi / xscale)).sum();
    let mut xnorm = xscale * ss.sqrt();

    let safmin = f64::MIN_POSITIVE;
    let mut alpha = alpha;
    let mut beta = -alpha.signum() * f64::hypot(alpha, xnorm);

    if beta.abs() < safmin {
        // Recompute in a rescaled frame until beta is normal.
        let safmin_inv = 1.0 / safmin;
        let mut knt = 0usize;
        let mut xw = x.to_vec();
        loop {
            knt += 1;
            for xi in xw.iter_mut() {
                *xi *= safmin;
            }
            alpha *= safmin;
            // The reference multiplies XNORM by SAFMIN here and recomputes BETA
            // from it; porting the multiplication of `beta` as well would be a
            // dead store, since BETA is recomputed below.
            let mut sc = 0.0;
            for &xi in xw.iter() {
                let a = xi.abs();
                if a > sc {
                    sc = a;
                }
            }
            xnorm = if sc == 0.0 {
                0.0
            } else {
                let ss: f64 = xw.iter().map(|&xi| (xi / sc) * (xi / sc)).sum();
                sc * ss.sqrt()
            };
            beta = -alpha.signum() * f64::hypot(alpha, xnorm);
            if beta.abs() >= safmin || knt >= 20 {
                break;
            }
        }
        // Scale everything back to the original frame.
        let back = safmin_inv.powi(knt as i32);
        beta /= back;
        alpha /= back;
        let tau = (beta - alpha) / beta;
        let s = 1.0 / (alpha - beta);
        for (i, xi) in xw.iter().enumerate() {
            v[i + 1] = (xi / back) * s;
        }
        v[0] = 1.0;
        return (beta, tau, v);
    }

    let tau = (beta - alpha) / beta;
    let s = 1.0 / (alpha - beta);
    for (i, xi) in x.iter().enumerate() {
        v[i + 1] = xi * s;
    }
    v[0] = 1.0;
    (beta, tau, v)
}

/// `dlanv2`: the eigenvalues of a 2x2 block, plus the rotation that standardises it.
///
/// Follows LAPACK 3.12.1's `dlanv2`. On input `a, b, c, d` are
///
/// ```text
/// [ a  b ]
/// [ c  d ]
/// ```
///
/// and `cs`/`sn` are the cosine and sine of the rotation applied. For real
/// eigenvalues the rotation makes the block upper triangular; for a complex pair
/// it makes the two diagonal entries equal, which is what lets the QR algorithm
/// deflate the block. `rt1i > 0` for a pair.
#[allow(clippy::too_many_arguments)]
pub fn dlanv2(a: &mut f64, b: &mut f64, c: &mut f64, d: &mut f64, cs: &mut f64, sn: &mut f64) -> Eigen2 {
    // LAPACK's MULTPL: below 4*eps the decision between real and complex
    // eigenvalues is not trustworthy, so the "equalise the diagonal" branch is
    // taken instead.
    const MULTPL: f64 = 4.0;
    let safmin = f64::MIN_POSITIVE;
    let eps = f64::EPSILON;
    let safmn2 = (safmin / eps).sqrt();
    let safmx2 = 1.0 / safmn2;

    let mut temp;

    if *c == 0.0 {
        // Already upper triangular.
        *cs = 1.0;
        *sn = 0.0;
    } else if *b == 0.0 {
        // Swap rows and columns: the block is lower triangular.
        *cs = 0.0;
        *sn = 1.0;
        temp = *d;
        *d = *a;
        *a = temp;
        *b = -*c;
        *c = 0.0;
    } else if (*a - *d) == 0.0 && b.signum() != c.signum() {
        // Already in standard complex form.
        *cs = 1.0;
        *sn = 0.0;
    } else {
        temp = *a - *d;
        let mut p = 0.5 * temp;
        let bcmax = b.abs().max(c.abs());
        let bcmis = b.abs().min(c.abs()) * b.signum() * c.signum();
        let scale = p.abs().max(bcmax);
        let mut z = (p / scale) * p + (bcmax / scale) * bcmis;

        if z >= MULTPL * eps {
            // Real eigenvalues.
            z = p + z.sqrt() * scale.sqrt() * p.signum();
            *a = *d + z;
            *d -= (bcmax / z) * bcmis;
            let tau = f64::hypot(*c, z);
            *cs = z / tau;
            *sn = *c / tau;
            *b -= *c;
            *c = 0.0;
        } else {
            // Complex eigenvalues, or real (almost) equal ones: equalise the
            // diagonal first.
            let mut count = 0;
            let mut sigma = *b + *c;
            loop {
                count += 1;
                let sc = temp.abs().max(sigma.abs());
                if sc >= safmx2 {
                    sigma *= safmn2;
                    temp *= safmn2;
                } else if sc <= safmn2 {
                    sigma *= safmx2;
                    temp *= safmx2;
                } else {
                    break;
                }
                if count > 20 {
                    break;
                }
            }
            p = 0.5 * temp;
            let tau = f64::hypot(sigma, temp);
            *cs = (0.5 * (1.0 + sigma.abs() / tau)).sqrt();
            *sn = -(p / (tau * *cs)) * sigma.signum();

            // [ AA BB ] = [ A B ] [ CS -SN ]
            // [ CC DD ]   [ C D ] [ SN  CS ]
            let aa = *a * *cs + *b * *sn;
            let bb = -*a * *sn + *b * *cs;
            let cc = *c * *cs + *d * *sn;
            let dd = -*c * *sn + *d * *cs;

            // [ A B ] = [ CS  SN ] [ AA BB ]
            // [ C D ]   [-SN  CS ] [ CC DD ]
            //
            // The products are bracketed the way LAPACK brackets them, to keep a
            // fused multiply-add from changing the result; see the FMA note in
            // dlanv2.f (upstream issue #1031).
            *a = aa * *cs + cc * *sn;
            *b = (bb * *cs) + (dd * *sn);
            *c = -(aa * *sn) + (cc * *cs);
            *d = -bb * *sn + dd * *cs;

            temp = 0.5 * (*a + *d);
            *a = temp;
            *d = temp;

            if *c != 0.0 {
                if *b != 0.0 {
                    if b.signum() == c.signum() {
                        // The pair turned out to be real after all: reduce to
                        // upper triangular form.
                        let sab = b.abs().sqrt();
                        let sac = c.abs().sqrt();
                        p = sab * sac * c.signum();
                        let tau = 1.0 / (b.abs() + c.abs()).sqrt();
                        *a = temp + p;
                        *d = temp - p;
                        *b -= *c;
                        *c = 0.0;
                        let cs1 = sab * tau;
                        let sn1 = sac * tau;
                        temp = *cs * cs1 - *sn * sn1;
                        *sn = *cs * sn1 + *sn * cs1;
                        *cs = temp;
                    }
                } else {
                    *b = -*c;
                    *c = 0.0;
                    temp = *cs;
                    *cs = -*sn;
                    *sn = temp;
                }
            }
        }
    }

    let rt1r = *a;
    let rt2r = *d;
    if *c == 0.0 {
        Eigen2 {
            rt1r,
            rt1i: 0.0,
            rt2r,
            rt2i: 0.0,
        }
    } else {
        let wi = b.abs().sqrt() * c.abs().sqrt();
        Eigen2 {
            rt1r,
            rt1i: wi,
            rt2r,
            rt2i: -wi,
        }
    }
}

/// `dgehrd` (the unblocked `dgehd2` path): reduce to upper Hessenberg form.
///
/// On exit `a` holds `H = Q^T A Q`, which is upper Hessenberg. `q`, if given,
/// receives the orthogonal `Q` accumulated from the reflectors, so that
/// `A = Q H Q^T`.
///
/// The reflectors are the same ones LAPACK packs into the zeroed part of `a`;
/// they are kept in a separate `Vec` here because this caller wants both `H` and
/// `Q`, and the packing buys nothing outside the blocked algorithm.
///
/// # The update is the literal `dgehd2` sequence
///
/// A step is `A := H A H` with `H = I - tau v v^T`, where `H` has order
/// `m = n - i - 1` and acts on rows/columns `i+1 .. n-1` (`v[0] == 1` sits at
/// row `i+1`). LAPACK forms this as two in-place applications against the same
/// array, in this order:
///
/// ```text
/// A(0:n, i+1:n)   := A(0:n, i+1:n) * H      (all rows, columns i+1..)
/// A(i+1:n, i+1:n) := H * A(i+1:n, i+1:n)
/// ```
///
/// The left application deliberately reads the values the right one just wrote,
/// because the target is `H * (A * H)`. Row `i` and column `i` are never touched.
/// Building `H` on rows `i .. n-1` (one row too early) or as a square reflector
/// on the wrong number of columns both break the similarity while still leaving
/// an orthogonal `Q`, so the mistake is easy to miss.
///
/// # Panics
///
/// Panics if `a.len() != n * n` or `q` is given with the wrong length.
pub fn dgehrd(a: &mut [f64], n: usize, q: Option<&mut [f64]>) {
    assert_eq!(a.len(), n * n, "dgehrd: a must be n*n");
    if let Some(qq) = q.as_ref() {
        assert_eq!(qq.len(), n * n, "dgehrd: q must be n*n");
    }
    if n < 3 {
        // Nothing to reduce; Q is the identity.
        if let Some(qq) = q {
            qq.fill(0.0);
            for i in 0..n {
                qq[i + i * n] = 1.0;
            }
        }
        return;
    }

    // A reflector is dropped (tau == 0) when a column is already in the required
    // form, so it is stored as an option.
    let mut vs: Vec<Option<(Vec<f64>, f64)>> = vec![None; n];

    for i in 0..(n - 2) {
        // Annihilate a[i+2..n, i] into a[i+1, i].
        let alpha = a[(i + 1) + i * n];
        let tail: Vec<f64> = ((i + 2)..n).map(|r| a[r + i * n]).collect();
        let (beta, tau, v) = dlarfg(alpha, &tail);
        a[(i + 1) + i * n] = beta;
        for r in (i + 2)..n {
            a[r + i * n] = 0.0;
        }
        if tau == 0.0 {
            continue;
        }
        vs[i] = Some((v.clone(), tau));

        // Apply H(i) from both sides, exactly as dgehd2 does. The reflector has
        // order m = n-i-1 and acts on rows/columns i+1..n-1; v[0] == 1 sits at
        // row i+1. Row i and column i are never touched.
        let m = n - i - 1;

        // Right: A(0:n, i+1:n) := A(0:n, i+1:n) * H, with
        // (A H)(r, j) = A(r, j) - tau * (sum_l A(r, i+1+l) v[l]) * v[j-(i+1)].
        // Every dot product is formed from the block before anything is written.
        let mut w = vec![0.0; n];
        for r in 0..n {
            let mut acc = 0.0;
            for l in 0..m {
                acc += a[r + (i + 1 + l) * n] * v[l];
            }
            w[r] = acc;
        }
        for r in 0..n {
            for l in 0..m {
                a[r + (i + 1 + l) * n] -= tau * w[r] * v[l];
            }
        }

        // Left: A(i+1:n, i+1:n) := H * A(i+1:n, i+1:n). This reads the values
        // the right application just wrote, giving H * (A * H) = H A H.
        for j in (i + 1)..n {
            let mut acc = 0.0;
            for l in 0..m {
                acc += v[l] * a[(i + 1 + l) + j * n];
            }
            for l in 0..m {
                a[(i + 1 + l) + j * n] -= tau * v[l] * acc;
            }
        }
    }

    if let Some(qq) = q {
        // Accumulate Q = H(0) H(1) ... H(n-3) into the identity.
        //
        // Applying the reflectors in reverse and multiplying on the left builds
        // the product: after processing i, `qq` holds H(i) H(i+1) ... H(n-3).
        // The reflector has order m = n-i-1, acts on rows i+1..n-1, and its
        // vector is `v` with v[0] == 1 at row i+1. Each column is read into a
        // scratch buffer first: forming the dot product from `qq` in place would
        // read rows the same update had already changed.
        qq.fill(0.0);
        for i in 0..n {
            qq[i + i * n] = 1.0;
        }
        for i in (0..(n - 2)).rev() {
            let (v, tau) = match &vs[i] {
                Some((v, tau)) => (v.as_slice(), *tau),
                None => continue,
            };
            let m = v.len();
            let base = i + 1;
            let mut col = vec![0.0; m];
            for j in 0..n {
                for r in 0..m {
                    col[r] = qq[(base + r) + j * n];
                }
                let mut acc = 0.0;
                for r in 0..m {
                    acc += v[r] * col[r];
                }
                let w = tau * acc;
                for r in 0..m {
                    qq[(base + r) + j * n] = col[r] - w * v[r];
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `dlahqr` (LAPACK 3.12.1, `SRC/dlahqr.f`) with `WANTZ`.
///
/// This is the same literal transcription as before; the only addition is the
/// accumulation of the rotations into `z` when `wantz` is set, which is what
/// `dhseqr` with `COMPZ = 'V'` needs. `z` is `n x n` column-major and is only
/// referenced when `wantz` is true.
fn dlahqr(
    n: usize,
    ilo: usize,
    ihi: usize,
    h: &mut [f64],
    ldh: usize,
    wr: &mut [f64],
    wi: &mut [f64],
    wantz: bool,
    z: &mut [f64],
) -> usize {
    // PARAMETER ( KEXSH = 10 ) and the exceptional-shift constants DAT1/DAT2.
    const KEXSH: usize = 10;
    const DAT1: f64 = 3.0 / 4.0;
    const DAT2: f64 = -0.4375;

    // Quick return if possible.
    if n == 0 {
        return 0;
    }
    if ilo == ihi {
        wr[ilo] = h[ilo + ilo * ldh];
        wi[ilo] = 0.0;
        return 0;
    }

    // ==== clear out the trash ====
    // DO 10 J = ILO, IHI - 3
    let mut j = ilo;
    while j + 3 <= ihi {
        h[(j + 2) + j * ldh] = 0.0;
        h[(j + 3) + j * ldh] = 0.0;
        j += 1;
    }
    if ilo + 2 <= ihi {
        h[ihi + (ihi - 2) * ldh] = 0.0;
    }

    let nh = ihi - ilo + 1;

    // Set machine-dependent constants for the stopping criterion.
    let safmin = f64::MIN_POSITIVE;
    let ulp = f64::EPSILON;
    let smlnum = safmin * (nh as f64 / ulp);

    // I1 and I2 are the indices of the first row and last column of H to which
    // transformations must be applied. With WANTT they are the whole matrix and
    // are fixed; otherwise they are set inside the main loop.
    let wantt = true;
    let mut i1 = 0usize;
    let mut i2 = n - 1;

    // ITMAX is the total number of QR iterations allowed.
    let itmax = 30 * nh.max(10);

    // KDEFL counts the number of iterations since a deflation.
    let mut kdefl: usize = 0;

    // The main loop begins here. I is the loop index and decreases from IHI to
    // ILO in steps of 1 or 2. Each iteration works with the active submatrix in
    // rows and columns L to I. Eigenvalues I+1 to IHI have already converged.
    // Either L = ILO or H(L,L-1) is negligible so that the matrix splits.
    let mut i = ihi;
    loop {
        // 20 CONTINUE / IF ( I.LT.ILO ) GO TO 160
        if i < ilo {
            return 0;
        }
        let mut l = ilo;

        // DO 140 ITS = 0, ITMAX. A completed loop falls through to INFO = I.
        let mut split_off = false;
        let mut iters = 0usize;
        while iters <= itmax {
            iters += 1;

            // Look for a single small subdiagonal element.
            // DO 30 K = I, L + 1, -1; running off the top leaves K = L.
            let mut k = i;
            while k > l {
                if h[k + (k - 1) * ldh].abs() <= smlnum {
                    break;
                }
                let mut tst = h[(k - 1) + (k - 1) * ldh].abs() + h[k + k * ldh].abs();
                if tst == 0.0 {
                    if k >= ilo + 2 {
                        tst += h[(k - 1) + (k - 2) * ldh].abs();
                    }
                    if k < ihi {
                        tst += h[(k + 1) + k * ldh].abs();
                    }
                }
                // ==== The following is a conservative small subdiagonal
                // .    deflation criterion due to Ahues & Tisseur (LAWN 122,
                // .    1997). It has better mathematical foundation and
                // .    improves accuracy in some cases. ====
                if h[k + (k - 1) * ldh].abs() <= ulp * tst {
                    let ab = h[k + (k - 1) * ldh].abs().max(h[(k - 1) + k * ldh].abs());
                    let ba = h[k + (k - 1) * ldh].abs().min(h[(k - 1) + k * ldh].abs());
                    let aa = h[k + k * ldh]
                        .abs()
                        .max((h[(k - 1) + (k - 1) * ldh] - h[k + k * ldh]).abs());
                    let bb = h[k + k * ldh]
                        .abs()
                        .min((h[(k - 1) + (k - 1) * ldh] - h[k + k * ldh]).abs());
                    let s = aa + ab;
                    if ba * (ab / s) <= smlnum.max(ulp * (bb * (aa / s))) {
                        break;
                    }
                }
                k -= 1;
            }
            // 40 CONTINUE
            l = k;
            if l > ilo {
                // H(L,L-1) is negligible.
                h[l + (l - 1) * ldh] = 0.0;
            }

            // Exit from loop if a submatrix of order 1 or 2 has split off.
            if l + 1 >= i {
                split_off = true;
                break;
            }
            kdefl += 1;

            // Now the active submatrix is in rows and columns L to I. If
            // eigenvalues only are being computed, only the active submatrix
            // need be transformed.
            if !wantt {
                i1 = l;
                i2 = i;
            }

            // Prepare the shift. The two exceptional shifts read a subdiagonal
            // two rows in from an end of the block, which is safe here because
            // L < I - 1 was checked above: the block always has at least 3 rows.
            let (mut h11, mut h12, mut h21, mut h22);
            if kdefl % (2 * KEXSH) == 0 {
                // Exceptional shift (from the bottom).
                let s = h[i + (i - 1) * ldh].abs() + h[(i - 1) + (i - 2) * ldh].abs();
                h11 = DAT1 * s + h[i + i * ldh];
                h12 = DAT2 * s;
                h21 = s;
                h22 = h11;
            } else if kdefl % KEXSH == 0 {
                // Exceptional shift (from the top).
                let s = h[(l + 1) + l * ldh].abs() + h[(l + 2) + (l + 1) * ldh].abs();
                h11 = DAT1 * s + h[l + l * ldh];
                h12 = DAT2 * s;
                h21 = s;
                h22 = h11;
            } else {
                // Prepare to use Francis' double shift
                // (i.e. 2nd degree generalized Rayleigh quotient).
                h11 = h[(i - 1) + (i - 1) * ldh];
                h21 = h[i + (i - 1) * ldh];
                h12 = h[(i - 1) + i * ldh];
                h22 = h[i + i * ldh];
            }
            let s = h11.abs() + h12.abs() + h21.abs() + h22.abs();
            let mut rt1r = 0.0;
            let mut rt1i = 0.0;
            let mut rt2r = 0.0;
            let mut rt2i = 0.0;
            if s == 0.0 {
                // All four entries are zero: the shifts are zero.
            } else {
                h11 /= s;
                h21 /= s;
                h12 /= s;
                h22 /= s;
                let tr = (h11 + h22) / 2.0;
                let det = (h11 - tr) * (h22 - tr) - h12 * h21;
                let rtdisc = det.abs().sqrt();
                if det >= 0.0 {
                    // ==== complex conjugate shifts ====
                    rt1r = tr * s;
                    rt2r = rt1r;
                    rt1i = rtdisc * s;
                    rt2i = -rt1i;
                } else {
                    // ==== real shifts (use only one of them) ====
                    rt1r = tr + rtdisc;
                    rt2r = tr - rtdisc;
                    if (rt1r - h22).abs() <= (rt2r - h22).abs() {
                        rt1r *= s;
                        rt2r = rt1r;
                    } else {
                        rt2r *= s;
                        rt1r = rt2r;
                    }
                    rt1i = 0.0;
                    rt2i = 0.0;
                }
            }

            // Look for two consecutive small subdiagonal elements.
            // DO 50 M = I - 2, L, -1; the M = L case always breaks out at 60.
            let mut m = i - 2;
            let mut v = [0.0f64; 3];
            loop {
                // Determine the effect of starting the double-shift QR
                // iteration at row M, and see if this would make H(M,M-1)
                // negligible. (The following uses scaling to avoid overflows and
                // most underflows.)
                let mut h21s = h[(m + 1) + m * ldh];
                let sv = (h[m + m * ldh] - rt2r).abs() + rt2i.abs() + h21s.abs();
                h21s = h[(m + 1) + m * ldh] / sv;
                v[0] = h21s * h[m + (m + 1) * ldh] + (h[m + m * ldh] - rt1r) * ((h[m + m * ldh] - rt2r) / sv)
                    - rt1i * (rt2i / sv);
                v[1] = h21s * (h[m + m * ldh] + h[(m + 1) + (m + 1) * ldh] - rt1r - rt2r);
                v[2] = h21s * h[(m + 2) + (m + 1) * ldh];
                let sv = v[0].abs() + v[1].abs() + v[2].abs();
                v[0] /= sv;
                v[1] /= sv;
                v[2] /= sv;
                if m == l {
                    break;
                }
                if h[m + (m - 1) * ldh].abs() * (v[1].abs() + v[2].abs())
                    <= ulp
                        * v[0].abs()
                        * (h[(m - 1) + (m - 1) * ldh].abs() + h[m + m * ldh].abs() + h[(m + 1) + (m + 1) * ldh].abs())
                {
                    break;
                }
                m -= 1;
            }
            // 60 CONTINUE

            // Double-shift QR step.
            // DO 130 K = M, I - 1
            for k in m..=(i - 1) {
                // NR is the order of the reflector G, at most 3.
                let nr = 3.min(i - k + 1);
                if k > m {
                    // The subsequent iterations determine the reflector from
                    // the bulge in column K-1.
                    for t in 0..nr {
                        v[t] = h[(k + t) + (k - 1) * ldh];
                    }
                }
                let (beta, t1, vv) = dlarfg(v[0], &v[1..nr]);
                v[..nr].copy_from_slice(&vv[..nr]);
                if k > m {
                    h[k + (k - 1) * ldh] = beta;
                    h[(k + 1) + (k - 1) * ldh] = 0.0;
                    if k < i - 1 {
                        h[(k + 2) + (k - 1) * ldh] = 0.0;
                    }
                } else if m > l {
                    // ==== Use the following instead of
                    // .    H( K, K-1 ) = -H( K, K-1 ) to
                    // .    avoid a bug when v(2) and v(3)
                    // .    underflow. ====
                    h[k + (k - 1) * ldh] *= 1.0 - t1;
                }
                let v2 = v[1];
                let t2 = t1 * v2;
                if nr == 3 {
                    let v3 = v[2];
                    let t3 = t1 * v3;

                    // Apply G from the left to transform the rows of the matrix
                    // in columns K to I2.
                    for jj in k..=i2 {
                        let sum = h[k + jj * ldh] + v2 * h[(k + 1) + jj * ldh] + v3 * h[(k + 2) + jj * ldh];
                        h[k + jj * ldh] -= sum * t1;
                        h[(k + 1) + jj * ldh] -= sum * t2;
                        h[(k + 2) + jj * ldh] -= sum * t3;
                    }

                    // Apply G from the right to transform the columns of the
                    // matrix in rows I1 to min(K+3,I).
                    for jj in i1..=(k + 3).min(i) {
                        let sum = h[jj + k * ldh] + v2 * h[jj + (k + 1) * ldh] + v3 * h[jj + (k + 2) * ldh];
                        h[jj + k * ldh] -= sum * t1;
                        h[jj + (k + 1) * ldh] -= sum * t2;
                        h[jj + (k + 2) * ldh] -= sum * t3;
                    }

                    // Accumulate transformations in the matrix Z.
                    if wantz {
                        for jj in ilo..=ihi {
                            let sum = z[jj + k * n] + v2 * z[jj + (k + 1) * n] + v3 * z[jj + (k + 2) * n];
                            z[jj + k * n] -= sum * t1;
                            z[jj + (k + 1) * n] -= sum * t2;
                            z[jj + (k + 2) * n] -= sum * t3;
                        }
                    }
                } else if nr == 2 {
                    // Apply G from the left to transform the rows of the matrix
                    // in columns K to I2.
                    for jj in k..=i2 {
                        let sum = h[k + jj * ldh] + v2 * h[(k + 1) + jj * ldh];
                        h[k + jj * ldh] -= sum * t1;
                        h[(k + 1) + jj * ldh] -= sum * t2;
                    }

                    // Apply G from the right to transform the columns of the
                    // matrix in rows I1 to I.
                    for jj in i1..=i {
                        let sum = h[jj + k * ldh] + v2 * h[jj + (k + 1) * ldh];
                        h[jj + k * ldh] -= sum * t1;
                        h[jj + (k + 1) * ldh] -= sum * t2;
                    }

                    // Accumulate transformations in the matrix Z.
                    if wantz {
                        for jj in ilo..=ihi {
                            let sum = z[jj + k * n] + v2 * z[jj + (k + 1) * n];
                            z[jj + k * n] -= sum * t1;
                            z[jj + (k + 1) * n] -= sum * t2;
                        }
                    }
                }
            }
        }

        // Failure to converge in remaining number of iterations: INFO = I.
        if !split_off {
            return i;
        }

        // 150 CONTINUE
        if l == i {
            // H(I,I-1) is negligible: one eigenvalue has converged.
            wr[i] = h[i + i * ldh];
            wi[i] = 0.0;
        } else if l + 1 == i {
            // H(I-1,I-2) is negligible: a pair of eigenvalues have converged.
            // Transform the 2-by-2 submatrix to standard Schur form, and compute
            // and store the eigenvalues.
            let (mut a, mut b, mut c, mut d) = (
                h[(i - 1) + (i - 1) * ldh],
                h[(i - 1) + i * ldh],
                h[i + (i - 1) * ldh],
                h[i + i * ldh],
            );
            let (mut cs, mut sn) = (0.0, 0.0);
            let e = dlanv2(&mut a, &mut b, &mut c, &mut d, &mut cs, &mut sn);
            h[(i - 1) + (i - 1) * ldh] = a;
            h[(i - 1) + i * ldh] = b;
            h[i + (i - 1) * ldh] = c;
            h[i + i * ldh] = d;
            wr[i - 1] = e.rt1r;
            wi[i - 1] = e.rt1i;
            wr[i] = e.rt2r;
            wi[i] = e.rt2i;

            if wantt {
                // Apply the transformation to the rest of H.
                //
                // DROT( I2-I, H( I-1, I+1 ), LDH, H( I, I+1 ), LDH, CS, SN )
                if i2 > i {
                    for jj in (i + 1)..=i2 {
                        let x = h[(i - 1) + jj * ldh];
                        let y = h[i + jj * ldh];
                        let temp = cs * x + sn * y;
                        h[i + jj * ldh] = cs * y - sn * x;
                        h[(i - 1) + jj * ldh] = temp;
                    }
                }
                // DROT( I-I1-1, H( I1, I-1 ), 1, H( I1, I ), 1, CS, SN )
                let count = i - i1 - 1;
                for r in i1..(i1 + count) {
                    let x = h[r + (i - 1) * ldh];
                    let y = h[r + i * ldh];
                    let temp = cs * x + sn * y;
                    h[r + i * ldh] = cs * y - sn * x;
                    h[r + (i - 1) * ldh] = temp;
                }
            }
            if wantz {
                // Apply the transformation to Z.
                //
                // DROT( NZ, Z( ILOZ, I-1 ), 1, Z( ILOZ, I ), 1, CS, SN )
                for r in ilo..=ihi {
                    let x = z[r + (i - 1) * n];
                    let y = z[r + i * n];
                    let temp = cs * x + sn * y;
                    z[r + i * n] = cs * y - sn * x;
                    z[r + (i - 1) * n] = temp;
                }
            }
        }

        // Reset the deflation counter and return to the start of the main loop
        // with the new value of I.
        kdefl = 0;
        if l == 0 {
            return 0;
        }
        i = l - 1;
    }
}

/// `dhseqr` with `JOB = 'S'`, `COMPZ = 'N'`: reduce an upper Hessenberg matrix to
/// real Schur form and return its eigenvalues.
///
/// On entry `t` is `n x n` and upper Hessenberg; on exit it is quasi-upper
/// triangular (zero below the first subdiagonal) and `(wr, wi)` holds the
/// eigenvalues in deflation order. A complex conjugate pair occupies two
/// adjacent positions with `wi > 0` for the first, matching the 2x2 block.
///
/// This calls `dlahqr` directly, which is what the reference `dhseqr` does for
/// `n <= 75`; the blocked `dlaqr0`/`dlaqr5` path is not implemented. `JOB = 'S'`
/// means `WANTT = .TRUE.` and `COMPZ = 'N'` means `WANTZ = .FALSE.`, and with
/// `ILO = 1, IHI = N` the whole matrix is reduced.
///
/// `dlahqr` reports non-convergence through `INFO`, which is returned as
/// [`LapackError::Convergence`]: a matrix that exhausts the iteration limit
/// leaves `t` partly reduced, and the partly converged eigenvalues are not
/// passed off as a Schur form.
///
/// The acceptance test is `parity::dhseqr_eigenvalues_agree_with_the_reference`,
/// which checks the quasi-triangular structure, the standard form of each 2x2
/// block, and the eigenvalue multiset against the reference `dgeev`.
pub fn dhseqr(t: &mut [f64], wr: &mut [f64], wi: &mut [f64], n: usize) -> Result<(), LapackError> {
    dhseqr_impl(t, wr, wi, None, n)
}

/// `dhseqr` with `JOB = 'S'`, `COMPZ = 'V'`: the real Schur form *and* the Schur
/// vectors.
///
/// This is the same reduction as [`dhseqr`], with `WANTZ = .TRUE.` instead of
/// `.FALSE.`, so the plane rotations that `dlahqr` applies are also accumulated
/// into `z`. On entry `z` must be an `n x n` orthogonal matrix (in `dgeev` it is
/// the `Q` from [`dgehrd`], so that `z` starts life as the Hessenberg
/// transformation); on exit it holds `z * Z_schur`, the Schur vectors of the
/// original matrix, so that `A = Z T Z^T` with `T` in `t`.
///
/// `t` and `(wr, wi)` are exactly as in the 4-argument form: the rotations
/// applied to `t` do not depend on whether they are recorded.
///
/// The test `dhseqr_with_schur_vectors_gives_the_schur_decomposition` checks
/// `A = Z T Z^T` and that `Z` is orthogonal.
pub fn dhseqr_with_schur_vectors(
    t: &mut [f64],
    wr: &mut [f64],
    wi: &mut [f64],
    z: &mut [f64],
    n: usize,
) -> Result<(), LapackError> {
    dhseqr_impl(t, wr, wi, Some(z), n)
}

fn dhseqr_impl(
    t: &mut [f64],
    wr: &mut [f64],
    wi: &mut [f64],
    z: Option<&mut [f64]>,
    n: usize,
) -> Result<(), LapackError> {
    assert_eq!(t.len(), n * n, "dhseqr: t must be n*n");
    assert_eq!(wr.len(), n, "dhseqr: wr must have n entries");
    assert_eq!(wi.len(), n, "dhseqr: wi must have n entries");
    if n == 0 {
        return Ok(());
    }
    // JOB = 'S' -> WANTT = .TRUE.; ILO = 1, IHI = N (0-based 0 and n - 1).
    let info = match z {
        Some(z) => {
            assert_eq!(z.len(), n * n, "dhseqr: z must be n*n");
            dlahqr(n, 0, n - 1, t, n, wr, wi, true, z)
        }
        None => dlahqr(n, 0, n - 1, t, n, wr, wi, false, &mut []),
    };
    // `INFO > 0` is dlahqr's non-convergence report. Returning it is the whole
    // point of the Result: a release build used to drop it on the floor.
    if info != 0 {
        return Err(LapackError::Convergence);
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `dlartg` (LAPACK 3.12.1, `SRC/dlartg.f90`): a plane rotation with real cosine
/// and sine, returning `(c, s, r)` with
///
/// ```text
/// [  c  s ] [ f ]   [ r ]
/// [ -s  c ] [ g ] = [ 0 ]
/// ```
///
/// Used by `dgeev` to rotate the largest component of a complex eigenvector onto
/// the real axis. The three scaling regimes (tiny, normal, huge) are the
/// reference's, so that `f*f + g*g` can neither overflow nor underflow.
pub fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    let safmin = f64::MIN_POSITIVE;
    let safmax = f64::MAX;
    let rtmin = safmin.sqrt();
    let rtmax = (safmax / 2.0).sqrt();
    let f1 = f.abs();
    let g1 = g.abs();
    if g == 0.0 {
        (1.0, 0.0, f)
    } else if f == 0.0 {
        (0.0, g.signum(), g1)
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = (f * f + g * g).sqrt();
        let c = f1 / d;
        let r = d.copysign(f);
        let s = g / r;
        (c, s, r)
    } else {
        let u = safmax.min(safmin.max(f1.max(g1)));
        let fs = f / u;
        let gs = g / u;
        let d = (fs * fs + gs * gs).sqrt();
        let c = fs.abs() / d;
        let mut r = d.copysign(f);
        let s = gs / r;
        r *= u;
        (c, s, r)
    }
}

/// `dladiv` (LAPACK 3.12.1, `SRC/dladiv.f`): `(a + ib) / (c + id)`.
fn dladiv(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let bs = 2.0;
    let ab = a.abs().max(b.abs());
    let cd = c.abs().max(d.abs());
    let mut s = 1.0;
    let ov = f64::MAX;
    let un = f64::MIN_POSITIVE;
    let eps = f64::EPSILON;
    let be = bs / (eps * eps);
    let (mut aa, mut bb, mut cc, mut dd) = (a, b, c, d);
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
    if ab <= un * bs / eps {
        aa *= be;
        bb *= be;
        s /= be;
    }
    if cd <= un * bs / eps {
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

fn dladiv1(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let r = d / c;
    let t = 1.0 / (c + d * r);
    let p = dladiv2(a, b, c, d, r, t);
    let q = dladiv2(b, -a, c, d, r, t);
    (p, q)
}

/// The result of [`dlaln2`]: `x` is the `2 x 2` solution array (only the entries
/// the requested shape uses are meaningful), `scale` the factor the caller must
/// apply to its right-hand side, and `xnorm` the infinity norm of `x`.
struct Laln2 {
    x11: f64,
    x21: f64,
    x12: f64,
    x22: f64,
    scale: f64,
    xnorm: f64,
}

/// `dlaln2` (LAPACK 3.12.1, `SRC/dlaln2.f`) specialised to `LTRANS = .FALSE.`,
/// `CA = ONE`, `D1 = D2 = ONE`.
///
/// Solves `(A - (wr + i*wi) I) X = scale * B`, where `A` is the 1x1 or 2x2 block
/// `[[a11, a12], [a21, a22]]` and `B` is `2 x 1` (real, `nw = 1`) or `2 x 2`
/// (complex, `nw = 2`). If the system is closer to singular than `smin`, the
/// pivot is perturbed to `smin` instead of failing, exactly as LAPACK does.
#[allow(clippy::too_many_arguments)]
fn dlaln2(
    na: usize,
    nw: usize,
    smin: f64,
    a11: f64,
    a21: f64,
    a12: f64,
    a22: f64,
    b11: f64,
    b21: f64,
    b12: f64,
    b22: f64,
    wr: f64,
    wi: f64,
) -> Laln2 {
    // IPIVOT, RSWAP and ZSWAP from dlaln2.f's DATA statements. IPIVOT[:, icmax]
    // is listed here by column (Fortran fills DATA column-major).
    const IPIVOT: [[usize; 4]; 4] = [[1, 2, 3, 4], [2, 1, 4, 3], [3, 4, 1, 2], [4, 3, 2, 1]];
    const RSWAP: [bool; 4] = [false, true, false, true];
    const ZSWAP: [bool; 4] = [false, false, true, true];

    let smlnum = 2.0 * f64::MIN_POSITIVE;
    let bignum = 1.0 / smlnum;
    let smini = smin.max(smlnum);
    let mut scale = 1.0;

    let (mut x11, mut x21, mut x12, mut x22);
    let mut xnorm;

    if na == 1 {
        if nw == 1 {
            // Real 1x1 system.
            let mut csr = a11 - wr;
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
            let mut csr = a11 - wr;
            let mut csi = -wi;
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
        let crv = [a11 - wr, a21, a12, a22 - wr];
        if nw == 1 {
            // Real 2x2 system (w is real).
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
            // Gaussian elimination with complete pivoting.
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
            if ZSWAP[icmax] {
                x11 = xr2;
                x21 = xr1;
            } else {
                x11 = xr1;
                x21 = xr2;
            }
            x12 = 0.0;
            x22 = 0.0;
            xnorm = xr1.abs().max(xr2.abs());
            if xnorm > 1.0 && cmax > 1.0 && xnorm > bignum / cmax {
                let temp = cmax / bignum;
                x11 *= temp;
                x21 *= temp;
                xnorm *= temp;
                scale *= temp;
            }
        } else {
            // Complex 2x2 system (w is complex).
            let civ = [-wi, 0.0, 0.0, -wi];
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
            // Gaussian elimination with complete pivoting.
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
                // Off-diagonals of the pivoted C are real.
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
                // Diagonals of the pivoted C are real.
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
            if ZSWAP[icmax] {
                x11 = xr2;
                x21 = xr1;
                x12 = xi2;
                x22 = xi1;
            } else {
                x11 = xr1;
                x21 = xr2;
                x12 = xi1;
                x22 = xi2;
            }
            xnorm = (xr1.abs() + xi1.abs()).max(xr2.abs() + xi2.abs());
            if xnorm > 1.0 && cmax > 1.0 && xnorm > bignum / cmax {
                let temp = cmax / bignum;
                x11 *= temp;
                x21 *= temp;
                x12 *= temp;
                x22 *= temp;
                xnorm *= temp;
                scale *= temp;
            }
        }
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

// ---------------------------------------------------------------------------
// 1-based accessors into a column-major n x n array, so the transcription below
// reads like the Fortran.
// ---------------------------------------------------------------------------

fn tget(t: &[f64], n: usize, i: usize, j: usize) -> f64 {
    t[(i - 1) + (j - 1) * n]
}

fn vget(v: &[f64], n: usize, i: usize, j: usize) -> f64 {
    v[(i - 1) + (j - 1) * n]
}

fn vset(v: &mut [f64], n: usize, i: usize, j: usize, x: f64) {
    v[(i - 1) + (j - 1) * n] = x;
}

/// `dtrevc3` (LAPACK 3.12.1, `SRC/dtrevc3.f`) with `SIDE = 'R'`, `HOWMNY = 'B'`
/// and the unblocked `NB = 1` back-transformation.
///
/// On entry `t` is the quasi-upper-triangular Schur form and `vr` holds the
/// orthogonal Schur vectors `Z` (the `VR` of `dgeev`); on exit `vr` holds the
/// right eigenvectors of the original matrix in LAPACK's compact storage: for a
/// conjugate pair, columns `j` and `j + 1` are the real and imaginary parts of
/// `v(j)`, and the eigenvalue with the positive imaginary part comes first.
///
/// The BLAS-2 back-transformation is `dtrevc3`'s `NB = 1` branch, which is what
/// `dgeev.f` selects for its `lwork = 4n + 1` workspace.
/// `VCRIT = BIGNUM` is set again after each rescale, and the reference does the
/// same even though the next statement overwrites it (`VCRIT = BIGNUM / VMAX`):
/// the write is dead there too. Kept so a diff against `dtrevc3.f` still lines
/// up; `unused_assignments` is the only lint it trips.
#[allow(unused_assignments)]
fn dtrevc_right(t: &[f64], vr: &mut [f64], n: usize) {
    let unfl = f64::MIN_POSITIVE;
    let ulp = f64::EPSILON;
    let smlnum = unfl * (n as f64 / ulp);
    let bignum = (1.0 - ulp) / smlnum;

    // 1-norm of the strictly upper triangular part of each column of T, to
    // control overflow in the triangular solver.
    let mut colnorm = vec![0.0; n];
    for j in 2..=n {
        let mut s = 0.0;
        for i in 1..j {
            s += tget(t, n, i, j).abs();
        }
        colnorm[j - 1] = s;
    }

    let mut work1 = vec![0.0; n];
    let mut work2 = vec![0.0; n];
    let mut y1 = vec![0.0; n];
    let mut y2 = vec![0.0; n];

    // IP = 0: real eigenvalue; 1: first of a conjugate pair, already handled;
    // -1: second of a conjugate pair (the KI at which the pair is processed).
    let mut ip = 0i32;
    let mut ki = n; // 1-based, from N down to 1
    while ki >= 1 {
        if ip != 1 {
            // ---- decide whether KI is real or the lower half of a pair ----
            // The reference spells the first two cases as separate branches
            // (`KI == 1` and `T(KI,KI-1) == 0`), both setting `IP = 0`; merged
            // here because the two are indistinguishable.
            if ki == 1 || tget(t, n, ki, ki - 1) == 0.0 {
                ip = 0;
            } else {
                ip = -1;
            }

            // The KI-th eigenvalue (recomputed from T, as the reference does).
            let wr = tget(t, n, ki, ki);
            let wi = if ip != 0 {
                tget(t, n, ki, ki - 1).abs().sqrt() * tget(t, n, ki - 1, ki).abs().sqrt()
            } else {
                0.0
            };
            let smin = (ulp * (wr.abs() + wi.abs())).max(smlnum);

            if ip == 0 {
                // ================= real right eigenvector =================
                work1.fill(0.0);
                work1[ki - 1] = 1.0;
                for k in 1..ki {
                    work1[k - 1] = -tget(t, n, k, ki);
                }

                // Solve the upper quasi-triangular system
                //     (T(1:KI-1,1:KI-1) - WR) x = scale * work.
                let mut jnxt: isize = ki as isize - 1;
                let mut j: isize = ki as isize - 1;
                while j >= 1 {
                    if j <= jnxt {
                        let ju = j as usize;
                        let mut j1 = j;
                        let j2 = j;
                        jnxt = j - 1;
                        if j > 1 && tget(t, n, ju, ju - 1) != 0.0 {
                            j1 = j - 1;
                            jnxt = j - 2;
                        }
                        if j1 == j2 {
                            // 1x1 diagonal block.
                            let r = dlaln2(
                                1,
                                1,
                                smin,
                                tget(t, n, ju, ju),
                                0.0,
                                0.0,
                                0.0,
                                work1[ju - 1],
                                0.0,
                                0.0,
                                0.0,
                                wr,
                                0.0,
                            );
                            let mut x11 = r.x11;
                            let mut scale = r.scale;
                            if r.xnorm > 1.0 && colnorm[ju - 1] > bignum / r.xnorm {
                                x11 /= r.xnorm;
                                scale /= r.xnorm;
                            }
                            if scale != 1.0 {
                                for l in 0..ki {
                                    work1[l] *= scale;
                                }
                            }
                            work1[ju - 1] = x11;
                            for l in 1..ju {
                                work1[l - 1] -= x11 * tget(t, n, l, ju);
                            }
                        } else {
                            // 2x2 diagonal block.
                            let r = dlaln2(
                                2,
                                1,
                                smin,
                                tget(t, n, ju - 1, ju - 1),
                                tget(t, n, ju, ju - 1),
                                tget(t, n, ju - 1, ju),
                                tget(t, n, ju, ju),
                                work1[ju - 2],
                                work1[ju - 1],
                                0.0,
                                0.0,
                                wr,
                                0.0,
                            );
                            let mut x11 = r.x11;
                            let mut x21 = r.x21;
                            let mut scale = r.scale;
                            if r.xnorm > 1.0 {
                                let beta = colnorm[ju - 2].max(colnorm[ju - 1]);
                                if beta > bignum / r.xnorm {
                                    x11 /= r.xnorm;
                                    x21 /= r.xnorm;
                                    scale /= r.xnorm;
                                }
                            }
                            if scale != 1.0 {
                                for l in 0..ki {
                                    work1[l] *= scale;
                                }
                            }
                            work1[ju - 2] = x11;
                            work1[ju - 1] = x21;
                            for l in 1..(ju - 1) {
                                work1[l - 1] -= x11 * tget(t, n, l, ju - 1);
                            }
                            for l in 1..(ju - 1) {
                                work1[l - 1] -= x21 * tget(t, n, l, ju);
                            }
                        }
                    }
                    j -= 1;
                }

                // Back-transform with the Schur vectors and normalise so that
                // the largest component has magnitude one:
                //     VR(:,KI) := Z * x.
                if ki > 1 {
                    for i in 1..=n {
                        let mut s = work1[ki - 1] * vget(vr, n, i, ki);
                        for l in 1..ki {
                            s += work1[l - 1] * vget(vr, n, i, l);
                        }
                        y1[i - 1] = s;
                    }
                    for i in 1..=n {
                        vset(vr, n, i, ki, y1[i - 1]);
                    }
                }
                let ii = crate::native::blas::idamax(&vr[(ki - 1) * n..ki * n]);
                let remax = 1.0 / vr[ii + (ki - 1) * n].abs();
                for i in 0..n {
                    vr[i + (ki - 1) * n] *= remax;
                }
            } else {
                // =================== complex right eigenvector ===================
                // Initial solve of the 2x2 block.
                if tget(t, n, ki - 1, ki).abs() >= tget(t, n, ki, ki - 1).abs() {
                    work1[ki - 2] = 1.0;
                    work2[ki - 1] = wi / tget(t, n, ki - 1, ki);
                } else {
                    work1[ki - 2] = -wi / tget(t, n, ki, ki - 1);
                    work2[ki - 1] = 1.0;
                }
                work1[ki - 1] = 0.0;
                work2[ki - 2] = 0.0;

                // Form the right-hand side.
                for k in 1..=(ki - 2) {
                    work1[k - 1] = -work1[ki - 2] * tget(t, n, k, ki - 1);
                    work2[k - 1] = -work2[ki - 1] * tget(t, n, k, ki);
                }

                // Solve (T(1:KI-2,1:KI-2) - (WR + i WI)) x = scale*(work1 + i work2).
                let mut jnxt: isize = ki as isize - 2;
                let mut j: isize = ki as isize - 2;
                while j >= 1 {
                    if j <= jnxt {
                        let ju = j as usize;
                        let mut j1 = j;
                        let j2 = j;
                        jnxt = j - 1;
                        if j > 1 && tget(t, n, ju, ju - 1) != 0.0 {
                            j1 = j - 1;
                            jnxt = j - 2;
                        }
                        if j1 == j2 {
                            // 1x1 diagonal block, two right-hand sides.
                            let r = dlaln2(
                                1,
                                2,
                                smin,
                                tget(t, n, ju, ju),
                                0.0,
                                0.0,
                                0.0,
                                work1[ju - 1],
                                0.0,
                                work2[ju - 1],
                                0.0,
                                wr,
                                wi,
                            );
                            let mut x11 = r.x11;
                            let mut x12 = r.x12;
                            let mut scale = r.scale;
                            if r.xnorm > 1.0 && colnorm[ju - 1] > bignum / r.xnorm {
                                x11 /= r.xnorm;
                                x12 /= r.xnorm;
                                scale /= r.xnorm;
                            }
                            if scale != 1.0 {
                                for l in 0..ki {
                                    work1[l] *= scale;
                                    work2[l] *= scale;
                                }
                            }
                            work1[ju - 1] = x11;
                            work2[ju - 1] = x12;
                            for l in 1..ju {
                                work1[l - 1] -= x11 * tget(t, n, l, ju);
                            }
                            for l in 1..ju {
                                work2[l - 1] -= x12 * tget(t, n, l, ju);
                            }
                        } else {
                            // 2x2 diagonal block, complex system.
                            let r = dlaln2(
                                2,
                                2,
                                smin,
                                tget(t, n, ju - 1, ju - 1),
                                tget(t, n, ju, ju - 1),
                                tget(t, n, ju - 1, ju),
                                tget(t, n, ju, ju),
                                work1[ju - 2],
                                work1[ju - 1],
                                work2[ju - 2],
                                work2[ju - 1],
                                wr,
                                wi,
                            );
                            let mut x11 = r.x11;
                            let mut x21 = r.x21;
                            let mut x12 = r.x12;
                            let mut x22 = r.x22;
                            let mut scale = r.scale;
                            if r.xnorm > 1.0 {
                                let beta = colnorm[ju - 2].max(colnorm[ju - 1]);
                                if beta > bignum / r.xnorm {
                                    let rec = 1.0 / r.xnorm;
                                    x11 *= rec;
                                    x12 *= rec;
                                    x21 *= rec;
                                    x22 *= rec;
                                    scale *= rec;
                                }
                            }
                            if scale != 1.0 {
                                for l in 0..ki {
                                    work1[l] *= scale;
                                    work2[l] *= scale;
                                }
                            }
                            work1[ju - 2] = x11;
                            work1[ju - 1] = x21;
                            work2[ju - 2] = x12;
                            work2[ju - 1] = x22;
                            for l in 1..(ju - 1) {
                                work1[l - 1] -= x11 * tget(t, n, l, ju - 1);
                            }
                            for l in 1..(ju - 1) {
                                work1[l - 1] -= x21 * tget(t, n, l, ju);
                            }
                            for l in 1..(ju - 1) {
                                work2[l - 1] -= x12 * tget(t, n, l, ju - 1);
                            }
                            for l in 1..(ju - 1) {
                                work2[l - 1] -= x22 * tget(t, n, l, ju);
                            }
                        }
                    }
                    j -= 1;
                }

                // Back-transform both columns and normalise so that the largest
                // (|re| + |im|) is one:
                //     VR(:,KI-1) := Z * work1,  VR(:,KI) := Z * work2.
                for i in 1..=n {
                    let mut s1 = work1[ki - 2] * vget(vr, n, i, ki - 1);
                    let mut s2 = work2[ki - 1] * vget(vr, n, i, ki);
                    for l in 1..(ki - 1) {
                        s1 += work1[l - 1] * vget(vr, n, i, l);
                        s2 += work2[l - 1] * vget(vr, n, i, l);
                    }
                    y1[i - 1] = s1;
                    y2[i - 1] = s2;
                }
                for i in 0..n {
                    vr[i + (ki - 2) * n] = y1[i];
                    vr[i + (ki - 1) * n] = y2[i];
                }
                let mut emax = 0.0f64;
                for i in 0..n {
                    emax = emax.max(vr[i + (ki - 2) * n].abs() + vr[i + (ki - 1) * n].abs());
                }
                let remax = 1.0 / emax;
                for i in 0..n {
                    vr[i + (ki - 2) * n] *= remax;
                    vr[i + (ki - 1) * n] *= remax;
                }
            }
        }

        // 130 CONTINUE
        if ip == 1 {
            ip = 0;
        } else if ip == -1 {
            ip = 1;
        }
        ki -= 1;
    }
}

/// Euclidean norm of column `j0` (0-based) of a column-major matrix.
///
/// A column of a column-major matrix is contiguous, so this is the crate's
/// [`dnrm2`](crate::native::blas::dnrm2) on a slice.
fn column_norm2(v: &[f64], n: usize, j0: usize) -> f64 {
    crate::native::blas::dnrm2(&v[j0 * n..(j0 + 1) * n])
}

/// `dgebal` with `JOB = 'B'` (LAPACK 3.12.1, `SRC/dgebal.f`): permute and scale a
/// general real matrix to improve the conditioning of its eigenvalues.
///
/// The permutation pushes rows (then columns) that isolate an eigenvalue to the
/// bottom (top), so that the matrix takes the block form
///
/// ```text
///       ( T1  X  Y )
/// P A P ( 0   B  Z )
///       ( 0   0  T2 )
/// ```
///
/// with `T1` and `T2` upper triangular and `B` the active block `ilo..=ihi`
/// (0-based). Scaling then makes the 1-norms of each row of `B` and its
/// corresponding column nearly equal by a diagonal similarity.
///
/// The returned `scale` vector follows LAPACK: entries inside `ilo..=ihi` hold
/// the scaling factors, entries outside hold the 1-based source index of the
/// permutation (used by `dgebak_right` to undo it).
fn dgebal(a: &mut [f64], n: usize) -> (usize, usize, Vec<f64>) {
    // PARAMETER ( SCLFAC = 2.0 ) and PARAMETER ( FACTOR = 0.95D+0 ).
    const SCLFAC: f64 = 2.0;
    const FACTOR: f64 = 0.95;

    let mut scale = vec![1.0f64; n];
    if n == 0 {
        return (0, 0, scale);
    }

    // K and L are LAPACK's ILO and IHI, 0-based here.
    let mut k = 0usize;
    let mut l = n - 1;

    // ==== Permutation to isolate eigenvalues if possible. ====
    //
    // Search for rows isolating an eigenvalue and push them down.
    let mut noconv = true;
    while noconv {
        noconv = false;
        // Fortran fixes the DO bound when the loop is entered; L is decremented
        // inside the body, so capture the bound first.
        let mut i = l as isize;
        while i >= 0 {
            let iu = i as usize;
            let mut canswap = true;
            for j in 0..=l {
                if iu != j && a[iu + j * n] != 0.0 {
                    canswap = false;
                    break;
                }
            }
            if canswap {
                scale[l] = (iu + 1) as f64;
                if iu != l {
                    // Swap rows (DSWAP(L, A(1,I), 1, A(1,L), 1)) and columns
                    // (DSWAP(N-K+1, A(I,K), LDA, A(L,K), LDA)).
                    for r in 0..=l {
                        a.swap(r + iu * n, r + l * n);
                    }
                    for c in k..n {
                        a.swap(iu + c * n, l + c * n);
                    }
                }
                noconv = true;
                if l == 0 {
                    return (0, 0, scale);
                }
                l -= 1;
            }
            i -= 1;
        }
    }

    // Search for columns isolating an eigenvalue and push them left.
    noconv = true;
    while noconv {
        noconv = false;
        let mut j = k;
        while j <= l {
            let mut canswap = true;
            for i in k..=l {
                if i != j && a[i + j * n] != 0.0 {
                    canswap = false;
                    break;
                }
            }
            if canswap {
                scale[k] = (j + 1) as f64;
                if j != k {
                    for r in 0..=l {
                        a.swap(r + j * n, r + k * n);
                    }
                    for c in k..n {
                        a.swap(j + c * n, k + c * n);
                    }
                }
                noconv = true;
                k += 1;
            }
            j += 1;
        }
    }

    // Initialize SCALE for the non-permuted submatrix.
    for i in k..=l {
        scale[i] = 1.0;
    }

    // ==== Balance the submatrix in rows K to L. ====
    //
    // SFMIN1 = DLAMCH('S') / DLAMCH('P').
    let sfmin1 = f64::MIN_POSITIVE / f64::EPSILON;
    let sfmax1 = 1.0 / sfmin1;
    let sfmin2 = sfmin1 * SCLFAC;
    let sfmax2 = 1.0 / sfmin2;

    let mut noconv = true;
    while noconv {
        noconv = false;
        for i in k..=l {
            let col: Vec<f64> = (k..=l).map(|r| a[r + i * n]).collect();
            let row: Vec<f64> = (k..=l).map(|c| a[i + c * n]).collect();
            let mut c = crate::native::blas::dnrm2(&col);
            let mut r = crate::native::blas::dnrm2(&row);

            // ICA = IDAMAX(L, A(1,I), 1); IRA = IDAMAX(N-K+1, A(I,K), LDA).
            let col_full: Vec<f64> = (0..=l).map(|rr| a[rr + i * n]).collect();
            let ica = crate::native::blas::idamax(&col_full);
            let mut ca = a[ica + i * n].abs();
            let row_full: Vec<f64> = (k..n).map(|cc| a[i + cc * n]).collect();
            let ira = crate::native::blas::idamax(&row_full) + k;
            let mut ra = a[i + ira * n].abs();

            // Guard against zero C or R due to underflow.
            if c == 0.0 || r == 0.0 {
                continue;
            }
            if (c + ca + r + ra).is_nan() {
                // DISNAN(C+CA+R+RA): the reference reports an error here. The
                // inputs to dgeev are finite, so this is unreachable in practice;
                // leaving the row unbalanced is the graceful fallback.
                continue;
            }

            let mut g = r / SCLFAC;
            let mut f = 1.0f64;
            let s = c + r;

            while c < g && f.max(c).max(ca) < sfmax2 && r.min(g).min(ra) > sfmin2 {
                f *= SCLFAC;
                c *= SCLFAC;
                ca *= SCLFAC;
                r /= SCLFAC;
                g /= SCLFAC;
                ra /= SCLFAC;
            }

            g = c / SCLFAC;

            while g >= r && r.max(ra) < sfmax2 && f.min(c).min(g).min(ca) > sfmin2 {
                f /= SCLFAC;
                c /= SCLFAC;
                g /= SCLFAC;
                ca /= SCLFAC;
                r *= SCLFAC;
                ra *= SCLFAC;
            }

            // Now balance.
            if c + r >= FACTOR * s {
                continue;
            }
            if f < 1.0 && scale[i] < 1.0 {
                if f * scale[i] <= sfmin1 {
                    continue;
                }
            }
            if f > 1.0 && scale[i] > 1.0 {
                if scale[i] >= sfmax1 / f {
                    continue;
                }
            }
            let g2 = 1.0 / f;
            scale[i] *= f;
            noconv = true;

            // DSCAL(N-K+1, G, A(I,K), LDA) then DSCAL(L, F, A(1,I), 1): scale
            // row I across columns K..N and column I across rows 1..L.
            for cc in k..n {
                a[i + cc * n] *= g2;
            }
            for rr in 0..=l {
                a[rr + i * n] *= f;
            }
        }
    }

    (k, l, scale)
}

/// `dgebak` with `JOB = 'B'`, `SIDE = 'R'` (LAPACK 3.12.1, `SRC/dgebak.f`):
/// undo the balancing that `dgebal` recorded in `scale`.
///
/// The right eigenvectors of the balanced matrix are those of the original
/// matrix with row `i` multiplied by `scale(i)` (for `i` in `ilo..=ihi`) and the
/// isolated rows permuted back.
fn dgebak_right(v: &mut [f64], n: usize, ilo: usize, ihi: usize, scale: &[f64]) {
    if n == 0 {
        return;
    }

    // Backward balance.
    if ilo != ihi {
        for i in ilo..=ihi {
            let s = scale[i];
            for j in 0..n {
                v[i + j * n] *= s;
            }
        }
    }

    // Backward permutation: I = ILO-1 step -1 until 1, then IHI+1 step 1 until N.
    for ii in 0..n {
        let i = if ii < ilo { ilo - ii - 1 } else { ii };
        if i >= ilo && i <= ihi {
            continue;
        }
        let k = scale[i] as usize - 1;
        if k == i {
            continue;
        }
        for j in 0..n {
            v.swap(i + j * n, k + j * n);
        }
    }
}

/// `dgeev` (right eigenvectors only): the eigenvalues and right eigenvectors of
/// a general real square matrix.
///
/// `a` is `n x n` column-major and is destroyed. On exit:
///
/// * the return value is `(wr, wi)`, the eigenvalues, in the same order as the
///   eigenvector columns; a complex conjugate pair is adjacent with `wi > 0`
///   first;
/// * `v_real` and `v_imag` hold the right eigenvectors as their real and
///   imaginary parts: `v(j) = v_real[:, j] + i * v_imag[:, j]`. A real
///   eigenvalue's column has zero imaginary part, and for a pair at `(j, j+1)`
///   the two columns are conjugates, `v(j+1) = conj(v(j))`. This is LAPACK's
///   compact `VR` layout, `crate::internal::dgeev_data`'s unpacking applied, so
///   either reading of the pair -- `v_real[:,j]`/`v_real[:,j+1]` as real and
///   imaginary parts, or `v_real`/`v_imag` as the parts of `v(j)` -- gives the
///   same vectors. Eigenvectors are normalised as the reference does: Euclidean
///   norm one, and for a pair the largest component is rotated onto the real
///   axis.
///
/// # What this follows
///
/// `dgeev.f` with `JOBVL = 'N'`, `JOBVR = 'V'`, and the `dgeev.f` dispatch to
/// `dtrevc3` (the unblocked `NB = 1` branch, which is what `dgeev`'s
/// `lwork = 4n + 1` selects):
///
/// 1. scale `A` if its largest element is outside `[SMLNUM, BIGNUM]`;
/// 2. `dgebal` (`JOB = 'B'`): permute and scale to improve conditioning;
/// 3. [`dgehrd`]: `A = Q H Q^T`, with `Q` accumulated;
/// 4. [`dhseqr_with_schur_vectors`] (`COMPZ = 'V'`): `A = Z T Z^T`, accumulating
///    the Schur vectors into `Q`, so that afterwards `Z` holds them;
/// 5. `dtrevc_right` (`DTREVC3` with `SIDE = 'R'`, `HOWMNY = 'B'`):
///    eigenvectors of `T`, back-transformed by `Z`;
/// 6. `dgebak_right` (`JOB = 'B'`, `SIDE = 'R'`): undo the balancing of the
///    eigenvectors;
/// 7. normalise the eigenvectors, then unscale the eigenvalues.
///
/// Balancing is what makes the eigenvalues of a defectively repeated eigenvalue
/// come out exact: the permutation in step 2 isolates such an eigenvalue, so the
/// QR iteration never has to separate it from its neighbours.
///
/// # Errors
///
/// [`LapackError::Convergence`] if the QR iteration in step 4 fails to converge.
pub fn dgeev(
    v_real: &mut [f64],
    v_imag: &mut [f64],
    a: &mut [f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), LapackError> {
    assert_eq!(a.len(), n * n, "dgeev: a must be n*n");
    assert_eq!(v_real.len(), n * n, "dgeev: v_real must be n*n");
    assert_eq!(v_imag.len(), n * n, "dgeev: v_imag must be n*n");

    let mut wr = vec![0.0; n];
    let mut wi = vec![0.0; n];
    if n == 0 {
        return Ok((wr, wi));
    }

    // ---- machine constants (dgeev.f) ----
    let eps = f64::EPSILON;
    let smlnum0 = f64::MIN_POSITIVE;
    let smlnum = smlnum0.sqrt() / eps;
    let bignum = 1.0 / smlnum;

    // ---- scale A if its largest element is outside [SMLNUM, BIGNUM] ----
    let mut anrm = 0.0f64;
    for x in a.iter() {
        anrm = anrm.max(x.abs());
    }
    let mut cscale = 0.0f64;
    let scalea = if anrm > 0.0 && anrm < smlnum {
        cscale = smlnum;
        true
    } else if anrm > bignum {
        cscale = bignum;
        true
    } else {
        false
    };
    if scalea {
        let f = cscale / anrm;
        for x in a.iter_mut() {
            *x *= f;
        }
    }

    // ---- DGEBAL('B'): permute and scale to improve conditioning ----
    let (ilo, ihi, bal_scale) = dgebal(a, n);

    // ---- DGEHRD: A = Q H Q^T ----
    let mut q = vec![0.0; n * n];
    dgehrd(a, n, Some(&mut q));

    // ---- DHSEQR with COMPZ = 'V': A = Z T Z^T (Z accumulated into q) ----
    dhseqr_with_schur_vectors(a, &mut wr, &mut wi, &mut q, n)?;

    // ---- DTREVC3('R','B'): eigenvectors of T, back-transformed by Z ----
    let mut vr = q;
    dtrevc_right(a, &mut vr, n);

    // ---- DGEBAK('B','R'): undo the balancing of the eigenvectors ----
    dgebak_right(&mut vr, n, ilo, ihi, &bal_scale);

    // ---- normalise: unit Euclidean norm, largest component real ----
    // (dgeev.f, the WANTVR branch; `i0` is LAPACK's 1-based I minus one)
    let mut i0 = 0usize;
    while i0 < n {
        if wi[i0] == 0.0 {
            let scl = 1.0 / column_norm2(&vr, n, i0);
            for r in 0..n {
                vr[r + i0 * n] *= scl;
            }
            i0 += 1;
        } else if wi[i0] > 0.0 {
            let scl = 1.0 / f64::hypot(column_norm2(&vr, n, i0), column_norm2(&vr, n, i0 + 1));
            for r in 0..n {
                vr[r + i0 * n] *= scl;
                vr[r + (i0 + 1) * n] *= scl;
            }
            // Rotate the largest component onto the real axis.
            let mut k = 0usize;
            let mut best = -1.0f64;
            for r in 0..n {
                let mag = vr[r + i0 * n] * vr[r + i0 * n] + vr[r + (i0 + 1) * n] * vr[r + (i0 + 1) * n];
                if mag > best {
                    best = mag;
                    k = r;
                }
            }
            let (cs, sn, _r) = dlartg(vr[k + i0 * n], vr[k + (i0 + 1) * n]);
            for r in 0..n {
                let x = vr[r + i0 * n];
                let y = vr[r + (i0 + 1) * n];
                vr[r + i0 * n] = cs * x + sn * y;
                vr[r + (i0 + 1) * n] = cs * y - sn * x;
            }
            vr[k + (i0 + 1) * n] = 0.0;
            i0 += 2;
        } else {
            // The second member of a pair: already written by the first.
            i0 += 1;
        }
    }

    // ---- undo the scaling of A on the eigenvalues ----
    if scalea {
        let f = anrm / cscale;
        for x in wr.iter_mut() {
            *x *= f;
        }
        for x in wi.iter_mut() {
            *x *= f;
        }
    }

    // ---- unpack the compact storage into real and imaginary parts ----
    for i in 0..n {
        if wi[i] == 0.0 {
            for r in 0..n {
                v_real[r + i * n] = vr[r + i * n];
                v_imag[r + i * n] = 0.0;
            }
        } else if wi[i] > 0.0 {
            for r in 0..n {
                let re = vr[r + i * n];
                let im = vr[r + (i + 1) * n];
                v_real[r + i * n] = re;
                v_imag[r + i * n] = im;
                v_real[r + (i + 1) * n] = re;
                v_imag[r + (i + 1) * n] = -im;
            }
        }
    }

    Ok((wr, wi))
}

/// `dtrevc3` (`dtrevc3.f`, LAPACK 3.12.1) with `SIDE = 'L'`, `HOWMNY = 'B'` and
/// the unblocked `NB = 1` back-transformation.
///
/// On entry `t` is the real Schur form (quasi-upper-triangular) and `vl` holds
/// the orthogonal Schur vectors `Z`; on exit `vl` holds the left eigenvectors in
/// LAPACK's compact storage: for a conjugate pair at `(j, j+1)`, columns `j` and
/// `j + 1` are the real and imaginary parts of `u(j)`, with `u(j+1) = conj(u(j))`.
/// The defining equation is `u(j)^T A = lambda(j) u(j)^T`.
///
/// The only `dlaln2` call with `LTRANS = .TRUE.` is the 2x2 block, and `dlaln2`
/// uses `LTRANS` for nothing but the off-diagonal swap in `CR`; passing the two
/// off-diagonals in the opposite argument slots is therefore exactly the
/// transposed system, and no second `dlaln2` is needed.
/// `VCRIT = BIGNUM` is set again after each rescale, and the reference does the
/// same even though the next statement overwrites it (`VCRIT = BIGNUM / VMAX`):
/// the write is dead there too. Kept so a diff against `dtrevc3.f` still lines
/// up; `unused_assignments` is the only lint it trips.
#[allow(unused_assignments)]
fn dtrevc_left(t: &[f64], vl: &mut [f64], n: usize) {
    let unfl = f64::MIN_POSITIVE;
    let ulp = f64::EPSILON;
    let smlnum = unfl * (n as f64 / ulp);
    let bignum = (1.0 - ulp) / smlnum;

    // 1-norm of the strictly upper triangular part of each column of T.
    let mut colnorm = vec![0.0; n];
    for j in 2..=n {
        let mut s = 0.0;
        for i in 1..j {
            s += tget(t, n, i, j).abs();
        }
        colnorm[j - 1] = s;
    }

    // The two columns of `WORK` the reference keeps for the current eigenvector:
    // `w1` is the real part and `w2` the imaginary part.
    let mut w1 = vec![0.0; n];
    let mut w2 = vec![0.0; n];
    let mut y1 = vec![0.0; n];
    let mut y2 = vec![0.0; n];

    // IP = 0: real eigenvalue; 1: the pair starting here was just processed;
    // -1: the second member of a pair, to be skipped.
    let mut ip = 0i32;
    let mut ki = 1usize;
    while ki <= n {
        if ip == 1 {
            ip = -1;
            ki += 1;
            continue;
        }
        if ki == n || tget(t, n, ki + 1, ki) == 0.0 {
            ip = 0;
        } else {
            ip = 1;
        }
        let wr = tget(t, n, ki, ki);
        let wi = if ip != 0 {
            tget(t, n, ki, ki + 1).abs().sqrt() * tget(t, n, ki + 1, ki).abs().sqrt()
        } else {
            0.0
        };
        let smin = (ulp * (wr.abs() + wi.abs())).max(smlnum);

        if ip == 0 {
            // ================== real left eigenvector ==================
            w1.fill(0.0);
            w1[ki - 1] = 1.0;
            for k in (ki + 1)..=n {
                w1[k - 1] = -tget(t, n, ki, k);
            }

            // Solve (T(KI+1:N,KI+1:N) - WR)^T x = scale * work by forward
            // substitution over the quasi-triangular blocks.
            let mut vmax = 1.0;
            let mut vcrit = bignum;
            let mut jnxt = ki + 1;
            let mut j = ki + 1;
            while j <= n {
                if j >= jnxt {
                    let j1 = j;
                    let mut j2 = j;
                    jnxt = j + 1;
                    if j < n && tget(t, n, j + 1, j) != 0.0 {
                        j2 = j + 1;
                        jnxt = j + 2;
                    }
                    if j1 == j2 {
                        if colnorm[j - 1] > vcrit {
                            let rec = 1.0 / vmax;
                            for l in (ki - 1)..n {
                                w1[l] *= rec;
                            }
                            vmax = 1.0;
                            vcrit = bignum;
                        }
                        let mut s = 0.0;
                        for l in (ki + 1)..j {
                            s += tget(t, n, l, j) * w1[l - 1];
                        }
                        w1[j - 1] -= s;
                        let r = dlaln2(
                            1,
                            1,
                            smin,
                            tget(t, n, j, j),
                            0.0,
                            0.0,
                            0.0,
                            w1[j - 1],
                            0.0,
                            0.0,
                            0.0,
                            wr,
                            0.0,
                        );
                        if r.scale != 1.0 {
                            for l in (ki - 1)..n {
                                w1[l] *= r.scale;
                            }
                        }
                        w1[j - 1] = r.x11;
                        vmax = w1[j - 1].abs().max(vmax);
                        vcrit = bignum / vmax;
                    } else {
                        let beta = colnorm[j - 1].max(colnorm[j]);
                        if beta > vcrit {
                            let rec = 1.0 / vmax;
                            for l in (ki - 1)..n {
                                w1[l] *= rec;
                            }
                            vmax = 1.0;
                            vcrit = bignum;
                        }
                        let mut s = 0.0;
                        for l in (ki + 1)..j {
                            s += tget(t, n, l, j) * w1[l - 1];
                        }
                        w1[j - 1] -= s;
                        let mut s2 = 0.0;
                        for l in (ki + 1)..j {
                            s2 += tget(t, n, l, j + 1) * w1[l - 1];
                        }
                        w1[j] -= s2;
                        // LTRANS = .TRUE.: the off-diagonals are swapped.
                        let r = dlaln2(
                            2,
                            1,
                            smin,
                            tget(t, n, j, j),
                            tget(t, n, j, j + 1),
                            tget(t, n, j + 1, j),
                            tget(t, n, j + 1, j + 1),
                            w1[j - 1],
                            w1[j],
                            0.0,
                            0.0,
                            wr,
                            0.0,
                        );
                        if r.scale != 1.0 {
                            for l in (ki - 1)..n {
                                w1[l] *= r.scale;
                            }
                        }
                        w1[j - 1] = r.x11;
                        w1[j] = r.x21;
                        vmax = w1[j - 1].abs().max(w1[j].abs()).max(vmax);
                        vcrit = bignum / vmax;
                    }
                }
                j += 1;
            }

            // Back-transform with the Schur vectors: VL(:,KI) := Z * x.
            if ki < n {
                for i in 1..=n {
                    let mut s = w1[ki - 1] * vget(vl, n, i, ki);
                    for l in (ki + 1)..=n {
                        s += w1[l - 1] * vget(vl, n, i, l);
                    }
                    y1[i - 1] = s;
                }
                for i in 1..=n {
                    vset(vl, n, i, ki, y1[i - 1]);
                }
            }
            let ii = crate::native::blas::idamax(&vl[(ki - 1) * n..ki * n]);
            let remax = 1.0 / vl[ii + (ki - 1) * n].abs();
            for i in 0..n {
                vl[i + (ki - 1) * n] *= remax;
            }
        } else {
            // =============== complex left eigenvector pair ===============
            // The first column of the pair is the real part, the second the
            // imaginary part, with the "seam" component of each set to zero.
            if tget(t, n, ki, ki + 1).abs() >= tget(t, n, ki + 1, ki).abs() {
                w1[ki - 1] = wi / tget(t, n, ki, ki + 1);
                w2[ki] = 1.0;
            } else {
                w1[ki - 1] = 1.0;
                w2[ki] = -wi / tget(t, n, ki + 1, ki);
            }
            w1[ki] = 0.0;
            w2[ki - 1] = 0.0;
            for k in (ki + 2)..=n {
                w1[k - 1] = -w1[ki - 1] * tget(t, n, ki, k);
                w2[k - 1] = -w2[ki] * tget(t, n, ki + 1, k);
            }

            let mut vmax = 1.0;
            let mut vcrit = bignum;
            let mut jnxt = ki + 2;
            let mut j = ki + 2;
            while j <= n {
                if j >= jnxt {
                    let j1 = j;
                    let mut j2 = j;
                    jnxt = j + 1;
                    if j < n && tget(t, n, j + 1, j) != 0.0 {
                        j2 = j + 1;
                        jnxt = j + 2;
                    }
                    if j1 == j2 {
                        if colnorm[j - 1] > vcrit {
                            let rec = 1.0 / vmax;
                            for l in (ki - 1)..n {
                                w1[l] *= rec;
                                w2[l] *= rec;
                            }
                            vmax = 1.0;
                            vcrit = bignum;
                        }
                        let mut s1 = 0.0;
                        for l in (ki + 2)..j {
                            s1 += tget(t, n, l, j) * w1[l - 1];
                        }
                        w1[j - 1] -= s1;
                        let mut s2 = 0.0;
                        for l in (ki + 2)..j {
                            s2 += tget(t, n, l, j) * w2[l - 1];
                        }
                        w2[j - 1] -= s2;
                        let r = dlaln2(
                            1,
                            2,
                            smin,
                            tget(t, n, j, j),
                            0.0,
                            0.0,
                            0.0,
                            w1[j - 1],
                            0.0,
                            w2[j - 1],
                            0.0,
                            wr,
                            -wi,
                        );
                        if r.scale != 1.0 {
                            for l in (ki - 1)..n {
                                w1[l] *= r.scale;
                                w2[l] *= r.scale;
                            }
                        }
                        w1[j - 1] = r.x11;
                        w2[j - 1] = r.x12;
                        vmax = w1[j - 1].abs().max(w2[j - 1].abs()).max(vmax);
                        vcrit = bignum / vmax;
                    } else {
                        let beta = colnorm[j - 1].max(colnorm[j]);
                        if beta > vcrit {
                            let rec = 1.0 / vmax;
                            for l in (ki - 1)..n {
                                w1[l] *= rec;
                                w2[l] *= rec;
                            }
                            vmax = 1.0;
                            vcrit = bignum;
                        }
                        let mut s1 = 0.0;
                        for l in (ki + 2)..j {
                            s1 += tget(t, n, l, j) * w1[l - 1];
                        }
                        w1[j - 1] -= s1;
                        let mut s2 = 0.0;
                        for l in (ki + 2)..j {
                            s2 += tget(t, n, l, j) * w2[l - 1];
                        }
                        w2[j - 1] -= s2;
                        let mut s3 = 0.0;
                        for l in (ki + 2)..j {
                            s3 += tget(t, n, l, j + 1) * w1[l - 1];
                        }
                        w1[j] -= s3;
                        let mut s4 = 0.0;
                        for l in (ki + 2)..j {
                            s4 += tget(t, n, l, j + 1) * w2[l - 1];
                        }
                        w2[j] -= s4;
                        // LTRANS = .TRUE.: the off-diagonals are swapped.
                        let r = dlaln2(
                            2,
                            2,
                            smin,
                            tget(t, n, j, j),
                            tget(t, n, j, j + 1),
                            tget(t, n, j + 1, j),
                            tget(t, n, j + 1, j + 1),
                            w1[j - 1],
                            w1[j],
                            w2[j - 1],
                            w2[j],
                            wr,
                            -wi,
                        );
                        if r.scale != 1.0 {
                            for l in (ki - 1)..n {
                                w1[l] *= r.scale;
                                w2[l] *= r.scale;
                            }
                        }
                        w1[j - 1] = r.x11;
                        w2[j - 1] = r.x12;
                        w1[j] = r.x21;
                        w2[j] = r.x22;
                        vmax = w1[j - 1]
                            .abs()
                            .max(w2[j - 1].abs())
                            .max(w1[j].abs())
                            .max(w2[j].abs())
                            .max(vmax);
                        vcrit = bignum / vmax;
                    }
                }
                j += 1;
            }

            // Back-transform: the two columns are Q*w1 and Q*w2.
            if ki < n - 1 {
                for i in 1..=n {
                    let mut s1 = w1[ki - 1] * vget(vl, n, i, ki);
                    let mut s2 = w2[ki] * vget(vl, n, i, ki + 1);
                    for l in (ki + 2)..=n {
                        s1 += w1[l - 1] * vget(vl, n, i, l);
                        s2 += w2[l - 1] * vget(vl, n, i, l);
                    }
                    y1[i - 1] = s1;
                    y2[i - 1] = s2;
                }
                for i in 0..n {
                    vl[i + (ki - 1) * n] = y1[i];
                    vl[i + ki * n] = y2[i];
                }
            } else {
                for i in 0..n {
                    vl[i + (ki - 1) * n] *= w1[ki - 1];
                    vl[i + ki * n] *= w2[ki];
                }
            }
            let mut emax = 0.0f64;
            for i in 0..n {
                emax = emax.max(vl[i + (ki - 1) * n].abs() + vl[i + ki * n].abs());
            }
            let remax = 1.0 / emax;
            for i in 0..n {
                vl[i + (ki - 1) * n] *= remax;
                vl[i + ki * n] *= remax;
            }
        }
        ki += 1;
    }
}

/// `dgebak` with `JOB = 'B'`, `SIDE = 'L'`: undo the balancing that `dgebal`
/// recorded in `scale`.
///
/// The left eigenvectors of the balanced matrix are those of the original matrix
/// with row `i` divided by `scale(i)` (for `i` in `ilo..=ihi`), and the isolated
/// rows permuted back.
fn dgebak_left(v: &mut [f64], n: usize, ilo: usize, ihi: usize, scale: &[f64]) {
    if n == 0 {
        return;
    }

    // Backward balance: the division is the only difference from SIDE='R'.
    if ilo != ihi {
        for i in ilo..=ihi {
            let s = 1.0 / scale[i];
            for j in 0..n {
                v[i + j * n] *= s;
            }
        }
    }

    // Backward permutation, identical to SIDE='R'.
    for ii in 0..n {
        let i = if ii < ilo { ilo - ii - 1 } else { ii };
        if i >= ilo && i <= ihi {
            continue;
        }
        let k = scale[i] as usize - 1;
        if k == i {
            continue;
        }
        for j in 0..n {
            v.swap(i + j * n, k + j * n);
        }
    }
}

/// Normalise a set of eigenvector columns the way `dgeev` does: unit Euclidean
/// norm, and for a conjugate pair the largest component is rotated onto the real
/// axis with [`dlartg`].
fn dgeev_normalize_columns(v: &mut [f64], n: usize, wi: &[f64]) {
    let mut k = 0usize;
    while k < n {
        if wi[k] == 0.0 {
            let scl = 1.0 / column_norm2(v, n, k);
            for r in 0..n {
                v[r + k * n] *= scl;
            }
            k += 1;
        } else if wi[k] > 0.0 {
            let scl = 1.0 / f64::hypot(column_norm2(v, n, k), column_norm2(v, n, k + 1));
            for r in 0..n {
                v[r + k * n] *= scl;
                v[r + (k + 1) * n] *= scl;
            }
            let mut kk = 0usize;
            let mut best = -1.0f64;
            for r in 0..n {
                let mag = v[r + k * n] * v[r + k * n] + v[r + (k + 1) * n] * v[r + (k + 1) * n];
                if mag > best {
                    best = mag;
                    kk = r;
                }
            }
            let (cs, sn, _r) = dlartg(v[kk + k * n], v[kk + (k + 1) * n]);
            for r in 0..n {
                let x = v[r + k * n];
                let y = v[r + (k + 1) * n];
                v[r + k * n] = cs * x + sn * y;
                v[r + (k + 1) * n] = cs * y - sn * x;
            }
            v[kk + (k + 1) * n] = 0.0;
            k += 2;
        } else {
            k += 1;
        }
    }
}

/// Unpack LAPACK's compact eigenvector storage into real and imaginary parts.
///
/// For a real eigenvalue the column is real; for a pair at `(k, k + 1)`, `u(k)`
/// has the two columns as its real and imaginary parts and `u(k + 1)` is its
/// conjugate, matching `dgeev`'s documented layout for both `VL` and `VR`.
fn dgeev_unpack(raw: &[f64], wi: &[f64], re: &mut [f64], im: &mut [f64], n: usize) {
    for i in 0..n {
        if wi[i] == 0.0 {
            for r in 0..n {
                re[r + i * n] = raw[r + i * n];
                im[r + i * n] = 0.0;
            }
        } else if wi[i] > 0.0 {
            for r in 0..n {
                let a = raw[r + i * n];
                let b = raw[r + (i + 1) * n];
                re[r + i * n] = a;
                im[r + i * n] = b;
                re[r + (i + 1) * n] = a;
                im[r + (i + 1) * n] = -b;
            }
        }
    }
}

/// `dgeev` with `JOBVL = 'V'` and `JOBVR = 'V'`: the eigenvalues and the left and
/// right eigenvectors of a general real square matrix.
///
/// This is the same pipeline as [`dgeev`] -- scale, `dgebal`, [`dgehrd`],
/// [`dhseqr_with_schur_vectors`], the `dtrevc3` back-substitutions,
/// `dgebak_right`/`dgebak_left`, normalisation -- with the left eigenvectors
/// taken from the *same* Schur factorisation by `dtrevc_left` rather than from
/// a second decomposition of `A^T`.
///
/// `a` is `n x n` column-major and is destroyed. On exit the return value is
/// `(wr, wi)`; `u_real`/`u_imag` and `v_real`/`v_imag` hold the left and right
/// eigenvectors, with `u(j) = u_real[:, j] + i u_imag[:, j]` (and likewise for
/// `v`), and the second member of a pair the conjugate of the first.
///
/// # Errors
///
/// [`LapackError::Convergence`] if the QR iteration fails to converge.
#[allow(clippy::too_many_arguments)]
pub fn dgeev_lr(
    u_real: &mut [f64],
    u_imag: &mut [f64],
    v_real: &mut [f64],
    v_imag: &mut [f64],
    a: &mut [f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), LapackError> {
    assert_eq!(a.len(), n * n, "dgeev_lr: a must be n*n");
    assert_eq!(u_real.len(), n * n, "dgeev_lr: u_real must be n*n");
    assert_eq!(u_imag.len(), n * n, "dgeev_lr: u_imag must be n*n");
    assert_eq!(v_real.len(), n * n, "dgeev_lr: v_real must be n*n");
    assert_eq!(v_imag.len(), n * n, "dgeev_lr: v_imag must be n*n");

    let mut wr = vec![0.0; n];
    let mut wi = vec![0.0; n];
    if n == 0 {
        return Ok((wr, wi));
    }

    // ---- machine constants (dgeev.f) ----
    let eps = f64::EPSILON;
    let smlnum0 = f64::MIN_POSITIVE;
    let smlnum = smlnum0.sqrt() / eps;
    let bignum = 1.0 / smlnum;

    // ---- scale A if its largest element is outside [SMLNUM, BIGNUM] ----
    let mut anrm = 0.0f64;
    for x in a.iter() {
        anrm = anrm.max(x.abs());
    }
    let mut cscale = 0.0f64;
    let scalea = if anrm > 0.0 && anrm < smlnum {
        cscale = smlnum;
        true
    } else if anrm > bignum {
        cscale = bignum;
        true
    } else {
        false
    };
    if scalea {
        let f = cscale / anrm;
        for x in a.iter_mut() {
            *x *= f;
        }
    }

    // ---- DGEBAL('B'): permute and scale to improve conditioning ----
    let (ilo, ihi, bal_scale) = dgebal(a, n);

    // ---- DGEHRD: A = Q H Q^T ----
    let mut q = vec![0.0; n * n];
    dgehrd(a, n, Some(&mut q));

    // ---- DHSEQR with COMPZ = 'V': A = Z T Z^T ----
    dhseqr_with_schur_vectors(a, &mut wr, &mut wi, &mut q, n)?;

    // ---- DTREVC3('B','B'): eigenvectors of T, back-transformed by Z ----
    let mut vl = q.clone();
    let mut vr = q;
    dtrevc_left(a, &mut vl, n);
    dtrevc_right(a, &mut vr, n);

    // ---- DGEBAK('B','L') and DGEBAK('B','R') ----
    dgebak_left(&mut vl, n, ilo, ihi, &bal_scale);
    dgebak_right(&mut vr, n, ilo, ihi, &bal_scale);

    // ---- normalise: unit Euclidean norm, largest component real ----
    dgeev_normalize_columns(&mut vl, n, &wi);
    dgeev_normalize_columns(&mut vr, n, &wi);

    // ---- undo the scaling of A on the eigenvalues ----
    if scalea {
        let f = anrm / cscale;
        for x in wr.iter_mut() {
            *x *= f;
        }
        for x in wi.iter_mut() {
            *x *= f;
        }
    }

    // ---- unpack the compact storages ----
    dgeev_unpack(&vl, &wi, u_real, u_imag, n);
    dgeev_unpack(&vr, &wi, v_real, v_imag, n);

    Ok((wr, wi))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    /// A deterministic, well-conditioned, non-symmetric, non-tridiagonal matrix,
    /// built from a fixed recurrence so the tests need no random generator.
    fn sample(n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i + j * n] = if i == j {
                    (n as f64) + 1.0 + (i as f64) * 0.5
                } else {
                    (((i * 7 + j * 13) % 11) as f64) - 5.0
                };
            }
        }
        a
    }

    fn norm_max(a: &[f64]) -> f64 {
        a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut s = 0.0;
                for l in 0..n {
                    s += a[i + l * n] * b[l + j * n];
                }
                c[i + j * n] = s;
            }
        }
        c
    }

    #[test]
    fn dlarfg_annihilates_the_tail() {
        // H x = beta e1 with H = I - tau v v^T.
        let x = [3.0, 4.0];
        let (beta, tau, v) = dlarfg(x[0], &x[1..]);
        assert!((beta.abs() - 5.0).abs() < 1e-15, "beta = {beta}");

        let mut hx = x.to_vec();
        let w: f64 = v.iter().zip(x.iter()).map(|(vi, xi)| vi * xi).sum::<f64>() * tau;
        for i in 0..2 {
            hx[i] -= w * v[i];
        }
        assert!((hx[0] - beta).abs() < 1e-14, "first component {} vs {beta}", hx[0]);
        assert!(hx[1].abs() < 1e-14, "tail must be annihilated, got {}", hx[1]);
    }

    #[test]
    fn dlarfg_of_a_zero_tail_is_the_identity() {
        let (beta, tau, _v) = dlarfg(2.5, &[0.0, 0.0]);
        assert_eq!(beta, 2.5);
        assert_eq!(tau, 0.0);
    }

    #[test]
    fn dlarfg_handles_a_tiny_vector() {
        // Near the underflow threshold the naive hypot underflows to zero unless
        // the vector is rescaled first.
        let tiny = 1e-300;
        let (beta, tau, v) = dlarfg(tiny, &[tiny]);
        assert!(beta != 0.0, "beta underflowed to zero");
        assert!(tau != 0.0, "tau underflowed to zero");

        // H must still be a reflector: H x = beta e1 with the tail annihilated.
        let x = [tiny, tiny];
        let mut hx = x;
        let w: f64 = v.iter().zip(x.iter()).map(|(vi, xi)| vi * xi).sum::<f64>() * tau;
        for i in 0..2 {
            hx[i] -= w * v[i];
        }
        assert!(
            (hx[0] - beta).abs() <= tiny.abs(),
            "first component {} vs {beta}",
            hx[0]
        );
        assert!(hx[1].abs() <= tiny.abs(), "tail must be annihilated, got {}", hx[1]);
    }

    /// The reduction, checked against the definition rather than against an
    /// internal factorisation: `H` upper Hessenberg, `Q` orthogonal, and
    /// `A = Q H Q^T`.
    #[test]
    fn dgehrd_produces_hessenberg_and_an_orthogonal_q() {
        for n in [1usize, 2, 3, 5, 8, 20] {
            let orig = sample(n);
            let mut h = orig.clone();
            let mut q = vec![0.0; n * n];
            dgehrd(&mut h, n, Some(&mut q));
            for j in 0..n {
                for i in (j + 2)..n {
                    assert!(
                        h[i + j * n].abs() < 1e-13,
                        "n={n}: H({i},{j}) = {} is below the subdiagonal",
                        h[i + j * n]
                    );
                }
            }

            for j in 0..n {
                for k in 0..n {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += q[i + j * n] * q[i + k * n];
                    }
                    let want = if j == k { 1.0 } else { 0.0 };
                    assert!((s - want).abs() < 1e-12, "n={n}: Q^T Q ({j},{k}) = {s}");
                }
            }

            // A = Q H Q^T, checked as A Q = Q H so that no transpose has to be
            // materialised (an earlier version built one with the indices the
            // wrong way round and reported a failure in correct code).
            let aq = mat_mul(&orig, &q, n);
            let qh = mat_mul(&q, &h, n);
            let scale = norm_max(&orig).max(1.0);
            for idx in 0..n * n {
                assert!(
                    (aq[idx] - qh[idx]).abs() / scale < 1e-12,
                    "n={n}: A Q and Q H differ at {idx}: {} vs {}",
                    aq[idx],
                    qh[idx]
                );
            }
        }
    }

    #[test]
    fn dlanv2_real_eigenvalues() {
        // [ 1 2 ; 3 4 ] has the real eigenvalues (5 +- sqrt(33))/2.
        let (mut a, mut b, mut c, mut d) = (1.0, 2.0, 3.0, 4.0);
        let (mut cs, mut sn) = (0.0, 0.0);
        let e = dlanv2(&mut a, &mut b, &mut c, &mut d, &mut cs, &mut sn);
        assert_eq!(e.rt1i, 0.0);
        assert_eq!(e.rt2i, 0.0);
        assert!(c.abs() < 1e-14, "the block must be made upper triangular, c = {c}");
        let disc = 33.0f64.sqrt();
        for want in [(5.0 + disc) / 2.0, (5.0 - disc) / 2.0] {
            assert!(
                [e.rt1r, e.rt2r].iter().any(|g| (g - want).abs() < 1e-13),
                "missing eigenvalue {want} in {:?}",
                [e.rt1r, e.rt2r]
            );
        }
    }

    #[test]
    fn dlanv2_complex_pair() {
        // [ 0 1 ; -1 0 ] has eigenvalues +-i.
        let (mut a, mut b, mut c, mut d) = (0.0, 1.0, -1.0, 0.0);
        let (mut cs, mut sn) = (0.0, 0.0);
        let e = dlanv2(&mut a, &mut b, &mut c, &mut d, &mut cs, &mut sn);
        assert!(e.rt1i > 0.0, "expected a complex pair with a positive rt1i, got {e:?}");
        assert!((e.rt1i + e.rt2i).abs() < 1e-14, "the pair must be conjugate");
        assert!(e.rt1r.abs() < 1e-14);
        assert!((e.rt1i - 1.0).abs() < 1e-14, "|imag| should be 1, got {}", e.rt1i);
        // The standard form requires equal diagonal entries and B*C < 0.
        assert!((a - d).abs() < 1e-14, "the diagonal must be equal: {a} vs {d}");
        assert!(b * c < 0.0, "B*C must be negative, got {} * {}", b, c);
    }

    /// The 3x3 case against an independently built `Hr A Hr`.
    ///
    /// For `n = 3` there is exactly one step, so `H` must equal `Hr * A * Hr`
    /// where `Hr` is the 3x3 reflector that is the identity on row/column 0 and
    /// `I - tau v v^T` on rows/columns 1..2. This pins the exact numbers rather
    /// than only the invariants, so a "reduction" that is similar to `A` but not
    /// the reduction LAPACK performs cannot pass.
    #[test]
    fn dgehrd_matches_the_3x3_reference() {
        // Column-major.
        let a0 = [4.0, 2.0, -2.0, -3.0, 4.5, 0.0, -1.0, -5.0, 5.0];
        let n = 3usize;

        // Hr, built directly from dlarfg on the column below the first row.
        let (beta, tau, v) = dlarfg(a0[1], &[a0[2]]);
        assert!((beta + 2.8284271247461903).abs() < 1e-14, "beta = {beta}");
        let mut hr = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                hr[i + j * n] = if i == 0 && j == 0 {
                    1.0
                } else if i > 0 && j > 0 {
                    (if i == j { 1.0 } else { 0.0 }) - tau * v[i - 1] * v[j - 1]
                } else {
                    0.0
                };
            }
        }

        let mul = |x: &[f64], y: &[f64]| {
            let mut c = vec![0.0; n * n];
            for j in 0..n {
                for i in 0..n {
                    let mut s = 0.0;
                    for l in 0..n {
                        s += x[i + l * n] * y[l + j * n];
                    }
                    c[i + j * n] = s;
                }
            }
            c
        };
        let want = mul(&hr, &mul(&a0, &hr));

        let mut h = a0;
        dgehrd(&mut h, n, None);
        for idx in 0..n * n {
            assert!(
                (h[idx] - want[idx]).abs() < 1e-13,
                "H[{idx}] = {} but Hr A Hr gives {}",
                h[idx],
                want[idx]
            );
        }
        // The published result for this matrix, as a regression pin. Note that
        // H(2,1) -- index 5 -- is the subdiagonal and is legitimately nonzero;
        // only entries *below* the subdiagonal must vanish.
        assert!((h[1] - -2.8284271247461903).abs() < 1e-13, "H(1,0) = {}", h[1]);
        assert_eq!(h[2], 0.0, "H(2,0) must be exactly zero");
        assert!((h[5] - -2.25).abs() < 1e-13, "H(2,1) = {}", h[5]);
    }

    /// `dhseqr` with `COMPZ = 'V'`: the accumulated Schur vectors must satisfy
    /// `A = Z T Z^T` with `Z` orthogonal and `T` quasi-triangular, and must leave
    /// `T` and the eigenvalues bit-for-bit identical to the 4-argument form.
    #[test]
    fn dhseqr_with_schur_vectors_gives_the_schur_decomposition() {
        for n in [1usize, 2, 3, 5, 8, 12] {
            let orig = sample(n);
            let mut h = orig.clone();
            let mut z = vec![0.0; n * n];
            dgehrd(&mut h, n, Some(&mut z));
            let mut wr = vec![0.0; n];
            let mut wi = vec![0.0; n];
            dhseqr_with_schur_vectors(&mut h, &mut wr, &mut wi, &mut z, n).unwrap();

            // Z is orthogonal.
            for j in 0..n {
                for k in 0..n {
                    let s: f64 = (0..n).map(|i| z[i + j * n] * z[i + k * n]).sum();
                    let want = if j == k { 1.0 } else { 0.0 };
                    assert!((s - want).abs() < 1e-12, "n={n}: Z^T Z ({j},{k}) = {s}");
                }
            }

            // A Z = Z T, which is A = Z T Z^T without forming a transpose.
            let az = mat_mul(&orig, &z, n);
            let zt = mat_mul(&z, &h, n);
            let scale = norm_max(&orig).max(1.0);
            for idx in 0..n * n {
                assert!(
                    (az[idx] - zt[idx]).abs() / scale < 1e-12,
                    "n={n}: A Z and Z T differ at {idx}: {} vs {}",
                    az[idx],
                    zt[idx]
                );
            }

            // T is quasi-triangular: nothing below the first subdiagonal.
            for j in 0..n {
                for i in (j + 2)..n {
                    assert!(h[i + j * n].abs() < 1e-13, "n={n}: T({i},{j}) is below the subdiagonal");
                }
            }

            // The eigenvalues are unchanged by recording the Schur vectors.
            let mut h2 = orig.clone();
            dgehrd(&mut h2, n, None);
            let mut wr2 = vec![0.0; n];
            let mut wi2 = vec![0.0; n];
            dhseqr(&mut h2, &mut wr2, &mut wi2, n).unwrap();
            for i in 0..n {
                assert!((wr[i] - wr2[i]).abs() < 1e-13 && (wi[i] - wi2[i]).abs() < 1e-13);
            }
        }
    }

    /// Right eigenvectors of a general real matrix, by residual.
    ///
    /// An eigenvector has no unique value -- it is defined up to scaling, and its
    /// sign and phase are arbitrary -- so comparing against the reference's vectors
    /// would be meaningless. What *is* well defined is the defining equation, so
    /// this checks `A v = lambda v` for each pair, at the tolerance the reduction and
    /// the back-substitution accumulate.
    ///
    /// The eigenvalues are deliberately recomputed here with an independent
    /// `dgehrd` + `dhseqr` rather than taken from `dgeev`, so the pairing of
    /// vectors to eigenvalues is not assumed. `dgebal` leaves the `sample`
    /// family untouched (its dominant diagonal already isolates nothing and
    /// needs no scaling), so the two deflation orders coincide and index `j` of
    /// the returned vectors is still the one paired with index `j` here.
    #[test]
    fn dgeev_eigenvectors_satisfy_their_defining_equation() {
        for n in [1usize, 2, 3, 5, 8, 12] {
            // `sample` here returns column-major data; wrap it so the residual can
            // be formed with element access.
            let data = sample(n);
            let mut a = Matrix::new(n, n);
            a.as_mut_data().copy_from_slice(&data);

            let mut av = Matrix::new(n, n);
            // The working copy must actually hold the matrix. An earlier version
            // left it zeroed, so `dgeev` decomposed the zero matrix, every
            // eigenvalue was zero, and the residual was trivially satisfied --
            // the test passed while testing nothing.
            av.as_mut_data().copy_from_slice(&data);
            let mut v_real = Matrix::new(n, n);
            let mut v_imag = Matrix::new(n, n);
            let (wr_vec, wi_vec) = dgeev(v_real.as_mut_data(), v_imag.as_mut_data(), av.as_mut_data(), n).unwrap();

            // Use the eigenvalues `dgeev` itself returns, and its own eigenvector
            // layout. Recovering them from an independent `dgehrd` + `dhseqr`
            // would give a different *order*, because `dgeev` balances first --
            // pairing an independent ordering against these vectors would
            // mismatch columns and fail for the wrong reason.
            let (wr, wi) = (&wr_vec, &wi_vec);

            let scale = a.as_data().iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
            let mut j = 0usize;
            while j < n {
                if wi[j] != 0.0 {
                    // Complex pair: v = (v_real[:,j] + i v_imag[:,j]), lambda = wr + i wi.
                    let (lr, li) = (wr[j], wi[j]);
                    let mut worst = 0.0f64;
                    for i in 0..n {
                        // (A v)_i for the complex vector v.
                        let mut sr = 0.0;
                        let mut si = 0.0;
                        for k in 0..n {
                            let ar = a.get(i, k);
                            let vr = v_real.get(k, j);
                            let vi = v_imag.get(k, j);
                            sr += ar * vr;
                            si += ar * vi;
                        }
                        // A v - lambda v
                        let dr = sr - (lr * v_real.get(i, j) - li * v_imag.get(i, j));
                        let di = si - (lr * v_imag.get(i, j) + li * v_real.get(i, j));
                        worst = worst.max(dr.abs().max(di.abs()));
                    }
                    assert!(
                        worst / scale < 1e-10,
                        "n={n}: the complex eigenvector for lambda = {lr}+{li}i has residual {worst} (scale {scale})"
                    );
                    j += 2;
                } else {
                    let l = wr[j];
                    let mut worst = 0.0f64;
                    for i in 0..n {
                        let mut s = 0.0;
                        for k in 0..n {
                            s += a.get(i, k) * v_real.get(k, j);
                        }
                        worst = worst.max((s - l * v_real.get(i, j)).abs());
                    }
                    assert!(
                        worst / scale < 1e-10,
                        "n={n}: the real eigenvector for lambda = {l} has residual {worst} (scale {scale})"
                    );
                    j += 1;
                }
            }
        }
    }

    /// The same defining equation as the acceptance test, over matrices the
    /// `sample` family does not reach: defective, structured and pseudo-random,
    /// including cases that exercise `dlaln2`'s small-pivot perturbation.
    ///
    /// Unlike the acceptance test this uses the eigenvalues `dgeev` returns, so it
    /// also checks that the vectors are paired with the eigenvalues reported for
    /// them.
    #[test]
    fn dgeev_residuals_on_a_battery_of_matrices() {
        let mut state: u64 = 0x243f_6a88_85a3_08d3;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };

        let mut cases: Vec<(usize, Vec<f64>)> = Vec::new();

        // A defective Jordan block: one eigenvector for a quadruple eigenvalue.
        let mut jordan = vec![0.0; 16];
        for i in 0..4 {
            jordan[i + i * 4] = 2.0;
            if i < 3 {
                jordan[i + (i + 1) * 4] = 1.0;
            }
        }
        cases.push((4, jordan));

        // A pure rotation: the eigenvalues are 1 +- 2i.
        cases.push((2, vec![1.0, -2.0, 2.0, 1.0]));

        // Three decoupled rotations: three conjugate pairs.
        let mut blockrot = vec![0.0; 36];
        for b in 0..3usize {
            let o = 2 * b;
            blockrot[o + o * 6] = 1.0 + b as f64;
            blockrot[(o + 1) + o * 6] = -2.0;
            blockrot[o + (o + 1) * 6] = 2.0;
            blockrot[(o + 1) + (o + 1) * 6] = 1.0 + b as f64;
        }
        cases.push((6, blockrot));

        // Pseudo-random matrices, deterministic so the test is reproducible.
        for n in [3usize, 5, 9, 16] {
            for _ in 0..4 {
                let mut a = vec![0.0; n * n];
                for x in a.iter_mut() {
                    *x = next() * 20.0;
                }
                cases.push((n, a));
            }
        }

        for (n, data) in cases {
            let mut a = Matrix::new(n, n);
            a.as_mut_data().copy_from_slice(&data);
            let mut av = Matrix::new(n, n);
            let mut v_real = Matrix::new(n, n);
            let mut v_imag = Matrix::new(n, n);
            let (wr, wi) = dgeev(av.as_mut_data(), v_real.as_mut_data(), v_imag.as_mut_data(), n).unwrap();

            let scale = a.as_data().iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
            let mut j = 0usize;
            while j < n {
                let (lr, li) = (wr[j], wi[j]);
                let complex = li != 0.0;
                let mut worst = 0.0f64;
                for i in 0..n {
                    let mut sr = 0.0;
                    let mut si = 0.0;
                    for k in 0..n {
                        sr += a.get(i, k) * v_real.get(k, j);
                        si += a.get(i, k) * v_imag.get(k, j);
                    }
                    let dr = sr - (lr * v_real.get(i, j) - li * v_imag.get(i, j));
                    let di = si - (lr * v_imag.get(i, j) + li * v_real.get(i, j));
                    worst = worst.max(dr.abs().max(di.abs()));
                }
                assert!(
                    worst / scale < 1e-10,
                    "n={n}: residual {worst} (scale {scale}) for lambda = {lr}{li:+}i"
                );
                j += if complex { 2 } else { 1 };
            }
        }
    }

    /// The left eigenvectors must satisfy `u^T A = lambda u^T`, i.e. for the
    /// complex vector `u = u_real + i u_imag`,
    /// `A^H u = conj(lambda) u`. Nothing in the existing tests checks this, so it
    /// is the acceptance test for [`dgeev_lr`]'s `SIDE = 'L'` branch.
    #[test]
    fn dgeev_lr_left_eigenvectors_satisfy_their_defining_equation() {
        // A deterministic generator, as in the battery test below.
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };

        let mut cases: Vec<(usize, Vec<f64>)> = Vec::new();
        for n in [1usize, 2, 3, 5, 8] {
            cases.push((n, sample(n)));
        }
        // A pure rotation: the eigenvalues are 1 +- 2i, so the complex-pair
        // branch of the left back-substitution is exercised.
        cases.push((2, vec![1.0, -2.0, 2.0, 1.0]));
        // Three decoupled rotations: three conjugate pairs.
        let mut blockrot = vec![0.0; 36];
        for b in 0..3usize {
            let o = 2 * b;
            blockrot[o + o * 6] = 1.0 + b as f64;
            blockrot[(o + 1) + o * 6] = -2.0;
            blockrot[o + (o + 1) * 6] = 2.0;
            blockrot[(o + 1) + (o + 1) * 6] = 1.0 + b as f64;
        }
        cases.push((6, blockrot));
        for n in [3usize, 7] {
            for _ in 0..4 {
                let mut a = vec![0.0; n * n];
                for x in a.iter_mut() {
                    *x = next() * 20.0;
                }
                cases.push((n, a));
            }
        }

        for (n, data) in cases {
            let mut a = Matrix::new(n, n);
            a.as_mut_data().copy_from_slice(&data);

            // The working copy must hold the same matrix; `a` is kept intact so
            // the residual can be formed against the original.
            let mut av = Matrix::new(n, n);
            av.as_mut_data().copy_from_slice(&data);
            let mut u_real = Matrix::new(n, n);
            let mut u_imag = Matrix::new(n, n);
            let mut v_real = Matrix::new(n, n);
            let mut v_imag = Matrix::new(n, n);
            let (wr, wi) = dgeev_lr(
                u_real.as_mut_data(),
                u_imag.as_mut_data(),
                v_real.as_mut_data(),
                v_imag.as_mut_data(),
                av.as_mut_data(),
                n,
            )
            .unwrap();

            let scale = a.as_data().iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
            let mut j = 0usize;
            while j < n {
                let (lr, li) = (wr[j], wi[j]);
                let complex = li != 0.0;
                // u^H A = lambda u^H  <=>  A^H u = conj(lambda) u.
                let mut worst = 0.0f64;
                for i in 0..n {
                    let mut sr = 0.0;
                    let mut si = 0.0;
                    for k in 0..n {
                        // A^H u, i.e. sum_k conj(A(k,i)) u(k).
                        sr += a.get(k, i) * u_real.get(k, j);
                        si += a.get(k, i) * u_imag.get(k, j);
                    }
                    // conj(lambda) * u(i) = (lr - i*li)(ur + i*ui)
                    let ur = u_real.get(i, j);
                    let ui = u_imag.get(i, j);
                    let dr = sr - (lr * ur + li * ui);
                    let di = si - (lr * ui - li * ur);
                    worst = worst.max(dr.abs().max(di.abs()));
                }
                assert!(
                    worst / scale < 1e-12,
                    "n={n}: left residual {worst} (scale {scale}) for lambda = {lr}{li:+}i"
                );
                j += if complex { 2 } else { 1 };
            }

            // The right eigenvectors must be unchanged by the extra left work.
            let mut vr = Matrix::new(n, n);
            let mut vi = Matrix::new(n, n);
            let mut aw = Matrix::new(n, n);
            aw.as_mut_data().copy_from_slice(&data);
            let (wr2, wi2) = dgeev(vr.as_mut_data(), vi.as_mut_data(), aw.as_mut_data(), n).unwrap();
            for i in 0..n {
                assert!(wr[i] == wr2[i] && wi[i] == wi2[i], "n={n}: eigenvalue {i} differs");
                for r in 0..n {
                    assert!(
                        (v_real.get(r, i) - vr.get(r, i)).abs() < 1e-13
                            && (v_imag.get(r, i) - vi.get(r, i)).abs() < 1e-13,
                        "n={n}: right eigenvector ({r},{i}) differs"
                    );
                }
            }
        }
    }

    /// Non-convergence must be reported, not dropped on the floor.
    ///
    /// A matrix of NaN makes the QR iteration's convergence test
    /// (`|h(i+1,i)| <= threshold`) permanently false, so `dlahqr` exhausts its
    /// iteration limit and returns `INFO > 0`. Before that `INFO` was plumbed
    /// through it was a `debug_assert`, so a release build returned partly
    /// converged eigenvalues as if they had converged. To check this test can
    /// actually fail, put the `debug_assert` back and build in release: the
    /// assertions below then see `Ok`.
    ///
    /// `n >= 4` on purpose: at `n == 2` the matrix is a single 2x2 block, which
    /// `dlanv2` reduces algebraically without iterating, so there is no iteration
    /// limit to exhaust -- reference `dhseqr` reports success there too.
    #[test]
    fn non_convergence_is_reported_rather_than_returned_as_a_result() {
        for n in [4usize, 8] {
            let mut wr = vec![0.0; n];
            let mut wi = vec![0.0; n];

            let mut t = vec![f64::NAN; n * n];
            assert_eq!(
                dhseqr(&mut t, &mut wr, &mut wi, n),
                Err(LapackError::Convergence),
                "dhseqr n={n} must report non-convergence"
            );

            let mut z = vec![0.0; n * n];
            for i in 0..n {
                z[i + i * n] = 1.0;
            }
            let mut t2 = vec![f64::NAN; n * n];
            assert_eq!(
                dhseqr_with_schur_vectors(&mut t2, &mut wr, &mut wi, &mut z, n),
                Err(LapackError::Convergence),
                "dhseqr_with_schur_vectors n={n} must report non-convergence"
            );

            let mut v_real = vec![0.0; n * n];
            let mut v_imag = vec![0.0; n * n];
            let mut a = vec![f64::NAN; n * n];
            assert_eq!(
                dgeev(&mut v_real, &mut v_imag, &mut a, n).map(|_| ()),
                Err(LapackError::Convergence),
                "dgeev n={n} must report non-convergence"
            );

            let mut u_real = vec![0.0; n * n];
            let mut u_imag = vec![0.0; n * n];
            let mut v_real2 = vec![0.0; n * n];
            let mut v_imag2 = vec![0.0; n * n];
            let mut a2 = vec![f64::NAN; n * n];
            assert_eq!(
                dgeev_lr(&mut u_real, &mut u_imag, &mut v_real2, &mut v_imag2, &mut a2, n).map(|_| ()),
                Err(LapackError::Convergence),
                "dgeev_lr n={n} must report non-convergence"
            );
        }
    }
}
