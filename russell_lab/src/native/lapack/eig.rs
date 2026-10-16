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
//! Steps 2-4 are here. Everything is column-major, so element `(i, j)` of an
//! `n x n` matrix lives at `data[i + j * n]`.
//!
//! # The Schur form is a *real* one
//!
//! A real matrix can have complex eigenvalues, so the real Schur form is only
//! *quasi*-triangular: 1x1 blocks on the diagonal for real eigenvalues and 2x2
//! blocks for complex conjugate pairs. Those 2x2 blocks are why the eigenvector
//! step needs `dlaln2`-style solving rather than a plain triangular substitution.

/// A 2x2 block's eigenvalues, as [`dlanv2`] returns them.
///
/// `rt1r`/`rt1i` and `rt2r`/`rt2i` are the two eigenvalues. They are real when
/// `rt1i == rt2i == 0`, and otherwise a complex conjugate pair with `rt1i > 0`,
/// which is LAPACK's convention and is relied on by the 2x2-block handling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eigen2 {
    pub rt1r: f64,
    pub rt1i: f64,
    pub rt2r: f64,
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
            beta *= safmin;
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
            *d = *d - (bcmax / z) * bcmis;
            let tau = f64::hypot(*c, z);
            *cs = z / tau;
            *sn = *c / tau;
            *b = *b - *c;
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
                        *b = *b - *c;
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

/// `dlahqr` (LAPACK 3.12.1, `SRC/dlahqr.f`): the unblocked double-shift QR step
/// that reduces a Hessenberg matrix to real Schur form.
///
/// This is a literal transcription of `dlahqr.f` with the indices shifted from
/// 1-based to 0-based; the Fortran loop labels survive in the comments. It is
/// called here with `WANTT = .TRUE.` and `WANTZ = .FALSE.`:
///
/// * `WANTT` true selects `I1 = 0, I2 = n - 1` (LAPACK's `I1 = 1, I2 = N`), so
///   every transformation is applied to the whole matrix. `dhseqr` with
///   `JOB = 'S'` needs that, because the caller gets the Schur form back in `h`
///   and not just the eigenvalues. With `WANTT` false, `dlahqr.f` narrows `I1`
///   and `I2` to the active block inside the main loop and leaves the rest of
///   the matrix stale; that is the "eigenvalues only" path.
/// * `WANTZ` false means no Schur vectors are accumulated (`COMPZ = 'N'`).
///
/// `ilo` and `ihi` delimit the Hessenberg submatrix to work on (`dhseqr` passes
/// `0` and `n - 1`). The return value is LAPACK's `INFO`: zero on success, or the
/// index at which the iteration gave up after `30 * max(10, nh)` sweeps. `h` is
/// column-major with leading dimension `ldh`; `(wr, wi)` receives the
/// eigenvalues in deflation order, a complex conjugate pair in two adjacent
/// positions with `wi > 0` first.
///
/// The structure is `dlahqr.f`'s: the `K` scan looks for one negligible
/// subdiagonal and leaves `K = L` when it runs off the top of the block (so `L`
/// never drops below `ILO`); once `L >= I - 1` a 1x1 or 2x2 block deflates and
/// the rotation is applied to the rest of `h`; otherwise one double-shift sweep
/// runs from `M` to `I - 1`. `KDEFL` counts the sweeps since the last deflation
/// and picks the two exceptional shifts at multiples of `KEXSH = 10` and
/// `2 * KEXSH = 20`.
fn dlahqr(n: usize, ilo: usize, ihi: usize, h: &mut [f64], ldh: usize, wr: &mut [f64], wi: &mut [f64]) -> usize {
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
            while k >= l + 1 {
                if h[k + (k - 1) * ldh].abs() <= smlnum {
                    break;
                }
                let mut tst = h[(k - 1) + (k - 1) * ldh].abs() + h[k + k * ldh].abs();
                if tst == 0.0 {
                    if k >= ilo + 2 {
                        tst += h[(k - 1) + (k - 2) * ldh].abs();
                    }
                    if k + 1 <= ihi {
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
            let mut s = h11.abs() + h12.abs() + h21.abs() + h22.abs();
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
/// This calls [`dlahqr`] directly, which is what the reference `dhseqr` does for
/// `n <= 75`; the blocked `dlaqr0`/`dlaqr5` path is not implemented. `JOB = 'S'`
/// means `WANTT = .TRUE.` and `COMPZ = 'N'` means `WANTZ = .FALSE.`, and with
/// `ILO = 1, IHI = N` the whole matrix is reduced.
///
/// `dlahqr` reports non-convergence through `INFO`, which this signature has no
/// way to return; a matrix that exhausts the iteration limit is left partly
/// reduced, as if `INFO > 0`.
///
/// The acceptance test is `parity::dhseqr_eigenvalues_agree_with_the_reference`,
/// which checks the quasi-triangular structure, the standard form of each 2x2
/// block, and the eigenvalue multiset against the reference `dgeev`.
pub fn dhseqr(t: &mut [f64], wr: &mut [f64], wi: &mut [f64], n: usize) {
    assert_eq!(t.len(), n * n, "dhseqr: t must be n*n");
    assert_eq!(wr.len(), n, "dhseqr: wr must have n entries");
    assert_eq!(wi.len(), n, "dhseqr: wi must have n entries");
    if n == 0 {
        return;
    }
    // JOB = 'S' -> WANTT = .TRUE.; COMPZ = 'N' -> WANTZ = .FALSE.; ILO = 1,
    // IHI = N (0-based 0 and n - 1).
    let info = dlahqr(n, 0, n - 1, t, n, wr, wi);
    debug_assert_eq!(info, 0, "dhseqr: the QR iteration failed to converge");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `dgeev` (right eigenvectors only): the eigenvalues and right eigenvectors of a
/// general real square matrix.
///
/// `a` is `n x n` column-major and is destroyed. `v_real` and `v_imag` receive
/// the eigenvectors in LAPACK's compact storage convention, which
/// `crate::internal::dgeev_data` unpacks: for a complex conjugate pair, column
/// `j` holds the real part and column `j+1` the imaginary part, and
/// `v(j) = col_j + i col_{j+1}`; a real eigenvalue's column has zero imaginary
/// part. Conjugate pairs are stored adjacently with `wi > 0` first.
///
/// **Not implemented yet.** The acceptance test is
/// `parity::dgeev_eigenvectors_satisfy_their_defining_equation`, which checks
/// `A v = lambda v` directly -- eigenvectors have no unique value to compare
/// against, so the defining equation is the criterion.
///
/// The plan, following `dgeev.f`: `dgebal` (optional; the reduction alone is
/// enough for a first version), then `dgehrd` (done), then `dhseqr` with
/// `COMPZ='V'` to accumulate the Schur vectors into `Q`, then `dtrevc` for the
/// eigenvectors of the quasi-triangular form, then transform back by `Q`.
pub fn dgeev(v_real: &mut [f64], v_imag: &mut [f64], a: &mut [f64], n: usize) {
    assert_eq!(a.len(), n * n, "dgeev: a must be n*n");
    assert_eq!(v_real.len(), n * n, "dgeev: v_real must be n*n");
    assert_eq!(v_imag.len(), n * n, "dgeev: v_imag must be n*n");
    let _ = (v_real, v_imag, a, n);
    unimplemented!("dgeev: the eigenvector back-substitution is not written yet")
}

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

    /// Right eigenvectors of a general real matrix, by residual.
    ///
    /// An eigenvector has no unique value -- it is defined up to scaling, and its
    /// sign and phase are arbitrary -- so comparing against the reference's vectors
    /// would be meaningless. What *is* well defined is the defining equation, so
    /// this checks `A v = lambda v` for each pair, at the tolerance the reduction and
    /// the back-substitution accumulate.
    ///
    /// This uses the compact storage convention the rest of the crate unpacks with
    /// `dgeev_data`: for a complex conjugate pair, column `j` holds the real part and
    /// column `j+1` the imaginary part, and `v(j) = col_j + i col_{j+1}`.
    #[test]
    #[ignore = "dgeev is not implemented yet; this test defines done for it"]
    fn dgeev_eigenvectors_satisfy_their_defining_equation() {
        for n in [1usize, 2, 3, 5, 8, 12] {
            // `sample` here returns column-major data; wrap it so the residual can
            // be formed with element access.
            let data = sample(n);
            let mut a = Matrix::new(n, n);
            a.as_mut_data().copy_from_slice(&data);

            let mut av = Matrix::new(n, n); // working copy, destroyed
            let mut v_real = Matrix::new(n, n);
            let mut v_imag = Matrix::new(n, n);
            dgeev(av.as_mut_data(), v_real.as_mut_data(), v_imag.as_mut_data(), n);

            // Recover the eigenvalues from the Schur form so the residual can be
            // formed; the driver returns them in (wr, wi) form for reuse.
            let mut t = a.clone();
            dgehrd(t.as_mut_data(), n, None);
            let mut wr = vec![0.0; n];
            let mut wi = vec![0.0; n];
            dhseqr(t.as_mut_data(), &mut wr, &mut wi, n);

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
}
