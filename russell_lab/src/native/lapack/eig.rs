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

/// `dhseqr` with `JOB = 'S'`, `COMPZ = 'N'`: reduce an upper Hessenberg matrix to
/// real Schur form and return its eigenvalues.
///
/// On entry `t` is `n x n` and upper Hessenberg; on exit it is quasi-upper
/// triangular (zero below the first subdiagonal) and `(wr, wi)` holds the
/// eigenvalues in deflation order. A complex conjugate pair occupies two
/// adjacent positions with `wi > 0` for the first, matching the 2x2 block.
///
/// **Not implemented yet.** The acceptance test is
/// `parity::dhseqr_eigenvalues_agree_with_the_reference`, which checks the
/// quasi-triangular structure, the standard form of each 2x2 block, and the
/// eigenvalue multiset against the reference `dgeev`. The QR iteration itself is
/// `dlahqr`; see `AGENTS-WIP.md` for the findings from a previous attempt.
pub fn dhseqr(t: &mut [f64], wr: &mut [f64], wi: &mut [f64], n: usize) {
    assert_eq!(t.len(), n * n, "dhseqr: t must be n*n");
    assert_eq!(wr.len(), n, "dhseqr: wr must have n entries");
    assert_eq!(wi.len(), n, "dhseqr: wi must have n entries");
    let _ = (t, wr, wi, n);
    unimplemented!("dhseqr: the QR iteration is not written yet")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

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
}
