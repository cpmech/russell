//! The reference's fixed example sets, ported from `TESTING/EIG`.
//!
//! `DCHKEC` does not build matrices from a random stream: it runs a routine over a
//! Cartesian grid of hand-chosen entries — values one ulp apart from each other,
//! magnitudes at the edges of the range — and checks the answer against quantities
//! that must hold for *any* correct implementation. That is the only route to the
//! rescaling and sign-flip branches of `dlanv2` and `dlaln2`: the families in
//! [`super::families`] are built to be solvable, so they never reach them.
//!
//! Each checker returns the reference's own `(RMAX, LMAX, KNT)`: the worst ratio,
//! the case that produced it, and how many cases ran — plus, for `DGET31`, the
//! `NINFO` pair that counts the cases the routine had to perturb. `DCHKEC`
//! compares `RMAX` with the threshold from its input file, which is 20 in the
//! standard `dtest.in`; the tests here quote what LAPACK's own run reports on the
//! same grid.
//!
//! Test-only: nothing here is part of the library's behavior.

use super::eig::dlanv2;

/// Port of `DGET33`: `DLANV2` over the reference's example grid.
///
/// The grid is `4^4` blocks times `3^4` magnitudes: entries drawn from
/// `{1, 1+2eps, 2, 2-4eps}` — three of them within a few ulp of each other, which
/// is what makes the "are the eigenvalues complex" decision delicate — and scaled
/// by `{smlnum, 1, bignum}`, with the (2,1) entry negated. 20,736 cases.
///
/// For each case the reference asks three things of the rotation `dlanv2` returns:
///
/// 1. `Q` is orthogonal, accumulated as `Q := Q G` from the identity;
/// 2. `Q^T T Q = T_new` reproduces the block the routine wrote back;
/// 3. structurally, the (2,1) entry is zero unless the diagonal entries are equal
///    and the off-diagonals have opposite signs — the two shapes `dlanv2` is
///    allowed to leave behind. A violation is scored as `1/eps`, which is why
///    `rmax` in the thousands means a branch was taken incorrectly rather than a
///    rounding difference.
///
/// Returns `(rmax, lmax, knt)`.
pub fn get33() -> (f64, usize, usize) {
    let eps = f64::EPSILON;
    let smlnum = f64::MIN_POSITIVE / eps;
    let bignum = 1.0 / smlnum;
    let val = [1.0, 1.0 + 2.0 * eps, 2.0, 2.0 - 4.0 * eps];
    let vm = [smlnum, 1.0, bignum];

    let mut knt = 0usize;
    let mut lmax = 0usize;
    let mut rmax = 0.0f64;

    for i1 in 0..4 {
        for i2 in 0..4 {
            for i3 in 0..4 {
                for i4 in 0..4 {
                    for im1 in 0..3 {
                        for im2 in 0..3 {
                            for im3 in 0..3 {
                                for im4 in 0..3 {
                                    // The block, row-major: T(1,1), T(1,2), T(2,1), T(2,2).
                                    let t = [
                                        [val[i1] * vm[im1], val[i2] * vm[im2]],
                                        [-val[i3] * vm[im3], val[i4] * vm[im4]],
                                    ];
                                    let tnrm = t[0][0].abs().max(t[0][1].abs()).max(t[1][0].abs()).max(t[1][1].abs());

                                    let (mut a, mut b, mut c, mut d) = (t[0][0], t[0][1], t[1][0], t[1][1]);
                                    let (mut cs, mut sn) = (0.0f64, 0.0f64);
                                    dlanv2(&mut a, &mut b, &mut c, &mut d, &mut cs, &mut sn);

                                    // Q := Q G, starting from the identity.
                                    let (mut q11, mut q12) = (1.0f64, 0.0f64);
                                    let (mut q21, mut q22) = (0.0f64, 1.0f64);
                                    let r = q11 * cs + q12 * sn;
                                    q12 = -q11 * sn + q12 * cs;
                                    q11 = r;
                                    let r = q21 * cs + q22 * sn;
                                    q22 = -q21 * sn + q22 * cs;
                                    q21 = r;
                                    let q = [[q11, q12], [q21, q22]];

                                    let mut res = 0.0f64;
                                    res += (q11 * q11 + q12 * q12 - 1.0).abs() / eps;
                                    res += (q22 * q22 + q21 * q21 - 1.0).abs() / eps;
                                    res += (q11 * q21 + q12 * q22).abs() / eps;

                                    // T2 = T Q, then T - Q^T T2.
                                    let mut t2 = [[0.0f64; 2]; 2];
                                    for (j1, row) in t2.iter_mut().enumerate() {
                                        for (j2, v) in row.iter_mut().enumerate() {
                                            let mut sum = 0.0;
                                            for k in 0..2 {
                                                sum += t[j1][k] * q[k][j2];
                                            }
                                            *v = sum;
                                        }
                                    }
                                    // The reference subtracts from `T`, which `DLANV2`
                                    // overwrote in place: it is the standardized block that
                                    // `Q^T T1 Q` has to reproduce, not the original one.
                                    let tnew = [[a, b], [c, d]];
                                    for (j1, row) in tnew.iter().enumerate() {
                                        for (j2, tval) in row.iter().enumerate() {
                                            let mut sum = *tval;
                                            for k in 0..2 {
                                                sum -= q[k][j1] * t2[k][j2];
                                            }
                                            res += sum.abs() / eps / tnrm;
                                        }
                                    }

                                    // The two shapes `dlanv2` may leave behind.
                                    let sign = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
                                    if c != 0.0 && (a != d || sign(b) * sign(c) > 0.0) {
                                        res += 1.0 / eps;
                                    }

                                    knt += 1;
                                    if res > rmax {
                                        lmax = knt;
                                        rmax = res;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (rmax, lmax, knt)
}

/// One `DGET31` example: the arguments `DLALN2` receives, in the reference's
/// order.
///
/// The entries a block does not read are zero rather than the stale values the
/// reference's local array happens to carry over from the previous block — no
/// `DLALN2` branch and no residual formula below reads them, so the two are the
/// same case.
pub struct Case {
    pub ltrans: bool,
    pub na: usize,
    pub nw: usize,
    pub smin: f64,
    pub ca: f64,
    pub a11: f64,
    pub a21: f64,
    pub a12: f64,
    pub a22: f64,
    pub d1: f64,
    pub d2: f64,
    pub b11: f64,
    pub b21: f64,
    pub b12: f64,
    pub b22: f64,
    pub wr: f64,
    pub wi: f64,
}

/// `DLALN2`'s `INFO = 1` decision, transcribed from `SRC/dlaln2.f`.
///
/// The routine reports `INFO = 1` when it had to raise a singular value to
/// `smin`: the 1x1 pivot below `smini`, the 2x2 norm below `smini`, or the
/// second pivot of the complete-pivoting elimination below `smini`. `DGET31`
/// needs that flag because it switches the residual's denominator on it, and
/// neither copy of the routine in this crate returns it — the one in
/// [`super::eig`] is the specialization its callers use, and the one in
/// [`super::qz`] returns `Laln2`, which has no field for it. So the adapters
/// pass it in, and a copy that perturbed a different case than the reference
/// does fails the residual rather than silently scoring its own decision as
/// correct.
///
/// `INFO` is never negative here: the reference does not check its arguments
/// ("In the interests of speed"), so `NINFO(1)` in `DGET31` stays zero.
pub fn get31_info(c: &Case) -> i32 {
    // `IPIVOT(:, icmax)` of the reference, by column.
    const IPIVOT: [[usize; 4]; 4] = [[1, 2, 3, 4], [2, 1, 4, 3], [3, 4, 1, 2], [4, 3, 2, 1]];

    let smlnum = 2.0 * f64::MIN_POSITIVE;
    let smini = c.smin.max(smlnum);

    if c.na == 1 {
        // `CNORM` of the 1x1 system: one entry, or the real and imaginary parts
        // of the complex divisor.
        let csr = c.ca * c.a11 - c.wr * c.d1;
        let cnorm = if c.nw == 1 {
            csr.abs()
        } else {
            csr.abs() + (c.wi * c.d1).abs()
        };
        return if cnorm < smini { 1 } else { 0 };
    }

    // CRV = [CR(1,1), CR(2,1), CR(1,2), CR(2,2)], the (1,2) and (2,1) entries
    // exchanged when the transpose is wanted.
    let crv = if c.ltrans {
        [
            c.ca * c.a11 - c.wr * c.d1,
            c.ca * c.a12,
            c.ca * c.a21,
            c.ca * c.a22 - c.wr * c.d2,
        ]
    } else {
        [
            c.ca * c.a11 - c.wr * c.d1,
            c.ca * c.a21,
            c.ca * c.a12,
            c.ca * c.a22 - c.wr * c.d2,
        ]
    };

    if c.nw == 1 {
        // Real 2x2: either the whole norm is below SMINI, or the second pivot of
        // the complete-pivoting elimination is.
        let mut cmax = 0.0;
        let mut icmax = 0usize;
        for (j, v) in crv.iter().enumerate() {
            if v.abs() > cmax {
                cmax = v.abs();
                icmax = j;
            }
        }
        if cmax < smini {
            return 1;
        }
        let ur11r = 1.0 / crv[icmax];
        let lr21 = ur11r * crv[IPIVOT[icmax][1] - 1];
        let ur22 = crv[IPIVOT[icmax][3] - 1] - crv[IPIVOT[icmax][2] - 1] * lr21;
        return if ur22.abs() < smini { 1 } else { 0 };
    }

    // Complex 2x2: same two conditions on |CR| + |CI|.
    let civ = [-c.wi * c.d1, 0.0, 0.0, -c.wi * c.d2];
    let mut cmax = 0.0;
    let mut icmax = 0usize;
    for (j, v) in crv.iter().enumerate() {
        let m = v.abs() + civ[j].abs();
        if m > cmax {
            cmax = m;
            icmax = j;
        }
    }
    if cmax < smini {
        return 1;
    }
    let ur11 = crv[icmax];
    let ui11 = civ[icmax];
    let cr21 = crv[IPIVOT[icmax][1] - 1];
    let ci21 = civ[IPIVOT[icmax][1] - 1];
    let ur12 = crv[IPIVOT[icmax][2] - 1];
    let ui12 = civ[IPIVOT[icmax][2] - 1];
    let cr22 = crv[IPIVOT[icmax][3] - 1];
    let ci22 = civ[IPIVOT[icmax][3] - 1];
    let u22abs = if icmax == 0 || icmax == 3 {
        // Off-diagonals of the pivoted C are real.
        let (ur11r, ui11r) = if ur11.abs() > ui11.abs() {
            let temp = ui11 / ur11;
            let r = 1.0 / (ur11 * (1.0 + temp * temp));
            (r, -temp * r)
        } else {
            let temp = ur11 / ui11;
            let i = -1.0 / (ui11 * (1.0 + temp * temp));
            (-temp * i, i)
        };
        let lr21 = cr21 * ur11r;
        let li21 = cr21 * ui11r;
        (cr22 - ur12 * lr21).abs() + (ci22 - ur12 * li21).abs()
    } else {
        // Diagonals of the pivoted C are real.
        let ur11r = 1.0 / ur11;
        let lr21 = cr21 * ur11r;
        let li21 = ci21 * ur11r;
        (cr22 - ur12 * lr21 + ui12 * li21).abs() + (-ur12 * li21 - ui12 * lr21).abs()
    };
    if u22abs < smini { 1 } else { 0 }
}

/// `DGET31`'s three penalties, applied to one case's ratio: a scale above one,
/// an `XNORM` that is not the norm of the `X` returned, and an `INFO` outside
/// `{0, 1}`. `zero` discards the ratio, which is how the reference treats a
/// solution whose right-hand side is negligible against the matrix.
fn scored(res: f64, zero: bool, scale: f64, xnorm: f64, xnorm_expected: f64, info: i32) -> f64 {
    let eps = f64::EPSILON;
    let smlnum = f64::MIN_POSITIVE / eps;
    let mut r = if zero { 0.0 } else { res };
    if scale > 1.0 {
        r += 1.0 / eps;
    }
    r += (xnorm - xnorm_expected).abs() / smlnum.max(xnorm) / eps;
    if info != 0 && info != 1 {
        r += 1.0 / eps;
    }
    r
}

#[derive(Default)]
struct Tally {
    knt: usize,
    lmax: usize,
    rmax: f64,
    ninfo_neg: usize,
    ninfo_pos: usize,
}

impl Tally {
    fn add(&mut self, res: f64, info: i32) {
        if info < 0 {
            self.ninfo_neg += 1;
        }
        if info > 0 {
            self.ninfo_pos += 1;
        }
        self.knt += 1;
        if res > self.rmax {
            self.lmax = self.knt;
            self.rmax = res;
        }
    }
}

/// Port of `DGET31`: `DLALN2` over the reference's example grid.
///
/// `DLALN2` solves `(ca A - w D) X = s B` for a 1x1 or 2x2 real `A`, a real or
/// complex `w = wr + i wi`, and a positive diagonal `D = diag(d1, d2)`; a pivot
/// below `smin` is raised to `smin`, and `s` is the scale that keeps `X` from
/// overflowing. Every combination of
///
/// | loop | values |
/// |---|---|
/// | `d1`, `d2` | `{sqrt(smlnum), 1, 2, sqrt(bignum)}` |
/// | `ca` | `{0, sqrt(smlnum), eps, 1/2, 1}` |
/// | `ltrans` | `{false, true}` |
/// | `smin` | `{smlnum, eps, 1/100, 1/eps}` |
///
/// runs four blocks — `(na, nw)` equal to `(1,1)`, `(1,2)`, `(2,1)`, `(2,2)` —
/// each over `a`, `b` drawn from `{sqrt(smlnum), 1, sqrt(bignum)}`, `wr` from
/// `{0, 1/2, 2, 1}` and, when `w` is complex, `wi` from `{smlnum, eps, 1, 2}`:
/// 36 + 144 + 36 + 144 = 360 cases per combination, 230,400 in all. The two
/// extremes are what reach the routines' rescaling branches — `ca = 0` makes the
/// system `-w D`, and `smin` at either end turns the perturbation on for almost
/// every case or for none.
///
/// `ca = d1 = d2 = 1` is special-cased by the reference: `wr` and `wi` are then
/// multiplied by `a(1,1)`, which puts `w` on the scale of the matrix instead of
/// on the scale of the perturbation threshold — the only cases where the
/// eigenvalue is comparable to `A` at all, and hence the only ones where a
/// recurrence draws its eigenvalue from its own block.
///
/// Each block checks its own residual, because the shapes of the equations
/// differ: its size is the largest ratio
///
/// ```text
///   norm( (ca A - w D) X - s B ) / ( max(ulp*norm(ca A - w D), smin') * norm(X) )
/// ```
///
/// where `smin'` falls back to `ulp*max(smin/ulp, norm)` when `INFO = 1`
/// perturbed the system. The reference's denominators differ by block in the
/// exact expression they use for the two norms (a single entry, an absolute sum
/// over a column, a maximum over rows) — transcribed rather than unified, since
/// those choices are what the tolerances were tuned against. `scale > 1`, an
/// `XNORM` that does not match the `X` returned, and an `INFO` outside `{0, 1}`
/// each add `1/eps`, so a ratio in the thousands means a wrong branch, not a
/// rounding difference.
///
/// The `ltrans` case carries a subtlety worth naming: the reference transposes
/// its `A` array *after* each call, inside the loops, so with `ltrans = .TRUE.`
/// the cases in one block alternate between `ca A - w D` and `ca A^T - w D`
/// while the residual — which reads the array at its current state — stays
/// consistent with the call that produced `X`. This port keeps a mutable `A`
/// per block for exactly that reason. Each row of the grid swaps an even number
/// of times and so starts from the untransposed block again.
///
/// Returns `(rmax, lmax, ninfo_neg, ninfo_pos, knt)`.
///
/// LAPACK 3.12.1's own run of `TESTING/EIG/dget31.f` reports `KNT = 230400`,
/// `LMAX = 249`, `NINFO = (0, 74796)` and `RMAX = 1.77272727272727271`.
pub fn get31<F>(solve: F) -> (f64, usize, usize, usize, usize)
where
    F: Fn(&Case) -> ([f64; 4], f64, f64, i32),
{
    get31_where(|_| true, solve)
}

/// [`get31`] restricted to the cases `keep` accepts.
///
/// A copy of `DLALN2` whose signature fixes some of the inputs cannot run the
/// whole grid; this lets it run the part its signature actually expresses
/// instead of emulating the rest, which would be a different computation and
/// would score a different residual. `lmax` then counts the kept cases.
pub fn get31_where<K, F>(keep: K, solve: F) -> (f64, usize, usize, usize, usize)
where
    K: Fn(&Case) -> bool,
    F: Fn(&Case) -> ([f64; 4], f64, f64, i32),
{
    let eps = f64::EPSILON;
    let unfl = f64::MIN_POSITIVE;
    let smlnum = f64::MIN_POSITIVE / eps;
    let bignum = 1.0 / smlnum;
    let vsm = [smlnum, eps, 1.0 / (10.0 * 10.0), 1.0 / eps];
    let vab = [smlnum.sqrt(), 1.0, bignum.sqrt()];
    let vwr = [0.0, 0.5, 2.0, 1.0];
    let vwi = [smlnum, eps, 1.0, 2.0];
    let vdd = [smlnum.sqrt(), 1.0, 2.0, bignum.sqrt()];
    let vca = [0.0, smlnum.sqrt(), eps, 0.5, 1.0];

    let mut tally = Tally::default();

    for &d1 in &vdd {
        for &d2 in &vdd {
            for &ca in &vca {
                for &ltrans in &[false, true] {
                    for &smin in &vsm {
                        let special = d1 == 1.0 && d2 == 1.0 && ca == 1.0;

                        // (na, nw) = (1, 1), real 1x1 system.
                        for &av in &vab {
                            for &bv in &vab {
                                for &wrv in &vwr {
                                    let wr = if special { wrv * av } else { wrv };
                                    let case = Case {
                                        ltrans,
                                        na: 1,
                                        nw: 1,
                                        smin,
                                        ca,
                                        a11: av,
                                        a21: 0.0,
                                        a12: 0.0,
                                        a22: 0.0,
                                        d1,
                                        d2,
                                        b11: bv,
                                        b21: 0.0,
                                        b12: 0.0,
                                        b22: 0.0,
                                        wr,
                                        wi: 0.0,
                                    };
                                    if !keep(&case) {
                                        continue;
                                    }
                                    let (x, scale, xnorm, info) = solve(&case);
                                    let x11 = x[0];
                                    let csr = ca * av - wr * d1;
                                    let mut res = (csr * x11 - scale * bv).abs();
                                    let den = if info == 0 {
                                        (eps * (csr * x11).abs()).max(smlnum)
                                    } else {
                                        (smin * x11.abs()).max(smlnum)
                                    };
                                    res /= den;
                                    let zero = x11.abs() < unfl && bv.abs() <= smlnum * csr.abs();
                                    tally.add(scored(res, zero, scale, xnorm, x11.abs(), info), info);
                                }
                            }
                        }

                        // (na, nw) = (1, 2), complex 1x1 system.
                        for &av in &vab {
                            for &bv in &vab {
                                let b1 = bv;
                                let b2 = -0.5 * bv;
                                for &wrv in &vwr {
                                    let wr = if special { wrv * av } else { wrv };
                                    for &wiv in &vwi {
                                        let wi = if special { wiv * av } else { wiv };
                                        let case = Case {
                                            ltrans,
                                            na: 1,
                                            nw: 2,
                                            smin,
                                            ca,
                                            a11: av,
                                            a21: 0.0,
                                            a12: 0.0,
                                            a22: 0.0,
                                            d1,
                                            d2,
                                            b11: b1,
                                            b21: 0.0,
                                            b12: b2,
                                            b22: 0.0,
                                            wr,
                                            wi,
                                        };
                                        if !keep(&case) {
                                            continue;
                                        }
                                        let (x, scale, xnorm, info) = solve(&case);
                                        let (x11, x12) = (x[0], x[2]);
                                        let csr = ca * av - wr * d1;
                                        let csi = wi * d1;
                                        let mut res = (csr * x11 + csi * x12 - scale * b1).abs();
                                        res += (csr * x12 - csi * x11 - scale * b2).abs();
                                        let den = if info == 0 {
                                            (eps * csr.abs().max(csi.abs()) * (x11.abs() + x12.abs())).max(smlnum)
                                        } else {
                                            (smin * (x11.abs() + x12.abs())).max(smlnum)
                                        };
                                        res /= den;
                                        let zero =
                                            x11.abs() < unfl && x12.abs() < unfl && b1.abs() <= smlnum * csr.abs();
                                        tally.add(scored(res, zero, scale, xnorm, x11.abs() + x12.abs(), info), info);
                                    }
                                }
                            }
                        }

                        // (na, nw) = (2, 1), real 2x2 system.
                        for &av in &vab {
                            // A(1,1), A(2,1), A(1,2), A(2,2). Mutable: the
                            // reference transposes it after every call below.
                            let mut a = [av, -7.0 * av, -3.0 * av, 21.0 * av];
                            for &bv in &vab {
                                let b = [bv, -2.0 * bv, 0.0, 0.0];
                                for &wrv in &vwr {
                                    let wr = if special { wrv * a[0] } else { wrv };
                                    let case = Case {
                                        ltrans,
                                        na: 2,
                                        nw: 1,
                                        smin,
                                        ca,
                                        a11: a[0],
                                        a21: a[1],
                                        a12: a[2],
                                        a22: a[3],
                                        d1,
                                        d2,
                                        b11: b[0],
                                        b21: b[1],
                                        b12: b[2],
                                        b22: b[3],
                                        wr,
                                        wi: 0.0,
                                    };
                                    let out = if keep(&case) { Some(solve(&case)) } else { None };
                                    if ltrans {
                                        a.swap(1, 2);
                                    }
                                    if let Some((x, scale, xnorm, info)) = out {
                                        let (x11, x21) = (x[0], x[1]);
                                        let r1 = ca * a[0] - wr * d1;
                                        let r2 = ca * a[3] - wr * d2;
                                        let c1 = r1.abs() + (ca * a[2]).abs();
                                        let c2 = (ca * a[1]).abs() + r2.abs();
                                        let mut res = (r1 * x11 + ca * a[2] * x21 - scale * b[0]).abs();
                                        res += (ca * a[1] * x11 + r2 * x21 - scale * b[1]).abs();
                                        let xmax = x11.abs().max(x21.abs());
                                        let den = if info == 0 {
                                            (eps * c1.max(c2) * xmax).max(smlnum)
                                        } else {
                                            (eps * (smin / eps).max(c1.max(c2)) * xmax).max(smlnum)
                                        };
                                        res /= den;
                                        let zero = x11.abs() < unfl
                                            && x21.abs() < unfl
                                            && b[0].abs() + b[1].abs() <= smlnum * (c1 + c2);
                                        tally.add(scored(res, zero, scale, xnorm, xmax, info), info);
                                    }
                                }
                            }
                        }

                        // (na, nw) = (2, 2), complex 2x2 system.
                        for &av in &vab {
                            let mut a = [2.0 * av, -7.0 * av, -3.0 * av, 21.0 * av];
                            for &bv in &vab {
                                let b = [bv, -2.0 * bv, 4.0 * bv, -7.0 * bv];
                                for &wrv in &vwr {
                                    let wr = if special { wrv * a[0] } else { wrv };
                                    for &wiv in &vwi {
                                        let wi = if special { wiv * a[0] } else { wiv };
                                        let case = Case {
                                            ltrans,
                                            na: 2,
                                            nw: 2,
                                            smin,
                                            ca,
                                            a11: a[0],
                                            a21: a[1],
                                            a12: a[2],
                                            a22: a[3],
                                            d1,
                                            d2,
                                            b11: b[0],
                                            b21: b[1],
                                            b12: b[2],
                                            b22: b[3],
                                            wr,
                                            wi,
                                        };
                                        let out = if keep(&case) { Some(solve(&case)) } else { None };
                                        if ltrans {
                                            a.swap(1, 2);
                                        }
                                        if let Some((x, scale, xnorm, info)) = out {
                                            let (x11, x21, x12, x22) = (x[0], x[1], x[2], x[3]);
                                            let r1 = ca * a[0] - wr * d1;
                                            let r2 = ca * a[3] - wr * d2;
                                            let s1 = wi * d1;
                                            let s2 = wi * d2;
                                            let c1 = r1.abs() + (ca * a[2]).abs() + s1.abs();
                                            let c2 = (ca * a[1]).abs() + r2.abs() + s2.abs();
                                            let mut res = (r1 * x11 + ca * a[2] * x21 + s1 * x12 - scale * b[0]).abs();
                                            res += (r1 * x12 + ca * a[2] * x22 - s1 * x11 - scale * b[2]).abs();
                                            res += (ca * a[1] * x11 + r2 * x21 + s2 * x22 - scale * b[1]).abs();
                                            res += (ca * a[1] * x12 + r2 * x22 - s2 * x21 - scale * b[3]).abs();
                                            let xmax = (x11.abs() + x21.abs()).max(x12.abs() + x22.abs());
                                            let den = if info == 0 {
                                                (eps * c1.max(c2) * xmax).max(smlnum)
                                            } else {
                                                (eps * (smin / eps).max(c1.max(c2)) * xmax).max(smlnum)
                                            };
                                            res /= den;
                                            let zero = x11.abs() < unfl
                                                && x21.abs() < unfl
                                                && x12.abs() < unfl
                                                && x22.abs() < unfl
                                                && b[0].abs() + b[1].abs() <= smlnum * (c1 + c2);
                                            let xnorm_expected = (x11.abs() + x12.abs()).max(x21.abs() + x22.abs());
                                            tally.add(scored(res, zero, scale, xnorm, xnorm_expected, info), info);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (tally.rmax, tally.lmax, tally.ninfo_neg, tally.ninfo_pos, tally.knt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `DGET33` on the port's `dlanv2`, against what LAPACK's own run reports.
    ///
    /// Captured from `dget33.f` compiled against `liblapack`: `KNT = 20736`,
    /// `LMAX = 11453`, `RMAX = 6.75`. The port reports the same three numbers — the
    /// same worst case, at the same score — because the checker's arithmetic is the
    /// same and `dlanv2` is a two-by-two routine whose branches are all taken here.
    ///
    /// The grid is small enough to run exhaustively and the routine is trivial, so
    /// this is cheap: the whole sweep is a hundredth of a second.
    ///
    /// The first version of this checker disagreed with LAPACK by `2/eps` and the
    /// routine was right: the reference subtracts the reconstruction from `T`,
    /// which `DLANV2` overwrote in place, while the checker compared against the
    /// original block. Worth remembering the next time one of these scores badly —
    /// read the reference's data flow before suspecting the port.
    #[test]
    fn dlanv2_matches_the_reference_examples() {
        let (rmax, lmax, knt) = get33();
        println!("dlanv2 over the reference's example grid: {knt} cases, rmax = {rmax:e} at case {lmax}");
        assert_eq!(knt, 20736, "the grid must have 4^4 * 3^4 cases");
        assert!(
            rmax <= 20.0,
            "dlanv2 on the reference's examples: rmax = {rmax:e} at case {lmax}, where LAPACK's own dlanv2 reports 6.75"
        );
        assert!(lmax > 0 || rmax == 0.0, "lmax must name a case when rmax is nonzero");
    }
}
