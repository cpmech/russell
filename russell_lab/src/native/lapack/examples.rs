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
//! the case that produced it, and how many cases ran. `DCHKEC` compares `RMAX`
//! with the threshold from its input file, which is 20 in the standard `dtest.in`;
//! the tests here quote what LAPACK's own run reports on the same grid.
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
