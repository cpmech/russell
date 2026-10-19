//! LAPACK's own test matrix families, so the native routines can be fed the same
//! inputs the reference suite uses.
//!
//! The reference checks its drivers on matrices built by `TESTING/MATGEN`: a random
//! stream with a fixed seed (`DLARUV`/`DLARNV`), decode tables that turn a small
//! integer into a spectrum family, and `DLATMS` to impose a prescribed condition
//! number. Alphabetical fixtures never reach the branches those families target --
//! clustered eigenvalues, deflation, a 2x2 block that forces a rescale -- which is
//! why the native layer's scaling paths stayed unexecuted.
//!
//! Everything here exists for the tests: the module is `#[cfg(test)]`, so it adds
//! nothing to a release build. Each item names the Fortran routine it follows, and
//! the transcription is pinned against the reference build rather than against
//! itself -- `larnv_matches_the_reference_stream` holds values captured from
//! LAPACK's own `libtmglib` for the seed every LAPACK test uses.

/// The `MM` table of `DLARUV`: 128 multipliers, 12 bits per word, indexed by
/// (position in the batch, word).
///
/// Transcribed from `SRC/dlaruv.f`; the values are the i-th power of the
/// multiplier 33952834046453 modulo 2**48, split into four 12-bit words.
const MM: [[u32; 4]; 128] = [
    [494, 322, 2508, 2549],
    [2637, 789, 3754, 1145],
    [255, 1440, 1766, 2253],
    [2008, 752, 3572, 305],
    [1253, 2859, 2893, 3301],
    [3344, 123, 307, 1065],
    [4084, 1848, 1297, 3133],
    [1739, 643, 3966, 2913],
    [3143, 2405, 758, 3285],
    [3468, 2638, 2598, 1241],
    [688, 2344, 3406, 1197],
    [1657, 46, 2922, 3729],
    [1238, 3814, 1038, 2501],
    [3166, 913, 2934, 1673],
    [1292, 3649, 2091, 541],
    [3422, 339, 2451, 2753],
    [1270, 3808, 1580, 949],
    [2016, 822, 1958, 2361],
    [154, 2832, 2055, 1165],
    [2862, 3078, 1507, 4081],
    [697, 3633, 1078, 2725],
    [1706, 2970, 3273, 3305],
    [491, 637, 17, 3069],
    [931, 2249, 854, 3617],
    [1444, 2081, 2916, 3733],
    [444, 4019, 3971, 409],
    [3577, 1478, 2889, 2157],
    [3944, 242, 3831, 1361],
    [2184, 481, 2621, 3973],
    [1661, 2075, 1541, 1865],
    [3482, 4058, 893, 2525],
    [657, 622, 736, 1409],
    [3023, 3376, 3992, 3445],
    [3618, 812, 787, 3577],
    [1267, 234, 2125, 77],
    [1828, 641, 2364, 3761],
    [164, 4005, 2460, 2149],
    [3798, 1122, 257, 1449],
    [3087, 3135, 1574, 3005],
    [2400, 2640, 3912, 225],
    [2870, 2302, 1216, 85],
    [3876, 40, 3248, 3673],
    [1905, 1832, 3401, 3117],
    [1593, 2247, 2124, 3089],
    [1797, 2034, 2762, 1349],
    [1234, 2637, 149, 2057],
    [3460, 1287, 2245, 413],
    [328, 1691, 166, 65],
    [2861, 496, 466, 1845],
    [1950, 1597, 4018, 697],
    [617, 2394, 1399, 3085],
    [2070, 2584, 190, 3441],
    [3331, 1843, 2879, 1573],
    [769, 336, 153, 3689],
    [1558, 1472, 2320, 2941],
    [2412, 2407, 18, 929],
    [2800, 433, 712, 533],
    [189, 2096, 2159, 2841],
    [287, 1761, 2318, 4077],
    [2045, 2810, 2091, 721],
    [1227, 566, 3443, 2821],
    [2838, 442, 1510, 2249],
    [209, 41, 449, 2397],
    [2770, 1238, 1956, 2817],
    [3654, 1086, 2201, 245],
    [3993, 603, 3137, 1913],
    [192, 840, 3399, 1997],
    [2253, 3168, 1321, 3121],
    [3491, 1499, 2271, 997],
    [2889, 1084, 3667, 1833],
    [2857, 3438, 2703, 2877],
    [2094, 2408, 629, 1633],
    [1818, 1589, 2365, 981],
    [688, 2391, 2431, 2009],
    [1407, 288, 1113, 941],
    [634, 26, 3922, 2449],
    [3231, 512, 2554, 197],
    [815, 1456, 184, 2441],
    [3524, 171, 2099, 285],
    [1914, 1677, 3228, 1473],
    [516, 2657, 4012, 2741],
    [164, 2270, 1921, 3129],
    [303, 2587, 3452, 909],
    [2144, 2961, 3901, 2801],
    [3480, 1970, 572, 421],
    [119, 1817, 3309, 4073],
    [3357, 676, 3171, 2813],
    [837, 1410, 817, 2337],
    [2826, 3723, 3039, 1429],
    [2332, 2803, 1696, 1177],
    [2089, 3185, 1256, 1901],
    [3780, 184, 3715, 81],
    [1700, 663, 2077, 1669],
    [3712, 499, 3019, 2633],
    [150, 3784, 1497, 2269],
    [2000, 1631, 1101, 129],
    [3375, 1925, 717, 1141],
    [1621, 3912, 51, 249],
    [3090, 1398, 981, 3917],
    [3765, 1349, 1978, 2481],
    [1149, 1441, 1813, 3941],
    [3146, 2224, 3881, 2217],
    [33, 2411, 76, 2749],
    [3082, 1907, 3846, 3041],
    [2741, 3192, 3694, 1877],
    [359, 2786, 1682, 345],
    [3316, 382, 124, 2861],
    [1749, 37, 1660, 1809],
    [185, 759, 3997, 3141],
    [2784, 2948, 479, 2825],
    [2202, 1862, 1141, 157],
    [2199, 3802, 886, 2881],
    [1364, 2423, 3514, 3637],
    [1244, 2051, 1301, 1465],
    [2020, 2295, 3604, 2829],
    [3160, 1332, 1888, 2161],
    [2785, 1832, 1836, 3365],
    [2772, 2405, 1990, 361],
    [1217, 3638, 2058, 2685],
    [1822, 3661, 692, 3745],
    [1245, 327, 1194, 2325],
    [2252, 3660, 20, 3609],
    [3904, 716, 3285, 3821],
    [2774, 1842, 2046, 3537],
    [997, 3987, 2107, 517],
    [2573, 1368, 3508, 3017],
    [1148, 1848, 3525, 2141],
    [545, 2366, 3801, 1537],
];

/// The modulus of one word of the 48-bit integer: `IPW2`.
const IPW2: u32 = 4096;

/// `R = 1/IPW2`, exact in binary.
const R: f64 = 1.0 / 4096.0;

/// Port of `DLARUV`: `n <= 128` numbers from the uniform (0,1) distribution, with
/// the seed advanced in place.
///
/// The reference keeps the 48-bit state in four 12-bit words so the arithmetic is
/// portable to 32-bit integers; the same layout is used here, and every
/// intermediate product fits in 31 bits (`4095 * 4095` and sums of four of them),
/// so `u32` holds it exactly as Fortran's `INTEGER` does.
pub fn laruv(seed: &mut [u32; 4], n: usize, x: &mut [f64]) {
    assert!(n <= 128, "laruv: n must be at most 128");
    assert!(x.len() >= n, "laruv: x is too short");
    if n < 1 {
        return;
    }
    let (mut i1, mut i2, mut i3, mut i4) = (seed[0], seed[1], seed[2], seed[3]);
    // IT1..IT4 hold the 48-bit product of the current iteration; the seed is the
    // last one computed, not the value it started from.
    let (mut it1, mut it2, mut it3, mut it4) = (0u32, 0u32, 0u32, 0u32);
    for i in 0..n.min(128) {
        let (m1, m2, m3, m4) = (MM[i][0], MM[i][1], MM[i][2], MM[i][3]);
        // Multiply the seed by the i-th power of the multiplier modulo 2**48.
        loop {
            it4 = i4 * m4;
            it3 = it4 / IPW2;
            it4 -= IPW2 * it3;
            it3 += i3 * m4 + i4 * m3;
            it2 = it3 / IPW2;
            it3 -= IPW2 * it2;
            it2 += i2 * m4 + i3 * m3 + i4 * m2;
            it1 = it2 / IPW2;
            it2 -= IPW2 * it1;
            it1 += i1 * m4 + i2 * m3 + i3 * m2 + i4 * m1;
            it1 %= IPW2;
            // Convert the 48-bit integer to a number in (0,1).
            let v = R * (it1 as f64 + R * (it2 as f64 + R * (it3 as f64 + R * it4 as f64)));
            if v == 1.0 {
                // About once every 2**n calls the first n bits are all ones and the
                // sum rounds to exactly 1.0; the reference steps the seed and
                // iterates again rather than returning an endpoint.
                i1 += 2;
                i2 += 2;
                i3 += 2;
                i4 += 2;
                continue;
            }
            x[i] = v;
            break;
        }
    }
    seed[0] = it1;
    seed[1] = it2;
    seed[2] = it3;
    seed[3] = it4;
}

/// The distribution selectors of `DLARNV`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dist {
    /// `IDIST = 1`: uniform (0,1).
    Uniform,
    /// `IDIST = 2`: uniform (-1,1).
    Symmetric,
    /// `IDIST = 3`: normal (0,1).
    Normal,
}

/// Port of `DLARNV`: `n` numbers from the chosen distribution, seed advanced in
/// place.
///
/// `DLARUV` produces at most 128 numbers per call, so `DLARNV` works in batches of
/// 64 (`LV/2`), doubling the batch for the normal distribution because Box-Muller
/// consumes two uniforms per output.
pub fn larnv(dist: Dist, seed: &mut [u32; 4], n: usize, x: &mut [f64]) {
    assert!(x.len() >= n, "larnv: x is too short");
    let lv = 128;
    let mut u = [0.0f64; 128];
    let mut iv = 0;
    while iv < n {
        let il = (lv / 2).min(n - iv);
        let il2 = if dist == Dist::Normal { 2 * il } else { il };
        laruv(seed, il2, &mut u);
        for i in 0..il {
            x[iv + i] = match dist {
                Dist::Uniform => u[i],
                Dist::Symmetric => 2.0 * u[i] - 1.0,
                // Box-Muller; TAU is the reference's TWOPI to the last bit.
                Dist::Normal => (-2.0 * u[2 * i].ln()).sqrt() * (std::f64::consts::TAU * u[2 * i + 1]).cos(),
            };
        }
        iv += lv / 2;
    }
}

/// Port of `DLARAN`: one number from the uniform (0,1) distribution, the seed
/// advanced in place.
///
/// The reference is `DLARUV` with `n = 1`; naming it separately keeps `DLATM1`'s
/// transcription readable where it draws a sign or a log-uniform magnitude.
pub fn laran(seed: &mut [u32; 4]) -> f64 {
    let mut x = [0.0f64];
    laruv(seed, 1, &mut x);
    x[0]
}

/// Port of `DLATM1`: the eigenvalue vector of the reference's spectra.
///
/// `mode` selects the distribution of `|d|` and `cond` its condition number
/// `max|d| / min|d|`; a negative `mode` reverses the vector, `irsign = 1` gives the
/// entries random signs, and `mode = +-6` draws them straight from `dist` instead.
/// The modes, as in the reference:
///
/// | `|mode|` | spectrum |
/// |---|---|
/// | 0 | all ones, `cond` ignored |
/// | 1 | one large value, the rest `1/cond` |
/// | 2 | one small value `1/cond`, the rest one |
/// | 3 | geometric: `cond**(-(i-1)/(n-1))` |
/// | 4 | arithmetic from 1 down to `1/cond` |
/// | 5 | log-uniform on `[1/cond, 1]` |
/// | 6 | `n` draws from `dist` |
///
/// Returns the reference's `INFO`: negative if an argument is out of range.
pub fn latm1(mode: i32, cond: f64, irsign: i32, dist: Dist, seed: &mut [u32; 4], d: &mut [f64]) -> Result<(), i32> {
    let n = d.len();
    // Decode and test the input parameters.
    let ordinary = mode != -6 && mode != 0 && mode != 6;
    let info = if !(-6..=6).contains(&mode) {
        -1
    } else if ordinary && irsign != 0 && irsign != 1 {
        -2
    } else if ordinary && cond < 1.0 {
        -3
    } else {
        // The reference's remaining check is `IDIST < 1 || IDIST > 3` for modes
        // +-6; `Dist` cannot hold a value outside that range, so it cannot fail.
        0
    };
    if info != 0 {
        return Err(info);
    }
    if n == 0 || mode == 0 {
        return Ok(());
    }

    // Compute D according to COND and MODE.
    match mode.abs() {
        1 => {
            d.fill(1.0 / cond);
            d[0] = 1.0;
        }
        2 => {
            d.fill(1.0);
            d[n - 1] = 1.0 / cond;
        }
        3 => {
            d[0] = 1.0;
            if n > 1 {
                let alpha = cond.powf(-1.0 / (n - 1) as f64);
                for i in 1..n {
                    d[i] = alpha.powi(i as i32);
                }
            }
        }
        4 => {
            d[0] = 1.0;
            if n > 1 {
                let temp = 1.0 / cond;
                let alpha = (1.0 - temp) / (n - 1) as f64;
                for i in 1..n {
                    // The reference reads D(I) = (N-I)*alpha + temp for I = 2..N,
                    // so the zero-based index sees one more.
                    d[i] = (n - i - 1) as f64 * alpha + temp;
                }
            }
        }
        5 => {
            let alpha = (1.0 / cond).ln();
            for v in d.iter_mut() {
                *v = (alpha * laran(seed)).exp();
            }
        }
        _ => larnv(dist, seed, n, d),
    }

    // Random signs, for the modes that build magnitudes.
    if ordinary && irsign == 1 {
        for v in d.iter_mut() {
            if laran(seed) > 0.5 {
                *v = -*v;
            }
        }
    }

    // Reverse for a negative MODE.
    if mode < 0 {
        d.reverse();
    }
    Ok(())
}

/// `DNRM2` as the reference BLAS computes it: the accumulation is rescaled by the
/// largest entry seen so far, so `ssq` is a running quantity. The crate's own
/// `dnrm2` divides by the global maximum instead, which is a different rounding --
/// fine for solving, but the harness exists to reproduce the reference's matrices,
/// and a difference of one bit in the norm moves every entry downstream of it.
fn dnrm2_reference(x: &[f64]) -> f64 {
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for &v in x {
        let absxi = v.abs();
        if absxi > 0.0 {
            if scale < absxi {
                let r = scale / absxi;
                ssq = 1.0 + ssq * r * r;
                scale = absxi;
            } else {
                let r = absxi / scale;
                ssq += r * r;
            }
        }
    }
    scale * ssq.sqrt()
}

/// Port of `DLARGE`: pre- and post-multiply `a` (column-major, `n x n`) by a random
/// orthogonal matrix, one Householder reflection at a time, drawn from the normal
/// distribution.
///
/// The reference builds each reflector from `DLARNV(3, ...)` and applies it as
/// `DLGEMV` + `DLGER` on both sides; the same two rank-one updates are written out
/// here, in the same order, so the arithmetic is identical.
pub fn large(n: usize, a: &mut [f64], seed: &mut [u32; 4]) {
    assert_eq!(a.len(), n * n, "large: a must be n*n");
    let mut w = vec![0.0f64; n];
    for i in (0..n).rev() {
        let len = n - i;
        let refl = &mut w[..len];
        larnv(Dist::Normal, seed, len, refl);
        let wn = dnrm2_reference(refl);
        // SIGN(WN, WORK(1)): the sign of the first entry, and + when it is zero.
        let wa = if refl[0] < 0.0 { -wn } else { wn };
        let tau = if wn == 0.0 {
            0.0
        } else {
            let wb = refl[0] + wa;
            if len > 1 {
                let inv = 1.0 / wb;
                for v in refl[1..].iter_mut() {
                    *v *= inv;
                }
            }
            refl[0] = 1.0;
            wb / wa
        };
        if tau == 0.0 {
            continue;
        }
        // A(i:n, 1:n) -= tau * w * (w^T A(i:n, 1:n)), column by column.
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..len {
                dot += w[k] * a[i + k + j * n];
            }
            if dot != 0.0 {
                for k in 0..len {
                    a[i + k + j * n] -= tau * w[k] * dot;
                }
            }
        }
        // A(1:n, i:n) -= tau * (A(1:n, i:n) w) * w^T, row by row.
        for r in 0..n {
            let mut dot = 0.0;
            for k in 0..len {
                dot += a[r + (i + k) * n] * w[k];
            }
            if dot != 0.0 {
                for k in 0..len {
                    a[r + (i + k) * n] -= tau * dot * w[k];
                }
            }
        }
    }
}

/// The arguments of `DLATME`, with the character switches as bools.
///
/// `kl` and `ku` are absent on purpose: the reference's bandwidth reduction only
/// runs when `kl < n-1` or `ku < n-1`, and the family sweeps want dense matrices,
/// where it is skipped. Passing `kl = ku = n-1` is what the wrapper does, so that
/// branch is not part of this port.
pub struct Latme {
    /// Distribution of the entries: `DIST`.
    pub dist: Dist,
    /// `MODE`: the spectrum's shape, as in [`latm1`]. `0` keeps the `d` given by the
    /// caller and is the mode a `.in` file exercises (the reference reads those
    /// eigenvalues from its input).
    pub mode: i32,
    /// `COND`: condition number of the spectrum.
    pub cond: f64,
    /// `DMAX`: target magnitude of the largest eigenvalue.
    pub dmax: f64,
    /// `EI`: only read when `mode == 0`. `true` marks a real eigenvalue; a `false`
    /// pairs with the entry before it to form a 2x2 block, exactly as LAPACK's `'I'`.
    pub real: Vec<bool>,
    /// `RSIGN`: random signs on the spectrum.
    pub rsign: bool,
    /// `UPPER`: fill the upper triangle with random numbers.
    pub upper: bool,
    /// `SIM`: apply the random orthogonal similarity.
    pub sim: bool,
    /// `MODES`: the shape of the eigenvector matrix's singular values.
    pub modes: i32,
    /// `CONDS`: condition number of those singular values.
    pub conds: f64,
    /// `ANORM`: target max-norm of the matrix; negative leaves it unscaled.
    pub anorm: f64,
}

/// Port of `DLATME`: the nonsymmetric matrix with a prescribed spectrum that
/// `ddrvev` feeds to `dgeev`.
///
/// The steps follow the reference's numbered comments: the spectrum from [`latm1`]
/// scaled by `dmax`, the diagonal of `a`, 2x2 blocks for complex pairs, the random
/// upper triangle, the two applications of [`large`] with the singular values of the
/// eigenvector matrix in between, and the final scaling to `anorm`.
///
/// Returns the reference's `INFO` on failure: `2` when the spectrum is all zeros
/// with a nonzero `dmax`, `3`/`4` when the similarity's own spectrum fails, `5` for a
/// zero singular value.
pub fn latme(p: &Latme, n: usize, a: &mut [f64], d: &mut [f64], seed: &mut [u32; 4]) -> Result<(), i32> {
    assert_eq!(a.len(), n * n, "latme: a must be n*n");
    assert_eq!(d.len(), n, "latme: d must be n");

    // 1) Decode and test the input parameters.
    if p.mode.abs() > 6 {
        return Err(-5);
    }
    if p.mode != 0 && p.mode.abs() != 6 && p.cond < 1.0 {
        return Err(-6);
    }
    if p.sim && p.modes.abs() > 5 {
        return Err(-13);
    }
    if p.sim && p.modes != 0 && p.conds < 1.0 {
        return Err(-14);
    }

    // The seed is normalized the reference's way: four 12-bit words, the last odd.
    for s in seed.iter_mut() {
        *s %= 4096;
    }
    if seed[3] % 2 != 1 {
        seed[3] += 1;
    }

    // 2) Set up the diagonal of A.
    latm1(p.mode, p.cond, p.rsign as i32, p.dist, seed, d).map_err(|_| 1)?;
    if p.mode != 0 && p.mode.abs() != 6 {
        let temp = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let alpha = if temp > 0.0 {
            p.dmax / temp
        } else if p.dmax != 0.0 {
            return Err(2);
        } else {
            0.0
        };
        for v in d.iter_mut() {
            *v *= alpha;
        }
    }
    a.fill(0.0);
    for i in 0..n {
        a[i + i * n] = d[i];
    }

    // Complex conjugate pairs.
    let pair = |a: &mut [f64], j: usize| {
        a[(j - 1) + j * n] = a[j + j * n];
        a[j + (j - 1) * n] = -a[j + j * n];
        a[j + j * n] = a[(j - 1) + (j - 1) * n];
    };
    if p.mode == 0 {
        for j in 1..n {
            if !p.real[j] {
                pair(a, j);
            }
        }
    } else if p.mode.abs() == 5 {
        for j in (1..n).step_by(2) {
            if laran(seed) > 0.5 {
                pair(a, j);
            }
        }
    }

    // 3) Random upper triangle, leaving the corners of a 2x2 block alone.
    if p.upper {
        for col in 1..n {
            let rows = if a[(col - 1) + col * n] != 0.0 { col - 1 } else { col };
            if rows > 0 {
                let start = col * n;
                larnv(p.dist, seed, rows, &mut a[start..start + rows]);
            }
        }
    }

    // 4) Similarity: X A X^-1 with X = U S V.
    if p.sim {
        let mut ds = vec![0.0f64; n];
        latm1(p.modes, p.conds, 0, Dist::Uniform, seed, &mut ds).map_err(|_| 3)?;
        large(n, a, seed);
        for j in 0..n {
            // Row j by DS(j), then column j by 1/DS(j).
            for k in 0..n {
                a[j + k * n] *= ds[j];
            }
            if ds[j] == 0.0 {
                return Err(5);
            }
            let inv = 1.0 / ds[j];
            for i in 0..n {
                a[i + j * n] *= inv;
            }
        }
        large(n, a, seed);
    }

    // 5) Scale the matrix to have max-norm ANORM.
    if p.anorm >= 0.0 {
        let temp = crate::native::blas::dlange('M', a, n, n);
        if temp > 0.0 {
            let alpha = p.anorm / temp;
            for v in a.iter_mut() {
                *v *= alpha;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seed every LAPACK test starts from (`ISEEDY` in the reference drivers).
    const SEEDY: [u32; 4] = [1988, 1989, 1990, 1991];

    /// Values captured from LAPACK 3.12.1's own `libtmglib` (`dlarnv_`) for
    /// `ISEEDY`, `n = 200`: the first twelve and the last two, plus the seed after
    /// the call. A transcription error in the multiplier table or in the carry
    /// arithmetic shows up at the first value.
    #[test]
    fn larnv_matches_the_reference_stream() {
        let cases: [(Dist, [f64; 14], [u32; 4]); 3] = [
            (
                Dist::Uniform,
                [
                    2.39108610563970814e-01,
                    4.59704946385553370e-01,
                    2.67450707486528216e-01,
                    5.37480921109899867e-01,
                    1.27181434037094476e-01,
                    3.77749846792422517e-01,
                    1.67801377711310096e-01,
                    2.89001310243659049e-01,
                    4.33095742785166493e-01,
                    5.21438670581879649e-02,
                    9.63368075751969144e-01,
                    3.70742051405354545e-01,
                    2.39748195870031822e-01,
                    6.98957663151670516e-01,
                ],
                [2862, 3811, 2824, 1639],
            ),
            (
                Dist::Symmetric,
                [
                    -5.21782778872058373e-01,
                    -8.05901072288932596e-02,
                    -4.65098585026943567e-01,
                    7.49618422197997347e-02,
                    -7.45637131925811047e-01,
                    -2.44500306415154967e-01,
                    -6.64397244577379809e-01,
                    -4.21997379512681903e-01,
                    -1.33808514429667014e-01,
                    -8.95712265883624070e-01,
                    9.26736151503938288e-01,
                    -2.58515897189290911e-01,
                    -5.20503608259936357e-01,
                    3.97915326303341033e-01,
                ],
                [2862, 3811, 2824, 1639],
            ),
            (
                Dist::Normal,
                [
                    -1.63771955772052569e+00,
                    -1.57925322198978191e+00,
                    -1.46061114264331215e+00,
                    -4.58389536009067289e-01,
                    1.22486082197039781e+00,
                    -1.87946296057196677e-01,
                    8.82352747833036077e-01,
                    8.64551373311237659e-01,
                    -5.82891802674063292e-02,
                    4.54479209347213420e-01,
                    1.18672119633679241e-01,
                    3.38440916077551557e-01,
                    -1.57964195594977208e+00,
                    9.87306889039197610e-01,
                ],
                [4005, 1507, 1338, 263],
            ),
        ];
        for (dist, want, want_seed) in cases {
            let mut seed = SEEDY;
            let mut x = vec![0.0; 200];
            larnv(dist, &mut seed, 200, &mut x);
            for (i, w) in want[..12].iter().enumerate() {
                assert_eq!(
                    x[i].to_bits(),
                    w.to_bits(),
                    "{dist:?}: x[{i}] = {:e}, reference {w:e}",
                    x[i]
                );
            }
            for (k, w) in want[12..].iter().enumerate() {
                assert_eq!(x[198 + k].to_bits(), w.to_bits(), "{dist:?}: x[{}]", 198 + k);
            }
            assert_eq!(seed, want_seed, "{dist:?}: seed after the call");
        }
    }

    /// The batch boundary: `DLARNV` fills 64 values per pass, so a length that is
    /// not a multiple of 64 must continue exactly where the previous pass stopped.
    #[test]
    fn larnv_is_continuous_across_batches() {
        for n in [1usize, 63, 64, 65, 129, 200] {
            let mut a = SEEDY;
            let mut all = vec![0.0; n];
            larnv(Dist::Uniform, &mut a, n, &mut all);

            // The same numbers, drawn one at a time, with the seed carried over.
            let mut b = SEEDY;
            let mut one = vec![0.0; n];
            for v in one.iter_mut() {
                larnv(Dist::Uniform, &mut b, 1, std::slice::from_mut(v));
            }
            assert_eq!(all, one, "n = {n}");
            assert_eq!(a, b, "n = {n}: seed");
        }
    }
    /// `DLATM1` pinned against the reference build, one case per mode.
    ///
    /// Values captured from LAPACK 3.12.1's `dlatm1_` with `n = 6` and `ISEEDY`; the
    /// cases cover the one-large and one-small spectra, the geometric and arithmetic
    /// ones, the log-uniform draw (`DLARAN`), the direct `DLARNV` draw, and a
    /// reversed spectrum with random signs. Modes 3 and 4 are where an off-by-one in
    /// the exponent or in the `n - i` factor shows up.
    #[test]
    fn latm1_matches_the_reference_spectra() {
        // (mode, cond, irsign, dist, |d|)
        let cases: [(i32, f64, i32, Dist, [f64; 6]); 7] = [
            (
                1,
                1e3,
                0,
                Dist::Uniform,
                [
                    1.0,
                    1e-3,
                    1.0000000000000000208e-3,
                    1.0000000000000000208e-3,
                    1.0000000000000000208e-3,
                    1.0000000000000000208e-3,
                ],
            ),
            (
                2,
                1e3,
                0,
                Dist::Uniform,
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0000000000000000208e-3],
            ),
            (
                3,
                1e6,
                0,
                Dist::Uniform,
                [
                    1.0,
                    6.30957344480193166e-2,
                    3.98107170553497168e-3,
                    2.51188643150957900e-4,
                    1.58489319246111277e-5,
                    9.99999999999999319e-7,
                ],
            ),
            (
                4,
                1e6,
                0,
                Dist::Uniform,
                [
                    1.0,
                    8.00000200000000050e-1,
                    6.00000399999999989e-1,
                    4.00000599999999984e-1,
                    2.00000800000000006e-1,
                    9.99999999999999955e-7,
                ],
            ),
            (
                5,
                1e4,
                0,
                Dist::Uniform,
                [
                    1.10551733715540487e-1,
                    1.44937315939409010e-2,
                    8.51524543305676590e-2,
                    7.08070197722824018e-3,
                    3.09937598639208056e-1,
                    3.08319230092024485e-2,
                ],
            ),
            (
                6,
                1.0,
                0,
                Dist::Normal,
                [
                    -1.63771955772052569,
                    -1.57925322198978191,
                    -1.46061114264331215,
                    -4.58389536009067289e-1,
                    1.22486082197039781,
                    -1.87946296057196677e-1,
                ],
            ),
            (
                -3,
                1e6,
                1,
                Dist::Uniform,
                [
                    9.99999999999999319e-7,
                    1.58489319246111277e-5,
                    -2.51188643150957900e-4,
                    3.98107170553497168e-3,
                    6.30957344480193166e-2,
                    1.0,
                ],
            ),
        ];
        for (mode, cond, irsign, dist, want) in cases {
            let mut seed = SEEDY;
            let mut d = [0.0f64; 6];
            latm1(mode, cond, irsign, dist, &mut seed, &mut d).expect("latm1");
            for (i, w) in want.iter().enumerate() {
                assert_eq!(
                    d[i].to_bits(),
                    w.to_bits(),
                    "mode {mode}, cond {cond:e}, irsign {irsign}, {dist:?}: d[{i}] = {:e}, reference {w:e}",
                    d[i]
                );
            }
        }
    }

    /// The argument checks the reference makes, and mode 0 with a zero length.
    #[test]
    fn latm1_rejects_what_the_reference_rejects() {
        let mut seed = SEEDY;
        let mut d = [0.0; 4];
        assert_eq!(latm1(7, 1.0, 0, Dist::Uniform, &mut seed, &mut d), Err(-1));
        assert_eq!(latm1(-7, 1.0, 0, Dist::Uniform, &mut seed, &mut d), Err(-1));
        assert_eq!(latm1(3, 1.0, 2, Dist::Uniform, &mut seed, &mut d), Err(-2));
        assert_eq!(latm1(3, 0.5, 0, Dist::Uniform, &mut seed, &mut d), Err(-3));
        // mode 0 and a negative mode ignore cond and irsign.
        assert_eq!(latm1(0, 0.5, 2, Dist::Uniform, &mut seed, &mut d), Ok(()));
        assert_eq!(d, [0.0; 4]);
        let mut empty: [f64; 0] = [];
        assert_eq!(latm1(3, 1e3, 0, Dist::Uniform, &mut seed, &mut empty), Ok(()));
    }
    /// `DLARGE` pinned against the reference build: a 4x4 matrix of small integers
    /// through the random orthogonal similarity, entry for entry, seed included.
    ///
    /// The comparison is at `1e-13` relative, not bit for bit: the reference applies
    /// each reflector with DGEMV+DGER from *its* BLAS, and the last bits of a dot
    /// product depend on that build's summation order (`DNRM2` alone took a running
    /// rescale to match). A transcription error moves these entries by O(1), so the
    /// tolerance still catches what this fixture exists to catch; the seed, which
    /// depends only on the draws, is compared exactly.
    #[test]
    fn large_matches_the_reference_orthogonal_similarity() {
        let n = 4usize;
        let mut a = [0.0f64; 16];
        for j in 0..n {
            for i in 0..n {
                a[i + j * n] = (10 * (i + 1) + (j + 1)) as f64;
            }
        }
        let mut seed = SEEDY;
        large(n, &mut a, &mut seed);

        let want: [f64; 16] = [
            2.3024147769153203e-1,
            -3.8619896079084226e-1,
            1.0582228391185904,
            -2.7512034544488131,
            -4.9125076648019270,
            -2.7454305584670942,
            -8.6945327915188884,
            2.9222577418995357e1,
            6.7788124210788379,
            2.5135150445843757,
            1.3608984701968311e1,
            -4.3745578309755906e1,
            -1.4896900680174987e1,
            -4.4904640952740786,
            -3.1212425521549907e1,
            9.8906204378807260e1,
        ];
        for j in 0..n {
            for i in 0..n {
                let got = a[i + j * n];
                let w = want[i + j * n];
                let tol = 1e-13 * w.abs().max(1.0);
                assert!(
                    (got - w).abs() <= tol,
                    "a({}, {}) = {got:e}, reference {w:e}",
                    i + 1,
                    j + 1
                );
            }
        }
        assert_eq!(seed, [523, 2892, 2970, 2903], "seed after the similarity");
    }
    /// `DLATME` pinned against the reference build, one case per mechanism.
    ///
    /// * case 1 — `MODE = 0` with `EI = "RIRI"`: the caller's spectrum, two 2x2
    ///   blocks, and the similarity with `MODES = 3`;
    /// * case 2 — `MODE = 3`, `DMAX = 10`, random signs, random upper triangle, all
    ///   of it conjugated (`MODES = 1`, `CONDS = 1e2`);
    /// * case 3 — `MODE = 5` with no similarity and `ANORM = 2`, which is the scaling
    ///   path on its own.
    ///
    /// Compared at 1e-13 relative for the same reason as [`large`]: the last bits
    /// belong to the reference's BLAS. The seed is compared exactly.
    #[test]
    fn latme_matches_the_reference_matrices() {
        let cases: [(Latme, usize, Option<[f64; 6]>, [[f64; 6]; 6], usize, [u32; 4]); 3] = [
            (
                Latme {
                    dist: Dist::Uniform,
                    mode: 0,
                    cond: 1.0,
                    dmax: 1.0,
                    real: vec![true, false, true, false, true, true],
                    rsign: false,
                    upper: false,
                    sim: true,
                    modes: 3,
                    conds: 1e2,
                    anorm: -1.0,
                },
                4,
                Some([1.0, 1.0, 2.0, 2.0, 0.0, 0.0]),
                [
                    [
                        -3.4358299714570606e1,
                        -3.1232682606951830e1,
                        2.1857744820860411e1,
                        -1.7318002903932477e1,
                        0.0,
                        0.0,
                    ],
                    [
                        5.9886570090465767e1,
                        5.2675038415048995e1,
                        -3.5490796756505702e1,
                        2.7429326504012714e1,
                        0.0,
                        0.0,
                    ],
                    [
                        -8.7771794030392680,
                        -9.1634094461913378,
                        8.4129308675480221,
                        -7.1512258361176526,
                        0.0,
                        0.0,
                    ],
                    [
                        -4.2671376152242544e1,
                        -3.7814331530471605e1,
                        2.6923357163109905e1,
                        -2.0729669568026424e1,
                        0.0,
                        0.0,
                    ],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                4,
                [3593, 2527, 3691, 1511],
            ),
            (
                Latme {
                    dist: Dist::Normal,
                    mode: 3,
                    cond: 1e3,
                    dmax: 10.0,
                    real: vec![true; 6],
                    rsign: true,
                    upper: true,
                    sim: true,
                    modes: 1,
                    conds: 1e2,
                    anorm: -1.0,
                },
                5,
                None,
                [
                    [
                        5.6818210826746565e1,
                        -1.2134452942258655e2,
                        7.0519304671240931e1,
                        2.3632111706016047e1,
                        -4.1501650066956316e1,
                        0.0,
                    ],
                    [
                        8.4625331783111903,
                        -1.7432470350849172e1,
                        9.8448507684683619,
                        3.0823716027616168,
                        -6.3561113226763801,
                        0.0,
                    ],
                    [
                        -1.4232958818183391e1,
                        2.9281440449711781e1,
                        -1.7690873786772720e1,
                        -6.0494189969748078,
                        9.8446755299916031,
                        0.0,
                    ],
                    [
                        1.4031453497192246e2,
                        -3.1220418686955975e2,
                        1.7479405622543953e2,
                        5.9505795852977720e1,
                        -1.0583721240354244e2,
                        0.0,
                    ],
                    [
                        9.1993945868447184e1,
                        -2.0618932806558507e2,
                        1.1637572936338884e2,
                        3.9916755411187836e1,
                        -6.9152389498565654e1,
                        0.0,
                    ],
                    [0.0; 6],
                ],
                5,
                [500, 161, 2574, 2627],
            ),
            (
                Latme {
                    dist: Dist::Uniform,
                    mode: 5,
                    cond: 1e2,
                    dmax: 1.0,
                    real: vec![true; 6],
                    rsign: false,
                    upper: false,
                    sim: false,
                    modes: 0,
                    conds: 0.0,
                    anorm: 2.0,
                },
                6,
                None,
                [
                    [1.1944710891888788, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 4.3249687215530030e-1, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0483141521287667, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0229504007241859e-1, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 6.3080215764373837e-1],
                ],
                6,
                [1773, 3932, 3380, 3219],
            ),
        ];
        for (k, (params, n, d_in, want, _, want_seed)) in cases.into_iter().enumerate() {
            let mut seed = SEEDY;
            let mut d = vec![0.0f64; n];
            if let Some(dv) = d_in {
                d[..n].copy_from_slice(&dv[..n]);
            }
            let mut a = vec![0.0f64; n * n];
            latme(&params, n, &mut a, &mut d, &mut seed).unwrap_or_else(|e| panic!("case {k}: INFO = {e}"));
            for i in 0..n {
                for j in 0..n {
                    let got = a[i + j * n];
                    let w = want[j][i];
                    let tol = 1e-13 * w.abs().max(1.0);
                    assert!(
                        (got - w).abs() <= tol,
                        "case {k}: a({}, {}) = {got:e}, reference {w:e}",
                        i + 1,
                        j + 1
                    );
                }
            }
            assert_eq!(seed, want_seed, "case {k}: seed after the call");
        }
    }

    /// The two failures the reference reports from the outside: a spectrum of zeros
    /// with a nonzero `DMAX`, and the argument checks.
    #[test]
    fn latme_reports_what_the_reference_reports() {
        let mut seed = SEEDY;
        let mut a = vec![0.0; 9];
        let mut d = vec![0.0; 3];
        let mut p = Latme {
            dist: Dist::Uniform,
            mode: 3,
            cond: 1e3,
            dmax: 1.0,
            real: vec![true; 3],
            rsign: false,
            upper: false,
            sim: false,
            modes: 0,
            conds: 0.0,
            anorm: -1.0,
        };
        p.cond = 0.5;
        assert_eq!(latme(&p, 3, &mut a, &mut d, &mut seed), Err(-6), "cond < 1");
        p.cond = 1e3;
        p.mode = 7;
        assert_eq!(latme(&p, 3, &mut a, &mut d, &mut seed), Err(-5), "|mode| > 6");
        p.mode = 3;
        p.sim = true;
        p.modes = 6;
        assert_eq!(latme(&p, 3, &mut a, &mut d, &mut seed), Err(-13), "|modes| > 5");
    }
}
