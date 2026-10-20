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

/// Port of `DLARND`: one number from the chosen distribution, drawn through
/// [`laran`] two at a time for the normal case, exactly as the reference does.
pub fn larnd(dist: Dist, seed: &mut [u32; 4]) -> f64 {
    let t1 = laran(seed);
    match dist {
        Dist::Uniform => t1,
        Dist::Symmetric => 2.0 * t1 - 1.0,
        Dist::Normal => {
            let t2 = laran(seed);
            (-2.0 * t1.ln()).sqrt() * (std::f64::consts::TAU * t2).cos()
        }
    }
}

/// The arguments of `DLATM4`, the generalized problem's matrix generator.
pub struct Latm4 {
    /// `ITYPE`: the shape of the matrix — identity, Jordan blocks, or a diagonal
    /// with one of the spectra `1..10`. A negative value reverses it.
    pub itype: i32,
    /// `NZ1`, `NZ2`: for `|itype| >= 4`, how many leading and trailing zeros the
    /// diagonal gets.
    pub nz1: i32,
    pub nz2: i32,
    /// `RCOND`: the ratio of the smallest to the largest diagonal entry.
    pub rcond: f64,
    /// `AMAGN`: the magnitude every entry is scaled to.
    pub amagn: f64,
    /// `ISIGN`: `0` no signs, `1` random signs, `2` random signs *and* random
    /// rotations that turn adjacent pairs into 2x2 blocks.
    pub isign: i32,
    /// `DIST`: the distribution for the random entries.
    pub dist: Dist,
    /// `TRIANG`: scale of the random upper triangle; zero leaves it empty.
    pub triang: f64,
}

/// Port of `DLATM4`: one of the matrices `dchkgg` builds the generalized problem
/// `(A, B)` from.
///
/// The reference is a single `GO TO` dispatch over `|ITYPE| = 1..10`; the cases are
/// written here as a `match`, and the rest of the routine (the `AMAGN` scaling, the
/// random signs, the reversal for a negative `ITYPE`, the rotations for `ISIGN = 2`,
/// and the upper triangle) follows in the same order.
pub fn latm4(p: &Latm4, n: usize, a: &mut [f64], seed: &mut [u32; 4]) {
    assert_eq!(a.len(), n * n, "latm4: a must be n*n");
    if n == 0 {
        return;
    }
    a.fill(0.0);
    if seed[3] % 2 != 1 {
        seed[3] += 1;
    }
    let itype = p.itype;
    if itype != 0 {
        let (kbeg, kend, klen) = if itype.abs() >= 4 {
            let kbeg = 1usize.max(n.min((p.nz1 + 1).max(0) as usize));
            let kend = kbeg.max(n.min((n as i32 - p.nz2).max(0) as usize));
            (kbeg, kend, kend + 1 - kbeg)
        } else {
            (1, n, n)
        };
        let isdb = 1usize;
        let mut isde = 0usize;
        // Zero-based helpers: the reference's JD is our jd - 1.
        let at = |a: &mut [f64], i: usize, j: usize, v: f64| a[(i - 1) + (j - 1) * n] = v;
        let get = |a: &[f64], i: usize, j: usize| a[(i - 1) + (j - 1) * n];
        match itype.abs() {
            1 => {
                for jd in 1..=n {
                    at(a, jd, jd, 1.0);
                }
            }
            2 => {
                for jd in 1..n {
                    at(a, jd + 1, jd, 1.0);
                }
                isde = n - 1;
            }
            3 => {
                let k = (n - 1) / 2;
                for jd in 1..=k {
                    at(a, jd + 1, jd, 1.0);
                }
                isde = k;
                for jd in (k + 2)..=(2 * k + 1) {
                    at(a, jd, jd, 1.0);
                }
            }
            4 => {
                for jd in kbeg..=kend {
                    at(a, jd, jd, jd as f64 - p.nz1 as f64);
                }
            }
            5 => {
                for jd in (kbeg + 1)..=kend {
                    at(a, jd, jd, p.rcond);
                }
                at(a, kbeg, kbeg, 1.0);
            }
            6 => {
                for jd in kbeg..kend {
                    at(a, jd, jd, 1.0);
                }
                at(a, kend, kend, p.rcond);
            }
            7 => {
                at(a, kbeg, kbeg, 1.0);
                if klen > 1 {
                    let alpha = p.rcond.powf(1.0 / (klen - 1) as f64);
                    for i in 2..=klen {
                        at(a, p.nz1 as usize + i, p.nz1 as usize + i, alpha.powi(i as i32 - 1));
                    }
                }
            }
            8 => {
                at(a, kbeg, kbeg, 1.0);
                if klen > 1 {
                    let alpha = (1.0 - p.rcond) / (klen - 1) as f64;
                    for i in 2..=klen {
                        let v = (klen - i) as f64 * alpha + p.rcond;
                        at(a, p.nz1 as usize + i, p.nz1 as usize + i, v);
                    }
                }
            }
            9 => {
                let alpha = p.rcond.ln();
                for jd in kbeg..=kend {
                    at(a, jd, jd, (alpha * laran(seed)).exp());
                }
            }
            _ => {
                for jd in kbeg..=kend {
                    at(a, jd, jd, larnd(p.dist, seed));
                }
            }
        }
        // Scale by AMAGN.
        for jd in kbeg..=kend {
            let v = get(a, jd, jd);
            at(a, jd, jd, p.amagn * v);
        }
        for jd in isdb..=isde {
            let v = get(a, jd + 1, jd);
            at(a, jd + 1, jd, p.amagn * v);
        }
        // Random signs on the diagonal and the subdiagonal.
        if p.isign > 0 {
            for jd in kbeg..=kend {
                let v = get(a, jd, jd);
                if v != 0.0 && laran(seed) > 0.5 {
                    at(a, jd, jd, -v);
                }
            }
            for jd in isdb..=isde {
                let v = get(a, jd + 1, jd);
                if v != 0.0 && laran(seed) > 0.5 {
                    at(a, jd + 1, jd, -v);
                }
            }
        }
        // Reverse for a negative ITYPE.
        if itype < 0 {
            for jd in kbeg..=((kbeg + kend - 1) / 2) {
                let lo = get(a, jd, jd);
                let hi = get(a, kbeg + kend - jd, kbeg + kend - jd);
                at(a, jd, jd, hi);
                at(a, kbeg + kend - jd, kbeg + kend - jd, lo);
            }
            for jd in 1..=((n - 1) / 2) {
                let lo = get(a, jd + 1, jd);
                let hi = get(a, n + 1 - jd, n - jd);
                at(a, jd + 1, jd, hi);
                at(a, n + 1 - jd, n - jd, lo);
            }
        }
        // ISIGN = 2: random rotations that make adjacent pairs into 2x2 blocks.
        if p.isign == 2 && itype != 2 && itype != 3 {
            let safmin = f64::MIN_POSITIVE;
            let mut jd = kbeg;
            while jd < kend {
                if laran(seed) > 0.5 {
                    let (mut cl, mut sl) = (2.0 * laran(seed) - 1.0, 2.0 * laran(seed) - 1.0);
                    let temp = 1.0 / safmin.max((cl * cl + sl * sl).sqrt());
                    cl *= temp;
                    sl *= temp;
                    let (mut cr, mut sr) = (2.0 * laran(seed) - 1.0, 2.0 * laran(seed) - 1.0);
                    let temp = 1.0 / safmin.max((cr * cr + sr * sr).sqrt());
                    cr *= temp;
                    sr *= temp;
                    let sv1 = get(a, jd, jd);
                    let sv2 = get(a, jd + 1, jd + 1);
                    at(a, jd, jd, cl * cr * sv1 + sl * sr * sv2);
                    at(a, jd + 1, jd, -sl * cr * sv1 + cl * sr * sv2);
                    at(a, jd, jd + 1, -cl * sr * sv1 + sl * cr * sv2);
                    at(a, jd + 1, jd + 1, sl * sr * sv1 + cl * cr * sv2);
                }
                jd += 2;
            }
        }
    }
    // Fill in the upper triangle, leaving 2x2 blocks alone.
    if p.triang != 0.0 {
        let ioff = if p.isign != 2 || itype == 2 || itype == 3 {
            1
        } else {
            for jr in 1..n {
                // The reference tests the subdiagonal, A(JR+1, JR), and fills
                // A(JR, JR+1) only where the pair is not a 2x2 block.
                if a[jr + (jr - 1) * n] == 0.0 {
                    let v = p.triang * larnd(p.dist, seed);
                    a[(jr - 1) + jr * n] = v;
                }
            }
            2
        };
        for jc in 2..=n {
            for jr in 1..=(jc - ioff) {
                let v = p.triang * larnd(p.dist, seed);
                a[(jr - 1) + (jc - 1) * n] = v;
            }
        }
    }
}

/// Which symmetry `DLATMS` builds into the matrix: its `SYM`.
///
/// `SYM = 'H'`, the Hermitian spelling the reference's complex twin uses, means
/// the same thing as `'S'` for a real matrix and is not a variant of its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DlatmsSym {
    /// `SYM = 'N'`: nonsymmetric, `A = U D V`, with the singular values in `D`.
    NonSymmetric,
    /// `SYM = 'P'`: symmetric, `A = U D U^T`, with non-negative eigenvalues in `D`.
    PositiveSymmetric,
    /// `SYM = 'S'`: symmetric, `A = U D U^T`, with random signs on `D` when `mode`
    /// is neither `0`, `6` nor `-6`.
    Symmetric,
}

/// How `DLATMS` stores the result: its `PACK`.
///
/// The triangular spellings `'U'` and `'L'` are `PACK` values, not `SYM` ones --
/// the reference's `SYM` takes `'N'`, `'P'`, `'S'` and `'H'` only -- so the two
/// half-matrix requests live here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DlatmsPack {
    /// `PACK = 'N'`: the whole matrix.
    Full,
    /// `PACK = 'U'`: symmetric, the strictly lower triangle zeroed.
    Upper,
    /// `PACK = 'L'`: symmetric, the strictly upper triangle zeroed.
    Lower,
}

/// The arguments of `DLATMS`.
///
/// `kl`, `ku` and `lda` are absent on purpose: the port builds the **dense**
/// matrix only, which is the case the SEP and bidiagonal families ask for
/// (`KL = KU = N`, clamped by the reference to the full bandwidth), and there the
/// bandwidth reduction runs zero iterations — neither `DLAGGE`'s nor `DLAGSY`'s
/// annihilation loop is entered. What that leaves out is everything the bandwidth
/// would otherwise decide:
///
/// - **Method B, the Givens rotations.** The reference switches to `DLAROT` when
///   the bandwidth is under `0.3 (m + n)` and `LDA >= M`, or when `LDA < M`. That
///   is the only route to the banded formats (`PACK = 'B'`, `'Q'`, `'Z'`) and to
///   `'C'`/`'R'`, so none of those are built here, and `DLAROT` is not ported.
/// - **The banded general case**, `KL < M-1` or `KU < N-1` under Method A. It is
///   reachable through `DLAGGE`'s annihilation loop, which is written for the
///   banded shapes the SVD and generalized sweeps do not need.
/// - **The diagonal shortcut** the reference takes before choosing a method
///   (`KL = KU = 0`), which only fires for a dense call when `m == n == 1`: the
///   condition is `LLB == 0 && UUB == 0` with `LLB = M-1` and `UUB = N-1`. That
///   case is handled here, because drawing a reflection for a `1 x 1` matrix
///   would disagree with the reference; `M x 1` and `1 x N` with `M, N > 1` do
///   not shortcut, and take the dense generation like any other matrix.
pub struct Dlatms {
    /// Distribution of the random entries: `DIST`.
    pub dist: Dist,
    /// `SYM`.
    pub sym: DlatmsSym,
    /// `MODE`: the shape of `D`, as in [`latm1`]. `0` keeps the `d` given by the
    /// caller.
    pub mode: i32,
    /// `COND`: condition number of the spectrum.
    pub cond: f64,
    /// `DMAX`: target magnitude of the largest entry of `D`.
    pub dmax: f64,
    /// `PACK`: the storage written.
    pub pack: DlatmsPack,
}

/// Port of `DLATMS`: a random matrix with a prescribed spectrum — the singular
/// values, or the eigenvalues when it is symmetric.
///
/// `a` is the `m x n` matrix in column-major order and `d` the `min(m, n)`-entry
/// spectrum, which is read when `mode == 0` and overwritten otherwise. The steps
/// are the reference's: the spectrum from [`latm1`], its scaling to `dmax`, the
/// random orthogonal factors by [`dlagge`] (nonsymmetric) or [`dlagsy`]
/// (symmetric), and the triangular zeroing `pack` asks for.
///
/// Returns the reference's `INFO` on failure: `-1` for `m != n` with a symmetric
/// `sym`, `-7` for `|mode| > 6`, `-8` for `cond < 1` where `mode` takes a
/// condition number, `-12` for a triangular `pack` with a nonsymmetric `sym`, `1`
/// when [`latm1`] rejects the mode, and `2` when the spectrum is all zeros.
pub fn dlatms(p: &Dlatms, m: usize, n: usize, a: &mut [f64], d: &mut [f64], seed: &mut [u32; 4]) -> Result<(), i32> {
    assert_eq!(a.len(), m * n, "dlatms: a must be m*n");
    let mnmin = m.min(n);
    assert_eq!(d.len(), mnmin, "dlatms: d must have min(m, n) entries");
    if m == 0 || n == 0 {
        return Ok(());
    }

    // 1) Decode and test the input parameters. The reference's negative lengths
    // (`INFO = -1`, `-2`) are unrepresentable in `usize`, so M != N is the only
    // shape check left; `INFO = -3`, `-5`, `-10`, `-11`, `-14` belong to `dist`,
    // `sym`, `kl`, `ku` and `lda`, which are enums or absent here.
    if p.sym != DlatmsSym::NonSymmetric && m != n {
        return Err(-1);
    }
    if p.mode.abs() > 6 {
        return Err(-7);
    }
    if p.mode != 0 && p.mode.abs() != 6 && p.cond < 1.0 {
        return Err(-8);
    }
    if p.sym == DlatmsSym::NonSymmetric && p.pack != DlatmsPack::Full {
        return Err(-12);
    }

    // The seed is normalized the reference's way: four 12-bit words, the last odd.
    for s in seed.iter_mut() {
        *s %= 4096;
    }
    if seed[3] % 2 != 1 {
        seed[3] += 1;
    }

    // 2) Set up D. Only `SYM = 'S'` and `'H'` sign it randomly.
    let irsign = if p.sym == DlatmsSym::Symmetric { 1 } else { 0 };
    latm1(p.mode, p.cond, irsign, p.dist, seed, d).map_err(|_| 1)?;
    if p.mode != 0 && p.mode.abs() != 6 {
        let temp = d.iter().fold(0.0f64, |mx, v| mx.max(v.abs()));
        if temp > 0.0 {
            let alpha = p.dmax / temp;
            for v in d.iter_mut() {
                *v *= alpha;
            }
        } else {
            return Err(2);
        }
    }

    // 3) Generate the dense matrix. `m == n == 1` is the reference's diagonal
    // shortcut (`LLB = UUB = 0`): it copies D and returns before choosing a
    // method. For a 1 x 1 the dense generation would otherwise draw a reflection
    // the reference never draws.
    if m == 1 && n == 1 {
        a.fill(0.0);
        for i in 0..mnmin {
            a[i + i * m] = d[i];
        }
    } else if p.sym == DlatmsSym::NonSymmetric {
        dlagge(m, n, d, a, seed);
    } else {
        dlagsy(m, d, a, seed);
    }

    // 4) Pack: zero the triangle the caller said to ignore. The reference packs
    // here because Method A always generates the full matrix (`IPACKG = 0`).
    match p.pack {
        DlatmsPack::Full => {}
        DlatmsPack::Upper => {
            for j in 0..n {
                for i in (j + 1)..n {
                    a[i + j * n] = 0.0;
                }
            }
        }
        DlatmsPack::Lower => {
            for j in 1..n {
                for i in 0..j {
                    a[i + j * n] = 0.0;
                }
            }
        }
    }
    Ok(())
}

/// Port of `DLAGGE` for the dense case: `A = U D V` with `U` and `V` random
/// orthogonal, built as one Householder reflection at a time from both sides.
///
/// The reference's bandwidth reduction (`DO 70 I = 1, MAX(M-1-KL, N-1-KU)`) is
/// not part of this port — with `KL = M-1` and `KU = N-1` its trip count is zero,
/// which is the only bandwidth the family sweeps ask for.
fn dlagge(m: usize, n: usize, d: &[f64], a: &mut [f64], seed: &mut [u32; 4]) {
    assert_eq!(a.len(), m * n, "dlagge: a must be m*n");
    assert!(d.len() >= m.min(n), "dlagge: d is too short");
    a.fill(0.0);
    for i in 0..m.min(n) {
        a[i + i * m] = d[i];
    }

    // The reference's WORK holds both the reflector and its image; they are kept
    // apart here, which changes no arithmetic.
    let lw = m.max(n);
    let mut w = vec![0.0f64; lw];
    let mut y = vec![0.0f64; lw];

    // `DO 40 I = MIN(M, N), 1, -1`, with the two blocks guarded by `I < M` and
    // `I < N` -- that is `i + 1 < m` and `i + 1 < n` in zero-based terms, so the
    // first iteration draws nothing when the matrix is square.
    for i in (0..m.min(n)).rev() {
        if i + 1 < m {
            // A(i:m, i:n) -= tau * w * (w^T A(i:m, i:n))
            let len = m - i;
            larnv(Dist::Normal, seed, len, &mut w[..len]);
            let wn = dnrm2_reference(&w[..len]);
            let wa = if w[0] < 0.0 { -wn } else { wn };
            let tau = if wn == 0.0 {
                0.0
            } else {
                let wb = w[0] + wa;
                let inv = 1.0 / wb;
                for v in w[1..len].iter_mut() {
                    *v *= inv;
                }
                w[0] = 1.0;
                wb / wa
            };
            if tau != 0.0 {
                let ncols = n - i;
                for j in 0..ncols {
                    let mut sum = 0.0;
                    for k in 0..len {
                        sum += a[(i + k) + (i + j) * m] * w[k];
                    }
                    y[j] = sum;
                }
                for j in 0..ncols {
                    for k in 0..len {
                        a[(i + k) + (i + j) * m] -= tau * w[k] * y[j];
                    }
                }
            }
        }
        if i + 1 < n {
            // A(i:m, i:n) -= tau * (A(i:m, i:n) w) * w^T
            let len = n - i;
            larnv(Dist::Normal, seed, len, &mut w[..len]);
            let wn = dnrm2_reference(&w[..len]);
            let wa = if w[0] < 0.0 { -wn } else { wn };
            let tau = if wn == 0.0 {
                0.0
            } else {
                let wb = w[0] + wa;
                let inv = 1.0 / wb;
                for v in w[1..len].iter_mut() {
                    *v *= inv;
                }
                w[0] = 1.0;
                wb / wa
            };
            if tau != 0.0 {
                let nrows = m - i;
                for k in 0..nrows {
                    let mut sum = 0.0;
                    for j in 0..len {
                        sum += a[(i + k) + (i + j) * m] * w[j];
                    }
                    y[k] = sum;
                }
                for j in 0..len {
                    for k in 0..nrows {
                        a[(i + k) + (i + j) * m] -= tau * y[k] * w[j];
                    }
                }
            }
        }
    }
}

/// Port of `DLAGSY` for the dense case: `A = U D U^T` on the lower triangle, one
/// symmetric rank-2 update at a time, then mirrored into the upper one.
///
/// The reference's bandwidth reduction (`DO 60 I = 1, N-1-K`) is not part of this
/// port: with `K = N-1` its trip count is zero. The `DSYMV` product is written in
/// the reference BLAS's order, column by column, because the running `t2` sum is
/// what makes the result bit-identical rather than merely accurate.
fn dlagsy(n: usize, d: &[f64], a: &mut [f64], seed: &mut [u32; 4]) {
    assert_eq!(a.len(), n * n, "dlagsy: a must be n*n");
    assert!(d.len() >= n, "dlagsy: d is too short");
    a.fill(0.0);
    for i in 0..n {
        a[i + i * n] = d[i];
    }
    let mut w = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];

    for i in (0..n.saturating_sub(1)).rev() {
        let len = n - i;
        larnv(Dist::Normal, seed, len, &mut w[..len]);
        let wn = dnrm2_reference(&w[..len]);
        let wa = if w[0] < 0.0 { -wn } else { wn };
        let tau = if wn == 0.0 {
            0.0
        } else {
            let wb = w[0] + wa;
            let inv = 1.0 / wb;
            for v in w[1..len].iter_mut() {
                *v *= inv;
            }
            w[0] = 1.0;
            wb / wa
        };
        if tau == 0.0 {
            continue;
        }
        // y := tau * A(i:n, i:n) * w, DSYMV('Lower') with beta = 0.
        //
        // The reference takes the diagonal term out of the inner loop and adds it
        // to `y[j]` on its own: the loop runs over `i = j+1..n` and the running
        // `t2` gathers the row-`j` contributions from the columns to its right.
        // Keeping the diagonal inside the loop doubles it -- measured, and it was
        // this port's first bug in the routine.
        for v in y[..len].iter_mut() {
            *v = 0.0;
        }
        for j in 0..len {
            let t1 = tau * w[j];
            let mut t2 = 0.0;
            y[j] += t1 * a[(i + j) + (i + j) * n];
            for k in (j + 1)..len {
                let akj = a[(i + k) + (i + j) * n];
                y[k] += t1 * akj;
                t2 += akj * w[k];
            }
            y[j] += tau * t2;
        }
        // v := y - 1/2 tau (y, w) w
        let mut dot = 0.0;
        for k in 0..len {
            dot += y[k] * w[k];
        }
        let alpha = -0.5 * tau * dot;
        for k in 0..len {
            y[k] += alpha * w[k];
        }
        // A(i:n, i:n) -= w v^T + v w^T, DSYR2('Lower', -1).
        for j in 0..len {
            if w[j] != 0.0 || y[j] != 0.0 {
                let t1 = -y[j];
                let t2 = -w[j];
                for k in j..len {
                    a[(i + k) + (i + j) * n] += w[k] * t1 + y[k] * t2;
                }
            }
        }
    }

    // The reference finishes by mirroring the lower triangle into the upper one.
    for j in 0..n {
        for i in (j + 1)..n {
            a[j + i * n] = a[i + j * n];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Complex64;
    use crate::native::lapack::checkers::get22;
    use crate::native::lapack::eig::dgeev;
    use crate::native::lapack::qz::dggev;

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
    /// The sweep `ddrvev` runs: `dgeev` over the reference's nonsymmetric families.
    ///
    /// The 21 cases of `ddrvev.f` are `KTYPE x KMAGN x KMODE x KCONDS`, where the
    /// spectrum's *mode* is `KMODE` and its magnitude `KMAGN` (O(1), large, small).
    /// What this test reproduces is that grid: every mode the table uses, at each of
    /// the three magnitudes. `COND` is fixed at 1e4 where the mode takes one, since
    /// the reference derives it from `KCONDS` and input files that are not part of
    /// this harness.
    ///
    /// The bound is the reference's own: `||A v - lambda v|| / (n ||A|| eps) <= 20`
    /// per eigenpair, with the returned vectors.
    ///
    /// Only the right vectors are checked, and that is a measured decision rather
    /// than a convenience. Asking the same question of the left ones gives ratios
    /// around `1e15` -- from LAPACK's own `dgeev` on the same matrix as from this
    /// port -- because the similarity that builds these families leaves the
    /// eigenvector matrix numerically singular (`dgecon` reports `rcond = 0`). The
    /// left residual is meaningless there, which is why the left vectors are checked
    /// in `dgeev_handles_matrices_at_the_edges_of_the_range`, on triangular
    /// matrices, where they are well conditioned.
    #[test]
    fn dgeev_solves_the_reference_families() {
        let n = 6usize;
        let eps = f64::EPSILON;
        let mut worst_ratio = 0.0f64;
        for mode in [0i32, 1, 2, 3, 4, 5] {
            for dmax in [1.0f64, 1e10, 1e-10] {
                let params = Latme {
                    dist: Dist::Uniform,
                    mode,
                    cond: if mode == 0 { 1.0 } else { 1e4 },
                    dmax,
                    // mode 0 keeps the caller's spectrum; alternate real and pairs so
                    // both the diagonal-only and the 2x2-block paths are swept.
                    real: vec![true, false, true, false, true, false],
                    rsign: mode != 0,
                    upper: mode != 0,
                    sim: true,
                    modes: if mode == 0 { 4 } else { 3 },
                    conds: 10.0,
                    anorm: -1.0,
                };
                let mut seed = SEEDY;
                let mut d = vec![1.0; n];
                if mode == 0 {
                    // A prescribed spectrum with a pair: two real values, two blocks.
                    d = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
                    for v in d.iter_mut() {
                        *v *= dmax;
                    }
                }
                let mut a0 = vec![0.0f64; n * n];
                latme(&params, n, &mut a0, &mut d, &mut seed).expect("latme");

                let mut a = a0.clone();
                let mut v_real = vec![0.0; n * n];
                let mut v_imag = vec![0.0; n * n];
                let (wr, wi) = dgeev(&mut v_real, &mut v_imag, &mut a, n).expect("dgeev");

                let anorm = a0.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                for j in 0..n {
                    assert!(
                        wr[j].is_finite() && wi[j].is_finite(),
                        "mode {mode}, dmax {dmax:e}: eigenvalue {j} is not finite"
                    );
                }
                let mut j = 0usize;
                while j < n {
                    let (lr, li) = (wr[j], wi[j]);
                    let complex = li != 0.0;
                    let mut worst = 0.0f64;
                    for i in 0..n {
                        let mut row = 0.0;
                        let mut row_i = 0.0;
                        for k in 0..n {
                            row += a0[i + k * n] * v_real[k + j * n];
                            row_i += a0[i + k * n] * v_imag[k + j * n];
                        }
                        let dr = row - (lr * v_real[i + j * n] - li * v_imag[i + j * n]);
                        let di = row_i - (lr * v_imag[i + j * n] + li * v_real[i + j * n]);
                        worst = worst.max(dr.abs().max(di.abs()));
                    }
                    let ratio = worst / (n as f64 * anorm * eps);
                    worst_ratio = worst_ratio.max(ratio);
                    assert!(ratio <= 20.0, "mode {mode}, dmax {dmax:e}, j {j}: ratio = {ratio:e}");
                    j += if complex { 2 } else { 1 };
                }
            }
        }
        println!("worst ratio over the families: {worst_ratio:e}");
    }

    /// `DLATM4` pinned against the reference build, one case per mechanism.
    ///
    /// The five cases cover the identity and the Jordan shapes, the `NZ1`/`NZ2`
    /// offsets of `|ITYPE| = 4`, the `ISIGN = 2` rotations that turn adjacent pairs
    /// into 2x2 blocks, and the reversal a negative `ITYPE` performs. Values captured
    /// from `dlatm4_` compiled from `TESTING/EIG/dlatm4.f`; the seed is compared
    /// exactly, which also pins how many draws each branch makes -- the two cases
    /// with no random entries leave it untouched.
    #[test]
    fn latm4_matches_the_reference_matrices() {
        // (params, n, reference in column-major order, seed after)
        let cases: [(Latm4, usize, &[f64], [u32; 4]); 5] = [
            // identity, scaled to AMAGN = 2, with a random upper triangle
            (
                Latm4 {
                    itype: 1,
                    nz1: 0,
                    nz2: 0,
                    rcond: 0.001,
                    amagn: 2.0,
                    isign: 0,
                    dist: Dist::Uniform,
                    triang: 0.5,
                },
                4,
                &[
                    2.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    1.1955430528198541e-01,
                    2.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    2.2985247319277671e-01,
                    1.3372535374326411e-01,
                    2.0000000000000000e+00,
                    0.0000000000000000e+00,
                    2.6874046055494988e-01,
                    6.3590717018547238e-02,
                    1.8887492339621131e-01,
                    2.0000000000000000e+00,
                ],
                [1547, 1078, 3168, 2783],
            ),
            // transposed Jordan block with random signs
            (
                Latm4 {
                    itype: 2,
                    nz1: 0,
                    nz2: 0,
                    rcond: 0.001,
                    amagn: 1.0,
                    isign: 1,
                    dist: Dist::Uniform,
                    triang: 0.3,
                },
                5,
                &[
                    0.0000000000000000e+00,
                    1.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    3.8154430211128340e-02,
                    0.0000000000000000e+00,
                    1.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    1.1332495403772670e-01,
                    5.0340413313393027e-02,
                    0.0000000000000000e+00,
                    1.0000000000000000e+00,
                    0.0000000000000000e+00,
                    8.6700393073097709e-02,
                    1.2992872283554990e-01,
                    1.5643160117456392e-02,
                    0.0000000000000000e+00,
                    -1.0000000000000000e+00,
                    2.8901042272559069e-01,
                    1.1122261542160640e-01,
                    1.6502349860833510e-01,
                    2.6983865223542020e-01,
                    0.0000000000000000e+00,
                ],
                [3684, 807, 733, 895],
            ),
            // diagonal 1..k with NZ1 leading and NZ2 trailing zeros
            (
                Latm4 {
                    itype: 4,
                    nz1: 1,
                    nz2: 2,
                    rcond: 0.001,
                    amagn: 1.0,
                    isign: 0,
                    dist: Dist::Uniform,
                    triang: 0.0,
                },
                7,
                &[
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    1.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    2.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    3.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    4.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                ],
                [1988, 1989, 1990, 1991],
            ),
            // log-uniform diagonal with ISIGN = 2 rotations turning pairs into 2x2 blocks
            (
                Latm4 {
                    itype: 9,
                    nz1: 0,
                    nz2: 0,
                    rcond: 0.01,
                    amagn: 1.0,
                    isign: 2,
                    dist: Dist::Uniform,
                    triang: 0.25,
                },
                4,
                &[
                    3.3249320852543812e-01,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    2.4084201893799231e-01,
                    1.2038991483484360e-01,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    2.2486554352951679e-01,
                    9.2685512851338636e-02,
                    2.9180893463115148e-01,
                    0.0000000000000000e+00,
                    1.3779487170530569e-01,
                    2.6125389868645321e-02,
                    1.3751958217361260e-01,
                    8.4146907116234765e-02,
                ],
                [428, 157, 965, 775],
            ),
            // one large value, reversed by the negative ITYPE
            (
                Latm4 {
                    itype: -5,
                    nz1: 0,
                    nz2: 0,
                    rcond: 0.1,
                    amagn: 3.0,
                    isign: 0,
                    dist: Dist::Uniform,
                    triang: 0.0,
                },
                5,
                &[
                    2.9999999999999999e-01,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    2.9999999999999999e-01,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    2.9999999999999999e-01,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    2.9999999999999999e-01,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    3.0000000000000000e+00,
                ],
                [1988, 1989, 1990, 1991],
            ),
        ];
        for (k, (params, n, want, want_seed)) in cases.into_iter().enumerate() {
            let mut seed = SEEDY;
            let mut a = vec![0.0f64; n * n];
            latm4(&params, n, &mut a, &mut seed);
            for idx in 0..n * n {
                let got = a[idx];
                let w = want[idx];
                let tol = 1e-13 * w.abs().max(1.0);
                assert!((got - w).abs() <= tol, "case {k}: a[{idx}] = {got:e}, reference {w:e}");
            }
            assert_eq!(seed, want_seed, "case {k}: seed after the call");
        }
    }
    /// `DLATMS` pinned against the reference build, case by case.
    ///
    /// Twelve cases: the dense general matrix at three aspect ratios, `MODE = 0`
    /// with a prescribed spectrum and `MODE = 6` with a random one, the symmetric
    /// forms with `SYM = 'S'` (random signs on the spectrum) and `'P'` (none), the
    /// two triangular packings, the `MODE < 0` reversal at `COND = 1e14`, the `1 x 1`
    /// and `N x 1` diagonal shortcuts, and a `2 x 2`.
    ///
    /// The seed after each call is compared exactly, which pins not just the values
    /// but the order and the count of the draws: a port that draws the right numbers
    /// in the wrong order still fails here, and that was the shape of every
    /// transcription bug in this module so far.
    ///
    /// `d` is passed in as the expected spectrum, which is correct for both halves:
    /// `MODE = 0` keeps what the caller gave it, and every other mode overwrites it.
    #[test]
    fn dlatms_matches_the_reference_matrices() {
        let d1: [f64; 5] = [
            1.0,
            0.1778279410038923,
            0.0316227766016838,
            0.005623413251903492,
            0.0010000000000000002,
        ];
        let a1: [f64; 25] = [
            0.34947618210725895,
            -0.3500407307662794,
            -0.03623859928904221,
            0.43458919724104705,
            0.10093006686815045, // column 1
            -0.19482004171386136,
            0.18008528903496984,
            0.0805346482341642,
            -0.20992597033142693,
            0.005061242779380713, // column 2
            0.11228271868242527,
            -0.14514089638932967,
            -0.06662409001525742,
            0.10039693313409319,
            -0.028125422974753077, // column 3
            0.06296619045603738,
            -0.07004807981653652,
            -0.06915579720032988,
            0.052560479319858786,
            -0.039047188007593395, // column 4
            -0.31983077789978837,
            0.2929893692413704,
            -0.025744238239034135,
            -0.4337853075195523,
            -0.15089266867457166, // column 5
        ];
        let d2: [f64; 4] = [10000000000.0, 10000.0, 10000.0, 10000.0];
        let a2: [f64; 24] = [
            4770788363.882683,
            -3744274313.070016,
            3644115892.749751,
            393935677.41646165,
            -4664894687.688498,
            -1067619599.4919305, // column 1
            -2410165637.096653,
            1891591976.769024,
            -1840982637.7584848,
            -199014075.53301436,
            2356670435.6023116,
            539354053.8868693, // column 2
            1425033288.1263006,
            -1118415774.6655731,
            1088499984.60656,
            117660701.9353508,
            -1393402549.5862982,
            -318903942.75177187, // column 3
            727459536.6919003,
            -570937601.4896286,
            555656053.0636292,
            60066514.71434181,
            -711319957.3847727,
            -162795640.34023362, // column 4
        ];
        let d3: [f64; 4] = [1e-10, 1e-10, 1e-10, 1e-12];
        let a3: [f64; 24] = [
            2.966176059666874e-11,
            -7.409087611884273e-11,
            1.2564656051463384e-11,
            -7.97982345360008e-12, // column 1
            1.0119741518886153e-11,
            -7.562612347112737e-12,
            7.500852201627117e-12,
            4.3324435176892676e-12, // column 2
            4.320032230366921e-11,
            -3.1894751767464455e-11,
            3.4809759603949126e-11,
            -3.996993823592404e-12, // column 3
            -3.9655960639169344e-11,
            -4.432825200894596e-11,
            -5.859436410627548e-11,
            2.2838921699193816e-11, // column 4
            2.1854777550169392e-11,
            -3.4972646349609665e-12,
            -4.982356013041861e-12,
            8.999131751407973e-11, // column 5
            2.1809074497100447e-11,
            3.35755697073258e-11,
            2.158486885254907e-11,
            2.982969927986295e-11, // column 6
        ];
        let d4: [f64; 4] = [4.0, 3.0, 2.0, 1.0];
        let a4: [f64; 16] = [
            -1.7633691453242482,
            1.3132480164870701,
            0.5797952918176301,
            1.9353518466185577, // column 1
            -2.052796122607038,
            -0.29560147731796005,
            -1.2801078303810702,
            -1.6610302269006387, // column 2
            -0.33918311392703726,
            -2.2589766479821485,
            0.9308219576414839,
            -0.9407789376876112, // column 3
            -1.0140108188642079,
            -1.5885226168245183,
            -1.1987593064306756,
            -0.653413192709643, // column 4
        ];
        let d5: [f64; 3] = [0.2391086105639708, 0.45970494638555337, 0.2674507074865282];
        let a5: [f64; 15] = [
            0.258216593237514,
            0.2128668781174073,
            0.09248150521675896,
            -0.1472347627783297,
            0.03443799166011409, // column 1
            -0.003981287109794673,
            0.238810312801468,
            -0.06552669724068982,
            0.10477341450825534,
            0.005757511298218453, // column 2
            0.16316934662674026,
            0.07135166514880575,
            -0.1318972108790163,
            -0.10790891438831977,
            0.25203048363142855, // column 3
        ];
        let d6: [f64; 5] = [
            1.0,
            0.1778279410038923,
            0.0316227766016838,
            -0.005623413251903492,
            0.0010000000000000002,
        ];
        let a6: [f64; 25] = [
            0.029461178707149482,
            -0.058684604527388134,
            0.047267964073456,
            0.06054036628673165,
            -0.14194140890920703, // column 1
            -0.058684604527388134,
            0.34046962988745094,
            -0.10273822060751979,
            -0.08740629969433211,
            0.2981651759031996, // column 2
            0.047267964073456,
            -0.10273822060751979,
            0.11195299968752104,
            0.07975112004832606,
            -0.23732203652927125, // column 3
            0.06054036628673165,
            -0.08740629969433211,
            0.07975112004832606,
            0.10589819888840743,
            -0.23752352675576444, // column 4
            -0.14194140890920703,
            0.2981651759031996,
            -0.23732203652927125,
            -0.23752352675576444,
            0.6170452971831442, // column 5
        ];
        let d7: [f64; 5] = [
            3.566902957270166e-11,
            4.6763386106029544e-12,
            2.7474064038836372e-11,
            2.284557281309629e-12,
            1e-10,
        ];
        let a7: [f64; 25] = [
            1.403402819047037e-11,
            0.0,
            0.0,
            0.0,
            0.0, // column 1
            -5.491868229251435e-12,
            4.1727887558099137e-11,
            0.0,
            0.0,
            0.0, // column 2
            -1.4724665225103058e-11,
            9.318370403439242e-12,
            2.4657826981078578e-11,
            0.0,
            0.0, // column 3
            5.5815670100303985e-12,
            -2.453470804881488e-11,
            -1.795058393446134e-11,
            8.110087671056663e-11,
            0.0, // column 4
            7.319355632171069e-12,
            -5.78509811993597e-12,
            -6.902161207076551e-12,
            -2.162192019870387e-12,
            8.583370063235891e-12, // column 5
        ];
        let d8: [f64; 5] = [10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 100000000.0];
        let a8: [f64; 25] = [
            8236041978.022697,
            -246282033.88396597,
            2068179110.815214,
            -2738970491.3992305,
            1584820963.6722574, // column 1
            0.0,
            9965614351.669193,
            288757074.4325004,
            -382412288.1415422,
            221271099.0920438, // column 2
            0.0,
            0.0,
            7575132298.433214,
            3211347143.681659,
            -1858147172.7852542, // column 3
            0.0,
            0.0,
            0.0,
            5747087368.696882,
            2460816980.6577473, // column 4
            0.0,
            0.0,
            0.0,
            0.0,
            8576124003.178024, // column 5
        ];
        let d9: [f64; 4] = [
            -1.0000000000000019e-14,
            4.6415888336127846e-10,
            2.154434690031885e-05,
            1.0,
        ];
        let a9: [f64; 16] = [
            6.141026689124035e-05,
            0.004435301378412105,
            0.0019516794995135562,
            -0.005743219274588545, // column 1
            0.004435301378412105,
            0.34829925058264555,
            0.15395786782475163,
            -0.45084825568525666, // column 2
            0.0019516794995135562,
            0.15395786782475163,
            0.06806949182007169,
            -0.19928369170791307, // column 3
            -0.005743219274588545,
            -0.45084825568525666,
            -0.19928369170791307,
            0.5835913921414408, // column 4
        ];
        let d10: [f64; 1] = [1.0];
        let a10: [f64; 1] = [
            1.0, // column 1
        ];
        let d11: [f64; 1] = [1.0];
        let a11: [f64; 3] = [
            -0.6057510633139085,
            -0.5841258437395268,
            -0.5402431378274554, // column 1
        ];
        let d12: [f64; 2] = [1.0, 0.1];
        let a12: [f64; 4] = [
            0.585072443172995,
            0.4486311666947325, // column 1
            0.4486311666947325,
            0.5149275568270046, // column 2
        ];

        let cases: [(Dlatms, usize, usize, [u32; 4], &[f64], &[f64]); 12] = [
            (
                Dlatms {
                    dist: Dist::Uniform,
                    sym: DlatmsSym::NonSymmetric,
                    mode: 3,
                    cond: 1000.0,
                    dmax: 1.0,
                    pack: DlatmsPack::Full,
                },
                5,
                5,
                [3609, 4014, 839, 2343],
                &d1,
                &a1,
            ),
            (
                Dlatms {
                    dist: Dist::Symmetric,
                    sym: DlatmsSym::NonSymmetric,
                    mode: 1,
                    cond: 1000000.0,
                    dmax: 10000000000.0,
                    pack: DlatmsPack::Full,
                },
                6,
                4,
                [1205, 1950, 294, 671],
                &d2,
                &a2,
            ),
            (
                Dlatms {
                    dist: Dist::Normal,
                    sym: DlatmsSym::NonSymmetric,
                    mode: 2,
                    cond: 100.0,
                    dmax: 1e-10,
                    pack: DlatmsPack::Full,
                },
                4,
                6,
                [1205, 1950, 294, 671],
                &d3,
                &a3,
            ),
            (
                Dlatms {
                    dist: Dist::Uniform,
                    sym: DlatmsSym::NonSymmetric,
                    mode: 0,
                    cond: 1.0,
                    dmax: 1.0,
                    pack: DlatmsPack::Full,
                },
                4,
                4,
                [585, 644, 3246, 663],
                &d4,
                &a4,
            ),
            (
                Dlatms {
                    dist: Dist::Uniform,
                    sym: DlatmsSym::NonSymmetric,
                    mode: 6,
                    cond: 1.0,
                    dmax: 1.0,
                    pack: DlatmsPack::Full,
                },
                5,
                3,
                [468, 100, 374, 2435],
                &d5,
                &a5,
            ),
            (
                Dlatms {
                    dist: Dist::Uniform,
                    sym: DlatmsSym::Symmetric,
                    mode: 3,
                    cond: 1000.0,
                    dmax: 1.0,
                    pack: DlatmsPack::Full,
                },
                5,
                5,
                [1819, 1031, 2352, 2291],
                &d6,
                &a6,
            ),
            (
                Dlatms {
                    dist: Dist::Symmetric,
                    sym: DlatmsSym::Symmetric,
                    mode: 5,
                    cond: 10000.0,
                    dmax: 1e-10,
                    pack: DlatmsPack::Upper,
                },
                5,
                5,
                [2733, 318, 317, 1375],
                &d7,
                &a7,
            ),
            (
                Dlatms {
                    dist: Dist::Normal,
                    sym: DlatmsSym::PositiveSymmetric,
                    mode: 2,
                    cond: 100.0,
                    dmax: 10000000000.0,
                    pack: DlatmsPack::Lower,
                },
                5,
                5,
                [875, 1624, 2364, 2295],
                &d8,
                &a8,
            ),
            (
                Dlatms {
                    dist: Dist::Uniform,
                    sym: DlatmsSym::Symmetric,
                    mode: -3,
                    cond: 100000000000000.0,
                    dmax: 1.0,
                    pack: DlatmsPack::Full,
                },
                4,
                4,
                [3186, 2038, 187, 2079],
                &d9,
                &a9,
            ),
            (
                Dlatms {
                    dist: Dist::Uniform,
                    sym: DlatmsSym::NonSymmetric,
                    mode: 3,
                    cond: 10.0,
                    dmax: 1.0,
                    pack: DlatmsPack::Full,
                },
                1,
                1,
                [1988, 1989, 1990, 1991],
                &d10,
                &a10,
            ),
            (
                Dlatms {
                    dist: Dist::Uniform,
                    sym: DlatmsSym::NonSymmetric,
                    mode: 1,
                    cond: 100.0,
                    dmax: 1.0,
                    pack: DlatmsPack::Full,
                },
                3,
                1,
                [1547, 1078, 3168, 2783],
                &d11,
                &a11,
            ),
            (
                Dlatms {
                    dist: Dist::Uniform,
                    sym: DlatmsSym::Symmetric,
                    mode: 1,
                    cond: 10.0,
                    dmax: 1.0,
                    pack: DlatmsPack::Full,
                },
                2,
                2,
                [1547, 1078, 3168, 2783],
                &d12,
                &a12,
            ),
        ];
        let mut worst = 0.0f64;
        for (k, (params, m, n, want_seed, want_d, want_a)) in cases.iter().enumerate() {
            let mut seed = SEEDY;
            let mut a = vec![0.0f64; m * n];
            let mut d = want_d.to_vec();
            dlatms(params, *m, *n, &mut a, &mut d, &mut seed).expect("dlatms");
            assert_eq!(seed, *want_seed, "case {}: seed after the call", k + 1);
            for (i, (got, want)) in d.iter().zip(want_d.iter()).enumerate() {
                if *want == 0.0 {
                    assert_eq!(*got, 0.0, "case {}, d[{}]", k + 1, i);
                } else {
                    let rel = (got - want).abs() / want.abs();
                    worst = worst.max(rel);
                    assert!(
                        rel <= 1e-13,
                        "case {}, d[{}] = {:e}, reference {:e}",
                        k + 1,
                        i,
                        got,
                        want
                    );
                }
            }
            for (i, (got, want)) in a.iter().zip(want_a.iter()).enumerate() {
                if *want == 0.0 {
                    assert_eq!(
                        *got,
                        0.0,
                        "case {}, a[{}] = {:e} where the reference has zero",
                        k + 1,
                        i,
                        got
                    );
                } else {
                    let rel = (got - want).abs() / want.abs();
                    worst = worst.max(rel);
                    assert!(
                        rel <= 1e-13,
                        "case {}, a[{}] = {:e}, reference {:e}",
                        k + 1,
                        i,
                        got,
                        want
                    );
                }
            }
        }
        println!("worst relative difference over the DLATMS cases: {worst:e}");
    }

    /// The `INFO` values `DLATMS` returns for the arguments this port can still get
    /// wrong: a symmetric request with `M != N` (`-1`), a mode out of range (`-7`), a
    /// condition number below one (`-8`), and a triangular `PACK` with a
    /// nonsymmetric `SYM` (`-12`).
    #[test]
    fn dlatms_reports_what_the_reference_reports() {
        let mut seed = SEEDY;
        let mut a = vec![0.0f64; 9];
        let mut d = vec![1.0f64; 3];
        let mut p = Dlatms {
            dist: Dist::Uniform,
            sym: DlatmsSym::Symmetric,
            mode: 3,
            cond: 1e3,
            dmax: 1.0,
            pack: DlatmsPack::Full,
        };
        assert_eq!(
            dlatms(&p, 3, 2, &mut a[..6], &mut d[..2], &mut seed),
            Err(-1),
            "symmetric with m != n"
        );
        p.mode = 7;
        assert_eq!(dlatms(&p, 3, 3, &mut a, &mut d, &mut seed), Err(-7), "|mode| > 6");
        p.mode = 3;
        p.cond = 0.5;
        assert_eq!(dlatms(&p, 3, 3, &mut a, &mut d, &mut seed), Err(-8), "cond < 1");
        p.cond = 1e3;
        p.sym = DlatmsSym::NonSymmetric;
        p.pack = DlatmsPack::Upper;
        assert_eq!(
            dlatms(&p, 3, 3, &mut a, &mut d, &mut seed),
            Err(-12),
            "packed nonsymmetric"
        );

        // The quick return for an empty matrix, and the seed normalization the
        // reference applies before every call: four 12-bit words, the last odd.
        p.pack = DlatmsPack::Full;
        assert_eq!(dlatms(&p, 0, 0, &mut [], &mut [], &mut seed), Ok(()), "empty matrix");
        let mut seed = [1u32, 2, 3, 4];
        dlatms(&p, 3, 3, &mut a, &mut d, &mut seed).expect("dlatms");
        assert_eq!(seed[3] % 2, 1, "the last seed word is made odd");
    }

    /// One column-major matrix with the triangle the generator zeroed restored, so
    /// the residual can be formed against the full symmetric matrix.
    fn full_symmetric(a: &[f64], n: usize, upper: bool) -> Vec<f64> {
        let mut f = a.to_vec();
        for j in 0..n {
            for i in 0..n {
                let stored = if upper { i <= j } else { i >= j };
                if !stored {
                    f[i + j * n] = a[j + i * n];
                }
            }
        }
        f
    }

    /// `||I - Q^T Q||_1 / (n eps)`: the orthonormality of a column-major matrix of
    /// vectors, as `DSYT21` and `DORT01` report it.
    fn orthonormality(q: &[f64], n: usize) -> f64 {
        let eps = f64::EPSILON;
        let mut w = vec![0.0f64; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[k + i * n] * q[k + j * n];
                }
                w[i + j * n] = if i == j { 1.0 - sum } else { -sum };
            }
        }
        crate::native::blas::dlange('1', &w, n, n) / (n as f64 * eps)
    }

    /// The complex 1-norm of an `m x n` matrix: the largest column sum of
    /// `|re| + |im|`, which is what `ZLANGE('1')` and the checkers use.
    fn clange1(a: &[Complex64], m: usize, n: usize) -> f64 {
        let mut best = 0.0f64;
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..m {
                sum += a[i + j * m].re.abs() + a[i + j * m].im.abs();
            }
            best = best.max(sum);
        }
        best
    }

    /// `||I - Q^H Q||_1 / (n eps)` for a complex `n x n` matrix of vectors.
    fn orthonormality_h(q: &[Complex64], n: usize) -> f64 {
        let eps = f64::EPSILON;
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let mut w = vec![0.0f64; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut sum = zero;
                for k in 0..n {
                    sum += q[k + i * n].conj() * q[k + j * n];
                }
                let d = if i == j { one } else { zero };
                let e = d - sum;
                w[i + j * n] = e.re.abs() + e.im.abs();
            }
        }
        let mut best = 0.0f64;
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..n {
                sum += w[i + j * n];
            }
            best = best.max(sum);
        }
        best / (n as f64 * eps)
    }

    /// The SEP sweep: `dsyev` over the reference's symmetric family.
    ///
    /// `SYM = 'S'` (random signs on the spectrum, so negative eigenvalues are
    /// exercised) and `PACK = 'U'`/`'L'`, which is what makes the `UPLO` argument
    /// matter: the half of the matrix the driver is told to ignore holds zeros, so
    /// a driver that reads the wrong triangle solves a different problem. Modes
    /// `0..6` at the three magnitudes the reference's `KMAGN` table uses, `n = 7`,
    /// `COND = 1e3` where the mode takes one.
    ///
    /// Three assertions, each normalized by `n ||A||_1 eps` so that one means "as
    /// accurate as the arithmetic allows" and the reference's own bound applies:
    ///
    /// 1. the eigenvalues against `D`, the spectrum the matrix was built from;
    /// 2. the residual `||A U - U W||_1 / (||A||_1 ||U||_1 ulp)`, which is
    ///    [`get22`]'s first result with `W` diagonal;
    /// 3. `||I - U^T U||_1 / (n ulp)`, the orthonormality — `DSYT21`'s second
    ///    result, and the quantity that plays the role `get22`'s second result
    ///    plays for a dense eigenproblem.
    ///
    /// `get22`'s second result does *not* transfer, and the measurement is the
    /// reason rather than a preference: it compares the largest entry of each
    /// column with one, which is `dgeev`'s convention of scaling an eigenvector so
    /// its largest component is one. `dsyev` returns unit-2-norm vectors, whose
    /// largest entry is below one by construction, so the metric reports
    /// `|0.5 - 1| / (n ulp) ~ 3e14` on *every* case here — the reference's own
    /// `dsyev` included. It is constant, so it carries no information about the
    /// answer; the orthonormality is what is left to check once the vectors are
    /// unit length. The same test on this grid, run on LAPACK 3.12.1's own
    /// `dsyev`, reports at most 3.1 for the residual, 2.3 for the orthonormality
    /// and 0.37 for the eigenvalues.
    #[test]
    fn dsyev_solves_the_reference_symmetric_families() {
        let n = 7usize;
        let eps = f64::EPSILON;
        let mut worst_ev = 0.0f64;
        let mut worst_res = 0.0f64;
        let mut worst_orth = 0.0f64;
        for mode in 0..=6i32 {
            for dmax in [1.0f64, 1e10, 1e-10] {
                for upper in [true, false] {
                    let p = Dlatms {
                        dist: Dist::Symmetric,
                        sym: DlatmsSym::Symmetric,
                        mode,
                        cond: 1e3,
                        dmax,
                        pack: if upper { DlatmsPack::Upper } else { DlatmsPack::Lower },
                    };
                    // MODE = 0 keeps the caller's spectrum, which is where the
                    // signed eigenvalues come from when the mode does not sign them.
                    let mut d: Vec<f64> = if mode == 0 {
                        (0..n)
                            .map(|i| {
                                let sign = if i % 2 == 0 { -1.0 } else { 1.0 };
                                sign * dmax / 2f64.powi(i as i32)
                            })
                            .collect()
                    } else {
                        vec![0.0; n]
                    };
                    let mut seed = SEEDY;
                    let mut a = vec![0.0f64; n * n];
                    dlatms(&p, n, n, &mut a, &mut d, &mut seed).expect("dlatms");
                    let full = full_symmetric(&a, n, upper);
                    let anorm = crate::native::blas::dlange('1', &full, n, n);

                    let mut u = a;
                    let mut w = vec![0.0f64; n];
                    crate::native::lapack::sym::dsyev(&mut u, &mut w, n, upper).expect("dsyev");

                    // 1) The spectrum, sorted on both sides: dsyev ascends, latm1
                    //    descends, and neither order is part of the contract here.
                    let mut ws = w.clone();
                    let mut ds = d.clone();
                    ws.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    ds.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    let mut ev = 0.0f64;
                    for (x, y) in ws.iter().zip(ds.iter()) {
                        ev = ev.max((x - y).abs());
                    }
                    let ev = ev / (n as f64 * anorm * eps);

                    // 2) The residual, in get22's form, with W diagonal.
                    let zeros = vec![0.0f64; n];
                    let (res, _) = get22(&full, &u, &w, &zeros, n);

                    // 3) The orthonormality.
                    let orth = orthonormality(&u, n);

                    worst_ev = worst_ev.max(ev);
                    worst_res = worst_res.max(res);
                    worst_orth = worst_orth.max(orth);
                    assert!(
                        ev <= 20.0,
                        "mode {mode}, dmax {dmax:e}, upper {upper}: eigenvalue ratio {ev:e}"
                    );
                    assert!(
                        res <= 20.0,
                        "mode {mode}, dmax {dmax:e}, upper {upper}: residual {res:e}"
                    );
                    assert!(
                        orth <= 20.0,
                        "mode {mode}, dmax {dmax:e}, upper {upper}: orthonormality {orth:e}"
                    );
                }
            }
        }
        println!(
            "dsyev over the family: eigenvalues {worst_ev:e}, residual {worst_res:e}, orthonormality {worst_orth:e}"
        );
    }

    /// The same sweep through `zheev`, which is fed the same real matrix as a
    /// Hermitian one with a zero imaginary part.
    ///
    /// That is a weaker test than a genuinely complex matrix would be, and the
    /// reason is that the reference's complex generator (`CLATMS`) is not part of
    /// this port, so there is no reference-built complex spectrum to compare
    /// against. What it does test is everything the complex path does differently:
    /// the reflectors, the tridiagonal reduction and the QR iteration all run in
    /// complex arithmetic and must leave the vectors real and the residual at the
    /// same order as `dsyev`'s. The quantities are the complex forms of the three
    /// above, and `ZGET22` is not ported, so the residual is formed here in the
    /// checker's own normalized shape.
    #[test]
    fn zheev_solves_the_reference_symmetric_families() {
        let n = 7usize;
        let eps = f64::EPSILON;
        let zero = Complex64::new(0.0, 0.0);
        let mut worst_ev = 0.0f64;
        let mut worst_res = 0.0f64;
        let mut worst_orth = 0.0f64;
        for mode in 0..=6i32 {
            for dmax in [1.0f64, 1e10, 1e-10] {
                for upper in [true, false] {
                    let p = Dlatms {
                        dist: Dist::Symmetric,
                        sym: DlatmsSym::Symmetric,
                        mode,
                        cond: 1e3,
                        dmax,
                        pack: if upper { DlatmsPack::Upper } else { DlatmsPack::Lower },
                    };
                    let mut d: Vec<f64> = if mode == 0 {
                        (0..n)
                            .map(|i| {
                                let sign = if i % 2 == 0 { -1.0 } else { 1.0 };
                                sign * dmax / 2f64.powi(i as i32)
                            })
                            .collect()
                    } else {
                        vec![0.0f64; n]
                    };
                    let mut seed = SEEDY;
                    let mut a = vec![0.0f64; n * n];
                    dlatms(&p, n, n, &mut a, &mut d, &mut seed).expect("dlatms");
                    let full: Vec<Complex64> = full_symmetric(&a, n, upper)
                        .iter()
                        .map(|&v| Complex64::new(v, 0.0))
                        .collect();
                    let anorm = clange1(&full, n, n);
                    let mut h: Vec<Complex64> = a.iter().map(|&v| Complex64::new(v, 0.0)).collect();
                    let mut w = vec![0.0f64; n];
                    crate::native::lapack::zeig::zheev(&mut h, &mut w, n, upper).expect("zheev");

                    let mut ws = w.clone();
                    let mut ds = d.clone();
                    ws.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    ds.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    let mut ev = 0.0f64;
                    for (x, y) in ws.iter().zip(ds.iter()) {
                        ev = ev.max((x - y).abs());
                    }
                    let ev = ev / (n as f64 * anorm * eps);

                    // E W, then A E - E W, with W diagonal and real.
                    let mut werr = vec![zero; n * n];
                    for j in 0..n {
                        for i in 0..n {
                            let mut sum = zero;
                            for k in 0..n {
                                sum += full[i + k * n] * h[k + j * n];
                            }
                            werr[i + j * n] = sum - h[i + j * n] * w[j];
                        }
                    }
                    let enorm = clange1(&h, n, n);
                    let res = (clange1(&werr, n, n) / enorm / anorm) / eps;
                    let orth = orthonormality_h(&h, n);

                    worst_ev = worst_ev.max(ev);
                    worst_res = worst_res.max(res);
                    worst_orth = worst_orth.max(orth);
                    assert!(
                        ev <= 20.0,
                        "mode {mode}, dmax {dmax:e}, upper {upper}: eigenvalue ratio {ev:e}"
                    );
                    assert!(
                        res <= 20.0,
                        "mode {mode}, dmax {dmax:e}, upper {upper}: residual {res:e}"
                    );
                    assert!(
                        orth <= 20.0,
                        "mode {mode}, dmax {dmax:e}, upper {upper}: orthonormality {orth:e}"
                    );
                }
            }
        }
        println!(
            "zheev over the family: eigenvalues {worst_ev:e}, residual {worst_res:e}, orthonormality {worst_orth:e}"
        );
    }

    /// The SVD sweep: `dgesvd` over the reference's bidiagonal family.
    ///
    /// `SYM = 'N'` with `MODE 0..6` at the three magnitudes, on four rectangular
    /// shapes — `6x4`, `7x3` and their transposes — because `m < n` is the case the
    /// driver has a separate path for (`m >= n` reduces A directly, `m < n` the
    /// same way on the transpose with `VT` and `U` swapped back).
    ///
    /// The three assertions are the reference's:
    ///
    /// 1. `||U^T A V - S||_1 / (max(m, n) ||A||_1 eps)` — `DBDT01`'s quantity with
    ///    `KD = 0`, which is the reconstruction `A = U S V^T` in the factors'
    ///    coordinates;
    /// 2. `||I - U^T U||_1 / (m eps)` and `||I - V^T V||_1 / (n eps)`, `DORT01`'s
    ///    number for each factor separately — reconstruction and orthogonality are
    ///    independent failures, as `dchkbd`'s two checkers assume;
    /// 3. the singular values against the spectrum the matrix was built from,
    ///    sorted on both sides since only the values are ordered, not `D`.
    ///
    /// Measured on LAPACK 3.12.1's own `dgesvd` over this grid: at most 4.5 for the
    /// reconstruction, 2.0 for either orthogonality and 0.43 for the values. The
    /// bound of 20 is the reference's own.
    #[test]
    fn dgesvd_solves_the_reference_bidiagonal_families() {
        let eps = f64::EPSILON;
        let mut worst_recon = 0.0f64;
        let mut worst_orth = 0.0f64;
        let mut worst_sv = 0.0f64;
        for (m, n) in [(6usize, 4usize), (4, 6), (7, 3), (3, 7)] {
            let mnmin = m.min(n);
            for mode in 0..=6i32 {
                for dmax in [1.0f64, 1e10, 1e-10] {
                    let p = Dlatms {
                        dist: Dist::Symmetric,
                        sym: DlatmsSym::NonSymmetric,
                        mode,
                        cond: 1e3,
                        dmax,
                        pack: DlatmsPack::Full,
                    };
                    let mut d: Vec<f64> = if mode == 0 {
                        (0..mnmin).map(|i| dmax / 2f64.powi(i as i32)).collect()
                    } else {
                        vec![0.0; mnmin]
                    };
                    let mut seed = SEEDY;
                    let mut a = vec![0.0f64; m * n];
                    dlatms(&p, m, n, &mut a, &mut d, &mut seed).expect("dlatms");
                    let a0 = a.clone();
                    let anorm = crate::native::blas::dlange('1', &a0, m, n);

                    let mut s = vec![0.0f64; mnmin];
                    let mut u = vec![0.0f64; m * m];
                    let mut vt = vec![0.0f64; n * n];
                    crate::native::lapack::svd::dgesvd(&mut a, m, n, &mut s, &mut u, &mut vt).expect("dgesvd");

                    // 1) U^T A V - S, formed in two passes to avoid transposing V.
                    let mut tmp = vec![0.0f64; m * n];
                    for j in 0..n {
                        for i in 0..m {
                            let mut sum = 0.0;
                            for k in 0..n {
                                sum += a0[i + k * m] * vt[j + k * n];
                            }
                            tmp[i + j * m] = sum;
                        }
                    }
                    let mut r = vec![0.0f64; m * n];
                    for j in 0..n {
                        for i in 0..m {
                            let mut sum = 0.0;
                            for k in 0..m {
                                sum += u[k + i * m] * tmp[k + j * m];
                            }
                            r[i + j * m] = sum;
                        }
                    }
                    for k in 0..mnmin {
                        r[k + k * m] -= s[k];
                    }
                    let recon = crate::native::blas::dlange('1', &r, m, n) / (m.max(n) as f64 * anorm * eps);

                    // 2) Both factors, orthonormality each.
                    let orth_u = orthonormality(&u, m);
                    let mut v = vec![0.0f64; n * n];
                    for j in 0..n {
                        for i in 0..n {
                            v[i + j * n] = vt[j + i * n];
                        }
                    }
                    let orth_v = orthonormality(&v, n);
                    let orth = orth_u.max(orth_v);

                    // 3) The singular values, both sides sorted descending.
                    let mut ss = s.clone();
                    let mut ds: Vec<f64> = d.iter().map(|v| v.abs()).collect();
                    ss.sort_by(|x, y| y.partial_cmp(x).unwrap());
                    ds.sort_by(|x, y| y.partial_cmp(x).unwrap());
                    let mut sv = 0.0f64;
                    for (x, y) in ss.iter().zip(ds.iter()) {
                        sv = sv.max((x - y).abs());
                    }
                    let sv = sv / (m.max(n) as f64 * anorm * eps);

                    worst_recon = worst_recon.max(recon);
                    worst_orth = worst_orth.max(orth);
                    worst_sv = worst_sv.max(sv);
                    assert!(
                        recon <= 20.0,
                        "{m}x{n}, mode {mode}, dmax {dmax:e}: reconstruction {recon:e}"
                    );
                    assert!(
                        orth <= 20.0,
                        "{m}x{n}, mode {mode}, dmax {dmax:e}: orthogonality {orth:e}"
                    );
                    assert!(
                        sv <= 20.0,
                        "{m}x{n}, mode {mode}, dmax {dmax:e}: singular values {sv:e}"
                    );
                }
            }
        }
        println!(
            "dgesvd over the family: reconstruction {worst_recon:e}, orthogonality {worst_orth:e}, singular values {worst_sv:e}"
        );
    }

    /// The same sweep through `zgesvd`, on the same real matrices held as complex
    /// numbers with a zero imaginary part.
    ///
    /// Same caveat and same reason as [`zheev_solves_the_reference_symmetric_families`]:
    /// the reference's complex generator is not ported, so the value here is that
    /// the complex bidiagonal reduction, the complex QR iteration and both sets of
    /// singular vectors carry the real problem unchanged. The reconstruction is
    /// formed with the complex inner products, since that is where a wrong
    /// conjugation shows up.
    #[test]
    fn zgesvd_solves_the_reference_bidiagonal_families() {
        let eps = f64::EPSILON;
        let zero = Complex64::new(0.0, 0.0);
        let mut worst_recon = 0.0f64;
        let mut worst_orth = 0.0f64;
        let mut worst_sv = 0.0f64;
        for (m, n) in [(6usize, 4usize), (4, 6), (7, 3), (3, 7)] {
            let mnmin = m.min(n);
            for mode in 0..=6i32 {
                for dmax in [1.0f64, 1e10, 1e-10] {
                    let p = Dlatms {
                        dist: Dist::Symmetric,
                        sym: DlatmsSym::NonSymmetric,
                        mode,
                        cond: 1e3,
                        dmax,
                        pack: DlatmsPack::Full,
                    };
                    let mut d: Vec<f64> = if mode == 0 {
                        (0..mnmin).map(|i| dmax / 2f64.powi(i as i32)).collect()
                    } else {
                        vec![0.0; mnmin]
                    };
                    let mut seed = SEEDY;
                    let mut ar = vec![0.0f64; m * n];
                    dlatms(&p, m, n, &mut ar, &mut d, &mut seed).expect("dlatms");
                    let a0: Vec<Complex64> = ar.iter().map(|&v| Complex64::new(v, 0.0)).collect();
                    let anorm = clange1(&a0, m, n);

                    let mut a = a0.clone();
                    let mut s = vec![0.0f64; mnmin];
                    let mut u = vec![zero; m * m];
                    let mut vt = vec![zero; n * n];
                    crate::native::lapack::zsvd::zgesvd(&mut a, m, n, &mut s, &mut u, &mut vt).expect("zgesvd");

                    // U^H A V - S, V recovered as VT^H on the fly.
                    let mut tmp = vec![zero; m * n];
                    for j in 0..n {
                        for i in 0..m {
                            let mut sum = zero;
                            for k in 0..n {
                                sum += a0[i + k * m] * vt[j + k * n].conj();
                            }
                            tmp[i + j * m] = sum;
                        }
                    }
                    let mut r = vec![zero; m * n];
                    for j in 0..n {
                        for i in 0..m {
                            let mut sum = zero;
                            for k in 0..m {
                                sum += u[k + i * m].conj() * tmp[k + j * m];
                            }
                            if i == j && i < mnmin {
                                sum -= Complex64::new(s[i], 0.0);
                            }
                            r[i + j * m] = sum;
                        }
                    }
                    let recon = (clange1(&r, m, n) / anorm) / eps / m.max(n) as f64;

                    let orth_u = orthonormality_h(&u, m);
                    let mut v = vec![zero; n * n];
                    for j in 0..n {
                        for i in 0..n {
                            v[i + j * n] = vt[j + i * n].conj();
                        }
                    }
                    let orth_v = orthonormality_h(&v, n);
                    let orth = orth_u.max(orth_v);

                    let mut ss = s.clone();
                    let mut ds: Vec<f64> = d.iter().map(|v| v.abs()).collect();
                    ss.sort_by(|x, y| y.partial_cmp(x).unwrap());
                    ds.sort_by(|x, y| y.partial_cmp(x).unwrap());
                    let mut sv = 0.0f64;
                    for (x, y) in ss.iter().zip(ds.iter()) {
                        sv = sv.max((x - y).abs());
                    }
                    let sv = sv / (m.max(n) as f64 * anorm * eps);

                    worst_recon = worst_recon.max(recon);
                    worst_orth = worst_orth.max(orth);
                    worst_sv = worst_sv.max(sv);
                    assert!(
                        recon <= 20.0,
                        "{m}x{n}, mode {mode}, dmax {dmax:e}: reconstruction {recon:e}"
                    );
                    assert!(
                        orth <= 20.0,
                        "{m}x{n}, mode {mode}, dmax {dmax:e}: orthogonality {orth:e}"
                    );
                    assert!(
                        sv <= 20.0,
                        "{m}x{n}, mode {mode}, dmax {dmax:e}: singular values {sv:e}"
                    );
                }
            }
        }
        println!(
            "zgesvd over the family: reconstruction {worst_recon:e}, orthogonality {worst_orth:e}, singular values {worst_sv:e}"
        );
    }

    /// The generalized sweep: `dggev` over the pairs `dchkgg` builds.
    ///
    /// `dchkgg`'s 26 types are `(ITYPE1, ITYPE2, ISIGN, RMOD)`; what is reproduced
    /// here is the grid of `ITYPE` values for `A` (1..10, in both sign conventions
    /// where the spectrum is a diagonal) against a fixed `B` that is the identity,
    /// with every `ISIGN` level including the rotations that make 2x2 blocks.
    ///
    /// The residual is the reference's own generalized form,
    /// `||A v - lambda B v|| / ((||A|| + |lambda| ||B||) n eps) <= 20`.
    #[test]
    fn dggev_solves_the_reference_generalized_families() {
        let n = 5usize;
        let eps = f64::EPSILON;
        let mut worst_ratio = 0.0f64;
        for itype in [1i32, -1, 2, 3, 4, -4, 5, -5, 6, -6, 7, -7, 8, 9, -9, 10] {
            for isign in [0i32, 1, 2] {
                if (itype == 2 || itype == 3) && isign == 2 {
                    // The reference skips the rotations for the Jordan shapes.
                    continue;
                }
                let pa = Latm4 {
                    itype,
                    nz1: 0,
                    nz2: 0,
                    rcond: 1e-3,
                    amagn: 1.0,
                    isign,
                    dist: Dist::Uniform,
                    triang: if isign == 2 { 0.25 } else { 0.0 },
                };
                let pb = Latm4 {
                    itype: 1,
                    nz1: 0,
                    nz2: 0,
                    rcond: 1.0,
                    amagn: 1.0,
                    isign: 0,
                    dist: Dist::Uniform,
                    triang: 0.0,
                };
                let mut seed = SEEDY;
                let mut a0 = vec![0.0f64; n * n];
                let mut b0 = vec![0.0f64; n * n];
                latm4(&pa, n, &mut a0, &mut seed);
                latm4(&pb, n, &mut b0, &mut seed);

                let mut a = a0.clone();
                let mut b = b0.clone();
                let mut alphar = vec![0.0; n];
                let mut alphai = vec![0.0; n];
                let mut beta = vec![0.0; n];
                let mut vl = vec![0.0; n * n];
                let mut vr = vec![0.0; n * n];
                dggev(
                    false,
                    true,
                    n,
                    &mut a,
                    &mut b,
                    &mut alphar,
                    &mut alphai,
                    &mut beta,
                    &mut vl,
                    &mut vr,
                )
                .unwrap_or_else(|e| panic!("itype {itype}, isign {isign}: {e:?}"));

                let anorm = a0.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                let bnorm = b0.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                for j in 0..n {
                    assert!(
                        beta[j].is_finite() && alphar[j].is_finite(),
                        "itype {itype}, isign {isign}: alpha/beta {j} is not finite"
                    );
                }
                let mut j = 0usize;
                while j < n {
                    let complex = alphai[j] != 0.0;
                    let (lr, li) = (alphar[j], alphai[j]);
                    let be = beta[j];
                    let mut worst = 0.0f64;
                    for i in 0..n {
                        // beta*A*v - alpha*B*v
                        let mut pa_r = 0.0;
                        let mut pa_i = 0.0;
                        let mut pb_r = 0.0;
                        let mut pb_i = 0.0;
                        for k in 0..n {
                            pa_r += a0[i + k * n] * vr[k + j * n];
                            pa_i += a0[i + k * n] * vr[k + (j + if complex { 1 } else { 0 }) * n];
                            pb_r += b0[i + k * n] * vr[k + j * n];
                            pb_i += b0[i + k * n] * vr[k + (j + if complex { 1 } else { 0 }) * n];
                        }
                        let dr = be * pa_r - (lr * pb_r - li * pb_i);
                        let di = be * pa_i - (lr * pb_i + li * pb_r);
                        worst = worst.max(dr.abs().max(di.abs()));
                    }
                    let scale = (n as f64) * eps * (anorm + (lr.hypot(li)) * bnorm);
                    let ratio = worst / scale;
                    worst_ratio = worst_ratio.max(ratio);
                    assert!(ratio <= 20.0, "itype {itype}, isign {isign}, j {j}: ratio = {ratio:e}");
                    j += if complex { 2 } else { 1 };
                }
            }
        }
        println!("worst generalized ratio: {worst_ratio:e}");
    }
    /// The generalized problem where `B` is singular, so some eigenvalues are
    /// infinite and `dggev` has to deflate rather than divide.
    ///
    /// The families above always pair `A` with a nonsingular `B`; that leaves the
    /// infinite-eigenvalue path alone. Here `B` carries trailing zeros on its
    /// diagonal (`NZ2`), which is a singular pencil, and the check is that the
    /// finite eigenpairs still satisfy `beta A v = alpha B v` while the rest come
    /// back as `beta = 0` with a finite `alpha`.
    #[test]
    fn dggev_handles_a_singular_b() {
        let n = 5usize;
        let eps = f64::EPSILON;
        let pa = Latm4 {
            itype: -4,
            nz1: 0,
            nz2: 0,
            rcond: 1e-3,
            amagn: 1.0,
            isign: 1,
            dist: Dist::Uniform,
            triang: 0.25,
        };
        // B must be the diagonal family (`itype = 4`): `itype = 1` sets the whole
        // diagonal and ignores NZ1/NZ2, so it stays nonsingular.
        let pb = Latm4 {
            itype: 4,
            nz1: 0,
            nz2: 2,
            rcond: 1.0,
            amagn: 1.0,
            isign: 0,
            dist: Dist::Uniform,
            triang: 0.0,
        };
        let mut seed = SEEDY;
        let mut a0 = vec![0.0f64; n * n];
        let mut b0 = vec![0.0f64; n * n];
        latm4(&pa, n, &mut a0, &mut seed);
        latm4(&pb, n, &mut b0, &mut seed);

        let mut a = a0.clone();
        let mut b = b0.clone();
        let mut alphar = vec![0.0; n];
        let mut alphai = vec![0.0; n];
        let mut beta = vec![0.0; n];
        let mut vl = vec![0.0; n * n];
        let mut vr = vec![0.0; n * n];
        dggev(
            false,
            true,
            n,
            &mut a,
            &mut b,
            &mut alphar,
            &mut alphai,
            &mut beta,
            &mut vl,
            &mut vr,
        )
        .expect("dggev");

        let anorm = a0.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let bnorm = b0.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let infinite = (0..n).filter(|&j| beta[j] == 0.0).count();
        assert_eq!(
            infinite, 2,
            "the two zero diagonal entries of B give two infinite eigenvalues"
        );
        for j in 0..n {
            assert!(
                alphar[j].is_finite() && alphai[j].is_finite() && beta[j].is_finite(),
                "eigenpair {j} is not finite"
            );
            if beta[j] == 0.0 {
                continue;
            }
            let mut worst = 0.0f64;
            for i in 0..n {
                let mut pa_r = 0.0;
                let mut pb_r = 0.0;
                for k in 0..n {
                    pa_r += a0[i + k * n] * vr[k + j * n];
                    pb_r += b0[i + k * n] * vr[k + j * n];
                }
                worst = worst.max((beta[j] * pa_r - alphar[j] * pb_r).abs());
            }
            let scale = n as f64 * eps * (anorm + alphar[j].abs() * bnorm);
            let ratio = worst / scale;
            assert!(ratio <= 20.0, "j {j}: ratio = {ratio:e}");
        }
    }
}
