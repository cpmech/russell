use super::{ONE_BY_3, PI};
use crate::StrError;

/// Computes the elliptic integral of the first kind F(φ, k)
///
/// ```text
///              φ
///             ⌠          dt
/// F(φ, k)  =  │  ___________________
///             │     _______________
///             ⌡   \╱ 1 - k² sin²(t)
///            0
///
/// 0 ≤ φ ≤ π/2   and   0 ≤ k·sin(φ) ≤ 1
/// ```
///
/// **Important:** Note that `k² = m`, where `m` is used in other tools
/// such as the Mathematica `EllipticF[ϕ, m]` function.
///
/// # Input
///
/// * `phi`  -- `0 ≤ φ ≤ π/2`
/// * `k` --  `0 ≤ k·sin(φ) ≤ 1`
///
/// # Special cases
///
/// * `F(φ, k) = Inf` if `k·sin(φ) == 1`
///	* `F(0.0, k) = 0.0`
///	* `F(φ, 0.0) = φ`
///
/// # References
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
pub fn elliptic_f(phi: f64, k: f64) -> Result<f64, StrError> {
    if phi < 0.0 || k < 0.0 {
        return Err("phi and k must be non-negative");
    }
    if phi > PI / 2.0 + f64::EPSILON {
        return Err("phi must be in 0 ≤ phi ≤ π/2");
    }
    if phi < f64::MIN_POSITIVE {
        return Ok(0.0);
    }
    if k < f64::MIN_POSITIVE {
        return Ok(phi);
    }
    let s = f64::sin(phi);
    if f64::abs(k * s - 1.0) < 10.0 * f64::EPSILON {
        return Ok(f64::INFINITY);
    }
    let ans = s * rf(f64::powf(f64::cos(phi), 2.0), (1.0 - s * k) * (1.0 + s * k), 1.0)?;
    Ok(ans)
}

/// Computes the elliptic integral of the second kind E(φ, k)
///
/// ```text
///              φ
///             ⌠     _______________
/// E(φ, k)  =  │   \╱ 1 - k² sin²(t)  dt
///             ⌡
///            0
///
/// 0 ≤ φ ≤ π/2   and   0 ≤ k·sin(φ) ≤ 1
/// ```
///
/// **Important:** Note that `k² = m`, where `m` is used in other tools
/// such as the Mathematica `EllipticE[ϕ, m]` function.
///
/// # Input
///
/// * `phi`  -- `0 ≤ φ ≤ π/2`
/// * `k` --  `0 ≤ k·sin(φ) ≤ 1`
///
/// # Special cases
///
///	* `E(0.0, k) = 0.0`
///	* `E(φ, 0.0) = φ`
///
/// # References
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
pub fn elliptic_e(phi: f64, k: f64) -> Result<f64, StrError> {
    if phi < 0.0 || k < 0.0 {
        return Err("phi and k must be non-negative");
    }
    if phi > PI / 2.0 + f64::EPSILON {
        return Err("phi must be in 0 ≤ phi ≤ π/2");
    }
    if phi < f64::MIN_POSITIVE {
        return Ok(0.0);
    }
    if k < f64::MIN_POSITIVE {
        return Ok(phi);
    }
    let s = f64::sin(phi);
    let cc = f64::powf(f64::cos(phi), 2.0);
    let q = (1.0 - s * k) * (1.0 + s * k);
    let ans = s * (rf(cc, q, 1.0)? - (f64::powf(s * k, 2.0)) * rd(cc, q, 1.0)? / 3.0);
    Ok(ans)
}

/// Computes the elliptic integral of the third kind Π(n, φ, k)
///
/// ```text
///                 φ
///                ⌠                  dt
/// Π(n, φ, k)  =  │  ___________________________________
///                │                     _______________
///                ⌡   (1 - n sin²(t)) \╱ 1 - k² sin²(t)
///               0
///
/// 0 ≤ φ ≤ π/2   and   0 ≤ k·sin(φ) ≤ 1
/// ```
///
/// **Important:** Note that `k² = m`, where `m` is used in other tools
/// such as the Mathematica `EllipticPi[ϕ, m]` function.
///
/// # Input
///
/// * `phi` -- `0 ≤ φ ≤ π/2`
/// * `k` -- `0 ≤ k·sin(φ) ≤ 1`
///
/// **Note:** The sign convention for `n` corresponds to that of Abramowitz and Stegun.
///
/// # Special cases
///
/// * `Π(n, φ, k) = Inf` if `k·sin(φ) == 1`
/// * `Π(n, φ, k) = Inf` if `n·sin²(φ) == 1`
///	* `Π(n, 0.0, k) = 0.0`
///
/// # References:
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
/// * Abramowitz M, Stegun IA (1972) Handbook of Mathematical Functions with Formulas, Graphs,
///   and Mathematical Tables. U.S. Department of Commerce, NIST
pub fn elliptic_pi(n: f64, phi: f64, k: f64) -> Result<f64, StrError> {
    if phi < 0.0 || k < 0.0 {
        return Err("phi and k must be non-negative");
    }
    if phi > PI / 2.0 + f64::EPSILON {
        return Err("phi must be in 0 ≤ phi ≤ π/2");
    }
    if phi < f64::MIN_POSITIVE {
        return Ok(0.0);
    }
    let s = f64::sin(phi);
    if f64::abs(k * s - 1.0) < 10.0 * f64::EPSILON {
        return Ok(f64::INFINITY);
    }
    if f64::abs(n * s * s - 1.0) < 10.0 * f64::EPSILON {
        return Ok(f64::INFINITY);
    }
    let minus_n = -n;
    let t = minus_n * s * s;
    let cc = f64::powf(f64::cos(phi), 2.0);
    let q = (1.0 - s * k) * (1.0 + s * k);
    let ans = s * (rf(cc, q, 1.0)? - t * rj(cc, q, 1.0, 1.0 + t)? / 3.0);
    Ok(ans)
}

/// Computes elliptic integral of the first kind using Carlson's formula
///
/// Computes Rf(x,y,z) where x,y,z must be non-negative and at most one can be zero.
///
/// # References:
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
/// * Carlson BC (1977) Elliptic Integrals of the First Kind, SIAM Journal on Mathematical
///   Analysis, vol. 8, pp. 231-242.
fn rf(x: f64, y: f64, z: f64) -> Result<f64, StrError> {
    let tiny = 5.0 * f64::MIN_POSITIVE;
    let big = 0.2 * f64::MAX;
    if f64::min(f64::min(x, y), z) < 0.0
        || f64::min(f64::min(x + y, x + z), y + z) < tiny
        || f64::max(f64::max(x, y), z) > big
    {
        return Err("(x,y,z) must be non-negative and at most one can be zero");
    }
    let mut xt = x;
    let mut yt = y;
    let mut zt = z;
    let mut ave: f64 = 0.0;
    let mut dx: f64 = 0.0;
    let mut dy: f64 = 0.0;
    let mut dz: f64 = 0.0;
    let mut it = 0;
    for _ in 0..N_MAX_ITERATIONS {
        let sqx = f64::sqrt(xt);
        let sqy = f64::sqrt(yt);
        let sqz = f64::sqrt(zt);
        let lam = sqx * (sqy + sqz) + sqy * sqz;
        xt = 0.25 * (xt + lam);
        yt = 0.25 * (yt + lam);
        zt = 0.25 * (zt + lam);
        ave = ONE_BY_3 * (xt + yt + zt);
        dx = (ave - xt) / ave;
        dy = (ave - yt) / ave;
        dz = (ave - zt) / ave;
        if f64::max(f64::max(f64::abs(dx), f64::abs(dy)), f64::abs(dz)) < RF_ERR_TOL {
            break;
        }
        it += 1;
    }
    if it == N_MAX_ITERATIONS {
        return Err("rf failed to converge");
    }
    let e2 = dx * dy - dz * dz;
    let e3 = dx * dy * dz;
    let ans = (1.0 + (RF_C1 * e2 - RF_C2 - RF_C3 * e3) * e2 + RF_C4 * e3) / f64::sqrt(ave);
    Ok(ans)
}

/// Computes elliptic integral of the second kind using Carlson's formula
///
/// Computes Rd(x,y,z) where x,y must be non-negative and at most one can be zero. z must be positive.
///
/// # References:
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
/// * Carlson BC (1977) Elliptic Integrals of the First Kind, SIAM Journal on Mathematical
///   Analysis, vol. 8, pp. 231-242.
fn rd(x: f64, y: f64, z: f64) -> Result<f64, StrError> {
    let tiny = 2.0 * f64::powf(f64::MAX, -2.0 / 3.0);
    let big = 0.1 * RD_ERR_TOL * f64::powf(f64::MIN_POSITIVE, -2.0 / 3.0);
    if f64::min(x, y) < 0.0 || f64::min(x + y, z) < tiny || f64::max(f64::max(x, y), z) > big {
        return Err("(x,y) must be non-negative and at most one can be zero. z must be positive");
    }
    let mut xt = x;
    let mut yt = y;
    let mut zt = z;
    let mut ave: f64 = 0.0;
    let mut dx: f64 = 0.0;
    let mut dy: f64 = 0.0;
    let mut dz: f64 = 0.0;
    let mut sum = 0.0;
    let mut fac = 1.0;
    let mut it = 0;
    for _ in 0..N_MAX_ITERATIONS {
        let sqx = f64::sqrt(xt);
        let sqy = f64::sqrt(yt);
        let sqz = f64::sqrt(zt);
        let lam = sqx * (sqy + sqz) + sqy * sqz;
        sum += fac / (sqz * (zt + lam));
        fac = 0.25 * fac;
        xt = 0.25 * (xt + lam);
        yt = 0.25 * (yt + lam);
        zt = 0.25 * (zt + lam);
        ave = 0.2 * (xt + yt + 3.0 * zt);
        dx = (ave - xt) / ave;
        dy = (ave - yt) / ave;
        dz = (ave - zt) / ave;
        if f64::max(f64::max(f64::abs(dx), f64::abs(dy)), f64::abs(dz)) < RD_ERR_TOL {
            break;
        }
        it += 1;
    }
    if it == N_MAX_ITERATIONS {
        return Err("rd failed to converge");
    }
    let ea = dx * dy;
    let eb = dz * dz;
    let ec = ea - eb;
    let ed = ea - 6.0 * eb;
    let ee = ed + ec + ec;
    let ans = 3.0 * sum
        + fac
            * (1.0
                + ed * (-RD_C1 + RD_C5 * ed - RD_C6 * dz * ee)
                + dz * (RD_C2 * ee + dz * (-RD_C3 * ec + dz * RD_C4 * ea)))
            / (ave * f64::sqrt(ave));
    Ok(ans)
}

/// Computes elliptic integral of the third kind using Carlson's formula
///
/// Computes Rj(x,y,z,p) where x,y,z must be nonnegative, and at most one can be zero.
/// p must be nonzero. If p < 0, the Cauchy principal value is returned.
///
/// # References:
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
/// * Carlson BC (1977) Elliptic Integrals of the First Kind, SIAM Journal on Mathematical
///   Analysis, vol. 8, pp. 231-242.
fn rj(x: f64, y: f64, z: f64, p: f64) -> Result<f64, StrError> {
    let tiny = f64::powf(5.0 * f64::MIN_POSITIVE, 1.0 / 3.0);
    let big = 0.3 * f64::powf(0.2 * f64::MAX, 1.0 / 3.0);
    if f64::min(f64::min(x, y), z) < 0.0
        || f64::min(f64::min(x + y, x + z), f64::min(y + z, f64::abs(p))) < tiny
        || f64::max(f64::max(x, y), f64::max(z, f64::abs(p))) > big
    {
        return Err("(x,y,z) must be non-negative and at most one can be zero. p must be nonzero");
    }
    let mut xt: f64;
    let mut yt: f64;
    let mut zt: f64;
    let mut ave: f64 = 0.0;
    let mut dx: f64 = 0.0;
    let mut dy: f64 = 0.0;
    let mut dz: f64 = 0.0;
    let mut dp: f64 = 0.0;
    let mut sum = 0.0;
    let mut fac = 1.0;
    let mut pt: f64;
    let mut a: f64 = 0.0;
    let mut b: f64 = 0.0;
    let mut rcx: f64 = 0.0;
    if p > 0.0 {
        xt = x;
        yt = y;
        zt = z;
        pt = p;
    } else {
        xt = f64::min(f64::min(x, y), z);
        zt = f64::max(f64::max(x, y), z);
        yt = x + y + z - xt - zt;
        a = 1.0 / (yt - p);
        b = a * (zt - yt) * (yt - xt);
        pt = yt + b;
        let rho = xt * zt / yt;
        let tau = p * pt / yt;
        rcx = rc(rho, tau)?;
    }
    let mut it = 0;
    for _ in 0..N_MAX_ITERATIONS {
        let sqx = f64::sqrt(xt);
        let sqy = f64::sqrt(yt);
        let sqz = f64::sqrt(zt);
        let lam = sqx * (sqy + sqz) + sqy * sqz;
        let alpha = f64::powf(pt * (sqx + sqy + sqz) + sqx * sqy * sqz, 2.0);
        let beta = pt * f64::powf(pt + lam, 2.0);
        sum += fac * rc(alpha, beta)?;
        fac = 0.25 * fac;
        xt = 0.25 * (xt + lam);
        yt = 0.25 * (yt + lam);
        zt = 0.25 * (zt + lam);
        pt = 0.25 * (pt + lam);
        ave = 0.2 * (xt + yt + zt + pt + pt);
        dx = (ave - xt) / ave;
        dy = (ave - yt) / ave;
        dz = (ave - zt) / ave;
        dp = (ave - pt) / ave;
        if f64::max(
            f64::max(f64::abs(dx), f64::abs(dy)),
            f64::max(f64::abs(dz), f64::abs(dp)),
        ) < RJ_ERR_TOL
        {
            break;
        }
        it += 1;
    }
    if it == N_MAX_ITERATIONS {
        return Err("rj failed to converge");
    }
    let ea = dx * (dy + dz) + dy * dz;
    let eb = dx * dy * dz;
    let ec = dp * dp;
    let ed = ea - 3.0 * ec;
    let ee = eb + 2.0 * dp * (ea - ec);
    let mut ans = 3.0 * sum
        + fac
            * (1.0
                + ed * (-RJ_C1 + RJ_C5 * ed - RJ_C6 * ee)
                + eb * (RJ_C7 + dp * (-RJ_C8 + dp * RJ_C4))
                + dp * ea * (RJ_C2 - dp * RJ_C3)
                - RJ_C2 * dp * ec)
            / (ave * f64::sqrt(ave));
    if p <= 0.0 {
        ans = a * (b * ans + 3.0 * (rcx - rf(xt, yt, zt)?));
    }
    Ok(ans)
}

/// Computes the degenerate elliptic integral using Carlson's formula
///
/// Computes Rc(x,y) where x must be non-negative and y must be nonzero.
/// If y < 0, the Cauchy principal value is returned.
///
/// # References:
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
pub fn rc(x: f64, y: f64) -> Result<f64, StrError> {
    let tiny = 5.0 * f64::MIN_POSITIVE;
    let big = 0.2 * f64::MAX;
    let comp1 = 2.236 / f64::sqrt(tiny);
    let comp2 = f64::powf(tiny * big, 2.0) / 25.0;
    if x < 0.0
        || y == 0.0
        || (x + f64::abs(y)) < tiny
        || (x + f64::abs(y)) > big
        || (y < -comp1 && x > 0.0 && x < comp2)
    {
        return Err("x must be non-negative. y must not be zero");
    }
    let (mut xt, mut yt, w) = if y > 0.0 {
        (x, y, 1.0)
    } else {
        (x - y, -y, f64::sqrt(x) / f64::sqrt(x - y))
    };
    let mut ave: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut it = 0;
    for _ in 0..N_MAX_ITERATIONS {
        let lam = 2.0 * f64::sqrt(xt) * f64::sqrt(yt) + yt;
        xt = 0.25 * (xt + lam);
        yt = 0.25 * (yt + lam);
        ave = ONE_BY_3 * (xt + yt + yt);
        s = (yt - ave) / ave;
        if f64::abs(s) < RC_ERR_TOL {
            break;
        }
        it += 1;
    }
    if it == N_MAX_ITERATIONS {
        return Err("rc failed to converge");
    }
    let ans = w * (1.0 + s * s * (RC_C1 + s * (RC_C2 + s * (RC_C3 + s * RC_C4)))) / f64::sqrt(ave);
    Ok(ans)
}

// constants --------------------------------------

const N_MAX_ITERATIONS: usize = 11;

const RF_ERR_TOL: f64 = 0.0025; // a value of 0.0025 for the error tolerance parameter gives full double precision (16 sig ant digits)
const RF_C1: f64 = 1.0 / 24.0;
const RF_C2: f64 = 0.1;
const RF_C3: f64 = 3.0 / 44.0;
const RF_C4: f64 = 1.0 / 14.0;

const RD_ERR_TOL: f64 = 0.0015;
const RD_C1: f64 = 3.0 / 14.0;
const RD_C2: f64 = 1.0 / 6.0;
const RD_C3: f64 = 9.0 / 22.0;
const RD_C4: f64 = 3.0 / 26.0;
const RD_C5: f64 = 0.25 * RD_C3;
const RD_C6: f64 = 1.5 * RD_C4;

const RJ_ERR_TOL: f64 = 0.0015;
const RJ_C1: f64 = 3.0 / 14.0;
const RJ_C2: f64 = 1.0 / 3.0;
const RJ_C3: f64 = 3.0 / 22.0;
const RJ_C4: f64 = 3.0 / 26.0;
const RJ_C5: f64 = 0.75 * RJ_C3;
const RJ_C6: f64 = 1.5 * RJ_C4;
const RJ_C7: f64 = 0.5 * RJ_C2;
const RJ_C8: f64 = RJ_C3 + RJ_C3;

const RC_ERR_TOL: f64 = 0.0012;
const RC_C1: f64 = 0.3;
const RC_C2: f64 = 1.0 / 7.0;
const RC_C3: f64 = 0.375;
const RC_C4: f64 = 9.0 / 22.0;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{elliptic_e, elliptic_f, elliptic_pi, rc, rd, rf, rj};
    use crate::approx_eq;
    use crate::math::{PI, SQRT_2};

    #[test]
    fn carlson_functions_capture_errors() {
        let cases = [
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ];
        for (x, y, z) in cases {
            assert_eq!(
                rf(x, y, z).err(),
                Some("(x,y,z) must be non-negative and at most one can be zero")
            );
            assert_eq!(
                rd(x, y, z).err(),
                Some("(x,y) must be non-negative and at most one can be zero. z must be positive")
            );
            assert_eq!(
                rj(x, y, z, 1.0).err(),
                Some("(x,y,z) must be non-negative and at most one can be zero. p must be nonzero")
            );
        }
        assert_eq!(
            rd(1.0, 1.0, 0.0).err(),
            Some("(x,y) must be non-negative and at most one can be zero. z must be positive")
        );
        assert_eq!(
            rj(1.0, 1.0, 1.0, 0.0).err(),
            Some("(x,y,z) must be non-negative and at most one can be zero. p must be nonzero")
        );
        let cases = [(-1.0, 1.0), (1.0, 0.0)];
        for (x, y) in cases {
            assert_eq!(rc(x, y).err(), Some("x must be non-negative. y must not be zero"));
        }
    }

    #[test]
    fn elliptic_f_captures_errors() {
        assert_eq!(elliptic_f(-1.0, 0.0).err(), Some("phi and k must be non-negative"));
        assert_eq!(elliptic_f(1.0, -1.0).err(), Some("phi and k must be non-negative"));
        assert_eq!(
            elliptic_f(PI / 2.0 + 1.0, 1.0).err(),
            Some("phi must be in 0 ≤ phi ≤ π/2")
        );
    }

    #[test]
    fn elliptic_f_edge_cases_work() {
        assert_eq!(elliptic_f(0.99 * f64::MIN_POSITIVE, 0.0).unwrap(), 0.0);
        assert_eq!(elliptic_f(PI / 4.0, 0.99 * f64::MIN_POSITIVE).unwrap(), PI / 4.0);
        let k_times_sin_phi = 1.0;
        let k = 2.0;
        assert_eq!(elliptic_f(f64::asin(k_times_sin_phi / k), k).unwrap(), f64::INFINITY);
        assert_eq!(elliptic_f(PI / 4.0, SQRT_2).unwrap(), f64::INFINITY);
        assert_eq!(elliptic_f(PI / 2.0, 1.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn elliptic_f_works() {
        // Mathematica:
        // list = Table[{phi, k, NumberForm[EllipticF[phi, k*k], 50]}, {phi, 0, Pi/2, Pi/8}, {k, 0, 1, 0.25}];
        // table = Flatten[list, {{1, 2}, {3}}];
        #[rustfmt::skip]
        let mathematica = [
            (0.0        , 0.0  , 1e-50, 0.0),
            (0.0        , 0.25 , 1e-50, 0.0),
            (0.0        , 0.5  , 1e-50, 0.0),
            (0.0        , 0.75 , 1e-50, 0.0),
            (0.0        , 1.0  , 1e-50, 0.0),
            (PI/8.0     , 0.0  , 1e-16, 0.3926990816987241),
            (PI/8.0     , 0.25 , 1e-16, 0.3933132893089199),
            (PI/8.0     , 0.5  , 1e-16, 0.395187276069818),
            (PI/8.0     , 0.75 , 1e-50, 0.3984206171209894),
            (PI/8.0     , 1.0  , 1e-50, 0.4031997191615115),
            (PI/4.0     , 0.0  , 1e-50, 0.7853981633974483),
            (PI/4.0     , 0.25 , 1e-15, 0.7899239996239404),
            (PI/4.0     , 0.5  , 1e-15, 0.804366101232066),
            (PI/4.0     , 0.75 , 1e-15, 0.831943296479276),
            (PI/4.0     , 1.0  , 1e-15, 0.881373587019543),
            (3.0*PI/8.0 , 0.0  , 1e-15, 1.178097245096172),
            (3.0*PI/8.0 , 0.25 , 1e-50, 1.191335209507002),
            (3.0*PI/8.0 , 0.5  , 1e-15, 1.235986172354044),
            (3.0*PI/8.0 , 0.75 , 1e-15, 1.33484364486983),
            (3.0*PI/8.0 , 1.0  , 1e-15, 1.614890916173095),
            (PI/2.0     , 0.0  , 1e-15, 1.570796326794897),
            (PI/2.0     , 0.25 , 1e-15, 1.596242222131783),
            (PI/2.0     , 0.5  , 1e-15, 1.685750354812596),
            (PI/2.0     , 0.75 , 1e-15, 1.910989780751829),
        ];
        for (phi, k, tol, reference) in mathematica {
            // println!("phi = {}π/8, k = {:?}", 8.0 * phi / PI, k);
            approx_eq(elliptic_f(phi, k).unwrap(), reference, tol);
        }
    }

    #[test]
    fn elliptic_e_captures_errors() {
        assert_eq!(elliptic_e(-1.0, 0.0).err(), Some("phi and k must be non-negative"));
        assert_eq!(elliptic_e(1.0, -1.0).err(), Some("phi and k must be non-negative"));
        assert_eq!(
            elliptic_e(PI / 2.0 + 1.0, 1.0).err(),
            Some("phi must be in 0 ≤ phi ≤ π/2")
        );
    }

    #[test]
    fn elliptic_e_edge_cases_work() {
        assert_eq!(elliptic_e(0.99 * f64::MIN_POSITIVE, 0.0).unwrap(), 0.0);
        assert_eq!(elliptic_e(PI / 4.0, 0.99 * f64::MIN_POSITIVE).unwrap(), PI / 4.0);
        approx_eq(
            elliptic_e(PI / 4.0, SQRT_2).unwrap(),
            0.59907011736779610371996124614016193911360633160783,
            1e-15,
        );
        assert_eq!(elliptic_e(PI / 2.0, 1.0).unwrap(), 1.0);
    }

    #[test]
    fn elliptic_e_works() {
        // Mathematica:
        // list = Table[{phi, k, NumberForm[EllipticE[phi, k*k], 50]}, {phi, 0, Pi/2, Pi/8}, {k, 0, 1, 0.25}];
        // table = Flatten[list, {{1, 2}, {3}}];
        #[rustfmt::skip]
        let mathematica = [
            (0.0        , 0.0  , 1e-50, 0.0),
            (0.0        , 0.25 , 1e-50, 0.0),
            (0.0        , 0.5  , 1e-50, 0.0),
            (0.0        , 0.75 , 1e-50, 0.0),
            (0.0        , 1.0  , 1e-50, 0.0),
            (PI/8.0     , 0.0  , 1e-16, 0.3926990816987241), 
            (PI/8.0     , 0.25 , 1e-16, 0.3920865800857932), 
            (PI/8.0     , 0.5  , 1e-16, 0.3902387362837457), 
            (PI/8.0     , 0.75 , 1e-50, 0.3871234648032271), 
            (PI/8.0     , 1.0  , 1e-50, 0.3826834323650898), 
            (PI/4.0     , 0.0  , 1e-50, 0.7853981633974483), 
            (PI/4.0     , 0.25 , 1e-50, 0.7809168245438629), 
            (PI/4.0     , 0.5  , 1e-15, 0.7671959857111226), 
            (PI/4.0     , 0.75 , 1e-15, 0.7432919634745029), 
            (PI/4.0     , 1.0  , 1e-50, 0.7071067811865475), 
            (3.0*PI/8.0 , 0.0  , 1e-15, 1.178097245096172), 
            (3.0*PI/8.0 , 0.25 , 1e-15, 1.165097224631938), 
            (3.0*PI/8.0 , 0.5  , 1e-15, 1.124570248666415), 
            (3.0*PI/8.0 , 0.75 , 1e-50, 1.050626001874075), 
            (3.0*PI/8.0 , 1.0  , 1e-15, 0.923879532511287), 
            (PI/2.0     , 0.0  , 1e-15, 1.570796326794897), 
            (PI/2.0     , 0.25 , 1e-50, 1.545957256105465), 
            (PI/2.0     , 0.5  , 1e-15, 1.467462209339427), 
            (PI/2.0     , 0.75 , 1e-15, 1.318472107994621), 
            (PI/2.0     , 1.0  , 1e-50, 1.0),
        ];
        for (phi, k, tol, reference) in mathematica {
            println!("phi = {}π/8, k = {:?}", 8.0 * phi / PI, k);
            approx_eq(elliptic_e(phi, k).unwrap(), reference, tol);
        }
    }

    #[test]
    fn elliptic_pi_captures_errors() {
        assert_eq!(
            elliptic_pi(1.0, -1.0, 0.0).err(),
            Some("phi and k must be non-negative")
        );
        assert_eq!(
            elliptic_pi(1.0, 1.0, -1.0).err(),
            Some("phi and k must be non-negative")
        );
        assert_eq!(
            elliptic_pi(2.0, PI / 2.0 + 1.0, 1.0).err(),
            Some("phi must be in 0 ≤ phi ≤ π/2")
        );
    }

    #[test]
    fn elliptic_pi_edge_cases_work() {
        assert_eq!(elliptic_pi(1.0, 0.99 * f64::MIN_POSITIVE, 0.0).unwrap(), 0.0);
        let k_times_sin_phi = 1.0;
        let k = 2.0;
        assert_eq!(
            elliptic_pi(1.0, f64::asin(k_times_sin_phi / k), k).unwrap(),
            f64::INFINITY
        );
        assert_eq!(elliptic_pi(1.0, PI / 4.0, SQRT_2).unwrap(), f64::INFINITY);
        assert_eq!(elliptic_pi(1.0, PI / 2.0, 1.0).unwrap(), f64::INFINITY);

        #[rustfmt::skip]
        let cases = [
            (-1.0, PI/2.0, 1.0),
            ( 0.0, PI/2.0, 1.0),
            ( 1.0, PI/2.0, 0.0),
            ( 1.0, PI/2.0, 0.25),
            ( 1.0, PI/2.0, 0.5),
            ( 1.0, PI/2.0, 0.75),
            ( 1.0, PI/2.0, 1.0),
            ( 2.0, PI/4.0, 0.0),
            ( 2.0, PI/4.0, 0.25),
            ( 2.0, PI/4.0, 0.5),
            ( 2.0, PI/4.0, 0.75),
            ( 2.0, PI/4.0, 1.0),
            ( 2.0, PI/2.0, 1.0),
        ];
        for (n, phi, k) in cases {
            // println!("n = {:>3}, phi = {}π/8, k = {:>4}", n, 8.0 * phi / PI, k,);
            assert!(elliptic_pi(n, phi, k).unwrap().is_infinite());
        }
    }

    #[test]
    fn elliptic_pi_works() {
        // Mathematica:
        // list = Table[{n, phi, k, NumberForm[EllipticPi[n, phi, k*k], 50]}, {n, -1, 2}, {phi, 0, Pi/2, Pi/8}, {k, 0, 1, 0.25}];
        // table = Flatten[list, {{1, 2, 3}, {4}}];
        #[rustfmt::skip]
        let mathematica = [
            (-1.0 , 0.0        , 0.0  , 1e-50, 0.0),
            (-1.0 , 0.0        , 0.25 , 1e-50, 0.0),
            (-1.0 , 0.0        , 0.5  , 1e-50, 0.0),
            (-1.0 , 0.0        , 0.75 , 1e-50, 0.0),
            (-1.0 , 0.0        , 1.0  , 1e-50, 0.0),
            (-1.0 , PI/8.0     , 0.0  , 1e-15, 0.3746978560011353),
            (-1.0 , PI/8.0     , 0.25 , 1e-16, 0.3752627090599716),
            (-1.0 , PI/8.0     , 0.5  , 1e-16, 0.3769856644986041),
            (-1.0 , PI/8.0     , 0.75 , 1e-16, 0.3799568324204776),
            (-1.0 , PI/8.0     , 1.0  , 1e-50, 0.3843447375648521),
            (-1.0 , PI/4.0     , 0.0  , 1e-15, 0.6755108588560399),
            (-1.0 , PI/4.0     , 0.25 , 1e-15, 0.6789938143103013),
            (-1.0 , PI/4.0     , 0.5  , 1e-15, 0.6900783531602186),
            (-1.0 , PI/4.0     , 0.75 , 1e-15, 0.7111192362634067),
            (-1.0 , PI/4.0     , 1.0  , 1e-50, 0.7484266478449651),
            (-1.0 , 3.0*PI/8.0 , 0.0  , 1e-15, 0.909248952481477),
            (-1.0 , 3.0*PI/8.0 , 0.25 , 1e-15, 0.91786771442656),
            (-1.0 , 3.0*PI/8.0 , 0.5  , 1e-15, 0.946735994205292),
            (-1.0 , 3.0*PI/8.0 , 0.75 , 1e-15, 1.009579293701023),
            (-1.0 , 3.0*PI/8.0 , 1.0  , 1e-14, 1.180371788473084),
            (-1.0 , PI/2.0     , 0.0  , 1e-15, 1.110720734539592),
            (-1.0 , PI/2.0     , 0.25 , 1e-15, 1.125595762079367),
            (-1.0 , PI/2.0     , 0.5  , 1e-50, 1.177446843000566),
            (-1.0 , PI/2.0     , 0.75 , 1e-15, 1.30499171065157),
            ( 0.0 , 0.0        , 0.0  , 1e-50, 0.0),
            ( 0.0 , 0.0        , 0.25 , 1e-50, 0.0),
            ( 0.0 , 0.0        , 0.5  , 1e-50, 0.0),
            ( 0.0 , 0.0        , 0.75 , 1e-50, 0.0),
            ( 0.0 , 0.0        , 1.0  , 1e-50, 0.0),
            ( 0.0 , PI/8.0     , 0.0  , 1e-15, 0.3926990816987241),
            ( 0.0 , PI/8.0     , 0.25 , 1e-16, 0.3933132893089199),
            ( 0.0 , PI/8.0     , 0.5  , 1e-16, 0.395187276069818),
            ( 0.0 , PI/8.0     , 0.75 , 1e-50, 0.3984206171209894),
            ( 0.0 , PI/8.0     , 1.0  , 1e-16, 0.4031997191615114),
            ( 0.0 , PI/4.0     , 0.0  , 1e-15, 0.7853981633974483),
            ( 0.0 , PI/4.0     , 0.25 , 1e-15, 0.7899239996239404),
            ( 0.0 , PI/4.0     , 0.5  , 1e-15, 0.804366101232066),
            ( 0.0 , PI/4.0     , 0.75 , 1e-15, 0.831943296479276),
            ( 0.0 , PI/4.0     , 1.0  , 1e-15, 0.881373587019543),
            ( 0.0 , 3.0*PI/8.0 , 0.0  , 1e-15, 1.178097245096172),
            ( 0.0 , 3.0*PI/8.0 , 0.25 , 1e-50, 1.191335209507002),
            ( 0.0 , 3.0*PI/8.0 , 0.5  , 1e-15, 1.235986172354044),
            ( 0.0 , 3.0*PI/8.0 , 0.75 , 1e-15, 1.33484364486983),
            ( 0.0 , 3.0*PI/8.0 , 1.0  , 1e-15, 1.614890916173095),
            ( 0.0 , PI/2.0     , 0.0  , 1e-15, 1.570796326794897),
            ( 0.0 , PI/2.0     , 0.25 , 1e-15, 1.596242222131783),
            ( 0.0 , PI/2.0     , 0.5  , 1e-15, 1.685750354812596),
            ( 0.0 , PI/2.0     , 0.75 , 1e-15, 1.910989780751829),
            ( 1.0 , 0.0        , 0.0  , 1e-50, 0.0),
            ( 1.0 , 0.0        , 0.25 , 1e-50, 0.0),
            ( 1.0 , 0.0        , 0.5  , 1e-50, 0.0),
            ( 1.0 , 0.0        , 0.75 , 1e-50, 0.0),
            ( 1.0 , 0.0        , 1.0  , 1e-50, 0.0),
            ( 1.0 , PI/8.0     , 0.0  , 1e-15, 0.414213562373095),
            ( 1.0 , PI/8.0     , 0.25 , 1e-16, 0.4148887499157372),
            ( 1.0 , PI/8.0     , 0.5  , 1e-50, 0.4169494122774874),
            ( 1.0 , PI/8.0     , 0.75 , 1e-15, 0.4205070481656367),
            ( 1.0 , PI/8.0     , 1.0  , 1e-16, 0.4257706241647383),
            ( 1.0 , PI/4.0     , 0.0  , 1e-50, 1.0),
            ( 1.0 , PI/4.0     , 0.25 , 1e-15, 1.006813769978728), 
            ( 1.0 , PI/4.0     , 0.5  , 1e-15, 1.028657249208549), 
            ( 1.0 , PI/4.0     , 0.75 , 1e-15, 1.070798803716203), 
            ( 1.0 , PI/4.0     , 1.0  , 1e-15, 1.147793574696319), 
            ( 1.0 , 3.0*PI/8.0 , 0.0  , 1e-15, 2.414213562373095), 
            ( 1.0 , 3.0*PI/8.0 , 0.25 , 1e-15, 2.454095930578143), 
            ( 1.0 , 3.0*PI/8.0 , 0.5  , 1e-15, 2.591483556460725), 
            ( 1.0 , 3.0*PI/8.0 , 0.75 , 1e-50, 2.912171502341037), 
            ( 1.0 , 3.0*PI/8.0 , 1.0  , 1e-15, 3.961767487985497), 
            ( 2.0 , 0.0        , 0.0  , 1e-50, 0.0),
            ( 2.0 , 0.0        , 0.25 , 1e-50, 0.0),
            ( 2.0 , 0.0        , 0.5  , 1e-50, 0.0),
            ( 2.0 , 0.0        , 0.75 , 1e-50, 0.0),
            ( 2.0 , 0.0        , 1.0  , 1e-50, 0.0),
            ( 2.0 , PI/8.0     , 0.0  , 1e-15, 0.4406867935097715), 
            ( 2.0 , PI/8.0     , 0.25 , 1e-16, 0.441439857788685), 
            ( 2.0 , PI/8.0     , 0.5  , 1e-16, 0.4437390242213649), 
            ( 2.0 , PI/8.0     , 0.75 , 1e-16, 0.4477113607770141), 
            ( 2.0 , PI/8.0     , 1.0  , 1e-16, 0.4535953429494243), 
            ( 2.0 , 3.0*PI/8.0 , 0.0  , 1e-15, 0.4406867935097716), 
            ( 2.0 , 3.0*PI/8.0 , 0.25 , 1e-15, 0.4285679279908611), 
            ( 2.0 , 3.0*PI/8.0 , 0.5  , 1e-16, 0.3836260537315499), 
            ( 2.0 , 3.0*PI/8.0 , 0.75 , 1e-15, 0.2631820774838872), 
            ( 2.0 , 3.0*PI/8.0 , 1.0  , 1e-15, -0.1878853428789266), 
            ( 2.0 , PI/2.0     , 0.0  , 1e-15, 0.0),
            ( 2.0 , PI/2.0     , 0.25 , 1e-15, -0.02574919276182852), 
            ( 2.0 , PI/2.0     , 0.5  , 1e-15, -0.1207208864079769), 
            ( 2.0 , PI/2.0     , 0.75 , 1e-50, -0.3816143599956563), 
        ];
        for (n, phi, k, tol, reference) in mathematica {
            // println!("n = {}, phi = {}π/8, k = {:?}", n, 8.0 * phi / PI, k);
            approx_eq(elliptic_pi(n, phi, k).unwrap(), reference, tol);
        }
    }
}
