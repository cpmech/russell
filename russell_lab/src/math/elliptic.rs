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
/// # Input
///
/// * `phi`  -- `0 ≤ φ ≤ π/2`
/// * `k` --  `0 ≤ k·sin(φ) ≤ 1`
///
/// # References
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
pub fn elliptic_1(phi: f64, k: f64) -> Result<f64, StrError> {
    if phi < 0.0 || k < 0.0 {
        return Err("phi and k must be non-negative");
    }
    if phi > PI / 2.0 {
        return Err("phi must be in 0 ≤ phi ≤ π/2");
    }
    if phi < f64::MIN_POSITIVE {
        return Ok(0.0);
    }
    if k < f64::MIN_POSITIVE {
        return Ok(phi);
    }
    let s = f64::sin(phi);
    if f64::abs(k * s - 1.0) < 1e-15 {
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
/// # Input
///
/// * `phi`  -- `0 ≤ φ ≤ π/2`
/// * `k` --  `0 ≤ k·sin(φ) ≤ 1`
///
/// # References
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
pub fn elliptic_2(phi: f64, k: f64) -> Result<f64, StrError> {
    if phi < 0.0 || k < 0.0 {
        return Err("phi and k must be non-negative");
    }
    if phi > PI / 2.0 {
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
/// # Input
///
/// * `phi`  -- `0 ≤ φ ≤ π/2`
/// * `k` --  `0 ≤ k·sin(φ) ≤ 1`
///
/// **Note:** The sign convention for `n` corresponds to that of Abramowitz and Stegun.
///
/// # References:
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
/// * Abramowitz M, Stegun IA (1972) Handbook of Mathematical Functions with Formulas, Graphs,
///   and Mathematical Tables. U.S. Department of Commerce, NIST
pub fn elliptic_3(n: f64, phi: f64, k: f64) -> Result<f64, StrError> {
    if phi < 0.0 || k < 0.0 {
        return Err("phi and k must be non-negative");
    }
    if phi > PI / 2.0 {
        return Err("phi must be in 0 ≤ phi ≤ π/2");
    }
    if phi < f64::MIN_POSITIVE {
        return Ok(0.0);
    }
    let minus_n = -n;
    let s = f64::sin(phi);
    if f64::abs(k * s - 1.0) < 1e-15 {
        return Ok(f64::INFINITY);
    }
    if f64::abs(n * s - 1.0) < 1e-15 {
        return Ok(f64::INFINITY);
    }
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
    let mut delx: f64 = 0.0;
    let mut dely: f64 = 0.0;
    let mut delz: f64 = 0.0;
    let mut it = 0;
    for _ in 0..N_MAX_ITERATIONS {
        let sqrtx = f64::sqrt(xt);
        let sqrty = f64::sqrt(yt);
        let sqrtz = f64::sqrt(zt);
        let alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        zt = 0.25 * (zt + alamb);
        ave = ONE_BY_3 * (xt + yt + zt);
        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;
        if f64::max(f64::max(f64::abs(delx), f64::abs(dely)), f64::abs(delz)) < RF_ERR_TOL {
            break;
        }
        it += 1;
    }
    if it == N_MAX_ITERATIONS {
        return Err("rf failed to converge");
    }
    let e2 = delx * dely - delz * delz;
    let e3 = delx * dely * delz;
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
    let mut delx: f64 = 0.0;
    let mut dely: f64 = 0.0;
    let mut delz: f64 = 0.0;
    let mut sum = 0.0;
    let mut fac = 1.0;
    let mut it = 0;
    for _ in 0..N_MAX_ITERATIONS {
        let sqrtx = f64::sqrt(xt);
        let sqrty = f64::sqrt(yt);
        let sqrtz = f64::sqrt(zt);
        let alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
        sum += fac / (sqrtz * (zt + alamb));
        fac = 0.25 * fac;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        zt = 0.25 * (zt + alamb);
        ave = 0.2 * (xt + yt + 3.0 * zt);
        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;
        if f64::max(f64::max(f64::abs(delx), f64::abs(dely)), f64::abs(delz)) < RD_ERR_TOL {
            break;
        }
        it += 1;
    }
    if it == N_MAX_ITERATIONS {
        return Err("rd failed to converge");
    }
    let ea = delx * dely;
    let eb = delz * delz;
    let ec = ea - eb;
    let ed = ea - 6.0 * eb;
    let ee = ed + ec + ec;
    let ans = 3.0 * sum
        + fac
            * (1.0
                + ed * (-RD_C1 + RD_C5 * ed - RD_C6 * delz * ee)
                + delz * (RD_C2 * ee + delz * (-RD_C3 * ec + delz * RD_C4 * ea)))
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
    let mut delx: f64 = 0.0;
    let mut dely: f64 = 0.0;
    let mut delz: f64 = 0.0;
    let mut delp: f64 = 0.0;
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
        let sqrtx = f64::sqrt(xt);
        let sqrty = f64::sqrt(yt);
        let sqrtz = f64::sqrt(zt);
        let alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
        let alpha = f64::powf(pt * (sqrtx + sqrty + sqrtz) + sqrtx * sqrty * sqrtz, 2.0);
        let beta = pt * f64::powf(pt + alamb, 2.0);
        sum += fac * rc(alpha, beta)?;
        fac = 0.25 * fac;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        zt = 0.25 * (zt + alamb);
        pt = 0.25 * (pt + alamb);
        ave = 0.2 * (xt + yt + zt + pt + pt);
        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;
        delp = (ave - pt) / ave;
        if f64::max(
            f64::max(f64::abs(delx), f64::abs(dely)),
            f64::max(f64::abs(delz), f64::abs(delp)),
        ) < RJ_ERR_TOL
        {
            break;
        }
        it += 1;
    }
    if it == N_MAX_ITERATIONS {
        return Err("rj failed to converge");
    }
    let ea = delx * (dely + delz) + dely * delz;
    let eb = delx * dely * delz;
    let ec = delp * delp;
    let ed = ea - 3.0 * ec;
    let ee = eb + 2.0 * delp * (ea - ec);
    let mut ans = 3.0 * sum
        + fac
            * (1.0
                + ed * (-RJ_C1 + RJ_C5 * ed - RJ_C6 * ee)
                + eb * (RJ_C7 + delp * (-RJ_C8 + delp * RJ_C4))
                + delp * ea * (RJ_C2 - delp * RJ_C3)
                - RJ_C2 * delp * ec)
            / (ave * f64::sqrt(ave));
    if p <= 0.0 {
        ans = a * (b * ans + 3.0 * (rcx - rf(xt, yt, zt)?));
    }
    Ok(ans)
}

/// Computes the degenerate elliptic integral using Carlson's formula
///
/// Computes Rc(x,y) where x must be nonnegative and y must be nonzero.
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
        return Err("(x,y) must be non-negative");
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
        let alamb = 2.0 * f64::sqrt(xt) * f64::sqrt(yt) + yt;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
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
    use super::{elliptic_1, elliptic_2, elliptic_3};
    use crate::approx_eq;

    #[test]
    fn elliptic_1_works() {
        println!("F = {:?}", elliptic_1(0.3, 0.8).unwrap());
        approx_eq(elliptic_1(0.3, 0.8).unwrap(), 0.303652, 1e-3);
    }

    #[test]
    fn elliptic_2_works() {
        println!("E = {:?}", elliptic_2(0.3, 0.8).unwrap());
        approx_eq(elliptic_2(0.3, 0.8).unwrap(), 0.296426, 1e-3);
    }

    #[test]
    fn elliptic_3_works() {
        println!("Π = {:?}", elliptic_3(2.0, 0.3, 0.8).unwrap());
        approx_eq(elliptic_3(2.0, 0.3, 0.8).unwrap(), 0.323907, 1e-3);
    }
}
