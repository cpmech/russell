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
    if f64::abs(k * s - 1.0) < f64::EPSILON {
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
    use super::{elliptic_e, elliptic_f};
    use crate::approx_eq;
    use crate::math::{PI, SQRT_2};

    #[test]
    fn elliptic_f_captures_errors() {
        assert_eq!(elliptic_f(-1.0, 0.0).err(), Some("phi and k must be non-negative"));
        assert_eq!(elliptic_f(1.0, -1.0).err(), Some("phi and k must be non-negative"));
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
}
