use super::{float_is_neg_int, gamma, ln_gamma};

// The code here is partially based on the beta.c file from Cephes
//
// Cephes Math Library Release 2.0:  April, 1987
// Copyright 1984, 1987 by Stephen L. Moshier
// Direct inquiries to 30 Frost Street, Cambridge, MA 02140
//
//    Some software in this archive may be from the book _Methods and
// Programs for Mathematical Functions_ (Prentice-Hall or Simon & Schuster
// International, 1989) or from the Cephes Mathematical Library, a
// commercial product. In either event, it is copyrighted by the author.
// What you see here may be used freely but it comes with no support or
// guarantee.
//
//    The two known misprints in the book are repaired here in the
// source listings for the gamma function and the incomplete beta
// integral.
//
// Stephen L. Moshier
// moshier@na-net.ornl.gov

const GAMMA_MAX: f64 = 171.624376956302725;

const LN_MAX: f64 = 7.09782712893383996732e2; // ln(DBL_MAX)

const ASYMPTOTIC_FACTOR: f64 = 1e6;

/// Evaluates the Euler beta function B(a, b)
///
/// ```text
///                         1
///           Γ(a) Γ(b)    ⌠
/// B(a, b) = ───────── =  │  tᵃ⁻¹ (1 - t)ᵇ⁻¹  dt
///            Γ(a + b)    ⌡
///                       0
/// ```
///
/// where `Γ(x)` is the Gamma function.
///
/// The function is evaluated using either the Γ(x) ([gamma()]) function or the
/// `ln(Γ(x))` ([ln_gamma()]) function--see below.
///
/// Considering first non-negative `ln(Γ(x))` values for `a`, `b`, and `c = a + b`:
///
/// ```text
/// B(a, b) = Γ(a) Γ(b) [Γ(c)]⁻¹
///         = exp{ln(Γ(a))} exp{ln(Γ(b))} exp{ln([Γ(c)]⁻¹)}
///         = exp{ln(Γ(a))} exp{ln(Γ(b))} exp{-ln([Γ(c)])}
///         = exp(la + lb - lb)
/// ```
///
/// where `la = ln(Γ(a))`, `lb = ln(Γ(b))`, and `lc = ln(Γ(a + b))`
///
/// Now, fixing the sign:
///
/// ```text
/// B(a, b) = sign(la) sign(lb) sign(lc) exp(la + lb - lb)
/// ```
///
/// Note: `B` is the greek capital beta; although we use the latin character B.
///
/// See: <https://mathworld.wolfram.com/BetaFunction.html>
///
/// See also: <https://en.wikipedia.org/wiki/Beta_function>
///
/// # Notable results
///
/// ```text
/// B(a, b)   = B(b, a)
/// B(1, x)   = 1/x
/// B(x, 1-x) = π/sin(π x)
/// B(1, 1)   = 1
/// B(-1, 1)  = -1
/// ```
pub fn beta_function(a: f64, b: f64) -> f64 {
    // special cases
    if f64::is_nan(a) || f64::is_nan(b) {
        return f64::NAN;
    } else if f64::is_infinite(a) || f64::is_infinite(b) {
        // Mathematica:
        //   Table[FunctionExpand[ Beta[a, b]], {a, {-\[Infinity], \[Infinity]}}, {b, {-\[Infinity], \ \[Infinity]}}]
        //   {{Indeterminate, Indeterminate}, {Indeterminate, Indeterminate}}
        return f64::NAN;
    }

    // handle negative integer a
    if a <= 0.0 {
        if a == f64::floor(a) {
            // the condition below checks for overflow, not only integer
            // because only overflows yield `a == floor(a) && a != int(a)`
            if a == (a as i32) as f64 {
                // ok
                return beta_negative_integer(a, b);
            } else {
                // overflow
                return f64::INFINITY;
            }
        }
    }

    // handle negative integer b
    if b <= 0.0 {
        if b == f64::floor(b) {
            if b == (b as i32) as f64 {
                return beta_negative_integer(b, a);
            } else {
                return f64::INFINITY;
            }
        }
    }

    // maybe swap a and b
    let (aa, bb) = if f64::abs(a) < f64::abs(b) { (b, a) } else { (a, b) };

    // avoid loss of precision in ln_gamma(a + b) - ln_gamma(a)
    if f64::abs(aa) > ASYMPTOTIC_FACTOR * f64::abs(bb) && aa > ASYMPTOTIC_FACTOR {
        let (ll, sign) = ln_beta_asymptotic(aa, bb);
        return (sign as f64) * f64::exp(ll);
    }

    // use the ln_gamma identity
    let cc = aa + bb;
    if f64::abs(cc) > GAMMA_MAX || f64::abs(aa) > GAMMA_MAX || f64::abs(bb) > GAMMA_MAX {
        let (la, sign_la) = ln_gamma(aa);
        let (lb, sign_lb) = ln_gamma(bb);
        let (lc, sign_lc) = ln_gamma(cc);
        let sign = sign_la * sign_lb * sign_lc;
        let ll = la + lb - lc;
        if ll > LN_MAX {
            return (sign as f64) * f64::INFINITY;
        }
        return (sign as f64) * f64::exp(ll);
    }

    // handle negative integer cc (yields undefined Gamma; NaN)
    if float_is_neg_int(cc) {
        // this case is not handled by Cephes; but it works in Cephes because their Gamma returns Inf for neg ints
        // in this case, the "horns" of the Gamma function on the negative axis
        // are "upside" and "downside" @ negative integers; thus, Gamma(neg_int) is undefined
        // however, Gamma(cc) being the denominator of Beta(a,b) and since neg_int a and b
        // have been already handled, the division of Gamma(a)*Gamma(b) by either
        // -Inf or Inf will yield Beta(a,b) = 0
        return 0.0;
    }

    // use the gamma identity
    let gc = gamma(cc);
    // the gamma function has no zeros, so there is no need to check for gc == 0.0
    // if gc == 0.0 { return f64::INFINITY; }
    let ga = gamma(aa);
    let gb = gamma(bb);
    let mut ans: f64;
    if f64::abs(f64::abs(ga) - f64::abs(gc)) > f64::abs(f64::abs(gb) - f64::abs(gc)) {
        ans = gb / gc;
        ans *= ga;
    } else {
        ans = ga / gc;
        ans *= gb;
    };
    ans
}

/// Handles the special case with a negative integer argument
fn beta_negative_integer(a_int: f64, b: f64) -> f64 {
    let b_int = (b as i32) as f64;
    if b == b_int && 1.0 - a_int - b > 0.0 {
        let sign = if (b_int as i32) % 2 == 0 { 1.0 } else { -1.0 };
        sign * beta_function(1.0 - a_int - b, b)
    } else {
        f64::INFINITY
    }
}

/// Implements an asymptotic expansion for the natural logarithm of the beta function
///
/// Returns `(lnb, sign)` where:
///
/// lnb = ln(|B(a, b)|)  for  a > ASYMPTOTIC_FACTOR * max(|b|, 1)
fn ln_beta_asymptotic(a: f64, b: f64) -> (f64, i32) {
    let (mut r, sign) = ln_gamma(b);
    r -= b * f64::ln(a);
    r += b * (1.0 - b) / (2.0 * a);
    r += b * (1.0 - b) * (1.0 - 2.0 * b) / (12.0 * a * a);
    r += -b * b * (1.0 - b) * (1.0 - b) / (12.0 * a * a * a);
    (r, sign)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{beta_function, ASYMPTOTIC_FACTOR};
    use crate::approx_eq;
    use crate::math::PI;

    #[test]
    fn beta_function_works_1() {
        approx_eq(beta_function(0.5, 0.5), PI, 1e-15);
        assert_eq!(beta_function(1.0, 1.0), 1.0);
        assert_eq!(beta_function(-1.0, 1.0), -1.0);
        approx_eq(beta_function(1.0, -0.5), -2.0, 1e-15);
        let aa = [1.0, 3.0, 10.0];
        let bb = [5.0, 2.0, 11.0, -0.5];
        let wx_maxima_solution = [
            [1.0 / 5.0, 1.0 / 2.0, 1.0 / 11.0, -2.0],
            [1.0 / 105.0, 1.0 / 12.0, 1.0 / 858.0, -16.0 / 3.0],
            [1.0 / 10010.0, 1.0 / 110.0, 1.0 / 1847560.0, -131072.0 / 12155.0],
        ];
        for (i, a) in aa.iter().enumerate() {
            for (j, b) in bb.iter().enumerate() {
                let tol = if i == 2 && j == 3 { 5e-14 } else { 1e-15 };
                let beta = beta_function(*a, *b);
                // println!("a = {:?}, b = {:?}, B(a,b) = {:?}", a, b, beta);
                approx_eq(beta, wx_maxima_solution[i][j], tol);
            }
        }
    }

    #[test]
    fn beta_function_handle_branches_1() {
        // c = a + b = -3 yielding Gamma(c) = NaN
        assert_eq!(beta_function(-1.4, -1.6), 0.0);
        assert_eq!(beta_function(-1.5, -1.5 + f64::EPSILON), 0.0);
        approx_eq(beta_function(-1.5, -1.5 + 10.0 * f64::EPSILON), 0.0, 1e-13);

        let tiny_neg_int = -f64::trunc(f64::MAX);
        // a <= 0.0  and  a == floor(a)  and  a != int(a)
        assert_eq!(beta_function(tiny_neg_int, 3.145), f64::INFINITY);

        // b <= 0.0  and  b == floor(b)  and  b != int(b)
        assert_eq!(beta_function(3.145, tiny_neg_int), f64::INFINITY);

        // in beta_negative_integer
        assert_eq!(beta_function(1.6, -2.0), f64::INFINITY);

        // int the recursion loop
        assert_eq!(beta_function(-2.0, -2.0), f64::INFINITY);
    }

    #[test]
    fn beta_function_handle_branches_2() {
        // abs(a) > ASYMPTOTIC_FACTOR * abs(b) && a > ASYMPTOTIC_FACTOR
        let b = 1.0;
        let a = ASYMPTOTIC_FACTOR * f64::abs(b) + 1.0; // 1_000_001
        approx_eq(
            beta_function(a, b),
            9.9999900000099999900000099999900000099999900000100e-7,
            1e-21,
        ); // Mathematica: N[Beta[1000001, 1], 50]

        // abs(a) > ASYMPTOTIC_FACTOR * abs(b) && a > ASYMPTOTIC_FACTOR
        let b = -0.5;
        let a = ASYMPTOTIC_FACTOR * f64::abs(b) + 1.0; // 500_001
        approx_eq(beta_function(a, b), -2506.62890, 1e-6); // Mathematica: NumberForm[N[Beta[500001, -0.5], 50], 50]

        // f64::abs(cc) > GAMMA_MAX || f64::abs(aa) > GAMMA_MAX || f64::abs(bb) > GAMMA_MAX
        let a = 172.0; // > GAMMA_MAX
        let b = 1.0;
        approx_eq(
            beta_function(a, b),
            0.0058139534883720930232558139534883720930232558139535,
            1e-15,
        ); // Mathematica: NumberForm[N[Beta[172, 1], 50], 50]

        // f64::abs(cc) > GAMMA_MAX || f64::abs(aa) > GAMMA_MAX || f64::abs(bb) > GAMMA_MAX
        let a = 1000.0;
        let b = -172.5;
        // Mathematica: NumberForm[N[Beta[1000, -172.5], 50], 50]
        //   -4.35702817322*10^(198)
        approx_eq(beta_function(a, b) / 1e198, -4.35702817322, 1e-11);

        // f64::abs(cc) > GAMMA_MAX || f64::abs(aa) > GAMMA_MAX || f64::abs(bb) > GAMMA_MAX
        // and ll > LN_MAX
        let a = 4500.0;
        let b = -172.5;
        // Mathematica: NumberForm[N[Beta[4500, -172.5], 50], 50]
        //   -5.8238626991*10^(316)
        assert_eq!(beta_function(a, b), f64::NEG_INFINITY);
    }

    #[test]
    fn beta_function_special_cases() {
        assert!(beta_function(f64::NAN, 3.145).is_nan());
        assert!(beta_function(f64::INFINITY, 3.145).is_nan());
        assert!(beta_function(f64::NEG_INFINITY, 3.145).is_nan());
        assert!(beta_function(3.145, f64::NAN).is_nan());
        assert!(beta_function(3.145, f64::INFINITY).is_nan());
        assert!(beta_function(3.145, f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn beta_function_works_2() {
        // Mathematica:
        // res = Table[{a, b, NumberForm[Beta[a, b], 50]}, {a, -2, 2, 0.3}, {b, -2, 2, 0.4}];
        // tab = Flatten[res, {{1, 2}, {3}}];
        // Export["test.txt", tab, "Table", "FieldSeparators" -> ", "]
        let mathematica = [
            (-2., -2., f64::INFINITY),
            (-2., -1.6, f64::INFINITY),
            (-2., -1.2, f64::INFINITY),
            (-2., -0.8, f64::INFINITY),
            (-2., -0.4, f64::INFINITY),
            (-2., 0., f64::INFINITY),
            (-2., 0.4, f64::INFINITY),
            (-2., 0.8, f64::INFINITY),
            (-2., 1.2, f64::INFINITY),
            (-2., 1.6, f64::INFINITY),
            (-2., 2., 0.5),
            (-1.7, -2., f64::INFINITY),
            (-1.7, -1.6, 13.24606205595081),
            (-1.7, -1.2, -6.356717816192493),
            (-1.7, -0.8, 15.26092710491747),
            (-1.7, -0.4, 2.023149527999408),
            (-1.7, 0., f64::INFINITY),
            (-1.7, 0.4, 1.675391248183499),
            (-1.7, 0.8, -0.2768806308323415),
            (-1.7, 1.2, -0.6511328898098121),
            (-1.7, 1.6, -0.2101973535583807),
            (-1.7, 2., 0.840336134453782),
            (-1.4, -2., f64::INFINITY),
            (-1.4, -1.6, 0.),
            (-1.4, -1.2, -14.51583351429044),
            (-1.4, -0.8, 6.920867017942697),
            (-1.4, -0.4, -3.105442553146037),
            (-1.4, 0., f64::INFINITY),
            (-1.4, 0.4, -2.73142155318082e-15),
            (-1.4, 0.8, -0.837451933516756),
            (-1.4, 1.2, -0.4194464859359214),
            (-1.4, 1.6, 0.5175737588576731),
            (-1.4, 2., 1.785714285714286),
            (-1.1, -2., f64::INFINITY),
            (-1.1, -1.6, -24.1083451046653),
            (-1.1, -1.2, -32.56573049035103),
            (-1.1, -0.8, -10.02056263672268),
            (-1.1, -0.4, -15.30422183849654),
            (-1.1, 0., f64::INFINITY),
            (-1.1, 0.4, -5.042268257184899),
            (-1.1, 0.8, -2.613971678155267),
            (-1.1, 1.2, 0.937596504020837),
            (-1.1, 1.6, 4.897350988318892),
            (-1.1, 2., 9.09090909090908),
            (-0.8, -2., f64::INFINITY),
            (-0.8, -1.6, 11.96664947264486),
            (-0.8, -1.2, 0.),
            (-0.8, -0.8, 14.25225208482654),
            (-0.8, -0.4, 4.40418810232718),
            (-0.8, 0., f64::INFINITY),
            (-0.8, 0.4, 3.419042706469955),
            (-0.8, 0.8, -1.483479078628329e-15),
            (-0.8, 1.2, -2.375375347471091),
            (-0.8, 1.6, -4.404188102327178),
            (-0.8, 2., -6.250000000000001),
            (-0.5, -2., f64::INFINITY),
            (-0.5, -1.6, 1.770563978070331),
            (-0.5, -1.2, -6.84038125988271),
            (-0.5, -0.8, 6.11193679598388),
            (-0.5, -0.4, -1.248525863317827),
            (-0.5, 0., f64::INFINITY),
            (-0.5, 0.4, 0.7358187960811731),
            (-0.5, 0.8, -1.379572691068783),
            (-0.5, 1.2, -2.507461249634415),
            (-0.5, 1.6, -3.329402302180867),
            (-0.5, 2., -4.),
            (-0.2, -2., f64::INFINITY),
            (-0.2, -1.6, -4.218909550127697),
            (-0.2, -1.2, -10.61874962955045),
            (-0.2, -0.8, 0.),
            (-0.2, -0.4, -5.86216353461729),
            (-0.2, 0., f64::INFINITY),
            (-0.2, 0.4, -2.812606366751801),
            (-0.2, 0.8, -4.550892698378767),
            (-0.2, 1.2, -5.344796660577971),
            (-0.2, 1.6, -5.862163534617291),
            (-0.2, 2., -6.249999999999996),
            (0.1, -2., f64::INFINITY),
            (0.1, -1.6, 9.30140485640914),
            (0.1, -1.2, 4.750441365819462),
            (0.1, -0.8, 12.77445005616922),
            (0.1, -0.4, 8.18576926242165),
            (0.1, 0., f64::INFINITY),
            (0.1, 0.4, 11.9057982162037),
            (0.1, 0.8, 10.36459934360612),
            (0.1, 1.2, 9.73291432850988),
            (0.1, 1.6, 9.35516487133903),
            (0.1, 2., 9.09090909090908),
            (0.4, -2., f64::INFINITY),
            (0.4, -1.6, 1.056542300792933),
            (0.4, -1.2, -1.875070911167869),
            (0.4, -0.8, 3.419042706469956),
            (0.4, -0.4, 0.),
            (0.4, 0., f64::INFINITY),
            (0.4, 0.4, 4.226169203171728),
            (0.4, 0.8, 2.812606366751803),
            (0.4, 1.2, 2.279361804313306),
            (0.4, 1.6, 1.981959599516475),
            (0.4, 2., 1.785714285714286),
            (0.7, -2., f64::INFINITY),
            (0.7, -1.6, -0.2837374022189633),
            (0.7, -1.2, -1.776297527149304),
            (0.7, -0.8, 0.6970579619812344),
            (0.7, -0.4, -1.615418153370992),
            (0.7, 0., f64::INFINITY),
            (0.7, 0.4, 3.026532290335616),
            (0.7, 0.8, 1.705245626063332),
            (0.7, 1.2, 1.239214154633306),
            (0.7, 1.6, 0.994103478997534),
            (0.7, 2., 0.840336134453782),
            (1., -2., -0.5),
            (1., -1.6, -0.625),
            (1., -1.2, -0.833333333333333),
            (1., -0.8, -1.25),
            (1., -0.4, -2.5),
            (1., 0., f64::INFINITY),
            (1., 0.4, 2.499999999999998),
            (1., 0.8, 1.25),
            (1., 1.2, 0.833333333333333),
            (1., 1.6, 0.625),
            (1., 2., 0.5),
            (1.3, -2., f64::INFINITY),
            (1.3, -1.6, -0.479258554167877),
            (1.3, -1.2, 0.4576221537367847),
            (1.3, -0.8, -2.905680520776285),
            (1.3, -0.4, -3.126685634180918),
            (1.3, 0., f64::INFINITY),
            (1.3, 0.4, 2.190896247624578),
            (1.3, 0.8, 0.998448335425713),
            (1.3, 1.2, 0.6198785110989412),
            (1.3, 1.6, 0.4388330714639885),
            (1.3, 2., 0.3344481605351171),
            (1.6, -2., f64::INFINITY),
            (1.6, -1.6, -9.16840492191685e-16),
            (1.6, -1.2, 1.954054511539095),
            (1.6, -0.8, -4.404188102327175),
            (1.6, -0.4, -3.623016312003708),
            (1.6, 0., f64::INFINITY),
            (1.6, 0.4, 1.981959599516473),
            (1.6, 0.8, 0.837451933516756),
            (1.6, 1.2, 0.4893542335919088),
            (1.6, 1.6, 0.3293651192730644),
            (1.6, 2., 0.2403846153846155),
            (1.9, -2., f64::INFINITY),
            (1.9, -1.6, 0.7428341628888476),
            (1.9, -1.2, 3.5942110574635),
            (1.9, -0.8, -5.801378368628927),
            (1.9, -0.4, -4.040314565363094),
            (1.9, 0., f64::INFINITY),
            (1.9, 0.4, 1.828514862495624),
            (1.9, 0.8, 0.72488290234558),
            (1.9, 1.2, 0.4018270731517874),
            (1.9, 1.6, 0.2585801321832379),
            (1.9, 2., 0.1814882032667877),
        ];
        for (a, b, reference) in mathematica {
            // println!("a = {:?}, b = {:?}", a, b);
            if reference == f64::INFINITY {
                assert_eq!(beta_function(a, b), f64::INFINITY);
            } else {
                approx_eq(beta_function(a, b), reference, 1e-13);
            }
        }
    }
}
