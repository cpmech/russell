use super::{Bracket, NoArgs};
use crate::math::{LN2, NAPIER, PI};
use crate::StrError;

/// Holds f(x) functions for tests
#[allow(unused)]
pub(super) struct TestFunction {
    /// Holds the name of the function
    pub name: &'static str,

    /// Holds the f(x) function
    pub f: fn(f64, &mut NoArgs) -> Result<f64, StrError>,

    /// Holds a bracketed local minimum
    pub min1: Option<Bracket>,

    // Holds another bracketed local minimum
    pub min2: Option<Bracket>,

    // Holds another bracketed local minimum
    pub min3: Option<Bracket>,

    /// Holds a bracketed root
    pub root1: Option<Bracket>,

    /// Holds another bracketed root
    pub root2: Option<Bracket>,

    /// Holds another bracketed root
    pub root3: Option<Bracket>,

    /// Holds the result of the integration `ii = ∫_a^b f(x) dx`
    ///
    /// The data is `(a, b, ii)`
    pub integral: Option<(f64, f64, f64)>,

    /// Tolerance for checking the minimum (using Brent's method)
    pub tol_min: f64,

    /// Tolerance for checking the root (using Brent's method)
    pub tol_root: f64,

    /// Tolerance for checking the integral
    pub tol_integral: f64,
}

/// Allocates f(x) test functions
#[allow(dead_code)]
pub(super) fn get_functions() -> Vec<TestFunction> {
    vec![
        TestFunction {
            name: "x² - 1",
            f: |x, _| Ok(x * x - 1.0),
            min1: Some(Bracket {
                a: -5.0,
                b: 5.0,
                fa: 24.0,
                fb: 24.0,
                xo: 0.0,
                fxo: -1.0,
            }),
            min2: None,
            min3: None,
            root1: Some(Bracket {
                a: -2.0,
                b: 0.0,
                fa: 3.0,
                fb: -1.0,
                xo: -1.0,
                fxo: 0.0,
            }),
            root2: Some(Bracket {
                a: 0.0,
                b: 2.0,
                fa: -1.0,
                fb: 3.0,
                xo: 1.0,
                fxo: 0.0,
            }),
            root3: None,
            integral: Some((-4.0, 4.0, 104.0 / 3.0)),
            tol_min: 1e-10,
            tol_root: 1e-10,
            tol_integral: 1e-50,
        },
        TestFunction {
            name: "1/2 - 1/(1 + 16 x²)", // (shifted) Runge equation
            f: |x, _| Ok(1.0 / 2.0 - 1.0 / (1.0 + 16.0 * x * x)),
            min1: Some(Bracket {
                a: -2.0,
                b: 2.0,
                fa: 63.0 / 130.0,
                fb: 63.0 / 130.0,
                xo: 0.0,
                fxo: -1.0 / 2.0,
            }),
            min2: None,
            min3: None,
            root1: Some(Bracket {
                a: -2.0,
                b: 0.0,
                fa: 63.0 / 130.0,
                fb: -1.0 / 2.0,
                xo: -1.0 / 4.0,
                fxo: 0.0,
            }),
            root2: Some(Bracket {
                a: 0.0,
                b: 2.0,
                fa: -1.0 / 2.0,
                fb: 63.0 / 130.0,
                xo: 1.0 / 4.0,
                fxo: 0.0,
            }),
            root3: None,
            integral: Some((-2.0, 2.0, 2.0 - f64::atan(8.0) / 2.0)),
            tol_min: 1e-8,
            tol_root: 1e-13,
            tol_integral: 1e-14,
        },
        TestFunction {
            name: "x⁵ + 3x⁴ - 2x³ + x - 1",
            f: |x, _| Ok(f64::powi(x, 5) + 3.0 * f64::powi(x, 4) - 2.0 * f64::powi(x, 3) + x - 1.0),
            min1: Some(Bracket {
                a: -2.0,
                b: 2.0,
                fa: 29.0,
                fb: 65.0,
                xo: -0.326434701525930898665902357162,
                fxo: -1.22650698564642753377955683652,
            }),
            min2: None,
            min3: None,
            root1: Some(Bracket {
                a: -4.0,
                b: -2.0,
                fa: -133.0,
                fb: 29.0,
                xo: -3.53652558839295230222542848627,
                fxo: 0.0,
            }),
            root2: Some(Bracket {
                a: -2.0,
                b: 0.0,
                fa: 29.0,
                fb: -1.0,
                xo: -0.781407421874263267559694073091,
                fxo: 0.0,
            }),
            root3: Some(Bracket {
                a: 0.0,
                b: 2.0,
                fa: -1.0,
                fb: 65.0,
                xo: 0.727096464661451721867714112038,
                fxo: 0.0,
            }),
            integral: Some((-3.0, 2.0, 475.0 / 6.0)),
            tol_min: 1e-8,
            tol_root: 1e-11,
            tol_integral: 1e-13,
        },
        TestFunction {
            name: "(x - 1)² + 5 sin(x)",
            f: |x, _| Ok(f64::powi(x - 1.0, 2) + 5.0 * f64::sin(x)),
            min1: Some(Bracket {
                a: -2.0,
                b: 2.0,
                fa: 4.45351286587159152301990067044,
                fb: 5.54648713412840847698009932956,
                xo: -0.779014930395140333216421108317,
                fxo: -0.347999771320472050094111906591,
            }),
            min2: Some(Bracket {
                a: 2.0,
                b: 5.0,
                fa: 5.54648713412840847698009932956,
                fb: 11.2053786266843076555342279692,
                xo: 3.41029230994771356210845446934,
                fxo: 4.48211912850661077326295235125,
            }),
            min3: None,
            root1: Some(Bracket {
                a: -2.0,
                b: -0.7,
                fa: 4.4535128658715915230199006704,
                fb: -0.3310884361884554,
                xo: -1.12294626691885210931752137234,
                fxo: 0.0,
            }),
            root2: Some(Bracket {
                a: -0.7,
                b: 1.0,
                fa: -0.3310884361884554,
                fb: 4.20735492403948253326251160815,
                xo: -0.407207140869762981181312249859,
                fxo: 0.0,
            }),
            root3: None,
            integral: Some((-3.0, 5.0, 128.0 / 3.0 + 5.0 * f64::cos(3.0) - 5.0 * f64::cos(5.0))),
            tol_min: 1e-8,
            tol_root: 1e-12,
            tol_integral: 1e-20,
        },
        TestFunction {
            name: "1/(1 - exp(-2 x) sin²(5 π x)) - 3/2",
            f: |x, _| Ok(1.0 / (1.0 - f64::exp(-2.0 * x) * f64::powi(f64::sin(5.0 * PI * x), 2)) - 1.5),
            min1: Some(Bracket {
                a: 0.1,
                b: 0.3,
                fa: 4.016655566126993,
                fb: 0.7163692151608707,
                xo: 0.2,
                fxo: -0.5,
            }),
            min2: Some(Bracket {
                a: 0.3,
                b: 0.5,
                fa: 0.7163692151608707,
                fb: 0.08197670686932645,
                xo: 0.4,
                fxo: -0.5,
            }),
            min3: Some(Bracket {
                a: 0.5,
                b: 0.7,
                fa: 0.08197670686932645,
                fb: -0.17268918209868533,
                xo: 0.6,
                fxo: -0.5,
            }),
            root1: Some(Bracket {
                a: 0.1,
                b: 0.2,
                fa: 4.016655566126993,
                fb: -0.5,
                xo: 0.153017232138599937021144629730,
                fxo: 0.0,
            }),
            root2: Some(Bracket {
                a: 0.2,
                b: 0.3,
                fa: -0.5,
                fb: 0.7163692151608707,
                xo: 0.253401241496921840876888995872,
                fxo: 0.0,
            }),
            root3: Some(Bracket {
                a: 0.3,
                b: 0.4,
                fa: 0.7163692151608707,
                fb: -0.5,
                xo: 0.339787495774806018201668030092,
                fxo: 0.0,
            }),
            integral: Some((0.0, 1.0, -0.0267552190488911754674985952882)), // From Mathematica NIntegrate
            tol_min: 1e-9,
            tol_root: 1e-12,
            tol_integral: 1e-14,
        },
        TestFunction {
            name: "sin(x) in [0, π]",
            f: |x, _| Ok(f64::sin(x)),
            min1: None,
            min2: None,
            min3: None,
            root1: None,
            root2: None,
            root3: None,
            integral: Some((0.0, PI, 2.0)),
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-15,
        },
        TestFunction {
            name: "sin(x) in [0, π/2]",
            f: |x, _| Ok(f64::sin(x)),
            min1: None,
            min2: None,
            min3: None,
            root1: None,
            root2: None,
            root3: None,
            integral: Some((0.0, PI / 2.0, 1.0)),
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-15,
        },
        TestFunction {
            name: "sin(x) in [-1, 1]",
            f: |x, _| Ok(f64::sin(x)),
            min1: None,
            min2: None,
            min3: None,
            root1: None,
            root2: None,
            root3: None,
            integral: Some((-1.0, 1.0, 0.0)),
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-20,
        },
        TestFunction {
            name: "0.092834 sin(77.0001 + 19.87 x) in [-2.34567, 12.34567]",
            f: |x, _| Ok(0.092834 * f64::sin(77.0001 + 19.87 * x)),
            min1: None,
            min2: None,
            min3: None,
            root1: None,
            root2: None,
            root3: None,
            integral: Some((-2.34567, 12.34567, 0.00378787099369719)),
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-15,
        },
        TestFunction {
            name: "0.092834 sin[7.0001 + 1.87 x) in [-2.34567, 1.34567]",
            f: |x, _| Ok(0.092834 * f64::sin(7.0001 + 1.87 * x)),
            min1: None,
            min2: None,
            min3: None,
            root1: None,
            root2: None,
            root3: None,
            integral: Some((-2.34567, 1.34567, 0.00654937363510264)),
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-16,
        },
        TestFunction {
            name: "(2 x⁵ - x + 3)/x²",
            f: |x, _| Ok((2.0 * f64::powi(x, 5) - x + 3.0) / (x * x)),
            min1: None,
            min2: None,
            min3: None,
            root1: None,
            root2: None,
            root3: None,
            integral: Some((1.0, 2.0, 9.0 - LN2)),
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-13,
        },
        TestFunction {
            name: "3/exp(-x) - 1/(3x)",
            f: |x, _| Ok(3.0 / f64::exp(-x) - 1.0 / (3.0 * x)),
            min1: None,
            min2: None,
            min3: None,
            root1: None,
            root2: None,
            root3: None,
            integral: Some((-20.0, -1.0, 3.0 / NAPIER - 3.0 / f64::exp(20.0) + f64::ln(20.0) / 3.0)),
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-14,
        },
        TestFunction {
            name: "log(2 Cos(x/2))",
            f: |x, _| Ok(f64::ln(2.0 * f64::cos(x / 2.0))),
            min1: None,
            min2: None,
            min3: None,
            root1: None,
            root2: None,
            root3: None,
            integral: Some((-PI, PI, 0.0)),
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-10,
        },
        TestFunction {
            name: "exp(x)",
            f: |x, _| Ok(f64::exp(x)),
            min1: None,
            min2: None,
            min3: None,
            root1: None,
            root2: None,
            root3: None,
            integral: Some((0.0, 10.1, f64::exp(10.1) - f64::exp(0.0))), // exp(b) - exp(a)
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-10,
        },
    ]
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::get_functions;
    use crate::algo::Bracket;
    use crate::approx_eq;

    fn check_consistency_min(bracket: &Bracket) {
        assert!(bracket.a < bracket.xo);
        assert!(bracket.xo < bracket.b);
        assert!(bracket.fa > bracket.fxo);
        assert!(bracket.fb > bracket.fxo);
    }

    fn check_consistency_root(bracket: &Bracket) {
        assert!(bracket.a < bracket.xo);
        assert!(bracket.xo < bracket.b);
        assert!(bracket.fa * bracket.fb < 0.0);
    }

    #[test]
    fn functions_are_consistent() {
        let args = &mut 0;
        for func in &get_functions() {
            println!("\n{}", func.name);
            if let Some(bracket) = &func.min1 {
                check_consistency_min(bracket);
                assert_eq!(bracket.fa, (func.f)(bracket.a, args).unwrap());
                assert_eq!(bracket.fb, (func.f)(bracket.b, args).unwrap());
                approx_eq(bracket.fxo, (func.f)(bracket.xo, args).unwrap(), 1e-15);
            }
            if let Some(bracket) = &func.min2 {
                check_consistency_min(bracket);
                assert_eq!(bracket.fa, (func.f)(bracket.a, args).unwrap());
                assert_eq!(bracket.fb, (func.f)(bracket.b, args).unwrap());
                assert_eq!(bracket.fxo, (func.f)(bracket.xo, args).unwrap());
            }
            if let Some(bracket) = &func.min3 {
                check_consistency_min(bracket);
                assert_eq!(bracket.fa, (func.f)(bracket.a, args).unwrap());
                assert_eq!(bracket.fb, (func.f)(bracket.b, args).unwrap());
                assert_eq!(bracket.fxo, (func.f)(bracket.xo, args).unwrap());
            }
            if let Some(bracket) = &func.root1 {
                check_consistency_root(bracket);
                assert_eq!(bracket.fa, (func.f)(bracket.a, args).unwrap());
                assert_eq!(bracket.fb, (func.f)(bracket.b, args).unwrap());
                approx_eq((func.f)(bracket.xo, args).unwrap(), 0.0, 1e-13);
                assert_eq!(bracket.fxo, 0.0);
            }
            if let Some(bracket) = &func.root2 {
                check_consistency_root(bracket);
                assert_eq!(bracket.fa, (func.f)(bracket.a, args).unwrap());
                assert_eq!(bracket.fb, (func.f)(bracket.b, args).unwrap());
                approx_eq((func.f)(bracket.xo, args).unwrap(), 0.0, 1e-15);
                assert_eq!(bracket.fxo, 0.0);
            }
            if let Some(bracket) = &func.root3 {
                check_consistency_root(bracket);
                assert_eq!(bracket.fa, (func.f)(bracket.a, args).unwrap());
                assert_eq!(bracket.fb, (func.f)(bracket.b, args).unwrap());
                approx_eq((func.f)(bracket.xo, args).unwrap(), 0.0, 1e-14);
                assert_eq!(bracket.fxo, 0.0);
            }
        }
    }
}
