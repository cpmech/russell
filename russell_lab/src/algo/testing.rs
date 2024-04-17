use super::{Bracket, NoArgs};
use crate::math::{LN2, NAPIER, PI};
use crate::StrError;

/// Holds an f(x) function that is useful for testing
pub struct TestFunction {
    /// Holds the name of the function
    pub name: &'static str,

    /// Holds the f(x) function
    pub f: fn(f64, &mut NoArgs) -> Result<f64, StrError>,

    /// Holds the first derivative of f(x) w.r.t x
    pub g: fn(f64, &mut NoArgs) -> Result<f64, StrError>,

    /// Holds the second derivative of f(x) w.r.t x
    pub h: fn(f64, &mut NoArgs) -> Result<f64, StrError>,

    /// Holds the range of interest of f(x)
    ///
    /// The values are `(xmin, xmax)` and are useful for
    /// plotting the function, for instance.
    pub range: (f64, f64),

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
    /// The data is `(a, b, ii)` where `a` and `b` are the lower
    /// and upper bounds of integration, respectively, and `ii`
    /// is the result of integration.
    pub integral: Option<(f64, f64, f64)>,

    /// Tolerance for checking the minimum (using Brent's method)
    pub tol_min: f64,

    /// Tolerance for checking the root (using Brent's method)
    pub tol_root: f64,

    /// Tolerance for checking the integral
    pub tol_integral: f64,
}

/// Generates f(x) functions for testing
///
/// ![001](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_001.svg)
///
/// ![002](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_002.svg)
///
/// ![003](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_003.svg)
///
/// ![004](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_004.svg)
///
/// ![005](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_005.svg)
///
/// ![006](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_006.svg)
///
/// ![007](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_007.svg)
///
/// ![008](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_008.svg)
///
/// ![009](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_009.svg)
///
/// ![010](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_010.svg)
///
/// ![011](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_011.svg)
///
/// ![012](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_012.svg)
///
/// ![013](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_013.svg)
///
/// ![014](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/test_function_014.svg)
pub fn get_test_functions() -> Vec<TestFunction> {
    vec![
        TestFunction {
            name: "0: f(x) = undefined",
            f: |_, _| Err("stop"),
            g: |_, _| Err("stop"),
            h: |_, _| Err("stop"),
            range: (-5.0, 5.0),
            min1: None,
            min2: None,
            min3: None,
            root1: None,
            root2: None,
            root3: None,
            integral: None,
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-10,
        },
        TestFunction {
            name: "1: f(x) = x² - 1",
            f: |x, _| Ok(x * x - 1.0),
            g: |x, _| Ok(2.0 * x),
            h: |_, _| Ok(2.0),
            range: (-5.0, 5.0),
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
            tol_integral: 1e-14,
        },
        TestFunction {
            name: "2: f(x) = 1/2 - 1/(1 + 16 x²)", // (shifted) Runge equation
            f: |x, _| Ok(1.0 / 2.0 - 1.0 / (1.0 + 16.0 * x * x)),
            g: |x, _| Ok((32.0 * x) / f64::powi(1.0 + 16.0 * f64::powi(x, 2), 2)),
            h: |x, _| {
                Ok((-2048.0 * f64::powi(x, 2)) / f64::powi(1.0 + 16.0 * f64::powi(x, 2), 3)
                    + 32.0 / f64::powi(1.0 + 16.0 * f64::powi(x, 2), 2))
            },
            range: (-2.0, 2.0),
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
            tol_integral: 1e-13,
        },
        TestFunction {
            name: "3: f(x) = x⁵ + 3x⁴ - 2x³ + x - 1",
            f: |x, _| Ok(f64::powi(x, 5) + 3.0 * f64::powi(x, 4) - 2.0 * f64::powi(x, 3) + x - 1.0),
            g: |x, _| Ok(1.0 - 6.0 * f64::powi(x, 2) + 12.0 * f64::powi(x, 3) + 5.0 * f64::powi(x, 4)),
            h: |x, _| Ok(-12.0 * x + 36.0 * f64::powi(x, 2) + 20.0 * f64::powi(x, 3)),
            range: (-3.6, 2.0),
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
            name: "4: f(x) = (x - 1)² + 5 sin(x)",
            f: |x, _| Ok(f64::powi(x - 1.0, 2) + 5.0 * f64::sin(x)),
            g: |x, _| Ok(2.0 * (-1.0 + x) + 5.0 * f64::cos(x)),
            h: |x, _| Ok(2.0 - 5.0 * f64::sin(x)),
            range: (-2.8, 5.0),
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
            tol_integral: 1e-14,
        },
        TestFunction {
            name: "5: f(x) = 1/(1 - exp(-2 x) sin²(5 π x)) - 3/2",
            f: |x, _| Ok(1.0 / (1.0 - f64::exp(-2.0 * x) * f64::powi(f64::sin(5.0 * PI * x), 2)) - 1.5),
            g: |x, _| {
                let s5px = f64::sin(5.0 * PI * x);
                let s5px2 = f64::powi(s5px, 2);
                let c5px = f64::cos(5.0 * PI * x);
                let e2x = f64::exp(2.0 * x);
                Ok(-(((-10.0 * PI * c5px * s5px) / e2x + (2.0 * s5px2) / e2x) / f64::powi(1.0 - s5px2 / e2x, 2)))
            },
            h: |x, _| {
                let s5px = f64::sin(5.0 * PI * x);
                let s5px2 = f64::powi(s5px, 2);
                let c5px = f64::cos(5.0 * PI * x);
                let e2x = f64::exp(2.0 * x);
                let pi2 = f64::powi(PI, 2);
                Ok(
                    (2.0 * f64::powi((-10.0 * PI * c5px * s5px) / e2x + (2.0 * s5px2) / e2x, 2))
                        / f64::powi(1.0 - s5px2 / e2x, 3)
                        - ((-50.0 * pi2 * f64::powi(c5px, 2)) / e2x + (40.0 * PI * c5px * s5px) / e2x
                            - (4.0 * s5px2) / e2x
                            + (50.0 * pi2 * s5px2) / e2x)
                            / f64::powi(1.0 - s5px2 / e2x, 2),
                )
            },
            range: (0.0, 1.0),
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
            name: "6: f(x) = sin(x) in [0, π]",
            f: |x, _| Ok(f64::sin(x)),
            g: |x, _| Ok(f64::cos(x)),
            h: |x, _| Ok(-f64::sin(x)),
            range: (0.0, PI),
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
            name: "7: f(x) = sin(x) in [0, π/2]",
            f: |x, _| Ok(f64::sin(x)),
            g: |x, _| Ok(f64::cos(x)),
            h: |x, _| Ok(-f64::sin(x)),
            range: (0.0, PI / 2.0),
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
            name: "8: f(x) = sin(x) in [-1, 1]",
            f: |x, _| Ok(f64::sin(x)),
            g: |x, _| Ok(f64::cos(x)),
            h: |x, _| Ok(-f64::sin(x)),
            range: (-1.0, 1.0),
            min1: None,
            min2: None,
            min3: None,
            root1: Some(Bracket {
                a: -1.0,
                b: 1.0,
                fa: -0.841470984807896506652502321630,
                fb: 0.841470984807896506652502321630,
                xo: 0.0,
                fxo: 0.0,
            }),
            root2: None,
            root3: None,
            integral: Some((-1.0, 1.0, 0.0)),
            tol_min: 0.0,
            tol_root: 0.0,
            tol_integral: 1e-20,
        },
        TestFunction {
            name: "9: f(x) = 0.092834 sin(77.0001 + 19.87 x) in [-2.34567, 12.34567]",
            f: |x, _| Ok(0.092834 * f64::sin(77.0001 + 19.87 * x)),
            g: |x, _| Ok(1.84461158 * f64::cos(77.0001 + 19.87 * x)),
            h: |x, _| Ok(-36.6524320946 * f64::sin(77.0001 + 19.87 * x)),
            range: (-2.34567, 12.34567),
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
            name: "10: f(x) = 0.092834 sin[7.0001 + 1.87 x) in [-2.34567, 1.34567]",
            f: |x, _| Ok(0.092834 * f64::sin(7.0001 + 1.87 * x)),
            g: |x, _| Ok(0.17359958 * f64::cos(7.0001 + 1.87 * x)),
            h: |x, _| Ok(-0.32463121460000005 * f64::sin(7.0001 + 1.87 * x)),
            range: (-2.5, 1.5),
            min1: Some(Bracket {
                a: -2.0,
                b: 1.0,
                fa: -0.010975778218986671,
                fb: 0.04889284300988581,
                xo: -1.22337487145893900992800289781,
                fxo: -0.0928339999999999998525623823298,
            }),
            min2: None,
            min3: None,
            root1: Some(Bracket {
                a: -2.5,
                b: -0.5,
                fa: 0.06765264302507541,
                fb: -0.020085627400799486,
                xo: -2.06337291251882714520714257579,
                fxo: 0.0,
            }),
            root2: Some(Bracket {
                a: -2.0,
                b: 1.0,
                fa: -0.010975778218986671,
                fb: 0.04889284300988581,
                xo: -0.383376841080435039077386755851,
                fxo: 0.0,
            }),
            root3: Some(Bracket {
                a: 0.0,
                b: 1.5,
                fa: 0.060997692376682885,
                fb: -0.03446179260577237,
                xo: 1.29661923035795706705236906408,
                fxo: 0.0,
            }),
            integral: Some((-2.34567, 1.34567, 0.00654937363510264)),
            tol_min: 1e-8,
            tol_root: 1e-15,
            tol_integral: 1e-16,
        },
        TestFunction {
            name: "11: f(x) = (2 x⁵ - x + 3)/x²",
            f: |x, _| Ok((2.0 * f64::powi(x, 5) - x + 3.0) / (x * x)),
            g: |x, _| {
                Ok((-1.0 + 10.0 * f64::powi(x, 4)) / f64::powi(x, 2)
                    - (2.0 * (3.0 - x + 2.0 * f64::powi(x, 5))) / f64::powi(x, 3))
            },
            h: |x, _| {
                Ok(40.0 * x - (4.0 * (-1.0 + 10.0 * f64::powi(x, 4))) / f64::powi(x, 3)
                    + (6.0 * (3.0 - x + 2.0 * f64::powi(x, 5))) / f64::powi(x, 4))
            },
            range: (1.0, 2.0),
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
            name: "12: f(x) = 3/exp(-x) - 1/(3x)",
            f: |x, _| Ok(3.0 / f64::exp(-x) - 1.0 / (3.0 * x)),
            g: |x, _| Ok(3.0 * f64::exp(x) + 1.0 / (3.0 * f64::powi(x, 2))),
            h: |x, _| Ok(3.0 * f64::exp(x) - 2.0 / (3.0 * f64::powi(x, 3))),
            range: (-20.0, -1.0),
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
            name: "13: f(x) = log(2 f64::cos(x/2))",
            f: |x, _| Ok(f64::ln(2.0 * f64::cos(x / 2.0))),
            g: |x, _| Ok(-0.5 * f64::tan(x / 2.0)),
            h: |x, _| Ok(-0.25 * f64::powi(1.0 / f64::cos(x / 2.0), 2)),
            range: (-PI, PI),
            min1: None,
            min2: None,
            min3: None,
            root1: Some(Bracket {
                a: -3.0,
                b: 1.0,
                fa: -1.9556364734184897,
                fb: 0.5625629401162227,
                xo: -2.09439510239319549230842892219,
                fxo: 0.0,
            }),
            root2: Some(Bracket {
                a: -1.0,
                b: 3.0,
                fa: 0.5625629401162227,
                fb: -1.9556364734184897,
                xo: 2.09439510239319549230842892219,
                fxo: 0.0,
            }),
            root3: None,
            integral: Some((-PI, PI, 0.0)),
            tol_min: 0.0,
            tol_root: 1e-13,
            tol_integral: 1e-10,
        },
        TestFunction {
            name: "14: f(x) = exp(x)",
            f: |x, _| Ok(f64::exp(x)),
            g: |x, _| Ok(f64::exp(x)),
            h: |x, _| Ok(f64::exp(x)),
            range: (0.0, 10.1),
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
    use super::get_test_functions;
    use crate::algo::Bracket;
    use crate::approx_eq;
    // use crate::Vector;
    // use plotpy::{Curve, Legend, Plot, RayEndpoint};

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
        for (i, func) in get_test_functions().iter().enumerate() {
            println!("\n{}", func.name);
            if i == 0 {
                assert_eq!((func.f)(0.0, args).err(), Some("stop"));
            }
            assert_eq!(format!("{}", i), func.name.split(":").next().unwrap()); // make sure index is correct
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

            // plot (do not delete the code below---to generate figures)
            /*
            if i > 0 {
                let mut curve_origin = Curve::new();
                let mut curve_f = Curve::new();
                let mut curve_min1 = Curve::new();
                let mut curve_min2 = Curve::new();
                let mut curve_min3 = Curve::new();
                let mut curve_root1 = Curve::new();
                let mut curve_root2 = Curve::new();
                let mut curve_root3 = Curve::new();
                curve_origin.set_line_color("#5c5c5c");
                curve_f.set_label("f(x)");
                curve_min1
                    .set_label("min1")
                    .set_line_style("None")
                    .set_marker_style("*")
                    .set_marker_line_color("red")
                    .set_marker_color("red");
                curve_min2
                    .set_label("min2")
                    .set_line_style("None")
                    .set_marker_style("*")
                    .set_marker_line_color("green")
                    .set_marker_color("green");
                curve_min3
                    .set_label("min3")
                    .set_line_style("None")
                    .set_marker_style("*")
                    .set_marker_line_color("blue")
                    .set_marker_color("blue");
                curve_root1
                    .set_label("root1")
                    .set_line_style("None")
                    .set_marker_style("o")
                    .set_marker_line_color("red")
                    .set_marker_void(true);
                curve_root2
                    .set_label("root2")
                    .set_line_style("None")
                    .set_marker_style("o")
                    .set_marker_line_color("green")
                    .set_marker_void(true);
                curve_root3
                    .set_label("root3")
                    .set_line_style("None")
                    .set_marker_style("o")
                    .set_marker_line_color("blue")
                    .set_marker_void(true);
                let npoint = if i == 9 || i == 13 { 1001 } else { 401 };
                let xx = Vector::linspace(func.range.0, func.range.1, npoint).unwrap();
                let yy = xx.get_mapped(|x| (func.f)(x, args).unwrap());
                curve_origin.draw_ray(0.0, 0.0, RayEndpoint::Horizontal);
                curve_origin.draw_ray(0.0, 0.0, RayEndpoint::Vertical);
                curve_f.draw(xx.as_data(), yy.as_data());
                let mut plot = Plot::new();
                plot.add(&curve_origin).add(&curve_f);
                if let Some(bracket) = &func.min1 {
                    curve_min1.draw(&[bracket.xo], &[bracket.fxo]);
                    plot.add(&curve_min1);
                }
                if let Some(bracket) = &func.min2 {
                    curve_min2.draw(&[bracket.xo], &[bracket.fxo]);
                    plot.add(&curve_min2);
                }
                if let Some(bracket) = &func.min3 {
                    curve_min3.draw(&[bracket.xo], &[bracket.fxo]);
                    plot.add(&curve_min3);
                }
                if let Some(bracket) = &func.root1 {
                    curve_root1.draw(&[bracket.xo], &[bracket.fxo]);
                    plot.add(&curve_root1);
                }
                if let Some(bracket) = &func.root2 {
                    curve_root2.draw(&[bracket.xo], &[bracket.fxo]);
                    plot.add(&curve_root2);
                }
                if let Some(bracket) = &func.root3 {
                    curve_root3.draw(&[bracket.xo], &[bracket.fxo]);
                    plot.add(&curve_root3);
                }
                let mut legend = Legend::new();
                if i == 13 {
                    legend.set_location("center");
                }
                legend.draw();
                let path = format!("/tmp/russell_lab/test_function_{:0>3}.svg", i);
                plot.set_title(&func.name)
                    .add(&legend)
                    .grid_and_labels("$x$", "$f(x)$")
                    .set_figure_size_points(600.0, 350.0)
                    .save(path.as_str())
                    .unwrap();
            }
            */
        }
        println!();
    }
}
