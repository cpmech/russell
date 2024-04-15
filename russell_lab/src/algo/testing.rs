use super::{Bracket, NoArgs};
use crate::StrError;

/// Holds f(x) functions for tests
#[allow(unused)]
pub(super) struct TestFunction {
    /// Holds the name of the function
    pub name: &'static str,

    /// Holds the f(x) function
    pub f: fn(f64, &mut NoArgs) -> Result<f64, StrError>,

    /// Holds a bracketed local minimum
    pub min_1: Option<Bracket>,

    // Holds another bracketed local minimum
    pub min_2: Option<Bracket>,

    /// Holds a bracketed root
    pub root_1: Option<Bracket>,

    /// Holds another bracketed root
    pub root_2: Option<Bracket>,
}

/// Allocates f(x) test functions
#[allow(dead_code)]
pub(super) fn get_functions() -> Vec<TestFunction> {
    vec![
        TestFunction {
            name: "x² - 1",
            f: |x, _| Ok(x * x - 1.0),
            min_1: Some(Bracket {
                a: -5.0,
                x_target: 0.0,
                b: 5.0,
                fa: 24.0,
                fx_target: -1.0,
                fb: 24.0,
            }),
            min_2: None,
            root_1: None,
            root_2: None,
        },
        TestFunction {
            name: "-1 / (1 + 16 x²)", // Runge equation
            f: |x, _| Ok(-1.0 / (1.0 + 16.0 * x * x)),
            min_1: Some(Bracket {
                a: -2.0,
                x_target: 0.0,
                b: 2.0,
                fa: -1.0 / 65.0,
                fx_target: -1.0,
                fb: -1.0 / 65.0,
            }),
            min_2: None,
            root_1: None, // no roots possible
            root_2: None, // no roots possible
        },
        TestFunction {
            name: "x⁵ + 3x⁴ - 2x³ + x - 1",
            f: |x, _| Ok(f64::powi(x, 5) + 3.0 * f64::powi(x, 4) - 2.0 * f64::powi(x, 3) + x - 1.0),
            min_1: Some(Bracket {
                a: -2.0,
                x_target: -0.326434701525930898665902357162,
                b: 2.0,
                fa: 29.0,
                fx_target: -1.22650698564642753377955683652,
                fb: 65.0,
            }),
            min_2: None,
            root_1: None,
            root_2: None,
        },
        TestFunction {
            name: "(x - 1)² + 5 sin(x)",
            f: |x, _| Ok(f64::powi(x - 1.0, 2) + 5.0 * f64::sin(x)),
            min_1: Some(Bracket {
                a: -2.0,
                x_target: -0.779014930395140333216421108317,
                b: 2.0,
                fa: 4.45351286587159152301990067044,
                fx_target: -0.347999771320472050094111906591,
                fb: 5.54648713412840847698009932956,
            }),
            min_2: Some(Bracket {
                a: 2.0,
                x_target: 3.41029230994771356210845446934,
                b: 5.0,
                fa: 5.54648713412840847698009932956,
                fx_target: 4.48211912850661077326295235125,
                fb: 11.2053786266843076555342279692,
            }),
            root_1: None,
            root_2: None,
        },
    ]
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::get_functions;
    use crate::approx_eq;

    #[test]
    fn functions_are_consistent() {
        let args = &mut 0;
        for func in &get_functions() {
            println!("{}", func.name);
            if let Some(bracket) = &func.min_1 {
                assert!(bracket.b > bracket.x_target);
                assert!(bracket.x_target > bracket.a);
                assert_eq!(bracket.fa, (func.f)(bracket.a, args).unwrap());
                approx_eq(bracket.fx_target, (func.f)(bracket.x_target, args).unwrap(), 1e-15);
                assert_eq!(bracket.fb, (func.f)(bracket.b, args).unwrap());
            }
            if let Some(bracket) = &func.min_2 {
                assert!(bracket.b > bracket.x_target);
                assert!(bracket.x_target > bracket.a);
                assert_eq!(bracket.fa, (func.f)(bracket.a, args).unwrap());
                assert_eq!(bracket.fx_target, (func.f)(bracket.x_target, args).unwrap());
                assert_eq!(bracket.fb, (func.f)(bracket.b, args).unwrap());
            }
        }
    }
}
