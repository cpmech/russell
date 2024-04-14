use super::{BracketMin, NoArgs};
use crate::StrError;

/// Holds f(x) functions for tests
#[allow(unused)]
pub(super) struct TestFunction {
    pub name: &'static str,                               // name
    pub f: fn(f64, &mut NoArgs) -> Result<f64, StrError>, // f(x)
    pub local_min_1: BracketMin,                          // local min (in given range)
    pub local_min_2: Option<BracketMin>,                  // another local min (in given range)
}

/// Allocates f(x) test functions
#[allow(dead_code)]
pub(super) fn get_functions() -> Vec<TestFunction> {
    vec![
        TestFunction {
            name: "x² - 1",
            f: |x, _| Ok(x * x - 1.0),
            local_min_1: BracketMin {
                a: -5.0,
                b: 0.0,
                c: 5.0,
                fa: 24.0,
                fb: -1.0,
                fc: 24.0,
            },
            local_min_2: None,
        },
        TestFunction {
            name: "-1 / (1 + 16 x²)", // Runge equation
            f: |x, _| Ok(-1.0 / (1.0 + 16.0 * x * x)),
            local_min_1: BracketMin {
                a: -2.0,
                b: 0.0,
                c: 2.0,
                fa: -1.0 / 65.0,
                fb: -1.0,
                fc: -1.0 / 65.0,
            },
            local_min_2: None,
        },
        TestFunction {
            name: "x⁵ + 3x⁴ - 2x³ + x - 1",
            f: |x, _| Ok(f64::powi(x, 5) + 3.0 * f64::powi(x, 4) - 2.0 * f64::powi(x, 3) + x - 1.0),
            local_min_1: BracketMin {
                a: -2.0,
                b: -0.326434701525930898665902357162,
                c: 2.0,
                fa: 29.0,
                fb: -1.22650698564642753377955683652,
                fc: 65.0,
            },
            local_min_2: None,
        },
        TestFunction {
            name: "(x - 1)² + 5 sin(x)",
            f: |x, _| Ok(f64::powi(x - 1.0, 2) + 5.0 * f64::sin(x)),
            local_min_1: BracketMin {
                a: -2.0,
                b: -0.779014930395140333216421108317,
                c: 2.0,
                fa: 4.45351286587159152301990067044,
                fb: -0.347999771320472050094111906591,
                fc: 5.54648713412840847698009932956,
            },
            local_min_2: Some(BracketMin {
                a: 2.0,
                b: 3.41029230994771356210845446934,
                c: 5.0,
                fa: 5.54648713412840847698009932956,
                fb: 4.48211912850661077326295235125,
                fc: 11.2053786266843076555342279692,
            }),
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
            // println!("{}", func.name);
            let bracket = &func.local_min_1;
            assert!(bracket.c > bracket.b);
            assert!(bracket.b > bracket.a);
            assert_eq!(bracket.fa, (func.f)(bracket.a, args).unwrap());
            approx_eq(bracket.fb, (func.f)(bracket.b, args).unwrap(), 1e-15);
            assert_eq!(bracket.fc, (func.f)(bracket.c, args).unwrap());
            if let Some(bracket) = &func.local_min_2 {
                assert!(bracket.c > bracket.b);
                assert!(bracket.b > bracket.a);
                assert_eq!(bracket.fa, (func.f)(bracket.a, args).unwrap());
                assert_eq!(bracket.fb, (func.f)(bracket.b, args).unwrap());
                assert_eq!(bracket.fc, (func.f)(bracket.c, args).unwrap());
            }
        }
    }
}
