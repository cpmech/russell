use crate::math::PI;
use crate::StrError;

/// Indicates that no arguments are needed
pub(super) type NoArgs = u8;

/// Holds f(x) functions for tests
#[allow(unused)]
pub(super) struct TestFunction {
    pub name: &'static str,                               // name
    pub f: fn(f64, &mut NoArgs) -> Result<f64, StrError>, // f(x)
    pub g: fn(f64, &mut NoArgs) -> Result<f64, StrError>, // g=df/dx
    pub h: fn(f64, &mut NoArgs) -> Result<f64, StrError>, // h=d²f/dx²
    pub at_x: f64,                                        // @x value
    pub tol_g: f64,                                       // tolerance for |num - ana|
    pub tol_g_err: f64,                                   // tolerance for truncation error
    pub tol_g_rerr: f64,                                  // tolerance for rounding error
    pub improv_tol_g_diff: f64,                           // tolerance for |num - ana|
    pub tol_h: f64,                                       // tolerance for |num - ana|
}

/// Allocates f(x) test functions
#[allow(dead_code)]
pub(super) fn get_functions() -> Vec<TestFunction> {
    vec![
        TestFunction {
            name: "x",
            f: |x, _| Ok(x),
            g: |_, _| Ok(1.0),
            h: |_, _| Ok(0.0),
            at_x: 0.0,
            tol_g: 1e-15,
            tol_g_err: 1e-15,
            tol_g_rerr: 1e-15,
            improv_tol_g_diff: 1e-15,
            tol_h: 1e-13,
        },
        TestFunction {
            name: "x²",
            f: |x, _| Ok(x * x),
            g: |x, _| Ok(2.0 * x),
            h: |_, _| Ok(2.0),
            at_x: 1.0,
            tol_g: 1e-12,
            tol_g_err: 1e-13,
            tol_g_rerr: 1e-11,
            improv_tol_g_diff: 1e-12,
            tol_h: 1e-9,
        },
        TestFunction {
            name: "exp(x)",
            f: |x, _| Ok(f64::exp(x)),
            g: |x, _| Ok(f64::exp(x)),
            h: |x, _| Ok(f64::exp(x)),
            at_x: 2.0,
            tol_g: 1e-11,
            tol_g_err: 1e-5,
            tol_g_rerr: 1e-10,
            improv_tol_g_diff: 1e-10, // worse
            tol_h: 1e-8,
        },
        TestFunction {
            name: "exp(-x²)",
            f: |x, _| Ok(f64::exp(-x * x)),
            g: |x, _| Ok(-2.0 * x * f64::exp(-x * x)),
            h: |x, _| Ok(-2.0 * f64::exp(-x * x) + 4.0 * x * x * f64::exp(-x * x)),
            at_x: 2.0,
            tol_g: 1e-13,
            tol_g_err: 1e-6,
            tol_g_rerr: 1e-13,
            improv_tol_g_diff: 1e-11, // worse
            tol_h: 1e-11,
        },
        TestFunction {
            name: "1/x",
            f: |x, _| Ok(1.0 / x),
            g: |x, _| Ok(-1.0 / (x * x)),
            h: |x, _| Ok(2.0 / (x * x * x)),
            at_x: 0.2,
            tol_g: 1e-8,
            tol_g_err: 1e-3,
            tol_g_rerr: 1e-11,
            improv_tol_g_diff: 1e-9, // better
            tol_h: 1e-8,
        },
        TestFunction {
            name: "x⋅√x",
            f: |x, _| Ok(x * f64::sqrt(x)),
            g: |x, _| Ok(1.5 * f64::sqrt(x)),
            h: |x, _| Ok(0.75 / f64::sqrt(x)),
            at_x: 25.0,
            tol_g: 1e-10,
            tol_g_err: 1e-9,
            tol_g_rerr: 1e-9,
            improv_tol_g_diff: 1e-10,
            tol_h: 1e-7,
        },
        TestFunction {
            name: "sin(1/x)",
            f: |x, _| Ok(f64::sin(1.0 / x)),
            g: |x, _| Ok(-f64::cos(1.0 / x) / (x * x)),
            h: |x, _| Ok(2.0 * f64::cos(1.0 / x) / (x * x * x) - f64::sin(1.0 / x) / (x * x * x * x)),
            at_x: 0.5,
            tol_g: 1e-10,
            tol_g_err: 1e-4,
            tol_g_rerr: 1e-11,
            improv_tol_g_diff: 1e-10,
            tol_h: 1e-9,
        },
        TestFunction {
            name: "cos(π⋅x/2)",
            f: |x, _| Ok(f64::cos(PI * x / 2.0)),
            g: |x, _| Ok(-f64::sin(PI * x / 2.0) * PI / 2.0),
            h: |x, _| Ok(-f64::cos(PI * x / 2.0) * PI * PI / 4.0),
            at_x: 1.0,
            tol_g: 1e-12,
            tol_g_err: 1e-6,
            tol_g_rerr: 1e-12,
            improv_tol_g_diff: 1e-10, // worse
            tol_h: 1e-14,
        },
    ]
}
