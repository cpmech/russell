use crate::StrError;
use russell_lab::Vector;
use russell_sparse::CooMatrix;

pub struct OdeSample {
    pub ndim: usize,
    pub system: Box<dyn FnMut(&mut Vector, f64, &Vector) -> Result<(), StrError>>,
    pub jacobian: Option<Box<dyn FnMut(&mut CooMatrix, f64, &Vector) -> Result<(), StrError>>>,
    pub analytical: Option<Box<dyn FnMut(&mut Vector, f64)>>,
    pub mass: Option<CooMatrix>,
    pub h_equal: Option<f64>,
    pub x0: f64,
    pub y0: Vector,
    pub x1: f64,
}

pub type OdeSampleArg = &'static mut u8;

pub struct Samples {}

impl Samples {
    pub fn hairer_wanner_eq1() -> OdeSample {
        const L: f64 = -50.0; // lambda
        OdeSample {
            ndim: 1,
            system: Box::new(|f, x, y| {
                f[0] = L * y[0] - L * f64::cos(x);
                Ok(())
            }),
            jacobian: None,
            analytical: Some(Box::new(|y, x| {
                y[0] = -L * (f64::sin(x) - L * f64::cos(x) + L * f64::exp(L * x)) / (L * L + 1.0);
            })),
            mass: None,
            h_equal: Some(1.875 / 50.0),
            x0: 0.0,
            y0: Vector::from(&[0.0]),
            x1: 1.5,
        }
    }

    pub fn van_der_pol(epsilon: Option<f64>, stationary: bool) -> OdeSample {
        let mut eps = match epsilon {
            Some(e) => e,
            None => 1.0e-6,
        };
        let x0 = 0.0;
        let mut y0 = Vector::from(&[2.0, -0.6]);
        let mut x1 = 2.0;
        if stationary {
            eps = 1.0;
            const A: f64 = 2.00861986087484313650940188;
            const T: f64 = 6.6632868593231301896996820305;
            y0[0] = A;
            y0[1] = 0.0;
            x1 = T;
        }
        let system = move |f: &mut Vector, _x: f64, y: &Vector| {
            f[0] = y[1];
            f[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / eps;
            Ok(())
        };
        let jacobian = move |jac: &mut CooMatrix, _x: f64, y: &Vector| {
            jac.reset();
            jac.put(0, 1, 1.0)?;
            jac.put(1, 0, (-2.0 * y[0] * y[1] - 1.0) / eps)?;
            jac.put(1, 1, (1.0 - y[0] * y[0]) / eps)?;
            Ok(())
        };
        OdeSample {
            ndim: 2,
            system: Box::new(system),
            jacobian: Some(Box::new(jacobian)),
            analytical: None,
            mass: None,
            h_equal: None,
            x0,
            y0,
            x1,
        }
    }
}
