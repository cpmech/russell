use crate::OdeSystem;
use crate::StrError;
use russell_lab::Vector;
use russell_sparse::CooMatrix;

pub struct OdeSample<'a> {
    pub system: OdeSystem<
        'a,
        fn(&mut Vector, f64, &Vector) -> Result<(), StrError>,
        fn(&mut CooMatrix, f64, &Vector, f64) -> Result<(), StrError>,
    >,
    pub h_equal: Option<f64>,
    pub x0: f64,
    pub y0: Vector,
    pub x1: f64,
    pub y_analytical: Option<Box<dyn 'a + FnMut(&mut Vector, f64)>>,
}

pub struct Samples {}

impl Samples {
    pub fn hairer_wanner_eq1<'a>() -> OdeSample<'a> {
        const L: f64 = -50.0; // lambda
        OdeSample {
            system: OdeSystem::new(
                1,
                |f: &mut Vector, x: f64, y: &Vector| {
                    f[0] = L * y[0] - L * f64::cos(x);
                    Ok(())
                },
                |jac: &mut CooMatrix, _x: f64, _y: &Vector, multiplier: f64| {
                    jac.reset();
                    jac.put(0, 0, multiplier * L)?;
                    Ok(())
                },
                false,
                None,
                None,
            ),
            h_equal: Some(1.875 / 50.0),
            x0: 0.0,
            y0: Vector::from(&[0.0]),
            x1: 1.5,
            y_analytical: Some(Box::new(|y, x| {
                y[0] = -L * (f64::sin(x) - L * f64::cos(x) + L * f64::exp(L * x)) / (L * L + 1.0);
            })),
        }
    }

    pub fn van_der_pol<'a>(epsilon: Option<f64>, stationary: bool) -> OdeSample<'a> {
        /*
        let mut eps = match epsilon {
            Some(e) => e,
            None => 1.0e-6,
        };
        */
        let x0 = 0.0;
        let mut y0 = Vector::from(&[2.0, -0.6]);
        let mut x1 = 2.0;
        if stationary {
            // eps = 1.0;
            const A: f64 = 2.00861986087484313650940188;
            const T: f64 = 6.6632868593231301896996820305;
            y0[0] = A;
            y0[1] = 0.0;
            x1 = T;
        }
        OdeSample {
            system: OdeSystem::new(
                2,
                |f: &mut Vector, _x: f64, y: &Vector| {
                    let eps = 1.0e-6;
                    f[0] = y[1];
                    f[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / eps;
                    Ok(())
                },
                |jac: &mut CooMatrix, _x: f64, y: &Vector, multiplier: f64| {
                    let eps = 1.0e-6;
                    jac.reset();
                    jac.put(0, 1, 1.0 * multiplier)?;
                    jac.put(1, 0, multiplier * (-2.0 * y[0] * y[1] - 1.0) / eps)?;
                    jac.put(1, 1, multiplier * (1.0 - y[0] * y[0]) / eps)?;
                    Ok(())
                },
                false,
                Some(3),
                None,
            ),
            h_equal: None,
            x0,
            y0,
            x1,
            y_analytical: None,
        }
    }
}
