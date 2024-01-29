use crate::OdeSystem;
use crate::StrError;
use russell_lab::Vector;
use russell_sparse::CooMatrix;

pub struct SampleSimData<'a> {
    pub x0: f64,
    pub y0: Vector,
    pub x1: f64,
    pub h_equal: Option<f64>,
    pub y_analytical: Option<Box<dyn 'a + FnMut(&mut Vector, f64)>>,
}

pub type NoArgs = u8;

pub struct Samples {}

impl Samples {
    /// Returns the Hairer-Wanner problem from VII-p2 Eq.(1.1)
    pub fn hairer_wanner_eq1<'a>() -> (
        OdeSystem<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleSimData<'a>,
    ) {
        const L: f64 = -50.0; // lambda
        let system = OdeSystem::new(
            1,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
                f[0] = L * y[0] - L * f64::cos(x);
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, _y: &Vector, multiplier: f64, _args: &mut NoArgs| {
                jj.reset();
                jj.put(0, 0, multiplier * L)?;
                Ok(())
            },
            false,
            None,
            None,
        );
        let data = SampleSimData {
            x0: 0.0,
            y0: Vector::from(&[0.0]),
            x1: 1.5,
            h_equal: Some(1.875 / 50.0),
            y_analytical: Some(Box::new(|y, x| {
                y[0] = -L * (f64::sin(x) - L * f64::cos(x) + L * f64::exp(L * x)) / (L * L + 1.0);
            })),
        };
        (system, data)
    }

    /// Returns the Van der Pol's equation as given in Hairer-Wanner VII-p5 Eq.(1.5)
    ///
    /// # Input
    ///
    /// * `epsilon` -- ε coefficient; use None for the default value (= 1.0e-6)
    /// * `stationary` -- use `ε = 1` and compute the period and amplitude such that
    ///   `y = [A, 0]` is a stationary point.
    pub fn van_der_pol<'a>(
        epsilon: Option<f64>,
        stationary: bool,
    ) -> (
        OdeSystem<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleSimData<'a>,
    ) {
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
        let system = OdeSystem::new(
            2,
            move |f: &mut Vector, _x: f64, y: &Vector, _args: &mut NoArgs| {
                f[0] = y[1];
                f[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / eps;
                Ok(())
            },
            move |jj: &mut CooMatrix, _x: f64, y: &Vector, multiplier: f64, _args: &mut NoArgs| {
                jj.reset();
                jj.put(0, 1, 1.0 * multiplier)?;
                jj.put(1, 0, multiplier * (-2.0 * y[0] * y[1] - 1.0) / eps)?;
                jj.put(1, 1, multiplier * (1.0 - y[0] * y[0]) / eps)?;
                Ok(())
            },
            false,
            Some(3),
            None,
        );
        let data = SampleSimData {
            x0,
            y0,
            x1,
            h_equal: None,
            y_analytical: None,
        };
        (system, data)
    }
}
