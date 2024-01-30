use crate::StrError;
use crate::{HasJacobian, OdeSystem};
use russell_lab::Vector;
use russell_sparse::CooMatrix;

/// Holds the control data corresponding to a sample ODE problem
pub struct SampleControl<'a> {
    /// Holds the initial x
    pub x0: f64,

    /// Holds the initial y
    pub y0: Vector,

    /// Holds the final x
    pub x1: f64,

    /// Holds the stepsize for simulations with equal-steps
    pub h_equal: Option<f64>,

    /// Holds the analytical solution `y(x)`
    pub y_analytical: Option<Box<dyn 'a + FnMut(&mut Vector, f64)>>,
}

/// Indicates that the sample ODE problem does not have extra arguments
pub type SampleNoArgs = u8;

/// Holds a collection of sample ODE problems
///
/// # References
///
/// 1. E. Hairer, S. P. Nørsett, G. Wanner (2008) Solving Ordinary Differential Equations I.
///    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
///    in Computational Mathematics, 528p
/// 2. E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
///    Stiff and Differential-Algebraic Problems. Second Revised Edition.
///    Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
pub struct Samples {}

impl Samples {
    /// Returns the Hairer-Wanner problem from the reference, Eq(1.1), page 2
    ///
    /// # Reference
    ///
    /// * E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn hairer_wanner_eq1<'a>() -> (
        OdeSystem<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut SampleNoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut SampleNoArgs) -> Result<(), StrError>,
            SampleNoArgs,
        >,
        SampleControl<'a>,
        SampleNoArgs,
    ) {
        const L: f64 = -50.0; // lambda
        let system = OdeSystem::new(
            1,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut SampleNoArgs| {
                f[0] = L * y[0] - L * f64::cos(x);
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, _y: &Vector, multiplier: f64, _args: &mut SampleNoArgs| {
                jj.reset();
                jj.put(0, 0, multiplier * L)?;
                Ok(())
            },
            HasJacobian::Yes,
            None,
            None,
        );
        let control = SampleControl {
            x0: 0.0,
            y0: Vector::from(&[0.0]),
            x1: 1.5,
            h_equal: Some(1.875 / 50.0),
            y_analytical: Some(Box::new(|y, x| {
                y[0] = -L * (f64::sin(x) - L * f64::cos(x) + L * f64::exp(L * x)) / (L * L + 1.0);
            })),
        };
        (system, control, 0)
    }

    /// Returns the Van der Pol's equation as given in Hairer-Wanner, Eq(1.5'), page 5
    ///
    /// Using data from Eq(7.29), page 113
    ///
    /// # Input
    ///
    /// * `epsilon` -- ε coefficient; use None for the default value (= 1.0e-6)
    /// * `stationary` -- use `ε = 1` and compute the period and amplitude such that
    ///   `y = [A, 0]` is a stationary point.
    ///
    /// # Reference
    ///
    /// * E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn van_der_pol<'a>(
        epsilon: Option<f64>,
        stationary: bool,
    ) -> (
        OdeSystem<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut SampleNoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut SampleNoArgs) -> Result<(), StrError>,
            SampleNoArgs,
        >,
        SampleControl<'a>,
        SampleNoArgs,
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
            move |f: &mut Vector, _x: f64, y: &Vector, _args: &mut SampleNoArgs| {
                f[0] = y[1];
                f[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / eps;
                Ok(())
            },
            move |jj: &mut CooMatrix, _x: f64, y: &Vector, multiplier: f64, _args: &mut SampleNoArgs| {
                jj.reset();
                jj.put(0, 1, 1.0 * multiplier)?;
                jj.put(1, 0, multiplier * (-2.0 * y[0] * y[1] - 1.0) / eps)?;
                jj.put(1, 1, multiplier * (1.0 - y[0] * y[0]) / eps)?;
                Ok(())
            },
            HasJacobian::Yes,
            Some(3),
            None,
        );
        let control = SampleControl {
            x0,
            y0,
            x1,
            h_equal: None,
            y_analytical: None,
        };
        (system, control, 0)
    }
}
