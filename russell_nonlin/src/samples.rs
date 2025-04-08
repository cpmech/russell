use super::{NoArgs, System};
use russell_lab::math::SQRT_2_BY_3;
use russell_lab::Vector;
use russell_sparse::{CooMatrix, Sym};

/// Holds a collection of nonlinear problems
pub struct Samples {}

impl Samples {
    /// Cubic polynomial (causing problems to Newton's method)
    ///
    /// ```text
    /// f = x³ - 2 x - 2
    /// J = 3 x² - 2
    /// ```
    ///
    /// Returns `(system, u_trial_ok, u_trial_oscillation, u_trial_indeterminate, u_ok, args)`
    pub fn cubic_poly_1<'a>() -> (System<'a, NoArgs>, Vector, Vector, Vector, Vector, NoArgs) {
        // system
        let ndim = 1;
        let mut system = System::new(ndim, |gg: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
            gg[0] = u[0] * u[0] * u[0] - 2.0 * u[0] - 2.0;
            Ok(())
        })
        .unwrap();

        // function to compute Gu
        let nnz = 1;
        system
            .set_calc_ggu(
                Some(nnz),
                Sym::No,
                |ggu: &mut CooMatrix, _l: f64, u: &Vector, _args: &mut NoArgs| {
                    ggu.reset();
                    ggu.put(0, 0, 3.0 * u[0] * u[0] - 2.0).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // trial u vector for Newton's method
        let u_trial_ok = Vector::from(&[1.0]);
        let u_trial_oscillation = Vector::from(&[0.0]);
        let u_trial_indeterminate = Vector::from(&[SQRT_2_BY_3]);

        // reference solution
        let u_reference = Vector::from(&[1.76929235423863]);

        // done
        let args = 0;
        (
            system,
            u_trial_ok,
            u_trial_oscillation,
            u_trial_indeterminate,
            u_reference,
            args,
        )
    }

    /// Cubic polynomial (causing divergence problems to Newton's method)
    ///
    /// ```text
    /// f = (x - 1)³ + 0.512
    /// J = 3 (x - 1)²
    /// ```
    ///
    /// Returns `(system, u_trial, u_reference,  args)`
    pub fn cubic_poly_2<'a>() -> (System<'a, NoArgs>, Vector, Vector, NoArgs) {
        // system
        let ndim = 1;
        let mut system = System::new(ndim, |gg: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
            gg[0] = f64::powi(u[0] - 1.0, 3) + 0.512;
            Ok(())
        })
        .unwrap();

        // function to compute Gu
        let nnz = 1;
        system
            .set_calc_ggu(
                Some(nnz),
                Sym::No,
                |ggu: &mut CooMatrix, _l: f64, u: &Vector, _args: &mut NoArgs| {
                    ggu.reset();
                    ggu.put(0, 0, 3.0 * f64::powi(u[0] - 1.0, 2)).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // trial u vector for Newton's method
        let u_trial = Vector::from(&[5.0]);

        // reference solution
        let u_reference = Vector::from(&[0.2]);

        // done
        let args = 0;
        (system, u_trial, u_reference, args)
    }

    /// Simple two-equation system
    ///
    /// Returns `(system, u_trial, u_reference, args)`
    pub fn simple_two_equations<'a>() -> (System<'a, NoArgs>, Vector, Vector, NoArgs) {
        // system
        let ndim = 2;
        let mut system = System::new(ndim, |gg: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
            gg[0] = u[0].powf(3.0) + u[1] - 1.0;
            gg[1] = -u[0] + u[1].powf(3.0) + 1.0;
            Ok(())
        })
        .unwrap();

        // function to compute Gu
        let nnz = 4;
        system
            .set_calc_ggu(
                Some(nnz),
                Sym::No,
                |ggu: &mut CooMatrix, _l: f64, u: &Vector, _args: &mut NoArgs| {
                    ggu.reset();
                    ggu.put(0, 0, 3.0 * u[0] * u[0]).unwrap();
                    ggu.put(0, 1, 1.0).unwrap();
                    ggu.put(1, 0, -1.0).unwrap();
                    ggu.put(1, 1, 3.0 * u[1] * u[1]).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // trial u vector for Newton's method
        let u_trial = Vector::from(&[0.5, 0.5]);

        // reference solution
        let u_reference = Vector::from(&[1.0, 0.0]);

        // done
        let args = 0;
        (system, u_trial, u_reference, args)
    }
}
