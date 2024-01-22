#![allow(unused)]

use crate::StrError;
use crate::{ExplicitRungeKutta, Func, JacF, Method, OdeOutput, OdeParams, OdeSolverTrait};
use russell_lab::Vector;
use russell_sparse::CooMatrix;

/// Defines the solver for systems of ODEs
///
/// Solves:
///
/// ```text
/// d{y}
/// ———— = f(x, {y})
///  dx
/// where x is a scalar and {y} is a vector
/// ```
struct OdeSolver<'a> {
    params: &'a OdeParams,
    // actual: Box<dyn OdeSolverTrait<A> + 'a>,
}

impl<'a> OdeSolver<'a> {
    pub fn new(
        params: &'a OdeParams,
        ndim: usize,
        // function: Func<A>,
        // jacobian: Option<JacF<A>>,
        mass: Option<&'a CooMatrix>,
    ) -> Result<Self, StrError> {
        // #[rustfmt::skip]
        // let actual: Box<dyn OdeSolverTrait<A>+'a> = match params.method {
        //     Method::Radau5     => panic!("<not available>"),
        //     Method::BwEuler    => panic!("<not available>"),
        //     Method::FwEuler    => panic!("<not available>"),
        //     Method::Rk2        => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::Rk3        => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::Heun3      => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::Rk4        => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::Rk4alt     => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::MdEuler    => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::Merson4    => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::Zonneveld4 => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::Fehlberg4  => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::DoPri5     => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::Verner6    => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::Fehlberg7  => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        //     Method::DoPri8     => Box::new(ExplicitRungeKutta::new(params, ndim, function)?),
        // };
        Ok(OdeSolver { params })
    }

    /// Solves the ODE system
    ///
    /// ```text
    /// d{y}
    /// ———— = f(x, {y})
    ///  dx
    /// where x is a scalar and {y} is a vector
    /// ```
    ///
    /// # Input
    ///
    /// * `y` -- the {y} vector (dependent variable)
    /// * `x` -- the independent variable
    /// * `h_approx` -- the stepsize (`h = dx`; `x_new = x_old + h`). Set this value to select a fixed-steps approach.
    ///   This function will employ an automatic sub-stepping approach is `h_approx = None` and the method is embedded.
    ///   If `h_approx = None` and the method is not embedded, the stepsize will be computed for 10 steps.
    pub fn solve(&mut self, y: &Vector, x: f64, xf: f64, h_approx: Option<f64>) -> Result<(), StrError> {
        // check
        if xf < x {
            return Err("xf must be greater than x");
        }

        // information
        let info = self.params.method.information();

        // fixed steps?
        let mut fixed_steps = false;
        let mut h_approximate = 0.0;
        let mut fixed_nstep = 0;
        let mut fixed_h = 0.0;
        if let Some(h) = h_approx {
            fixed_steps = true;
            h_approximate = h;
        };
        if !info.embedded && !fixed_steps {
            const N_STEPS: usize = 10;
            fixed_steps = true;
            h_approximate = (xf - xf) / (N_STEPS as f64);
        }
        if fixed_steps {
            fixed_nstep = f64::ceil(xf / h_approximate) as usize;
            fixed_h = xf / (fixed_nstep as f64);
            let x_final = (fixed_nstep as f64) * fixed_h;
            if f64::abs(x_final - xf) > 1e-14 {
                println!("INTERNAL ERROR: x_final - xf = {:?} > 1e-14", x_final - xf);
                panic!("INTERNAL ERROR: x_final should be equal to xf");
            }
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_works() {}
}
