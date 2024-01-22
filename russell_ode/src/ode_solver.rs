use crate::StrError;
use crate::{ExplicitRungeKutta, Func, JacF, Method, OdeParams, OdeSolverTrait};
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
struct OdeSolver<'a, A> {
    params: &'a OdeParams,
    actual: Box<dyn OdeSolverTrait<A> + 'a>,
}

impl<'a, A: 'a> OdeSolver<'a, A> {
    pub fn new(
        params: &'a OdeParams,
        ndim: usize,
        function: Func<A>,
        jacobian: Option<JacF<A>>,
        mass: Option<&'a CooMatrix>,
    ) -> Result<Self, StrError> {
        if params.method == Method::Radau5 {
            panic!("TODO: Radau5");
        }
        if params.method == Method::BwEuler {
            panic!("TODO: BwEuler");
        }
        if params.method == Method::FwEuler {
            panic!("TODO: FwEuler");
        }
        let actual: Box<ExplicitRungeKutta<'a, A>> = Box::new(ExplicitRungeKutta::new(params, ndim, function)?);
        Ok(OdeSolver { params, actual })
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
    /// * `h_approx` -- the approximate stepsize for solving with fixed-steps (`h = dx`; `x_new = x_old + h`);
    ///   otherwise use the automatic sub-stepping approach, if available.
    ///
    /// **Note:** If the method is not embedded and `h_approx = None`, a fixed stepsize will be computed.
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
