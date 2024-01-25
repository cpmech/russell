use crate::constants::N_EQUAL_STEPS;
use crate::StrError;
use crate::{EulerForward, ExplicitRungeKutta, Method, OdeParams, OdeSolverTrait, OdeSys, OdeSysJac};
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
pub struct OdeSolver<'a, A> {
    params: &'a OdeParams,
    actual: Box<dyn OdeSolverTrait<A> + 'a>,

    /// Scaling vector
    ///
    /// ```text
    /// scaling[i] = abs_tol + rel_tol ⋅ |x[i]|
    /// ```
    scaling: Vector,

    n_performed_steps: usize,
}

impl<'a, A: 'a> OdeSolver<'a, A> {
    pub fn new(
        params: &'a OdeParams,
        ndim: usize,
        function: OdeSys<A>,
        _jacobian: Option<OdeSysJac<A>>,
        _mass: Option<&'a CooMatrix>,
    ) -> Result<Self, StrError> {
        params.validate()?;
        let actual: Box<dyn OdeSolverTrait<A>> = if params.method == Method::Radau5 {
            panic!("TODO: Radau5");
        } else if params.method == Method::BwEuler {
            panic!("TODO: BwEuler");
        } else if params.method == Method::FwEuler {
            Box::new(EulerForward::new(ndim, function))
        } else {
            Box::new(ExplicitRungeKutta::new(params, ndim, function)?)
        };
        Ok(OdeSolver {
            params,
            actual,
            scaling: Vector::new(ndim),
            n_performed_steps: 0,
        })
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
    /// * `y0` -- the initial vector of dependent variables; it will be updated to `y1`
    /// * `x0` -- the initial independent variable
    /// * `x1` -- the final independent variable
    /// * `h_equal` -- a constant stepsize for solving with equal-steps; otherwise,
    ///   if possible, variable step sizes are automatically calculated. If automatic
    ///   sub-stepping is not possible (e.g., the RK method is not embedded),
    ///   a constant (and equal) stepsize will be calculated for [N_EQUAL_STEPS] steps.
    pub fn solve(
        &mut self,
        y0: &mut Vector,
        x0: f64,
        x1: f64,
        h_equal: Option<f64>,
        args: &mut A,
    ) -> Result<(), StrError> {
        // check
        if x1 < x0 {
            return Err("x1 must be greater than x0");
        }

        // information
        let info = self.params.method.information();

        // configure fixed steps if not embedded or if h_approx is given
        let mut fixed_steps = false;
        let mut h_approximate = 0.0;
        let mut fixed_nstep = 0;
        let mut fixed_h = 0.0;
        if let Some(h) = h_equal {
            if h < 0.0 {
                return Err("h_approx must be greater than zero");
            }
            fixed_steps = true;
            h_approximate = h;
        };
        if !info.embedded && !fixed_steps {
            fixed_steps = true;
            h_approximate = (x1 - x0) / (N_EQUAL_STEPS as f64);
        }
        if fixed_steps {
            fixed_nstep = f64::ceil(x1 / h_approximate) as usize;
            fixed_h = x1 / (fixed_nstep as f64);
            let x_final = (fixed_nstep as f64) * fixed_h;
            if f64::abs(x_final - x1) > 1e-14 {
                println!("INTERNAL ERROR: x_final - x1 = {:?} > 1e-14", x_final - x1);
                panic!("INTERNAL ERROR: x_final should be equal to x1");
            }
        }

        // initial stepsize
        let mut h = x1 - x0;
        if fixed_steps {
            h = fixed_h;
        } else {
            h = f64::min(h, self.params.initial_stepsize);
        }

        // x0 for sub-stepping (will become x1 at the end)
        let mut x0 = x0;

        // fixed steps loop
        if fixed_steps {
            const IGNORED: f64 = 0.0;
            for n in 0..fixed_nstep {
                // TODO: compute f0 for numerical Jacobian

                // perform step
                let first_step = n == 0;
                self.actual.step(x0, &y0, h, first_step, args)?;
                self.n_performed_steps += 1;

                // update x0
                x0 = ((n + 1) as f64) * h;

                // accept step
                self.actual.accept(y0, x0, h, IGNORED, IGNORED, args)?;
            }
            return Ok(());
        }

        // first scaling variables
        let ndim = self.scaling.dim();
        for i in 0..ndim {
            self.scaling[i] = self.params.abs_tol + self.params.rel_tol * f64::abs(y0[i]);
        }

        // variable steps
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::OdeSolver;
    use crate::StrError;
    use crate::{Method, OdeParams};
    use russell_lab::Vector;

    #[test]
    fn solve_works_1() {
        struct Args {}
        let params = OdeParams::new(Method::FwEuler, None, None);
        let function = |f: &mut Vector, _: f64, _: &Vector, _: &mut Args| -> Result<(), StrError> {
            f[0] = 1.0;
            Ok(())
        };
        let mut solver = OdeSolver::new(&params, 1, function, None, None).unwrap();
        let mut y0 = Vector::new(1);
        let mut args = Args {};
        solver.solve(&mut y0, 0.0, 1.0, None, &mut args).unwrap();
    }
}
