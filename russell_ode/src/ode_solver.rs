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
    /// Holds the parameters
    params: &'a OdeParams,

    /// Dimension of the ODE system
    ndim: usize,

    /// Holds a pointer to the actual ODE system solver
    actual: Box<dyn OdeSolverTrait<A> + 'a>,

    /// Scaling vector
    ///
    /// ```text
    /// scaling[i] = abs_tol + rel_tol ⋅ |x[i]|
    /// ```
    scaling: Vector,

    /// Collects the number of steps, successful or not
    n_performed_steps: usize,

    /// Collects the number of rejected steps
    n_rejected_steps: usize,
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
            ndim,
            actual,
            scaling: Vector::new(ndim),
            n_performed_steps: 0,
            n_rejected_steps: 0,
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
        // check data
        if y0.dim() != self.ndim {
            return Err("y0.dim() must be equal to ndim");
        }
        if x1 < x0 {
            return Err("x1 must be greater than x0");
        }

        // information
        let info = self.params.method.information();

        // initial stepsize
        let (equal_stepping, h) = match h_equal {
            Some(h_eq) => {
                if h_eq < 0.0 {
                    return Err("h_equal must be greater than zero");
                }
                let n = f64::ceil((x1 - x0) / h_eq) as usize;
                let h = (x1 - x0) / (n as f64);
                (true, h)
            }
            None => {
                if info.embedded {
                    let h = f64::min(self.params.initial_stepsize, x1 - x0);
                    (false, h)
                } else {
                    let h = (x1 - x0) / (N_EQUAL_STEPS as f64);
                    (true, h)
                }
            }
        };

        // mutable x0 (will become x1 at the end)
        let mut x0 = x0;

        // restart variables
        self.actual.initialize();
        self.n_performed_steps = 0;
        self.n_rejected_steps = 0;

        // equal-stepping loop
        if equal_stepping {
            const IGNORED: f64 = 0.0;
            while x0 < x1 {
                // step
                self.actual.step(x0, &y0, h, args)?;
                self.n_performed_steps += 1;

                // update x0
                x0 += h;

                // update y0
                self.actual.accept(y0, x0, h, IGNORED, IGNORED, args)?;
            }
            return Ok(());
        }

        // first scaling variables
        for i in 0..self.ndim {
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
