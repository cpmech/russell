use crate::constants::N_EQUAL_STEPS;
use crate::StrError;
use crate::{EulerBackward, EulerForward, ExplicitRungeKutta, Method, NumSolver, OdeParams, OdeSystem};
use russell_lab::Vector;
use russell_sparse::CooMatrix;

/// Defines the solver for systems of ODEs
///
/// The system is defined by:
///
/// ```text
/// d{y}
/// ———— = {f}(x, {y})
///  dx
/// where x is a scalar and {y} and {f} are vectors
/// ```
///
/// The Jacobian is defined by:
///
/// ```text
///               ∂{f}
/// [J](x, {y}) = ————
///               ∂{y}
/// where [J] is the Jacobian matrix
/// ```
pub struct OdeSolver<'a, A> {
    /// Holds the parameters
    params: &'a OdeParams,

    /// Dimension of the ODE system
    ndim: usize,

    /// Holds a pointer to the actual ODE system solver
    actual: Box<dyn NumSolver<A> + 'a>,

    /// Collects the number of steps, successful or not
    n_performed_steps: usize,

    /// Collects the number of rejected steps
    n_rejected_steps: usize,
}

impl<'a, A> OdeSolver<'a, A> {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `params` -- holds all parameters, including the selection of the numerical [Method]
    /// * `system` -- defines the ODE system
    ///
    /// # Generics
    ///
    /// See [OdeSystem] for an explanation of the generic parameters.
    pub fn new<F, J>(params: &'a OdeParams, system: OdeSystem<'a, F, J, A>) -> Result<Self, StrError>
    where
        F: 'a + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
        J: 'a + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
        A: 'a,
    {
        params.validate()?;
        let ndim = system.ndim;
        let actual: Box<dyn NumSolver<A>> = if params.method == Method::Radau5 {
            panic!("TODO: Radau5");
        } else if params.method == Method::BwEuler {
            Box::new(EulerBackward::new(params, system))
        } else if params.method == Method::FwEuler {
            Box::new(EulerForward::new(system))
        } else {
            Box::new(ExplicitRungeKutta::new(params, system)?)
        };
        Ok(OdeSolver {
            params,
            ndim,
            actual,
            n_performed_steps: 0,
            n_rejected_steps: 0,
        })
    }

    /// Solves the ODE system
    ///
    /// # Input
    ///
    /// * `y0` -- the initial value of the vector of dependent variables; it will be updated to `y1` at the end
    /// * `x0` -- the initial value of the independent variable
    /// * `x1` -- the final value of the independent variable
    /// * `args` -- holds some extra arguments for the function `F` and jacobian `J`
    /// * `h_equal` -- a constant stepsize for solving with equal-steps; otherwise,
    ///   if possible, variable step sizes are automatically calculated. If automatic
    ///   sub-stepping is not possible (e.g., the RK method is not embedded),
    ///   a constant (and equal) stepsize will be calculated for [N_EQUAL_STEPS] steps.
    /// * `output_step` -- handles the output of results during accepted steps
    /// * `output_dense` -- handles the dense output
    ///
    /// # Generics
    ///
    /// * `S` -- step output function such as `fn(step: usize, h: f64, x: f64, y: &Vector)`
    /// * `D` -- dense output function such as `fn(y_out: &mut Vector, x_out: f64, step: usize, h: f64, x: f64, y: &Vector)`
    pub fn solve<S, D>(
        &mut self,
        y0: &mut Vector,
        x0: f64,
        x1: f64,
        args: &mut A,
        h_equal: Option<f64>,
        mut output_step: S,
        mut _output_dense: D,
    ) -> Result<(), StrError>
    where
        S: FnMut(usize, f64, f64, &Vector) -> Result<bool, StrError>,
        D: FnMut(&mut Vector, f64, usize, f64, f64, &Vector) -> Result<bool, StrError>,
    {
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
        assert!(h > 0.0);

        // restart variables
        self.actual.initialize(x0, y0);
        self.n_performed_steps = 0;
        self.n_rejected_steps = 0;

        // current values
        let mut x = x0; // will become x1 at the end
        let y = y0; // will become y1 at the end

        // equal-stepping loop
        if equal_stepping {
            const IGNORED: f64 = 0.0;
            let nstep = f64::ceil((x1 - x) / h) as usize;
            for step in 0..nstep {
                // step
                self.actual.step(x, &y, h, args)?;
                self.n_performed_steps += 1;

                // update x
                x = ((step + 1) as f64) * h;

                // update y
                self.actual.accept(y, x, h, IGNORED, IGNORED, args)?;

                // output
                let stop = (output_step)(step, h, x, y)?;
                if stop {
                    return Ok(());
                }
            }
            return Ok(());
        }

        // variable steps
        Ok(())
    }
}

/// Disables the output of accepted steps
pub fn no_step_output(_step: usize, _h: f64, _x: f64, _y: &Vector) -> Result<bool, StrError> {
    Ok(false)
}

/// Disables the dense output
pub fn np_dense_output(
    _y_out: &mut Vector,
    _x_out: f64,
    _step: usize,
    _h: f64,
    _x: f64,
    _y: &Vector,
) -> Result<bool, StrError> {
    Ok(false)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{np_dense_output, OdeSolver};
    use crate::{no_jacobian, HasJacobian, Method, OdeParams, OdeSystem, N_EQUAL_STEPS};
    use russell_lab::{vec_approx_eq, Vector};

    #[test]
    fn solve_works_1() {
        // solving
        //
        // dy
        // —— = 1   with   y(x=0)=0    thus   y(x) = x
        // dx

        let params = OdeParams::new(Method::FwEuler, None, None);
        let system = OdeSystem::new(
            1,
            |f, _, _, _| {
                f[0] = 1.0;
                Ok(())
            },
            no_jacobian,
            HasJacobian::No,
            true,
            None,
            None,
        );

        // consistent initial conditions
        let y_ana = |x| x;
        let mut x0 = 0.0;
        let mut y0 = Vector::from(&[y_ana(x0)]);

        // output arrays
        let mut h_values = Vec::new();
        let mut x_values = vec![x0];
        let mut y_values = vec![y0[0]];
        let mut e_values = vec![0.0]; // global errors
        let output_step = |_, h, x, y: &Vector| {
            h_values.push(h);
            x_values.push(x);
            y_values.push(y[0]);
            e_values.push(y_ana(x) - y[0]);
            Ok(false)
        };

        // arguments
        struct Args {}
        let mut args = Args {};

        // solve the ODE system
        let mut solver = OdeSolver::new(&params, system).unwrap();
        let xf = 1.0;
        solver
            .solve(&mut y0, x0, xf, &mut args, None, output_step, np_dense_output)
            .unwrap();

        // check
        assert_eq!(h_values.len(), N_EQUAL_STEPS);
        assert_eq!(x_values.len(), N_EQUAL_STEPS + 1);
        assert_eq!(y_values.len(), N_EQUAL_STEPS + 1);
        assert_eq!(e_values.len(), N_EQUAL_STEPS + 1);
        let h_equal_correct = (xf - x0) / (N_EQUAL_STEPS as f64);
        let h_values_correct = Vector::filled(N_EQUAL_STEPS, h_equal_correct);
        let x_values_correct = Vector::linspace(x0, xf, N_EQUAL_STEPS + 1).unwrap();
        let e_values_correct = Vector::new(N_EQUAL_STEPS + 1); // all 0.0
        vec_approx_eq(&h_values, h_values_correct.as_data(), 1e-17);
        vec_approx_eq(&x_values, x_values_correct.as_data(), 1e-17);
        vec_approx_eq(&y_values, x_values_correct.as_data(), 1e-15);
        vec_approx_eq(&e_values, e_values_correct.as_data(), 1e-15);

        // reset problem
        x0 = 0.0;
        y0[0] = y_ana(x0);
        h_values.clear();
        x_values.clear();
        y_values.clear();
        e_values.clear();
        x_values.push(x0);
        y_values.push(y0[0]);
        e_values.push(0.0);
        let output_step = |_, h, x, y: &Vector| {
            h_values.push(h);
            x_values.push(x);
            y_values.push(y[0]);
            e_values.push(y_ana(x) - y[0]);
            Ok(false)
        };

        // solve the ODE system again with prescribed h_equal
        let h_equal = Some(0.3);
        let xf = 1.2; // => will generate 4 steps
        solver
            .solve(&mut y0, x0, xf, &mut args, h_equal, output_step, np_dense_output)
            .unwrap();

        // check again
        let nstep = 4;
        let h_values_correct = Vector::filled(nstep, 0.3);
        let x_values_correct = Vector::linspace(x0, xf, nstep + 1).unwrap();
        let e_values_correct = Vector::new(nstep + 1); // all 0.0
        vec_approx_eq(&h_values, h_values_correct.as_data(), 1e-17);
        vec_approx_eq(&x_values, x_values_correct.as_data(), 1e-17);
        vec_approx_eq(&y_values, x_values_correct.as_data(), 1e-15);
        vec_approx_eq(&e_values, e_values_correct.as_data(), 1e-15);
    }
}
