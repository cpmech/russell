use crate::constants::N_EQUAL_STEPS;
use crate::{Benchmark, Method, OdeSolverTrait, Params, System, Workspace};
use crate::{EulerBackward, EulerForward, ExplicitRungeKutta, Radau5};
use crate::{Output, StrError};
use russell_lab::{vec_all_finite, Vector};
use russell_sparse::CooMatrix;

/// Implements a numerical solver for systems of ODEs
///
/// The ODE and DAE systems are represented as follows:
///
/// ```text
///     d{y}
/// [M] ———— = {f}(x, {y})
///      dx
/// ```
///
/// where `x` is the independent scalar variable (e.g., time), `{y}` is the solution vector,
/// `{f}` is the right-hand side vector, and `[M]` is the so-called "mass matrix".
///
/// **Note:** The mass matrix is optional and need not be specified
/// (unless the DAE under study requires it).
///
/// The Jacobian is defined by:
///
/// ```text
///               ∂{f}
/// [J](x, {y}) = ————
///               ∂{y}
/// ```
///
/// where `[J]` is the Jacobian matrix.
///
/// **Note:** The Jacobian function is not required for explicit Runge-Kutta methods
/// (see [crate::Method] and [crate::Information]). Thus, one may simply pass the [crate::no_jacobian]
/// function and set [crate::HasJacobian::No] in the system.
///
/// The flag [crate::ParamsNewton::use_numerical_jacobian] may be set to true to compute the
/// Jacobian matrix numerically. This option works with or without specifying the analytical
/// Jacobian function.
///
/// # Recommended methods
///
/// * [Method::DoPri5] for ODE systems and non-stiff problems using moderate tolerances
/// * [Method::DoPri8] for ODE systems and non-stiff problems using strict tolerances
/// * [Method::Radau5] for ODE and DAE systems, possibly stiff, with moderate to strict tolerances
///
/// **Note:** A *Stiff problem* arises due to a combination of conditions, such as
/// the ODE system equations, the initial values, the stepsize, and the numerical method.
///
/// # Limitations
///
/// * Currently, the only method that can solve DAE systems is [Method::Radau5]
/// * Currently, *dense output* is only available for [Method::DoPri5], [Method::DoPri8], and [Method::Radau5]
///
/// # References
///
/// 1. E. Hairer, S. P. Nørsett, G. Wanner (2008) Solving Ordinary Differential Equations I.
///    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
///    in Computational Mathematics, 528p
/// 2. E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
///    Stiff and Differential-Algebraic Problems. Second Revised Edition.
///    Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
pub struct OdeSolver<'a, A> {
    /// Holds the parameters
    params: Params,

    /// Dimension of the ODE system
    ndim: usize,

    /// Holds a pointer to the actual ODE system solver
    actual: Box<dyn OdeSolverTrait<A> + 'a>,

    /// Holds benchmark and work variables
    work: Workspace,
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
    /// The generic arguments here are:
    ///
    /// * `F` -- function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    /// * `J` -- function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    /// * `A` -- generic argument to assist in the `F` and `J` functions. It may be simply the [crate::NoArgs] type indicating that no arguments are needed.
    pub fn new<F, J>(params: Params, system: &'a System<F, J, A>) -> Result<Self, StrError>
    where
        F: 'a + Send + Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
        J: 'a + Send + Fn(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
        A: 'a,
    {
        if system.mass_matrix.is_some() && params.method != Method::Radau5 {
            return Err("the method must be Radau5 for systems with a mass matrix");
        }
        params.validate()?;
        let ndim = system.ndim;
        let actual: Box<dyn OdeSolverTrait<A>> = if params.method == Method::Radau5 {
            Box::new(Radau5::new(params, system))
        } else if params.method == Method::BwEuler {
            Box::new(EulerBackward::new(params, system))
        } else if params.method == Method::FwEuler {
            Box::new(EulerForward::new(system))
        } else {
            Box::new(ExplicitRungeKutta::new(params, system).unwrap()) // unwrap here because an error cannot occur
        };
        Ok(OdeSolver {
            params,
            ndim,
            actual,
            work: Workspace::new(params.method),
        })
    }

    /// Returns some benchmarking data
    pub fn bench(&self) -> &Benchmark {
        &self.work.bench
    }

    /// Solves the ODE system
    ///
    /// # Input
    ///
    /// * `y0` -- the initial value of the vector of dependent variables; it will be updated to `y1` at the end
    /// * `x0` -- the initial value of the independent variable
    /// * `x1` -- the final value of the independent variable
    /// * `h_equal` -- a constant stepsize for solving with equal-steps; otherwise,
    ///   if possible, variable step sizes are automatically calculated. If automatic
    ///   stepping is not possible (e.g., the RK method is not embedded),
    ///   a constant (and equal) stepsize will be calculated for [N_EQUAL_STEPS] steps.
    /// * `output` -- structure to hold the results at accepted steps or at specified stations (continuous/dense output)
    pub fn solve(
        &mut self,
        y0: &mut Vector,
        x0: f64,
        x1: f64,
        h_equal: Option<f64>,
        mut output: Option<&mut Output>,
        args: &mut A,
    ) -> Result<(), StrError> {
        // check data
        if y0.dim() != self.ndim {
            return Err("y0.dim() must be equal to ndim");
        }
        if x1 <= x0 {
            return Err("x1 must be greater than x0");
        }

        // information
        let info = self.params.method.information();

        // initial stepsize
        let (equal_stepping, mut h) = match h_equal {
            Some(h_eq) => {
                if h_eq < 10.0 * f64::EPSILON {
                    return Err("h_equal must be ≥ 10.0 * f64::EPSILON");
                }
                let n = f64::ceil((x1 - x0) / h_eq) as usize;
                let h = (x1 - x0) / (n as f64);
                (true, h)
            }
            None => {
                if info.embedded {
                    let h = f64::min(self.params.step.h_ini, x1 - x0);
                    (false, h)
                } else {
                    let h = (x1 - x0) / (N_EQUAL_STEPS as f64);
                    (true, h)
                }
            }
        };
        assert!(h > 0.0);

        // reset variables
        self.work.reset(h, self.params.step.rel_error_prev_min);

        // current values
        let mut x = x0; // will become x1 at the end
        let y = y0; // will become y1 at the end

        // first output
        if let Some(out) = output.as_mut() {
            if out.with_dense_output() {
                self.actual.enable_dense_output()?;
            }
            out.save_stiff = self.params.stiffness.save_results;
            out.push(&self.work, x, y, h, &self.actual)?;
        }

        // equal-stepping loop
        if equal_stepping {
            let nstep = f64::ceil((x1 - x) / h) as usize;
            for _ in 0..nstep {
                self.work.bench.sw_step.reset();

                // step
                self.work.bench.n_steps += 1;
                self.actual.step(&mut self.work, x, &y, h, args)?;

                // update x and y
                self.work.bench.n_accepted += 1; // this must be after `self.actual.step`
                self.actual.accept(&mut self.work, &mut x, y, h, args)?;

                // check for anomalies
                vec_all_finite(&y, self.params.debug)?;

                // output
                if let Some(out) = output.as_mut() {
                    out.push(&self.work, x, y, h, &self.actual)?;
                }
                self.work.bench.stop_sw_step();
            }
            self.work.bench.stop_sw_total();
            return Ok(());
        }

        // variable steps: control variables
        let mut success = false;
        let mut last_step = false;

        // variable stepping loop
        for _ in 0..self.params.step.n_step_max {
            self.work.bench.sw_step.reset();

            // converged?
            let dx = x1 - x;
            if dx <= 10.0 * f64::EPSILON {
                success = true;
                self.work.bench.stop_sw_step();
                break;
            }

            // update and check the stepsize
            h = f64::min(self.work.h_new, dx);
            if h <= 10.0 * f64::EPSILON {
                return Err("the stepsize becomes too small");
            }

            // step
            self.work.bench.n_steps += 1;
            self.actual.step(&mut self.work, x, &y, h, args)?;

            // handle diverging iterations
            if self.work.iterations_diverging {
                self.work.iterations_diverging = false;
                self.work.follows_reject_step = true;
                last_step = false;
                self.work.h_new = h * self.work.h_multiplier_diverging;
                continue;
            }

            // accept step
            if self.work.rel_error < 1.0 {
                // update x and y
                self.work.bench.n_accepted += 1;
                self.actual.accept(&mut self.work, &mut x, y, h, args)?;

                // check for anomalies
                vec_all_finite(&y, self.params.debug)?;

                // do not allow h to grow if previous step was a reject
                if self.work.follows_reject_step {
                    self.work.h_new = f64::min(self.work.h_new, h);
                }
                self.work.follows_reject_step = false;

                // save previous stepsize, relative error, and accepted/suggested stepsize
                self.work.h_prev = h;
                self.work.rel_error_prev = f64::max(self.params.step.rel_error_prev_min, self.work.rel_error);
                self.work.bench.h_accepted = self.work.h_new;

                // output
                if let Some(out) = output.as_mut() {
                    out.push(&self.work, x, y, h, &self.actual)?;
                }

                // converged?
                if last_step {
                    success = true;
                    self.work.bench.stop_sw_step();
                    break;
                }

                // check if the last step is approaching
                if x + self.work.h_new >= x1 {
                    last_step = true;
                }

            // reject step
            } else {
                // set flags
                if self.work.bench.n_accepted > 0 {
                    self.work.bench.n_rejected += 1;
                }
                self.work.follows_reject_step = true;
                last_step = false;

                // recompute stepsize
                if self.work.bench.n_accepted == 0 && self.params.step.m_first_reject > 0.0 {
                    self.work.h_new = h * self.params.step.m_first_reject;
                } else {
                    self.actual.reject(&mut self.work, h);
                }
            }
        }

        // done
        self.work.bench.stop_sw_total();
        if success {
            Ok(())
        } else {
            Err("variable stepping did not converge")
        }
    }

    /// Update the parameters (e.g., for sensitive analyses)
    pub fn update_params(&mut self, params: Params) -> Result<(), StrError> {
        if params.method != self.params.method {
            return Err("update_params must not change the method");
        }
        params.validate()?;
        self.actual.update_params(params);
        self.params = params;
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::OdeSolver;
    use crate::{Method, Output, Params, Samples, N_EQUAL_STEPS};
    use russell_lab::{vec_approx_eq, vec_copy, Vector};
    use russell_sparse::Genie;

    #[test]
    fn new_captures_errors() {
        let (system, _, _) = Samples::simple_system_with_mass_matrix(false, Genie::Umfpack);
        let params = Params::new(Method::MdEuler);
        assert_eq!(
            OdeSolver::new(params, &system).err(),
            Some("the method must be Radau5 for systems with a mass matrix")
        );
    }

    #[test]
    fn solve_with_step_output_works() {
        // ODE system
        let (system, data, mut args) = Samples::simple_equation_constant();

        // output
        let mut out = Output::new();
        out.y_analytical = data.y_analytical;
        out.enable_step(&[0]);

        // params and solver
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // solve the ODE system (will run with N_EQUAL_STEPS)
        let mut y = data.y0.clone();
        solver
            .solve(&mut y, data.x0, data.x1, None, Some(&mut out), &mut args)
            .unwrap();

        // check
        let h_equal_correct = (data.x1 - data.x0) / (N_EQUAL_STEPS as f64);
        let h_values_correct = Vector::filled(N_EQUAL_STEPS + 1, h_equal_correct);
        let x_values_correct = Vector::linspace(data.x0, data.x1, N_EQUAL_STEPS + 1).unwrap();
        let e_values_correct = Vector::new(N_EQUAL_STEPS + 1); // all 0.0
        vec_approx_eq(y.as_data(), &[1.0], 1e-15);
        vec_approx_eq(&out.step_h, h_values_correct.as_data(), 1e-15);
        vec_approx_eq(&out.step_x, x_values_correct.as_data(), 1e-15);
        vec_approx_eq(&out.step_y.get(&0).unwrap(), x_values_correct.as_data(), 1e-15);
        vec_approx_eq(&out.step_global_error, e_values_correct.as_data(), 1e-15);

        // run again without step output
        out.clear();
        out.disable_step();
        vec_copy(&mut y, &data.y0).unwrap();
        solver
            .solve(&mut y, data.x0, data.x1, None, Some(&mut out), &mut args)
            .unwrap();
        vec_approx_eq(y.as_data(), &[1.0], 1e-15);
        assert_eq!(out.step_h.len(), 0);
        assert_eq!(out.step_x.len(), 0);
        assert_eq!(out.step_y.get(&0).unwrap().len(), 0);
        assert_eq!(out.step_global_error.len(), 0);
    }

    #[test]
    fn solve_with_h_equal_works() {
        // ODE system
        let (system, mut data, mut args) = Samples::simple_equation_constant();

        // output
        let mut out = Output::new();
        out.enable_step(&[0]);

        // params and solver
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        let x1 = 1.2; // => will generate 4 steps

        // capture error
        let h_equal = Some(f64::EPSILON); // will cause error
        assert_eq!(
            solver
                .solve(&mut data.y0, data.x0, x1, h_equal, Some(&mut out), &mut args)
                .err(),
            Some("h_equal must be ≥ 10.0 * f64::EPSILON")
        );

        // solve the ODE system
        let h_equal = Some(0.3);
        solver
            .solve(&mut data.y0, data.x0, x1, h_equal, Some(&mut out), &mut args)
            .unwrap();

        // check
        let nstep = 4;
        let h_values_correct = Vector::filled(nstep + 1, 0.3);
        let x_values_correct = Vector::linspace(data.x0, x1, nstep + 1).unwrap();
        vec_approx_eq(data.y0.as_data(), &[x1], 1e-15);
        vec_approx_eq(&out.step_h, h_values_correct.as_data(), 1e-15);
        vec_approx_eq(&out.step_x, x_values_correct.as_data(), 1e-15);
        vec_approx_eq(&out.step_y.get(&0).unwrap(), x_values_correct.as_data(), 1e-15);
        assert_eq!(out.step_global_error.len(), 0); // not available when y_analytical is not provided
    }

    #[test]
    fn solve_with_variable_steps_works() {
        // ODE system
        let (system, mut data, mut args) = Samples::simple_equation_constant();

        // output
        let mut out = Output::new();
        out.y_analytical = data.y_analytical;
        out.enable_step(&[0]);

        // params and solver
        let mut params = Params::new(Method::MdEuler);
        params.step.h_ini = 0.1;
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // solve the ODE system
        solver
            .solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)
            .unwrap();

        // check
        vec_approx_eq(data.y0.as_data(), &[1.0], 1e-15);
        vec_approx_eq(&out.step_h, &[0.1, 0.1, 0.9], 1e-15);
        vec_approx_eq(&out.step_x, &[0.0, 0.1, 1.0], 1e-15);
        vec_approx_eq(&out.step_y.get(&0).unwrap(), &[0.0, 0.1, 1.0], 1e-15);
        vec_approx_eq(&&out.step_global_error, &[0.0, 0.0, 0.0], 1e-15);
    }

    #[test]
    fn solve_with_dense_output_works() {
        // ODE system
        let (system, data, mut args) = Samples::simple_equation_constant();

        // output
        let mut out = Output::new();
        out.enable_dense(0.25, &[0]).unwrap();

        // params and solver
        let mut params = Params::new(Method::DoPri5);
        params.step.h_ini = 0.1;
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // solve the ODE system
        let mut y = data.y0.clone();
        solver
            .solve(&mut y, data.x0, data.x1, None, Some(&mut out), &mut args)
            .unwrap();

        // check
        vec_approx_eq(y.as_data(), &[1.0], 1e-15);
        vec_approx_eq(&out.dense_x, &[0.0, 0.25, 0.5, 0.75], 1e-15); // will not store the last x here
        vec_approx_eq(&out.dense_y.get(&0).unwrap(), &[0.0, 0.25, 0.5, 0.75], 1e-15);
        assert_eq!(&out.dense_step_index, &[0, 2, 2, 2]);

        // run again without dense output
        out.clear();
        out.disable_dense();
        vec_copy(&mut y, &data.y0).unwrap();
        solver
            .solve(&mut y, data.x0, data.x1, None, Some(&mut out), &mut args)
            .unwrap();
        vec_approx_eq(y.as_data(), &[1.0], 1e-15);
        assert_eq!(out.dense_x.len(), 0);
        assert_eq!(out.dense_y.get(&0).unwrap().len(), 0);
        assert_eq!(out.dense_step_index.len(), 0);
    }

    #[test]
    fn solve_handles_zero_stepsize() {
        // ODE system
        let (system, mut data, mut args) = Samples::simple_equation_constant();

        // output
        let mut out = Output::new();
        out.enable_step(&[0]);

        // params and solver
        let mut params = Params::new(Method::DoPri5);
        params.step.h_ini = 20.0; // since the problem is linear, this stepsize will lead to a jump from x=0.0 to 1.0
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // solve the ODE system
        solver
            .solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)
            .unwrap();

        // check
        vec_approx_eq(data.y0.as_data(), &[1.0], 1e-15);
        vec_approx_eq(&out.step_h, &[1.0, 1.0], 1e-15);
        vec_approx_eq(&out.step_x, &[0.0, 1.0], 1e-15);
        vec_approx_eq(&out.step_y.get(&0).unwrap(), &[0.0, 1.0], 1e-15);
    }

    #[test]
    fn solve_and_output_handle_errors() {
        let (system, mut data, mut args) = Samples::simple_equation_constant();
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        let mut out = Output::new();
        assert_eq!(out.enable_dense(-0.1, &[0]).err(), Some("h_out must be positive"));
        out.enable_dense(0.1, &[0]).unwrap();
        assert_eq!(
            solver
                .solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)
                .err(),
            Some("dense output is not available for the FwEuler method")
        );
    }

    #[test]
    fn nan_and_infinity_are_captured() {
        let (system, data, mut args, _) = Samples::brusselator_ode();
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        let mut y = data.y0.clone();
        let x = data.x0;
        let h = 0.5;
        assert_eq!(
            solver.solve(&mut y, x, data.x1, Some(h), None, &mut args).err(),
            Some("an element of the vector is either infinite or NaN")
        );
    }

    #[test]
    fn lack_of_convergence_is_captured() {
        let (system, data, mut args) = Samples::simple_equation_constant();
        let mut params = Params::new(Method::MdEuler);
        params.step.n_step_max = 1;
        let mut solver = OdeSolver::new(params, &system).unwrap();
        let mut y = data.y0.clone();
        let x = data.x0;
        assert_eq!(
            solver.solve(&mut y, x, data.x1, None, None, &mut args).err(),
            Some("variable stepping did not converge")
        );
    }

    #[test]
    fn update_params_works() {
        let (system, _, _) = Samples::simple_equation_constant();
        let mut params = Params::new(Method::MdEuler);
        params.step.n_step_max = 0;
        assert_eq!(
            OdeSolver::new(params, &system).err(),
            Some("parameter must satisfy: n_step_max ≥ 1")
        );
        params.step.n_step_max = 1000;
        let mut solver = OdeSolver::new(params, &system).unwrap();
        assert_eq!(solver.params.step.n_step_max, 1000);
        params.step.n_step_max = 2; // this will not change the solver until update_params is called
        assert_eq!(solver.params.step.n_step_max, 1000);
        solver.update_params(params).unwrap();
        assert_eq!(solver.params.step.n_step_max, 2);
        params.method = Method::FwEuler;
        assert_eq!(
            solver.update_params(params).err(),
            Some("update_params must not change the method")
        );
    }

    #[test]
    fn solve_capture_errors() {
        let (system, mut data, mut args) = Samples::simple_equation_constant();
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        let mut y0 = Vector::new(system.ndim + 1); // wrong dim
        assert_eq!(
            solver.solve(&mut y0, data.x0, data.x1, None, None, &mut args).err(),
            Some("y0.dim() must be equal to ndim")
        );
        let x1 = data.x0; // wrong value
        assert_eq!(
            solver.solve(&mut data.y0, data.x0, x1, None, None, &mut args).err(),
            Some("x1 must be greater than x0")
        );
    }
}
