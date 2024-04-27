use crate::constants::N_EQUAL_STEPS;
use crate::{EulerBackward, EulerForward, ExplicitRungeKutta, Radau5};
use crate::{Method, OdeSolverTrait, Params, Stats, System, Workspace};
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
/// The (scaled) Jacobian matrix is defined by:
///
/// ```text
///                 ∂{f}
/// [J](x, {y}) = α ————
///                 ∂{y}
/// ```
///
/// where `[J]` is the scaled Jacobian matrix and `α` is a scaling coefficient.
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
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_approx_eq, StrError, Vector};
/// use russell_ode::prelude::*;
///
/// fn main() -> Result<(), StrError> {
///     // ODE system
///     let ndim = 1;
///     let jac_nnz = 1;
///     let system = System::new(
///         ndim,
///         |f, x, y, _args: &mut NoArgs| {
///             f[0] = x + y[0];
///             Ok(())
///         },
///         |jj, alpha, _x, _y, _args: &mut NoArgs| {
///             jj.reset();
///             jj.put(0, 0, alpha * (1.0))?;
///             Ok(())
///         },
///         HasJacobian::Yes,
///         Some(jac_nnz),
///         None,
///     );
///
///     // solver
///     let params = Params::new(Method::Radau5);
///     let mut solver = OdeSolver::new(params, &system)?;
///
///     // initial values
///     let x = 0.0;
///     let mut y = Vector::from(&[0.0]);
///
///     // solve from x = 0 to x = 1
///     let x1 = 1.0;
///     let mut args = 0;
///     solver.solve(&mut y, x, x1, None, None, &mut args)?;
///
///     // check the results
///     let y_ana = Vector::from(&[f64::exp(x1) - x1 - 1.0]);
///     vec_approx_eq(&y, &y_ana, 1e-5);
///
///     // print stats
///     println!("{}", solver.stats());
///     Ok(())
/// }
/// ```
pub struct OdeSolver<'a, A> {
    /// Holds the parameters
    params: Params,

    /// Dimension of the ODE system
    ndim: usize,

    /// Holds a pointer to the actual ODE system solver
    actual: Box<dyn OdeSolverTrait<A> + 'a>,

    /// Holds statistics, benchmarking and "work" variables
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
    /// * `J` -- function to compute the Jacobian: `(jj: &mut CooMatrix, alpha: f64, x: f64, y: &Vector, args: &mut A)`
    /// * `A` -- generic argument to assist in the `F` and `J` functions. It may be simply [crate::NoArgs] indicating that no arguments are needed.
    pub fn new<F, J>(params: Params, system: &'a System<F, J, A>) -> Result<Self, StrError>
    where
        F: 'a + Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
        J: 'a + Fn(&mut CooMatrix, f64, f64, &Vector, &mut A) -> Result<(), StrError>,
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
    pub fn stats(&self) -> &Stats {
        &self.work.stats
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
        mut output: Option<&mut Output<A>>,
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
            out.stiff_record = self.params.stiffness.save_results;
            let stop = out.execute(&self.work, h, x, y, &self.actual, args)?;
            if stop {
                return Ok(());
            }
        }

        // equal-stepping loop
        if equal_stepping {
            let nstep = f64::ceil((x1 - x) / h) as usize;
            for _ in 0..nstep {
                self.work.stats.sw_step.reset();

                // step
                self.work.stats.n_steps += 1;
                self.actual.step(&mut self.work, x, &y, h, args)?;

                // update x and y
                self.work.stats.n_accepted += 1; // this must be after `self.actual.step`
                self.actual.accept(&mut self.work, &mut x, y, h, args)?;

                // check for anomalies
                vec_all_finite(&y, self.params.debug)?;

                // output
                if let Some(out) = output.as_mut() {
                    let stop = out.execute(&self.work, h, x, y, &self.actual, args)?;
                    if stop {
                        self.work.stats.stop_sw_step();
                        self.work.stats.stop_sw_total();
                        return Ok(());
                    }
                }
                self.work.stats.stop_sw_step();
            }
            if let Some(out) = output.as_mut() {
                out.last(&self.work, h, x, y, args)?;
            }
            self.work.stats.stop_sw_total();
            return Ok(());
        }

        // variable steps: control variables
        let mut success = false;
        let mut last_step = false;

        // variable stepping loop
        for _ in 0..self.params.step.n_step_max {
            self.work.stats.sw_step.reset();

            // converged?
            let dx = x1 - x;
            if dx <= 10.0 * f64::EPSILON {
                success = true;
                self.work.stats.stop_sw_step();
                break;
            }

            // update and check the stepsize
            h = f64::min(self.work.h_new, dx);
            if h <= 10.0 * f64::EPSILON {
                return Err("the stepsize becomes too small");
            }

            // step
            self.work.stats.n_steps += 1;
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
                self.work.stats.n_accepted += 1;
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
                self.work.stats.h_accepted = self.work.h_new;

                // output
                if let Some(out) = output.as_mut() {
                    let stop = out.execute(&self.work, h, x, y, &self.actual, args)?;
                    if stop {
                        self.work.stats.stop_sw_step();
                        self.work.stats.stop_sw_total();
                        return Ok(());
                    }
                }

                // converged?
                if last_step {
                    success = true;
                    self.work.stats.stop_sw_step();
                    break;
                }

                // check if the last step is approaching
                if x + self.work.h_new >= x1 {
                    last_step = true;
                }

            // reject step
            } else {
                // set flags
                if self.work.stats.n_accepted > 0 {
                    self.work.stats.n_rejected += 1;
                }
                self.work.follows_reject_step = true;
                last_step = false;

                // recompute stepsize
                if self.work.stats.n_accepted == 0 && self.params.step.m_first_reject > 0.0 {
                    self.work.h_new = h * self.params.step.m_first_reject;
                } else {
                    self.actual.reject(&mut self.work, h);
                }
            }
        }

        // last output
        if let Some(out) = output.as_mut() {
            out.last(&self.work, h, x, y, args)?;
        }

        // done
        self.work.stats.stop_sw_total();
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
    use crate::{no_jacobian, HasJacobian, NoArgs, OutCallback, OutCount, OutData, Output};
    use crate::{Method, Params, Samples, System};
    use russell_lab::{approx_eq, array_approx_eq, vec_approx_eq, Vector};
    use russell_sparse::Genie;

    #[test]
    fn new_captures_errors() {
        let (system, _, _, _, _) = Samples::simple_system_with_mass_matrix(false, Genie::Umfpack);
        let mut params = Params::new(Method::MdEuler);
        assert_eq!(
            OdeSolver::new(params, &system).err(),
            Some("the method must be Radau5 for systems with a mass matrix")
        );
        let (system, _, _, _, _) = Samples::simple_equation_constant();
        params.step.m_max = 0.0; // wrong
        assert_eq!(
            OdeSolver::new(params, &system).err(),
            Some("parameter must satisfy: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
    }

    #[test]
    fn solve_captures_errors() {
        let (system, _, _, mut args, _) = Samples::simple_equation_constant();
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        let mut y0 = Vector::new(system.ndim + 1); // wrong dim
        assert_eq!(
            solver.solve(&mut y0, 0.0, 1.0, None, None, &mut args).err(),
            Some("y0.dim() must be equal to ndim")
        );
        let mut y0 = Vector::new(system.ndim);
        assert_eq!(
            solver.solve(&mut y0, 0.0, 0.0, None, None, &mut args).err(),
            Some("x1 must be greater than x0")
        );
        let h_equal = Some(f64::EPSILON); // will cause an error
        assert_eq!(
            solver.solve(&mut y0, 0.0, 1.0, h_equal, None, &mut args).err(),
            Some("h_equal must be ≥ 10.0 * f64::EPSILON")
        );
    }

    #[test]
    fn nan_and_infinity_are_captured() {
        // this problem cannot be solved by FwEuler or MdEuler and these parameters;
        // it becomes stiff and yield infinite results
        let (system, _, mut y0, mut args, _) = Samples::brusselator_ode();
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        assert_eq!(
            solver.solve(&mut y0, 0.0, 9.0, Some(1.0), None, &mut args).err(),
            Some("an element of the vector is either infinite or NaN")
        );
        let params = Params::new(Method::MdEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        assert_eq!(
            solver.solve(&mut y0, 0.0, 1.0, None, None, &mut args).err(),
            Some("an element of the vector is either infinite or NaN")
        );
    }

    #[test]
    fn lack_of_convergence_is_captured() {
        let (system, _, mut y0, mut args, _) = Samples::simple_equation_constant();
        let mut params = Params::new(Method::MdEuler);
        params.step.n_step_max = 1; // will make the solver to fail (too few steps)
        let mut solver = OdeSolver::new(params, &system).unwrap();
        assert_eq!(
            solver.solve(&mut y0, 0.0, 1.0, None, None, &mut args).err(),
            Some("variable stepping did not converge")
        );
    }

    #[test]
    fn solve_with_n_equal_steps_works() {
        // solve the ODE system (will run with N_EQUAL_STEPS)
        let (system, x0, y0, mut args, _) = Samples::simple_equation_constant();
        let x1 = 1.0;
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        let mut y = y0.clone();
        solver.solve(&mut y, x0, x1, None, None, &mut args).unwrap();
        vec_approx_eq(&y, &[1.0], 1e-15);
    }

    #[test]
    fn solve_completes_after_a_single_step() {
        // since the problem is linear, this stepsize will lead to a jump from x=0.0 to 1.0
        let (system, _, mut y0, mut args, _) = Samples::simple_equation_constant();
        let mut params = Params::new(Method::DoPri5);
        params.step.h_ini = 20.0; // will be truncated to 1 yielding a single step
        let mut solver = OdeSolver::new(params, &system).unwrap();
        solver.solve(&mut y0, 0.0, 1.0, None, None, &mut args).unwrap();
        assert_eq!(solver.work.stats.n_accepted, 1);
        vec_approx_eq(&y0, &[1.0], 1e-15);
    }

    #[test]
    fn solve_with_variable_steps_works() {
        let (system, _, mut y0, mut args, _) = Samples::simple_equation_constant();
        let mut params = Params::new(Method::MdEuler);
        params.step.h_ini = 0.1;
        let mut solver = OdeSolver::new(params, &system).unwrap();
        solver.solve(&mut y0, 0.0, 0.3, None, None, &mut args).unwrap();
        vec_approx_eq(&y0, &[0.3], 1e-15);
    }

    #[test]
    fn update_params_captures_errors() {
        let (system, _, _, _, _) = Samples::simple_equation_constant();
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
        params.method = Method::MdEuler;
        params.step.m_max = 0.0;
        assert_eq!(
            solver.update_params(params).err(),
            Some("parameter must satisfy: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
    }

    #[test]
    fn solve_with_out_step_captures_errors() {
        let (system, x0, mut y0, mut args, _) = Samples::simple_equation_constant();
        let x1 = 1.0;
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        let mut out = Output::new();
        out.set_dense_recording(true, 0.1, &[0]).unwrap();
        assert_eq!(
            solver.solve(&mut y0, x0, x1, None, Some(&mut out), &mut args).err(),
            Some("dense output is not available for the FwEuler method")
        );
    }

    #[test]
    fn solve_with_step_output_works() {
        // system and solver
        let (system, _, y0, mut args, y_fn_x) = Samples::simple_equation_constant();
        let params = Params::new(Method::DoPri5);
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // output
        let mut out = Output::new();
        let path_key = "/tmp/russell_ode/test_solve_step_output_works";
        out.set_yx_correct(y_fn_x)
            .set_step_file_writing(true, path_key)
            .set_step_recording(true, &[0])
            .set_step_callback(true, |stats, h, x, y, _args| {
                assert_eq!(h, 0.2);
                approx_eq(x, (stats.n_accepted as f64) * h, 1e-15);
                approx_eq(y[0], (stats.n_accepted as f64) * h, 1e-15);
                Ok(false)
            });

        // solve
        let h_equal = Some(0.2);
        let mut y = y0.clone();
        solver
            .solve(&mut y, 0.0, 0.4, h_equal, Some(&mut out), &mut args)
            .unwrap();

        // check
        vec_approx_eq(&y, &[0.4], 1e-15);
        array_approx_eq(&out.step_h, &[0.2, 0.2, 0.2], 1e-15);
        array_approx_eq(&out.step_x, &[0.0, 0.2, 0.4], 1e-15);
        array_approx_eq(&out.step_y.get(&0).unwrap(), &[0.0, 0.2, 0.4], 1e-15);
        array_approx_eq(&out.step_global_error, &[0.0, 0.0, 0.0], 1e-15);

        // check count file
        let count = OutCount::read_json(&format!("{}_count.json", path_key)).unwrap();
        assert_eq!(count.n, 3);

        // check output files
        for i in 0..count.n {
            let res = OutData::read_json(&format!("{}_{}.json", path_key, i)).unwrap();
            assert_eq!(res.h, 0.2);
            approx_eq(res.x, (i as f64) * 0.2, 1e-15);
            approx_eq(res.y[0], (i as f64) * 0.2, 1e-15);
        }

        // define the callback function
        let cb: OutCallback<NoArgs> = |_stats, _h, _x, _y, _args| Err("unreachable");
        assert_eq!(cb(&solver.stats(), 0.0, 0.0, &y0, &mut args).err(), Some("unreachable"));

        // run again without step output
        out.clear();
        out.set_step_file_writing(false, path_key)
            .set_step_recording(false, &[])
            .set_step_callback(false, cb);
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, None, Some(&mut out), &mut args).unwrap();
        vec_approx_eq(&y, &[0.4], 1e-15);
        assert_eq!(out.step_h.len(), 0);
        assert_eq!(out.step_x.len(), 0);
        assert_eq!(out.step_y.len(), 0);
        assert_eq!(out.step_global_error.len(), 0);

        // run again and stop earlier
        out.clear();
        out.set_step_callback(true, |stats, _h, _x, _y, _args| {
            if stats.n_accepted > 0 {
                Ok(true) // stop
            } else {
                Ok(false) // do not stop
            }
        });
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, None, Some(&mut out), &mut args).unwrap();
        assert!(y[0] > 0.0 && y[0] < 0.4);

        // run again and stop due to error
        out.clear();
        out.set_step_callback(true, |stats, _h, _x, _y, _args| {
            if stats.n_accepted > 0 {
                Err("stop with error")
            } else {
                Ok(false) // do not stop
            }
        });
        let mut y = y0.clone();
        assert_eq!(
            solver.solve(&mut y, 0.0, 0.4, None, Some(&mut out), &mut args).err(),
            Some("stop with error")
        );
    }

    #[test]
    fn solve_with_dense_output_works() {
        // system and solver
        let (system, _, y0, mut args, y_fn_x) = Samples::simple_equation_constant();
        let params = Params::new(Method::DoPri5);
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // output
        let mut out = Output::new();
        const H_OUT: f64 = 0.1;
        let path_key = "/tmp/russell_ode/test_solve_dense_output_works";
        out.set_yx_correct(y_fn_x);
        out.set_dense_file_writing(true, H_OUT, path_key).unwrap();
        out.set_dense_recording(true, H_OUT, &[0]).unwrap();
        out.set_dense_callback(true, H_OUT, |stats, h, x, y, _args| {
            assert_eq!(h, 0.2);
            if stats.n_accepted < 2 {
                approx_eq(x, (stats.n_accepted as f64) * H_OUT, 1e-15);
                approx_eq(y[0], (stats.n_accepted as f64) * H_OUT, 1e-15);
            } else {
                approx_eq(y[0], x, 1e-15);
            }
            Ok(false)
        })
        .unwrap();

        // solve
        let h_equal = Some(0.2);
        let mut y = y0.clone();
        solver
            .solve(&mut y, 0.0, 0.4, h_equal, Some(&mut out), &mut args)
            .unwrap();

        // check
        vec_approx_eq(&y, &[0.4], 1e-15);
        assert_eq!(&out.dense_step_index, &[0, 1, 2, 2, 2]);
        array_approx_eq(&out.dense_x, &[0.0, 0.1, 0.2, 0.3, 0.4], 1e-15);
        array_approx_eq(&out.dense_y.get(&0).unwrap(), &[0.0, 0.1, 0.2, 0.3, 0.4], 1e-15);

        // check count file
        let count = OutCount::read_json(&format!("{}_count.json", path_key)).unwrap();
        assert_eq!(count.n, 5);

        // check output files
        for i in 0..count.n {
            let res = OutData::read_json(&format!("{}_{}.json", path_key, i)).unwrap();
            assert_eq!(res.h, 0.2); // fixed h, not h_out
            approx_eq(res.x, (i as f64) * H_OUT, 1e-15);
            approx_eq(res.y[0], (i as f64) * H_OUT, 1e-15);
        }

        // define the callback function
        let cb: OutCallback<NoArgs> = |_stats, _h, _x, _y, _args| Err("unreachable");
        assert_eq!(cb(&solver.stats(), 0.0, 0.0, &y0, &mut args).err(), Some("unreachable"));

        // run again without dense output
        out.clear();
        out.set_dense_file_writing(false, H_OUT, path_key).unwrap();
        out.set_dense_recording(false, H_OUT, &[]).unwrap();
        out.set_dense_callback(false, H_OUT, cb).unwrap();
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, None, Some(&mut out), &mut args).unwrap();
        vec_approx_eq(&y, &[0.4], 1e-15);
        assert_eq!(out.dense_step_index.len(), 0);
        assert_eq!(out.dense_x.len(), 0);
        assert_eq!(out.dense_y.len(), 0);

        // run again but stop at the first output
        out.clear();
        out.set_dense_callback(true, H_OUT, |_stats, _h, _x, _y, _args| {
            Ok(true) // stop
        })
        .unwrap();
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, None, Some(&mut out), &mut args).unwrap();
        assert_eq!(solver.work.stats.n_accepted, 0);
        assert_eq!(y[0], 0.0);

        // run again and stop earlier
        out.clear();
        out.set_dense_callback(true, H_OUT, |stats, _h, _x, _y, _args| {
            if stats.n_accepted > 0 {
                Ok(true) // stop
            } else {
                Ok(false) // do not stop
            }
        })
        .unwrap();
        // ... equal steps
        let mut y = y0.clone();
        solver
            .solve(&mut y, 0.0, 0.4, Some(0.2), Some(&mut out), &mut args)
            .unwrap();
        assert!(y[0] > 0.0 && y[0] < 0.4);
        // ... variable steps
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, None, Some(&mut out), &mut args).unwrap();
        assert!(y[0] > 0.0 && y[0] < 0.4);

        // run again and stop due to error
        out.clear();
        // ... first step
        out.set_dense_callback(true, H_OUT, |_stats, _h, _x, _y, _args| Err("stop with error"))
            .unwrap();
        let mut y = y0.clone();
        assert_eq!(
            solver.solve(&mut y, 0.0, 0.4, None, Some(&mut out), &mut args).err(),
            Some("stop with error")
        );
        // ... next steps
        out.set_dense_callback(true, H_OUT, |stats, _h, _x, _y, _args| {
            if stats.n_accepted > 0 {
                Err("stop with error")
            } else {
                Ok(false) // do not stop
            }
        })
        .unwrap();
        let mut y = y0.clone();
        assert_eq!(
            solver.solve(&mut y, 0.0, 0.4, None, Some(&mut out), &mut args).err(),
            Some("stop with error")
        );
    }

    #[test]
    fn solve_captures_errors_from_f_and_out() {
        // args
        struct Args {
            f_count: usize,
            f_barrier: usize,
            out_count: usize,
            out_barrier: usize,
        }
        let mut args = Args {
            f_count: 0,
            f_barrier: 0,
            out_count: 0,
            out_barrier: 2, // first and second outputs
        };

        // system
        let ndim = 1;
        let system = System::new(
            ndim,
            |f: &mut Vector, _x: f64, _y: &Vector, args: &mut Args| {
                if args.f_count == args.f_barrier {
                    return Err("f: artificial error");
                }
                f[0] = 1.0;
                args.f_count += 1;
                Ok(())
            },
            no_jacobian,
            HasJacobian::No,
            None,
            None,
        );

        // initial values and final x
        let x0 = 0.0;
        let x1 = 0.2;
        let mut y = Vector::from(&[0.0]);

        // parameters and solver
        let mut params = Params::new(Method::DoPri8);
        params.step.h_ini = 0.2;
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // output
        let mut out = Output::new();
        out.set_dense_callback(true, 0.1, |_stats, _h, _x, _y, args: &mut Args| {
            if args.out_count == args.out_barrier {
                return Err("out: artificial error");
            }
            args.out_count += 1;
            Ok(false) // do not stop
        })
        .unwrap();

        // equal steps -----------------------------------------------------------

        // first error @ actual.step
        assert_eq!(
            solver.solve(&mut y, x0, x1, Some(0.2), None, &mut args).err(),
            Some("f: artificial error")
        );

        // second error @ actual.accept
        // There are only two places in ERK where an error in 'accept' may occur:
        // (the other methods/solvers do not return Err in their 'accept')
        // 1. when saving data for dense output (only DoPri8)
        // 2. in computations related to the stiffness detection
        args.f_barrier += 12; // nstage = 12 (need to skip the next call to 'step')
        assert_eq!(
            solver.solve(&mut y, x0, x1, Some(0.2), Some(&mut out), &mut args).err(),
            Some("f: artificial error")
        );

        // third error @ the second output
        args.f_barrier += 2 * 12; // skip next calls to 'step'
        assert_eq!(
            solver.solve(&mut y, x0, x1, Some(0.2), Some(&mut out), &mut args).err(),
            Some("out: artificial error")
        );

        // fourth error @ the last output
        args.f_barrier += 2 * 12; // skip next calls to 'step'
        args.out_barrier += 2; // skip first and second output
        assert_eq!(
            solver.solve(&mut y, x0, x1, Some(0.2), Some(&mut out), &mut args).err(),
            Some("out: artificial error")
        );

        // variable steps --------------------------------------------------------

        // first error @ actual.step
        args.f_count = 0;
        args.f_barrier = 0;
        args.out_count = 0;
        args.out_barrier = 2; // first and second outputs
        assert_eq!(
            solver.solve(&mut y, x0, x1, None, None, &mut args).err(),
            Some("f: artificial error")
        );

        // second error @ actual.accept
        // There are only two places in ERK where an error in 'accept' may occur:
        // (the other methods/solvers do not return Err in their 'accept')
        // 1. when saving data for dense output (only DoPri8)
        // 2. in computations related to the stiffness detection
        args.f_barrier += 12; // nstage = 12 (need to skip the next call to 'step')
        assert_eq!(
            solver.solve(&mut y, x0, x1, None, Some(&mut out), &mut args).err(),
            Some("f: artificial error")
        );

        // error @ last output
        args.f_count = 0;
        args.f_barrier = 15 + 1; // 12 stages, 3 dense output, 1 safety
        args.out_count = 0;
        args.out_barrier = 2; // first and second outputs
        assert_eq!(
            solver.solve(&mut y, x0, x1, None, Some(&mut out), &mut args).err(),
            Some("out: artificial error")
        );
    }
}
