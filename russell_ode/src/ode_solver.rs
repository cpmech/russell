use crate::constants::N_EQUAL_STEPS;
use crate::{EulerBackward, EulerForward, ExplicitRungeKutta, Radau5};
use crate::{Method, OdeSolverTrait, Params, Stats, System, Workspace};
use crate::{Output, StrError};
use russell_lab::{vec_all_finite, Vector};

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
/// (see [crate::Method] and [crate::Information]).
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
/// use russell_sparse::Sym;
///
/// fn main() -> Result<(), StrError> {
///     // ODE system
///     let ndim = 1;
///     let jac_nnz = 1;
///     let mut system = System::new(ndim, |f, x, y, _args: &mut NoArgs| {
///         f[0] = x + y[0];
///         Ok(())
///     });
///     system.set_jacobian(Some(jac_nnz), Sym::No, |jj, alpha, _x, _y, _args: &mut NoArgs| {
///         jj.reset();
///         jj.put(0, 0, alpha * (1.0))?;
///         Ok(())
///     });
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
///     solver.solve(&mut y, x, x1, None, &mut args)?;
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

    /// Assists in generating the output of results (steps or dense)
    output: Output<'a, A>,

    /// Indicates whether the output is enabled or not
    output_enabled: bool,
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
    /// * `A` -- generic argument to assist in the f(x,y) and Jacobian functions.
    ///   It may be simply [NoArgs] indicating that no arguments are needed.
    pub fn new(params: Params, system: &'a System<'a, A>) -> Result<Self, StrError>
    where
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
            output: Output::new(),
            output_enabled: false,
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
        if self.output_enabled {
            self.output.initialize(x0, x1, self.params.stiffness.save_results)?;
            if self.output.with_dense_output() {
                self.actual.enable_dense_output()?;
            }
            let stop = self.output.execute(&self.work, h, x, y, &self.actual, args)?;
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
                if self.output_enabled {
                    let stop = self.output.execute(&self.work, h, x, y, &self.actual, args)?;
                    if stop {
                        self.work.stats.stop_sw_step();
                        self.work.stats.stop_sw_total();
                        return Ok(());
                    }
                }
                self.work.stats.stop_sw_step();
            }
            if self.output_enabled {
                self.output.last(&self.work, h, x, y, args)?;
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
                if self.output_enabled {
                    let stop = self.output.execute(&self.work, h, x, y, &self.actual, args)?;
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
        if self.output_enabled {
            self.output.last(&self.work, h, x, y, args)?;
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

    /// Enables the output of results
    ///
    /// Returns an access to the output structure for further configuration
    pub fn enable_output(&mut self) -> &mut Output<'a, A> {
        self.output_enabled = true;
        &mut self.output
    }

    /// Returns an access to the output during accepted steps: stepsizes (h)
    pub fn out_step_h(&self) -> &Vec<f64> {
        &self.output.step_h
    }

    /// Returns an access to the output during accepted steps: x values
    pub fn out_step_x(&self) -> &Vec<f64> {
        &self.output.step_x
    }

    /// Returns an access to the output during accepted steps: y values
    ///
    /// # Panics
    ///
    /// A panic will occur if `m` is out of range
    pub fn out_step_y(&self, m: usize) -> &Vec<f64> {
        &self.output.step_y.get(&m).unwrap()
    }

    /// Returns an access to the output during accepted steps: global error
    pub fn out_step_global_error(&self) -> &Vec<f64> {
        &self.output.step_global_error
    }

    /// Returns an access to the dense output: x values
    pub fn out_dense_x(&self) -> &Vec<f64> {
        &self.output.dense_x
    }

    /// Returns an access to the dense output: y values
    ///
    /// # Panics
    ///
    /// A panic will occur if `m` is out of range
    pub fn out_dense_y(&self, m: usize) -> &Vec<f64> {
        &self.output.dense_y.get(&m).unwrap()
    }

    /// Returns an access to the stiffness detection results: accepted step index
    ///
    /// This is the index of the accepted step for which stiffness has been detected.
    pub fn out_stiff_step_index(&self) -> &Vec<usize> {
        &self.output.stiff_step_index
    }

    /// Returns an access to the stiffness detection results: x values
    pub fn out_stiff_x(&self) -> &Vec<f64> {
        &self.output.stiff_x
    }

    /// Returns an access to the stiffness detection results: h times ρ
    ///
    /// `h·ρ` is the approximation of the boundary of the stability region.
    ///
    /// Note: `ρ` is an approximation of `|λ|`, where `λ` is the dominant eigenvalue of the Jacobian
    /// (see Hairer-Wanner Part II page 22).
    pub fn out_stiff_h_times_rho(&self) -> &Vec<f64> {
        &self.output.stiff_h_times_rho
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::OdeSolver;
    use crate::{Method, Params, Samples, System};
    use crate::{NoArgs, OutCount, OutData, Stats, StrError};
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
            solver.solve(&mut y0, 0.0, 1.0, None, &mut args).err(),
            Some("y0.dim() must be equal to ndim")
        );
        let mut y0 = Vector::new(system.ndim);
        assert_eq!(
            solver.solve(&mut y0, 0.0, 0.0, None, &mut args).err(),
            Some("x1 must be greater than x0")
        );
        let h_equal = Some(f64::EPSILON); // will cause an error
        assert_eq!(
            solver.solve(&mut y0, 0.0, 1.0, h_equal, &mut args).err(),
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
            solver.solve(&mut y0, 0.0, 9.0, Some(1.0), &mut args).err(),
            Some("an element of the vector is either infinite or NaN")
        );
        let params = Params::new(Method::MdEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        assert_eq!(
            solver.solve(&mut y0, 0.0, 1.0, None, &mut args).err(),
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
            solver.solve(&mut y0, 0.0, 1.0, None, &mut args).err(),
            Some("variable stepping did not converge")
        );
    }

    #[test]
    fn out_initialize_errors_are_captured() {
        let (system, _, mut y0, mut args, _) = Samples::simple_equation_constant();
        let params = Params::new(Method::DoPri5);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        solver
            .enable_output()
            .set_dense_x_out(&[0.0, 1.0])
            .unwrap()
            .set_dense_recording(&[0]);
        assert_eq!(
            solver.solve(&mut y0, 0.0, 1.0, None, &mut args).err(),
            Some("the first interior x_out for dense output must be > x0")
        );
        solver.enable_output().set_dense_x_out(&[0.1, 1.0]).unwrap();
        assert_eq!(
            solver.solve(&mut y0, 0.0, 1.0, None, &mut args).err(),
            Some("the last interior x_out for dense output must be < x1")
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
        solver.solve(&mut y, x0, x1, None, &mut args).unwrap();
        vec_approx_eq(&y, &[1.0], 1e-15);
    }

    #[test]
    fn solve_completes_after_a_single_step() {
        // since the problem is linear, this stepsize will lead to a jump from x=0.0 to 1.0
        let (system, _, mut y0, mut args, _) = Samples::simple_equation_constant();
        let mut params = Params::new(Method::DoPri5);
        params.step.h_ini = 20.0; // will be truncated to 1 yielding a single step
        let mut solver = OdeSolver::new(params, &system).unwrap();
        solver.solve(&mut y0, 0.0, 1.0, None, &mut args).unwrap();
        assert_eq!(solver.work.stats.n_accepted, 1);
        vec_approx_eq(&y0, &[1.0], 1e-15);
    }

    #[test]
    fn solve_with_variable_steps_works() {
        let (system, _, mut y0, mut args, _) = Samples::simple_equation_constant();
        let mut params = Params::new(Method::MdEuler);
        params.step.h_ini = 0.1;
        let mut solver = OdeSolver::new(params, &system).unwrap();
        solver.solve(&mut y0, 0.0, 0.3, None, &mut args).unwrap();
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
    fn solve_with_out_dense_captures_errors() {
        let (system, x0, mut y0, mut args, _) = Samples::simple_equation_constant();
        let x1 = 1.0;
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();
        solver.enable_output().set_dense_recording(&[0]);
        assert_eq!(
            solver.solve(&mut y0, x0, x1, None, &mut args).err(),
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
        let path_key = "/tmp/russell_ode/test_solve_step_output_works";
        solver
            .enable_output()
            .set_yx_correct(y_fn_x)
            .set_step_file_writing(path_key)
            .set_step_recording(&[0])
            .set_step_callback(|stats, h, x, y, _args| {
                assert_eq!(h, 0.2);
                approx_eq(x, (stats.n_accepted as f64) * h, 1e-15);
                approx_eq(y[0], (stats.n_accepted as f64) * h, 1e-15);
                Ok(false)
            });

        // solve
        let h_equal = Some(0.2);
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, h_equal, &mut args).unwrap();

        // check
        vec_approx_eq(&y, &[0.4], 1e-15);
        array_approx_eq(&solver.out_step_h(), &[0.2, 0.2, 0.2], 1e-15);
        array_approx_eq(&solver.out_step_x(), &[0.0, 0.2, 0.4], 1e-15);
        array_approx_eq(&solver.out_step_y(0), &[0.0, 0.2, 0.4], 1e-15);
        array_approx_eq(&solver.out_step_global_error(), &[0.0, 0.0, 0.0], 1e-15);

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
        let cb = |_stats: &Stats, _h: f64, _x: f64, _y: &Vector, _args: &mut NoArgs| -> Result<bool, StrError> {
            Err("unreachable")
        };
        assert_eq!(cb(&solver.stats(), 0.0, 0.0, &y0, &mut args).err(), Some("unreachable"));

        // run again and stop earlier
        solver.enable_output().set_step_callback(|stats, _h, _x, _y, _args| {
            if stats.n_accepted > 0 {
                Ok(true) // stop
            } else {
                Ok(false) // do not stop
            }
        });
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, None, &mut args).unwrap();
        assert!(y[0] > 0.0 && y[0] < 0.4);
    }

    #[test]
    fn solve_with_step_captures_errors() {
        // system and solver
        let (system, _, y0, mut args, _) = Samples::simple_equation_constant();
        let params = Params::new(Method::FwEuler);
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // output
        solver.enable_output().set_step_recording(&[0]);

        // solve and stop due to error on the first accepted step
        solver
            .enable_output()
            .set_step_callback(|_stats, _h, _x, _y, _args| Err("stop with error (first accepted step)"));
        let mut y = y0.clone();
        assert_eq!(
            solver.solve(&mut y, 0.0, 0.4, None, &mut args).err(),
            Some("stop with error (first accepted step)")
        );

        // solve again and stop due to error on the next steps
        solver.enable_output().set_step_callback(|stats, _h, _x, _y, _args| {
            if stats.n_accepted > 0 {
                Err("stop with error (subsequent steps)")
            } else {
                Ok(false) // do not stop
            }
        });
        let mut y = y0.clone();
        assert_eq!(
            solver.solve(&mut y, 0.0, 0.4, None, &mut args).err(),
            Some("stop with error (subsequent steps)")
        );
    }

    #[test]
    fn solve_with_dense_output_h_out_works_1() {
        // system and solver
        let (system, _, _, mut args, _) = Samples::simple_equation_constant();
        let params = Params::new(Method::DoPri5);
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // output
        solver
            .enable_output()
            .set_dense_h_out(0.2)
            .unwrap()
            .set_dense_recording(&[0]);

        // solve
        let h_equal = Some(0.201); // this will make x_out < x - h with x_out = 0.4
        let x0 = 0.0;
        let x1 = 1.0;
        let mut y = Vector::from(&[x0]);
        solver.solve(&mut y, x0, x1, h_equal, &mut args).unwrap();

        // check
        vec_approx_eq(&y, &[x1], 1e-15);
        let correct = &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        array_approx_eq(&solver.out_dense_x(), correct, 1e-15);
        array_approx_eq(&solver.out_dense_y(0), correct, 1e-15);
    }

    #[test]
    fn solve_with_dense_output_h_out_works_2() {
        // system and solver
        let (system, _, y0, mut args, y_fn_x) = Samples::simple_equation_constant();
        let params = Params::new(Method::DoPri5);
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // output
        const H_OUT: f64 = 0.1;
        let path_key = "/tmp/russell_ode/test_solve_dense_output_h_out_works";
        solver
            .enable_output()
            .set_yx_correct(y_fn_x)
            .set_dense_h_out(H_OUT)
            .unwrap()
            .set_dense_file_writing(path_key)
            .unwrap()
            .set_dense_recording(&[0])
            .set_dense_callback(|_stats, h, x, y, _args| {
                assert_eq!(h, 0.2);
                approx_eq(y[0], x, 1e-15);
                Ok(false)
            });

        // solve
        let h_equal = Some(0.2);
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, h_equal, &mut args).unwrap();

        // check
        vec_approx_eq(&y, &[0.4], 1e-15);
        array_approx_eq(&solver.out_dense_x(), &[0.0, 0.1, 0.2, 0.3, 0.4], 1e-15);
        array_approx_eq(&solver.out_dense_y(0), &[0.0, 0.1, 0.2, 0.3, 0.4], 1e-15);

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
        let cb = |_stats: &Stats, _h: f64, _x: f64, _y: &Vector, _args: &mut NoArgs| -> Result<bool, StrError> {
            Err("unreachable")
        };
        assert_eq!(cb(&solver.stats(), 0.0, 0.0, &y0, &mut args).err(), Some("unreachable"));

        // run again but stop at the first output
        solver.enable_output().set_dense_callback(|_stats, _h, _x, _y, _args| {
            Ok(true) // stop
        });
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, None, &mut args).unwrap();
        assert_eq!(solver.work.stats.n_accepted, 0);
        assert_eq!(y[0], 0.0);

        // run again and stop earlier
        solver.enable_output().set_dense_callback(|stats, _h, _x, _y, _args| {
            if stats.n_accepted > 0 {
                Ok(true) // stop
            } else {
                Ok(false) // do not stop
            }
        });
        // ... equal steps
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, Some(0.2), &mut args).unwrap();
        assert!(y[0] > 0.0 && y[0] < 0.4);
        // ... variable steps
        let mut y = y0.clone();
        solver.solve(&mut y, 0.0, 0.4, None, &mut args).unwrap();
        assert!(y[0] > 0.0 && y[0] < 0.4);

        // run again and stop due to error
        // ... first step
        solver
            .enable_output()
            .set_dense_callback(|_stats, _h, _x, _y, _args| Err("stop with error"));
        let mut y = y0.clone();
        assert_eq!(
            solver.solve(&mut y, 0.0, 0.4, None, &mut args).err(),
            Some("stop with error")
        );
        // ... next steps
        solver.enable_output().set_dense_callback(|stats, _h, _x, _y, _args| {
            if stats.n_accepted > 0 {
                Err("stop with error")
            } else {
                Ok(false) // do not stop
            }
        });
        let mut y = y0.clone();
        assert_eq!(
            solver.solve(&mut y, 0.0, 0.4, None, &mut args).err(),
            Some("stop with error")
        );
    }

    #[test]
    fn solve_with_dense_output_x_out_works() {
        // system and solver
        let (system, _, _, mut args, _) = Samples::simple_equation_constant();
        let params = Params::new(Method::DoPri5);
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // interior output stations and selected y component
        let interior_x_out = &[-0.5, 0.0, 0.5];
        let selected_y = &[0];

        // output
        solver
            .enable_output()
            .set_dense_x_out(interior_x_out)
            .unwrap()
            .set_dense_recording(selected_y);

        // solve
        let h_equal = Some(0.2);
        let x0 = -1.0;
        let x1 = 1.0;
        let mut y = Vector::from(&[x0]);
        solver.solve(&mut y, x0, x1, h_equal, &mut args).unwrap();

        // check
        vec_approx_eq(&y, &[x1], 1e-15);
        let correct = &[-1.0, -0.5, 0.0, 0.5, 1.0];
        assert_eq!(solver.out_dense_x(), correct);
        array_approx_eq(solver.out_dense_y(0), correct, 1e-15);
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
        let system = System::new(ndim, |f: &mut Vector, _x: f64, _y: &Vector, args: &mut Args| {
            if args.f_count == args.f_barrier {
                return Err("f: artificial error");
            }
            f[0] = 1.0;
            args.f_count += 1;
            Ok(())
        });

        // initial values and final x
        let x0 = 0.0;
        let x1 = 0.2;
        let mut y = Vector::from(&[0.0]);

        // parameters and solver
        let mut params = Params::new(Method::DoPri8);
        params.step.h_ini = 0.2;
        let mut solver = OdeSolver::new(params, &system).unwrap();

        // output
        solver.enable_output().set_dense_h_out(0.1).unwrap().set_dense_callback(
            |_stats, _h, _x, _y, args: &mut Args| {
                if args.out_count == args.out_barrier {
                    return Err("out: artificial error");
                }
                args.out_count += 1;
                Ok(false) // do not stop
            },
        );

        // equal steps -----------------------------------------------------------

        // first error @ actual.step
        assert_eq!(
            solver.solve(&mut y, x0, x1, Some(0.2), &mut args).err(),
            Some("f: artificial error")
        );

        // second error @ actual.accept
        // There are only two places in ERK where an error in 'accept' may occur:
        // (the other methods/solvers do not return Err in their 'accept')
        // 1. when saving data for dense output (only DoPri8)
        // 2. in computations related to the stiffness detection
        args.f_barrier += 12; // nstage = 12 (need to skip the next call to 'step')
        assert_eq!(
            solver.solve(&mut y, x0, x1, Some(0.2), &mut args).err(),
            Some("f: artificial error")
        );

        // third error @ the second output
        args.f_barrier += 2 * 12; // skip next calls to 'step'
        assert_eq!(
            solver.solve(&mut y, x0, x1, Some(0.2), &mut args).err(),
            Some("out: artificial error")
        );

        // fourth error @ the last output
        args.f_barrier += 2 * 12; // skip next calls to 'step'
        args.out_barrier += 2; // skip first and second output
        assert_eq!(
            solver.solve(&mut y, x0, x1, Some(0.2), &mut args).err(),
            Some("out: artificial error")
        );

        // variable steps --------------------------------------------------------

        // first error @ actual.step
        args.f_count = 0;
        args.f_barrier = 0;
        args.out_count = 0;
        args.out_barrier = 2; // first and second outputs
        assert_eq!(
            solver.solve(&mut y, x0, x1, None, &mut args).err(),
            Some("f: artificial error")
        );

        // second error @ actual.accept
        // There are only two places in ERK where an error in 'accept' may occur:
        // (the other methods/solvers do not return Err in their 'accept')
        // 1. when saving data for dense output (only DoPri8)
        // 2. in computations related to the stiffness detection
        args.f_barrier += 12; // nstage = 12 (need to skip the next call to 'step')
        assert_eq!(
            solver.solve(&mut y, x0, x1, None, &mut args).err(),
            Some("f: artificial error")
        );

        // error @ last output
        args.f_count = 0;
        args.f_barrier = 15 + 1; // 12 stages, 3 dense output, 1 safety
        args.out_count = 0;
        args.out_barrier = 2; // first and second outputs
        assert_eq!(
            solver.solve(&mut y, x0, x1, None, &mut args).err(),
            Some("out: artificial error")
        );
    }
}
