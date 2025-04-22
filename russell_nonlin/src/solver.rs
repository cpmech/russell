use super::{Config, Method, SolverTrait, State, Stop, System};
use super::{Output, SolverArclength, SolverNatural, Stats, Workspace};
use crate::StrError;
use russell_lab::vec_all_finite;

/// Default number of steps
pub const N_EQUAL_STEPS: usize = 10;

pub struct Solver<'a, A> {
    /// Configuration options
    config: Config,

    /// Dimension of the ODE system
    ndim: usize,

    /// Holds a pointer to the actual ODE system solver
    actual: Box<dyn SolverTrait<A> + 'a>,

    /// Holds statistics, benchmarking and "work" variables
    work: Workspace<'a>,

    /// Assists in generating the output of results
    output: Output<'a, A>,

    /// Indicates whether the output is enabled or not
    output_enabled: bool,
}

impl<'a, A> Solver<'a, A> {
    /// Allocates a new instance
    pub fn new(config: Config, system: System<'a, A>) -> Result<Self, StrError>
    where
        A: 'a,
    {
        config.validate()?;
        let ndim = system.ndim;
        let work = Workspace::new(&config, &system);
        let actual: Box<dyn SolverTrait<A>> = match config.method {
            Method::Arclength => Box::new(SolverArclength::new(config, system.clone())),
            Method::Natural => Box::new(SolverNatural::new(config, system.clone())),
        };
        Ok(Solver {
            config,
            ndim,
            actual,
            work,
            output: Output::new(),
            output_enabled: false,
        })
    }

    /// Returns some benchmarking data
    pub fn stats(&self) -> &Stats {
        &self.work.stats
    }

    /// Returns the error messages
    pub fn errors(&self) -> Vec<String> {
        self.work.errors()
    }

    /// Solves the nonlinear system
    ///
    /// Solves:
    ///
    /// ```text
    /// Natural:   G(u, λ) = 0
    /// Arclength: G(u(s), λ(s)) = 0
    /// ```
    ///
    /// # Input
    ///
    /// * `state` -- the initial state (u0,λ0) with a non-singular Gu = du/dλ (Jacobian) matrix.
    ///    The state also includes an option to compute or reuse the initial tangent vector.
    ///    The state will be updated with the new solution (u,λ) util a stop criterion is reached.
    /// * `stop` -- stop criterion (e.g, either a final λ value or a number of steps)
    /// * `h_equal` -- use a constant stepsize; otherwise, variable step sizes are automatically calculated (auto mode).
    pub fn solve(&mut self, state: &mut State, stop: Stop, h_equal: Option<f64>, args: &mut A) -> Result<(), StrError> {
        // check data
        if state.u.dim() != self.ndim {
            return Err("u.dim() must be equal to ndim");
        }
        match stop {
            Stop::Lambda(l1) => {
                if l1 <= state.l {
                    return Err("Stopping criterion error: l1 must be greater than l0");
                }
            }
            Stop::Steps(n) => {
                if n < 1 {
                    return Err("Stopping criterion error: number of steps must be greater than 0");
                }
            }
        }

        // determine auto-stepping (substepping) flag and initial stepsize
        let (auto, h_ini) = match h_equal {
            Some(h_eq) => {
                // equal stepsize (not automatic substepping)
                if h_eq < 10.0 * f64::EPSILON {
                    return Err("h_equal must be ≥ 10.0 * f64::EPSILON");
                }
                let h0 = match stop {
                    Stop::Lambda(l1) => {
                        let n = f64::ceil((l1 - state.l) / h_eq) as usize;
                        (l1 - state.l) / (n as f64)
                    }
                    Stop::Steps(_) => h_eq,
                };
                (false, h0)
            }
            None => {
                // automatic substepping
                let h0 = match stop {
                    Stop::Lambda(l1) => f64::min(self.config.h_ini, l1 - state.l),
                    Stop::Steps(_) => self.config.h_ini,
                };
                (true, h0)
            }
        };
        assert!(h_ini > 0.0);
        state.h = h_ini;

        // reset variables
        self.work.reset(h_ini, self.config.rel_error_prev_min, auto);

        // perform initialization such as computing the first tangent vector in pseudo-arclength
        self.actual.initialize(&mut self.work, &state, args)?;

        // first output
        if self.output_enabled {
            self.output.initialize();
            let terminate = self.output.execute(&self.work, &state, args)?;
            if terminate {
                return Ok(());
            }
        }

        // message
        self.work.log.header();

        // solve with equal stepsize
        if !auto {
            let nstep = match stop {
                Stop::Lambda(l1) => f64::ceil((l1 - state.l) / h_ini) as usize,
                Stop::Steps(n) => n,
            };
            for _ in 0..nstep {
                // log
                self.work.stats.sw_step.reset();
                self.work.log.step(&state);

                // step
                self.work.stats.n_steps += 1;
                self.actual.step(&mut self.work, state, args)?;
                if self.work.err.failed() {
                    break;
                }

                // update u and λ
                self.work.stats.n_accepted += 1;
                self.actual.accept(&mut self.work, state);

                // check for anomalies
                vec_all_finite(&state.u, self.config.verbose)?;

                // output
                if self.output_enabled {
                    let terminate = self.output.execute(&self.work, &state, args)?;
                    if terminate {
                        self.work.stats.stop_sw_step();
                        break;
                    }
                }
                self.work.stats.stop_sw_step();
            }

            // last output
            if self.output_enabled {
                self.output.last()?;
            }

            // print last message and footer
            self.work.log.step(&state);
            self.work.stats.stop_sw_total();
            self.work.log.footer(&self.work);

            // handle errors
            if self.work.err.failed() {
                return Err("failed to solve the nonlinear problem with equal stepsize");
            } else {
                return Ok(());
            }
        }

        // variable steps: control variables
        let mut success = false;
        let mut last_step = false;

        // variable stepping loop
        for i in 0..self.config.n_step_max {
            // log
            self.work.stats.sw_step.reset();
            self.work.log.step(&state);

            // check final stepsize and stopping criterion
            let h_final = match stop {
                Stop::Lambda(l1) => {
                    let dl = l1 - state.l;
                    if dl <= 10.0 * f64::EPSILON {
                        success = true;
                        self.work.stats.stop_sw_step();
                        break;
                    }
                    dl
                }
                Stop::Steps(n) => {
                    if i >= n && !self.work.follows_reject_step {
                        success = true;
                        self.work.stats.stop_sw_step();
                        break;
                    }
                    self.work.h_new
                }
            };

            // check number of continued rejections
            self.work.n_continued_rejection += 1;
            if self.work.n_continued_rejection >= self.config.n_cont_reject_allowed {
                self.work.stop_continued_rejection = true;
                break;
            }

            // update and check the stepsize
            state.h = f64::min(self.work.h_new, h_final);
            if state.h <= 10.0 * f64::EPSILON {
                self.work.stop_small_stepsize = true;
                break;
            }

            // perform the step calculations
            self.work.stats.n_steps += 1;
            self.actual.step(&mut self.work, state, args)?;

            // handle diverging iterations
            if self.work.iterations_failed {
                self.work.iterations_failed = false;
                self.work.follows_reject_step = true;
                last_step = false;
                self.work.h_new = state.h * self.work.h_multiplier_failure;
                continue;
            }

            // accept step
            if self.work.rel_error < 1.0 {
                // update u and λ
                self.work.stats.n_accepted += 1;
                self.actual.accept(&mut self.work, state);

                // check for anomalies
                vec_all_finite(&state.u, self.config.verbose)?;

                // do not allow h to grow if previous step was a reject
                if self.work.follows_reject_step {
                    self.work.h_new = f64::min(self.work.h_new, state.h);
                }
                self.work.n_continued_rejection = 0;
                self.work.follows_reject_step = false;

                // save previous stepsize, relative error, and accepted/suggested stepsize
                self.work.h_prev = state.h;
                self.work.rel_error_prev = f64::max(self.config.rel_error_prev_min, self.work.rel_error);
                self.work.stats.h_accepted = self.work.h_new;

                // output
                if self.output_enabled {
                    let terminate = self.output.execute(&self.work, &state, args)?;
                    if terminate {
                        success = true;
                        self.work.stats.stop_sw_step();
                        break;
                    }
                }

                // stop calculations if last step
                if last_step {
                    success = true;
                    self.work.stats.stop_sw_step();
                    break;
                }

                // check if the last step is approaching
                match stop {
                    Stop::Lambda(l1) => last_step = state.l + self.work.h_new >= l1,
                    Stop::Steps(n) => last_step = i + 1 >= n,
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
                if self.work.stats.n_accepted == 0 && self.config.m_first_reject > 0.0 {
                    self.work.h_new = state.h * self.config.m_first_reject;
                } else {
                    self.actual.reject(&mut self.work, state.h, args);
                }
            }
        }

        // last output
        if self.output_enabled {
            self.output.last()?;
        }

        // print footer
        self.work.stats.stop_sw_total();
        self.work.log.footer(&self.work);

        // handle errors
        if success {
            Ok(())
        } else {
            return Err("failed to solve the nonlinear problem with automatic stepsize");
        }
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
        &self.output.h
    }

    /// Returns an access to the output during accepted steps: λ values
    pub fn out_step_l(&self) -> &Vec<f64> {
        &self.output.l
    }

    /// Returns an access to the output during accepted steps: u values
    ///
    /// # Panics
    ///
    /// A panic will occur if `m` is out of range
    pub fn out_step_u(&self, m: usize) -> &Vec<f64> {
        &self.output.u.get(&m).unwrap()
    }

    /// Returns an access to the output during accepted steps: global error
    pub fn out_step_global_error(&self) -> &Vec<f64> {
        &self.output.error
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Solver;
    use crate::{Config, Method, Samples, State, Stop};
    use russell_lab::{vec_approx_eq, Vector};

    #[test]
    fn new_captures_errors() {
        let (system, _, _, _) = Samples::two_eq_ref();
        let mut config = Config::new(Method::Natural);
        config.m_max = 0.0; // wrong
        assert_eq!(
            Solver::new(config, system).err(),
            Some("requirement: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
    }

    #[test]
    fn solve_captures_errors() {
        let (system, _, _, mut args) = Samples::two_eq_ref();
        let mut state = State::new(system.ndim + 1, false); // wrong dim
        let ndim = system.ndim;
        let config = Config::new(Method::Natural);
        let mut solver = Solver::new(config, system).unwrap();
        assert_eq!(
            solver.solve(&mut state, Stop::Lambda(1.0), None, &mut args).err(),
            Some("u.dim() must be equal to ndim")
        );
        state.u = Vector::new(ndim); // fix dim
        assert_eq!(
            solver.solve(&mut state, Stop::Lambda(0.0), None, &mut args).err(),
            Some("Stopping criterion error: l1 must be greater than l0")
        );
        let h_equal = Some(f64::EPSILON); // will cause an error
        assert_eq!(
            solver.solve(&mut state, Stop::Lambda(1.0), h_equal, &mut args).err(),
            Some("h_equal must be ≥ 10.0 * f64::EPSILON")
        );
    }

    #[test]
    fn lack_of_convergence_is_captured() {
        let (system, mut state, _u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new(Method::Natural);
        config.n_step_max = 1; // will make the solver to fail (too few steps)
        let mut solver = Solver::new(config, system).unwrap();
        assert_eq!(
            solver.solve(&mut state, Stop::Lambda(1.0), None, &mut args).err(),
            Some("failed to solve the nonlinear problem with automatic stepsize")
        );
    }

    #[test]
    fn solve_with_one_step_works_fixed() {
        let (system, mut state, u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new(Method::Natural);
        config.set_verbose(true, true, true).set_tol_delta(1e-12, 1e-10);
        let mut solver = Solver::new(config, system).unwrap();
        let stop = Stop::Steps(1); // just one step
        solver.solve(&mut state, stop, Some(1.0), &mut args).unwrap();
        vec_approx_eq(&state.u, &u_ref, 1e-15);
        let stats = solver.stats();
        assert_eq!(stats.n_function, 7);
        assert_eq!(stats.n_jacobian, 6);
        assert_eq!(stats.n_factor, 6);
        assert_eq!(stats.n_lin_sol, 6);
        assert_eq!(stats.n_steps, 1);
        assert_eq!(stats.n_accepted, 1);
        assert_eq!(stats.n_rejected, 0);
        assert_eq!(stats.n_iterations_max, 7);
        assert_eq!(stats.n_iterations_total, 7);
    }

    #[test]
    fn solve_with_one_step_works_auto() {
        let (system, mut state, u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new(Method::Natural);
        config.set_verbose(true, true, true).set_tol_delta(1e-12, 1e-10);
        let mut solver = Solver::new(config, system).unwrap();
        let stop = Stop::Steps(1); // just one step
        solver.solve(&mut state, stop, None, &mut args).unwrap();
        vec_approx_eq(&state.u, &u_ref, 1e-15);
        let stats = solver.stats();
        assert_eq!(stats.n_function, 7);
        assert_eq!(stats.n_jacobian, 6);
        assert_eq!(stats.n_factor, 6);
        assert_eq!(stats.n_lin_sol, 6);
        assert_eq!(stats.n_steps, 1);
        assert_eq!(stats.n_accepted, 1);
        assert_eq!(stats.n_rejected, 0);
        assert_eq!(stats.n_iterations_max, 7);
        assert_eq!(stats.n_iterations_total, 7);
    }
}
