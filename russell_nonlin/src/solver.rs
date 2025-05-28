use super::{AutoStep, Config, Direction, Method, SolverTrait, State, Status, Stop, System};
use super::{Output, SolverArclength, SolverNatural, Stats, Workspace, CONFIG_H_MIN};
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
    /// * `args` -- extra arguments to be passed to the system functions
    /// * `state` -- the initial state `(u₀, λ₀)` with a non-singular `Gu₀ = ∂u/∂λ|₀` (Jacobian) matrix.
    ///    The state will be updated with the new solution (u, λ) util a stop criterion is reached.
    /// * `dir` -- the direction to follow on the solution branch (pseudo-arclength method).
    /// * `stop` -- stop criterion (e.g, either a final λ value or a number of steps)
    /// * `auto` -- defines the stepsize control method (variable stepsize control or fixed stepsize)
    /// * `output` -- output object to write results files, record the results, or execute a callback function
    ///
    /// # Returns
    ///
    /// Returns the status code.
    pub fn solve<'b>(
        &mut self,
        args: &mut A,
        state: &mut State,
        dir: Direction,
        stop: Stop,
        auto: AutoStep,
        mut output: Option<&mut Output<'b, A>>,
    ) -> Result<Status, StrError> {
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

        // reset stats and flags
        self.work.reset_stats_and_flags(auto.yes());

        // perform initialization (compute initial stepsize and tangent vector)
        self.actual.initialize(&mut self.work, state, dir, stop, auto, args)?;

        // first output
        if let Some(out) = output.as_deref_mut() {
            let terminate = out.execute(&self.work, &state, args)?;
            if terminate {
                return Ok(Status::Stopped);
            }
        }

        // message
        self.work.log.header();

        // perform continuation
        let mut continuation_completed = false;
        if auto.no() {
            // fixed/equal stepsize
            self.work.stats.h_accepted = self.work.h;
            for i in 0..self.config.n_step_max {
                // log
                self.work.stats.sw_step.reset();
                self.work.log.step(self.work.h, &state);

                // step
                self.work.stats.n_steps += 1;
                self.actual.step(&mut self.work, state, args)?;
                if self.work.err.failed() {
                    break;
                }

                // update u and λ
                self.work.stats.n_accepted += 1;
                self.actual.accept(&mut self.work, state, args)?;

                // check for anomalies
                vec_all_finite(&state.u, self.config.verbose)?;

                // output
                if let Some(out) = output.as_deref_mut() {
                    let terminate = out.execute(&self.work, &state, args)?;
                    if terminate {
                        self.work.stats.stop_sw_step();
                        break;
                    }
                }

                // exit point
                self.work.stats.stop_sw_step();
                if match stop {
                    Stop::Lambda(l1) => state.l > l1 || f64::abs(state.l - l1) < CONFIG_H_MIN,
                    Stop::Steps(n) => (i + 1) == n,
                } {
                    continuation_completed = true;
                    break;
                }
            }
        } else {
            // variable stepsize
            for i in 0..self.config.n_step_max {
                // log
                self.work.stats.sw_step.reset();
                self.work.log.step(self.work.h, &state);

                // handle small stepsize
                if self.work.h < CONFIG_H_MIN {
                    self.work.stopped_due_to_small_stepsize = true;
                    break;
                }

                // step
                self.work.stats.n_steps += 1;
                self.actual.step(&mut self.work, state, args)?;

                // handle iteration failure
                if self.work.err.failed() {
                    self.work.n_continued_failure += 1;
                    self.work.follows_failure = true;
                }
                if self.work.n_continued_failure >= 3 {
                    self.work.stopped_due_to_continued_failure = true;
                    break;
                }

                // handle rejections (due to large curvatures, etc.)
                if self.work.n_continued_rejection >= 3 {
                    self.work.stopped_due_to_continued_rejection = true;
                    break;
                }

                // handle lambda target
                if let Some(l1) = stop.lambda_target() {
                    if self.work.l > l1 {
                        // redo with target stepsize
                        self.work.stats.n_steps += 1;
                        self.work.stats.sw_step.reset();
                        self.work.log.step(self.work.h, &state);
                        self.actual.target_stepsize(&mut self.work, state, l1);
                        self.actual.step(&mut self.work, state, args)?;
                    }
                }

                // accept step
                if self.work.acceptable {
                    // update u and λ
                    self.work.stats.n_accepted += 1;
                    self.actual.accept(&mut self.work, state, args)?;

                    // check for anomalies
                    vec_all_finite(&state.u, self.config.verbose)?;

                    // fix stepsize estimate
                    if self.work.follows_failure || self.work.follows_rejection {
                        // avoid stepsize growth on failure/rejection
                        self.work.h_estimate = f64::min(self.work.h_estimate, self.work.h);
                    }
                    self.work.stats.h_accepted = self.work.h_estimate;

                    // reset flags
                    if self.work.follows_failure {
                        self.work.err.clear_error_flags();
                    }
                    self.work.n_continued_failure = 0;
                    self.work.n_continued_rejection = 0;
                    self.work.follows_failure = false;
                    self.work.follows_rejection = false;

                    // output
                    if let Some(out) = output.as_deref_mut() {
                        let terminate = out.execute(&self.work, &state, args)?;
                        if terminate {
                            self.work.stats.stop_sw_step();
                            break;
                        }
                    }

                    // exit point
                    self.work.stats.stop_sw_step();
                    if match stop {
                        Stop::Lambda(l1) => state.l > l1 || f64::abs(state.l - l1) < CONFIG_H_MIN,
                        Stop::Steps(n) => (i + 1) == n,
                    } {
                        continuation_completed = true;
                        break;
                    }

                // reject step
                } else {
                    // set flags
                    self.work.stats.n_rejected += 1;
                    self.work.follows_rejection = true;

                    // perform the reject operations (e.g., restore external vars) and recompute stepsize
                    self.actual.reject(&mut self.work, args);
                }

                // adjust stepsize
                self.work.h = self.work.h_estimate;
            }
        }

        // last output
        if let Some(out) = output.as_deref_mut() {
            out.last()?;
        }

        // print last message and footer
        self.work.log.step(self.work.h, &state);
        self.work.stats.stop_sw_total();
        self.work.log.footer(&self.work);

        // done
        if continuation_completed {
            return Ok(Status::Success);
        } else {
            return Ok(Status::Failure);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Solver;
    use crate::{AutoStep, Config, Direction, Method, Samples, State, Status, Stop};
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
        let mut state = State::new(system.ndim + 1); // wrong dim
        let ndim = system.ndim;
        let config = Config::new(Method::Natural);
        let mut solver = Solver::new(config, system).unwrap();
        assert_eq!(
            solver
                .solve(
                    &mut args,
                    &mut state,
                    Direction::Pos,
                    Stop::Lambda(1.0),
                    AutoStep::Yes,
                    None
                )
                .err(),
            Some("u.dim() must be equal to ndim")
        );
        state.u = Vector::new(ndim); // fix dim
        assert_eq!(
            solver
                .solve(
                    &mut args,
                    &mut state,
                    Direction::Pos,
                    Stop::Lambda(0.0),
                    AutoStep::Yes,
                    None,
                )
                .err(),
            Some("Stopping criterion error: l1 must be greater than l0")
        );
        assert_eq!(
            solver
                .solve(
                    &mut args,
                    &mut state,
                    Direction::Pos,
                    Stop::Lambda(1.0),
                    AutoStep::No(f64::EPSILON), // will cause an error
                    None,
                )
                .err(),
            Some("h must be ≥ 10.0 * f64::EPSILON for fixed stepsize")
        );
    }

    #[test]
    fn lack_of_convergence_is_captured() {
        let (system, mut state, _u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new(Method::Natural);
        config.n_step_max = 1; // will make the solver to fail (too few steps)
        let mut solver = Solver::new(config, system).unwrap();
        assert_eq!(
            solver
                .solve(
                    &mut args,
                    &mut state,
                    Direction::Pos,
                    Stop::Lambda(1.0),
                    AutoStep::Yes,
                    None,
                )
                .unwrap(),
            Status::Failure
        );
    }

    #[test]
    fn solve_with_one_step_works_fixed() {
        let (system, mut state, u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new(Method::Natural);
        config.set_verbose(true, true, true).set_tol_delta(1e-12, 1e-10);
        let mut solver = Solver::new(config, system).unwrap();
        solver
            .solve(
                &mut args,
                &mut state,
                Direction::Pos,
                Stop::Steps(1),
                AutoStep::No(1.0),
                None,
            )
            .unwrap();
        vec_approx_eq(&state.u, &u_ref, 1e-15);
        let stats = solver.stats();
        assert_eq!(stats.n_function, 7);
        assert_eq!(stats.n_jacobian, 6);
        assert_eq!(stats.n_factor, 6);
        assert_eq!(stats.n_lin_sol, 6);
        assert_eq!(stats.n_steps, 1);
        assert_eq!(stats.n_accepted, 1);
        assert_eq!(stats.n_rejected, 0);
        assert_eq!(stats.n_iteration_max, 7);
        assert_eq!(stats.n_iteration_total, 7);
    }

    #[test]
    fn solve_with_one_step_works_auto() {
        let (system, mut state, u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new(Method::Natural);
        config.set_verbose(true, true, true).set_tol_delta(1e-12, 1e-10);
        let mut solver = Solver::new(config, system).unwrap();
        solver
            .solve(
                &mut args,
                &mut state,
                Direction::Pos,
                Stop::Steps(1),
                AutoStep::Yes,
                None,
            )
            .unwrap();
        vec_approx_eq(&state.u, &u_ref, 1e-15);
        let stats = solver.stats();
        assert_eq!(stats.n_function, 7);
        assert_eq!(stats.n_jacobian, 6);
        assert_eq!(stats.n_factor, 6);
        assert_eq!(stats.n_lin_sol, 6);
        assert_eq!(stats.n_steps, 1);
        assert_eq!(stats.n_accepted, 1);
        assert_eq!(stats.n_rejected, 0);
        assert_eq!(stats.n_iteration_max, 7);
        assert_eq!(stats.n_iteration_total, 7);
    }
}
