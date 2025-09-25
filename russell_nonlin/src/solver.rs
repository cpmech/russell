use super::{AutoStep, Config, Direction, Method, SolverTrait, State, Stop, System};
use super::{Output, SolverArclength, SolverNatural, Stats, Status, Workspace, CONFIG_H_MIN};
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

    // variables for the stepsize adaptation
    rerr_prev: f64,
    rerr_anc: f64,
    h_prev: f64,
    h_anc: f64,
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
            Method::Arclength => Box::new(SolverArclength::new(config.clone(), system.clone())?),
            Method::Natural => Box::new(SolverNatural::new(config.clone(), system.clone())),
        };
        Ok(Solver {
            config,
            ndim,
            actual,
            work,
            rerr_prev: 0.0,
            rerr_anc: 0.0,
            h_prev: 0.0,
            h_anc: 0.0,
        })
    }

    /// Returns some benchmarking data
    pub fn get_stats(&self) -> &Stats {
        &self.work.stats
    }

    /// Returns λ and the first two components of u (if available), calculated by the predictor step (for debugging)
    ///
    /// Returns `(l_predictor, u0_predictor, u1_predictor)`.
    pub fn get_debug_predictor_values(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        if let Some((l, u0, u1)) = &self.work.predictor_values_debug {
            (l.clone(), u0.clone(), u1.clone())
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        }
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
        // validate input
        if state.u.dim() != self.ndim {
            return Err("u.dim() must be equal to ndim");
        }
        stop.validate(state)?;
        auto.validate()?;

        // reset stats and flags
        self.work.reset_stats_and_flags(auto.yes());

        // perform initialization (compute initial stepsize and tangent vector)
        self.actual.initialize(&mut self.work, state, dir, stop, auto, args)?;

        // first output
        if let Some(out) = output.as_deref_mut() {
            let stop_gracefully = out.execute(&self.work, &state, args)?;
            if stop_gracefully {
                return Ok(Status::Success);
            }
        }

        // message
        self.work.log.header();

        // default status
        let mut status = Status::Success;

        // perform continuation
        if auto.no() {
            // fixed/equal stepsize
            self.work.stats.h_accepted = self.work.h;
            for i in 0..self.config.n_step_max {
                // log
                self.work.stats.sw_step.reset();
                self.work.log.step(self.work.h, &state);

                // step
                self.work.stats.n_steps += 1;
                status = self.actual.step(&mut self.work, state, stop, args)?;

                // handle failures
                if status.failure() {
                    break;
                }

                // update u and λ
                self.work.stats.n_accepted += 1;
                self.actual.accept(&mut self.work, state, args)?;

                // check for anomalies
                vec_all_finite(&state.u, self.config.verbose)?;

                // output
                if let Some(out) = output.as_deref_mut() {
                    let stop_gracefully = out.execute(&self.work, &state, args)?;
                    if stop_gracefully {
                        self.work.stats.stop_sw_step();
                        break;
                    }
                }

                // exit point
                self.work.stats.stop_sw_step();
                if stop.now(i, state) {
                    break;
                }
            }
        } else {
            // variable stepsize
            for i in 0..self.config.n_step_max {
                // log
                self.work.stats.sw_step.reset();
                self.work.log.step(self.work.h, &state);

                // step
                self.work.stats.n_steps += 1;
                status = self.actual.step(&mut self.work, state, stop, args)?;

                // handle failures
                if status.failure() {
                    if status.try_again() {
                        self.work.n_continued_failure += 1;
                        self.work.follows_failure = true;
                    } else {
                        break;
                    }
                }

                // handle continued failure (allowed to "try again")
                if self.work.n_continued_failure >= self.config.n_cont_failure_allowed {
                    status = Status::ContinuedFailure;
                    break;
                }

                // handle rejections (due to large curvatures, etc.)
                if self.work.n_continued_rejection >= self.config.allowed_continued_rejection {
                    status = Status::ContinuedRejection;
                    break;
                }

                // accept step
                if self.work.acceptable {
                    // update u and λ
                    self.work.stats.n_accepted += 1;
                    let rerr = self.actual.accept(&mut self.work, state, args)?;

                    // check for anomalies
                    vec_all_finite(&state.u, self.config.verbose)?;

                    // handle target u or λ reached
                    if self.work.target_reached {
                        break;
                    }

                    // adapt stepsize
                    let mut h_estimate = self.adapt_stepsize(rerr);
                    if self.work.follows_failure || self.work.follows_rejection {
                        // avoid stepsize growth on failure/rejection
                        h_estimate = f64::min(h_estimate, self.work.h);
                    }
                    self.work.h = h_estimate;
                    self.work.stats.h_accepted = h_estimate;

                    // reset flags
                    self.work.n_continued_failure = 0;
                    self.work.n_continued_rejection = 0;
                    self.work.follows_failure = false;
                    self.work.follows_rejection = false;

                    // output
                    if let Some(out) = output.as_deref_mut() {
                        let stop_gracefully = out.execute(&self.work, &state, args)?;
                        if stop_gracefully {
                            self.work.stats.stop_sw_step();
                            break;
                        }
                    }

                    // exit point
                    self.work.stats.stop_sw_step();
                    if stop.now(i, state) {
                        break;
                    }

                // reject step
                } else {
                    // set flags
                    self.work.stats.n_rejected += 1;
                    self.work.follows_rejection = true;

                    // perform the reject operations (e.g., restore external vars) and recompute stepsize
                    self.actual.reject(&mut self.work, args);

                    // adapt stepsize
                    self.work.h *= self.config.m_failure;
                }

                // check allowed stepsize change
                if self.work.h < CONFIG_H_MIN {
                    status = Status::SmallStepsize;
                    break;
                }
            }
        }

        // last output
        if let Some(out) = output.as_deref_mut() {
            out.last()?;
        }

        // print last message and footer
        self.work.log.step(self.work.h, &state);
        self.work.stats.stop_sw_total();
        self.work.log.footer(&self.work.stats, status)?;

        // done
        Ok(status)
    }

    /// Calculates the new stepsize based on the relative error `rerr`
    fn adapt_stepsize(&mut self, rerr: f64) -> f64 {
        let mut rho = f64::powf(1.0 / rerr, self.config.tg_control_beta1);
        if self.work.stats.n_accepted > 1 {
            rho *= f64::powf(1.0 / self.rerr_prev, self.config.tg_control_beta2);
            rho *= f64::powf(self.work.h / self.h_prev, -self.config.tg_control_alpha2);
        }
        if self.work.stats.n_accepted > 2 {
            rho *= f64::powf(1.0 / self.rerr_anc, self.config.tg_control_beta3);
            rho *= f64::powf(self.h_prev / self.h_anc, -self.config.tg_control_alpha3);
        }
        self.rerr_anc = self.rerr_prev;
        self.rerr_prev = rerr;
        self.h_anc = self.h_prev;
        self.h_prev = self.work.h;

        // calculate the relative convergence behavior of the Newton-Raphson iterations
        let nn = self.work.n_iteration as f64;
        let nn_opt = self.config.nr_control_n_opt as f64;
        let ksi = f64::powf(nn_opt / nn, self.config.nr_control_beta);

        // calculate the new stepsize using the convergence behavior and the error in the tangent vector
        let m = 1.0 + f64::atan(rho * ksi - 1.0); // smoothing formula by Soderlind and Wang (2006)
        self.work.h * m
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
        config.tol_abs_residual = 0.0; // wrong
        assert_eq!(
            Solver::new(config, system).err(),
            Some("requirement: tol_abs_residual ≥ CONFIG_TOL_MIN")
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
                    Stop::MaxLambda(1.0),
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
                    Stop::MaxLambda(0.0),
                    AutoStep::Yes,
                    None,
                )
                .err(),
            Some("Stop enum error: MaxLambda value must be greater than the initial lambda value")
        );
        assert_eq!(
            solver
                .solve(
                    &mut args,
                    &mut state,
                    Direction::Pos,
                    Stop::MaxLambda(1.0),
                    AutoStep::No(f64::EPSILON), // will cause an error
                    None,
                )
                .err(),
            Some("AutoStep enum error: fixed stepsize h_eq must be ≥ 10.0 * f64::EPSILON")
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
                    Stop::MaxLambda(1.0),
                    AutoStep::Yes,
                    None,
                )
                .unwrap(),
            Status::Success
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
        let stats = solver.get_stats();
        assert_eq!(stats.n_function, 7);
        assert_eq!(stats.n_jacobian, 6);
        assert_eq!(stats.n_factor, 6);
        assert_eq!(stats.n_lin_sol, 6);
        assert_eq!(stats.n_steps, 1);
        assert_eq!(stats.n_accepted, 1);
        assert_eq!(stats.n_rejected, 0);
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
        let stats = solver.get_stats();
        assert_eq!(stats.n_function, 7);
        assert_eq!(stats.n_jacobian, 6);
        assert_eq!(stats.n_factor, 6);
        assert_eq!(stats.n_lin_sol, 6);
        assert_eq!(stats.n_steps, 1);
        assert_eq!(stats.n_accepted, 1);
        assert_eq!(stats.n_rejected, 0);
        assert_eq!(stats.n_iteration_total, 7);
    }
}
