use super::{CONFIG_H_MIN, Output, SolverArclength, SolverNatural, Stats, Status, StepSummary, Workspace};
use super::{Config, DeltaLambda, IniDir, Method, SolverTrait, Stop, System};
use crate::StrError;
use russell_lab::{Vector, vec_all_finite};

/// Default number of equal steps for generating evenly spaced solution points
pub const N_EQUAL_STEPS: usize = 10;

/// Solver for parameterized nonlinear systems using Numerical Continuation
///
/// This is the main entry point of the crate. Given a [`System`] and a [`Config`],
/// it drives either the Natural parameter continuation ([`Method::Natural`]) or the
/// pseudo-arclength continuation ([`Method::Arclength`]) until a [`Stop`] criterion
/// is met.
pub struct Solver<'a, A> {
    /// Configuration options
    config: &'a Config,

    /// Dimension of the nonlinear system
    ndim: usize,

    /// Holds a pointer to the actual solver
    actual: Box<dyn SolverTrait<A> + 'a>,

    /// Holds statistics, benchmarking and "work" variables
    work: Workspace,

    // variables for the stepsize adaptation
    rerr_prev: f64,
    rerr_anc: f64,
    h_prev: f64,
    h_anc: f64,
}

impl<'a, A> Solver<'a, A> {
    /// Allocates a new instance
    ///
    /// # Arguments
    ///
    /// * `config` -- solver configuration (method, tolerances, step control, linear solver)
    /// * `system` -- the nonlinear system definition, including `G(u, λ)` and its Jacobian
    ///
    /// # Returns
    ///
    /// * `Ok(Solver)` if the configuration is valid and all internal structures are allocated
    /// * `Err(StrError)` if the configuration is invalid (e.g., tolerances out of range)
    pub fn new(config: &'a Config, system: System<'a, A>) -> Result<Self, StrError>
    where
        A: 'a,
    {
        config.validate()?;
        let ndim = system.ndim;
        let work = Workspace::new(config, &system);
        let actual: Box<dyn SolverTrait<A>> = match config.method {
            Method::Arclength => Box::new(SolverArclength::new(config, system.clone())?),
            Method::Natural => Box::new(SolverNatural::new(config, system.clone())?),
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
    /// * `(u, l)` -- the initial state `(u₀, λ₀)` with a non-singular `Gu₀ = ∂G/∂u|₀` (Jacobian) matrix.
    ///   The state will be updated with the new solution (u, λ) until a stop criterion is reached.
    /// * `dir` -- the direction to follow on the solution branch (pseudo-arclength method).
    /// * `stop` -- stop criterion (e.g, either a final λ value or a number of steps)
    /// * `ddl` -- specifies how Δλ is adjusted
    /// * `output` -- output object to write results files, record the results, or execute a callback function
    ///
    /// # Returns
    ///
    /// Returns the status code.
    ///
    /// # External state variables
    ///
    /// When the run ends in a terminal failure while a step is still in flight, the external state
    /// variables are restored to the snapshot taken at the start of that step, so `args` never keeps
    /// the trial values of a step that was not accepted.
    pub fn solve<'b>(
        &mut self,
        args: &mut A,
        u: &mut Vector,
        l: &mut f64,
        dir: IniDir,
        stop: Stop,
        ddl: &DeltaLambda,
        output: Option<&mut Output<'b, A>>,
    ) -> Result<Status, StrError> {
        let res = self.solve_inner(args, u, l, dir, stop, ddl, output);
        let terminal_failure = match &res {
            Err(_) => true,
            Ok(status) => status.failure(),
        };
        if terminal_failure && self.work.step_in_flight {
            // The failing step was never accepted nor rejected, so the external state variables are
            // still at the trial values: put them back at the snapshot taken at the start of the step.
            self.actual.restore_secondary_state(args);
            self.work.step_in_flight = false;
        }
        res
    }

    /// Solves the nonlinear system, without the terminal restore of the external state variables
    ///
    /// The input validation lives here so that a validation error is not treated as an in-flight
    /// step failure by [`Solver::solve`].
    fn solve_inner<'b>(
        &mut self,
        args: &mut A,
        u: &mut Vector,
        l: &mut f64,
        dir: IniDir,
        stop: Stop,
        ddl: &DeltaLambda,
        mut output: Option<&mut Output<'b, A>>,
    ) -> Result<Status, StrError> {
        // validate input
        if u.dim() != self.ndim {
            return Err("u.dim() must be equal to ndim");
        }
        stop.validate(u, *l)?;

        // reset stats, flags, and stepsize history
        self.work.reset_stats_and_flags(ddl.auto);
        self.rerr_prev = 0.0;
        self.rerr_anc = 0.0;
        self.h_prev = 0.0;
        self.h_anc = 0.0;

        // calculate the default initial Δλ
        let ddl_ini = ddl.ini(&stop, *l)?;

        // perform initialization (compute the actual initial stepsize and tangent vector)
        self.actual.initialize(&mut self.work, ddl_ini, u, *l, dir, args)?;
        if self.work.h > self.config.h_max {
            self.work.h = self.config.h_max;
        }
        self.work.stats.h_next = self.work.h;

        // first output
        if let Some(out) = output.as_deref_mut() {
            let stop_gracefully = out.execute(&self.work, u, *l, self.work.h, args)?;
            if stop_gracefully {
                return Ok(Status::Success);
            }
        }

        // message
        if self.config.verbose_header_footer {
            self.work.log.header();
        }

        // default status and termination flags
        let mut status = Status::Success;
        let mut stop_reached = false;
        let mut stopped_gracefully = false;

        // perform continuation
        if !ddl.auto {
            // constant or list-based stepsize
            let n_step_max = if !ddl.list.is_empty() { ddl.list.len() } else { self.config.n_step_max };
            for i in 0..n_step_max {
                // log
                self.work.stats.sw_step.reset();
                self.work.log.step(self.work.h, *l, false);

                // step
                self.work.stats.n_steps += 1;
                status = self.actual.step(&mut self.work, u, *l, stop, args)?;

                // handle failures
                if status.failure() {
                    let h_used = self.work.h;
                    let (n_iter, res_0, res_last) = self.current_step_residuals();
                    self.work.stats.record_rejection_status(&status);
                    self.work.stats.h_next = h_used;
                    self.work.stats.record_step_attempt(StepSummary {
                        step_index: self.work.stats.n_steps,
                        accepted: false,
                        status: status.clone(),
                        h_used,
                        h_next: h_used,
                        rdiff: 0.0,
                        n_iteration: n_iter,
                        residual_0: res_0,
                        residual_last: res_last,
                        lambda: *l,
                    });
                    break;
                }

                // update u and λ
                self.work.stats.n_accepted += 1;
                let rdiff = self.actual.accept(&mut self.work, u, l, args)?;
                let h_used = self.work.h;

                // update the stepsize if a list is given
                if i + 1 < ddl.list.len() {
                    let ddl_next = ddl.list[i + 1];
                    if ddl_next <= CONFIG_H_MIN {
                        return Err("requirement: ddl > 1e-10");
                    }
                    self.work.h = match self.config.method {
                        Method::Arclength => {
                            let den = f64::abs(self.work.dlds);
                            if den < CONFIG_H_MIN {
                                return Err("dλ/ds is too small to calculate the stepsize");
                            }
                            ddl_next / den
                        }
                        Method::Natural => ddl_next,
                    };
                }
                let h_next = self.work.h;
                self.work.stats.h_accepted = h_used;
                self.work.stats.h_next = h_next;
                let (n_iter, res_0, res_last) = self.current_step_residuals();
                self.work.stats.record_step_attempt(StepSummary {
                    step_index: self.work.stats.n_steps,
                    accepted: true,
                    status: Status::Success,
                    h_used,
                    h_next,
                    rdiff,
                    n_iteration: n_iter,
                    residual_0: res_0,
                    residual_last: res_last,
                    lambda: *l,
                });

                // check for anomalies
                vec_all_finite(u, self.config.verbose)?;

                // output
                if let Some(out) = output.as_deref_mut() {
                    let stop_gracefully = out.execute(&self.work, u, *l, h_used, args)?;
                    if stop_gracefully {
                        stopped_gracefully = true;
                        self.work.stats.stop_sw_step();
                        break;
                    }
                }

                // exit point
                self.work.stats.stop_sw_step();
                if stop.now(i, u, *l) {
                    stop_reached = true;
                    break;
                }
            }
        } else {
            // variable stepsize
            for i in 0..self.config.n_step_max {
                // log
                self.work.stats.sw_step.reset();
                self.work.log.step(self.work.h, *l, false);

                // step
                self.work.stats.n_steps += 1;
                status = self.actual.step(&mut self.work, u, *l, stop, args)?;

                // redo the step if it violates the stop u-component criterion
                if self.config.enable_precise_stop_u_comp {
                    if let Some((i, u1i, is_min)) = stop.u_comp() {
                        if (is_min && self.work.u[i] < u1i - CONFIG_H_MIN) || (!is_min && self.work.u[i] > u1i + CONFIG_H_MIN) {
                            status = Status::UnmetStopCriterion;
                        }
                    }
                }

                let (n_iter, res_0, res_last) = self.current_step_residuals();

                // check for failures
                if status.failure() {
                    let h_used = self.work.h;
                    let attempt_status = status.clone();
                    self.work.stats.record_rejection_status(&attempt_status);
                    if status.try_again() {
                        self.work.n_continued_failure += 1;
                        self.work.follows_failure = true;
                    } else {
                        self.work.stats.h_next = h_used;
                        self.work.stats.record_step_attempt(StepSummary {
                            step_index: self.work.stats.n_steps,
                            accepted: false,
                            status: attempt_status,
                            h_used,
                            h_next: h_used,
                            rdiff: 0.0,
                            n_iteration: n_iter,
                            residual_0: res_0,
                            residual_last: res_last,
                            lambda: *l,
                        });
                        break;
                    }

                    // handle continued failure
                    if self.work.n_continued_failure >= self.config.n_cont_failure_max {
                        status = Status::ContinuedFailure;
                        self.work.stats.h_next = h_used;
                        self.work.stats.record_step_attempt(StepSummary {
                            step_index: self.work.stats.n_steps,
                            accepted: false,
                            status: attempt_status,
                            h_used,
                            h_next: h_used,
                            rdiff: 0.0,
                            n_iteration: n_iter,
                            residual_0: res_0,
                            residual_last: res_last,
                            lambda: *l,
                        });
                        break;
                    }

                    // handle continued rejections
                    if self.work.n_continued_rejection >= self.config.n_cont_rejection_max {
                        status = Status::ContinuedRejection;
                        self.work.stats.h_next = h_used;
                        self.work.stats.record_step_attempt(StepSummary {
                            step_index: self.work.stats.n_steps,
                            accepted: false,
                            status: attempt_status,
                            h_used,
                            h_next: h_used,
                            rdiff: 0.0,
                            n_iteration: n_iter,
                            residual_0: res_0,
                            residual_last: res_last,
                            lambda: *l,
                        });
                        break;
                    }

                    // set flags
                    self.work.stats.n_rejected += 1;
                    self.work.follows_rejection = true;

                    // perform the reject operations (e.g., restore external vars) and reset PID history
                    self.actual.reject(&mut self.work, args);
                    self.rerr_prev = 0.0;
                    self.rerr_anc = 0.0;

                    // adapt stepsize
                    self.work.h *= self.config.m_failure;
                    let h_next = self.work.h;
                    self.work.stats.h_next = h_next;
                    self.work.stats.record_step_attempt(StepSummary {
                        step_index: self.work.stats.n_steps,
                        accepted: false,
                        status: attempt_status,
                        h_used,
                        h_next,
                        rdiff: 0.0,
                        n_iteration: n_iter,
                        residual_0: res_0,
                        residual_last: res_last,
                        lambda: *l,
                    });
                } else {
                    // update u and λ
                    self.work.stats.n_accepted += 1;
                    let rdiff = self.actual.accept(&mut self.work, u, l, args)?;
                    let h_used = self.work.h;

                    // check for anomalies
                    vec_all_finite(u, self.config.verbose)?;

                    // exit point: target u or λ reached
                    if self.work.target_reached {
                        self.work.stats.h_accepted = h_used;
                        self.work.stats.h_next = h_used;
                        self.work.stats.record_step_attempt(StepSummary {
                            step_index: self.work.stats.n_steps,
                            accepted: true,
                            status: Status::Success,
                            h_used,
                            h_next: h_used,
                            rdiff,
                            n_iteration: n_iter,
                            residual_0: res_0,
                            residual_last: res_last,
                            lambda: *l,
                        });
                        stop_reached = true;
                        break;
                    }

                    // adapt stepsize
                    let mut h_estimate = self.adapt_stepsize(rdiff);
                    if self.work.follows_failure || self.work.follows_rejection {
                        // hold stepsize after fast recovery; block growth after slow recovery
                        if self.work.n_iteration <= self.config.nr_control_n_opt {
                            h_estimate = self.work.h;
                        } else {
                            h_estimate = f64::min(h_estimate, self.work.h);
                        }
                    }
                    let h_next = f64::min(h_estimate, self.config.h_max);
                    self.work.h = h_next;
                    self.work.stats.h_accepted = h_used;
                    self.work.stats.h_next = h_next;
                    self.work.stats.record_step_attempt(StepSummary {
                        step_index: self.work.stats.n_steps,
                        accepted: true,
                        status: Status::Success,
                        h_used,
                        h_next,
                        rdiff,
                        n_iteration: n_iter,
                        residual_0: res_0,
                        residual_last: res_last,
                        lambda: *l,
                    });

                    // reset flags
                    self.work.n_continued_failure = 0;
                    self.work.n_continued_rejection = 0;
                    self.work.follows_failure = false;
                    self.work.follows_rejection = false;

                    // output
                    if let Some(out) = output.as_deref_mut() {
                        let stop_gracefully = out.execute(&self.work, u, *l, h_used, args)?;
                        if stop_gracefully {
                            stopped_gracefully = true;
                            self.work.stats.stop_sw_step();
                            break;
                        }
                    }

                    // exit point
                    self.work.stats.stop_sw_step();
                    if stop.now(i, u, *l) {
                        stop_reached = true;
                        break;
                    }
                }

                // check minimum allowed stepsize
                if f64::abs(self.work.h) < self.config.h_min {
                    status = Status::SmallStepsize;
                    break;
                }
            }
        }

        // Mark runs that exhausted n_step_max (under automatic or constant stepsize) without
        // meeting the stop criterion or stopping gracefully via an output callback.
        if status.success() && !stop_reached && !stopped_gracefully && (ddl.auto || ddl.list.is_empty()) {
            status = Status::ReachedMaxSteps;
        }

        // Close the drift of an arclength run that ended short of a lambda target: a listed Δλ is
        // converted to a step in s through the previous |dλ/ds|, so the last step can land just
        // short. The polish step is a no-op for the natural method and for any stop that is not a
        // lambda target, and only runs when a listed sequence completed without meeting Stop or
        // stopping gracefully.
        if status.success() && !stop_reached && !stopped_gracefully {
            if let Some((l1, _)) = stop.lambda() {
                if f64::abs(*l - l1) > CONFIG_H_MIN {
                    if let Some(polish_status) = self.actual.polish(&mut self.work, u, *l, stop, args)? {
                        let (n_iter, res_0, res_last) = self.current_step_residuals();
                        if polish_status.success() {
                            self.work.stats.n_steps += 1;
                            self.work.stats.n_accepted += 1;
                            let rdiff = self.actual.accept(&mut self.work, u, l, args)?;
                            let h_used = self.work.h;
                            self.work.stats.h_accepted = h_used;
                            self.work.stats.h_next = h_used;
                            self.work.stats.record_step_attempt(StepSummary {
                                step_index: self.work.stats.n_steps,
                                accepted: true,
                                status: Status::Success,
                                h_used,
                                h_next: h_used,
                                rdiff,
                                n_iteration: n_iter,
                                residual_0: res_0,
                                residual_last: res_last,
                                lambda: *l,
                            });
                            vec_all_finite(u, self.config.verbose)?;
                            if let Some(out) = output {
                                let _ = out.execute(&self.work, u, *l, h_used, args)?;
                            }
                        } else {
                            let h_used = self.work.h;
                            self.work.stats.record_rejection_status(&polish_status);
                            self.work.stats.h_next = h_used;
                            self.work.stats.record_step_attempt(StepSummary {
                                step_index: self.work.stats.n_steps + 1,
                                accepted: false,
                                status: polish_status.clone(),
                                h_used,
                                h_next: h_used,
                                rdiff: 0.0,
                                n_iteration: n_iter,
                                residual_0: res_0,
                                residual_last: res_last,
                                lambda: *l,
                            });
                            status = polish_status;
                        }
                    }
                }
            }
        }

        // stop total stopwatch
        self.work.stats.stop_sw_total();

        // print last message and footer
        if self.config.verbose_header_footer {
            self.work.log.step(self.work.h, *l, true);
            self.work.log.footer(&self.work.stats, &status)?;
        }

        // done
        Ok(status)
    }

    /// Logs the header (method name and column headers)
    ///
    /// Writes the solver method name and column headers to the configured
    /// log output (stdout or a log file). Only has an effect if
    /// `Config.verbose_header_footer` is enabled.
    pub fn log_header(&mut self) {
        self.work.log.header();
    }

    /// Logs the footer with statistics and final status
    ///
    /// Writes a summary of the solver statistics and the final status
    /// to the configured log output (stdout or a log file).
    ///
    /// # Panics
    ///
    /// Panics if the logger's `footer` method returns an error.
    pub fn log_footer(&mut self) {
        self.work.log.footer(&self.work.stats, &Status::Success).unwrap();
    }

    /// Returns `(n_iteration_evals, residual_0, residual_last)` for the current step attempt
    fn current_step_residuals(&self) -> (usize, f64, f64) {
        if self.work.err.has_residual {
            (self.work.n_iteration + 1, self.work.err.residual_0, self.work.err.residual_max)
        } else {
            (0, f64::NAN, f64::NAN)
        }
    }

    /// Adapts the stepsize
    fn adapt_stepsize(&mut self, rdiff: f64) -> f64 {
        // calculate the relative convergence behavior of the Newton-Raphson iterations
        let ksi = if self.config.nr_control_enabled {
            let nn = f64::max(1.0, self.work.n_iteration as f64);
            let nn_opt = self.config.nr_control_n_opt as f64;
            f64::powf(nn_opt / nn, self.config.nr_control_beta)
        } else {
            1.0
        };

        // set rdiff to zero if it is too small so that tiny differences in
        // the solver yield deterministic stepsize control behavior
        assert!(rdiff >= 0.0, "rdiff must be non-negative");
        let rdiff = if rdiff < self.config.tg_control_rdiff_min { 0.0 } else { rdiff };

        // calculates the relative changes on the tangent vector
        let rho = if self.config.tg_control_enabled {
            if rdiff == 0.0 {
                self.config.tg_control_rho_for_tiny_rdiff
            } else {
                let raw_rho = if self.config.tg_control_pid_vcc {
                    const KP: f64 = 0.075;
                    const KI: f64 = 0.175;
                    const KD: f64 = 0.01;
                    let mut p = 1.0;
                    let mut d = 1.0;
                    let i = 1.0 / rdiff;
                    // The previous error may have been clamped to zero by tg_control_rdiff_min. Dividing
                    // by it would make p = 0 and turn the first error above the floor into the maximal
                    // shrink, which keeps the stepsize in a limit cycle. With p = 1 the first real error
                    // is judged on its own.
                    if self.work.stats.n_accepted > 1 && self.rerr_prev > 0.0 {
                        p = self.rerr_prev / rdiff;
                    }
                    if self.work.stats.n_accepted > 2 && self.rerr_anc > 0.0 {
                        d = self.rerr_prev * self.rerr_prev / (rdiff * self.rerr_anc);
                    }
                    f64::powf(p, KP) * f64::powf(i, KI) * f64::powf(d, KD)
                } else {
                    let mut rho = f64::powf(1.0 / rdiff, self.config.tg_control_beta1);
                    if self.work.stats.n_accepted > 1 {
                        rho *= f64::powf(1.0 / self.rerr_prev, self.config.tg_control_beta2);
                        rho *= f64::powf(self.work.h / self.h_prev, -self.config.tg_control_alpha2);
                    }
                    if self.work.stats.n_accepted > 2 {
                        rho *= f64::powf(1.0 / self.rerr_anc, self.config.tg_control_beta3);
                        rho *= f64::powf(self.h_prev / self.h_anc, -self.config.tg_control_alpha3);
                    }
                    rho
                };
                f64::min(raw_rho, self.config.tg_control_rho_for_tiny_rdiff)
            }
        } else {
            1.0
        };

        // records the previous values
        self.rerr_anc = self.rerr_prev;
        self.rerr_prev = rdiff;
        self.h_anc = self.h_prev;
        self.h_prev = self.work.h;

        // calculate the new stepsize using the convergence behavior and the error in the tangent vector
        let m = 1.0 + f64::atan(ksi * rho - 1.0); // smoothing formula by Soderlind and Wang (2006)
        self.work.h * m
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Solver;
    use crate::{Config, DeltaLambda, IniDir, Method, NoArgs, Output, Samples, Status, Stop, System};
    use russell_lab::{Vector, approx_eq, vec_approx_eq};
    use russell_sparse::Sym;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn new_captures_errors() {
        let (system, _, _, _) = Samples::two_eq_ref();
        let mut config = Config::new();
        config.tol_abs_residual = 0.0; // wrong
        assert_eq!(Solver::new(&config, system).err(), Some("requirement: tol_abs_residual ≥ 1e-12"));
    }

    #[test]
    fn solve_captures_errors() {
        let (system, _, _, mut args) = Samples::two_eq_ref();
        let mut u = Vector::new(system.ndim + 1); // wrong dim
        let mut l = 0.0;
        let ndim = system.ndim;
        let config = Config::new();
        let mut solver = Solver::new(&config, system).unwrap();
        assert_eq!(
            solver
                .solve(
                    &mut args,
                    &mut u,
                    &mut l,
                    IniDir::Pos,
                    Stop::MaxLambda(1.0),
                    &DeltaLambda::auto(1e-4),
                    None
                )
                .err(),
            Some("u.dim() must be equal to ndim")
        );
        u = Vector::new(ndim); // fix dim
        assert_eq!(
            solver
                .solve(
                    &mut args,
                    &mut u,
                    &mut l,
                    IniDir::Pos,
                    Stop::MaxLambda(0.0),
                    &DeltaLambda::auto(1e-4),
                    None,
                )
                .err(),
            Some("Stop enum error: MaxLambda value must be greater than the initial lambda value")
        );
        assert_eq!(
            solver
                .solve(
                    &mut args,
                    &mut u,
                    &mut l,
                    IniDir::Pos,
                    Stop::MaxLambda(1.0),
                    &DeltaLambda::constant(f64::EPSILON), // will cause an error
                    None,
                )
                .err(),
            Some("requirement: ddl_ini > 1e-10")
        );
    }

    #[test]
    fn lack_of_convergence_is_captured() {
        let (system, mut u, _u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new();
        config.n_step_max = 1; // will make the solver fail (too few steps)
        let mut solver = Solver::new(&config, system).unwrap();
        let mut l = 0.0;
        assert_eq!(
            solver
                .solve(
                    &mut args,
                    &mut u,
                    &mut l,
                    IniDir::Pos,
                    Stop::MaxLambda(1.0),
                    &DeltaLambda::auto(1e-4),
                    None,
                )
                .unwrap(),
            Status::ReachedMaxSteps
        );
    }

    /// The callback runs on the initial point as well, and returning `true` there must
    /// end the computation with success before any step is taken.
    #[test]
    fn solve_stops_gracefully_at_the_first_output() {
        let (system, mut u, _u_ref, mut args) = Samples::two_eq_ref();
        let config = Config::new();
        let mut solver = Solver::new(&config, system).unwrap();
        let mut l = 0.0;
        let mut out = Output::new();
        out.set_callback(|_stats, _u, _l, _h, _args| Ok(true));
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::Steps(3),
                &DeltaLambda::constant(1.0),
                Some(&mut out),
            )
            .unwrap();
        assert_eq!(status, Status::Success);
        assert_eq!(solver.get_stats().n_steps, 0);
    }

    /// Returning `true` from the accepted-step callback must end the computation with
    /// success after that step, rather than continuing to the stop criterion.
    #[test]
    fn solve_stops_gracefully_after_an_accepted_step() {
        let (system, mut u, _u_ref, mut args) = Samples::two_eq_ref();
        let config = Config::new();
        let mut solver = Solver::new(&config, system).unwrap();
        let mut l = 0.0;
        let calls = AtomicUsize::new(0);
        let mut out = Output::new();
        out.set_callback(|_stats, _u, _l, _h, _args| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            Ok(n > 0) // false on the initial output, true after the first step
        });
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::Steps(5),
                &DeltaLambda::constant(1.0),
                Some(&mut out),
            )
            .unwrap();
        assert_eq!(status, Status::Success);
        assert_eq!(solver.get_stats().n_steps, 1);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// A listed Δλ below the solver's minimum must be refused instead of producing a
    /// step the solver cannot take.
    #[test]
    fn solve_rejects_a_listed_step_below_the_minimum() {
        let (system, mut u, _u_ref, mut args) = Samples::two_eq_ref();
        let config = Config::new();
        let mut solver = Solver::new(&config, system).unwrap();
        let mut l = 0.0;
        let res = solver.solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::Steps(3),
            &DeltaLambda::list(&[1.0, 1e-12]),
            None,
        );
        assert_eq!(res.err(), Some("requirement: ddl > 1e-10"));
    }

    #[test]
    fn solve_with_one_step_works_fixed() {
        let (system, mut u, u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new();
        config.set_verbose(false, true, true).set_tol_delta(1e-12, 1e-10);
        let mut solver = Solver::new(&config, system).unwrap();
        let mut l = 0.0;
        solver
            .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(1), &DeltaLambda::constant(1.0), None)
            .unwrap();
        vec_approx_eq(&u, &u_ref, 1e-15);
        let stats = solver.get_stats();
        assert_eq!(stats.n_function, 7);
        assert_eq!(stats.n_jacobian, 7);
        assert_eq!(stats.n_factor, 7);
        assert_eq!(stats.n_lin_sol, 6);
        assert_eq!(stats.n_steps, 1);
        assert_eq!(stats.n_accepted, 1);
        assert_eq!(stats.n_rejected, 0);
        assert_eq!(stats.n_iteration_total, 7);
    }

    #[test]
    fn solve_with_one_step_works_auto() {
        let (system, mut u, u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new();
        config.set_verbose(false, true, true).set_tol_delta(1e-12, 1e-10);
        let mut solver = Solver::new(&config, system).unwrap();
        let mut l = 0.0;
        solver
            .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(1), &DeltaLambda::auto(1e-4), None)
            .unwrap();
        vec_approx_eq(&u, &u_ref, 1e-15);
        let stats = solver.get_stats();
        assert_eq!(stats.n_function, 7);
        assert_eq!(stats.n_jacobian, 7);
        assert_eq!(stats.n_factor, 7);
        assert_eq!(stats.n_lin_sol, 6);
        assert_eq!(stats.n_steps, 1);
        assert_eq!(stats.n_accepted, 1);
        assert_eq!(stats.n_rejected, 0);
        assert_eq!(stats.n_iteration_total, 7);
    }

    #[test]
    fn debug_predictor_values_disabled_returns_empty() {
        let (system, mut u, _u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new();
        config.set_debug_predictor(false);
        let mut solver = Solver::new(&config, system).unwrap();
        let mut l = 0.0;
        solver
            .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(3), &DeltaLambda::auto(1e-4), None)
            .unwrap();
        let (l_pred, u0_pred, u1_pred) = solver.get_debug_predictor_values();
        assert!(l_pred.is_empty());
        assert!(u0_pred.is_empty());
        assert!(u1_pred.is_empty());
    }

    #[test]
    fn debug_predictor_values_enabled_populates_data() {
        let (system, mut u, _u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new();
        config.set_debug_predictor(true);
        let mut solver = Solver::new(&config, system).unwrap();
        let mut l = 0.0;
        solver
            .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(5), &DeltaLambda::auto(1e-4), None)
            .unwrap();
        let (l_pred, u0_pred, u1_pred) = solver.get_debug_predictor_values();
        // With 5 steps and debug enabled, we should have at least 5 predictor entries
        assert!(l_pred.len() >= 5);
        assert!(u0_pred.len() >= 5);
        assert!(u1_pred.len() >= 5);
        // All values should be finite
        for &v in &l_pred {
            assert!(v.is_finite());
        }
        for &v in &u0_pred {
            assert!(v.is_finite());
        }
        for &v in &u1_pred {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn log_header_and_footer_disabled_do_not_panic() {
        let (system, _, _, _) = Samples::two_eq_ref();
        let mut config = Config::new();
        config.set_verbose_header_footer(false);
        let mut solver = Solver::new(&config, system).unwrap();
        solver.log_header();
        solver.log_footer();
    }

    #[test]
    fn log_header_and_footer_to_file_work() {
        let (system, _, _, _) = Samples::two_eq_ref();
        let full_path = "/tmp/russell_nonlin/test_log_header_footer.txt";
        let _ = std::fs::remove_file(full_path);

        let mut config = Config::new();
        config.set_log_file(full_path);
        config.set_verbose_header_footer(true);
        config.set_verbose_stats(true);

        let mut solver = Solver::new(&config, system).unwrap();
        solver.log_header();
        solver.log_footer();

        let contents = std::fs::read_to_string(full_path).unwrap();
        assert!(contents.contains("λ"));
        assert!(contents.contains("Δλ"));
        assert!(contents.contains("‖G‖∞"));
        assert!(contents.contains("Natural parameter continuation"));
        assert!(contents.contains("Number of function evaluations"));
    }

    #[test]
    fn log_header_and_footer_arclength_to_file_work() {
        let (system, _, _, _) = Samples::two_eq_ref();
        let full_path = "/tmp/russell_nonlin/test_log_header_footer_arc.txt";
        let _ = std::fs::remove_file(full_path);

        let mut config = Config::new();
        config.set_method(Method::Arclength);
        config.set_log_file(full_path);
        config.set_verbose_header_footer(true);
        config.set_verbose_stats(true);

        let mut solver = Solver::new(&config, system).unwrap();
        solver.log_header();
        solver.log_footer();

        let contents = std::fs::read_to_string(full_path).unwrap();
        assert!(contents.contains("‖(G,N)‖∞"));
        assert!(contents.contains("Pseudo-arclength continuation"));
    }

    #[test]
    fn reached_max_steps_skips_polish_for_constant_and_auto() {
        // 1. Constant stepsize with MaxLambda
        let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);
        let mut config = Config::new();
        config.set_method(Method::Arclength).set_n_step_max(3);
        let mut solver = Solver::new(&config, system).unwrap();
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::MaxLambda(1.0),
                &DeltaLambda::constant(0.1),
                None,
            )
            .unwrap();
        assert_eq!(status, Status::ReachedMaxSteps);
        assert_eq!(solver.get_stats().n_steps, 3);
        assert_eq!(solver.get_stats().n_accepted, 3);
        approx_eq(l, 0.3, 1e-13);

        // 2. Constant stepsize with MaxCompU
        let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);
        let mut config = Config::new();
        config.set_method(Method::Arclength).set_n_step_max(3);
        let mut solver = Solver::new(&config, system).unwrap();
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::MaxCompU(0, 1.0),
                &DeltaLambda::constant(0.1),
                None,
            )
            .unwrap();
        assert_eq!(status, Status::ReachedMaxSteps);
        assert_eq!(solver.get_stats().n_steps, 3);
        approx_eq(u[0], 0.3, 1e-13);

        // 3. Auto stepsize with MaxLambda
        let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);
        let mut config = Config::new();
        config.set_method(Method::Arclength).set_n_step_max(2);
        let mut solver = Solver::new(&config, system).unwrap();
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::MaxLambda(1.0),
                &DeltaLambda::auto(0.1),
                None,
            )
            .unwrap();
        assert_eq!(status, Status::ReachedMaxSteps);
        assert_eq!(solver.get_stats().n_steps, 2);
        assert!(l < 0.5, "polish must not jump to lambda = 1.0; got l = {l}");
    }

    #[test]
    fn graceful_stop_with_lambda_target_skips_polish() {
        let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);
        let mut config = Config::new();
        config.set_method(Method::Arclength);
        let mut solver = Solver::new(&config, system).unwrap();
        let calls = AtomicUsize::new(0);
        let mut out = Output::new();
        out.set_callback(|_stats, _u, _l, _h, _args| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            Ok(n > 0) // stop gracefully right after step 1
        });
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::MaxLambda(1.0),
                &DeltaLambda::list(&[0.2, 0.2, 0.2]),
                Some(&mut out),
            )
            .unwrap();
        assert_eq!(status, Status::Success);
        assert_eq!(solver.get_stats().n_steps, 1);
        approx_eq(l, 0.2, 1e-13);
    }

    #[test]
    fn configurable_h_min_aborts_with_small_stepsize() {
        // Quadratic curve that hits LargeDelta at h = 0.015, then halves to 0.0075 < h_min = 0.01.
        let system = System::new(
            1,
            Some(1),
            Sym::No,
            |gg, l, u, _args: &mut NoArgs| {
                gg[0] = u[0] - 1000.0 * l * l - 1.0e-6 * l;
                Ok(())
            },
            |ggu, ggl, l, _u, _args: &mut NoArgs| {
                ggu.put(0, 0, 1.0).unwrap();
                ggl[0] = -2000.0 * l - 1.0e-6;
                Ok(())
            },
        )
        .unwrap();
        let mut config = Config::new();
        config.set_method(Method::Arclength).set_h_min(0.01);
        let mut solver = Solver::new(&config, system).unwrap();
        let mut u = Vector::from(&[0.0]);
        let mut l = 0.0;
        let mut args = 0;
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::MaxLambda(1.0),
                &DeltaLambda::auto(0.015),
                None,
            )
            .unwrap();
        assert_eq!(status, Status::SmallStepsize);
        assert_eq!(solver.get_stats().n_rejected, 1);
        assert_eq!(solver.get_stats().n_reject_large_delta, 1);
    }

    #[test]
    fn telemetry_records_h_used_and_rejection_breakdown() {
        let system = System::new(
            1,
            Some(1),
            Sym::No,
            |gg, l, u, _args: &mut NoArgs| {
                gg[0] = u[0] - 1000.0 * l * l - 1.0e-6 * l;
                Ok(())
            },
            |ggu, ggl, l, _u, _args: &mut NoArgs| {
                ggu.put(0, 0, 1.0).unwrap();
                ggl[0] = -2000.0 * l - 1.0e-6;
                Ok(())
            },
        )
        .unwrap();
        let mut config = Config::new();
        config.set_method(Method::Arclength).set_record_step_summary(true);
        let mut solver = Solver::new(&config, system).unwrap();
        let mut out = Output::new();
        out.set_recording(true, &[0], &[]);
        let mut u = Vector::from(&[0.0]);
        let mut l = 0.0;
        let mut args = 0;
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::Steps(3), // step 0 rejects, steps 1 and 2 accept -> 3 total attempts
                &DeltaLambda::auto(0.015),
                Some(&mut out),
            )
            .unwrap();
        assert_eq!(status, Status::Success);
        let stats = solver.get_stats();
        assert_eq!(stats.n_steps, 3);
        assert_eq!(stats.n_rejected, 1);
        assert_eq!(stats.n_accepted, 2);
        assert_eq!(stats.n_reject_large_delta, 1);
        assert_eq!(stats.last_rejection_status, Some(Status::LargeDelta));
        let summaries = stats.step_summaries.as_ref().unwrap();
        assert_eq!(summaries.len(), 3);
        assert!(!summaries[0].accepted);
        assert_eq!(summaries[0].status, Status::LargeDelta);
        assert!(summaries[1].accepted);
        assert!(summaries[2].accepted);

        let h_vals = out.get_h_values();
        // h_vals = [h_initial, h_used_step1, h_used_step2]
        assert_eq!(h_vals.len(), 3);
        approx_eq(h_vals[1], summaries[1].h_used, 1e-15);
        approx_eq(h_vals[2], summaries[2].h_used, 1e-15);
        approx_eq(stats.h_accepted, summaries[2].h_used, 1e-15);
        approx_eq(stats.h_next, summaries[2].h_next, 1e-15);
        assert!(f64::abs(stats.h_accepted - stats.h_next) > 1e-12);
    }

    #[test]
    fn arclength_near_zero_dof_keeps_rdiff_bounded() {
        // 3-DOF arclength system (u0, u1, lambda) where u1 = 1e-10*lambda + 1e-4*lambda^2
        let system = System::new(
            2,
            Some(2),
            Sym::No,
            |gg, l, u, _args: &mut NoArgs| {
                gg[0] = u[0] - l;
                gg[1] = u[1] - (1e-10 * l + 1e-4 * l * l);
                Ok(())
            },
            |ggu, ggl, l, _u, _args: &mut NoArgs| {
                ggu.put(0, 0, 1.0).unwrap();
                ggu.put(1, 1, 1.0).unwrap();
                ggl[0] = -1.0;
                ggl[1] = -(1e-10 + 2e-4 * l);
                Ok(())
            },
        )
        .unwrap();
        let mut config = Config::new();
        config.set_method(Method::Arclength).set_record_step_summary(true);
        let mut solver = Solver::new(&config, system).unwrap();
        let mut u = Vector::from(&[0.0, 0.0]);
        let mut l = 0.0;
        let mut args = 0;
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::MaxLambda(1.0),
                &DeltaLambda::auto(0.1),
                None,
            )
            .unwrap();
        assert_eq!(status, Status::Success);
        let stats = solver.get_stats();
        assert!(stats.n_accepted < 25, "n_accepted = {}", stats.n_accepted);
        approx_eq(l, 1.0, 1e-12);
        let summaries = stats.step_summaries.as_ref().unwrap();
        for s in summaries {
            assert!(s.rdiff < 0.01, "step {} rdiff = {:e}", s.step_index, s.rdiff);
        }
    }

    #[test]
    fn arclength_crosses_c0_softening_plateau_kink_without_collapse() {
        // 2-DOF system with an elastic branch (u0 <= 0.2), softening (0.2 < u0 <= 0.4, H = -0.8 K),
        // and a C0 residual plateau (u0 > 0.4, H = 0), coupled in series with spring stiffness 1.0.
        let elem = |u0: f64| -> (f64, f64) {
            if u0 <= 0.2 {
                (u0, 1.0)
            } else if u0 <= 0.4 {
                (0.2 - 0.8 * (u0 - 0.2), -0.8)
            } else {
                (0.04, 0.0)
            }
        };
        let system = System::new(
            2,
            Some(3),
            Sym::No,
            move |gg, l, u, _args: &mut NoArgs| {
                let (f0, _) = elem(u[0]);
                gg[0] = f0 - (l - u[0]);
                gg[1] = u[1] - 0.5 * u[0];
                Ok(())
            },
            move |ggu, ggl, _l, u, _args: &mut NoArgs| {
                let (_, df0) = elem(u[0]);
                ggu.put(0, 0, df0 + 1.0).unwrap();
                ggu.put(1, 0, -0.5).unwrap();
                ggu.put(1, 1, 1.0).unwrap();
                ggl[0] = -1.0;
                ggl[1] = 0.0;
                Ok(())
            },
        )
        .unwrap();

        let mut config = Config::new();
        config.set_method(Method::Arclength).set_record_step_summary(true);
        let mut solver = Solver::new(&config, system).unwrap();
        let mut u = Vector::from(&[0.0, 0.0]);
        let mut l = 0.0;
        let mut args = 0;
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::MaxLambda(1.0),
                &DeltaLambda::auto(0.05),
                None,
            )
            .unwrap();
        assert_eq!(status, Status::Success);
        let stats = solver.get_stats();
        assert!(stats.n_accepted < 40, "n_accepted = {}", stats.n_accepted);
        approx_eq(l, 1.0, 1e-12);
        approx_eq(u[0], 0.96, 1e-12);

        let summaries = stats.step_summaries.as_ref().unwrap();
        let h_ini = summaries[0].h_used;
        let h_min_seen = summaries.iter().map(|s| s.h_used).fold(f64::INFINITY, f64::min);
        assert!(h_min_seen >= 0.2 * h_ini, "h_min_seen = {:e}, h_ini = {:e}", h_min_seen, h_ini);
    }

    #[test]
    fn adapt_stepsize_is_monotonic_across_rdiff_min() {
        for pid_vcc in [true, false] {
            let mut config = Config::new();
            config.set_tg_control_pid_vcc(pid_vcc);
            let (sys0, _, _, _) = Samples::simple_linear_problem(Sym::No);
            let (sys1, _, _, _) = Samples::simple_linear_problem(Sym::No);
            let mut s0 = Solver::new(&config, sys0).unwrap();
            let mut s1 = Solver::new(&config, sys1).unwrap();
            s0.work.h = 0.1;
            s1.work.h = 0.1;
            let h_zero = s0.adapt_stepsize(0.0);
            let h_small = s1.adapt_stepsize(2e-6);
            let max_growth = 1.0 + f64::atan(config.tg_control_rho_for_tiny_rdiff - 1.0);
            assert!(h_small <= h_zero + 1e-15, "pid_vcc={pid_vcc}: h_small={h_small:e} > h_zero={h_zero:e}");
            approx_eq(h_zero, 0.1 * max_growth, 1e-15);
            approx_eq(h_small, h_zero, 1e-15);
        }
    }
}
