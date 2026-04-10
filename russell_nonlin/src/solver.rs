use super::{Config, DeltaLambda, IniDir, Method, SolverTrait, Stop, System};
use super::{Output, SolverArclength, SolverNatural, Stats, Status, Workspace, CONFIG_H_MIN};
use crate::StrError;
use russell_lab::{vec_all_finite, Vector};

/// Default number of steps
pub const N_EQUAL_STEPS: usize = 10;

pub struct Solver<'a, A> {
    /// Configuration options
    config: &'a Config,

    /// Dimension of the ODE system
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
    pub fn new(config: &'a Config, system: System<'a, A>) -> Result<Self, StrError>
    where
        A: 'a,
    {
        config.validate()?;
        let ndim = system.ndim;
        let work = Workspace::new(&config, &system);
        let actual: Box<dyn SolverTrait<A>> = match config.method {
            Method::Arclength => Box::new(SolverArclength::new(&config, system.clone())?),
            Method::Natural => Box::new(SolverNatural::new(&config, system.clone())?),
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
    /// * `(u, l)` -- the initial state `(u₀, λ₀)` with a non-singular `Gu₀ = ∂u/∂λ|₀` (Jacobian) matrix.
    ///    The state will be updated with the new solution (u, λ) util a stop criterion is reached.
    /// * `dir` -- the direction to follow on the solution branch (pseudo-arclength method).
    /// * `stop` -- stop criterion (e.g, either a final λ value or a number of steps)
    /// * `ddl` -- specifies how Δλ is adjusted
    /// * `auto` -- defines the stepsize control method (variable stepsize control or fixed stepsize)
    /// * `output` -- output object to write results files, record the results, or execute a callback function
    ///
    /// # Returns
    ///
    /// Returns the status code.
    pub fn solve<'b>(
        &mut self,
        args: &mut A,
        u: &mut Vector,
        l: &mut f64,
        dir: IniDir,
        stop: Stop,
        ddl: DeltaLambda,
        mut output: Option<&mut Output<'b, A>>,
    ) -> Result<Status, StrError> {
        // validate input
        if u.dim() != self.ndim {
            return Err("u.dim() must be equal to ndim");
        }
        stop.validate(u, *l)?;

        // reset stats and flags
        self.work.reset_stats_and_flags(ddl.auto);

        // calculate the default initial Δλ
        let ddl_ini = ddl.ini(&stop, *l)?;

        // perform initialization (compute the actual initial stepsize and tangent vector)
        self.actual.initialize(&mut self.work, ddl_ini, u, *l, dir, args)?;

        // first output
        if let Some(out) = output.as_deref_mut() {
            let stop_gracefully = out.execute(&self.work, u, *l, args)?;
            if stop_gracefully {
                return Ok(Status::Success);
            }
        }

        // message
        if self.config.verbose_header_footer {
            self.work.log.header();
        }

        // default status
        let mut status = Status::Success;

        // perform continuation
        if !ddl.auto {
            // constant or list-based stepsize
            let n_step_max = if ddl.list.len() > 0 {
                ddl.list.len()
            } else {
                self.config.n_step_max
            };
            for i in 0..n_step_max {
                // log
                self.work.stats.sw_step.reset();
                self.work.log.step(self.work.h, *l, false);

                // step
                self.work.stats.n_steps += 1;
                status = self.actual.step(&mut self.work, u, *l, stop, args)?;

                // handle failures
                if status.failure() {
                    break;
                }

                // update u and λ
                self.work.stats.n_accepted += 1;
                self.actual.accept(&mut self.work, u, l, args)?;

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
                self.work.stats.h_accepted = self.work.h;

                // check for anomalies
                vec_all_finite(&u, self.config.verbose)?;

                // output
                if let Some(out) = output.as_deref_mut() {
                    let stop_gracefully = out.execute(&self.work, u, *l, args)?;
                    if stop_gracefully {
                        self.work.stats.stop_sw_step();
                        break;
                    }
                }

                // exit point
                self.work.stats.stop_sw_step();
                if stop.now(i, u, *l) {
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

                // check for failures
                if status.failure() {
                    if status.try_again() {
                        self.work.n_continued_failure += 1;
                        self.work.follows_failure = true;
                    } else {
                        break;
                    }
                }

                // handle continued failure
                if self.work.n_continued_failure >= self.config.n_cont_failure_max {
                    status = Status::ContinuedFailure;
                    break;
                }

                // handle continued rejections
                if self.work.n_continued_rejection >= self.config.n_cont_rejection_max {
                    status = Status::ContinuedRejection;
                    break;
                }

                // reject or accept
                if status.failure() {
                    // set flags
                    self.work.stats.n_rejected += 1;
                    self.work.follows_rejection = true;

                    // perform the reject operations (e.g., restore external vars) and recompute stepsize
                    self.actual.reject(&mut self.work, args);

                    // adapt stepsize
                    self.work.h *= self.config.m_failure;
                } else {
                    // update u and λ
                    self.work.stats.n_accepted += 1;
                    let rdiff = self.actual.accept(&mut self.work, u, l, args)?;

                    // check for anomalies
                    vec_all_finite(&u, self.config.verbose)?;

                    // exit point: target u or λ reached
                    if self.work.target_reached {
                        break;
                    }

                    // adapt stepsize
                    let mut h_estimate = self.adapt_stepsize(rdiff);
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
                        let stop_gracefully = out.execute(&self.work, u, *l, args)?;
                        if stop_gracefully {
                            self.work.stats.stop_sw_step();
                            break;
                        }
                    }

                    // exit point
                    self.work.stats.stop_sw_step();
                    if stop.now(i, u, *l) {
                        break;
                    }
                }

                // check allowed stepsize change
                if self.work.h < CONFIG_H_MIN {
                    status = Status::SmallStepsize;
                    break;
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

    /// Logs the header
    pub fn log_header(&mut self) {
        self.work.log.header();
    }

    /// Logs the footer
    pub fn log_footer(&mut self) {
        self.work.log.footer(&self.work.stats, &Status::Success).unwrap();
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
        let rdiff = if rdiff < self.config.tg_control_rdiff_zero {
            0.0
        } else {
            rdiff
        };

        // calculates the the relative changes on the tangent vector
        let rho = if self.config.tg_control_enabled {
            if rdiff == 0.0 {
                self.config.tg_control_rho_for_tiny_rdiff
            } else if self.config.tg_control_pid_vcc {
                const KP: f64 = 0.075;
                const KI: f64 = 0.175;
                const KD: f64 = 0.01;
                let mut p = 1.0;
                let mut d = 1.0;
                let i = 1.0 / rdiff;
                if self.work.stats.n_accepted > 1 {
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
    use crate::{Config, DeltaLambda, IniDir, Samples, Status, Stop};
    use russell_lab::{vec_approx_eq, Vector};

    #[test]
    fn new_captures_errors() {
        let (system, _, _, _) = Samples::two_eq_ref();
        let mut config = Config::new();
        config.tol_abs_residual = 0.0; // wrong
        assert_eq!(
            Solver::new(&config, system).err(),
            Some("requirement: tol_abs_residual ≥ 1e-12")
        );
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
                    DeltaLambda::auto(1e-4),
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
                    DeltaLambda::auto(1e-4),
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
                    DeltaLambda::constant(f64::EPSILON), // will cause an error
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
        config.n_step_max = 1; // will make the solver to fail (too few steps)
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
                    DeltaLambda::auto(1e-4),
                    None,
                )
                .unwrap(),
            Status::Success
        );
    }

    #[test]
    fn solve_with_one_step_works_fixed() {
        let (system, mut u, u_ref, mut args) = Samples::two_eq_ref();
        let mut config = Config::new();
        config.set_verbose(false, true, true).set_tol_delta(1e-12, 1e-10);
        let mut solver = Solver::new(&config, system).unwrap();
        let mut l = 0.0;
        solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::Steps(1),
                DeltaLambda::constant(1.0),
                None,
            )
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
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::Steps(1),
                DeltaLambda::auto(1e-4),
                None,
            )
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
}
