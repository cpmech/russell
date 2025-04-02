#![allow(unused)]

use super::{NlConfig, NlMethod, NlSolverTrait, NlStop, NlSystem};
use super::{Output, SolverArclength, SolverNatural, Stats, Workspace};
use crate::{State, StrError};
use russell_lab::{vec_all_finite, Vector};

/// Default number of steps
pub const N_EQUAL_STEPS: usize = 10;

pub struct NlSolver<'a, A> {
    /// Configuration options
    config: NlConfig,

    /// Dimension of the ODE system
    ndim: usize,

    /// Holds a pointer to the actual ODE system solver
    actual: Box<dyn NlSolverTrait<A> + 'a>,

    /// Holds statistics, benchmarking and "work" variables
    work: Workspace<'a>,

    /// Assists in generating the output of results
    output: Output<'a, A>,

    /// Indicates whether the output is enabled or not
    output_enabled: bool,
}

impl<'a, A> NlSolver<'a, A> {
    /// Allocates a new instance
    pub fn new(config: NlConfig, system: NlSystem<'a, A>) -> Result<Self, StrError>
    where
        A: 'a,
    {
        config.validate()?;
        let ndim = system.ndim;
        let work = Workspace::new(&config, &system);
        let actual: Box<dyn NlSolverTrait<A>> = match config.method {
            NlMethod::Arclength => Box::new(SolverArclength::new(config, system.clone())),
            NlMethod::Natural => Box::new(SolverNatural::new(config, system.clone())),
        };
        Ok(NlSolver {
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
    /// * `u0` -- the initial value of the vector of unknowns
    /// * `l0` -- the initial value of λ
    /// * `l1` -- the final value of λ
    /// * `h_equal` -- a constant stepsize for solving with equal-steps; otherwise,
    ///   variable step sizes are automatically calculated (auto mode).
    pub fn solve(
        &mut self,
        u0: &mut Vector,
        l0: &mut f64,
        stop: NlStop,
        h_equal: Option<f64>,
        args: &mut A,
    ) -> Result<(), StrError> {
        // check data
        if u0.dim() != self.ndim {
            return Err("u0.dim() must be equal to ndim");
        }
        match stop {
            NlStop::Lambda(l1) => {
                if l1 <= *l0 {
                    return Err("Stopping criterion error: l1 must be greater than l0");
                }
            }
            NlStop::Steps(n) => {
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
                    NlStop::Lambda(l1) => {
                        let n = f64::ceil((l1 - *l0) / h_eq) as usize;
                        (l1 - *l0) / (n as f64)
                    }
                    NlStop::Steps(_) => h_eq,
                };
                (false, h0)
            }
            None => {
                // automatic substepping
                let h0 = match stop {
                    NlStop::Lambda(l1) => f64::min(self.config.h_ini, l1 - *l0),
                    NlStop::Steps(_) => self.config.h_ini,
                };
                (true, h0)
            }
        };
        assert!(h_ini > 0.0);

        // reset variables
        self.work.reset(h_ini, self.config.rel_error_prev_min);

        // current state
        let mut state = State {
            u: u0,
            l: l0,
            s: 0.0,
            h: h_ini,
        };

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
                NlStop::Lambda(l1) => f64::ceil((l1 - *state.l) / h_ini) as usize,
                NlStop::Steps(n) => n,
            };
            for _ in 0..nstep {
                self.work.stats.sw_step.reset();

                // step
                self.work.stats.n_steps += 1;
                self.actual.step(&mut self.work, &mut state, args, auto)?;

                // update u and λ
                self.work.stats.n_accepted += 1;
                self.actual.accept(&mut self.work, &mut state, args);

                // check for anomalies
                vec_all_finite(state.u, self.config.verbose)?;

                // output
                if self.output_enabled {
                    let terminate = self.output.execute(&self.work, &state, args)?;
                    if terminate {
                        self.work.stats.stop_sw_step();
                        self.work.stats.stop_sw_total();
                        return Ok(());
                    }
                }
                self.work.stats.stop_sw_step();
            }
            if self.output_enabled {
                self.output.last()?;
            }
            self.work.stats.stop_sw_total();
            return Ok(());
        }

        // variable steps: control variables
        let mut success = false;
        let mut last_step = false;

        // variable stepping loop
        for step in 0..self.config.n_step_max {
            self.work.stats.sw_step.reset();

            // check final stepsize and stopping criterion
            let h_final = match stop {
                NlStop::Lambda(l1) => {
                    let dl = l1 - *state.l;
                    if dl <= 10.0 * f64::EPSILON {
                        success = true;
                        self.work.stats.stop_sw_step();
                        break;
                    }
                    dl
                }
                NlStop::Steps(n) => {
                    if step >= n {
                        success = true;
                        self.work.stats.stop_sw_step();
                        break;
                    }
                    self.work.h_new
                }
            };

            // update and check the stepsize
            state.h = f64::min(self.work.h_new, h_final);
            if state.h <= 10.0 * f64::EPSILON {
                return Err("the stepsize becomes too small");
            }

            // perform the step calculations
            self.work.stats.n_steps += 1;
            self.actual.step(&mut self.work, &mut state, args, auto)?;

            // handle diverging iterations
            if self.work.iterations_diverging {
                self.work.iterations_diverging = false;
                self.work.follows_reject_step = true;
                last_step = false;
                self.work.h_new = state.h * self.work.h_multiplier_diverging;
                continue;
            }

            // accept step
            if self.work.rel_error < 1.0 {
                // update u and λ
                self.work.stats.n_accepted += 1;
                self.actual.accept(&mut self.work, &mut state, args);

                // check for anomalies
                vec_all_finite(state.u, self.config.verbose)?;

                // do not allow h to grow if previous step was a reject
                if self.work.follows_reject_step {
                    self.work.h_new = f64::min(self.work.h_new, state.h);
                }
                self.work.follows_reject_step = false;

                // save previous stepsize, relative error, and accepted/suggested stepsize
                self.work.h_prev = state.h;
                self.work.rel_error_prev = f64::max(self.config.rel_error_prev_min, self.work.rel_error);
                self.work.stats.h_accepted = self.work.h_new;

                // output
                if self.output_enabled {
                    let terminate = self.output.execute(&self.work, &state, args)?;
                    if terminate {
                        self.work.stats.stop_sw_step();
                        self.work.stats.stop_sw_total();
                        return Ok(());
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
                    NlStop::Lambda(l1) => last_step = *state.l + self.work.h_new >= l1,
                    NlStop::Steps(n) => last_step = step + 1 >= n,
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
                    self.actual.reject(&mut self.work, state.h, args, auto);
                }
            }
        }

        // last output
        if self.output_enabled {
            self.output.last()?;
        }

        // print footer and handle errors
        self.work.log.footer(&self.work.stats);

        // done
        self.work.stats.stop_sw_total();
        if success {
            Ok(())
        } else {
            Err("variable stepping did not converge")
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
    use super::NlSolver;
    use crate::{NlConfig, NlMethod, NlStop, Samples};
    use russell_lab::{vec_approx_eq, Vector};

    #[test]
    fn new_captures_errors() {
        let (system, _u_trial, _u_ref, _args) = Samples::simple_two_equations();
        let mut config = NlConfig::new(NlMethod::Natural);
        config.m_max = 0.0; // wrong
        assert_eq!(
            NlSolver::new(config, system).err(),
            Some("requirement: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
    }

    #[test]
    fn solve_captures_errors() {
        let (system, _u_trial, _u_ref, mut args) = Samples::simple_two_equations();
        let mut l0 = 0.0;
        let ndim = system.ndim;
        let config = NlConfig::new(NlMethod::Natural);
        let mut solver = NlSolver::new(config, system).unwrap();
        let mut u0 = Vector::new(ndim + 1); // wrong dim
        assert_eq!(
            solver
                .solve(&mut u0, &mut l0, NlStop::Lambda(1.0), None, &mut args)
                .err(),
            Some("u0.dim() must be equal to ndim")
        );
        let mut y0 = Vector::new(ndim);
        assert_eq!(
            solver
                .solve(&mut y0, &mut l0, NlStop::Lambda(0.0), None, &mut args)
                .err(),
            Some("l1 must be greater than l0")
        );
        let h_equal = Some(f64::EPSILON); // will cause an error
        assert_eq!(
            solver
                .solve(&mut y0, &mut l0, NlStop::Lambda(1.0), h_equal, &mut args)
                .err(),
            Some("h_equal must be ≥ 10.0 * f64::EPSILON")
        );
    }

    #[test]
    fn lack_of_convergence_is_captured() {
        let (system, mut u0, _u_ref, mut args) = Samples::simple_two_equations();
        let mut l0 = 0.0;
        let mut config = NlConfig::new(NlMethod::Natural);
        config.n_step_max = 1; // will make the solver to fail (too few steps)
        let mut solver = NlSolver::new(config, system).unwrap();
        assert_eq!(
            solver
                .solve(&mut u0, &mut l0, NlStop::Lambda(1.0), None, &mut args)
                .err(),
            Some("TODO") // Some("variable stepping did not converge")
        );
    }

    #[test]
    fn solve_with_n_equal_steps_works() {
        // solve the nonlinear system (will run with N_EQUAL_STEPS)
        let (system, mut u, u_ref, mut args) = Samples::simple_two_equations();
        let config = NlConfig::new(NlMethod::Natural);
        let mut solver = NlSolver::new(config, system).unwrap();
        // solver.solve(&mut u, 0.0, 1.0, None, &mut args).unwrap();
        // vec_approx_eq(&u, &u_ref, 1e-15);
    }
}
