#![allow(unused)]

use super::{NlMethod, NlParams, NlSolverTrait, NlSystem, Output, Stats, Workspace};
use super::{SolverArclength, SolverParametric, SolverSimple};
use crate::{State, StrError};
use russell_lab::{vec_all_finite, Vector};

/// Default number of steps
pub const N_EQUAL_STEPS: usize = 10;

pub struct NlSolver<'a, A> {
    /// Holds the parameters
    params: NlParams,

    /// Dimension of the ODE system
    ndim: usize,

    /// Holds a pointer to the actual ODE system solver
    actual: Box<dyn NlSolverTrait<A> + 'a>,

    /// Holds statistics, benchmarking and "work" variables
    work: Workspace<'a>,

    /// Assists in generating the output of results (steps or dense)
    output: Output<'a, A>,

    /// Indicates whether the output is enabled or not
    output_enabled: bool,
}

impl<'a, A> NlSolver<'a, A> {
    /// Allocates a new instance
    pub fn new(params: NlParams, system: NlSystem<'a, A>) -> Result<Self, StrError>
    where
        A: 'a,
    {
        params.validate()?;
        let ndim = system.ndim;
        let work = Workspace::new(&params, &system);
        let actual: Box<dyn NlSolverTrait<A>> = match params.method {
            NlMethod::Arclength => Box::new(SolverArclength::new(params, system.clone())),
            NlMethod::Parametric => Box::new(SolverParametric::new(params, system.clone())),
            NlMethod::Simple => Box::new(SolverSimple::new(params, system.clone())),
        };
        Ok(NlSolver {
            params,
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
    /// Simple:     G(u) = 0
    /// Parametric: G(u, λ) = 0
    /// Arclength:  G(u(s), λ(s)) = 0
    /// ```
    ///
    /// # Input
    ///
    /// * `u0` -- the initial value of the vector of unknowns
    /// * `l0` -- the initial value of λ
    /// * `l1` -- the final value of λ
    /// * `h_equal` -- a constant stepsize for solving with equal-steps; otherwise,
    ///   variable step sizes are automatically calculated.
    pub fn solve(
        &mut self,
        u0: &mut Vector,
        l0: &mut f64,
        l1: f64,
        h_equal: Option<f64>,
        args: &mut A,
    ) -> Result<(), StrError> {
        // check data
        if u0.dim() != self.ndim {
            return Err("u0.dim() must be equal to ndim");
        }
        if l1 <= *l0 {
            return Err("l1 must be greater than l0");
        }

        // initial stepsize
        let (equal_stepping, mut h) = match h_equal {
            Some(h_eq) => {
                if h_eq < 10.0 * f64::EPSILON {
                    return Err("h_equal must be ≥ 10.0 * f64::EPSILON");
                }
                let n = f64::ceil((l1 - *l0) / h_eq) as usize;
                let h = (l1 - *l0) / (n as f64);
                (true, h)
            }
            None => {
                let h = f64::min(self.params.h_ini, l1 - *l0);
                (false, h)
            }
        };
        assert!(h > 0.0);

        // reset variables
        self.work.reset(h, self.params.rel_error_prev_min);

        // current state
        let mut state = State {
            u: u0,
            l: l0,
            s: 0.0,
            h,
        };

        // first output
        if self.output_enabled {
            self.output.initialize();
            let stop = self.output.execute(&self.work, &state, args)?;
            if stop {
                return Ok(());
            }
        }

        // message
        self.work.log.header();

        // equal-stepping loop
        if equal_stepping {
            let nstep = f64::ceil((l1 - *state.l) / h) as usize;
            for _ in 0..nstep {
                self.work.stats.sw_step.reset();

                // step
                self.work.stats.n_steps += 1;
                self.actual.step(&mut self.work, &mut state, args, false)?;
                self.work.stats.n_accepted += 1;

                // check for anomalies
                vec_all_finite(state.u, self.params.verbose)?;

                // output
                if self.output_enabled {
                    let stop = self.output.execute(&self.work, &state, args)?;
                    if stop {
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
        for _ in 0..self.params.n_step_max {
            self.work.stats.sw_step.reset();

            // converged?
            let dx = l1 - *state.l;
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
            self.actual.step(&mut self.work, &mut state, args, true)?;

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

                // check for anomalies
                vec_all_finite(state.u, self.params.verbose)?;

                // do not allow h to grow if previous step was a reject
                if self.work.follows_reject_step {
                    self.work.h_new = f64::min(self.work.h_new, h);
                }
                self.work.follows_reject_step = false;

                // save previous stepsize, relative error, and accepted/suggested stepsize
                self.work.h_prev = h;
                self.work.rel_error_prev = f64::max(self.params.rel_error_prev_min, self.work.rel_error);
                self.work.stats.h_accepted = self.work.h_new;

                // output
                if self.output_enabled {
                    let stop = self.output.execute(&self.work, &state, args)?;
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
                if *state.l + self.work.h_new >= l1 {
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
                if self.work.stats.n_accepted == 0 && self.params.m_first_reject > 0.0 {
                    self.work.h_new = h * self.params.m_first_reject;
                } else {
                    // self.actual.reject(&mut self.work, h);
                }
                panic!("TODO: handle reject");
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

    /// Returns an access to the output during accepted steps: x values
    pub fn out_step_x(&self) -> &Vec<f64> {
        &self.output.l
    }

    /// Returns an access to the output during accepted steps: y values
    ///
    /// # Panics
    ///
    /// A panic will occur if `m` is out of range
    pub fn out_step_y(&self, m: usize) -> &Vec<f64> {
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
    use crate::{NlMethod, NlParams, Samples};
    use russell_lab::{vec_approx_eq, Vector};

    #[test]
    fn new_captures_errors() {
        let (system, _u_trial, _u_ref, _args) = Samples::simple_two_equations();
        let mut params = NlParams::new(NlMethod::Simple);
        params.m_max = 0.0; // wrong
        assert_eq!(
            NlSolver::new(params, system).err(),
            Some("parameter must satisfy: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
    }

    #[test]
    fn solve_captures_errors() {
        let (system, _u_trial, _u_ref, mut args) = Samples::simple_two_equations();
        let mut l0 = 0.0;
        let ndim = system.ndim;
        let params = NlParams::new(NlMethod::Simple);
        let mut solver = NlSolver::new(params, system).unwrap();
        let mut u0 = Vector::new(ndim + 1); // wrong dim
        assert_eq!(
            solver.solve(&mut u0, &mut l0, 1.0, None, &mut args).err(),
            Some("u0.dim() must be equal to ndim")
        );
        let mut y0 = Vector::new(ndim);
        assert_eq!(
            solver.solve(&mut y0, &mut l0, 0.0, None, &mut args).err(),
            Some("l1 must be greater than l0")
        );
        let h_equal = Some(f64::EPSILON); // will cause an error
        assert_eq!(
            solver.solve(&mut y0, &mut l0, 1.0, h_equal, &mut args).err(),
            Some("h_equal must be ≥ 10.0 * f64::EPSILON")
        );
    }

    #[test]
    fn lack_of_convergence_is_captured() {
        let (system, mut u0, _u_ref, mut args) = Samples::simple_two_equations();
        let mut l0 = 0.0;
        let mut params = NlParams::new(NlMethod::Simple);
        params.n_step_max = 1; // will make the solver to fail (too few steps)
        let mut solver = NlSolver::new(params, system).unwrap();
        assert_eq!(
            solver.solve(&mut u0, &mut l0, 1.0, None, &mut args).err(),
            Some("TODO") // Some("variable stepping did not converge")
        );
    }

    #[test]
    fn solve_with_n_equal_steps_works() {
        // solve the nonlinear system (will run with N_EQUAL_STEPS)
        let (system, mut u, u_ref, mut args) = Samples::simple_two_equations();
        let params = NlParams::new(NlMethod::Simple);
        let mut solver = NlSolver::new(params, system).unwrap();
        // solver.solve(&mut u, 0.0, 1.0, None, &mut args).unwrap();
        // vec_approx_eq(&u, &u_ref, 1e-15);
    }
}
