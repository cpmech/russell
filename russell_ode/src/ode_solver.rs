#![allow(unused)]

use crate::{Func, JacF, Method, OdeSolParams, Output, RkWork};
use russell_lab::Vector;
use russell_sparse::CooMatrix;

// Solver implements an ODE solver
struct Solver<'a> {
    // structures
    conf: &'a OdeSolParams, // configuration parameters
    out: Output<'a>,        // output handler
    // stat: &'a Stat,         // statistics

    // problem definition
    ndim: usize, // size of y
    fcn: Func,   // dy/dx := f(x,y)
    jac: JacF,   // Jacobian: df/dy

    // method, info and workspace
    // rkm: OdeMethod,   // Runge-Kutta method
    fixed_only: bool, // method can only be used with fixed steps
    implicit: bool,   // method is implicit
    work: RkWork,     // Runge-Kutta workspace
}

impl<'a> Solver<'a> {
    fn new(ndim: usize, conf: &'a OdeSolParams, fcn: Func, jac: JacF, m: &'a CooMatrix) -> Self {
        // main
        let mut solver = Solver {
            conf,
            out: Output::new(ndim, conf),
            // stat: &Stat::new(conf.ls_kind, false), // assuming ls_kind is a field in Config
            ndim,
            fcn,
            jac,
            // rkm: RkMethod::new(conf.method),
            fixed_only: false,          // to be updated based on method info
            implicit: false,            // to be updated based on method info
            work: RkWork::new(0, ndim), // to be updated based on method info
        };

        // information
        // let (fixed_only, implicit, nstg) = solver.rkm.info();
        // solver.fixed_only = fixed_only;
        // solver.implicit = implicit;

        // stat
        // solver.stat = &Stat::new(conf.ls_kind, solver.implicit);

        // workspace
        // solver.work = &RkWork::new(nstg, solver.ndim);

        // initialize method
        // solver.rkm.init(ndim, conf, solver.work, solver.stat, fcn, jac, m);

        // connect dense output function
        // if solver.out != &Output::default() {
        //     solver.out.dout = solver.rkm.dense_out;
        // }
        solver
    }
}

impl<'a> Solver<'a> {
    fn solve(&mut self, y: &Vector, x: f64, xf: f64) {
        // benchmark
        // let start_time = Instant::now();
        // defer(|| self.stat.update_nanoseconds_total(start_time));

        // check
        if xf < x {
            panic!("xf={} must be greater than x={}", xf, x);
        }
        if self.fixed_only && !self.conf.fixed {
            panic!(
                "method {} can only be used with fixed steps. make sure to call conf.set_fixed_h > 0",
                "self.conf.method"
            );
        }

        // initial step size
        self.work.h = xf - x;
        if self.conf.fixed {
            self.work.h = self.conf.fixedH;
        } else {
            self.work.h = self.work.h.min(self.conf.IniH);
        }

        // stat and output
        // self.stat.reset();
        // self.stat.hopt = self.work.h;
        // if let Some(out) = &self.out {
        //     let stop = out.execute(0, false, self.work.rs, self.work.h, x, y);
        //     if stop {
        //         return;
        //     }
        // }

        // set control flags
        self.work.first = true;

        // first scaling variable
        // vec_scale_abs(&mut self.work.scal, &self.conf.atol, &self.conf.rtol, y); // scal = atol + rtol * abs(y)

        // make sure that final x is equal to xf in the end
        // defer(|| {
        //     if (x - xf).abs() > 1e-10 {
        //         println!("warning: |x - xf| = {} > 1e-8", (x - xf).abs());
        //     }
        // });

        // fixed steps //////////////////////////////
        if self.conf.fixed {
            let mut istep = 1;
            if self.conf.Verbose {
                // io::pfgreen!("x = {}\n", x);
                // io::pf!("y = {}\n", y);
            }
            for n in 0..self.conf.fixedNsteps {
                // if self.implicit && self.jac.is_none() {
                // f0 for numerical Jacobian
                // self.stat.nfeval += 1;
                // self.fcn(&mut self.work.f0, &self.work.h, x, y);
                // }
                // self.rkm.step(x, y);
                // self.stat.nsteps += 1;
                // self.work.first = false;
                // x = f64::from(n + 1) * self.work.h;
                // self.rkm.accept(y, x);
                // if let Some(out) = &self.out {
                //     let stop = out.execute(istep, false, &self.work.rs, &self.work.h, x, y);
                //     if stop {
                //         return;
                //     }
                // }
                // if self.conf.verbose {
                //     io::pfgreen!("x = {}\n", x);
                //     io::pf!("y = {}\n", y);
                // }
                istep += 1;
            }
            return;
        }

        // variable steps //////////////////////////////

        // control variables
        self.work.reuse_jac_and_dec_once = false;
        self.work.reuse_jac_once = false;
        self.work.jac_is_ok = false;
        self.work.h_prev = self.work.h;
        self.work.nit = 0;
        self.work.eta = 1.0;
        self.work.theta = self.conf.ThetaMax;
        self.work.dvfac = 0.0;
        self.work.diverg = false;
        self.work.reject = false;
        self.work.rerr_prev = 1e-4;
        self.work.stiff_yes = 0;
        self.work.stiff_not = 0;

        // first function evaluation
        // self.stat.nfeval += 1;
        // self.fcn(&mut self.work.f0, self.work.h, x, y); // f0 := f(x,y)

        // time loop
        let mut x = x;
        let delta_x = xf - x;
        let mut dxmax: f64 = 0.0;
        let mut xstep: f64 = 0.0;
        let mut dxnew: f64 = 0.0;
        let mut dxratio: f64 = 0.0;
        let mut last: bool = false;
        let mut failed: bool = false;
        while x < xf {
            dxmax = delta_x;
            xstep = x + delta_x;
            failed = false;
            for iss in 0..=self.conf.NmaxSS {
                // total number of substeps
                // self.stat.nsteps += 1;

                // error: did not converge
                // if iss == self.conf.nmax_ss {
                //     failed = true;
                //     break;
                // }

                // converged?
                if x - xstep >= 0.0 {
                    break;
                }

                // step update
                // let start_time_step = Instant::now();
                // self.rkm.step(x, y);
                // self.stat.update_nanoseconds_step(start_time_step);

                // iterations diverging ?
                if self.work.diverg {
                    self.work.diverg = false;
                    self.work.reject = true;
                    last = false;
                    self.work.h *= self.work.dvfac;
                    continue;
                }

                // accepted
                if self.work.rerr < 1.0 {
                    // set flags
                    // self.stat.naccepted += 1;
                    self.work.first = false;
                    self.work.jac_is_ok = false;

                    // stiffness detection
                    // if self.conf.stiff_nstp > 0 {
                    //     if self.stat.naccepted % self.conf.stiff_nstp == 0 || self.work.stiff_yes > 0 {
                    //         if self.work.rs > self.conf.stiff_rs_max {
                    //             self.work.stiff_not = 0;
                    //             self.work.stiff_yes += 1;
                    //             if self.work.stiff_yes == self.conf.stiff_nyes {
                    //                 println!("stiff step detected @ x = {}", x);
                    //             }
                    //         } else {
                    //             self.work.stiff_not += 1;
                    //             if self.work.stiff_not == self.conf.stiff_nnot {
                    //                 self.work.stiff_yes = 0;
                    //             }
                    //         }
                    //     }
                    // }

                    // update x and y
                    // dxnew = self.rkm.accept(y, x);
                    x += self.work.h;

                    // output
                    // if let Some(out) = &self.out {
                    //     let stop = out.execute(self.stat.naccepted, last, self.work.rs, self.work.h, x, y);
                    //     if stop {
                    //         return;
                    //     }
                    // }

                    // converged ?
                    // if last {
                    //     self.stat.hopt = self.work.h; // optimal stepsize
                    //     break;
                    // }

                    // save previous stepsize and relative error
                    self.work.h_prev = self.work.h;
                    // self.work.rerr_prev = self.conf.rerr_prev_min.max(self.work.rerr);

                    // calc new scal and f0
                    if self.implicit {
                        // vec_scale_abs(&mut self.work.scal, &self.conf.atol, &self.conf.rtol, y);
                        // self.stat.nfeval += 1;
                        // self.fcn(&mut self.work.f0, self.work.h, x, y); // f0 := f(x,y)
                    }

                    // check new step size
                    dxnew = f64::min(dxnew, dxmax);
                    if self.work.reject {
                        dxnew = self.work.h.min(dxnew);
                    }
                    self.work.reject = false;

                    // do not reuse current Jacobian and decomposition by default
                    self.work.reuse_jac_and_dec_once = false;

                    // last step ?
                    if x + dxnew - xstep >= 0.0 {
                        last = true;
                        self.work.h = xstep - x;
                    } else {
                        if self.implicit {
                            // dxratio = dxnew / self.work.h;
                            // self.work.reuse_jac_and_dec_once = self.work.theta <= self.conf.theta_max
                            //     && dxratio >= self.conf.c1h
                            //     && dxratio <= self.conf.c2h;
                            // if !self.work.reuse_jac_and_dec_once {
                            //     self.work.h = dxnew;
                            // }
                        } else {
                            self.work.h = dxnew;
                        }
                    }

                    // check θ to decide if at least the Jacobian can be reused
                    if self.implicit {
                        // if !self.work.reuse_jac_and_dec_once {
                        //     self.work.reuse_jac_once = self.work.theta <= self.conf.theta_max;
                        // }
                    }
                } else {
                    // set flags
                    // if self.stat.naccepted > 0 {
                    //     self.stat.nrejected += 1;
                    // }
                    self.work.reject = true;
                    last = false;

                    // compute next stepsize
                    // dxnew = self.rkm.reject();

                    // new step size
                    // if self.work.first && self.conf.mfirst_rej > 0 {
                    //     self.work.h = self.conf.mfirst_rej * self.work.h;
                    // } else {
                    //     self.work.h = dxnew;
                    // }

                    // last step
                    if x + self.work.h > xstep {
                        self.work.h = xstep - x;
                    }
                }
            }

            // sub-stepping failed
            if failed {
                panic!("substepping did not converge after {} steps", self.conf.NmaxSS);
                break;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_works() {}
}
