#![allow(non_snake_case)]

use crate::{constants::*, JacF};
use crate::{Configuration, Func, Information, Method, RungeKuttaTrait, Statistics, StrError, Workspace};
use russell_lab::{vec_add, vec_copy, vec_update, Matrix, Vector};

pub struct ExplicitRungeKutta<A> {
    conf: Configuration, // configuration
    info: Information,

    // constants
    A: Matrix, // A coefficients
    B: Vector, // B coefficients
    C: Vector, // C coefficients

    E: Option<Vector>, // (embedded) error coefficients. difference between B and Be: e = b - be (if be is not nil)

    Ad: Option<Matrix>, // A coefficients for dense output
    Cd: Option<Vector>, // C coefficients for dense output
    D: Option<Matrix>,  // dense output coefficients. [may be nil]

    P: usize, // order of y1 (corresponding to b)
    Q: usize, // order of error estimator (embedded only); e.g. DoPri5(4) ⇒ q = 4 (=min(order(y1) , order(y1bar))

    // data
    ndim: usize,      // problem dimension
    work: Workspace,  // workspace
    stat: Statistics, // statistics
    fcn: Func<A>,     // dy/dx = f(x, y) function

    // auxiliary
    w: Vector, // local workspace
    n: f64,    // exponent n = 1/(q+1) (or 1/(q+1)-0.75⋅β) of rerrⁿ

    // dense output
    dout: Option<Vec<Vector>>, // dense output coefficients [nstgDense][ndim] (partially allocated by newERK method)
    kd: Option<Vec<Vector>>,   // k values for dense output [nextraKs] (partially allocated by newERK method)
    yd: Option<Vector>,        // y values for dense output (allocated here if len(kd)>0)
}

impl<A> ExplicitRungeKutta<A> {
    /// Allocates a new instance
    pub fn new(conf: Configuration, ndim: usize, function: Func<A>) -> Result<Self, StrError> {
        let info = conf.method.information();
        if info.implicit {
            return Err("the method must not be implicit");
        }
        if conf.method == Method::FwEuler {
            return Err("the method must not be FwEuler");
        }
        #[rustfmt::skip]
        let (A, B, C) = match conf.method {
            Method::Radau5     => panic!("<not available>"),
            Method::BwEuler    => panic!("<not available>"),
            Method::FwEuler    => panic!("<not available>"),
            Method::Rk2        => (Matrix::from(&RUNGE_KUTTA_2_A)     , Vector::from(&RUNGE_KUTTA_2_B)     , Vector::from(&RUNGE_KUTTA_2_C)    ),
            Method::Rk3        => (Matrix::from(&RUNGE_KUTTA_3_A)     , Vector::from(&RUNGE_KUTTA_3_B)     , Vector::from(&RUNGE_KUTTA_3_C)    ),
            Method::Heun3      => (Matrix::from(&HEUN_3_A)            , Vector::from(&HEUN_3_B)            , Vector::from(&HEUN_3_C)           ),
            Method::Rk4        => (Matrix::from(&RUNGE_KUTTA_4_A)     , Vector::from(&RUNGE_KUTTA_4_B)     , Vector::from(&RUNGE_KUTTA_4_C)    ),
            Method::Rk4alt     => (Matrix::from(&RUNGE_KUTTA_ALT_4_A) , Vector::from(&RUNGE_KUTTA_ALT_4_B) , Vector::from(&RUNGE_KUTTA_ALT_4_C)),
            Method::MdEuler    => (Matrix::from(&MODIFIED_EULER_A)    , Vector::from(&MODIFIED_EULER_B)    , Vector::from(&MODIFIED_EULER_C)   ),
            Method::Merson4    => (Matrix::from(&MERSON_4_A)          , Vector::from(&MERSON_4_B)          , Vector::from(&MERSON_4_C)         ),
            Method::Zonneveld4 => (Matrix::from(&ZONNEVELD_4_A)       , Vector::from(&ZONNEVELD_4_B)       , Vector::from(&ZONNEVELD_4_C)      ),
            Method::Fehlberg4  => (Matrix::from(&FEHLBERG_4_A)        , Vector::from(&FEHLBERG_4_B)        , Vector::from(&FEHLBERG_4_C)       ),
            Method::DoPri5     => (Matrix::from(&DORMAND_PRINCE_5_A)  , Vector::from(&DORMAND_PRINCE_5_B)  , Vector::from(&DORMAND_PRINCE_5_C) ),
            Method::Verner6    => (Matrix::from(&VERNER_6_A)          , Vector::from(&VERNER_6_B)          , Vector::from(&VERNER_6_C)         ),
            Method::Fehlberg7  => (Matrix::from(&FEHLBERG_7_A)        , Vector::from(&FEHLBERG_7_B)        , Vector::from(&FEHLBERG_7_C)       ),
            Method::DoPri8     => (Matrix::from(&DORMAND_PRINCE_8_A)  , Vector::from(&DORMAND_PRINCE_8_B)  , Vector::from(&DORMAND_PRINCE_8_C) ),
        };
        #[rustfmt::skip]
        let (Be, E) = if info.embedded {
            match conf.method {
                Method::Radau5     => (None, None),
                Method::BwEuler    => (None, None),
                Method::FwEuler    => (None, None),
                Method::Rk2        => (None, None),
                Method::Rk3        => (None, None),
                Method::Heun3      => (None, None),
                Method::Rk4        => (None, None),
                Method::Rk4alt     => (None, None),
                Method::MdEuler    => (Some(Vector::from(&MODIFIED_EULER_BE))   , Some(Vector::from(&MODIFIED_EULER_E))  ),
                Method::Merson4    => (Some(Vector::from(&ZONNEVELD_4_BE))      , Some(Vector::from(&ZONNEVELD_4_E))     ),
                Method::Zonneveld4 => (Some(Vector::from(&ZONNEVELD_4_BE))      , Some(Vector::from(&ZONNEVELD_4_E))     ),
                Method::Fehlberg4  => (Some(Vector::from(&FEHLBERG_4_BE))       , Some(Vector::from(&FEHLBERG_4_E))      ),
                Method::DoPri5     => (Some(Vector::from(&DORMAND_PRINCE_5_BE)) , Some(Vector::from(&DORMAND_PRINCE_5_E))),
                Method::Verner6    => (Some(Vector::from(&VERNER_6_BE))         , Some(Vector::from(&VERNER_6_E))        ),
                Method::Fehlberg7  => (Some(Vector::from(&FEHLBERG_7_BE))       , Some(Vector::from(&FEHLBERG_7_E))      ),
                Method::DoPri8     => (None, Some(Vector::from(&DORMAND_PRINCE_8_E))),
            }
        } else {
            (None, None)
        };
        let (mut Ad, mut Cd, mut D) = (None, None, None);
        let (mut dout, mut kd, mut yd) = (None, None, None);
        if conf.denseOut && conf.method == Method::DoPri5 {
            D = Some(Matrix::from(&DORMAND_PRINCE_5_D));
            dout = Some(vec![Vector::new(ndim); 5]);
        }
        if conf.denseOut && conf.method == Method::DoPri8 {
            Ad = Some(Matrix::from(&DORMAND_PRINCE_8_AD));
            Cd = Some(Vector::from(&DORMAND_PRINCE_8_CD));
            D = Some(Matrix::from(&DORMAND_PRINCE_8_D));
            dout = Some(vec![Vector::new(ndim); 8]);
            kd = Some(vec![Vector::new(ndim); 3]);
            yd = Some(Vector::new(ndim));
        }
        let lund_stabilization_factor = if conf.StabBeta > 0.0 {
            1.0 / ((info.order_of_estimator + 1) as f64) - conf.StabBeta * conf.stabBetaM
        } else {
            1.0 / ((info.order_of_estimator + 1) as f64)
        };
        let stat = Statistics::new(info.implicit, conf.genie);
        let n_stage = B.dim();
        Ok(ExplicitRungeKutta {
            conf,
            info,
            A,
            B,
            C,
            E,
            Ad,
            Cd,
            D,
            P: info.order,
            Q: info.order_of_estimator,
            ndim,
            work: Workspace::new(n_stage, ndim),
            stat,
            fcn: function,
            w: Vector::new(ndim),
            n: lund_stabilization_factor,
            dout,
            kd,
            yd,
        })
    }

    /// Performs  the next step
    pub fn next_step(&mut self, xa: f64, ya: &Vector, args: &mut A) {
        // auxiliary
        let n_stage = self.work.nstg;
        let h = self.work.h;
        let k = &mut self.work.f;
        let v = &mut self.work.v;
        let dmin = 1.0 / self.conf.Mmin;
        let dmax = 1.0 / self.conf.Mmax;

        // compute k0 (otherwise, use k0 saved in Accept)
        if (self.work.first || !self.info.first_step_same_as_last) && !self.work.reject {
            let u0 = xa + h * self.C[0];
            self.stat.Nfeval += 1;
            (self.fcn)(&mut k[0], h, u0, ya, args); // k0 := f(ui,vi)
        }

        // compute ki
        for i in 1..n_stage {
            let ui = xa + h * self.C[i];
            vec_copy(&mut v[1], &ya).unwrap(); // vi := ya
            for j in 0..i {
                // j goes from 0 to (i-1) because the method is explicit (lower diagonal)
                vec_update(&mut v[i], h * self.A.get(i, j), &k[j]).unwrap();
                // vec_add(&mut v[i], 1.0, &v[i], h * self.A.get(i, j), &k[j]).unwrap();
                // vi += h ⋅ aij ⋅ kj
            }
            self.stat.Nfeval += 1;
            (self.fcn)(&mut k[i], h, ui, &v[i], args); // ki := f(ui,vi)
        }

        // update
        if !self.info.embedded {
            for m in 0..self.ndim {
                self.w[m] = ya[m];
                for i in 0..n_stage {
                    self.w[m] += self.B[i] * k[i][m] * h;
                }
            }
            return;
        }

        // auxiliary
        let ee = self.E.as_ref().unwrap();
        let mut snum = 0.0;
        let mut sden = 0.0;
        let dim = self.ndim as f64;

        // error estimation for Dormand-Prince 8 with 5 and 3 orders
        if self.conf.method == Method::DoPri8 {
            let mut errA = 0.0;
            let mut errB = 0.0;
            let mut err3 = 0.0;
            let mut err5 = 0.0;
            for m in 0..self.ndim {
                self.w[m] = ya[m];
                let mut errA = 0.0;
                let mut errB = 0.0;
                for i in 0..n_stage {
                    self.w[m] += self.B[i] * k[i][m] * h;
                    errA += self.B[i] * k[i][m];
                    errB += ee[i] * k[i][m];
                }
                let sk = self.conf.atol + self.conf.rtol * f64::max(f64::abs(ya[m]), f64::abs(self.w[m]));
                errA -= (DORMAND_PRINCE_8_BHH1 * k[0][m]
                    + DORMAND_PRINCE_8_BHH2 * k[8][m]
                    + DORMAND_PRINCE_8_BHH3 * k[11][m]);
                err3 += (errA / sk) * (errA / sk);
                err5 += (errB / sk) * (errB / sk);
                // stiffness estimation
                let dk = k[n_stage - 1][m] - k[n_stage - 2][m];
                let dv = v[n_stage - 1][m] - v[n_stage - 2][m];
                snum += dk * dk;
                sden += dv * dv;
            }
            let mut den = err5 + 0.01 * err3; // similar to Eq. (10.17) of [1, page 255]
            if den <= 0.0 {
                den = 1.0;
            }
            self.work.rerr = f64::abs(h) * err5 * f64::sqrt(1.0 / (dim * den));
            if sden > 0.0 {
                self.work.rs = h * f64::sqrt(snum / sden);
            }
            return;
        }

        // update, error and stiffness estimation
        let mut sum = 0.0;
        for m in 0..self.ndim {
            self.w[m] = ya[m];
            let mut lerrm = 0.0;
            for i in 0..n_stage {
                let kh = k[i][m] * h;
                self.w[m] += self.B[i] * kh;
                lerrm += ee[i] * kh;
            }
            let sk = self.conf.atol + self.conf.rtol * f64::max(f64::abs(ya[m]), f64::abs(self.w[m]));
            let ratio = lerrm / sk;
            sum += ratio * ratio;
            // stiffness estimation
            let dk = k[n_stage - 1][m] - k[n_stage - 2][m];
            let dv = v[n_stage - 1][m] - v[n_stage - 2][m];
            snum += dk * dk;
            sden += dv * dv;
        }
        self.work.rerr = f64::max(f64::sqrt(sum / dim), 1.0e-10);
        if sden > 0.0 {
            self.work.rs = h * f64::sqrt(snum / sden);
        }
    }

    /// Accepts the update and computes the next stepsize
    ///
    /// Returns `stepsize_new`
    pub fn accept_update(&mut self, y0: &mut Vector, x0: f64, args: &mut A) -> f64 {
        // store data for future dense output (Dormand-Prince 5)
        if self.conf.denseOut && self.conf.method == Method::DoPri5 {
            let dd = self.D.as_ref().unwrap();
            let dout = self.dout.as_mut().unwrap();
            let h = self.work.h;
            let k = &self.work.f;
            for m in 0..self.ndim {
                let ydiff = self.w[m] - y0[m];
                let bspl = h * k[0][m] - ydiff;
                dout[0][m] = y0[m];
                dout[1][m] = ydiff;
                dout[2][m] = bspl;
                dout[3][m] = ydiff - h * k[6][m] - bspl;
                dout[4][m] = dd.get(0, 0) * k[0][m]
                    + dd.get(0, 2) * k[2][m]
                    + dd.get(0, 3) * k[3][m]
                    + dd.get(0, 4) * k[4][m]
                    + dd.get(0, 5) * k[5][m]
                    + dd.get(0, 6) * k[6][m];
                dout[4][m] *= self.work.h;
            }
        }

        // store data for future dense output (Dormand-Prince 8)
        if self.conf.denseOut && self.conf.method == Method::DoPri8 {
            // auxiliary variables
            let aad = self.Ad.as_ref().unwrap();
            let cd = self.Cd.as_ref().unwrap();
            let dd = self.D.as_ref().unwrap();
            let dout = self.dout.as_mut().unwrap();
            let kd = self.kd.as_mut().unwrap();
            let yd = self.yd.as_mut().unwrap();
            let h = self.work.h;
            let k = &self.work.f;

            // first function evaluation
            for m in 0..self.ndim {
                yd[m] = y0[m]
                    + h * (aad.get(0, 0) * k[0][m]
                        + aad.get(0, 6) * k[6][m]
                        + aad.get(0, 7) * k[7][m]
                        + aad.get(0, 8) * k[8][m]
                        + aad.get(0, 9) * k[9][m]
                        + aad.get(0, 10) * k[10][m]
                        + aad.get(0, 11) * k[11][m]
                        + aad.get(0, 12) * k[11][m]);
            }
            self.stat.Nfeval += 1;
            let u = x0 + cd[0] * h;
            (self.fcn)(&mut kd[0], h, u, yd, args);

            // second function evaluation
            for m in 0..self.ndim {
                yd[m] = y0[m]
                    + h * (aad.get(1, 0) * k[0][m]
                        + aad.get(1, 5) * k[5][m]
                        + aad.get(1, 6) * k[6][m]
                        + aad.get(1, 7) * k[7][m]
                        + aad.get(1, 10) * k[10][m]
                        + aad.get(1, 11) * k[11][m]
                        + aad.get(1, 12) * k[11][m]
                        + aad.get(1, 13) * kd[0][m]);
            }
            self.stat.Nfeval += 1;
            let u = x0 + cd[1] * h;
            (self.fcn)(&mut kd[1], h, u, yd, args);

            // next third function evaluation
            for m in 0..self.ndim {
                yd[m] = y0[m]
                    + h * (aad.get(2, 0) * k[0][m]
                        + aad.get(2, 5) * k[5][m]
                        + aad.get(2, 6) * k[6][m]
                        + aad.get(2, 7) * k[7][m]
                        + aad.get(2, 8) * k[8][m]
                        + aad.get(2, 12) * k[11][m]
                        + aad.get(2, 13) * kd[0][m]
                        + aad.get(2, 14) * kd[1][m]);
            }
            self.stat.Nfeval += 1;
            let u = x0 + cd[2] * h;
            (self.fcn)(&mut kd[2], h, u, yd, args);

            // final results
            for m in 0..self.ndim {
                let ydiff = self.w[m] - y0[m];
                let bspl = h * k[0][m] - ydiff;
                dout[0][m] = y0[m];
                dout[1][m] = ydiff;
                dout[2][m] = bspl;
                dout[3][m] = ydiff - h * k[11][m] - bspl;
                dout[4][m] = h
                    * (dd.get(0, 0) * k[0][m]
                        + dd.get(0, 5) * k[5][m]
                        + dd.get(0, 6) * k[6][m]
                        + dd.get(0, 7) * k[7][m]
                        + dd.get(0, 8) * k[8][m]
                        + dd.get(0, 9) * k[9][m]
                        + dd.get(0, 10) * k[10][m]
                        + dd.get(0, 11) * k[11][m]
                        + dd.get(0, 12) * k[11][m]
                        + dd.get(0, 13) * kd[0][m]
                        + dd.get(0, 14) * kd[1][m]
                        + dd.get(0, 15) * kd[2][m]);
                dout[5][m] = h
                    * (dd.get(1, 0) * k[0][m]
                        + dd.get(1, 5) * k[5][m]
                        + dd.get(1, 6) * k[6][m]
                        + dd.get(1, 7) * k[7][m]
                        + dd.get(1, 8) * k[8][m]
                        + dd.get(1, 9) * k[9][m]
                        + dd.get(1, 10) * k[10][m]
                        + dd.get(1, 11) * k[11][m]
                        + dd.get(1, 12) * k[11][m]
                        + dd.get(1, 13) * kd[0][m]
                        + dd.get(1, 14) * kd[1][m]
                        + dd.get(1, 15) * kd[2][m]);
                dout[6][m] = h
                    * (dd.get(2, 0) * k[0][m]
                        + dd.get(2, 5) * k[5][m]
                        + dd.get(2, 6) * k[6][m]
                        + dd.get(2, 7) * k[7][m]
                        + dd.get(2, 8) * k[8][m]
                        + dd.get(2, 9) * k[9][m]
                        + dd.get(2, 10) * k[10][m]
                        + dd.get(2, 11) * k[11][m]
                        + dd.get(2, 12) * k[11][m]
                        + dd.get(2, 13) * kd[0][m]
                        + dd.get(2, 14) * kd[1][m]
                        + dd.get(2, 15) * kd[2][m]);
                dout[7][m] = h
                    * (dd.get(3, 0) * k[0][m]
                        + dd.get(3, 5) * k[5][m]
                        + dd.get(3, 6) * k[6][m]
                        + dd.get(3, 7) * k[7][m]
                        + dd.get(3, 8) * k[8][m]
                        + dd.get(3, 9) * k[9][m]
                        + dd.get(3, 10) * k[10][m]
                        + dd.get(3, 11) * k[11][m]
                        + dd.get(3, 12) * k[11][m]
                        + dd.get(3, 13) * kd[0][m]
                        + dd.get(3, 14) * kd[1][m]
                        + dd.get(3, 15) * kd[2][m]);
            }
        }

        // auxiliary
        let n_stage = self.work.nstg;
        let dmin = 1.0 / self.conf.Mmin;
        let dmax = 1.0 / self.conf.Mmax;

        // update y
        vec_copy(y0, &self.w).unwrap();

        // update k0
        if self.info.first_step_same_as_last {
            for m in 0..self.ndim {
                self.work.f[0][m] = self.work.f[n_stage - 1][m]; // k0 := ks for next step
            }
        }

        // return zero if not embedded
        if !self.info.embedded {
            return 0.0;
        }

        // estimate new stepsize
        let mut d = f64::powf(self.work.rerr, self.n);
        if self.conf.StabBeta > 0.0 {
            // lund-stabilization
            d = d / f64::powf(self.work.rerr_prev, self.conf.StabBeta);
        }
        d = f64::max(dmax, f64::min(dmin, d / self.conf.Mfac)); // we require  fac1 <= hnew/h <= fac2
        let dxnew = self.work.h / d;
        dxnew
    }

    /// Rejects the update
    ///
    /// Returns the `relative_error`
    pub fn reject_update(&mut self) -> f64 {
        // estimate new stepsize
        let dmin = 1.0 / self.conf.Mmin;
        let d = f64::powf(self.work.rerr, self.n) / self.conf.Mfac;
        let dxnew = self.work.h / f64::min(dmin, d);
        dxnew
    }

    /// Computes the dense output
    pub fn dense_output(&self, yout: &mut Vector, h: f64, x: f64, y: &Vector, xout: f64) {
        if self.conf.denseOut && self.conf.method == Method::DoPri5 {
            let dout = self.dout.as_ref().unwrap();
            let xold = x - h;
            let theta = (xout - xold) / h;
            let u_theta = 1.0 - theta;
            for m in 0..self.ndim {
                yout[m] = dout[0][m]
                    + theta * (dout[1][m] + u_theta * (dout[2][m] + theta * (dout[3][m] + u_theta * dout[4][m])));
            }
        }
        if self.conf.denseOut && self.conf.method == Method::DoPri8 {
            let dout = self.dout.as_ref().unwrap();
            let xold = x - h;
            let theta = (xout - xold) / h;
            let u_theta = 1.0 - theta;
            for m in 0..self.ndim {
                let par = dout[4][m] + theta * (dout[5][m] + u_theta * (dout[6][m] + theta * dout[7][m]));
                yout[m] =
                    dout[0][m] + theta * (dout[1][m] + u_theta * (dout[2][m] + theta * (dout[3][m] + u_theta * par)));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_lab::approx_eq;

    #[test]
    fn constants_are_consistent() {
        let ndim = 1;
        let function = |_: &mut Vector, _: f64, _: f64, _: &Vector, _: &mut i32| -> Result<(), StrError> { Ok(()) };
        let methods = Method::explicit_methods();
        let staged = methods.iter().filter(|&&m| m != Method::FwEuler);
        for method in staged {
            println!("\n... {:?} ...", method);
            let conf = Configuration::new(*method, None, None);
            let erk = ExplicitRungeKutta::new(conf, ndim, function).unwrap();
            let nstage = erk.work.nstg;
            assert_eq!(erk.A.dims(), (nstage, nstage));
            assert_eq!(erk.B.dim(), nstage);
            assert_eq!(erk.C.dim(), nstage);
            let info = method.information();
            if info.embedded {
                let ee = erk.E.as_ref().unwrap();
                assert_eq!(ee.dim(), nstage);
            }
            println!("c coefficients: ci = Σ_j aij");
            for i in 0..nstage {
                let mut sum = 0.0;
                for j in 0..nstage {
                    sum += erk.A.get(i, j);
                }
                approx_eq(sum, erk.C[i], 1e-14);
            }
            if info.first_step_same_as_last && info.embedded {
                let ee = erk.E.as_ref().unwrap();
                // let ee = erk.E.as_ref().unwrap();
            }
        }
    }
}
