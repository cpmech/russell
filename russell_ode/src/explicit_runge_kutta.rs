#![allow(unused, non_snake_case)]

use crate::{constants::*, JacF};
use crate::{Configuration, Func, Information, Method, RungeKuttaTrait, Statistics, StrError, Workspace};
use russell_lab::{vec_add, vec_copy, vec_update, Matrix, Vector};

pub struct ExplicitRungeKutta {
    conf: Configuration, // configuration
    info: Information,

    // constants
    A: Matrix, // A coefficients
    B: Vector, // B coefficients
    C: Vector, // C coefficients

    Be: Option<Vector>, // B coefficients [may be nil, e.g. if FSAL = false]

    E: Option<Vector>, // error coefficients. difference between B and Be: e = b - be (if be is not nil)

    Ad: Option<Matrix>, // A coefficients for dense output
    Cd: Option<Vector>, // C coefficients for dense output
    D: Option<Matrix>,  // dense output coefficients. [may be nil]

    Nstg: usize, // number of stages = len(A) = len(B) = len(C)
    P: usize,    // order of y1 (corresponding to b)
    Q: usize,    // order of error estimator (embedded only); e.g. DoPri5(4) ⇒ q = 4 (=min(order(y1) , order(y1bar))

    // data
    ndim: usize,      // problem dimension
    work: Workspace,  // workspace
    stat: Statistics, // statistics
    fcn: Func,        // dy/dx = f(x, y) function

    // auxiliary
    w: Vector, // local workspace
    n: f64,    // exponent n = 1/(q+1) (or 1/(q+1)-0.75⋅β) of rerrⁿ

               // dense output
               // dout: Vec<Vector>, // dense output coefficients [nstgDense][ndim] (partially allocated by newERK method)
               // kd: Vec<Vector>,   // k values for dense output [nextraKs] (partially allocated by newERK method)
               // yd: Vector,        // y values for dense output (allocated here if len(kd)>0)

               // functions to compute variables for dense output
               // dfunA: fn(&mut Vector, f64),                    // in Accept function
               // dfunB: fn(&mut Vector, f64, f64, &Vector, f64), // DenseOut function
}

impl ExplicitRungeKutta {
    /// Allocates a new instance
    pub fn new(conf: Configuration, ndim: usize, function: Func) -> Result<Self, StrError> {
        let info = conf.method.information();
        if info.implicit {
            return Err("the method must be explicit (must not be Radau5 nor BwEuler)");
        }
        if info.multiple_stages {
            return Err("the method must have multiple-stages (must not be FwEuler");
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
        if conf.method == Method::DoPri5 {
            D = Some(Matrix::from(&DORMAND_PRINCE_5_D));
        }
        if conf.method == Method::DoPri8 {
            Ad = Some(Matrix::from(&DORMAND_PRINCE_8_AD));
            Cd = Some(Vector::from(&DORMAND_PRINCE_8_CD));
            D = Some(Matrix::from(&DORMAND_PRINCE_8_D));
        }
        let Nstg = B.dim();
        let lund_stabilization_factor = if conf.StabBeta > 0.0 {
            1.0 / ((info.order_of_estimator + 1) as f64) - conf.StabBeta * conf.stabBetaM
        } else {
            1.0 / ((info.order_of_estimator + 1) as f64)
        };
        let stat = Statistics::new(info.implicit, conf.genie);
        Ok(ExplicitRungeKutta {
            conf,
            info,
            A,
            B,
            C,
            Be,
            E,
            Ad,
            Cd,
            D,
            Nstg,
            P: info.order,
            Q: info.order_of_estimator,
            ndim,
            work: Workspace::new(Nstg, ndim),
            stat,
            fcn: function,
            w: Vector::new(ndim),
            n: lund_stabilization_factor,
            // dout: (),
            // kd: (),
            // yd: (),
            // dfunA: (),
            // dfunB: (),
        })
    }

    /// Performs  the next step
    pub fn next_step(&mut self, xa: f64, ya: &Vector) {
        // auxiliary
        let h = self.work.h;
        let k = &mut self.work.f;
        let v = &mut self.work.v;

        let dmin = 1.0 / self.conf.Mmin;
        let dmax = 1.0 / self.conf.Mmax;

        // compute k0 (otherwise, use k0 saved in Accept)
        if (self.work.first || !self.info.first_step_same_as_last) && !self.work.reject {
            let u0 = xa + h * self.C[0];
            self.stat.Nfeval += 1;
            (self.fcn)(&mut k[0], h, u0, ya); // k0 := f(ui,vi)
        }

        // compute ki
        for i in 1..self.work.nstg {
            let ui = xa + h * self.C[i];
            vec_copy(&mut v[1], &ya).unwrap(); // vi := ya
            for j in 0..i {
                // j goes from 0 to (i-1) because the method is explicit (lower diagonal)
                vec_update(&mut v[i], h * self.A.get(i, j), &k[j]).unwrap();
                // vec_add(&mut v[i], 1.0, &v[i], h * self.A.get(i, j), &k[j]).unwrap();
                // vi += h ⋅ aij ⋅ kj
            }
            self.stat.Nfeval += 1;
            (self.fcn)(&mut k[i], h, ui, &v[i]); // ki := f(ui,vi)
        }

        // update
        if !self.info.embedded {
            for m in 0..self.ndim {
                self.w[m] = ya[m];
                for i in 0..self.work.nstg {
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
            const BHH1: f64 = 0.244094488188976377952755905512e+00;
            const BHH2: f64 = 0.733846688281611857341361741547e+00;
            const BHH3: f64 = 0.220588235294117647058823529412e-01;
            let mut errA = 0.0;
            let mut errB = 0.0;
            let mut err3 = 0.0;
            let mut err5 = 0.0;
            for m in 0..self.ndim {
                self.w[m] = ya[m];
                let mut errA = 0.0;
                let mut errB = 0.0;
                for i in 0..self.work.nstg {
                    self.w[m] += self.B[i] * k[i][m] * h;
                    errA += self.B[i] * k[i][m];
                    errB += ee[i] * k[i][m];
                }
                let sk = self.conf.atol + self.conf.rtol * f64::max(f64::abs(ya[m]), f64::abs(self.w[m]));
                errA -= (BHH1 * k[0][m] + BHH2 * k[8][m] + BHH3 * k[11][m]);
                err3 += (errA / sk) * (errA / sk);
                err5 += (errB / sk) * (errB / sk);
                // stiffness estimation
                let dk = k[self.Nstg - 1][m] - k[self.Nstg - 2][m];
                let dv = v[self.Nstg - 1][m] - v[self.Nstg - 2][m];
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
            for i in 0..self.work.nstg {
                let kh = k[i][m] * h;
                self.w[m] += self.B[i] * kh;
                lerrm += ee[i] * kh;
            }
            let sk = self.conf.atol + self.conf.rtol * f64::max(f64::abs(ya[m]), f64::abs(self.w[m]));
            let ratio = lerrm / sk;
            sum += ratio * ratio;
            // stiffness estimation
            let dk = k[self.Nstg - 1][m] - k[self.Nstg - 2][m];
            let dv = v[self.Nstg - 1][m] - v[self.Nstg - 2][m];
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
    /// Returns `(stepsize_new, relative_error)`
    pub fn accept_update(&mut self) -> (f64, f64) {
        (0.0, 0.0)
    }

    /// Rejects the update
    ///
    /// Returns the `relative_error`
    pub fn reject_update(&mut self) -> f64 {
        0.0
    }

    /// Computes the dense output
    pub fn dense_output(&self) {}
}
