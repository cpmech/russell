#![allow(unused, non_snake_case)]

use crate::{constants::*, JacF};
use crate::{Configuration, Func, Information, Method, RungeKuttaTrait, Statistics, StrError, Workspace};
use russell_lab::{Matrix, Vector};

pub struct ExplicitRungeKutta {
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
    ndim: usize,         // problem dimension
    conf: Configuration, // configuration
    work: Workspace,     // workspace
    stat: Statistics,    // statistics
    fcn: Func,           // dy/dx = f(x, y) function

    // auxiliary
    w: Vector, // local workspace
    n: f64,    // exponent n = 1/(q+1) (or 1/(q+1)-0.75⋅β) of rerrⁿ

               // 5(3) error estimator
               // err53: bool, // use 5-3 error estimator
               // bhh1: f64,   // error estimator: coefficient of k0
               // bhh2: f64,   // error estimator: coefficient of k8
               // bhh3: f64,   // error estimator: coefficient of k11

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
        if !info.implicit && info.multiple_stages {
            #[rustfmt::skip]
            let (A, B, C) = match conf.method {
                Method::Radau5     => panic!("<not available>"),
                Method::BwEuler    => panic!("<not available>"),
                Method::FwEuler    => panic!("<not available>"),
                Method::MdEuler    => (Matrix::from(&MODIFIED_EULER_A)    , Vector::from(&MODIFIED_EULER_B)    , Vector::from(&MODIFIED_EULER_C)   ),
                Method::Rk2        => (Matrix::from(&RUNGE_KUTTA_2_A)     , Vector::from(&RUNGE_KUTTA_2_B)     , Vector::from(&RUNGE_KUTTA_2_C)    ),
                Method::Rk3        => (Matrix::from(&RUNGE_KUTTA_3_A)     , Vector::from(&RUNGE_KUTTA_3_B)     , Vector::from(&RUNGE_KUTTA_3_C)    ),
                Method::Heun3      => (Matrix::from(&HEUN_3_A)            , Vector::from(&HEUN_3_B)            , Vector::from(&HEUN_3_C)           ),
                Method::Rk4        => (Matrix::from(&RUNGE_KUTTA_4_A)     , Vector::from(&RUNGE_KUTTA_4_B)     , Vector::from(&RUNGE_KUTTA_4_C)    ),
                Method::Rk4alt     => (Matrix::from(&RUNGE_KUTTA_ALT_4_A) , Vector::from(&RUNGE_KUTTA_ALT_4_B) , Vector::from(&RUNGE_KUTTA_ALT_4_C)),
                Method::Merson4    => (Matrix::from(&MERSON_4_A)          , Vector::from(&MERSON_4_B)          , Vector::from(&MERSON_4_C)         ),
                Method::Zonneveld4 => (Matrix::from(&ZONNEVELD_4_A)       , Vector::from(&ZONNEVELD_4_B)       , Vector::from(&ZONNEVELD_4_C)      ),
                Method::Fehlberg4  => (Matrix::from(&FEHLBERG_4_A)        , Vector::from(&FEHLBERG_4_B)        , Vector::from(&FEHLBERG_4_C)       ),
                Method::DoPri5     => (Matrix::from(&DORMAND_PRINCE_5_A)  , Vector::from(&DORMAND_PRINCE_5_B)  , Vector::from(&DORMAND_PRINCE_5_C) ),
                Method::Verner6    => (Matrix::from(&VERNER_6_A)          , Vector::from(&VERNER_6_B)          , Vector::from(&VERNER_6_C)         ),
                Method::Fehlberg7  => (Matrix::from(&FEHLBERG_7_A)        , Vector::from(&FEHLBERG_7_B)        , Vector::from(&FEHLBERG_7_C)       ),
                Method::DoPri8     => (Matrix::from(&DORMAND_PRINCE_8_A)  , Vector::from(&DORMAND_PRINCE_8_B)  , Vector::from(&DORMAND_PRINCE_8_C) ),
            };
            let (mut Be, mut E) = (None, None);
            let (mut Ad, mut Cd, mut D) = (None, None, None);
            let Nstg = B.dim();
            let lund_stabilization_factor = if conf.StabBeta > 0.0 {
                1.0 / ((info.order_of_estimator + 1) as f64) - conf.StabBeta * conf.stabBetaM
            } else {
                1.0 / ((info.order_of_estimator + 1) as f64)
            };
            let stat = Statistics::new(info.implicit, conf.genie);
            Ok(ExplicitRungeKutta {
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
                conf,
                work: Workspace::new(Nstg, ndim),
                stat,
                fcn: function,
                w: Vector::new(ndim),
                n: lund_stabilization_factor,
                // err53: (),
                // bhh1: (),
                // bhh2: (),
                // bhh3: (),
                // dout: (),
                // kd: (),
                // yd: (),
                // dfunA: (),
                // dfunB: (),
            })
        } else {
            Err("The Runge-Kutta method must be explicit and multi-stage")
        }
    }

    /// Performs  the next step
    pub fn next_step(&mut self, xa: f64, ya: &Vector) {
        // auxiliary
        let h = self.work.h;
        let k = &mut self.work.f;
        let v = &self.work.v;

        let dmin = 1.0 / self.conf.Mmin;
        let dmax = 1.0 / self.conf.Mmax;
        let ndf = self.ndim as f64;

        // compute k0 (otherwise, use k0 saved in Accept)
        if (self.work.first || !self.info.first_step_same_as_last) && !self.work.reject {
            let u0 = xa + h * self.C[0];
            self.stat.Nfeval += 1;
            (self.fcn)(&mut k[0], h, u0, ya); // k0 := f(ui,vi)
        }

        /*
        // compute ki
        for i in 1..self.work.nstg {
           ui = xa + h*self.C[i];
            v[i].Apply(1, ya);        // vi := ya
            for j in 0..i {           // lower diagonal ⇒ explicit
                la.VecAdd(v[i], 1, v[i], h*self.A[i][j], k[j]); // vi += h⋅aij⋅kj
            }
           self.stat.Nfeval+=1;
            self.fcn(k[i], h, ui, v[i]); // ki := f(ui,vi)
        }

        // update
        if !self.Embedded {
            for m in 0..self.ndim {
                self.w[m] = ya[m];
                for i in 0..self.nstg {
                    self.w[m] += self.B[i] * k[i][m] * h;
                }
            }
            return;
        }

        // error estimation with 5 and 3 orders (e.g. DoPri853)
        if self.err53 {
            for m in 0..self.ndim {
                self.w[m] = ya[m];
                let mut errA = 0.0;
                let mut errB = 0.0;
                for i in 0..self.nstg {
                    self.w[m] += self.B[i] * k[i][m] * h;
                    errA += self.B[i] * k[i][m];
                    errB += self.E[i] * k[i][m];
                }
                let sk = self.conf.atol + self.conf.rtol*utl.Max(math.Abs(ya[m]), math.Abs(self.w[m]));
                errA -= (self.bhh1*k[0][m] + self.bhh2*k[8][m] + self.bhh3*k[11][m]);
                err3 += (errA / sk) * (errA / sk);
                err5 += (errB / sk) * (errB / sk);
                // stiffness estimation
                dk = k[self.Nstg-1][m] - k[self.Nstg-2][m];
                dv = v[self.Nstg-1][m] - v[self.Nstg-2][m];
                snum += dk * dk;
                sden += dv * dv;
            }
            let den = err5 + 0.01*err3; // similar to Eq. (10.17) of [1, page 255]
            if den <= 0.0 {
                den = 1.0;
            }
            self.work.rerr = math.Abs(h) * err5 * math.Sqrt(1.0/(self.ndf*den));
            if sden > 0 {
                self.work.rs = h * math.Sqrt(snum/sden);
            }
            return ;
        }

        // update, error and stiffness estimation
        for m in 0..self.ndim {
            self.w[m] = ya[m];
            let lerrm = 0.0; // must be zeroed for each m
            for i in 0..nstg {
                kh = k[i][m] * h;
                self.w[m] += self.B[i] * kh;
                lerrm += self.E[i] * kh;
            }
            let sk = self.conf.atol + self.conf.rtol*utl.Max(math.Abs(ya[m]), math.Abs(self.w[m]));
            let ratio = lerrm / sk;
            sum += ratio * ratio;
            // stiffness estimation
            dk = k[self.Nstg-1][m] - k[self.Nstg-2][m];
            dv = v[self.Nstg-1][m] - v[self.Nstg-2][m];
            snum += dk * dk;
            sden += dv * dv;
        }
        self.work.rerr = utl.Max(math.Sqrt(sum/self.ndf), 1.0e-10);
        if sden > 0 {
            self.work.rs = h * math.Sqrt(snum/sden);
        }
        */
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
