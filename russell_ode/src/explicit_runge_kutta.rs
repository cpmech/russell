use crate::constants::*;
use crate::StrError;
use crate::{detect_stiffness, ErkDenseOut, Information, Method, OdeSolverTrait, Params, System, Workspace};
use russell_lab::{format_fortran, vec_copy, vec_update, Matrix, Vector};
use russell_sparse::CooMatrix;

pub(crate) struct ExplicitRungeKutta<'a, F, J, A>
where
    F: Send + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Holds the parameters
    params: Params,

    /// ODE system
    system: System<'a, F, J, A>,

    /// Information such as implicit, embedded, etc.
    info: Information,

    /// Runge-Kutta A coefficients
    aa: Matrix,

    /// Runge-Kutta B coefficients
    bb: Vector,

    /// Runge-Kutta C coefficients
    cc: Vector,

    /// (embedded) error coefficients
    ///
    /// difference between B and Be: e = b - be
    ee: Option<Vector>,

    /// Number of stages
    nstage: usize,

    /// Lund stabilization factor (n)
    ///
    /// `n = 1/(q+1)-0.75⋅β` of `rel_err ⁿ`
    lund_factor: f64,

    /// Auxiliary variable: 1 / m_min
    d_min: f64,

    /// Auxiliary variable: 1 / m_max
    d_max: f64,

    /// Array of vectors holding the updates
    ///
    /// v[stg][dim] = ya[dim] + h*sum(a[stg][j]*f[j][dim], j, nstage)
    v: Vec<Vector>,

    /// Array of vectors holding the function evaluations
    ///
    /// k[stg][dim] = f(u[stg], v[stg][dim])
    k: Vec<Vector>,

    /// Auxiliary workspace (will contain y0 to be used in accept_update)
    w: Vector,

    /// Handles the dense output
    dense_out: Option<ErkDenseOut>,
}

impl<'a, F, J, A> ExplicitRungeKutta<'a, F, J, A>
where
    F: Send + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Allocates a new instance
    pub fn new(params: Params, system: System<'a, F, J, A>) -> Result<Self, StrError> {
        // Runge-Kutta coefficients
        #[rustfmt::skip]
        let (aa, bb, cc) = match params.method {
            Method::Radau5     => return Err("cannot use Radau5 with ExplicitRungeKutta"),
            Method::BwEuler    => return Err("cannot use BwEuler with ExplicitRungeKutta"),
            Method::FwEuler    => return Err("cannot use FwEuler with ExplicitRungeKutta"),
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

        // information
        let info = params.method.information();
        assert!(!info.implicit);

        // coefficients for error estimate
        let ee = if info.embedded {
            match params.method {
                Method::Radau5 => None,
                Method::BwEuler => None,
                Method::FwEuler => None,
                Method::Rk2 => None,
                Method::Rk3 => None,
                Method::Heun3 => None,
                Method::Rk4 => None,
                Method::Rk4alt => None,
                Method::MdEuler => Some(Vector::from(&MODIFIED_EULER_E)),
                Method::Merson4 => Some(Vector::from(&MERSON_4_E)),
                Method::Zonneveld4 => Some(Vector::from(&ZONNEVELD_4_E)),
                Method::Fehlberg4 => Some(Vector::from(&FEHLBERG_4_E)),
                Method::DoPri5 => Some(Vector::from(&DORMAND_PRINCE_5_E)),
                Method::Verner6 => Some(Vector::from(&VERNER_6_E)),
                Method::Fehlberg7 => Some(Vector::from(&FEHLBERG_7_E)),
                Method::DoPri8 => Some(Vector::from(&DORMAND_PRINCE_8_E)),
            }
        } else {
            None
        };

        // number of stages
        let nstage = bb.dim();

        // Lund stabilization factor (n)
        let lund_factor = 1.0 / ((info.order_of_estimator + 1) as f64) - params.erk.lund_beta * params.erk.lund_m;

        // return structure
        let ndim = system.ndim;
        Ok(ExplicitRungeKutta {
            params,
            system,
            info,
            aa,
            bb,
            cc,
            ee,
            nstage,
            lund_factor,
            d_min: 1.0 / params.step.m_min,
            d_max: 1.0 / params.step.m_max,
            v: vec![Vector::new(ndim); nstage],
            k: vec![Vector::new(ndim); nstage],
            w: Vector::new(ndim),
            dense_out: None,
        })
    }
}

impl<'a, F, J, A> OdeSolverTrait<A> for ExplicitRungeKutta<'a, F, J, A>
where
    F: Send + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Enables dense output
    fn enable_dense_output(&mut self) -> Result<(), StrError> {
        self.dense_out = Some(ErkDenseOut::new(self.params.method, self.system.ndim)?);
        Ok(())
    }

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        // auxiliary
        let k = &mut self.k;
        let v = &mut self.v;

        // compute k0 (otherwise, use k0 saved in accept)
        if (work.bench.n_accepted == 0 || !self.info.first_step_same_as_last) && !work.follows_reject_step {
            let u0 = x + h * self.cc[0];
            work.bench.n_function += 1;
            (self.system.function)(&mut k[0], u0, y, args)?; // k0 := f(ui,vi)
        }

        // compute ki
        for i in 1..self.nstage {
            let ui = x + h * self.cc[i];
            vec_copy(&mut v[i], &y).unwrap(); // vi := ya
            for j in 0..i {
                vec_update(&mut v[i], h * self.aa.get(i, j), &k[j]).unwrap(); // vi += h ⋅ aij ⋅ kj
            }
            work.bench.n_function += 1;
            (self.system.function)(&mut k[i], ui, &v[i], args)?; // ki := f(ui,vi)
        }

        // update (methods without error estimation)
        if !self.info.embedded {
            for m in 0..self.system.ndim {
                self.w[m] = y[m];
                for i in 0..self.nstage {
                    self.w[m] += self.bb[i] * k[i][m] * h;
                }
            }
            return Ok(());
        }

        // auxiliary
        let ee = self.ee.as_ref().unwrap();
        let dim = self.system.ndim as f64;

        // update and error estimation
        if self.params.method == Method::DoPri8 {
            //  Dormand-Prince 8 with 5 and 3 orders
            let (bhh1, bhh2, bhh3) = (DORMAND_PRINCE_8_BHH1, DORMAND_PRINCE_8_BHH2, DORMAND_PRINCE_8_BHH3);
            let mut err_3 = 0.0;
            let mut err_5 = 0.0;
            for m in 0..self.system.ndim {
                self.w[m] = y[m];
                let mut err_a = 0.0;
                let mut err_b = 0.0;
                for i in 0..self.nstage {
                    self.w[m] += self.bb[i] * k[i][m] * h;
                    err_a += self.bb[i] * k[i][m];
                    err_b += ee[i] * k[i][m];
                }
                let sk = self.params.tol.abs + self.params.tol.rel * f64::max(f64::abs(y[m]), f64::abs(self.w[m]));
                err_a -= bhh1 * k[0][m] + bhh2 * k[8][m] + bhh3 * k[11][m];
                err_3 += (err_a / sk) * (err_a / sk);
                err_5 += (err_b / sk) * (err_b / sk);
            }
            let mut den = err_5 + 0.01 * err_3; // similar to Eq. (10.17) of [1, page 255]
            if den <= 0.0 {
                den = 1.0;
            }
            work.rel_error = f64::abs(h) * err_5 * f64::sqrt(1.0 / (dim * den));
        } else {
            // all other ERK methods
            let mut sum = 0.0;
            for m in 0..self.system.ndim {
                self.w[m] = y[m];
                let mut err_m = 0.0;
                for i in 0..self.nstage {
                    let kh = k[i][m] * h;
                    self.w[m] += self.bb[i] * kh;
                    err_m += ee[i] * kh;
                }
                let sk = self.params.tol.abs + self.params.tol.rel * f64::max(f64::abs(y[m]), f64::abs(self.w[m]));
                let ratio = err_m / sk;
                sum += ratio * ratio;
            }
            work.rel_error = f64::max(f64::sqrt(sum / dim), 1.0e-10);
        }
        Ok(())
    }

    /// Updates x and y and computes the next stepsize
    fn accept(
        &mut self,
        work: &mut Workspace,
        x: &mut f64,
        y: &mut Vector,
        h: f64,
        args: &mut A,
    ) -> Result<(), StrError> {
        // save data for dense output
        if let Some(out) = self.dense_out.as_mut() {
            work.bench.n_function += out.update(&mut self.system, *x, y, h, &self.w, &self.k, args)?;
        }

        // update x and y
        *x += h;
        vec_copy(y, &self.w).unwrap();

        // update k0
        if self.info.first_step_same_as_last {
            for m in 0..self.system.ndim {
                self.k[0][m] = self.k[self.nstage - 1][m]; // k0 := ks for next step
            }
        }

        // exit if not embedded method
        if !self.info.embedded {
            return Ok(());
        }

        // estimate the new stepsize
        let mut fac = f64::powf(work.rel_error, self.lund_factor); // line 463 of dopri5.f
        if self.params.erk.lund_beta > 0.0 && work.rel_error_prev > 0.0 {
            // lund-stabilization (line 465 of dopri5.f)
            fac = fac / f64::powf(work.rel_error_prev, self.params.erk.lund_beta);
        }
        fac = f64::max(self.d_max, f64::min(self.d_min, fac / self.params.step.m_safety)); // line 467 of dopri5.f
        work.h_new = h / fac;

        // stiffness detection
        if self.params.stiffness.enabled {
            if self.params.method == Method::DoPri5 {
                let mut num = 0.0;
                let mut den = 0.0;
                for m in 0..self.system.ndim {
                    let delta_k = self.k[6][m] - self.k[5][m]; // k7 - k6  (Eq 2.26, HW-PartII, page 22)
                    let delta_v = self.v[6][m] - self.v[5][m]; // v7 - v6  (Eq 2.26, HW-PartII, page 22)
                    num += delta_k * delta_k;
                    den += delta_v * delta_v;
                }
                if den > f64::EPSILON {
                    work.stiff_h_times_lambda = h * f64::sqrt(num / den);
                }
                detect_stiffness(work, &self.params)?;
            } else if self.params.method == Method::DoPri8 {
                const NEW: usize = 10; // to use k[NEW] as a temporary workspace
                work.bench.n_function += 1;
                (self.system.function)(&mut self.k[NEW], *x, y, args)?; // line 663 of dop853.f
                let mut num = 0.0;
                let mut den = 0.0;
                for m in 0..self.system.ndim {
                    let delta_k = self.k[NEW][m] - self.k[11][m]; // line 670 of dop843.f
                    let delta_v = y[m] - self.v[11][m];
                    num += delta_k * delta_k;
                    den += delta_v * delta_v;
                }
                if den > f64::EPSILON {
                    work.stiff_h_times_lambda = h * f64::sqrt(num / den);
                }
                detect_stiffness(work, &self.params)?;
            }
        };

        // print debug messages
        if self.params.debug {
            if work.stiff_detected {
                println!(
                    "THE PROBLEM SEEMS TO BECOME STIFF AT X ={}, ACCEPTED STEP ={:>5}",
                    format_fortran(*x - h),
                    work.bench.n_accepted
                );
            }
            println!(
                "step(A) ={:>5}, err ={}, h_new ={}, n_yes ={:>4}, n_no ={:>4}, h*lambda ={}",
                work.bench.n_steps,
                format_fortran(work.rel_error),
                format_fortran(work.h_new),
                work.stiff_n_detection_yes,
                work.stiff_n_detection_no,
                format_fortran(work.stiff_h_times_lambda),
            );
        }
        Ok(())
    }

    /// Rejects the update
    fn reject(&mut self, work: &mut Workspace, h: f64) {
        // estimate new stepsize
        let d = f64::powf(work.rel_error, self.lund_factor) / self.params.step.m_safety;
        work.h_new = h / f64::min(self.d_min, d);

        // print debug messages
        if self.params.debug {
            println!(
                "step(R) ={:>5}, err ={}, h_new ={}",
                work.bench.n_steps,
                format_fortran(work.rel_error),
                format_fortran(work.h_new),
            );
        }
    }

    /// Computes the dense output with x-h ≤ x_out ≤ x
    fn dense_output(&self, y_out: &mut Vector, x_out: f64, x: f64, _y: &Vector, h: f64) {
        if let Some(out) = self.dense_out.as_ref() {
            out.calculate(y_out, x_out, x, h);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::ExplicitRungeKutta;
    use crate::{no_jacobian, HasJacobian, Method, OdeSolverTrait, Params, Samples, System, Workspace};
    use russell_lab::{approx_eq, vec_approx_eq, Vector};

    #[test]
    fn constants_are_consistent() {
        let methods = Method::explicit_methods();
        let staged = methods.iter().filter(|&&m| m != Method::FwEuler);
        struct Args {}
        for method in staged {
            println!("\n... {:?} ...", method);
            let params = Params::new(*method);
            let system = System::new(
                1,
                |_, _, _, _args: &mut Args| Ok(()),
                no_jacobian,
                HasJacobian::No,
                None,
                None,
            );
            let erk = ExplicitRungeKutta::new(params, system).unwrap();
            let nstage = erk.nstage;
            assert_eq!(erk.aa.dims(), (nstage, nstage));
            assert_eq!(erk.bb.dim(), nstage);
            assert_eq!(erk.cc.dim(), nstage);
            let info = method.information();
            if info.embedded {
                let ee = erk.ee.as_ref().unwrap();
                assert_eq!(ee.dim(), nstage);
            }

            println!("Σi bi = 1                                 (Eq. 1.11a, page 135)");
            let mut sum = 0.0;
            for i in 0..nstage {
                sum += erk.bb[i];
            }
            approx_eq(sum, 1.0, 1e-15);

            println!("Σi bi ci = 1/2                            (Eq. 1.11b, page 135)");
            sum = 0.0;
            for i in 0..nstage {
                sum += erk.bb[i] * erk.cc[i];
            }
            approx_eq(sum, 1.0 / 2.0, 1e-15);

            if erk.info.order < 4 {
                continue;
            }

            println!("Σi bi ci² = 1/3                           (Eq. 1.11c, page 135)");
            sum = 0.0;
            for i in 0..nstage {
                sum += erk.bb[i] * erk.cc[i] * erk.cc[i];
            }
            approx_eq(sum, 1.0 / 3.0, 1e-15);

            println!("Σi,j bi aij cj = 1/6                      (Eq. 1.11d, page 135)");
            sum = 0.0;
            for i in 0..nstage {
                for j in 0..nstage {
                    sum += erk.bb[i] * erk.aa.get(i, j) * erk.cc[j];
                }
            }
            approx_eq(sum, 1.0 / 6.0, 1e-15);

            println!("Σi bi ci³ = 1/4                           (Eq. 1.11e, page 135)");
            sum = 0.0;
            for i in 0..nstage {
                sum += erk.bb[i] * erk.cc[i] * erk.cc[i] * erk.cc[i];
            }
            approx_eq(sum, 1.0 / 4.0, 1e-15);

            println!("Σi,j bi ci aij cj = 1/8                   (Eq. 1.11f, page 135)");
            sum = 0.0;
            for i in 0..nstage {
                for j in 0..nstage {
                    sum += erk.bb[i] * erk.cc[i] * erk.aa.get(i, j) * erk.cc[j];
                }
            }
            approx_eq(sum, 1.0 / 8.0, 1e-15);

            println!("Σi,j bi aij cj² = 1/12                    (Eq. 1.11g, page 136)");
            sum = 0.0;
            for i in 0..nstage {
                for j in 0..nstage {
                    sum += erk.bb[i] * erk.aa.get(i, j) * erk.cc[j] * erk.cc[j];
                }
            }
            approx_eq(sum, 1.0 / 12.0, 1e-15);

            println!("Σi,j,k bi aij ajk ck = 1/24               (Eq. 1.11h, page 136)");
            sum = 0.0;
            for i in 0..nstage {
                for j in 0..nstage {
                    for k in 0..nstage {
                        sum += erk.bb[i] * erk.aa.get(i, j) * erk.aa.get(j, k) * erk.cc[k];
                    }
                }
            }
            approx_eq(sum, 1.0 / 24.0, 1e-15);

            println!("Σ(i=1,s) bi ciⁿ⁻¹ = 1 / n   for n = 1..8  (Eq. 5.20a, page 181)");
            for n in 1..erk.info.order {
                let mut sum = 0.0;
                for i in 1..(nstage + 1) {
                    sum += erk.bb[i - 1] * f64::powf(erk.cc[i - 1], (n - 1) as f64);
                }
                approx_eq(sum, 1.0 / (n as f64), 1e-15);
            }

            println!("Σ(j=1,i-1) aij = ci         for i = 1..s  (Eq. 5.20b, page 181)");
            for i in 1..(nstage + 1) {
                let mut sum = 0.0;
                for j in 1..i {
                    sum += erk.aa.get(i - 1, j - 1);
                }
                approx_eq(sum, erk.cc[i - 1], 1e-14);
            }

            if erk.info.order < 5 {
                continue;
            }

            println!("Σ(j=1,i-1) aij cj = ci² / 2 for i = 3..s  (Eq. 5.20c, page 181)");
            for i in 3..(nstage + 1) {
                let mut sum = 0.0;
                for j in 1..i {
                    sum += erk.aa.get(i - 1, j - 1) * erk.cc[j - 1];
                }
                approx_eq(sum, erk.cc[i - 1] * erk.cc[i - 1] / 2.0, 1e-14);
            }
        }
    }

    #[test]
    fn modified_euler_works() {
        // This test relates to Table 21.2 of Kreyszig's book, page 904

        // problem
        let (system, data, mut args) = Samples::kreyszig_eq6_page902();
        let mut yfx = data.y_analytical.unwrap();
        let ndim = system.ndim;

        // allocate structs
        let params = Params::new(Method::MdEuler); // aka the Improved Euler in Kreyszig's book
        let mut solver = ExplicitRungeKutta::new(params, system).unwrap();
        let mut work = Workspace::new(Method::FwEuler);

        // check dense output availability
        assert_eq!(
            solver.enable_dense_output().err(),
            Some("dense output is not available for the MdEuler method")
        );

        // numerical approximation
        let h = 0.2;
        let mut x = data.x0;
        let mut y = data.y0.clone();
        let mut y_ana = Vector::new(ndim);
        yfx(&mut y_ana, x);
        let mut xx = vec![x];
        let mut yy_num = vec![y[0]];
        let mut yy_ana = vec![y_ana[0]];
        let mut errors = vec![f64::abs(yy_num[0] - yy_ana[0])];
        for n in 0..5 {
            solver.step(&mut work, x, &y, h, &mut args).unwrap();
            assert_eq!(work.bench.n_function, (n + 1) * 2);

            solver.accept(&mut work, &mut x, &mut y, h, &mut args).unwrap();
            xx.push(x);
            yy_num.push(y[0]);

            yfx(&mut y_ana, x);
            yy_ana.push(y_ana[0]);
            errors.push(f64::abs(yy_num.last().unwrap() - yy_ana.last().unwrap()));
        }

        // Mathematica code:
        //
        // MdEulerSingleEq[f_, x0_, y0_, x1_, h_] := Module[{x, y, nstep, k1, k2},
        //    x[1] = x0;
        //    y[1] = y0;
        //    nstep = IntegerPart[(x1 - x0)/h] + 1;
        //    Do[
        //     k1 = f[x[i], y[i]];
        //     k2 = f[x[i] + h, y[i] + h k1];
        //     x[i + 1] = x[i] + h;
        //     y[i + 1] = y[i] + h/2 (k1 + k2);
        //     , {i, 1, nstep}];
        //    Table[{x[i], y[i]}, {i, 1, nstep}]
        // ];
        //
        // f[x_, y_] := x + y;
        // x0 = 0;  y0 = 0;  x1 = 1;  h = 0.2;
        // xy = MdEulerSingleEq[f, x0, y0, x1, h];
        // err = Abs[#[[2]] - (Exp[#[[1]]] - #[[1]] - 1)] & /@ xy;
        //
        // Print["x = ", NumberForm[xy[[All, 1]], 20]]
        // Print["y = ", NumberForm[xy[[All, 2]], 20]]
        // Print["err = ", NumberForm[err, 20]]

        // compare with Mathematica results
        let xx_correct = &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let yy_correct = &[0.0, 0.02, 0.0884, 0.215848, 0.41533456, 0.7027081632000001];
        let errors_correct = &[
            0.0,
            0.001402758160169895,
            0.00342469764127043,
            0.006270800390509007,
            0.01020636849246781,
            0.01557366525904502,
        ];
        vec_approx_eq(&xx, xx_correct, 1e-15);
        vec_approx_eq(&yy_num, yy_correct, 1e-15);
        vec_approx_eq(&errors, errors_correct, 1e-15);
    }

    #[test]
    fn rk4_works() {
        // This test relates to Table 21.4 of Kreyszig's book, page 904

        // problem
        let (system, data, mut args) = Samples::kreyszig_eq6_page902();
        let mut yfx = data.y_analytical.unwrap();
        let ndim = system.ndim;

        // allocate structs
        let params = Params::new(Method::Rk4); // aka the Classical RK in Kreyszig's book
        let mut solver = ExplicitRungeKutta::new(params, system).unwrap();
        let mut work = Workspace::new(Method::FwEuler);

        // check dense output availability
        assert_eq!(
            solver.enable_dense_output().err(),
            Some("dense output is not available for the Rk4 method")
        );

        // numerical approximation
        let h = 0.2;
        let mut x = data.x0;
        let mut y = data.y0.clone();
        let mut y_ana = Vector::new(ndim);
        yfx(&mut y_ana, x);
        let mut xx = vec![x];
        let mut yy_num = vec![y[0]];
        let mut yy_ana = vec![y_ana[0]];
        let mut errors = vec![f64::abs(yy_num[0] - yy_ana[0])];
        for n in 0..5 {
            solver.step(&mut work, x, &y, h, &mut args).unwrap();
            assert_eq!(work.bench.n_function, (n + 1) * 4);

            solver.accept(&mut work, &mut x, &mut y, h, &mut args).unwrap();
            xx.push(x);
            yy_num.push(y[0]);

            yfx(&mut y_ana, x);
            yy_ana.push(y_ana[0]);
            errors.push(f64::abs(yy_num.last().unwrap() - yy_ana.last().unwrap()));
        }

        // Mathematica code:
        //
        // RK4SingleEq[f_, x0_, y0_, x1_, h_] := Module[{x, y, nstep, k1, k2, k3, k4},
        //    x[1] = x0;
        //    y[1] = y0;
        //    nstep = IntegerPart[(x1 - x0)/h] + 1;
        //    Do[
        //     k1 = f[x[i], y[i]];
        //     k2 = f[x[i] + 1/2 h, y[i] + h/2 k1];
        //     k3 = f[x[i] + 1/2 h, y[i] + h/2 k2];
        //     k4 = f[x[i] + h, y[i] + h k3];
        //     x[i + 1] = x[i] + h;
        //     y[i + 1] = y[i] + h/6 (k1 + 2 k2 + 2 k3 + k4);
        //     , {i, 1, nstep}];
        //    Table[{x[i], y[i]}, {i, 1, nstep}]
        // ];
        //
        // f[x_, y_] := x + y;
        // x0 = 0;  y0 = 0;  x1 = 1;  h = 0.2;
        // xy = RK4SingleEq[f, x0, y0, x1, h];
        // err = Abs[#[[2]] - (Exp[#[[1]]] - #[[1]] - 1)] & /@ xy;
        //
        // Print["x = ", NumberForm[xy[[All, 1]], 20]]
        // Print["y = ", NumberForm[xy[[All, 2]], 20]]
        // Print["err = ", NumberForm[err, 20]]

        // compare with Mathematica results
        let xx_correct = &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let yy_correct = &[
            0.0,
            0.0214,
            0.09181796,
            0.222106456344,
            0.4255208257785617,
            0.7182511366059352,
        ];
        let errors_correct = &[
            0.0,
            2.758160169896717e-6,
            6.737641270432304e-6,
            0.00001234404650901633,
            0.00002010271390617824,
            0.00003069185310988765,
        ];
        vec_approx_eq(&xx, xx_correct, 1e-15);
        vec_approx_eq(&yy_num, yy_correct, 1e-15);
        vec_approx_eq(&errors, errors_correct, 1e-15);
    }

    #[test]
    fn fehlberg4_step_works() {
        // Solving Equation (11) from Kreyszig's page 908 - Example 3
        //
        // ```text
        // dy
        // —— = (y - x - 1)² + 2   with   y(x=0)=1
        // dx
        //
        // y(x) = tan(x) + x + 1
        // ```
        let system = System::new(
            1,
            |f, x, y, _: &mut u8| {
                let d = y[0] - x - 1.0;
                f[0] = d * d + 2.0;
                Ok(())
            },
            no_jacobian,
            HasJacobian::No,
            None,
            None,
        );

        // allocate solver
        let params = Params::new(Method::Fehlberg4);
        let mut solver = ExplicitRungeKutta::new(params, system).unwrap();
        let mut work = Workspace::new(Method::FwEuler);

        // perform one step (compute k)
        let x = 0.0;
        let y = Vector::from(&[1.0]);
        let h = 0.1;
        let mut args = 0;
        solver.step(&mut work, x, &y, h, &mut args).unwrap();

        // compare with Kreyszig's results (Example 3, page 908)
        let kh: Vec<_> = solver.k.iter().map(|k| k[0] * h).collect();
        let kh_correct = &[
            0.200000000000,
            0.200062500000,
            0.200140756867,
            0.200856926154,
            0.201006676700,
            0.200250418651,
        ];
        vec_approx_eq(&kh, kh_correct, 1e-12);
    }
}
