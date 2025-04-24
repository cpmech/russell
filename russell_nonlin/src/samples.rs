use super::{NoArgs, State, System};
use russell_lab::math::{SQRT_2_BY_3, SQRT_3};
use russell_lab::Vector;
use russell_sparse::{CooMatrix, Sym};

/// Holds a collection of nonlinear problems
pub struct Samples {}

impl Samples {
    /// Cubic polynomial (causing problems to Newton's method)
    ///
    /// Returns `(system, state_ok, state_oscillation, state_indeterminate, u_ref, args)`
    ///
    /// ```text
    /// f = x³ - 2 x - 2
    /// J = 3 x² - 2
    /// ```
    pub fn cubic_poly_1<'a>() -> (System<'a, NoArgs>, State, State, State, Vector, NoArgs) {
        // system
        let ndim = 1;
        let mut system = System::new(ndim, |gg: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
            gg[0] = u[0] * u[0] * u[0] - 2.0 * u[0] - 2.0;
            Ok(())
        })
        .unwrap();

        // function to compute Gu
        let nnz = 1;
        system
            .set_calc_ggu(
                Some(nnz),
                Sym::No,
                |ggu: &mut CooMatrix, _l: f64, u: &Vector, _args: &mut NoArgs| {
                    ggu.put(0, 0, 3.0 * u[0] * u[0] - 2.0).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial states = trial for Newton's method
        let mut state_ok = State::new(ndim, false);
        let mut state_oscillation = State::new(ndim, false);
        let mut state_indeterminate = State::new(ndim, false);
        state_ok.u[0] = 1.0;
        state_oscillation.u[0] = 0.0;
        state_indeterminate.u[0] = SQRT_2_BY_3;

        // reference solution
        let u_reference = Vector::from(&[1.76929235423863]);

        // done
        let args = 0;
        (
            system,
            state_ok,
            state_oscillation,
            state_indeterminate,
            u_reference,
            args,
        )
    }

    /// Cubic polynomial (causing divergence problems to Newton's method)
    ///
    /// Returns `(system, state, u_reference,  args)`
    ///
    /// ```text
    /// f = (x - 1)³ + 0.512
    /// J = 3 (x - 1)²
    /// ```
    pub fn cubic_poly_2<'a>() -> (System<'a, NoArgs>, State, Vector, NoArgs) {
        // system
        let ndim = 1;
        let mut system = System::new(ndim, |gg: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
            gg[0] = f64::powi(u[0] - 1.0, 3) + 0.512;
            Ok(())
        })
        .unwrap();

        // function to compute Gu
        let nnz = 1;
        system
            .set_calc_ggu(
                Some(nnz),
                Sym::No,
                |ggu: &mut CooMatrix, _l: f64, u: &Vector, _args: &mut NoArgs| {
                    ggu.put(0, 0, 3.0 * f64::powi(u[0] - 1.0, 2)).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial state = trial for Newton's method
        let mut state = State::new(ndim, false);
        state.u[0] = 5.0;

        // reference solution
        let u_reference = Vector::from(&[0.2]);

        // done
        let args = 0;
        (system, state, u_reference, args)
    }

    /// Simple two-equation system with reference solution
    ///
    /// Returns `(system, state, u_reference, args)`
    pub fn two_eq_ref<'a>() -> (System<'a, NoArgs>, State, Vector, NoArgs) {
        // system
        let ndim = 2;
        let mut system = System::new(ndim, |gg: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
            gg[0] = u[0].powf(3.0) + u[1] - 1.0;
            gg[1] = -u[0] + u[1].powf(3.0) + 1.0;
            Ok(())
        })
        .unwrap();

        // function to compute Gu
        let nnz = 4;
        system
            .set_calc_ggu(
                Some(nnz),
                Sym::No,
                |ggu: &mut CooMatrix, _l: f64, u: &Vector, _args: &mut NoArgs| {
                    ggu.put(0, 0, 3.0 * u[0] * u[0]).unwrap();
                    ggu.put(0, 1, 1.0).unwrap();
                    ggu.put(1, 0, -1.0).unwrap();
                    ggu.put(1, 1, 3.0 * u[1] * u[1]).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial state = trial for Newton's method
        let mut state = State::new(ndim, false);
        state.u[0] = 0.5;
        state.u[1] = 0.5;

        // reference solution
        let u_reference = Vector::from(&[1.0, 0.0]);

        // done
        let args = 0;
        (system, state, u_reference, args)
    }

    /// Two-equation system causing problems to Newton-Raphson (singular)
    ///
    /// Returns `(system, state, u_reference, args)`
    ///
    /// The solution is (0,0) and the Jacobian is singular at this point.
    pub fn two_eq_nr_prob_1<'a>() -> (System<'a, NoArgs>, State, Vector, NoArgs) {
        // system
        let ndim = 2;
        let mut system = System::new(ndim, |gg: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
            gg[0] = u[0] * u[0] + u[1] * u[1];
            gg[1] = u[0] * u[0] - u[1] * u[1];
            Ok(())
        })
        .unwrap();

        // function to compute Gu
        let nnz = 4;
        system
            .set_calc_ggu(
                Some(nnz),
                Sym::No,
                |ggu: &mut CooMatrix, _l: f64, u: &Vector, _args: &mut NoArgs| {
                    ggu.put(0, 0, 2.0 * u[0]).unwrap();
                    ggu.put(0, 1, 2.0 * u[1]).unwrap();
                    ggu.put(1, 0, 2.0 * u[0]).unwrap();
                    ggu.put(1, 1, -2.0 * u[1]).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial state = trial for Newton's method
        let eps = 1e-5;
        let mut state = State::new(ndim, false);
        state.u[0] = 0.0;
        state.u[1] = eps;

        // reference solution
        let u_reference = Vector::from(&[0.0, 0.0]);

        // done
        let args = 0;
        (system, state, u_reference, args)
    }

    /// Two-equation system with two solutions and causing problems to Newton-Raphson
    ///
    /// Returns `(system, state_ok1, state_ok2, state_bad, u_ref1, u_ref2, args)`
    pub fn two_eq_nr_prob_2<'a>() -> (System<'a, NoArgs>, State, State, State, Vector, Vector, NoArgs) {
        // system
        let ndim = 2;
        let mut system = System::new(ndim, |gg: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
            gg[0] = f64::powi(u[0], 2) + f64::powi(u[1], 2) - 2.0; // circle centered @ (0, 0) with r = √2
            gg[1] = f64::powi(u[0] - 1.0, 2) + f64::powi(u[1] - 1.0, 2) - 2.0; // circle centered @ (1, 1) with r = √2
            Ok(())
        })
        .unwrap();

        // function to compute Gu
        let nnz = 4;
        system
            .set_calc_ggu(
                Some(nnz),
                Sym::No,
                |ggu: &mut CooMatrix, _l: f64, u: &Vector, _args: &mut NoArgs| {
                    ggu.put(0, 0, 2.0 * u[0]).unwrap();
                    ggu.put(0, 1, 2.0 * u[1]).unwrap();
                    ggu.put(1, 0, 2.0 * (u[0] - 1.0)).unwrap();
                    ggu.put(1, 1, 2.0 * (u[1] - 1.0)).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial state = trial for Newton's method
        // Note: det(J) = -4 u₀ + 4 u₁, thus, it may be zero when u₀ = u₁
        let mut state_ok1 = State::new(ndim, false);
        let mut state_ok2 = State::new(ndim, false);
        let mut state_bad = State::new(ndim, false);
        state_ok1.u[0] = -0.1;
        state_ok1.u[1] = 0.1;
        state_ok2.u[0] = 0.1;
        state_ok2.u[1] = -0.1;
        state_bad.u[0] = 0.0;
        state_bad.u[1] = 0.0;

        // reference solutions
        let u_ref1 = Vector::from(&[(1.0 - SQRT_3) / 2.0, (1.0 + SQRT_3) / 2.0]);
        let u_ref2 = Vector::from(&[(1.0 + SQRT_3) / 2.0, (1.0 - SQRT_3) / 2.0]);

        // done
        let args = 0;
        (system, state_ok1, state_ok2, state_bad, u_ref1, u_ref2, args)
    }

    /// Single equation with a fold point
    ///
    /// Returns `(system, state, lambda_ana, args)` where `lambda_ana` is `f(u) -> λ`
    ///
    /// See Reference 1, page 70.
    ///
    /// 1. Bank RE and Mittelmann HD (1990) Stepsize selection in continuation procedures and damped Newton's method.
    ///    In Continuation Techniques and Bifurcation Problems, Ed. by Mittelmann HD and Roose D (1990), Springer.
    pub fn single_eq_with_fold_point<'a>() -> (System<'a, NoArgs>, State, fn(f64) -> f64, NoArgs) {
        // system
        let ndim = 1;
        let mut system = System::new(ndim, |gg: &mut Vector, l: f64, u: &Vector, _args: &mut NoArgs| {
            gg[0] = u[0] - l * f64::exp(u[0]);
            Ok(())
        })
        .unwrap();

        // function to compute Gu
        let nnz_ggu = 1;
        system
            .set_calc_ggu(
                Some(nnz_ggu),
                Sym::No,
                |ggu: &mut CooMatrix, l: f64, u: &Vector, _args: &mut NoArgs| {
                    ggu.put(0, 0, 1.0 - l * f64::exp(u[0])).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // function to compute Gl
        system
            .set_calc_ggl(|ggl: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
                ggl[0] = -f64::exp(u[0]);
                Ok(())
            })
            .unwrap();

        // initial state
        let mut state = State::new(ndim, true);
        state.u[0] = 0.0;
        state.l = 0.0;

        // reference solution
        let lambda_ana = |u: f64| f64::exp(-u) * u;

        // done
        let args = 0;
        (system, state, lambda_ana, args)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Samples;
    use russell_lab::{algo::num_jacobian, Vector};
    use russell_lab::{mat_approx_eq, vec_approx_eq};
    use russell_sparse::{CooMatrix, Sym};

    #[test]
    fn test_cubic_poly_1() {
        // system
        let (system, state, _, _, _, mut args) = Samples::cubic_poly_1();

        // analytical Jacobian
        let mut ggu = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        let ggu_fn = system.calc_ggu.as_ref().unwrap();
        (ggu_fn)(&mut ggu, 0.0, &state.u, &mut args).unwrap();

        // numerical Jacobian
        let num = num_jacobian(system.ndim, 0.0, &state.u, 1.0, &mut args, system.calc_gg.as_ref()).unwrap();
        let ana = ggu.as_dense();

        // check
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn test_cubic_poly_2() {
        // system
        let (system, state, _, mut args) = Samples::cubic_poly_2();

        // analytical Jacobian
        let mut ggu = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        let ggu_fn = system.calc_ggu.as_ref().unwrap();
        (ggu_fn)(&mut ggu, 0.0, &state.u, &mut args).unwrap();

        // numerical Jacobian
        let num = num_jacobian(system.ndim, 0.0, &state.u, 1.0, &mut args, system.calc_gg.as_ref()).unwrap();
        let ana = ggu.as_dense();

        // check
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-10);
    }

    #[test]
    fn test_simple_two_equations() {
        // system
        let (system, state, _, mut args) = Samples::two_eq_ref();

        // analytical Jacobian
        let mut ggu = CooMatrix::new(2, 2, 4, Sym::No).unwrap();
        let ggu_fn = system.calc_ggu.as_ref().unwrap();
        (ggu_fn)(&mut ggu, 0.0, &state.u, &mut args).unwrap();

        // numerical Jacobian
        let num = num_jacobian(system.ndim, 0.0, &state.u, 1.0, &mut args, system.calc_gg.as_ref()).unwrap();
        let ana = ggu.as_dense();

        // check
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-10);
    }

    #[test]
    fn test_two_eq_nr_prob_1() {
        // system
        let (system, state, _, mut args) = Samples::two_eq_nr_prob_1();

        // analytical Jacobian
        let mut ggu = CooMatrix::new(2, 2, 4, Sym::No).unwrap();
        let ggu_fn = system.calc_ggu.as_ref().unwrap();
        (ggu_fn)(&mut ggu, 0.0, &state.u, &mut args).unwrap();

        // numerical Jacobian
        let num = num_jacobian(system.ndim, 0.0, &state.u, 1.0, &mut args, system.calc_gg.as_ref()).unwrap();
        let ana = ggu.as_dense();

        // check
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-15);
    }

    #[test]
    fn test_two_eq_nr_prob_2() {
        // system
        let (system, state, _, _, _, _, mut args) = Samples::two_eq_nr_prob_2();

        // analytical Jacobian
        let mut ggu = CooMatrix::new(2, 2, 4, Sym::No).unwrap();
        let ggu_fn = system.calc_ggu.as_ref().unwrap();
        (ggu_fn)(&mut ggu, 0.0, &state.u, &mut args).unwrap();

        // numerical Jacobian
        let num = num_jacobian(system.ndim, 0.0, &state.u, 1.0, &mut args, system.calc_gg.as_ref()).unwrap();
        let ana = ggu.as_dense();

        // check
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-12);
    }

    #[test]
    fn test_single_eq_with_fold_point() {
        // system
        let (system, state, _, mut args) = Samples::single_eq_with_fold_point();

        // analytical Jacobian
        let mut ggu = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        let ggu_fn = system.calc_ggu.as_ref().unwrap();
        (ggu_fn)(&mut ggu, state.l, &state.u, &mut args).unwrap();

        // numerical Jacobian
        let num = num_jacobian(system.ndim, state.l, &state.u, 1.0, &mut args, system.calc_gg.as_ref()).unwrap();
        let ana = ggu.as_dense();

        // check Gu
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-15);

        // check Gl
        let ggl_fn = system.calc_ggl.as_ref().unwrap();
        let mut ggl = Vector::new(1);
        (ggl_fn)(&mut ggl, state.l, &state.u, &mut args).unwrap();
        vec_approx_eq(&ggl, &[-1.0], 1e-15);
    }
}
