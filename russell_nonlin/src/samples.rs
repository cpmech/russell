use super::{NoArgs, System};
use russell_lab::math::{SQRT_2, SQRT_2_BY_3, SQRT_3};
use russell_lab::{Bspline, Vector};
use russell_sparse::Sym;

/// Holds a collection of nonlinear problems
pub struct Samples {}

/// Holds extra arguments for the B-spline problem
pub struct SampleBsplineArgs {
    pub bspline: Bspline,
    pub coords: Vector,
}

impl Samples {
    /// Simple linear problem: G(u, λ) = u - λ
    ///
    /// Returns `(system, u, args)`
    ///
    /// ```text
    /// G(u, λ) = u - λ
    /// Gu = ∂G/∂u = 1
    /// Gl = ∂G/∂λ = -1
    /// ```
    pub fn simple_linear_problem<'a>(sym: Sym) -> (System<'a, NoArgs>, Vector, f64, NoArgs) {
        // nonlinear problem: G(u, λ) = u - λ
        let ndim = 1;
        let nnz = Some(1);
        let system = System::new(
            ndim,
            nnz,
            sym,
            |gg, l, u, _args: &mut NoArgs| {
                gg[0] = u[0] - l;
                Ok(())
            },
            |ggu, ggl, _l, _u, _args: &mut NoArgs| {
                ggu.put(0, 0, 1.0).unwrap();
                ggl[0] = -1.0;
                Ok(())
            },
        )
        .unwrap();

        // initial state
        let u = Vector::from(&[0.0]);
        let l = 0.0;

        // done
        let args = 0;
        (system, u, l, args)
    }

    /// Cubic polynomial (causing problems to Newton's method)
    ///
    /// Returns `(system, u_ok, u_oscillation, u_indeterminate, u_ref, args)`
    ///
    /// ```text
    /// f = x³ - 2 x - 2
    /// J = 3 x² - 2
    /// ```
    pub fn cubic_poly_1<'a>() -> (System<'a, NoArgs>, Vector, Vector, Vector, Vector, NoArgs) {
        // system
        let ndim = 1;
        let nnz = Some(1);
        let sym = Sym::No;
        let system = System::new(
            ndim,
            nnz,
            sym,
            |gg, _l, u, _args| {
                gg[0] = u[0] * u[0] * u[0] - 2.0 * u[0] - 2.0;
                Ok(())
            },
            |ggu, _ggl, _l, u, _args| {
                ggu.put(0, 0, 3.0 * u[0] * u[0] - 2.0).unwrap();
                Ok(())
            },
        )
        .unwrap();

        // initial states = trial for Newton's method
        let u_ok = Vector::from(&[1.0]);
        let u_oscillation = Vector::from(&[0.0]);
        let u_indeterminate = Vector::from(&[SQRT_2_BY_3]);

        // reference solution
        let u_reference = Vector::from(&[1.76929235423863]);

        // done
        let args = 0;
        (system, u_ok, u_oscillation, u_indeterminate, u_reference, args)
    }

    /// Cubic polynomial (causing divergence problems to Newton's method)
    ///
    /// Returns `(system, u, u_reference,  args)`
    ///
    /// ```text
    /// f = (x - 1)³ + 0.512
    /// J = 3 (x - 1)²
    /// ```
    pub fn cubic_poly_2<'a>() -> (System<'a, NoArgs>, Vector, Vector, NoArgs) {
        // system
        let ndim = 1;
        let nnz = Some(1);
        let sym = Sym::No;
        let system = System::new(
            ndim,
            nnz,
            sym,
            |gg, _l, u, _args| {
                gg[0] = f64::powi(u[0] - 1.0, 3) + 0.512;
                Ok(())
            },
            |ggu, _ggl, _l, u, _args| {
                ggu.put(0, 0, 3.0 * f64::powi(u[0] - 1.0, 2)).unwrap();
                Ok(())
            },
        )
        .unwrap();

        // initial state = trial for Newton's method
        let u = Vector::from(&[5.0]);

        // reference solution
        let u_reference = Vector::from(&[0.2]);

        // done
        let args = 0;
        (system, u, u_reference, args)
    }

    /// Simple two-equation system with reference solution
    ///
    /// Returns `(system, u, u_reference, args)`
    pub fn two_eq_ref<'a>() -> (System<'a, NoArgs>, Vector, Vector, NoArgs) {
        // system
        let ndim = 2;
        let nnz = Some(4);
        let sym = Sym::No;
        let system = System::new(
            ndim,
            nnz,
            sym,
            |gg, _l, u, _args| {
                gg[0] = u[0].powf(3.0) + u[1] - 1.0;
                gg[1] = -u[0] + u[1].powf(3.0) + 1.0;
                Ok(())
            },
            |ggu, _ggl, _l, u, _args| {
                ggu.put(0, 0, 3.0 * u[0] * u[0]).unwrap();
                ggu.put(0, 1, 1.0).unwrap();
                ggu.put(1, 0, -1.0).unwrap();
                ggu.put(1, 1, 3.0 * u[1] * u[1]).unwrap();
                Ok(())
            },
        )
        .unwrap();

        // initial state = trial for Newton's method
        let u = Vector::from(&[0.5, 0.5]);

        // reference solution
        let u_reference = Vector::from(&[1.0, 0.0]);

        // done
        let args = 0;
        (system, u, u_reference, args)
    }

    /// Two-equation system causing problems to Newton-Raphson (singular)
    ///
    /// Returns `(system, u, u_reference, args)`
    ///
    /// The solution is (0,0) and the Jacobian is singular at this point.
    pub fn two_eq_nr_prob_1<'a>() -> (System<'a, NoArgs>, Vector, Vector, NoArgs) {
        // system
        let ndim = 2;
        let nnz = Some(4);
        let sym = Sym::No;
        let system = System::new(
            ndim,
            nnz,
            sym,
            |gg, _l, u, _args| {
                gg[0] = u[0] * u[0] + u[1] * u[1];
                gg[1] = u[0] * u[0] - u[1] * u[1];
                Ok(())
            },
            |ggu, _ggl, _l, u, _args| {
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
        let u = Vector::from(&[0.0, eps]);

        // reference solution
        let u_reference = Vector::from(&[0.0, 0.0]);

        // done
        let args = 0;
        (system, u, u_reference, args)
    }

    /// Two-equation system with two solutions and causing problems to Newton-Raphson
    ///
    /// Returns `(system, u_ok1, u_ok2, u_bad, u_ref1, u_ref2, args)`
    pub fn two_eq_nr_prob_2<'a>() -> (System<'a, NoArgs>, Vector, Vector, Vector, Vector, Vector, NoArgs) {
        // system
        let ndim = 2;
        let nnz = Some(4);
        let sym = Sym::No;
        let system = System::new(
            ndim,
            nnz,
            sym,
            |gg, _l, u, _args| {
                gg[0] = f64::powi(u[0], 2) + f64::powi(u[1], 2) - 2.0; // circle centered @ (0, 0) with r = √2
                gg[1] = f64::powi(u[0] - 1.0, 2) + f64::powi(u[1] - 1.0, 2) - 2.0; // circle centered @ (1, 1) with r = √2
                Ok(())
            },
            |ggu, _ggl, _l, u, _args| {
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
        let u_ok1 = Vector::from(&[-0.1, 0.1]);
        let u_ok2 = Vector::from(&[0.1, -0.1]);
        let u_bad = Vector::from(&[0.0, 0.0]);

        // reference solutions
        let u_ref1 = Vector::from(&[(1.0 - SQRT_3) / 2.0, (1.0 + SQRT_3) / 2.0]);
        let u_ref2 = Vector::from(&[(1.0 + SQRT_3) / 2.0, (1.0 - SQRT_3) / 2.0]);

        // done
        let args = 0;
        (system, u_ok1, u_ok2, u_bad, u_ref1, u_ref2, args)
    }

    /// One equation with a fold point
    ///
    /// Returns `(system, u, l, lambda_ana, args)` where `lambda_ana` is `f(u) -> λ`
    ///
    /// See Reference 1, page 70.
    ///
    /// 1. Bank RE and Mittelmann HD (1990) Stepsize selection in continuation procedures and damped Newton's method.
    ///    In Continuation Techniques and Bifurcation Problems, Ed. by Mittelmann HD and Roose D (1990), Springer.
    pub fn one_eq_with_fold_point<'a>() -> (System<'a, NoArgs>, Vector, f64, fn(f64) -> f64, NoArgs) {
        // system
        let ndim = 1;
        let nnz = Some(1);
        let sym = Sym::No;
        let system = System::new(
            ndim,
            nnz,
            sym,
            |gg, l, u, _args| {
                gg[0] = u[0] - l * f64::exp(u[0]);
                Ok(())
            },
            |ggu, ggl, l, u, _args| {
                ggu.put(0, 0, 1.0 - l * f64::exp(u[0])).unwrap();
                ggl[0] = -f64::exp(u[0]);
                Ok(())
            },
        )
        .unwrap();

        // initial state
        let u = Vector::from(&[0.0]);
        let l = 0.0;

        // reference solution
        let lambda_ana = |u: f64| f64::exp(-u) * u;

        // done
        let args = 0;
        (system, u, l, lambda_ana, args)
    }

    /// One equation with singular initial state
    ///
    /// Returns `(system, u, l, lambda_ana, args)` where `lambda_ana` is `f(u) -> λ`
    pub fn singular_initial_state<'a>(
        alpha: f64,
        perturbation: f64,
    ) -> (System<'a, NoArgs>, Vector, f64, impl Fn(f64) -> f64, NoArgs) {
        // system
        let ndim = 1;
        let nnz = Some(1);
        let sym = Sym::No;
        let system = System::new(
            ndim,
            nnz,
            sym,
            move |gg, l, u, _args| {
                gg[0] = f64::powf(u[0], alpha) - l;
                Ok(())
            },
            move |ggu, ggl, _l, u, _args| {
                ggu.put(0, 0, alpha * f64::powf(u[0], alpha - 1.0)).unwrap();
                ggl[0] = -1.0;
                Ok(())
            },
        )
        .unwrap();

        // initial state
        let u = Vector::from(&[perturbation]);
        let l = 0.0;

        // reference solution
        let lambda_ana = move |u: f64| f64::powf(u, alpha);

        // done
        let args = 0;
        (system, u, l, lambda_ana, args)
    }

    /// B-spline problem # 1
    ///
    /// Returns `(system, u, l, args)`
    pub fn bspline_problem_1<'a>(
        snap_back_delta: f64,
    ) -> (System<'a, SampleBsplineArgs>, Vector, f64, SampleBsplineArgs) {
        // define the nonlinear system: G(u, λ)
        let ndim = 2;
        let nnz = Some(2);
        let sym = Sym::No;
        let system = System::new(
            ndim,
            nnz,
            sym,
            |gg: &mut Vector, l: f64, u: &Vector, args: &mut SampleBsplineArgs| {
                let t = f64::min(1.0, f64::max(0.0, l));
                args.bspline.calc_point(&mut args.coords, t, false)?;
                gg[0] = u[0] - args.coords[0];
                gg[1] = u[1] - args.coords[1];
                Ok(())
            },
            |ggu, ggl, l, _u, args: &mut SampleBsplineArgs| {
                // Gu = ∂G/∂u
                ggu.put(0, 0, 1.0).unwrap();
                ggu.put(1, 1, 1.0).unwrap();
                // Gl = ∂G/∂λ
                let t = f64::min(1.0, f64::max(0.0, l));
                args.bspline.calc_curve_derivs(t, 1, false);
                args.bspline.get_curve_deriv(&mut args.coords, 1);
                ggl[0] = -args.coords[0];
                ggl[1] = -args.coords[1];
                Ok(())
            },
        )
        .unwrap();

        // allocate B-spline and extra arguments
        let degree = 2;
        let knots = &[0.0, 0.0, 0.0, 2.0 / 5.0, 3.0 / 5.0, 1.0, 1.0, 1.0];
        let mut args = SampleBsplineArgs {
            bspline: Bspline::new(degree, knots).unwrap(),
            coords: Vector::new(2),
        };
        let control = &[
            [0.0, 0.0],                   // P0
            [0.5, 1.0],                   // P1
            [1.75, 1.0],                  // P2
            [2.0 - snap_back_delta, 0.0], // P3
            [2.5, 0.5],                   // P4
        ];
        args.bspline.set_control_points(control).unwrap();

        // initial state
        let u = Vector::from(&[0.0, 0.0]);
        let l = 0.0;

        // done
        (system, u, l, args)
    }

    /// Circle on the u-λ space
    ///
    /// Returns `(system, u, l, args)`
    pub fn circle_ul<'a>(radius: f64) -> (System<'a, NoArgs>, Vector, f64, NoArgs) {
        // system
        let ndim = 1;
        let nnz = Some(1);
        let sym = Sym::No;
        let system = System::new(
            ndim,
            nnz,
            sym,
            move |gg, l, u, _args| {
                gg[0] = u[0] * u[0] + l * l - radius * radius;
                Ok(())
            },
            |ggu, ggl, l, u, _args| {
                ggu.put(0, 0, 2.0 * u[0]).unwrap();
                ggl[0] = 2.0 * l;
                Ok(())
            },
        )
        .unwrap();

        // initial state: point on the first quadrant with u = l
        let u = Vector::from(&[radius / SQRT_2]);
        let l = radius / SQRT_2;

        // done
        let args = 0;
        (system, u, l, args)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Samples;
    use russell_lab::approx_eq;
    use russell_lab::math::SQRT_2;
    use russell_lab::Vector;

    #[test]
    fn test_samples_cubic_poly_1() {
        let (system, u, _, _, _, mut args) = Samples::cubic_poly_1();
        system.check_ggu(0.0, &u, &mut args, 1e-11).unwrap();
    }

    #[test]
    fn test_samples_cubic_poly_2() {
        let (system, u, _, mut args) = Samples::cubic_poly_2();
        system.check_ggu(0.0, &u, &mut args, 1e-10).unwrap();
    }

    #[test]
    fn test_samples_simple_two_equations() {
        let (system, u, _, mut args) = Samples::two_eq_ref();
        system.check_ggu(0.0, &u, &mut args, 1e-10).unwrap();
    }

    #[test]
    fn test_samples_two_eq_nr_prob_1() {
        let (system, u, _, mut args) = Samples::two_eq_nr_prob_1();
        system.check_ggu(0.0, &u, &mut args, 1e-15).unwrap();
    }

    #[test]
    fn test_samples_two_eq_nr_prob_2() {
        let (system, u, _, _, _, _, mut args) = Samples::two_eq_nr_prob_2();
        system.check_ggu(0.0, &u, &mut args, 1e-12).unwrap();
    }

    #[test]
    fn test_samples_one_eq_with_fold_point() {
        let (system, u, l, _, mut args) = Samples::one_eq_with_fold_point();
        system.check_ggu(l, &u, &mut args, 1e-15).unwrap();
    }

    #[test]
    fn test_samples_arc_singular_initial_state() {
        let (system, u, l, _, mut args) = Samples::singular_initial_state(2.0, 1e-2);
        system.check_ggu(l, &u, &mut args, 1e-15).unwrap();
    }

    #[test]
    fn test_samples_bspline_problem_1() {
        let (system, u, l, mut args) = Samples::bspline_problem_1(0.5);
        system.check_ggu(l, &u, &mut args, 1e-15).unwrap();
    }

    #[test]
    fn test_samples_circle_ul() {
        // system
        let radius = SQRT_2;
        let (system, u, l, mut args) = Samples::circle_ul(radius);
        system.check_ggu(l, &u, &mut args, 1e-12).unwrap();

        // check initial G(u, λ) = 0
        let ndim = system.get_ndim();
        let gg_fn = system.calc_gg.as_ref();
        let mut gg = Vector::new(ndim);
        gg_fn(&mut gg, l, &u, &mut args).unwrap();
        approx_eq(gg[0], 0.0, 1e-15);
    }
}
