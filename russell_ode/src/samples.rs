use crate::StrError;
use crate::{HasJacobian, System};
use russell_lab::Vector;
use russell_sparse::CooMatrix;

/// Holds data corresponding to a sample ODE problem
pub struct SampleData<'a> {
    /// Holds the initial x
    pub x0: f64,

    /// Holds the initial y
    pub y0: Vector,

    /// Holds the final x
    pub x1: f64,

    /// Holds the stepsize for simulations with equal-steps
    pub h_equal: Option<f64>,

    /// Holds the analytical solution `y(x)`
    pub y_analytical: Option<Box<dyn 'a + FnMut(&mut Vector, f64)>>,
}

/// Indicates that the sample ODE problem does not have extra arguments
pub type SampleNoArgs = u8;

/// Holds a collection of sample ODE problems
///
/// # References
///
/// 1. Hairer E, Nørsett, SP, Wanner G (2008) Solving Ordinary Differential Equations I.
///    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
///    in Computational Mathematics, 528p
/// 2. Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
///    Stiff and Differential-Algebraic Problems. Second Revised Edition.
///    Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
/// 3. Kreyszig, E (2011) Advanced engineering mathematics; in collaboration with Kreyszig H,
///    Edward JN 10th ed 2011, Hoboken, New Jersey, Wiley
pub struct Samples {}

impl Samples {
    /// Implements Equation (6) from Kreyszig's book on page 902
    ///
    /// ```text
    /// dy/dx = x + y
    /// y(0) = 0
    /// ```
    ///
    /// # Output
    ///
    /// Returns `(system, data, args)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `SampleNoArgs`
    /// * `data: SampleData` -- holds the initial values and the analytical solution
    /// * `args: SampleNoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Kreyszig, E (2011) Advanced engineering mathematics; in collaboration with Kreyszig H,
    ///    Edward JN 10th ed 2011, Hoboken, New Jersey, Wiley
    pub fn kreyszig_eq6_page902<'a>() -> (
        System<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut SampleNoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut SampleNoArgs) -> Result<(), StrError>,
            SampleNoArgs,
        >,
        SampleData<'a>,
        SampleNoArgs,
    ) {
        let ndim = 1;
        let jac_nnz = 1;
        let system = System::new(
            ndim,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut SampleNoArgs| {
                f[0] = x + y[0];
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, _y: &Vector, multiplier: f64, _args: &mut SampleNoArgs| {
                jj.reset();
                jj.put(0, 0, 1.0 * multiplier)?;
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            None,
        );
        let data = SampleData {
            x0: 0.0,
            y0: Vector::from(&[0.0]),
            x1: 1.0,
            h_equal: Some(0.2),
            y_analytical: Some(Box::new(|y, x| {
                y[0] = f64::exp(x) - x - 1.0;
            })),
        };
        (system, data, 0)
    }

    /// Implements a simple system with two equations (with analytical solution)
    ///
    /// ```text
    /// dy0/dx = -x y1
    /// dy1/dx =  x y0
    /// y0(0) = P
    /// y1(0) = Q
    /// ```
    ///
    /// # Output
    ///
    /// Returns `(system, data, args)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `SampleNoArgs`
    /// * `data: SampleData` -- holds the initial values and the analytical solution
    /// * `args: SampleNoArgs` -- is a placeholder variable with the arguments to F and J
    pub fn simple_system<'a>() -> (
        System<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut SampleNoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut SampleNoArgs) -> Result<(), StrError>,
            SampleNoArgs,
        >,
        SampleData<'a>,
        SampleNoArgs,
    ) {
        const P: f64 = -1.0;
        const Q: f64 = 1.0;
        let ndim = 2;
        let jac_nnz = 2;
        let system = System::new(
            ndim,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut SampleNoArgs| {
                f[0] = -x * y[1];
                f[1] = x * y[0];
                Ok(())
            },
            |jj: &mut CooMatrix, x: f64, _y: &Vector, multiplier: f64, _args: &mut SampleNoArgs| {
                jj.reset();
                jj.put(0, 1, -x * multiplier)?;
                jj.put(1, 0, x * multiplier)?;
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            None,
        );
        let data = SampleData {
            x0: 0.0,
            y0: Vector::from(&[P, Q]),
            x1: 5.0,
            h_equal: None,
            y_analytical: Some(Box::new(|y, x| {
                let v = x * x / 2.0;
                let c = f64::cos(v);
                let s = f64::sin(v);
                y[0] = P * c - Q * s;
                y[1] = Q * c + P * s;
            })),
        };
        (system, data, 0)
    }

    /// Returns the Hairer-Wanner problem from the reference, Part II, Eq(1.1), page 2 (with analytical solution)
    ///
    /// # Output
    ///
    /// Returns `(system, data, args)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `SampleNoArgs`
    /// * `data: SampleData` -- holds the initial values and the analytical solution
    /// * `args: SampleNoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn hairer_wanner_eq1<'a>() -> (
        System<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut SampleNoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut SampleNoArgs) -> Result<(), StrError>,
            SampleNoArgs,
        >,
        SampleData<'a>,
        SampleNoArgs,
    ) {
        const L: f64 = -50.0; // lambda
        let ndim = 1;
        let jac_nnz = 1;
        let system = System::new(
            ndim,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut SampleNoArgs| {
                f[0] = L * (y[0] - f64::cos(x));
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, _y: &Vector, multiplier: f64, _args: &mut SampleNoArgs| {
                jj.reset();
                jj.put(0, 0, multiplier * L)?;
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            None,
        );
        let data = SampleData {
            x0: 0.0,
            y0: Vector::from(&[0.0]),
            x1: 1.5,
            h_equal: Some(1.875 / 50.0),
            y_analytical: Some(Box::new(|y, x| {
                y[0] = -L * (f64::sin(x) - L * f64::cos(x) + L * f64::exp(L * x)) / (L * L + 1.0);
            })),
        };
        (system, data, 0)
    }

    /// Returns the Robertson's equation, Hairer-Wanner, Part II, Eq(1.4), page 3
    ///
    /// # Output
    ///
    /// Returns `(system, data, args)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `SampleNoArgs`
    /// * `data: SampleData` -- holds the initial values
    /// * `args: SampleNoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn robertson<'a>() -> (
        System<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut SampleNoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut SampleNoArgs) -> Result<(), StrError>,
            SampleNoArgs,
        >,
        SampleData<'a>,
        SampleNoArgs,
    ) {
        let ndim = 3;
        let jac_nnz = 7;
        let system = System::new(
            ndim,
            |f: &mut Vector, _x: f64, y: &Vector, _args: &mut SampleNoArgs| {
                f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
                f[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
                f[2] = 3.0e7 * y[1] * y[1];
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, y: &Vector, multiplier: f64, _args: &mut SampleNoArgs| {
                jj.reset();
                jj.put(0, 0, -0.04 * multiplier)?;
                jj.put(0, 1, 1.0e4 * y[2] * multiplier)?;
                jj.put(0, 2, 1.0e4 * y[1] * multiplier)?;
                jj.put(1, 0, 0.04 * multiplier)?;
                jj.put(1, 1, (-1.0e4 * y[2] - 6.0e7 * y[1]) * multiplier)?;
                jj.put(1, 2, (-1.0e4 * y[1]) * multiplier)?;
                jj.put(2, 1, 6.0e7 * y[1] * multiplier)?;
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            None,
        );
        let data = SampleData {
            x0: 0.0,
            y0: Vector::from(&[1.0, 0.0, 0.0]),
            x1: 0.3,
            h_equal: None,
            y_analytical: None,
        };
        (system, data, 0)
    }

    /// Returns the Van der Pol's equation as given in Hairer-Wanner, Part II, Eq(1.5'), page 5
    ///
    /// **Note:** Using the data from Eq(7.29), page 113.
    ///
    /// # Output
    ///
    /// Returns `(system, data, args)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `SampleNoArgs`
    /// * `data: SampleData` -- holds the initial values
    /// * `args: SampleNoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Input
    ///
    /// * `epsilon` -- ε coefficient; use None for the default value (= 1.0e-6)
    /// * `stationary` -- use `ε = 1` and compute the period and amplitude such that
    ///   `y = [A, 0]` is a stationary point.
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn van_der_pol<'a>(
        epsilon: Option<f64>,
        stationary: bool,
    ) -> (
        System<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut SampleNoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut SampleNoArgs) -> Result<(), StrError>,
            SampleNoArgs,
        >,
        SampleData<'a>,
        SampleNoArgs,
    ) {
        let mut eps = match epsilon {
            Some(e) => e,
            None => 1.0e-6,
        };
        let x0 = 0.0;
        let mut y0 = Vector::from(&[2.0, -0.6]);
        let mut x1 = 2.0;
        if stationary {
            eps = 1.0;
            const A: f64 = 2.00861986087484313650940188;
            const T: f64 = 6.6632868593231301896996820305;
            y0[0] = A;
            y0[1] = 0.0;
            x1 = T;
        }
        let ndim = 2;
        let jac_nnz = 3;
        let system = System::new(
            ndim,
            move |f: &mut Vector, _x: f64, y: &Vector, _args: &mut SampleNoArgs| {
                f[0] = y[1];
                f[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / eps;
                Ok(())
            },
            move |jj: &mut CooMatrix, _x: f64, y: &Vector, multiplier: f64, _args: &mut SampleNoArgs| {
                jj.reset();
                jj.put(0, 1, 1.0 * multiplier)?;
                jj.put(1, 0, multiplier * (-2.0 * y[0] * y[1] - 1.0) / eps)?;
                jj.put(1, 1, multiplier * (1.0 - y[0] * y[0]) / eps)?;
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            None,
        );
        let data = SampleData {
            x0,
            y0,
            x1,
            h_equal: None,
            y_analytical: None,
        };
        (system, data, 0)
    }

    /// Returns the Arenstorf orbit problem, Hairer-Wanner, Part I, Eq(0.1), page 129
    ///
    /// From Hairer-Wanner:
    ///
    /// "(...) an example from Astronomy, the restricted three body problem. (...)
    /// two bodies of masses μ' = 1 − μ and μ in circular rotation in a plane and
    /// a third body of negligible mass moving around in the same plane. (...)"
    ///
    /// ```text
    /// y0'' = y0 + 2 y1' - μ' (y0 + μ) / d0 - μ (y0 - μ') / d1
    /// y1'' = y1 - 2 y0' - μ' y1 / d0 - μ y1 / d1
    /// ```
    ///
    /// ```text
    /// y2 := y0'  ⇒  y2' = y0''
    /// y3 := y1'  ⇒  y3' = y1''
    /// ```
    ///
    /// ```text
    /// f0 := y0' = y2
    /// f1 := y1' = y3
    /// f2 := y2' = y0 + 2 y3 - μ' (y0 + μ) / d0 - μ (y0 - μ') / d1
    /// f3 := y3' = y1 - 2 y2 - μ' y1 / d0 - μ y1 / d1
    /// ```
    ///
    /// # Output
    ///
    /// Returns `(system, data, args)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `SampleNoArgs`
    /// * `data: SampleData` -- holds the initial values
    /// * `args: SampleNoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn arenstorf<'a>() -> (
        System<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut SampleNoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut SampleNoArgs) -> Result<(), StrError>,
            SampleNoArgs,
        >,
        SampleData<'a>,
        SampleNoArgs,
    ) {
        const MU: f64 = 0.012277471;
        const MD: f64 = 1.0 - MU;
        let x0 = 0.0;
        let y0 = Vector::from(&[0.994, 0.0, 0.0, -2.00158510637908252240537862224]);
        let x1 = 17.0652165601579625588917206249;
        let ndim = 4;
        let jac_nnz = 8;
        let system = System::new(
            ndim,
            |f: &mut Vector, _x: f64, y: &Vector, _args: &mut SampleNoArgs| {
                let t0 = (y[0] + MU) * (y[0] + MU) + y[1] * y[1];
                let t1 = (y[0] - MD) * (y[0] - MD) + y[1] * y[1];
                let d0 = t0 * f64::sqrt(t0);
                let d1 = t1 * f64::sqrt(t1);
                f[0] = y[2];
                f[1] = y[3];
                f[2] = y[0] + 2.0 * y[3] - MD * (y[0] + MU) / d0 - MU * (y[0] - MD) / d1;
                f[3] = y[1] - 2.0 * y[2] - MD * y[1] / d0 - MU * y[1] / d1;
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, y: &Vector, m: f64, _args: &mut SampleNoArgs| {
                let t0 = (y[0] + MU) * (y[0] + MU) + y[1] * y[1];
                let t1 = (y[0] - MD) * (y[0] - MD) + y[1] * y[1];
                let s0 = f64::sqrt(t0);
                let s1 = f64::sqrt(t1);
                let d0 = t0 * s0;
                let d1 = t1 * s1;
                let dd0 = d0 * d0;
                let dd1 = d1 * d1;
                let a = y[0] + MU;
                let b = y[0] - MD;
                let c = -MD / d0 - MU / d1;
                let dj00 = 3.0 * a * s0;
                let dj01 = 3.0 * y[1] * s0;
                let dj10 = 3.0 * b * s1;
                let dj11 = 3.0 * y[1] * s1;
                jj.reset();
                jj.put(0, 2, 1.0 * m)?;
                jj.put(1, 3, 1.0 * m)?;
                jj.put(2, 0, (1.0 + a * dj00 * MD / dd0 + b * dj10 * MU / dd1 + c) * m)?;
                jj.put(2, 1, (a * dj01 * MD / dd0 + b * dj11 * MU / dd1) * m)?;
                jj.put(2, 3, 2.0 * m)?;
                jj.put(3, 0, (dj00 * y[1] * MD / dd0 + dj10 * y[1] * MU / dd1) * m)?;
                jj.put(3, 1, (1.0 + dj01 * y[1] * MD / dd0 + dj11 * y[1] * MU / dd1 + c) * m)?;
                jj.put(3, 2, -2.0 * m)?;
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            None,
        );
        let data = SampleData {
            x0,
            y0,
            x1,
            h_equal: None,
            y_analytical: None,
        };
        (system, data, 0)
    }

    /// Returns the transistor amplifier problem described by Hairer-Wanner, Part II, page 376
    ///
    /// **Note:** The equations hare are taken from Hairer's website, not the book.
    ///
    /// # Output
    ///
    /// Returns `(system, data, args, gen_mass_matrix)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `SampleNoArgs`
    /// * `data: SampleData` -- holds the initial values
    /// * `args: SampleNoArgs` -- is a placeholder variable with the arguments to F and J
    /// * `gen_mass_matrix: fn(one_based: bool) -> CooMatrix` -- is a function to generate the mass matrix.
    ///    Note: the mass matrix needs to be allocated externally because a reference to it is required by System.
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn amplifier<'a>() -> (
        System<
            'a,
            impl FnMut(&mut Vector, f64, &Vector, &mut SampleNoArgs) -> Result<(), StrError>,
            impl FnMut(&mut CooMatrix, f64, &Vector, f64, &mut SampleNoArgs) -> Result<(), StrError>,
            SampleNoArgs,
        >,
        SampleData<'a>,
        SampleNoArgs,
        fn(bool) -> CooMatrix,
    ) {
        // constants
        let (ue, ub, uf, alpha, beta) = (0.1, 6.0, 0.026, 0.99, 1.0e-6);
        let (r0, r1, r2, r3, r4, r5) = (1000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0);
        let (r6, r7, r8, r9) = (9000.0, 9000.0, 9000.0, 9000.0);
        let w = 2.0 * 3.141592654 * 100.0;

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[
            0.0,
            ub,
            ub / (r6 / r5 + 1.0),
            ub / (r6 / r5 + 1.0),
            ub,
            ub / (r2 / r1 + 1.0),
            ub / (r2 / r1 + 1.0),
            0.0,
        ]);

        // ODE system
        let x1 = 0.05;
        let ndim = 8;
        let jac_nnz = 16;
        let system = System::new(
            ndim,
            move |f: &mut Vector, x: f64, y: &Vector, _args: &mut SampleNoArgs| {
                let uet = ue * f64::sin(w * x);
                let fac1 = beta * (f64::exp((y[3] - y[2]) / uf) - 1.0);
                let fac2 = beta * (f64::exp((y[6] - y[5]) / uf) - 1.0);
                f[0] = y[0] / r9;
                f[1] = (y[1] - ub) / r8 + alpha * fac1;
                f[2] = y[2] / r7 - fac1;
                f[3] = y[3] / r5 + (y[3] - ub) / r6 + (1.0 - alpha) * fac1;
                f[4] = (y[4] - ub) / r4 + alpha * fac2;
                f[5] = y[5] / r3 - fac2;
                f[6] = y[6] / r1 + (y[6] - ub) / r2 + (1.0 - alpha) * fac2;
                f[7] = (y[7] - uet) / r0;
                Ok(())
            },
            move |jj: &mut CooMatrix, _x: f64, y: &Vector, m: f64, _args: &mut SampleNoArgs| {
                let fac14 = beta * f64::exp((y[3] - y[2]) / uf) / uf;
                let fac27 = beta * f64::exp((y[6] - y[5]) / uf) / uf;
                jj.reset();
                jj.put(0, 0, (1.0 / r9) * m)?;
                jj.put(1, 1, (1.0 / r8) * m)?;
                jj.put(1, 2, (-alpha * fac14) * m)?;
                jj.put(1, 3, (alpha * fac14) * m)?;
                jj.put(2, 2, (1.0 / r7 + fac14) * m)?;
                jj.put(2, 3, (-fac14) * m)?;
                jj.put(3, 3, (1.0 / r5 + 1.0 / r6 + (1.0 - alpha) * fac14) * m)?;
                jj.put(3, 2, (-(1.0 - alpha) * fac14) * m)?;
                jj.put(4, 4, (1.0 / r4) * m)?;
                jj.put(4, 5, (-alpha * fac27) * m)?;
                jj.put(4, 6, (alpha * fac27) * m)?;
                jj.put(5, 5, (1.0 / r3 + fac27) * m)?;
                jj.put(5, 6, (-fac27) * m)?;
                jj.put(6, 6, (1.0 / r1 + 1.0 / r2 + (1.0 - alpha) * fac27) * m)?;
                jj.put(6, 5, (-(1.0 - alpha) * fac27) * m)?;
                jj.put(7, 7, (1.0 / r0) * m)?;
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            None,
        );

        // function that generates the mass matrix
        let gen_mass_matrix = |one_based: bool| -> CooMatrix {
            let (c1, c2, c3, c4, c5) = (1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6, 5.0e-6);
            let ndim = 8;
            let nnz = 14;
            let mut mass = CooMatrix::new(ndim, ndim, nnz, None, one_based).unwrap();
            mass.put(0, 0, -c5).unwrap();
            mass.put(0, 1, c5).unwrap();
            mass.put(1, 0, c5).unwrap();
            mass.put(1, 1, -c5).unwrap();
            mass.put(2, 2, -c4).unwrap();
            mass.put(3, 3, -c3).unwrap();
            mass.put(3, 4, c3).unwrap();
            mass.put(4, 3, c3).unwrap();
            mass.put(4, 4, -c3).unwrap();
            mass.put(5, 5, -c2).unwrap();
            mass.put(6, 6, -c1).unwrap();
            mass.put(6, 7, c1).unwrap();
            mass.put(7, 6, c1).unwrap();
            mass.put(7, 7, -c1).unwrap();
            mass
        };

        // results
        let data = SampleData {
            x0,
            y0,
            x1,
            h_equal: None,
            y_analytical: None,
        };
        (system, data, 0, gen_mass_matrix)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{SampleNoArgs, Samples};
    use crate::StrError;
    use russell_lab::{deriv_central5, mat_approx_eq, vec_approx_eq, Matrix, Vector};
    use russell_sparse::{CooMatrix, Symmetry};

    fn numerical_jacobian<F>(ndim: usize, x0: f64, y0: Vector, mut function: F, multiplier: f64) -> Matrix
    where
        F: FnMut(&mut Vector, f64, &Vector, &mut SampleNoArgs) -> Result<(), StrError>,
    {
        struct Extra {
            x: f64,
            f: Vector,
            y: Vector,
            i: usize, // index i of ∂fᵢ/∂yⱼ
            j: usize, // index j of ∂fᵢ/∂yⱼ
        }
        let mut extra = Extra {
            x: x0,
            f: Vector::new(ndim),
            y: y0.clone(),
            i: 0,
            j: 0,
        };
        let mut jac = Matrix::new(ndim, ndim);
        let mut args: u8 = 0;
        for i in 0..ndim {
            extra.i = i;
            for j in 0..ndim {
                extra.j = j;
                let at_yj = y0[j];
                let res = deriv_central5(at_yj, &mut extra, |yj: f64, extra: &mut Extra| {
                    let original = extra.y[extra.j];
                    extra.y[extra.j] = yj;
                    function(&mut extra.f, extra.x, &extra.y, &mut args).unwrap();
                    extra.y[extra.j] = original;
                    extra.f[extra.i]
                });
                jac.set(i, j, res * multiplier);
            }
        }
        jac
    }

    #[test]
    fn single_equation_works() {
        let multiplier = 2.0;
        let (mut system, mut data, mut args) = Samples::kreyszig_eq6_page902();

        // check initial values
        if let Some(y_ana) = data.y_analytical.as_mut() {
            let mut y = Vector::new(data.y0.dim());
            y_ana(&mut y, data.x0);
            println!("y0 = {:?} = {:?}", y.as_data(), data.y0.as_data());
            vec_approx_eq(y.as_data(), data.y0.as_data(), 1e-15);
        }

        // compute the analytical Jacobian matrix
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn simple_system_works() {
        let multiplier = 2.0;
        let (mut system, mut data, mut args) = Samples::simple_system();

        // check initial values
        if let Some(y_ana) = data.y_analytical.as_mut() {
            let mut y = Vector::new(data.y0.dim());
            y_ana(&mut y, data.x0);
            println!("y0 = {:?} = {:?}", y.as_data(), data.y0.as_data());
            vec_approx_eq(y.as_data(), data.y0.as_data(), 1e-15);
        }

        // compute the analytical Jacobian matrix
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn hairer_wanner_eq1_works() {
        let multiplier = 2.0;
        let (mut system, mut data, mut args) = Samples::hairer_wanner_eq1();

        // check initial values
        if let Some(y_ana) = data.y_analytical.as_mut() {
            let mut y = Vector::new(data.y0.dim());
            y_ana(&mut y, data.x0);
            println!("y0 = {:?} = {:?}", y.as_data(), data.y0.as_data());
            vec_approx_eq(y.as_data(), data.y0.as_data(), 1e-15);
        }

        // compute the analytical Jacobian matrix
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn robertson_works() {
        let multiplier = 2.0;
        let (mut system, data, mut args) = Samples::robertson();

        // compute the analytical Jacobian matrix
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-14);
    }

    #[test]
    fn van_der_pol_works() {
        let multiplier = 2.0;
        let (mut system, data, mut args) = Samples::van_der_pol(None, false);

        // compute the analytical Jacobian matrix
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1.5e-6);
    }

    #[test]
    fn van_der_pol_works_stationary() {
        let multiplier = 3.0;
        let (mut system, data, mut args) = Samples::van_der_pol(None, true);

        // compute the analytical Jacobian matrix
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn arenstorf_works() {
        let multiplier = 1.5;
        let (mut system, data, mut args) = Samples::arenstorf();

        // compute the analytical Jacobian matrix
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1.6e-4);
    }

    #[test]
    fn amplifier_works() {
        let multiplier = 2.0;
        let (mut system, data, mut args, gen_mass_matrix) = Samples::amplifier();

        // compute the analytical Jacobian matrix
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        // println!("{}", ana);
        // println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-13);

        // generate the mass matrix
        let mass = gen_mass_matrix(false);
        println!("{}", mass.as_dense());
        let ndim = system.ndim;
        let nnz_mass = ndim + 6;
        assert_eq!(mass.get_info(), (ndim, ndim, nnz_mass, Symmetry::No));
    }
}
