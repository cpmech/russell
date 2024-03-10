use crate::StrError;
use crate::{HasJacobian, NoArgs, System};
use russell_lab::math::PI;
use russell_lab::Vector;
use russell_sparse::{CooMatrix, Genie, Sym};

/// Holds data corresponding to a sample ODE problem
pub struct SampleData {
    /// Holds the initial x
    pub x0: f64,

    /// Holds the initial y
    pub y0: Vector,

    /// Holds the final x
    pub x1: f64,

    /// Holds the stepsize for simulations with equal-steps
    pub h_equal: Option<f64>,

    /// Holds a function to compute the analytical solution y(x)
    pub y_analytical: Option<fn(&mut Vector, f64)>,
}

/// Holds a collection of sample ODE problems
///
/// **Note:** Click on the *source* link in the documentation to access the
/// source code illustrating the allocation of System.
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
    /// Implements a simple ODE with a single equation and constant derivative
    ///
    /// ```text
    /// dy
    /// —— = 1   with   y(x=0)=0    thus   y(x) = x
    /// dx
    /// ```
    ///
    /// # Output
    ///
    /// Returns `(system, data, args)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `NoArgs`
    /// * `data: SampleData` -- holds the initial values and the analytical solution
    /// * `args: NoArgs` -- is a placeholder variable with the arguments to F and J
    pub fn simple_equation_constant() -> (
        System<
            impl Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl Fn(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleData,
        NoArgs,
    ) {
        let ndim = 1;
        let jac_nnz = 1; // CooMatrix requires at least one value (thus the 0.0 must be stored)
        let system = System::new(
            ndim,
            |f: &mut Vector, _x: f64, _y: &Vector, _args: &mut NoArgs| {
                f[0] = 1.0;
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, _y: &Vector, multiplier: f64, _args: &mut NoArgs| {
                jj.reset();
                jj.put(0, 0, 0.0 * multiplier).unwrap();
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
            y_analytical: Some(|y, x| {
                y[0] = x;
            }),
        };
        (system, data, 0)
    }

    /// Returns a simple system with a mass matrix
    ///
    /// The system is:
    ///
    /// ```text
    /// y0' + y1'     = -y0 + y1
    /// y0' - y1'     =  y0 + y1
    ///           y2' = 1/(1 + x)
    ///
    /// y0(0) = 1,  y1(0) = 0,  y2(0) = 0
    /// ```
    ///
    /// Thus:
    ///
    /// ```text
    /// M y' = f(x, y)
    /// ```
    ///
    /// with:
    ///
    /// ```text
    ///     ┌          ┐       ┌           ┐
    ///     │  1  1  0 │       │ -y0 + y1  │
    /// M = │  1 -1  0 │   f = │  y0 + y1  │
    ///     │  0  0  1 │       │ 1/(1 + x) │
    ///     └          ┘       └           ┘
    /// ```
    ///
    /// The Jacobian matrix is:
    ///
    /// ```text
    ///          ┌          ┐
    ///     df   │ -1  1  0 │
    /// J = —— = │  1  1  0 │
    ///     dy   │  0  0  0 │
    ///          └          ┘
    /// ```
    ///
    /// The analytical solution is:
    ///
    /// ```text
    /// y0(x) = cos(x)
    /// y1(x) = -sin(x)
    /// y2(x) = log(1 + x)
    /// ```
    ///
    /// # Input
    ///
    /// * `symmetric` -- considers the symmetry of the Jacobian and Mass matrices
    /// * `genie` -- if symmetric, this information is required to decide on the lower-triangle/full
    ///   representation which is consistent with the linear solver employed
    ///
    /// # Output
    ///
    /// Returns `(system, data, args)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `NoArgs`
    /// * `data: SampleData` -- holds the initial values
    /// * `args: NoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Mathematica, Numerical Solution of Differential-Algebraic Equations: Solving Systems with a Mass Matrix
    /// <https://reference.wolfram.com/language/tutorial/NDSolveDAE.html>
    pub fn simple_system_with_mass_matrix(
        symmetric: bool,
        genie: Genie,
    ) -> (
        System<
            impl Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl Fn(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleData,
        NoArgs,
    ) {
        // selected symmetric option (for both Mass and Jacobian matrices)
        let sym = genie.symmetry(symmetric);
        let triangular = sym.triangular();

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[1.0, 0.0, 0.0]);
        let x1 = 20.0;

        // ODE system
        let ndim = 3;
        let jac_nnz = if triangular { 3 } else { 4 };
        let mut system = System::new(
            ndim,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
                f[0] = -y[0] + y[1];
                f[1] = y[0] + y[1];
                f[2] = 1.0 / (1.0 + x);
                Ok(())
            },
            move |jj: &mut CooMatrix, _x: f64, _y: &Vector, m: f64, _args: &mut NoArgs| {
                jj.reset();
                jj.put(0, 0, m * (-1.0)).unwrap();
                if !triangular {
                    jj.put(0, 1, m * (1.0)).unwrap();
                }
                jj.put(1, 0, m * (1.0)).unwrap();
                jj.put(1, 1, m * (1.0)).unwrap();
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            if sym == Sym::No { None } else { Some(sym) },
        );

        // mass matrix
        let mass_nnz = if triangular { 4 } else { 5 };
        system.init_mass_matrix(mass_nnz).unwrap();
        system.mass_put(0, 0, 1.0).unwrap();
        if !triangular {
            system.mass_put(0, 1, 1.0).unwrap();
        }
        system.mass_put(1, 0, 1.0).unwrap();
        system.mass_put(1, 1, -1.0).unwrap();
        system.mass_put(2, 2, 1.0).unwrap();

        // control
        let data = SampleData {
            x0,
            y0,
            x1,
            h_equal: None,
            y_analytical: Some(|y, x| {
                y[0] = f64::cos(x);
                y[1] = -f64::sin(x);
                y[2] = f64::ln(1.0 + x);
            }),
        };
        (system, data, 0)
    }

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
    ///     * `A` -- is `NoArgs`
    /// * `data: SampleData` -- holds the initial values and the analytical solution
    /// * `args: NoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Kreyszig, E (2011) Advanced engineering mathematics; in collaboration with Kreyszig H,
    ///    Edward JN 10th ed 2011, Hoboken, New Jersey, Wiley
    pub fn kreyszig_eq6_page902() -> (
        System<
            impl Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl Fn(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleData,
        NoArgs,
    ) {
        let ndim = 1;
        let jac_nnz = 1;
        let system = System::new(
            ndim,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
                f[0] = x + y[0];
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, _y: &Vector, multiplier: f64, _args: &mut NoArgs| {
                jj.reset();
                jj.put(0, 0, 1.0 * multiplier).unwrap();
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
            y_analytical: Some(|y, x| {
                y[0] = f64::exp(x) - x - 1.0;
            }),
        };
        (system, data, 0)
    }

    /// Implements Example 4 from Kreyszig's book on page 920
    ///
    /// With proper initial conditions, this problem becomes "stiff".
    ///
    /// ```text
    /// y'' + 11 y' + 10 y = 10 x + 11
    /// y(0) = 2
    /// y'(0) = -10
    /// ```
    ///
    /// Converting into a system:
    ///
    /// ```text
    /// y = y1 and y' = y2
    /// y0' = y1
    /// y1' = -10 y0 - 11 y1 + 10 x + 11
    /// ```
    ///
    /// # Output
    ///
    /// Returns `(system, data, args)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `NoArgs`
    /// * `data: SampleData` -- holds the initial values and the analytical solution
    /// * `args: NoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Kreyszig, E (2011) Advanced engineering mathematics; in collaboration with Kreyszig H,
    ///    Edward JN 10th ed 2011, Hoboken, New Jersey, Wiley
    pub fn kreyszig_ex4_page920() -> (
        System<
            impl Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl Fn(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleData,
        NoArgs,
    ) {
        let ndim = 2;
        let jac_nnz = 3;
        let system = System::new(
            ndim,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
                f[0] = y[1];
                f[1] = -10.0 * y[0] - 11.0 * y[1] + 10.0 * x + 11.0;
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, _y: &Vector, multiplier: f64, _args: &mut NoArgs| {
                jj.reset();
                jj.put(0, 1, 1.0 * multiplier).unwrap();
                jj.put(1, 0, -10.0 * multiplier).unwrap();
                jj.put(1, 1, -11.0 * multiplier).unwrap();
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            None,
        );
        let data = SampleData {
            x0: 0.0,
            y0: Vector::from(&[2.0, -10.0]),
            x1: 1.0,
            h_equal: Some(0.2),
            y_analytical: Some(|y, x| {
                y[0] = f64::exp(-x) + f64::exp(-10.0 * x) + x;
                y[1] = -f64::exp(-x) - 10.0 * f64::exp(-10.0 * x) + 1.0;
            }),
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
    ///     * `A` -- is `NoArgs`
    /// * `data: SampleData` -- holds the initial values and the analytical solution
    /// * `args: NoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn hairer_wanner_eq1() -> (
        System<
            impl Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl Fn(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleData,
        NoArgs,
    ) {
        const L: f64 = -50.0; // lambda
        let ndim = 1;
        let jac_nnz = 1;
        let system = System::new(
            ndim,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
                f[0] = L * (y[0] - f64::cos(x));
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, _y: &Vector, multiplier: f64, _args: &mut NoArgs| {
                jj.reset();
                jj.put(0, 0, multiplier * L).unwrap();
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
            y_analytical: Some(|y, x| {
                y[0] = -L * (f64::sin(x) - L * f64::cos(x) + L * f64::exp(L * x)) / (L * L + 1.0);
            }),
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
    ///     * `A` -- is `NoArgs`
    /// * `data: SampleData` -- holds the initial values
    /// * `args: NoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn robertson() -> (
        System<
            impl Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl Fn(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleData,
        NoArgs,
    ) {
        let ndim = 3;
        let jac_nnz = 7;
        let system = System::new(
            ndim,
            |f: &mut Vector, _x: f64, y: &Vector, _args: &mut NoArgs| {
                f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
                f[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
                f[2] = 3.0e7 * y[1] * y[1];
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, y: &Vector, multiplier: f64, _args: &mut NoArgs| {
                jj.reset();
                jj.put(0, 0, -0.04 * multiplier).unwrap();
                jj.put(0, 1, 1.0e4 * y[2] * multiplier).unwrap();
                jj.put(0, 2, 1.0e4 * y[1] * multiplier).unwrap();
                jj.put(1, 0, 0.04 * multiplier).unwrap();
                jj.put(1, 1, (-1.0e4 * y[2] - 6.0e7 * y[1]) * multiplier).unwrap();
                jj.put(1, 2, (-1.0e4 * y[1]) * multiplier).unwrap();
                jj.put(2, 1, 6.0e7 * y[1] * multiplier).unwrap();
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
    ///     * `A` -- is `NoArgs`
    /// * `data: SampleData` -- holds the initial values
    /// * `args: NoArgs` -- is a placeholder variable with the arguments to F and J
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
    pub fn van_der_pol(
        epsilon: Option<f64>,
        stationary: bool,
    ) -> (
        System<
            impl Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl Fn(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleData,
        NoArgs,
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
            move |f: &mut Vector, _x: f64, y: &Vector, _args: &mut NoArgs| {
                f[0] = y[1];
                f[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / eps;
                Ok(())
            },
            move |jj: &mut CooMatrix, _x: f64, y: &Vector, multiplier: f64, _args: &mut NoArgs| {
                jj.reset();
                jj.put(0, 1, 1.0 * multiplier).unwrap();
                jj.put(1, 0, multiplier * (-2.0 * y[0] * y[1] - 1.0) / eps).unwrap();
                jj.put(1, 1, multiplier * (1.0 - y[0] * y[0]) / eps).unwrap();
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
    ///     * `A` -- is `NoArgs`
    /// * `data: SampleData` -- holds the initial values
    /// * `args: NoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn arenstorf() -> (
        System<
            impl Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl Fn(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleData,
        NoArgs,
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
            |f: &mut Vector, _x: f64, y: &Vector, _args: &mut NoArgs| {
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
            |jj: &mut CooMatrix, _x: f64, y: &Vector, m: f64, _args: &mut NoArgs| {
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
                jj.put(0, 2, 1.0 * m).unwrap();
                jj.put(1, 3, 1.0 * m).unwrap();
                jj.put(2, 0, (1.0 + a * dj00 * MD / dd0 + b * dj10 * MU / dd1 + c) * m)
                    .unwrap();
                jj.put(2, 1, (a * dj01 * MD / dd0 + b * dj11 * MU / dd1) * m).unwrap();
                jj.put(2, 3, 2.0 * m).unwrap();
                jj.put(3, 0, (dj00 * y[1] * MD / dd0 + dj10 * y[1] * MU / dd1) * m)
                    .unwrap();
                jj.put(3, 1, (1.0 + dj01 * y[1] * MD / dd0 + dj11 * y[1] * MU / dd1 + c) * m)
                    .unwrap();
                jj.put(3, 2, -2.0 * m).unwrap();
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

    /// Returns the one-transistor amplifier problem described by Hairer-Wanner, Part II, page 376
    ///
    /// # Output
    ///
    /// Returns `(system, data, args)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `NoArgs`
    /// * `data: SampleData` -- holds the initial values
    /// * `args: NoArgs` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn amplifier1t() -> (
        System<
            impl Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl Fn(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleData,
        NoArgs,
    ) {
        // constants
        const ALPHA: f64 = 0.99;
        const GAMMA: f64 = 1.0 - ALPHA;
        const C: f64 = 0.4;
        const D: f64 = 200.0 * PI;
        const BETA: f64 = 1e-6;
        const UB: f64 = 6.0;
        const UF: f64 = 0.026;
        const R: f64 = 1000.0;
        const S: f64 = 9000.0;

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[0.0, UB / 2.0, UB / 2.0, UB, 0.0]);
        let x1 = 0.05;

        // ODE system
        let ndim = 5;
        let jac_nnz = 9;
        let mut system = System::new(
            ndim,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
                let ue = C * f64::sin(D * x);
                let f12 = BETA * (f64::exp((y[1] - y[2]) / UF) - 1.0);
                f[0] = (y[0] - ue) / R;
                f[1] = (2.0 * y[1] - UB) / S + GAMMA * f12;
                f[2] = y[2] / S - f12;
                f[3] = (y[3] - UB) / S + ALPHA * f12;
                f[4] = y[4] / S;
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, y: &Vector, m: f64, _args: &mut NoArgs| {
                let g12 = BETA * f64::exp((y[1] - y[2]) / UF) / UF;
                jj.reset();
                jj.put(0, 0, m * (1.0 / R)).unwrap();
                jj.put(1, 1, m * (2.0 / S + GAMMA * g12)).unwrap();
                jj.put(1, 2, m * (-GAMMA * g12)).unwrap();
                jj.put(2, 1, m * (-g12)).unwrap();
                jj.put(2, 2, m * (1.0 / S + g12)).unwrap();
                jj.put(3, 1, m * (ALPHA * g12)).unwrap();
                jj.put(3, 2, m * (-ALPHA * g12)).unwrap();
                jj.put(3, 3, m * (1.0 / S)).unwrap();
                jj.put(4, 4, m * (1.0 / S)).unwrap();
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            None,
        );

        // function that generates the mass matrix
        const C1: f64 = 1e-6;
        const C2: f64 = 2e-6;
        const C3: f64 = 3e-6;
        let mass_nnz = 9;
        system.init_mass_matrix(mass_nnz).unwrap();
        system.mass_put(0, 0, -C1).unwrap();
        system.mass_put(0, 1, C1).unwrap();
        system.mass_put(1, 0, C1).unwrap();
        system.mass_put(1, 1, -C1).unwrap();
        system.mass_put(2, 2, -C2).unwrap();
        system.mass_put(3, 3, -C3).unwrap();
        system.mass_put(3, 4, C3).unwrap();
        system.mass_put(4, 3, C3).unwrap();
        system.mass_put(4, 4, -C3).unwrap();

        // control
        let data = SampleData {
            x0,
            y0,
            x1,
            h_equal: None,
            y_analytical: None,
        };
        (system, data, 0)
    }

    /// Returns the Brusselator problem (ODE version) described in Hairer-Nørsett-Wanner, Part I, page 116
    ///
    /// # Output
    ///
    /// Returns `(system, data, args, y_ref)` where:
    ///
    /// * `system: System<F, J, A>` with:
    ///     * `F` -- is a function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    ///     * `J` -- is a function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    ///     * `A` -- is `NoArgs`
    /// * `data: SampleData` -- holds the initial values
    /// * `args: NoArgs` -- is a placeholder variable with the arguments to F and J
    /// * `y_ref` -- is a reference solution, computed with high-accuracy by Mathematica
    ///
    /// # Reference
    ///
    /// * Hairer E, Nørsett, SP, Wanner G (2008) Solving Ordinary Differential Equations I.
    ///   Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
    ///   in Computational Mathematics, 528p
    pub fn brusselator_ode() -> (
        System<
            impl Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
            impl Fn(&mut CooMatrix, f64, &Vector, f64, &mut NoArgs) -> Result<(), StrError>,
            NoArgs,
        >,
        SampleData,
        NoArgs,
        Vector,
    ) {
        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[3.0 / 2.0, 3.0]);
        let x1 = 20.0;

        // ODE system
        let ndim = 2;
        let jac_nnz = 4;
        let system = System::new(
            ndim,
            |f: &mut Vector, _x: f64, y: &Vector, _args: &mut NoArgs| {
                f[0] = 1.0 - 4.0 * y[0] + y[0] * y[0] * y[1];
                f[1] = 3.0 * y[0] - y[0] * y[0] * y[1];
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, y: &Vector, m: f64, _args: &mut NoArgs| {
                jj.reset();
                jj.put(0, 0, m * (-4.0 + 2.0 * y[0] * y[1])).unwrap();
                jj.put(0, 1, m * (y[0] * y[0])).unwrap();
                jj.put(1, 0, m * (3.0 - 2.0 * y[0] * y[1])).unwrap();
                jj.put(1, 1, m * (-y[0] * y[0])).unwrap();
                Ok(())
            },
            HasJacobian::Yes,
            Some(jac_nnz),
            None,
        );

        // control
        let data = SampleData {
            x0,
            y0,
            x1,
            h_equal: Some(0.1),
            y_analytical: None,
        };

        // reference solution; using the following Mathematica code:
        // ```Mathematica
        // Needs["DifferentialEquations`NDSolveProblems`"];
        // Needs["DifferentialEquations`NDSolveUtilities`"];
        // sys = GetNDSolveProblem["BrusselatorODE"];
        // sol = NDSolve[sys, Method -> "StiffnessSwitching", WorkingPrecision -> 32];
        // ref = First[FinalSolutions[sys, sol]]
        // ```
        let y_ref = Vector::from(&[0.4986370712683478291402659846476, 4.596780349452011024598321237263]);
        (system, data, 0, y_ref)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{NoArgs, Samples};
    use crate::StrError;
    use russell_lab::{deriv_central5, mat_approx_eq, vec_approx_eq, Matrix, Vector};
    use russell_sparse::{CooMatrix, Sym};

    fn numerical_jacobian<F>(ndim: usize, x0: f64, y0: Vector, function: F, multiplier: f64) -> Matrix
    where
        F: Fn(&mut Vector, f64, &Vector, &mut NoArgs) -> Result<(), StrError>,
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
    fn simple_constant_works() {
        let multiplier = 2.0;
        let (system, mut data, mut args) = Samples::simple_equation_constant();

        // check initial values
        if let Some(y_ana) = data.y_analytical.as_mut() {
            let mut y = Vector::new(data.y0.dim());
            y_ana(&mut y, data.x0);
            println!("y0 = {:?} = {:?}", y.as_data(), data.y0.as_data());
            vec_approx_eq(y.as_data(), data.y0.as_data(), 1e-15);
        }

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.jac_sym).unwrap();
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
    fn kreyszig_eq6_page902_works() {
        let multiplier = 2.0;
        let (system, mut data, mut args) = Samples::kreyszig_eq6_page902();

        // check initial values
        if let Some(y_ana) = data.y_analytical.as_mut() {
            let mut y = Vector::new(data.y0.dim());
            y_ana(&mut y, data.x0);
            println!("y0 = {:?} = {:?}", y.as_data(), data.y0.as_data());
            vec_approx_eq(y.as_data(), data.y0.as_data(), 1e-15);
        }

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.jac_sym).unwrap();
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
    fn kreyszig_ex4_page920() {
        let multiplier = 2.0;
        let (system, mut data, mut args) = Samples::kreyszig_ex4_page920();

        // check initial values
        if let Some(y_ana) = data.y_analytical.as_mut() {
            let mut y = Vector::new(data.y0.dim());
            y_ana(&mut y, data.x0);
            println!("y0 = {:?} = {:?}", y.as_data(), data.y0.as_data());
            vec_approx_eq(y.as_data(), data.y0.as_data(), 1e-15);
        }

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.jac_sym).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-10);
    }

    #[test]
    fn hairer_wanner_eq1_works() {
        let multiplier = 2.0;
        let (system, mut data, mut args) = Samples::hairer_wanner_eq1();

        // check initial values
        if let Some(y_ana) = data.y_analytical.as_mut() {
            let mut y = Vector::new(data.y0.dim());
            y_ana(&mut y, data.x0);
            println!("y0 = {:?} = {:?}", y.as_data(), data.y0.as_data());
            vec_approx_eq(y.as_data(), data.y0.as_data(), 1e-15);
        }

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.jac_sym).unwrap();
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
        let (system, data, mut args) = Samples::robertson();

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.jac_sym).unwrap();
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
        let (system, data, mut args) = Samples::van_der_pol(None, false);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.jac_sym).unwrap();
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
        let (system, data, mut args) = Samples::van_der_pol(None, true);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.jac_sym).unwrap();
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
        let (system, data, mut args) = Samples::arenstorf();

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.jac_sym).unwrap();
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
    fn amplifier1t_works() {
        let multiplier = 2.0;
        let (system, data, mut args) = Samples::amplifier1t();

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.jac_sym).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{:.15}", ana);
        println!("{:.15}", num);
        mat_approx_eq(&ana, &num, 1e-13);

        // check the mass matrix
        let mass = system.mass_matrix.unwrap();
        println!("{}", mass.as_dense());
        let ndim = system.ndim;
        let nnz_mass = 5 + 4;
        assert_eq!(mass.get_info(), (ndim, ndim, nnz_mass, Sym::No));
    }

    #[test]
    fn brusselator_ode_works() {
        let multiplier = 2.0;
        let (system, data, mut args, _) = Samples::brusselator_ode();

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.jac_sym).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function, multiplier);

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{:.15}", ana);
        println!("{:.15}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }
}
