use crate::{NoArgs, PdeDiscreteLaplacian2d, Side, System};
use russell_lab::math::PI;
use russell_lab::Vector;
use russell_sparse::{CooMatrix, Genie, Sym};

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
    /// Returns a simple ODE with a constant derivative
    ///
    /// Returns `(system, x0, y0, args, y_fn_x)`
    ///
    /// Problem:
    ///
    /// ```text
    ///      dy
    /// y' = —— = 1   with   y(x=0)=0    thus   y(x) = x
    ///      dx
    /// ```
    ///
    /// # Output
    ///
    /// * `system` -- the ODE system
    /// * `x0` -- initial x
    /// * `y0` -- initial y
    /// * `args` -- is a placeholder variable with the arguments to F and J
    /// * `y_fn_x` -- is a function to compute the analytical solution
    pub fn simple_equation_constant<'a>() -> (
        System<'a, NoArgs>,
        f64,
        Vector,
        NoArgs,
        impl Fn(&mut Vector, f64, &mut NoArgs),
    ) {
        // system
        let ndim = 1;
        let mut system = System::new(ndim, |f: &mut Vector, _x: f64, _y: &Vector, _args: &mut NoArgs| {
            f[0] = 1.0;
            Ok(())
        });

        // function to compute the Jacobian matrix
        let jac_nnz = 1; // CooMatrix requires at least one value (thus the 0.0 must be stored)
        system
            .set_jacobian(
                Some(jac_nnz),
                Sym::No,
                |jj: &mut CooMatrix, alpha: f64, _x: f64, _y: &Vector, _args: &mut NoArgs| {
                    jj.reset();
                    jj.put(0, 0, 0.0 * alpha).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[0.0]);

        // analytical solution
        let y_fn_x = |y: &mut Vector, x: f64, _args: &mut NoArgs| {
            y[0] = x;
        };

        // results
        let args = 0;
        (system, x0, y0, args, y_fn_x)
    }

    /// Returns a simple system with a mass matrix
    ///
    /// Returns `(system, x0, y0, args, y_fn_x)`
    ///
    /// Problem:
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
    /// * `symmetric` -- specifies the symmetric type of the Jacobian and Mass matrices
    /// * `genie` -- if symmetric, this information is required to decide on the lower-triangle/full
    ///   representation which is consistent with the linear solver employed
    ///
    /// # Output
    ///
    /// * `system` -- the ODE system
    /// * `x0` -- initial x
    /// * `y0` -- initial y
    /// * `args` -- is a placeholder variable with the arguments to F and J
    /// * `y_fn_x` -- is a function to compute the analytical solution
    ///
    /// # Reference
    ///
    /// * Mathematica, Numerical Solution of Differential-Algebraic Equations: Solving Systems with a Mass Matrix
    /// <https://reference.wolfram.com/language/tutorial/NDSolveDAE.html>
    pub fn simple_system_with_mass_matrix<'a>(
        symmetric: bool,
        genie: Genie,
    ) -> (
        System<'a, NoArgs>,
        f64,
        Vector,
        NoArgs,
        impl Fn(&mut Vector, f64, &mut NoArgs),
    ) {
        // selected symmetric type (for both Mass and Jacobian matrices)
        let sym = genie.get_sym(symmetric);
        let triangular = sym.triangular();

        // system
        let ndim = 3;
        let mut system = System::new(ndim, |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = -y[0] + y[1];
            f[1] = y[0] + y[1];
            f[2] = 1.0 / (1.0 + x);
            Ok(())
        });

        // function to compute the Jacobian matrix
        let jac_nnz = if triangular { 3 } else { 4 };
        system
            .set_jacobian(
                Some(jac_nnz),
                sym,
                move |jj: &mut CooMatrix, alpha: f64, _x: f64, _y: &Vector, _args: &mut NoArgs| {
                    jj.reset();
                    jj.put(0, 0, alpha * (-1.0)).unwrap();
                    if !triangular {
                        jj.put(0, 1, alpha * (1.0)).unwrap();
                    }
                    jj.put(1, 0, alpha * (1.0)).unwrap();
                    jj.put(1, 1, alpha * (1.0)).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // mass matrix
        let mass_nnz = if triangular { 4 } else { 5 };
        system
            .set_mass(Some(mass_nnz), sym, move |mm: &mut CooMatrix| {
                mm.put(0, 0, 1.0).unwrap();
                if !triangular {
                    mm.put(0, 1, 1.0).unwrap();
                }
                mm.put(1, 0, 1.0).unwrap();
                mm.put(1, 1, -1.0).unwrap();
                mm.put(2, 2, 1.0).unwrap();
            })
            .unwrap();

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[1.0, 0.0, 0.0]);

        // analytical solution
        let y_fn_x = |y: &mut Vector, x: f64, _args: &mut NoArgs| {
            y[0] = f64::cos(x);
            y[1] = -f64::sin(x);
            y[2] = f64::ln(1.0 + x);
        };

        // results
        let args = 0;
        (system, x0, y0, args, y_fn_x)
    }

    /// Returns the Brusselator problem (ODE version)
    ///
    /// Returns `(system, x0, y0, args, y_ref)`
    ///
    /// This example corresponds to Fig 16.4 on page 116 of the reference.
    /// The problem is defined in Eq (16.12) on page 116 of the reference.
    ///
    /// The system is:
    ///
    /// ```text
    /// y0' = 1 - 4 y0 + y0² y1
    /// y1' = 3 y0 - y0² y1
    ///
    /// with  y0(x=0) = 3/2  and  y1(x=0) = 3
    /// ```
    ///
    /// The Jacobian matrix is:
    ///
    /// ```text
    ///          ┌                     ┐
    ///     df   │ -4 + 2 y0 y1    y0² │
    /// J = —— = │                     │
    ///     dy   │  3 - 2 y0 y1   -y0² │
    ///          └                     ┘
    /// ```
    ///
    /// # Output
    ///
    /// * `system` -- the ODE system
    /// * `x0` -- initial x
    /// * `y0` -- initial y
    /// * `args` -- is a placeholder variable with the arguments to F and J
    /// * `y_ref` -- is a reference solution, computed with high-accuracy by Mathematica
    ///
    /// # Reference
    ///
    /// * Hairer E, Nørsett, SP, Wanner G (2008) Solving Ordinary Differential Equations I.
    ///   Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
    ///   in Computational Mathematics, 528p
    pub fn brusselator_ode<'a>() -> (System<'a, NoArgs>, f64, Vector, NoArgs, Vector) {
        // system
        let ndim = 2;
        let jac_nnz = 4;
        let mut system = System::new(ndim, |f: &mut Vector, _x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = 1.0 - 4.0 * y[0] + y[0] * y[0] * y[1];
            f[1] = 3.0 * y[0] - y[0] * y[0] * y[1];
            Ok(())
        });

        // function to compute the Jacobian matrix
        system
            .set_jacobian(
                Some(jac_nnz),
                Sym::No,
                |jj: &mut CooMatrix, alpha: f64, _x: f64, y: &Vector, _args: &mut NoArgs| {
                    jj.reset();
                    jj.put(0, 0, alpha * (-4.0 + 2.0 * y[0] * y[1])).unwrap();
                    jj.put(0, 1, alpha * (y[0] * y[0])).unwrap();
                    jj.put(1, 0, alpha * (3.0 - 2.0 * y[0] * y[1])).unwrap();
                    jj.put(1, 1, alpha * (-y[0] * y[0])).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[3.0 / 2.0, 3.0]);

        // reference solution; using the following Mathematica code:
        // ```Mathematica
        // Needs["DifferentialEquations`NDSolveProblems`"];
        // Needs["DifferentialEquations`NDSolveUtilities`"];
        // sys = GetNDSolveProblem["BrusselatorODE"];
        // sol = NDSolve[sys, Method -> "StiffnessSwitching", WorkingPrecision -> 32];
        // ref = First[FinalSolutions[sys, sol]]
        // ```
        let y_ref = Vector::from(&[0.4986370712683478291402659846476, 4.596780349452011024598321237263]);

        // results
        let args = 0;
        (system, x0, y0, args, y_ref)
    }

    /// Returns the Brusselator reaction-diffusion problem in 2D (parabolic PDE)
    ///
    /// Returns `(system, t0, yy0, args)`
    ///
    /// This example corresponds to Fig 10.4(a,b) on pages 250 and 251 of Reference #1.
    /// The problem is defined in Eqs (10.10-10.14) on pages 248 and 249 of Reference #1.
    ///
    /// If `second_book` is true, this example corresponds to Fig 10.7 on page 151 of Reference #2.
    /// Also, in this case, the problem is defined in Eqs (10.15-10.16) on pages 151 and 152 of Reference #2.
    ///
    /// While in the first book the boundary conditions are Neumann-type, in the second book the
    /// boundary conditions are periodic. Also the initial values in the second book are different
    /// that those in the first book.
    ///
    /// The model is given by:
    ///
    /// ```text
    /// ∂u                         ⎛ ∂²u   ∂²u ⎞
    /// ——— = 1 - 4.4 u + u² v + α ⎜ ——— + ——— ⎟ + I(t,x,y)
    /// ∂t                         ⎝ ∂x²   ∂y² ⎠
    ///
    /// ∂v                         ⎛ ∂²v   ∂²v ⎞
    /// ——— =     3.4 u - u² v + α ⎜ ——— + ——— ⎟
    /// ∂t                         ⎝ ∂x²   ∂y² ⎠
    ///
    /// with:  t ≥ 0,  0 ≤ x ≤ 1,  0 ≤ y ≤ 1
    /// ```
    ///
    /// where `I(t,x,y)` is the inhomogeneity function (second book) given by:
    ///
    /// ```text
    ///             ⎧ 5  if (x-0.3)²+(y-0.6)² ≤ 0.1² and t ≥ 1.1
    /// I(t,x,y) =  ⎨
    ///             ⎩ 0  otherwise
    /// ```
    ///
    /// The first book considers the following Neumann boundary conditions:
    ///
    /// ```text
    /// ∂u          ∂v     
    /// ——— = 0     ——— = 0
    ///  →           →
    /// ∂n          ∂n     
    /// ```
    ///
    /// and the following initial conditions (first book):
    ///
    /// ```text
    /// u(t=0,x,y) = 0.5 + y    v(t=0,x,y) = 1 + 5 x
    /// ```
    ///
    /// The second book considers periodic boundary conditions on `u`.
    /// However, here we assume periodic on `u` and `v`:
    ///
    /// ```text
    /// u(t, 0, y) = u(t, 1, y)
    /// u(t, x, 0) = u(t, x, 1)
    /// v(t, 0, y) = v(t, 1, y)   ← Not in the book
    /// v(t, x, 0) = v(t, x, 1)   ← Not in the book
    /// ```
    ///
    /// The second book considers the following initial conditions:
    ///
    /// ```text
    /// u(0, x, y) = 22 y pow(1 - y, 1.5)
    /// v(0, x, y) = 27 x pow(1 - x, 1.5)
    /// ```
    ///
    /// The scalar fields u(t, x, y) and v(t, x, y) are mapped over a rectangular grid with
    /// their discrete counterparts represented by:
    ///
    /// ```text
    /// pᵢⱼ(t) := u(t, xᵢ, yⱼ)
    /// qᵢⱼ(t) := v(t, xᵢ, yⱼ)
    /// ```
    ///
    /// Thus `ndim = 2 npoint²` with npoint being the number of points along the x or y line.
    ///
    /// The second partial derivatives over x and y (Laplacian) are approximated using the
    /// Finite Differences Method (FDM).
    ///
    /// The pᵢⱼ and qᵢⱼ values are mapped onto the vectors `U` and `V` as follows:
    ///
    /// ```text
    /// pᵢⱼ → Uₘ
    /// qᵢⱼ → Vₘ
    ///
    /// with m = i + j nx
    /// ```
    ///
    /// Then, they are stored in a single vector `Y`:
    ///
    /// ```text
    ///     ┌   ┐
    ///     │ U │
    /// Y = │   │
    ///     │ V │
    ///     └   ┘
    /// ```
    ///
    /// Thus:
    ///
    /// ```text
    /// Uₘ = Yₘ  and  Vₘ = Yₛ₊ₘ
    ///
    /// where  0 ≤ m ≤ s - 1  and (shift)  s = npoint²
    /// ```
    ///
    /// In terms of components, we can write:
    ///
    /// ```text
    ///       ⎧ Uₐ    if a < s
    /// Yₐ =  ⎨
    ///       ⎩ Vₐ₋ₛ  if a ≥ s
    ///
    /// where  0 ≤ a ≤ ndim - 1  and  ndim = 2 s
    /// ```
    ///
    /// The components of the resulting system of equations are defined by:
    /// (the prime indicates time-derivative; no summation over repeated indices):
    ///
    /// ```text
    /// Uₘ' = 1 - 4.4 Uₘ + Uₘ² Vₘ + Σ Aₘₖ Uₖ
    ///                            k
    /// Vₘ' =     3.4 Uₘ - Uₘ² Vₘ + Σ Aₘₖ Uₖ
    ///                            k
    ///
    /// where Aₘₖ are the elements of the discrete Laplacian matrix
    /// ```
    ///
    /// The components to build the Jacobian matrix are:
    /// (no summation over repeated indices):
    ///
    /// ```text
    /// ∂Uₘ'
    /// ———— = -4.4 δₘₙ + 2 Uₘ δₘₙ Vₘ + Aₘₙ
    /// ∂Uₙ
    ///
    /// ∂Uₘ'
    /// ———— = Uₘ² δₘₙ
    /// ∂Vₙ
    ///
    /// ∂Vₘ'
    /// ———— = 3.4 δₘₙ - 2 Uₘ δₘₙ Vₘ
    /// ∂Uₙ
    ///
    /// ∂Vₘ'
    /// ———— = -Uₘ² δₘₙ + Aₘₙ
    /// ∂Vₙ
    ///
    /// where δₘₙ is the Kronecker delta
    /// ```
    ///
    /// With `Fₐ := ∂Yₐ/∂t`, the components of the Jacobian matrix can be "assembled" as follows:
    ///
    /// ```text
    ///       ⎧  ⎧ ∂Uₐ'/∂Uₑ      if e < s
    ///       │  ⎨                          if a < s
    /// ∂Fₐ   │  ⎩ ∂Uₐ'/∂Vₑ₋ₛ    if e ≥ s
    /// ——— = ⎨
    /// ∂Yₑ   │  ⎧ ∂Vₐ₋ₛ'/∂Uₑ    if e < s
    ///       │  ⎨                          if a ≥ s
    ///       ⎩  ⎩ ∂Vₐ₋ₛ'/∂Vₑ₋ₛ  if e ≥ s
    ///
    /// where  0 ≤ a ≤ ndim - 1  and  0 ≤ e ≤ ndim - 1
    /// ```
    ///
    /// # Input
    ///
    /// * `alpha` -- the α coefficient
    /// * `npoint` -- the number of points along one direction on the grid
    /// * `second_book` -- implements the model from the second book (Reference #2)
    /// * `ignore_diffusion` -- ignore the diffusion term (convenient for debugging)
    ///
    /// # Output
    ///
    /// * `system` -- the ODE system
    /// * `t0` -- initial t
    /// * `yy0` -- initial `Y0 = [U0, V0]ᵀ`
    /// * `args: PdeDiscreteLaplacian2d` -- the discrete Laplacian
    ///
    /// # References
    ///
    /// 1. Hairer E, Nørsett, SP, Wanner G (2008) Solving Ordinary Differential Equations I.
    ///    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
    ///    in Computational Mathematics, 528p
    /// 2. Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///    Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///    Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn brusselator_pde<'a>(
        alpha: f64,
        npoint: usize,
        second_book: bool,
        ignore_diffusion: bool,
    ) -> (System<'a, PdeDiscreteLaplacian2d>, f64, Vector, PdeDiscreteLaplacian2d) {
        // constants
        let (kx, ky) = (alpha, alpha);
        let (xmin, xmax) = (0.0, 1.0);
        let (ymin, ymax) = (0.0, 1.0);
        let (nx, ny) = (npoint, npoint);
        let s = npoint * npoint;
        let ndim = 2 * s;

        // number of non-zeros in the Jacobian
        let band = 5;
        let jac_nnz = if ignore_diffusion {
            4 * s // 4 diagonal matrices + 2 banded (laplacian) matrices
        } else {
            4 * s + 2 * s * band // 4 diagonal matrices + 2 banded (laplacian) matrices
        };

        // system
        let mut system = System::new(ndim, move |f, t, yy, fdm: &mut PdeDiscreteLaplacian2d| {
            fdm.loop_over_grid_points(|m, x, y| {
                let um = yy[m];
                let vm = yy[s + m];
                let um2 = um * um;
                f[m] = 1.0 - 4.4 * um + um2 * vm;
                f[s + m] = 3.4 * um - um2 * vm;
                if !ignore_diffusion {
                    fdm.loop_over_coef_mat_row(m, |k, amk| {
                        let uk = yy[k];
                        let vk = yy[s + k];
                        f[m] += amk * uk;
                        f[s + m] += amk * vk;
                    });
                }
                if second_book {
                    if t >= 1.1 {
                        let dx = x - 0.3;
                        let dy = y - 0.6;
                        let inhomogeneity = if dx * dx + dy * dy <= 0.01 { 5.0 } else { 0.0 };
                        f[m] += inhomogeneity;
                    }
                }
            });
            Ok(())
        });

        // function to compute the Jacobian matrix
        system
            .set_jacobian(
                Some(jac_nnz),
                Sym::No,
                move |jj, aa, _x, yy, fdm: &mut PdeDiscreteLaplacian2d| {
                    jj.reset();
                    let mut nnz_count = 0;
                    for m in 0..s {
                        let um = yy[m];
                        let vm = yy[s + m];
                        let um2 = um * um;
                        jj.put(m, m, aa * (-4.4 + 2.0 * um * vm)).unwrap();
                        jj.put(m, s + m, aa * (um2)).unwrap();
                        jj.put(s + m, m, aa * (3.4 - 2.0 * um * vm)).unwrap();
                        jj.put(s + m, s + m, aa * (-um2)).unwrap();
                        nnz_count += 4;
                        if !ignore_diffusion {
                            fdm.loop_over_coef_mat_row(m, |n, amn| {
                                jj.put(m, n, aa * (amn)).unwrap();
                                jj.put(s + m, s + n, aa * (amn)).unwrap();
                                nnz_count += 2;
                            });
                        }
                    }
                    assert_eq!(nnz_count, jac_nnz);
                    Ok(())
                },
            )
            .unwrap();

        // discrete laplacian
        let mut fdm = PdeDiscreteLaplacian2d::new(kx, ky, xmin, xmax, ymin, ymax, nx, ny).unwrap();
        if second_book {
            fdm.set_periodic_boundary_condition(Side::Left);
            fdm.set_periodic_boundary_condition(Side::Bottom);
        }

        // initial values
        let t0 = 0.0;
        let mut yy0 = Vector::new(ndim);
        if second_book {
            fdm.loop_over_grid_points(|m, x, y| {
                yy0[m] = 22.0 * y * f64::powf(1.0 - y, 1.5);
                yy0[s + m] = 27.0 * x * f64::powf(1.0 - x, 1.5);
            });
        } else {
            fdm.loop_over_grid_points(|m, x, y| {
                yy0[m] = 0.5 + y; // u0
                yy0[s + m] = 1.0 + 5.0 * x; // v0
            });
        }

        // results
        (system, t0, yy0, fdm)
    }

    /// Returns the Arenstorf orbit problem
    ///
    /// Returns `(system, x0, y0, x1, args, y_ref)`
    ///
    /// This example corresponds to Fig 0.1 on page 130 of the reference.
    /// The problem is defined in Eqs (0.1) and (0.2) on page 129 and 130 of the reference.
    ///
    /// From Hairer-Nørsett-Wanner:
    ///
    /// "(...) an example from Astronomy, the restricted three body problem. (...)
    /// two bodies of masses μ' = 1 − μ and μ in circular rotation in a plane and
    /// a third body of negligible mass moving around in the same plane. (...)"
    ///
    /// The system equations are:
    ///
    /// ```text
    /// y0'' = y0 + 2 y1' - μ' (y0 + μ) / d0 - μ (y0 - μ') / d1
    /// y1'' = y1 - 2 y0' - μ' y1 / d0 - μ y1 / d1
    /// ```
    ///
    /// With the assignments:
    ///
    /// ```text
    /// y2 := y0'  ⇒  y2' = y0''
    /// y3 := y1'  ⇒  y3' = y1''
    /// ```
    ///
    /// We obtain a 4-dim problem:
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
    /// * `system` -- the ODE system
    /// * `x0` -- initial x
    /// * `y0` -- initial y
    /// * `x1` -- final x
    /// * `args` -- is a placeholder variable with the arguments to F and J
    /// * `y_ref` -- is a reference solution, computed with high-accuracy by Mathematica
    ///
    /// # Reference
    ///
    /// * Hairer E, Nørsett, SP, Wanner G (2008) Solving Ordinary Differential Equations I.
    ///   Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
    ///   in Computational Mathematics, 528p
    pub fn arenstorf<'a>() -> (System<'a, NoArgs>, f64, Vector, f64, NoArgs, Vector) {
        // constants
        const MU: f64 = 0.012277471;
        const MD: f64 = 1.0 - MU;

        // system
        let ndim = 4;
        let mut system = System::new(ndim, |f: &mut Vector, _x: f64, y: &Vector, _args: &mut NoArgs| {
            let t0 = (y[0] + MU) * (y[0] + MU) + y[1] * y[1];
            let t1 = (y[0] - MD) * (y[0] - MD) + y[1] * y[1];
            let d0 = t0 * f64::sqrt(t0);
            let d1 = t1 * f64::sqrt(t1);
            f[0] = y[2];
            f[1] = y[3];
            f[2] = y[0] + 2.0 * y[3] - MD * (y[0] + MU) / d0 - MU * (y[0] - MD) / d1;
            f[3] = y[1] - 2.0 * y[2] - MD * y[1] / d0 - MU * y[1] / d1;
            Ok(())
        });

        // function to compute the Jacobian matrix
        let jac_nnz = 8;
        system
            .set_jacobian(
                Some(jac_nnz),
                Sym::No,
                |jj: &mut CooMatrix, alpha: f64, _x: f64, y: &Vector, _args: &mut NoArgs| {
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
                    jj.put(0, 2, 1.0 * alpha).unwrap();
                    jj.put(1, 3, 1.0 * alpha).unwrap();
                    jj.put(2, 0, (1.0 + a * dj00 * MD / dd0 + b * dj10 * MU / dd1 + c) * alpha)
                        .unwrap();
                    jj.put(2, 1, (a * dj01 * MD / dd0 + b * dj11 * MU / dd1) * alpha)
                        .unwrap();
                    jj.put(2, 3, 2.0 * alpha).unwrap();
                    jj.put(3, 0, (dj00 * y[1] * MD / dd0 + dj10 * y[1] * MU / dd1) * alpha)
                        .unwrap();
                    jj.put(
                        3,
                        1,
                        (1.0 + dj01 * y[1] * MD / dd0 + dj11 * y[1] * MU / dd1 + c) * alpha,
                    )
                    .unwrap();
                    jj.put(3, 2, -2.0 * alpha).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial values and final x
        let x0 = 0.0;
        let y0 = Vector::from(&[0.994, 0.0, 0.0, -2.00158510637908252240537862224]);
        let x1 = 17.0652165601579625588917206249;

        // reference solution from Mathematica
        let y_ref = Vector::from(&[
            0.99399999999999280751004722382642,
            2.4228439406717e-14,
            3.6631563591513e-12,
            -2.0015851063802005176067408813970,
        ]);

        // results
        let args = 0;
        (system, x0, y0, x1, args, y_ref)
    }

    /// Returns equation 1.1 from Hairer-Wanner Part II book
    ///
    /// Returns `(system, x0, y0, args, y_fn_x)`
    ///
    /// This example corresponds to Fig 1.1 and Fig 1.2 on page 2 of the reference.
    /// The problem is defined in Eq (1.1) on page 2 of the reference
    ///
    /// The system is:
    ///
    /// ```text
    /// y0' = -50 (y0 - cos(x))
    ///
    /// with  y0(x=0) = 0
    /// ```
    ///
    /// The Jacobian matrix is:
    ///
    /// ```text
    ///     df   ┌     ┐
    /// J = —— = │ -50 │
    ///     dy   └     ┘
    /// ```
    ///
    /// # Output
    ///
    /// * `system` -- the ODE system
    /// * `x0` -- initial x
    /// * `y0` -- initial y
    /// * `args` -- is a placeholder variable with the arguments to F and J
    /// * `y_fn_x` -- is a function to compute the analytical solution
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn hairer_wanner_eq1<'a>() -> (
        System<'a, NoArgs>,
        f64,
        Vector,
        NoArgs,
        impl Fn(&mut Vector, f64, &mut NoArgs),
    ) {
        // constants
        const L: f64 = -50.0; // lambda

        // system
        let ndim = 1;
        let mut system = System::new(ndim, |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = L * (y[0] - f64::cos(x));
            Ok(())
        });

        // function to compute the Jacobian matrix
        let jac_nnz = 1;
        system
            .set_jacobian(
                Some(jac_nnz),
                Sym::No,
                |jj: &mut CooMatrix, alpha: f64, _x: f64, _y: &Vector, _args: &mut NoArgs| {
                    jj.reset();
                    jj.put(0, 0, alpha * L).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[0.0]);

        // analytical solution
        let y_fn_x = |y: &mut Vector, x: f64, _args: &mut NoArgs| {
            y[0] = -L * (f64::sin(x) - L * f64::cos(x) + L * f64::exp(L * x)) / (L * L + 1.0);
        };

        // results
        let args = 0;
        (system, x0, y0, args, y_fn_x)
    }

    /// Returns the Robertson's problem
    ///
    /// Returns `(system, x0, y0, args)`:
    ///
    /// This example corresponds to Fig 1.3 on page 4 of the reference.
    /// The problem is defined in Eq (1.4) on page 3 of the reference.
    ///
    /// The system is:
    ///
    /// ```text
    /// y0' = -0.04 y0 + 1.0e4 y1 y2
    /// y1' =  0.04 y0 - 1.0e4 y1 y2 - 3.0e7 y1²
    /// y2' =                          3.0e7 y1²
    ///
    /// with  y0(0) = 1, y1(0) = 0, y2(0) = 0
    /// ```
    ///
    /// # Output
    ///
    /// * `system` -- the ODE system
    /// * `x0` -- initial x
    /// * `y0` -- initial y
    /// * `args` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn robertson<'a>() -> (System<'a, NoArgs>, f64, Vector, NoArgs) {
        // system
        let ndim = 3;
        let mut system = System::new(ndim, |f: &mut Vector, _x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
            f[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
            f[2] = 3.0e7 * y[1] * y[1];
            Ok(())
        });

        // function to compute the Jacobian matrix
        let jac_nnz = 7;
        system
            .set_jacobian(
                Some(jac_nnz),
                Sym::No,
                |jj: &mut CooMatrix, alpha: f64, _x: f64, y: &Vector, _args: &mut NoArgs| {
                    jj.reset();
                    jj.put(0, 0, -0.04 * alpha).unwrap();
                    jj.put(0, 1, 1.0e4 * y[2] * alpha).unwrap();
                    jj.put(0, 2, 1.0e4 * y[1] * alpha).unwrap();
                    jj.put(1, 0, 0.04 * alpha).unwrap();
                    jj.put(1, 1, (-1.0e4 * y[2] - 6.0e7 * y[1]) * alpha).unwrap();
                    jj.put(1, 2, (-1.0e4 * y[1]) * alpha).unwrap();
                    jj.put(2, 1, 6.0e7 * y[1] * alpha).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[1.0, 0.0, 0.0]);

        // results
        let args = 0;
        (system, x0, y0, args)
    }

    /// Returns the Van der Pol's problem
    ///
    /// Returns `(system, x0, y0, x1, args)`
    ///
    /// This example corresponds to Eq (1.5') on page 5 of the reference and is used to compare with
    /// Fig 2.6 on page 23 and Fig 8.1 on page 125 of the reference.
    ///
    /// The system is:
    ///
    /// ```text
    /// y0' = y1
    /// y1' = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / ε
    /// ```
    ///
    /// where ε defines the *stiffness* of the problem + conditions (equation + initial conditions + step size + method).
    ///
    /// **Note:** Using the data from Eq (7.29), page 113.
    ///
    /// # Input
    ///
    /// * `epsilon` -- ε coefficient; use None for the default value (= 1.0e-6)
    /// * `stationary` -- use `ε = 1` and compute the period and amplitude such that
    ///   `y = [A, 0]` is a stationary point.
    ///
    /// # Output
    ///
    /// * `system` -- the ODE system
    /// * `x0` -- initial x
    /// * `y0` -- initial y
    /// * `x1` -- final x
    /// * `args` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn van_der_pol<'a>(epsilon: f64, stationary: bool) -> (System<'a, NoArgs>, f64, Vector, f64, NoArgs) {
        // constants
        let x0 = 0.0;
        let mut y0 = Vector::from(&[2.0, -0.6]);
        let mut x1 = 2.0;
        let eps = if stationary {
            const A: f64 = 2.00861986087484313650940188;
            const T: f64 = 6.6632868593231301896996820305;
            y0[0] = A;
            y0[1] = 0.0;
            x1 = T;
            1.0
        } else {
            epsilon
        };

        // system
        let ndim = 2;
        let mut system = System::new(ndim, move |f: &mut Vector, _x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = y[1];
            f[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / eps;
            Ok(())
        });

        // function to compute the Jacobian matrix
        let jac_nnz = 3;
        system
            .set_jacobian(
                Some(jac_nnz),
                Sym::No,
                move |jj: &mut CooMatrix, alpha: f64, _x: f64, y: &Vector, _args: &mut NoArgs| {
                    jj.reset();
                    jj.put(0, 1, 1.0 * alpha).unwrap();
                    jj.put(1, 0, alpha * (-2.0 * y[0] * y[1] - 1.0) / eps).unwrap();
                    jj.put(1, 1, alpha * (1.0 - y[0] * y[0]) / eps).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // results
        let args = 0;
        (system, x0, y0, x1, args)
    }

    /// Returns the one-transistor amplifier problem
    ///
    /// Returns `(system, x0, y0, args)`
    ///
    /// This example corresponds to Fig 1.3 on page 377 and Fig 1.4 on page 379 of the reference.
    /// The problem is defined in Eq (1.14) on page 377 of the reference.
    ///
    /// This is a differential-algebraic problem modelling the nodal voltages of a one-transistor amplifier.
    ///
    /// The DAE is expressed in the so-called *mass-matrix* form (ndim = 5):
    ///
    /// ```text
    /// M y' = f(x, y)
    ///
    /// with: y0(0)=0, y1(0)=Ub/2, y2(0)=Ub/2, y3(0)=Ub, y4(0)=0
    /// ```
    ///
    /// where the elements of the right-hand side function are:
    ///
    /// ```text
    /// f0 = (y0 - ue) / R
    /// f1 = (2 y1 - UB) / S + γ g12
    /// f2 = y2 / S - g12
    /// f3 = (y3 - UB) / S + α g12
    /// f4 = y4 / S
    ///
    /// with:
    ///
    /// ue = A sin(ω x)
    /// g12 = β (exp((y1 - y2) / UF) - 1)
    /// ```
    ///
    /// Compared to Eq (1.14), we set all resistances Rᵢ to S, except the first one (R := R₀).
    ///
    /// The mass matrix is:
    ///
    /// ```text
    ///     ┌                     ┐
    ///     │ -C1  C1             │
    ///     │  C1 -C1             │
    /// M = │         -C2         │
    ///     │             -C3  C3 │
    ///     │              C3 -C3 │
    ///     └                     ┘
    /// ```
    ///
    /// and the Jacobian matrix is:
    ///
    /// ```text
    ///     ┌                                           ┐
    ///     │ 1/R                                       │
    ///     │       2/S + γ h12      -γ h12             │
    /// J = │              -h12   1/S + h12             │
    ///     │             α h12      -α h12             │
    ///     │                                 1/S       │
    ///     │                                       1/S │
    ///     └                                           ┘
    ///
    /// with:
    ///
    /// h12 = β exp((y1 - y2) / UF) / UF
    /// ```
    ///
    /// # Output
    ///
    /// * `system` -- the ODE system
    /// * `x0` -- initial x
    /// * `y0` -- initial y
    /// * `args` -- is a placeholder variable with the arguments to F and J
    ///
    /// # Reference
    ///
    /// * Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
    ///   Stiff and Differential-Algebraic Problems. Second Revised Edition.
    ///   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
    pub fn amplifier1t<'a>() -> (System<'a, NoArgs>, f64, Vector, NoArgs) {
        // constants
        const ALPHA: f64 = 0.99;
        const GAMMA: f64 = 1.0 - ALPHA;
        const BETA: f64 = 1e-6;
        const A: f64 = 0.4;
        const OM: f64 = 200.0 * PI;
        const UB: f64 = 6.0;
        const UF: f64 = 0.026;
        const R: f64 = 1000.0;
        const S: f64 = 9000.0;

        // system
        let ndim = 5;
        let mut system = System::new(ndim, |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
            let ue = A * f64::sin(OM * x);
            let g12 = BETA * (f64::exp((y[1] - y[2]) / UF) - 1.0);
            f[0] = (y[0] - ue) / R;
            f[1] = (2.0 * y[1] - UB) / S + GAMMA * g12;
            f[2] = y[2] / S - g12;
            f[3] = (y[3] - UB) / S + ALPHA * g12;
            f[4] = y[4] / S;
            Ok(())
        });

        // function to compute the Jacobian matrix
        let jac_nnz = 9;
        let symmetric = Sym::No;
        system
            .set_jacobian(
                Some(jac_nnz),
                symmetric,
                |jj: &mut CooMatrix, aa: f64, _x: f64, y: &Vector, _args: &mut NoArgs| {
                    let h12 = BETA * f64::exp((y[1] - y[2]) / UF) / UF;
                    jj.reset();
                    jj.put(0, 0, aa * (1.0 / R)).unwrap();
                    jj.put(1, 1, aa * (2.0 / S + GAMMA * h12)).unwrap();
                    jj.put(1, 2, aa * (-GAMMA * h12)).unwrap();
                    jj.put(2, 1, aa * (-h12)).unwrap();
                    jj.put(2, 2, aa * (1.0 / S + h12)).unwrap();
                    jj.put(3, 1, aa * (ALPHA * h12)).unwrap();
                    jj.put(3, 2, aa * (-ALPHA * h12)).unwrap();
                    jj.put(3, 3, aa * (1.0 / S)).unwrap();
                    jj.put(4, 4, aa * (1.0 / S)).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // mass matrix
        const C1: f64 = 1e-6;
        const C2: f64 = 2e-6;
        const C3: f64 = 3e-6;
        let mass_nnz = 9;
        system
            .set_mass(Some(mass_nnz), symmetric, |mm: &mut CooMatrix| {
                mm.put(0, 0, -C1).unwrap();
                mm.put(0, 1, C1).unwrap();
                mm.put(1, 0, C1).unwrap();
                mm.put(1, 1, -C1).unwrap();
                mm.put(2, 2, -C2).unwrap();
                mm.put(3, 3, -C3).unwrap();
                mm.put(3, 4, C3).unwrap();
                mm.put(4, 3, C3).unwrap();
                mm.put(4, 4, -C3).unwrap();
            })
            .unwrap();

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[0.0, UB / 2.0, UB / 2.0, UB, 0.0]);

        // results
        let args = 0;
        (system, x0, y0, args)
    }

    /// Implements Equation (6) from Kreyszig's book on page 902
    ///
    /// Returns `(system, x0, y0, args, y_fn_x)`
    ///
    /// Problem:
    ///
    /// ```text
    /// dy/dx = x + y
    /// y(0) = 0
    /// ```
    ///
    /// # Output
    ///
    /// * `system` -- the ODE system
    /// * `x0` -- initial x
    /// * `y0` -- initial y
    /// * `args` -- is a placeholder variable with the arguments to F and J
    /// * `y_fn_x` -- is a function to compute the analytical solution
    ///
    /// # Reference
    ///
    /// * Kreyszig, E (2011) Advanced engineering mathematics; in collaboration with Kreyszig H,
    ///    Edward JN 10th ed 2011, Hoboken, New Jersey, Wiley
    pub fn kreyszig_eq6_page902<'a>() -> (
        System<'a, NoArgs>,
        f64,
        Vector,
        NoArgs,
        impl Fn(&mut Vector, f64, &mut NoArgs),
    ) {
        // system
        let ndim = 1;
        let mut system = System::new(ndim, |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = x + y[0];
            Ok(())
        });

        // function to compute the Jacobian matrix
        let jac_nnz = 1;
        system
            .set_jacobian(
                Some(jac_nnz),
                Sym::No,
                |jj: &mut CooMatrix, alpha: f64, _x: f64, _y: &Vector, _args: &mut NoArgs| {
                    jj.reset();
                    jj.put(0, 0, 1.0 * alpha).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[0.0]);

        // analytical solution
        let y_fn_x = |y: &mut Vector, x: f64, _args: &mut NoArgs| {
            y[0] = f64::exp(x) - x - 1.0;
        };

        // results
        let args = 0;
        (system, x0, y0, args, y_fn_x)
    }

    /// Implements Example 4 from Kreyszig's book on page 920
    ///
    /// Returns `(system, x0, y0, args, y_fn_x)`
    ///
    /// With proper initial conditions, this problem becomes "stiff".
    ///
    /// Problem:
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
    /// * `system` -- the ODE system
    /// * `x0` -- initial x
    /// * `y0` -- initial y
    /// * `args` -- is a placeholder variable with the arguments to F and J
    /// * `y_fn_x` -- is a function to compute the analytical solution
    ///
    /// # Reference
    ///
    /// * Kreyszig, E (2011) Advanced engineering mathematics; in collaboration with Kreyszig H,
    ///    Edward JN 10th ed 2011, Hoboken, New Jersey, Wiley
    pub fn kreyszig_ex4_page920<'a>() -> (
        System<'a, NoArgs>,
        f64,
        Vector,
        NoArgs,
        impl Fn(&mut Vector, f64, &mut NoArgs),
    ) {
        // system
        let ndim = 2;
        let mut system = System::new(ndim, |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = y[1];
            f[1] = -10.0 * y[0] - 11.0 * y[1] + 10.0 * x + 11.0;
            Ok(())
        });

        // function to compute the Jacobian matrix
        let jac_nnz = 3;
        system
            .set_jacobian(
                Some(jac_nnz),
                Sym::No,
                |jj: &mut CooMatrix, alpha: f64, _x: f64, _y: &Vector, _args: &mut NoArgs| {
                    jj.reset();
                    jj.put(0, 1, 1.0 * alpha).unwrap();
                    jj.put(1, 0, -10.0 * alpha).unwrap();
                    jj.put(1, 1, -11.0 * alpha).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // initial values
        let x0 = 0.0;
        let y0 = Vector::from(&[2.0, -10.0]);

        // analytical solution
        let y_fn_x = |y: &mut Vector, x: f64, _args: &mut NoArgs| {
            y[0] = f64::exp(-x) + f64::exp(-10.0 * x) + x;
            y[1] = -f64::exp(-x) - 10.0 * f64::exp(-10.0 * x) + 1.0;
        };

        // results
        let args = 0;
        (system, x0, y0, args, y_fn_x)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Samples;
    use russell_lab::algo::num_jacobian;
    use russell_lab::{mat_approx_eq, vec_approx_eq, Vector};
    use russell_sparse::{CooMatrix, Genie, Sym};

    #[test]
    fn simple_equation_constant_works() {
        let alpha = 2.0;
        let (system, x0, y0, mut args, y_fn_x) = Samples::simple_equation_constant();

        // check initial values
        let mut y0_correct = Vector::new(y0.dim());
        y_fn_x(&mut y0_correct, x0, &mut args);
        println!("y0 = {:?} = {:?}", y0_correct.as_data(), y0.as_data());
        vec_approx_eq(&y0, &y0_correct, 1e-15);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn simple_system_with_mass_matrix_works() {
        let alpha = 2.0;
        let (system, x0, y0, mut args, y_fn_x) = Samples::simple_system_with_mass_matrix(true, Genie::Umfpack);

        // check initial values
        let mut y0_correct = Vector::new(y0.dim());
        y_fn_x(&mut y0_correct, x0, &mut args);
        println!("y0 = {:?} = {:?}", y0_correct.as_data(), y0.as_data());
        vec_approx_eq(&y0, &y0_correct, 1e-15);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn brusselator_ode_works() {
        let alpha = 2.0;
        let (system, x0, y0, mut args, _) = Samples::brusselator_ode();

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{:.15}", ana);
        println!("{:.15}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn brusselator_pde_no_diffusion_works() {
        let jac_alpha = 2.0;
        let second_book = false;
        let ignore_diffusion = true;
        let (system, x0, y0, mut args) = Samples::brusselator_pde(2e-3, 3, second_book, ignore_diffusion);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, jac_alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, jac_alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{:.2}", ana);
        println!("{:.2}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn brusselator_pde_works() {
        let jac_alpha = 2.0;
        let second_book = false;
        let ignore_diffusion = false;
        let (system, x0, y0, mut args) = Samples::brusselator_pde(2e-3, 3, second_book, ignore_diffusion);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, jac_alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, jac_alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{:.2}", ana);
        println!("{:.2}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn brusselator_pde_2nd_works() {
        let alpha = 0.1;
        let npoint = 6;
        let second_book = true;
        let ignore_diffusion = false;
        let (system, _, y0, mut args) = Samples::brusselator_pde(alpha, npoint, second_book, ignore_diffusion);

        for t in [0.9, 1.2] {
            // compute the analytical Jacobian matrix
            let jac_alpha = 2.0;
            let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
            let jacobian = system.jacobian.as_ref().unwrap();
            (jacobian)(&mut jj, jac_alpha, t, &y0, &mut args).unwrap();

            // compute the numerical Jacobian matrix
            let num = num_jacobian(system.ndim, t, &y0, jac_alpha, &mut args, system.function.as_ref()).unwrap();

            // check the Jacobian matrix
            let ana = jj.as_dense();
            // println!("{:.2}", ana);
            // println!("{:.2}", num);
            mat_approx_eq(&ana, &num, 1e-9);
        }
    }

    #[test]
    fn arenstorf_works() {
        let alpha = 1.5;
        let (system, x0, y0, _, mut args, _) = Samples::arenstorf();

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1.6e-4);
    }

    #[test]
    fn hairer_wanner_eq1_works() {
        let alpha = 2.0;
        let (system, x0, y0, mut args, y_fn_x) = Samples::hairer_wanner_eq1();

        // check initial values
        let mut y0_correct = Vector::new(y0.dim());
        y_fn_x(&mut y0_correct, x0, &mut args);
        println!("y0 = {:?} = {:?}", y0_correct.as_data(), y0.as_data());
        vec_approx_eq(&y0, &y0_correct, 1e-15);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn robertson_works() {
        let alpha = 2.0;
        let (system, x0, y0, mut args) = Samples::robertson();

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-14);
    }

    #[test]
    fn van_der_pol_works() {
        let alpha = 2.0;
        let (system, x0, y0, _, mut args) = Samples::van_der_pol(0.03, false);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1.5e-6);
    }

    #[test]
    fn van_der_pol_works_stationary() {
        let alpha = 3.0;
        let (system, x0, y0, _, mut args) = Samples::van_der_pol(1.0, true);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn amplifier1t_works() {
        let alpha = 2.0;
        let (system, x0, y0, mut args) = Samples::amplifier1t();

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{:.15}", ana);
        println!("{:.15}", num);
        mat_approx_eq(&ana, &num, 1e-13);

        // check the mass matrix
        let mut mass = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let calc_mass = system.calc_mass.as_ref().unwrap();
        (calc_mass)(&mut mass);
        println!("{}", mass.as_dense());
        let ndim = system.ndim;
        let nnz_mass = 5 + 4;
        assert_eq!(mass.get_info(), (ndim, ndim, nnz_mass, Sym::No));
    }

    #[test]
    fn kreyszig_eq6_page902_works() {
        let alpha = 2.0;
        let (system, x0, y0, mut args, y_fn_x) = Samples::kreyszig_eq6_page902();

        // check initial values
        let mut y0_correct = Vector::new(y0.dim());
        y_fn_x(&mut y0_correct, x0, &mut args);
        println!("y0 = {:?} = {:?}", y0_correct.as_data(), y0.as_data());
        vec_approx_eq(&y0, &y0_correct, 1e-15);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn kreyszig_ex4_page920() {
        let alpha = 2.0;
        let (system, x0, y0, mut args, y_fn_x) = Samples::kreyszig_ex4_page920();

        // check initial values
        let mut y0_correct = Vector::new(y0.dim());
        y_fn_x(&mut y0_correct, x0, &mut args);
        println!("y0 = {:?} = {:?}", y0_correct.as_data(), y0.as_data());
        vec_approx_eq(&y0, &y0_correct, 1e-15);

        // compute the analytical Jacobian matrix
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, system.symmetric).unwrap();
        let jacobian = system.jacobian.as_ref().unwrap();
        (jacobian)(&mut jj, alpha, x0, &y0, &mut args).unwrap();

        // compute the numerical Jacobian matrix
        let num = num_jacobian(system.ndim, x0, &y0, alpha, &mut args, system.function.as_ref()).unwrap();

        // check the Jacobian matrix
        let ana = jj.as_dense();
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-10);
    }
}
