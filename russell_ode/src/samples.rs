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
/// 1. E. Hairer, S. P. Nørsett, G. Wanner (2008) Solving Ordinary Differential Equations I.
///    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
///    in Computational Mathematics, 528p
/// 2. E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
///    Stiff and Differential-Algebraic Problems. Second Revised Edition.
///    Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
pub struct Samples {}

impl Samples {
    /// Returns the Hairer-Wanner problem from the reference, Eq(1.1), page 2
    ///
    /// # Output
    ///
    /// Returns `(System<F, J, A>, SampleData, A)` where:
    ///
    /// * `F` -- is a function to compute the `f` vector; e.g., `fn(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    /// * `J` -- is a function to compute the Jacobian; e.g., `fn(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    /// * `A` -- is `SampleNoArgs`
    ///
    /// # Reference
    ///
    /// * E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
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
        let system = System::new(
            1,
            |f: &mut Vector, x: f64, y: &Vector, _args: &mut SampleNoArgs| {
                f[0] = L * y[0] - L * f64::cos(x);
                Ok(())
            },
            |jj: &mut CooMatrix, _x: f64, _y: &Vector, multiplier: f64, _args: &mut SampleNoArgs| {
                jj.reset();
                jj.put(0, 0, multiplier * L)?;
                Ok(())
            },
            HasJacobian::Yes,
            None,
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

    /// Returns the Van der Pol's equation as given in Hairer-Wanner, Eq(1.5'), page 5
    ///
    /// Using data from Eq(7.29), page 113
    ///
    /// # Input
    ///
    /// * `epsilon` -- ε coefficient; use None for the default value (= 1.0e-6)
    /// * `stationary` -- use `ε = 1` and compute the period and amplitude such that
    ///   `y = [A, 0]` is a stationary point.
    ///
    /// # Output
    ///
    /// Returns `(System<F, J, A>, SampleData, A)` where:
    ///
    /// * `F` -- is a function to compute the `f` vector; e.g., `fn(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    /// * `J` -- is a function to compute the Jacobian; e.g., `fn(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    /// * `A` -- is `SampleNoArgs`
    ///
    /// # Reference
    ///
    /// * E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
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
        let system = System::new(
            2,
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
            Some(3),
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

    /// Returns the Arenstorf orbit problem, Hairer-Wanner, Eq(0.1), page 129
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
    /// Returns `(System<F, J, A>, SampleData, A)` where:
    ///
    /// * `F` -- is a function to compute the `f` vector; e.g., `fn(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
    /// * `J` -- is a function to compute the Jacobian; e.g., `fn(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
    /// * `A` -- is `SampleNoArgs`
    ///
    /// # Reference
    ///
    /// * E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
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
        let system = System::new(
            4,
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
            Some(8),
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{SampleNoArgs, Samples};
    use crate::StrError;
    use russell_lab::{deriv_central5, mat_approx_eq, Matrix, Vector};
    use russell_sparse::CooMatrix;

    fn numerical_jacobian<F>(ndim: usize, x0: f64, y0: Vector, mut function: F) -> Matrix
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
        for i in 0..ndim {
            extra.i = i;
            for j in 0..ndim {
                extra.j = j;
                let at_yj = y0[j];
                let res = deriv_central5(at_yj, &mut extra, |yj: f64, extra: &mut Extra| {
                    let mut args: u8 = 0;
                    let original = extra.y[extra.j];
                    extra.y[extra.j] = yj;
                    function(&mut extra.f, extra.x, &extra.y, &mut args).unwrap();
                    extra.y[extra.j] = original;
                    extra.f[extra.i]
                });
                jac.set(i, j, res);
            }
        }
        jac
    }

    #[test]
    fn sample_hairer_wanner_eq1_jacobian_works() {
        let mut args: u8 = 0;
        let multiplier = 1.0;
        let (mut system, data, _) = Samples::hairer_wanner_eq1();
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();
        let ana = jj.as_dense();
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function);
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-11);
    }

    #[test]
    fn sample_van_der_pol_jacobian_works() {
        let mut args: u8 = 0;
        let multiplier = 1.0;

        // non-stationary
        let (mut system, data, _) = Samples::van_der_pol(None, false);
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();
        let ana = jj.as_dense();
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function);
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-6);

        // stationary
        let (mut system, data, _) = Samples::van_der_pol(None, true);
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();
        let ana = jj.as_dense();
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function);
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-12);
    }

    #[test]
    fn sample_arenstorf_jacobian_works() {
        let mut args: u8 = 0;
        let multiplier = 1.0;
        let (mut system, data, _) = Samples::arenstorf();
        let symmetry = Some(system.jac_symmetry);
        let mut jj = CooMatrix::new(system.ndim, system.ndim, system.jac_nnz, symmetry, false).unwrap();
        (system.jacobian)(&mut jj, data.x0, &data.y0, multiplier, &mut args).unwrap();
        let ana = jj.as_dense();
        let num = numerical_jacobian(system.ndim, data.x0, data.y0, system.function);
        println!("{}", ana);
        println!("{}", num);
        mat_approx_eq(&ana, &num, 1e-3);
    }
}
