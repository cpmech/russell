use crate::StrError;
use russell_lab::Vector;
use russell_sparse::{CooMatrix, Sym};
use std::sync::Arc;

/// Indicates that the system functions do not require extra arguments
pub type NoArgs = u8;

/// Defines a system of first order ordinary differential equations (ODE) or a differential-algebraic equations (DAE) of Index-1
///
/// The system is defined by:
///
/// ```text
///     d{y}
/// [M] ———— = {f}(x, {y})
///      dx
/// ```
///
/// where `x` is the independent scalar variable (e.g., time), `{y}` is the solution vector,
/// `{f}` is the right-hand side vector, and `[M]` is the so-called "mass matrix".
///
/// **Note:** The mass matrix is optional and need not be specified.
/// (unless the DAE under study requires it).
///
/// The (scaled) Jacobian matrix is defined by:
///
/// ```text
///                 ∂{f}
/// [J](x, {y}) = α ————
///                 ∂{y}
/// ```
///
/// where `[J]` is the scaled Jacobian matrix and `α` is a scaling coefficient.
///
/// See [crate::Samples] for many examples on how to define the system (in [crate::Samples], click on the *source*
/// link in the documentation to access the source code illustrating the allocation of System).
///
/// # Generics
///
/// * `A` -- generic argument to assist in the f(x,y) and Jacobian functions.
///   It may be simply [NoArgs] indicating that no arguments are needed.
///
/// # Important
///
/// The implementation requires the `alpha` parameter in the Jacobian function
/// to scale the Jacobian matrix. For example:
///
/// ```text
/// |jj: &mut CooMatrix, alpha: f64, x: f64, y: &Vector, args: &mut Args| {
///     jj.reset();
///     jj.put(0, 0, alpha * y[0])?;
///     Ok(())
/// },
/// ```
///
/// # References
///
/// 1. E. Hairer, S. P. Nørsett, G. Wanner (2008) Solving Ordinary Differential Equations I.
///    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
///    in Computational Mathematics, 528p
/// 2. E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
///    Stiff and Differential-Algebraic Problems. Second Revised Edition.
///    Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
pub struct System<'a, A> {
    /// System dimension
    pub(crate) ndim: usize,

    /// ODE system function
    pub(crate) function: Arc<dyn Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + 'a>,

    /// Jacobian function
    pub(crate) jacobian: Option<Arc<dyn Fn(&mut CooMatrix, f64, f64, &Vector, &mut A) -> Result<(), StrError> + 'a>>,

    /// Calc mass matrix function
    pub(crate) calc_mass: Option<Arc<dyn Fn(&mut CooMatrix) + 'a>>,

    /// Number of non-zeros in the Jacobian matrix
    pub(crate) jac_nnz: usize,

    /// Number of non-zeros in the mass matrix
    pub(crate) mass_nnz: usize,

    /// Symmetric type of the Jacobian matrix (for error checking; to make sure it is equal to sym_mass)
    sym_jac: Option<Sym>,

    /// Symmetric type of the mass matrix (for error checking; to make sure it is equal to sym_jacobian)
    sym_mass: Option<Sym>,

    /// Symmetric type of the Jacobian and mass matrices
    pub(crate) symmetric: Sym,
}

impl<'a, A> System<'a, A> {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `ndim` -- dimension of the ODE system (number of equations)
    /// * `function` -- implements the function: `dy/dx = f(x, y)`
    ///
    /// **Note:** Even if the (analytical) Jacobian function is not configured,
    /// a numerical Jacobian matrix may be computed (see [crate::Params] and [crate::ParamsNewton]).
    ///
    /// # Generics
    ///
    /// * `A` -- generic argument to assist in the f(x,y) and Jacobian functions.
    ///   It may be simply [NoArgs] indicating that no arguments are needed.
    ///
    /// # Examples
    ///
    /// ## One equation (ndim = 1) without Jacobian callback function
    ///
    /// ```rust
    /// use russell_ode::prelude::*;
    /// use russell_ode::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let system = System::new(1, |f, x, y, _args: &mut NoArgs| {
    ///         f[0] = x + y[0];
    ///         Ok(())
    ///     });
    ///     Ok(())
    /// }
    /// ```
    ///
    /// ## Two equation system (ndim = 2) with Jacobian
    ///
    /// ```rust
    /// use russell_ode::prelude::*;
    /// use russell_ode::StrError;
    /// use russell_sparse::Sym;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let ndim = 2;
    ///     let mut system = System::new(ndim, |f, x, y, _args: &mut NoArgs| {
    ///         f[0] = x + 2.0 * y[0] + 3.0 * y[1];
    ///         f[1] = x - 4.0 * y[0] - 5.0 * y[1];
    ///         Ok(())
    ///     });
    ///
    ///     let jac_nnz = 4;
    ///     system.set_jacobian(Some(jac_nnz), Sym::No, |jj, alpha, _x, _y, _args: &mut NoArgs| {
    ///         jj.reset();
    ///         jj.put(0, 0, alpha * (2.0)).unwrap();
    ///         jj.put(0, 1, alpha * (3.0)).unwrap();
    ///         jj.put(1, 0, alpha * (-4.0)).unwrap();
    ///         jj.put(1, 1, alpha * (-5.0)).unwrap();
    ///         Ok(())
    ///     });
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn new(ndim: usize, function: impl Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + 'a) -> Self {
        System {
            ndim,
            function: Arc::new(function),
            jacobian: None,
            calc_mass: None,
            jac_nnz: ndim * ndim,
            mass_nnz: 0,
            sym_jac: None,
            sym_mass: None,
            symmetric: Sym::No,
        }
    }

    /// Returns a copy of this struct
    pub fn clone(&self) -> Self {
        System {
            ndim: self.ndim,
            function: self.function.clone(),
            jacobian: self.jacobian.clone(),
            calc_mass: self.calc_mass.clone(),
            jac_nnz: self.jac_nnz,
            mass_nnz: self.mass_nnz,
            sym_jac: self.sym_jac,
            sym_mass: self.sym_mass,
            symmetric: self.symmetric,
        }
    }

    /// Sets a function to calculate the Jacobian matrix (analytical Jacobian)
    ///
    /// Use `|jj, alpha, x, y, args|` or `|jj: &mut CooMatrix, alpha: f64, x: f64, y: &Vector, args: &mut A|`
    ///
    /// # Input
    ///
    /// * `nnz` -- the number of non-zeros in the Jacobian; use None to indicate a dense matrix with:
    ///     * `nnz = (ndim + ndim²) / 2` if triangular
    ///     * `nnz = ndim²` otherwise
    /// * `symmetric` -- specifies the symmetric type of the Jacobian and **mass** matrices
    /// * `callback` -- the function to calculate the Jacobian matrix
    pub fn set_jacobian(
        &mut self,
        nnz: Option<usize>,
        symmetric: Sym,
        callback: impl Fn(&mut CooMatrix, f64, f64, &Vector, &mut A) -> Result<(), StrError> + 'a,
    ) -> Result<(), StrError> {
        if let Some(sym) = self.sym_mass {
            if symmetric != sym {
                return Err("the Jacobian matrix must have the same symmetric type as the mass matrix");
            }
        }
        self.jac_nnz = if let Some(value) = nnz {
            value
        } else {
            if symmetric.triangular() {
                (self.ndim + self.ndim * self.ndim) / 2
            } else {
                self.ndim * self.ndim
            }
        };
        self.sym_jac = Some(symmetric);
        self.symmetric = symmetric;
        self.jacobian = Some(Arc::new(callback));
        Ok(())
    }

    /// Sets a function to calculate the constant mass matrix
    ///
    /// # Input
    ///
    /// * `nnz` -- the number of non-zeros in the mass matrix; use None to indicate a dense matrix with:
    ///     * `nnz = (ndim + ndim²) / 2` if triangular
    ///     * `nnz = ndim²` otherwise
    /// * `symmetric` -- specifies the symmetric type for the mass and **Jacobian** matrices
    /// * `callback` -- the function to calculate the mass matrix (will be called just once)
    pub fn set_mass(
        &mut self,
        nnz: Option<usize>,
        symmetric: Sym,
        callback: impl Fn(&mut CooMatrix) + 'a,
    ) -> Result<(), StrError> {
        if let Some(sym) = self.sym_jac {
            if symmetric != sym {
                return Err("the mass matrix must have the same symmetric type as the Jacobian matrix");
            }
        }
        self.mass_nnz = if let Some(value) = nnz {
            value
        } else {
            if symmetric.triangular() {
                (self.ndim + self.ndim * self.ndim) / 2
            } else {
                self.ndim * self.ndim
            }
        };
        self.sym_mass = Some(symmetric);
        self.symmetric = symmetric;
        self.calc_mass = Some(Arc::new(callback));
        Ok(())
    }

    /// Returns the dimension of the ODE system
    pub fn get_ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the number of non-zero values in the Jacobian matrix
    pub fn get_jac_nnz(&self) -> usize {
        self.jac_nnz
    }

    /// Returns the number of non-zero values in the mass matrix
    pub fn get_mass_nnz(&self) -> usize {
        self.mass_nnz
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::System;
    use crate::NoArgs;
    use russell_lab::Vector;
    use russell_sparse::{CooMatrix, Sym};

    #[test]
    fn ode_system_handles_errors() {
        let mut system = System::new(1, |f, _, _, _: &mut NoArgs| {
            f[0] = 1.0;
            Ok(())
        });
        let mut f = Vector::new(1);
        let x = 0.0;
        let y = Vector::new(1);
        let mut args = 0;
        (system.function)(&mut f, x, &y, &mut args).unwrap();
        let jac_cb = |_: &mut CooMatrix, _: f64, _: f64, _: &Vector, _: &mut NoArgs| Ok(());
        let mas_cb = |_: &mut CooMatrix| ();
        let mut jj = CooMatrix::new(1, 1, 1, Sym::YesLower).unwrap();
        let mut mm = CooMatrix::new(1, 1, 1, Sym::YesLower).unwrap();
        let y = Vector::new(1);
        (jac_cb)(&mut jj, 0.0, 0.0, &y, &mut 0).unwrap();
        (mas_cb)(&mut mm);
        system.set_jacobian(None, Sym::YesLower, jac_cb).unwrap();
        assert_eq!(
            system.set_mass(None, Sym::YesUpper, mas_cb).err(),
            Some("the mass matrix must have the same symmetric type as the Jacobian matrix")
        );
        system.sym_jac = None;
        system.set_mass(None, Sym::YesLower, mas_cb).unwrap();
        assert_eq!(
            system.set_jacobian(None, Sym::YesUpper, jac_cb).err(),
            Some("the Jacobian matrix must have the same symmetric type as the mass matrix")
        );
        system.set_jacobian(None, Sym::YesLower, jac_cb).unwrap(); // ok
    }

    #[test]
    fn ode_system_works() {
        struct Args {
            n_function_eval: usize,
            more_data_goes_here: bool,
        }
        let mut args = Args {
            n_function_eval: 0,
            more_data_goes_here: false,
        };
        let system = System::new(2, |f, x, y, args: &mut Args| {
            args.n_function_eval += 1;
            f[0] = -x * y[1];
            f[1] = x * y[0];
            args.more_data_goes_here = true;
            Ok(())
        });
        assert_eq!(system.get_ndim(), 2);
        assert_eq!(system.get_jac_nnz(), 4);
        // call system function
        let x = 0.0;
        let y = Vector::new(2);
        let mut k = Vector::new(2);
        (system.function)(&mut k, x, &y, &mut args).unwrap();
        // check that jacobian function is none
        assert!(system.jacobian.is_none());
        // check
        println!("n_function_eval = {}", args.n_function_eval);
        assert_eq!(args.n_function_eval, 1);
        assert_eq!(args.more_data_goes_here, true);
        // clone
        let clone = system.clone();
        assert_eq!(clone.ndim, 2);
        assert_eq!(clone.jac_nnz, 4);
        assert_eq!(clone.mass_nnz, 0);
        assert_eq!(clone.sym_jac, None);
        assert_eq!(clone.sym_mass, None);
        assert_eq!(clone.symmetric, Sym::No);
    }

    #[test]
    fn ode_system_set_jacobian_works() {
        struct Args {
            n_function_eval: usize,
            n_jacobian_eval: usize,
            more_data_goes_here_fn: bool,
            more_data_goes_here_jj: bool,
        }
        let mut args = Args {
            n_function_eval: 0,
            n_jacobian_eval: 0,
            more_data_goes_here_fn: false,
            more_data_goes_here_jj: false,
        };
        let mut system = System::new(2, |f, x, y, args: &mut Args| {
            args.n_function_eval += 1;
            f[0] = -x * y[1];
            f[1] = x * y[0];
            args.more_data_goes_here_fn = true;
            Ok(())
        });
        let symmetric = Sym::No;
        system
            .set_jacobian(Some(2), symmetric, |jj, alpha, x, _y, args: &mut Args| {
                args.n_jacobian_eval += 1;
                jj.reset();
                jj.put(0, 1, alpha * (-x)).unwrap();
                jj.put(1, 0, alpha * (x)).unwrap();
                args.more_data_goes_here_jj = true;
                Ok(())
            })
            .unwrap();
        // analytical_solution:
        // y[0] = f64::cos(x * x / 2.0) - 2.0 * f64::sin(x * x / 2.0);
        // y[1] = 2.0 * f64::cos(x * x / 2.0) + f64::sin(x * x / 2.0);
        // call system function
        let x = 0.0;
        let y = Vector::new(2);
        let mut k = Vector::new(2);
        (system.function)(&mut k, x, &y, &mut args).unwrap();
        // call jacobian function
        let mut jj = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        let alpha = 1.0;
        (system.jacobian.as_ref().unwrap())(&mut jj, alpha, x, &y, &mut args).unwrap();
        // check
        println!("n_function_eval = {}", args.n_function_eval);
        println!("n_jacobian_eval = {}", args.n_jacobian_eval);
        assert_eq!(args.n_function_eval, 1);
        assert_eq!(args.n_jacobian_eval, 1);
        assert_eq!(args.more_data_goes_here_fn, true);
        assert_eq!(args.more_data_goes_here_jj, true);
    }

    #[test]
    fn ode_system_set_mass_works() {
        let mut system = System::new(2, |f, _, _, _: &mut NoArgs| {
            f[0] = 1.0;
            f[1] = 1.0;
            Ok(())
        });
        let mut f = Vector::new(2);
        let x = 0.0;
        let y = Vector::new(2);
        let mut args = 0;
        (system.function)(&mut f, x, &y, &mut args).unwrap();
        let mas_cb = |_: &mut CooMatrix| ();
        let mut mm = CooMatrix::new(2, 2, 4, Sym::YesLower).unwrap();
        (mas_cb)(&mut mm);
        system.set_mass(None, Sym::YesLower, mas_cb).unwrap();
        assert_eq!(system.get_mass_nnz(), 3);
        system.set_mass(None, Sym::No, mas_cb).unwrap();
        assert_eq!(system.get_mass_nnz(), 4);
    }
}
