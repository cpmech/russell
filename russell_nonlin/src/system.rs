use crate::StrError;
use russell_lab::{Vector, algo::num_jacobian, mat_approx_eq};
use russell_sparse::{CooMatrix, Sym};
use std::sync::Arc;

/// Indicates that the system functions do not require extra arguments
pub type NoArgs = u8;

/// Defines the non-linear system of equations
///
/// The system is defined by:
///
/// ```text
/// Natural:   G(u, λ) = 0
/// Arclength: G(u(s), λ(s)) = 0
/// ```
///
/// Here, `gg` corresponds to `G` and `l` to `λ` (lambda).
///
/// The required derivatives are:
///
/// ```text
/// ggu := Gu = dG/du
/// ggl := Gλ = dG/dλ
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::Vector;
/// use russell_sparse::Sym;
/// use russell_nonlin::System;
///
/// // one equation with a fold point: G(u, λ) = u - λ·exp(u)
/// // Gu = 1 - λ·exp(u), Gλ = -exp(u)
/// let system = System::new(
///     1,
///     Some(1),
///     Sym::No,
///     |gg, l, u, _args| {
///         gg[0] = u[0] - l * f64::exp(u[0]);
///         Ok(())
///     },
///     |ggu, ggl, l, u, _args| {
///         ggu.put(0, 0, 1.0 - l * f64::exp(u[0])).unwrap();
///         ggl[0] = -f64::exp(u[0]);
///         Ok(())
///     },
/// )
/// .unwrap();
///
/// let u = Vector::from(&[0.0]);
/// let l = 0.0;
/// system.check_ggu(l, &u, &mut 0u8, 1e-15).unwrap();
/// ```
pub struct System<'a, A> {
    /// Dimension of `u` and `G`
    pub(crate) ndim: usize,

    /// Number of non-zeros in the Gu matrix
    pub(crate) nnz_ggu: usize,

    /// Symmetric type of the Gu matrix
    pub(crate) sym_ggu: Sym,

    /// Calculates the function G(u) or G(u, λ) or G(u(s), λ(s))
    ///
    /// The function is `calc_gg(gg, l, u, args)`
    pub(crate) calc_gg: Arc<dyn Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a>,

    /// Calculates the Gu = dG/du and Gλ = dG/dλ derivatives
    ///
    /// The function is `calc_jac(ggu, ggl, l, u, args)`
    pub(crate) calc_jac:
        Arc<dyn Fn(&mut CooMatrix, &mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a>,

    /// Creates a copy of external state variables at the beginning of a step
    ///
    /// The function is `fn (args)`
    pub(crate) backup_secondary_state: Option<Arc<dyn Fn(&mut A) + Send + Sync + 'a>>,

    /// Restores external state variables at the end of a step, if the step failed
    ///
    /// The function is `fn (args)`
    pub(crate) restore_secondary_state: Option<Arc<dyn Fn(&mut A) + Send + Sync + 'a>>,

    /// Prepares to iterate (e.g., reset algorithmic variables in the FEM)
    ///
    /// The function is `fn (args)`
    pub(crate) prepare_to_iterate: Option<Arc<dyn Fn(&mut A) + Send + Sync + 'a>>,

    /// Updates secondary variables (e.g., FEM stresses and starred variables)
    ///
    /// The function is `fn (do_backup, u0, u1, l0, l1, args) -> stop_gracefully` with `(u0, l0)` being the
    /// value at the beginning of the step and `(u1, l1)` the value at the updated step.
    pub(crate) update_secondary_state:
        Option<Arc<dyn Fn(bool, &Vector, &Vector, f64, f64, &mut A) -> Result<bool, StrError> + Send + Sync + 'a>>,
}

impl<'a, A> System<'a, A> {
    /// Allocates a new instance
    ///
    /// The functions are: `calc_gg(gg, l, u, args)` and `calc_jac(ggu, ggl, l, u, args)`.
    ///
    /// For simple nonlinear systems, l may be ignored (i.e., not a continuation problem).
    ///
    /// In the Natural method, only `ggu` is needed, so `ggl` may be ignored.
    ///
    /// In the Arclength method, if `bordering = true`, then `ggu` is the actual Gu matrix,
    /// otherwise, `ggu` is either the Gu matrix or the A matrix, depending on the context.
    /// This is necessary to build the system shown below:
    ///
    /// ```text
    /// ┌           ┐ ┌    ┐   ┌    ┐
    /// │ Gu    Gλ  │ │ δu │   │ -G │
    /// │           │ │    │ = │    │
    /// │ Nu₀ᵀ  Nλ₀ │ │ δλ │   │ -N │
    /// └           ┘ └    ┘   └    ┘
    ///       A         x         b
    /// ```
    ///
    /// # Arguments
    ///
    /// * `ndim` -- the dimension of the nonlinear system
    /// * `nnz_ggu` -- the number of non-zeros in the Gu matrix. If None, a **dense** matrix is assumed with:
    ///     * `nnz = (ndim + ndim²) / 2` if triangular
    ///     * `nnz = ndim²` otherwise
    /// * `sym_ggu` -- specifies the symmetry of the Gu matrix
    /// * `calc_gg` -- the callback function to calculate G (in `gg`)
    /// * `calc_jac` -- the callback function to calculate Gu (in `ggu`) and Gλ (in `ggl`). There is no need
    ///   to call `ggu.reset()` inside this function, as it is done already before the call.
    ///
    /// # Returns
    ///
    /// * `Ok(System)` if the parameters are valid
    /// * `Err(StrError)` if `ndim < 1` or `nnz_ggu < 1` (when specified)
    pub fn new(
        ndim: usize,
        nnz_ggu: Option<usize>,
        sym_ggu: Sym,
        calc_gg: impl Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a,
        calc_jac: impl Fn(&mut CooMatrix, &mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a,
    ) -> Result<Self, StrError> {
        if ndim < 1 {
            return Err("ndim must be at least 1");
        }
        let nnz_ggu = match nnz_ggu {
            Some(nnz) => {
                if nnz < 1 {
                    return Err("nnz_ggu must be at least 1");
                }
                nnz
            }
            None => {
                if sym_ggu.triangular() {
                    (ndim + ndim * ndim) / 2
                } else {
                    ndim * ndim
                }
            }
        };
        Ok(System {
            ndim,
            nnz_ggu,
            sym_ggu,
            calc_gg: Arc::new(calc_gg),
            calc_jac: Arc::new(calc_jac),
            backup_secondary_state: None,
            restore_secondary_state: None,
            prepare_to_iterate: None,
            update_secondary_state: None,
        })
    }

    /// Returns a copy of this struct
    ///
    /// Note: `Clone` cannot be derived here because the closure fields are stored as
    /// `Arc<dyn Fn(...)>`, which are not `Clone` in the trait-object sense. This manual
    /// implementation clones the underlying `Arc` pointers (cheap reference-count bump),
    /// so the returned copy shares the same closures as the original.
    pub fn clone(&self) -> Self {
        System {
            ndim: self.ndim,
            nnz_ggu: self.nnz_ggu,
            sym_ggu: self.sym_ggu,
            calc_gg: self.calc_gg.clone(),
            calc_jac: self.calc_jac.clone(),
            backup_secondary_state: self.backup_secondary_state.clone(),
            restore_secondary_state: self.restore_secondary_state.clone(),
            prepare_to_iterate: self.prepare_to_iterate.clone(),
            update_secondary_state: self.update_secondary_state.clone(),
        }
    }

    /// Sets a function to create a copy of external state variables at the beginning of a step
    ///
    /// The function is `fn (args)`
    pub fn set_backup_secondary_state(&mut self, callback: impl Fn(&mut A) + Send + Sync + 'a) -> &mut Self {
        self.backup_secondary_state = Some(Arc::new(callback));
        self
    }

    /// Sets a function to restore external state variables at the end of a step, if the step failed
    ///
    /// The function is `fn (args)`
    pub fn set_restore_secondary_state(&mut self, callback: impl Fn(&mut A) + Send + Sync + 'a) -> &mut Self {
        self.restore_secondary_state = Some(Arc::new(callback));
        self
    }

    /// Sets a function to prepare to iterate (e.g., reset algorithmic variables in the FEM)
    ///
    /// The function is `fn (args)`
    pub fn set_prepare_to_iterate(&mut self, callback: impl Fn(&mut A) + Send + Sync + 'a) -> &mut Self {
        self.prepare_to_iterate = Some(Arc::new(callback));
        self
    }

    /// Sets a function to update secondary variables (e.g., FEM stresses and starred variables)
    ///
    /// The function is `fn (do_backup, u0, u1, l0, l1, args) -> stop_gracefully` with `(u0, l0)` being the
    /// value at the beginning of the step and `(u1, l1)` the value at the updated step.
    pub fn set_update_secondary_state(
        &mut self,
        callback: impl Fn(bool, &Vector, &Vector, f64, f64, &mut A) -> Result<bool, StrError> + Send + Sync + 'a,
    ) -> &mut Self {
        self.update_secondary_state = Some(Arc::new(callback));
        self
    }

    /// Returns the dimension of the nonlinear system
    pub fn get_ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the number of non-zero values in the Gu matrix
    pub fn get_nnz_ggu(&self) -> usize {
        self.nnz_ggu
    }

    /// Returns the symmetric type of the Gu matrix
    pub fn get_sym_ggu(&self) -> Sym {
        self.sym_ggu
    }

    /// Checks Gu using numerical derivative
    pub fn check_ggu(&self, l_at: f64, u_at: &Vector, args: &mut A, tol: f64) -> Result<(), StrError> {
        // analytical Gu
        let mut ggu = CooMatrix::new(self.ndim, self.ndim, self.nnz_ggu, self.sym_ggu).unwrap();
        let mut ggl = Vector::new(self.ndim);
        (self.calc_jac)(&mut ggu, &mut ggl, l_at, &u_at, args).unwrap();

        // numerical Jacobian
        let num = num_jacobian(self.ndim, 0.0, &u_at, 1.0, args, self.calc_gg.as_ref()).unwrap();
        let ana = ggu.as_dense();

        // check
        mat_approx_eq(&ana, &num, tol);
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::System;
    use russell_lab::Vector;
    use russell_sparse::{CooMatrix, Sym};

    fn dummy_calc_gg(_gg: &mut Vector, _l: f64, _u: &Vector, _args: &mut u8) -> Result<(), &'static str> {
        Ok(())
    }

    fn dummy_calc_jac(
        _ggu: &mut CooMatrix,
        _ggl: &mut Vector,
        _l: f64,
        _u: &Vector,
        _args: &mut u8,
    ) -> Result<(), &'static str> {
        Ok(())
    }

    #[test]
    fn new_works() {
        let system = System::new(3, Some(9), Sym::No, dummy_calc_gg, dummy_calc_jac).unwrap();
        assert_eq!(system.get_ndim(), 3);
        assert_eq!(system.get_nnz_ggu(), 9);
        assert_eq!(system.get_sym_ggu(), Sym::No);

        // exercise the closures for coverage
        let mut gg = Vector::new(1);
        let u = Vector::new(1);
        let mut ggu = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        let mut ggl = Vector::new(1);
        (system.calc_gg)(&mut gg, 0.0, &u, &mut 0u8).unwrap();
        (system.calc_jac)(&mut ggu, &mut ggl, 0.0, &u, &mut 0u8).unwrap();
    }

    #[test]
    fn new_captures_ndim_zero() {
        assert_eq!(
            System::new(0, Some(1), Sym::No, dummy_calc_gg, dummy_calc_jac).err(),
            Some("ndim must be at least 1")
        );
    }

    #[test]
    fn new_captures_nnz_ggu_zero() {
        assert_eq!(
            System::new(3, Some(0), Sym::No, dummy_calc_gg, dummy_calc_jac).err(),
            Some("nnz_ggu must be at least 1")
        );
    }

    #[test]
    fn new_with_nnz_none_triangular() {
        // Sym::YesLower is triangular → nnz = (ndim + ndim²) / 2 = (3 + 9) / 2 = 6
        let system = System::new(3, None, Sym::YesLower, dummy_calc_gg, dummy_calc_jac).unwrap();
        assert_eq!(system.get_nnz_ggu(), 6);
    }

    #[test]
    fn new_with_nnz_none_non_triangular() {
        // Sym::No is not triangular → nnz = ndim² = 9
        let system = System::new(3, None, Sym::No, dummy_calc_gg, dummy_calc_jac).unwrap();
        assert_eq!(system.get_nnz_ggu(), 9);
    }

    #[test]
    fn getters_work() {
        let system = System::new(5, Some(7), Sym::YesUpper, dummy_calc_gg, dummy_calc_jac).unwrap();
        assert_eq!(system.get_ndim(), 5);
        assert_eq!(system.get_nnz_ggu(), 7);
        assert_eq!(system.get_sym_ggu(), Sym::YesUpper);
    }

    #[test]
    fn clone_works() {
        let system = System::new(3, Some(9), Sym::No, dummy_calc_gg, dummy_calc_jac).unwrap();
        let cloned = system.clone();
        assert_eq!(cloned.get_ndim(), 3);
        assert_eq!(cloned.get_nnz_ggu(), 9);
        assert_eq!(cloned.get_sym_ggu(), Sym::No);
    }

    #[test]
    fn secondary_state_setters_work() {
        let mut system = System::new(3, Some(9), Sym::No, dummy_calc_gg, dummy_calc_jac).unwrap();

        system.set_backup_secondary_state(|_args: &mut u8| {});
        assert!(system.backup_secondary_state.is_some());

        system.set_restore_secondary_state(|_args: &mut u8| {});
        assert!(system.restore_secondary_state.is_some());

        system.set_prepare_to_iterate(|_args: &mut u8| {});
        assert!(system.prepare_to_iterate.is_some());

        system.set_update_secondary_state(
            |_do_backup: bool,
             _u0: &Vector,
             _u1: &Vector,
             _l0: f64,
             _l1: f64,
             _args: &mut u8|
             -> Result<bool, &'static str> { Ok(false) },
        );
        assert!(system.update_secondary_state.is_some());

        // exercise the closures for coverage
        (system.backup_secondary_state.as_ref().unwrap())(&mut 0u8);
        (system.restore_secondary_state.as_ref().unwrap())(&mut 0u8);
        (system.prepare_to_iterate.as_ref().unwrap())(&mut 0u8);
        let u = Vector::new(3);
        let _ = (system.update_secondary_state.as_ref().unwrap())(false, &u, &u, 0.0, 0.0, &mut 0u8);
    }
}
