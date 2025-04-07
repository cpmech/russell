use crate::StrError;
use russell_lab::Vector;
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
/// Here, we use `gg` to represent `G` because capital letters are const in Rust.
///
/// The required derivatives are:
///
/// ```text
/// ggu := Gu = dG/du
/// ggl := Gλ = dG/dλ
/// ```
pub struct NlSystem<'a, A> {
    /// Dimension of `u` and `G`
    pub(crate) ndim: usize,

    /// Calculates the function G(u) or G(u, λ) or G(u(s), λ(s))
    ///
    /// The function is `fn (gg, λ, u, args)` where λ can be ignored for the simple case.
    pub(crate) calc_gg: Arc<dyn Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a>,

    /// Calculates the Gu = dG/du derivative
    ///
    /// The function is `fn (ggu, λ, u, args)` where λ can be ignored for the simple case.
    pub(crate) calc_ggu:
        Option<Arc<dyn Fn(&mut CooMatrix, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a>>,

    /// Calculates the Gλ = dG/dλ derivative
    ///
    /// The function is `fn (ggl, λ, u, args)` where λ can be ignored for the simple case.
    pub(crate) calc_ggl:
        Option<Arc<dyn Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a>>,

    /// Number of non-zeros in the Gu matrix
    pub(crate) nnz_ggu: usize,

    /// Symmetric type of the Gu matrix
    pub(crate) sym_ggu: Sym,

    /// Creates a copy of external state variables
    ///
    /// The function is `fn (args)`
    pub(crate) step_backup_state: Option<Arc<dyn Fn(&mut A) + Send + Sync + 'a>>,

    /// Restores external state variables
    ///
    /// The function is `fn (args)`
    pub(crate) step_restore_state: Option<Arc<dyn Fn(&mut A) + Send + Sync + 'a>>,

    /// Prepares to iterate (e.g., reset algorithmic variables in the FEM)
    ///
    /// The function is `fn (args)`
    pub(crate) step_reset_algorithmic_variables: Option<Arc<dyn Fn(&mut A) + Send + Sync + 'a>>,

    /// Updates starred variables (e.g., FEM transient starred variables)
    ///
    /// The function is `fn (mdu, u_new, args)`
    pub(crate) iteration_update_starred: Option<Arc<dyn Fn(&Vector, &mut A) + Send + Sync + 'a>>,

    /// Prepares to update secondary variables (e.g., FEM stresses)
    ///
    /// The function is `fn (first_iteration, args)`
    pub(crate) iteration_prepare_to_update_secondary: Option<Arc<dyn Fn(bool, &mut A) + Send + Sync + 'a>>,

    /// Updates secondary variables (e.g., FEM stresses)
    ///
    /// The function is `fn (mdu, u_new, args)`
    pub(crate) iteration_update_secondary:
        Option<Arc<dyn Fn(&Vector, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a>>,
}

impl<'a, A> NlSystem<'a, A> {
    /// Allocates a new instance
    ///
    /// use `|gg, l, u, args|` or `|gg: &mut Vector, l: f64, u: &Vector, args: &mut A|`
    pub fn new(
        ndim: usize,
        calc_gg: impl Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a,
    ) -> Result<Self, StrError> {
        if ndim < 1 {
            return Err("ndim must be at least 1");
        }
        Ok(NlSystem {
            ndim,
            calc_gg: Arc::new(calc_gg),
            calc_ggu: None,
            calc_ggl: None,
            nnz_ggu: ndim * ndim,
            sym_ggu: Sym::No,
            step_backup_state: None,
            step_restore_state: None,
            step_reset_algorithmic_variables: None,
            iteration_update_starred: None,
            iteration_prepare_to_update_secondary: None,
            iteration_update_secondary: None,
        })
    }

    /// Returns a copy of this struct
    pub fn clone(&self) -> Self {
        NlSystem {
            ndim: self.ndim,
            calc_gg: self.calc_gg.clone(),
            calc_ggu: self.calc_ggu.clone(),
            calc_ggl: self.calc_ggl.clone(),
            nnz_ggu: self.nnz_ggu,
            sym_ggu: self.sym_ggu,
            step_reset_algorithmic_variables: self.step_reset_algorithmic_variables.clone(),
            step_backup_state: self.step_backup_state.clone(),
            step_restore_state: self.step_restore_state.clone(),
            iteration_update_starred: self.iteration_update_starred.clone(),
            iteration_prepare_to_update_secondary: self.iteration_prepare_to_update_secondary.clone(),
            iteration_update_secondary: self.iteration_update_secondary.clone(),
        }
    }

    /// Sets a function to calculate the `Gu = dG/du` matrix
    ///
    /// Use `|ggu, λ, u, args|` or `|ggu: &mut CooMatrix, l: f64, u: &Vector, args: &mut A|`
    ///
    /// # Input
    ///
    /// * `nnz` -- the number of non-zeros in the Gu matrix; use None to indicate a dense matrix with:
    ///     * `nnz = (ndim + ndim²) / 2` if triangular
    ///     * `nnz = ndim²` otherwise
    /// * `symmetric` -- specifies the symmetric type of the Gu matrix
    /// * `callback` -- the function to calculate the Jacobian matrix
    pub fn set_calc_ggu(
        &mut self,
        nnz: Option<usize>,
        symmetric: Sym,
        callback: impl Fn(&mut CooMatrix, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a,
    ) -> Result<(), StrError> {
        self.nnz_ggu = if let Some(value) = nnz {
            if value < 1 {
                return Err("nnz must be at least 1");
            }
            value
        } else {
            if symmetric.triangular() {
                (self.ndim + self.ndim * self.ndim) / 2
            } else {
                self.ndim * self.ndim
            }
        };
        self.sym_ggu = symmetric;
        self.calc_ggu = Some(Arc::new(callback));
        Ok(())
    }

    /// Returns the dimension of the ODE system
    pub fn get_ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the number of non-zero values in the Gu matrix
    pub fn get_nnz_ggu(&self) -> usize {
        self.nnz_ggu
    }
}
