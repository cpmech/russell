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
pub struct System<'a, A> {
    /// Dimension of `u` and `G`
    pub(crate) ndim: usize,

    /// Calculates the function G(u) or G(u, λ) or G(u(s), λ(s))
    ///
    /// The function is `fn (gg, λ, u, args)` where λ can be ignored for the simple case.
    pub(crate) calc_gg: Arc<dyn Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a>,

    /// Calculates the Gu = dG/du derivative
    ///
    /// The function is `fn (ggu, λ, u, args)` where λ can be ignored for the simple case.
    ///
    /// **Note:** there is no need to call `reset` on the `CooMatrix` object because this is done already.
    pub(crate) calc_ggu:
        Option<Arc<dyn Fn(&mut CooMatrix, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a>>,

    /// Calculates the Gλ = dG/dλ derivative
    ///
    /// The function is `fn (ggl, λ, u, args)`
    pub(crate) calc_ggl:
        Option<Arc<dyn Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a>>,

    /// Number of non-zeros in the Gu matrix
    pub(crate) nnz_ggu: usize,

    /// Symmetric type of the Gu matrix
    pub(crate) sym_ggu: Sym,

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
    /// The function is `fn (do_backup, u0, u1, args) -> stop_gracefully` with `u0` being the
    /// value at the beginning of the step and `u1` the value at the updated step.
    pub(crate) update_secondary_state:
        Option<Arc<dyn Fn(bool, &Vector, &Vector, &mut A) -> Result<bool, StrError> + Send + Sync + 'a>>,
}

impl<'a, A> System<'a, A> {
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
        Ok(System {
            ndim,
            calc_gg: Arc::new(calc_gg),
            calc_ggu: None,
            calc_ggl: None,
            nnz_ggu: ndim * ndim,
            sym_ggu: Sym::No,
            backup_secondary_state: None,
            restore_secondary_state: None,
            prepare_to_iterate: None,
            update_secondary_state: None,
        })
    }

    /// Returns a copy of this struct
    pub fn clone(&self) -> Self {
        System {
            ndim: self.ndim,
            calc_gg: self.calc_gg.clone(),
            calc_ggu: self.calc_ggu.clone(),
            calc_ggl: self.calc_ggl.clone(),
            nnz_ggu: self.nnz_ggu,
            sym_ggu: self.sym_ggu,
            backup_secondary_state: self.backup_secondary_state.clone(),
            restore_secondary_state: self.restore_secondary_state.clone(),
            prepare_to_iterate: self.prepare_to_iterate.clone(),
            update_secondary_state: self.update_secondary_state.clone(),
        }
    }

    /// Sets a function to calculate the `Gu = dG/du` matrix
    ///
    /// Use `|ggu_or_aa, λ, u, args|` or `|ggu_or_aa: &mut CooMatrix, l: f64, u: &Vector, args: &mut A|`
    ///
    /// **Important:** If `bordering = true`, then `ggu_or_aa` is called with the actual `Gu` matrix.
    /// Otherwise, the function may be called with either the `Gu` matrix or the `A` matrix, so we can
    /// build the system shown below:
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
    /// # Input
    ///
    /// * `nnz` -- the number of non-zeros in the Gu matrix; use None to indicate a dense matrix with:
    ///     * `nnz = (ndim + ndim²) / 2` if triangular
    ///     * `nnz = ndim²` otherwise
    /// * `symmetric` -- specifies the symmetric type of the Gu matrix
    /// * `callback` -- the function to calculate the Gu matrix
    pub fn set_calc_ggu(
        &mut self,
        nnz: Option<usize>,
        symmetric: Sym,
        callback: impl Fn(&mut CooMatrix, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a,
    ) -> Result<&mut Self, StrError> {
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
        Ok(self)
    }

    /// Sets a function to calculate the `Gλ = dG/dλ` vector
    ///
    /// Use `|ggl, λ, u, args|` or `|ggl: &mut Vector, l: f64, u: &Vector, args: &mut A|`
    ///
    /// # Input
    ///
    /// * `callback` -- the function to calculate the Gλ vector
    pub fn set_calc_ggl(
        &mut self,
        callback: impl Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError> + Send + Sync + 'a,
    ) -> &mut Self {
        self.calc_ggl = Some(Arc::new(callback));
        self
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
    /// The function is `fn (do_backup, u0, u1, args) -> stop_gracefully` with `u0` being the
    /// value at the beginning of the step and `u1` the value at the updated step.
    pub fn set_update_secondary_state(
        &mut self,
        callback: impl Fn(bool, &Vector, &Vector, &mut A) -> Result<bool, StrError> + Send + Sync + 'a,
    ) -> &mut Self {
        self.update_secondary_state = Some(Arc::new(callback));
        self
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
