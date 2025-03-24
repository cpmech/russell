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
/// Simple:     G(u) = 0
/// Parametric: G(u, λ) = 0
/// Arclength:  G(u(s), λ(s)) = 0
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

    /// Calculates the function `G(u)` or `G(u, λ)` or `G(u(s), λ(s))`
    ///
    /// The function is `fn (gg, u, λ, s, args)` where `λ` and `s` can be ignored for the simple case.
    pub(crate) calc_gg: Arc<dyn Fn(&mut Vector, &Vector, f64, f64, &mut A) -> Result<(), StrError> + Send + Sync + 'a>,

    /// Calculates the `Gu = dG/du` derivative
    ///
    /// The function is `fn (ggu, u, λ, s, args)` where `λ` and `s` can be ignored for the simple case.
    pub(crate) calc_ggu:
        Option<Arc<dyn Fn(&mut CooMatrix, &Vector, f64, f64, &mut A) -> Result<(), StrError> + Send + Sync + 'a>>,

    /// Calculates the `Gλ = dG/dλ` derivative
    ///
    /// The function is `fn (ggl, u, λ, s, args)` where `λ` and `s` can be ignored for the simple case.
    pub(crate) calc_ggl:
        Option<Arc<dyn Fn(&mut Vector, &Vector, f64, f64, &mut A) -> Result<(), StrError> + Send + Sync + 'a>>,

    /// Number of non-zeros in the Gu matrix
    pub(crate) nnz_ggu: usize,

    /// Symmetric type of the Gu matrix
    pub(crate) sym_ggu: Sym,
}

impl<'a, A> NlSystem<'a, A> {
    /// Allocates a new instance
    pub fn new(
        ndim: usize,
        calc_gg: impl Fn(&mut Vector, &Vector, f64, f64, &mut A) -> Result<(), StrError> + Send + Sync + 'a,
    ) -> Self {
        NlSystem {
            ndim,
            calc_gg: Arc::new(calc_gg),
            calc_ggu: None,
            calc_ggl: None,
            nnz_ggu: ndim * ndim,
            sym_ggu: Sym::No,
        }
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
        }
    }

    /// Sets a function to calculate the `Gu = dG/du` matrix
    ///
    /// Use `|ggu, u, λ, s, args|` or `|ggu: &mut CooMatrix, u: &Vector, l: f64, s: args: &mut A|`
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
        callback: impl Fn(&mut CooMatrix, &Vector, f64, f64, &mut A) -> Result<(), StrError> + Send + Sync + 'a,
    ) -> Result<(), StrError> {
        self.nnz_ggu = if let Some(value) = nnz {
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
