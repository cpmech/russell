use crate::{Grid1d, Side};
use std::sync::Arc;

/// Manages natural (Neumann) boundary conditions for 1D PDE problems
///
/// Natural boundary conditions (NBCs), also known as Neumann boundary conditions,
/// specify the flux or derivative of the solution at the domain boundaries. For a 1D domain,
/// these conditions take the form:
///
/// * At x = xₘᵢₙ: wₙ(xₘᵢₙ) = f(xₘᵢₙ)
/// * At x = xₘₐₓ: wₙ(xₘₐₓ) = f(xₘₐₓ)
///
/// where wₙ is the normal component of the flux and f is a user-defined function.
///
/// # Flux Convention
///
/// The flux vector is defined as:
///
/// ```text
/// →         →
/// w = - ḵ · ∇ϕ
/// ```
///
/// The normal component of the flux crossing a boundary is denoted by:
///
/// ```text
///      →   →
/// wₙ = w · n̂
/// ```
///
/// where n̂ is the unit outward normal vector on the boundary.
///
/// A **positive** flux value indicates flow **leaving** the domain (outward direction).
///
/// # Notes
///
/// * Natural BCs cannot coexist with essential BCs on the same side
/// * See [EssentialBcs1d](crate::EssentialBcs1d) for Dirichlet boundary conditions
///
/// # Examples
///
/// ```
/// use russell_pde::{NaturalBcs1d, Side};
///
/// let mut nbcs = NaturalBcs1d::new();
///
/// // Insulated boundary (zero flux)
/// nbcs.set(Side::Xmin, |_x| 0.0);
///
/// // Constant outward flux
/// nbcs.set(Side::Xmax, |_x| 10.0);
///
/// // Spatially-varying flux
/// nbcs.set(Side::Xmin, |x| x * 5.0);
/// ```
pub struct NaturalBcs1d<'a> {
    /// Functions to compute natural boundary condition flux values
    ///
    /// Each function has the signature `f(x) -> flux` where:
    /// * `x` is the spatial coordinate at the boundary
    /// * `flux` is the normal flux component wₙ at that location
    ///   * Positive values: flux leaving the domain (outward)
    ///   * Negative values: flux entering the domain (inward)
    ///
    /// The vector contains 2 functions corresponding to [Xmin, Xmax] boundaries.
    pub(crate) functions: Vec<Arc<dyn Fn(f64) -> f64 + Send + Sync + 'a>>,

    /// Flags indicating which sides have natural boundary conditions
    ///
    /// Array of 2 booleans: [Xmin, Xmax]
    /// * `true`: Natural boundary condition is active on this side
    /// * `false`: No natural boundary condition on this side
    pub(crate) sides: [bool; 2], // Xmin, Xmax
}

impl<'a> NaturalBcs1d<'a> {
    /// Creates a new instance with no boundary conditions set
    ///
    /// # Returns
    ///
    /// Returns a `NaturalBcs1d` instance with:
    /// * No active natural boundary conditions
    /// * Default zero-valued functions for all sides
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::NaturalBcs1d;
    ///
    /// let nbcs = NaturalBcs1d::new();
    /// // Initially, no boundary conditions are active
    /// ```
    pub fn new() -> Self {
        NaturalBcs1d {
            functions: vec![
                Arc::new(|_| 0.0), // xmin
                Arc::new(|_| 0.0), // xmax
            ],
            sides: [false; 2],
        }
    }

    /// Sets a flux boundary condition
    ///
    /// The boundary condition is defined as:
    ///
    /// ```text
    /// wₙ = f(x) = q̄
    /// ```
    ///
    /// where a **positive** value of f(x) indicates a flux **leaving** the domain.
    /// It is worth noting that this convention is opposite to the one commonly used
    /// in the literature. The convention here is such that a positive flux is pointing
    /// in the same direction as the outward normal vector on the boundary.
    ///
    /// The function is `f(x) -> q̄`
    ///
    /// # Panics
    ///
    /// A panic may occur if an invalid side is provided for a 1D grid. It must be
    /// either `Side::Xmin` or `Side::Xmax`.
    ///
    /// # Theory
    ///
    /// The flux vector is defined by:
    ///
    /// ```text
    /// →         →
    /// w = - ḵ · ∇ϕ
    /// ```
    ///
    /// The normal component of the flux crossing a boundary is denoted by:
    ///
    /// ```text
    ///      →   →
    /// wₙ = w · n̂
    /// ```
    ///
    /// where n̂ is the unit outward normal vector on the boundary.
    ///
    /// In 1D, the flux vector reduces to `w = [wx, 0]ᵀ`, where
    ///
    /// ```text
    /// wx = -kx ∂ϕ/∂x
    /// ```
    ///
    /// The normal vectors at the boundaries are illustrated below:
    ///
    /// ```text
    /// @ Xmin:                                     @ Xmax:
    ///      ┌    ┐   ┌    ┐                             ┌    ┐   ┌    ┐
    ///      │ wx │   │ -1 │    ┌──────────────┐         │ wx │   │  1 │
    /// wₙ = │    │ · │    │  ← │              │ →  wₙ = │    │ · │    │
    ///      │  0 │   │  0 │    └──────────────┘         │  0 │   │  0 │
    ///      └    ┘   └    ┘                             └    ┘   └    ┘
    ///    = kx ∂ϕ/∂x                                  = -kx ∂ϕ/∂x
    /// ```
    pub fn set(&mut self, side: Side, f: impl Fn(f64) -> f64 + Send + Sync + 'a) {
        let index = side as usize;
        self.functions[index] = Arc::new(f);
        self.sides[index] = true;
    }

    /// Checks if a natural boundary condition is active at a given node index
    ///
    /// This method determines whether a natural BC should be applied at a specific
    /// grid node based on its position.
    ///
    /// # Input
    ///
    /// * `m` - Grid node index to check
    /// * `grid` - Reference to the 1D grid
    ///
    /// # Returns
    ///
    /// * `true` if a natural BC is active at node `m`
    /// * `false` if no natural BC is active (interior node or no BC set)
    ///
    /// # Notes
    ///
    /// * Returns `true` for node 0 if Xmin has an NBC
    /// * Returns `true` for node (nx-1) if Xmax has an NBC
    /// * Returns `false` for all interior nodes
    pub(crate) fn enabled_m(&self, m: usize, grid: &Grid1d) -> bool {
        if m == 0 {
            // Xmin
            self.sides[0]
        } else if m == grid.nx() - 1 {
            // Xmax
            self.sides[1]
        } else {
            // Interior
            false
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NaturalBcs1d;
    use crate::{Grid1d, Side};

    #[test]
    fn new_works() {
        let natural_bcs = NaturalBcs1d::new();
        assert_eq!(natural_bcs.sides, [false, false]);
        assert_eq!((natural_bcs.functions[0])(0.0), 0.0);
        assert_eq!((natural_bcs.functions[1])(0.0), 0.0);
    }

    #[test]
    fn set_works() {
        let mut natural_bcs = NaturalBcs1d::new();
        natural_bcs.set(Side::Xmin, |_| -10.0);
        natural_bcs.set(Side::Xmax, |_| 10.0);
        assert_eq!(natural_bcs.sides, [true, true]);
        assert_eq!((natural_bcs.functions[0])(0.0), -10.0);
        assert_eq!((natural_bcs.functions[1])(0.0), 10.0);
    }

    #[test]
    #[should_panic]
    fn set_panics_on_invalid_side() {
        let mut natural_bcs = NaturalBcs1d::new();
        natural_bcs.set(Side::Ymin, |_| 0.0);
    }

    #[test]
    fn enabled_m_works() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 3).unwrap();
        let mut natural_bcs = NaturalBcs1d::new();
        natural_bcs.set(Side::Xmin, |_| -10.0);
        assert!(natural_bcs.enabled_m(0, &grid));
        assert!(!natural_bcs.enabled_m(1, &grid));
        assert!(!natural_bcs.enabled_m(2, &grid));

        natural_bcs.set(Side::Xmax, |_| 10.0);
        assert!(natural_bcs.enabled_m(0, &grid));
        assert!(!natural_bcs.enabled_m(1, &grid));
        assert!(natural_bcs.enabled_m(2, &grid));
    }
}
