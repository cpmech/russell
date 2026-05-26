use crate::{Grid2d, Side};
use std::sync::Arc;

/// Manages natural (Neumann) boundary conditions for 2D PDE problems
///
/// Natural boundary conditions (NBCs), also known as Neumann boundary conditions,
/// specify the flux or derivative of the solution at the domain boundaries. For a 2D domain,
/// these conditions take the form:
///
/// * At `x = xₘᵢₙ`: `wₙ(xₘᵢₙ, y) = f(xₘᵢₙ, y)`
/// * At `x = xₘₐₓ`: `wₙ(xₘₐₓ, y) = f(xₘₐₓ, y)`
/// * At `y = yₘᵢₙ`: `wₙ(x, yₘᵢₙ) = f(x, yₘᵢₙ)`
/// * At `y = yₘₐₓ`: `wₙ(x, yₘₐₓ) = f(x, yₘₐₓ)`
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
/// **Important:** This convention is opposite to what is commonly found in some literature.
///
/// # Notes
///
/// * Natural BCs cannot coexist with essential BCs on the same side
/// * See [EssentialBcs2d](crate::EssentialBcs2d) for Dirichlet boundary conditions
///
/// # Examples
///
/// ```
/// use russell_pde::{NaturalBcs2d, Side};
///
/// let mut nbcs = NaturalBcs2d::new();
///
/// // Insulated boundaries (zero flux)
/// nbcs.set(Side::Xmin, |_x, _y| 0.0);
/// nbcs.set(Side::Xmax, |_x, _y| 0.0);
///
/// // Constant outward flux at top
/// nbcs.set(Side::Ymax, |_x, _y| 10.0);
///
/// // Spatially-varying flux at bottom
/// nbcs.set(Side::Ymin, |x, y| x * y * 5.0);
/// ```
pub struct NaturalBcs2d<'a> {
    /// Functions to compute natural boundary condition flux values
    ///
    /// Each function has the signature `f(x, y) -> flux` where:
    /// * `x`, `y` are the spatial coordinates at the boundary
    /// * `flux` is the normal flux component wₙ at that location
    ///   * Positive values: flux leaving the domain (outward)
    ///   * Negative values: flux entering the domain (inward)
    ///
    /// The vector contains 4 functions corresponding to [Xmin, Xmax, Ymin, Ymax] boundaries.
    pub(crate) functions: Vec<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'a>>,

    /// Flags indicating which sides have natural boundary conditions
    ///
    /// Array of 4 booleans: [Xmin, Xmax, Ymin, Ymax]
    /// * `true`: Natural boundary condition is active on this side
    /// * `false`: No natural boundary condition on this side
    pub(crate) sides: [bool; 4], // Xmin, Xmax, Ymin, Ymax
}

impl<'a> NaturalBcs2d<'a> {
    /// Creates a new instance with no boundary conditions set
    ///
    /// # Returns
    ///
    /// Returns a `NaturalBcs2d` instance with:
    /// * No active natural boundary conditions
    /// * Default zero-valued functions for all sides
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::NaturalBcs2d;
    ///
    /// let nbcs = NaturalBcs2d::new();
    /// // Initially, no boundary conditions are active
    /// ```
    pub fn new() -> Self {
        NaturalBcs2d {
            functions: vec![
                Arc::new(|_, _| 0.0), // xmin
                Arc::new(|_, _| 0.0), // xmax
                Arc::new(|_, _| 0.0), // ymin
                Arc::new(|_, _| 0.0), // ymax
            ],
            sides: [false; 4],
        }
    }

    /// Sets a natural (Neumann) boundary condition specifying the flux at a boundary
    ///
    /// # Input
    ///
    /// * `side` - The boundary side (`Side::Xmin`, `Side::Xmax`, `Side::Ymin`, or `Side::Ymax`)
    /// * `f` - Function with signature `f(x, y) -> flux` that computes the normal flux value
    ///   * `x`, `y`: Spatial coordinates at the boundary
    ///   * `flux`: Normal flux component wₙ = f(x, y) = q̄
    ///
    /// # Flux Convention
    ///
    /// A **positive** value of f(x, y) indicates a flux **leaving** the domain. This convention
    /// ensures that the flux direction aligns with the outward normal vector on the boundary.
    ///
    /// **Important:** This convention is opposite to what is commonly found in some literature.
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
    /// In 2D, the flux vector reduces to `w = [wx, wy]ᵀ`, where
    ///
    /// ```text
    /// wx = -kx ∂ϕ/∂x
    /// wy = -ky ∂ϕ/∂y
    /// ```
    ///
    /// The normal vectors at the boundaries are illustrated below:
    ///
    /// ```text
    ///                          @ Ymax:
    ///                               ┌    ┐   ┌    ┐
    ///                               │ wx │   │  0 │
    ///                          wₙ = │    │ · │    │
    ///                               │ wy │   │  1 │
    ///                               └    ┘   └    ┘
    ///                             = -ky ∂ϕ/∂y
    ///
    ///                                   ↑
    /// @ Xmin:                  ┌─────────────────┐     @ Xmax:
    ///      ┌    ┐   ┌    ┐     │                 │          ┌    ┐   ┌    ┐
    ///      │ wx │   │ -1 │     │                 │          │ wx │   │  1 │
    /// wₙ = │    │ · │    │     │                 │     wₙ = │    │ · │    │
    ///      │ wy │   │  0 │   ← │                 │ →        │ wy │   │  0 │
    ///      └    ┘   └    ┘     │                 │          └    ┘   └    ┘
    ///    = kx ∂ϕ/∂x            │                 │        = -kx ∂ϕ/∂x
    ///                          │                 │
    ///                          └─────────────────┘
    ///                                   ↓
    ///
    ///                          @ Ymin:
    ///                               ┌    ┐   ┌    ┐
    ///                               │ wx │   │  0 │
    ///                          wₙ = │    │ · │    │
    ///                               │ wy │   │ -1 │
    ///                               └    ┘   └    ┘
    ///                             = ky ∂ϕ/∂y
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::{NaturalBcs2d, Side};
    ///
    /// let mut nbcs = NaturalBcs2d::new();
    ///
    /// // Insulated boundary (zero flux)
    /// nbcs.set(Side::Xmin, |_x, _y| 0.0);
    ///
    /// // Constant heat flux entering the domain
    /// nbcs.set(Side::Xmax, |_x, _y| -5.0);
    ///
    /// // Spatially-varying flux
    /// nbcs.set(Side::Ymin, |x, y| 2.0 * x * y);
    /// ```
    pub fn set(&mut self, side: Side, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'a) {
        let index = side as usize;
        self.functions[index] = Arc::new(f);
        self.sides[index] = true;
    }

    /// Checks if a natural boundary condition is active at a given grid location
    ///
    /// This method determines whether a natural BC should be applied at a specific
    /// grid node based on its position in the 2D grid.
    ///
    /// # Input
    ///
    /// * `i` - Grid index in the x-direction
    /// * `j` - Grid index in the y-direction
    /// * `grid` - Reference to the 2D grid
    ///
    /// # Returns
    ///
    /// * `true` if a natural BC is active at location (i, j)
    /// * `false` if no natural BC is active (interior node or no BC set)
    ///
    /// # Notes
    ///
    /// * Edge nodes: Returns `true` if the corresponding side has an NBC
    /// * Corner nodes: Returns `true` if either adjacent side has an NBC
    /// * Interior nodes: Always returns `false`
    pub(crate) fn enabled_ij(&self, i: usize, j: usize, grid: &Grid2d) -> bool {
        // edges
        if i == 0 {
            self.sides[0]
        } else if i == grid.nx() - 1 {
            self.sides[1]
        } else if j == 0 {
            self.sides[2]
        } else if j == grid.ny() - 1 {
            self.sides[3]
        }
        // corners
        else if i == 0 && j == 0 {
            self.sides[0] || self.sides[2]
        } else if i == 0 && j == grid.ny() - 1 {
            self.sides[0] || self.sides[3]
        } else if i == grid.nx() - 1 && j == 0 {
            self.sides[1] || self.sides[2]
        } else if i == grid.nx() - 1 && j == grid.ny() - 1 {
            self.sides[1] || self.sides[3]
        }
        // Interior
        else {
            false
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NaturalBcs2d;
    use crate::{Grid2d, Side};

    #[test]
    fn new_works() {
        let natural_bcs = NaturalBcs2d::new();
        assert_eq!(natural_bcs.sides, [false, false, false, false]);
        assert_eq!((natural_bcs.functions[0])(0.0, 0.0), 0.0);
        assert_eq!((natural_bcs.functions[1])(0.0, 0.0), 0.0);
        assert_eq!((natural_bcs.functions[2])(0.0, 0.0), 0.0);
        assert_eq!((natural_bcs.functions[3])(0.0, 0.0), 0.0);
    }

    #[test]
    fn set_works() {
        let mut natural_bcs = NaturalBcs2d::new();
        natural_bcs.set(Side::Xmin, |_, _| -10.0);
        natural_bcs.set(Side::Xmax, |_, _| 10.0);
        natural_bcs.set(Side::Ymin, |_, _| -20.0);
        natural_bcs.set(Side::Ymax, |_, _| 20.0);
        assert_eq!(natural_bcs.sides, [true, true, true, true]);
        assert_eq!((natural_bcs.functions[0])(0.0, 0.0), -10.0);
        assert_eq!((natural_bcs.functions[1])(0.0, 0.0), 10.0);
        assert_eq!((natural_bcs.functions[2])(0.0, 0.0), -20.0);
        assert_eq!((natural_bcs.functions[3])(0.0, 0.0), 20.0);
    }

    #[test]
    fn enabled_ij_works() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3).unwrap();
        let mut natural_bcs = NaturalBcs2d::new();
        natural_bcs.set(Side::Xmin, |_, _| -10.0);
        assert!(natural_bcs.enabled_ij(0, 0, &grid)); // corner
        assert!(natural_bcs.enabled_ij(0, 1, &grid));
        assert!(natural_bcs.enabled_ij(0, 2, &grid)); // corner
        assert!(!natural_bcs.enabled_ij(1, 0, &grid));
        assert!(!natural_bcs.enabled_ij(1, 1, &grid));
        assert!(!natural_bcs.enabled_ij(1, 2, &grid));
        assert!(!natural_bcs.enabled_ij(2, 0, &grid));
        assert!(!natural_bcs.enabled_ij(2, 1, &grid));
        assert!(!natural_bcs.enabled_ij(2, 2, &grid));

        natural_bcs.set(Side::Xmax, |_, _| 10.0);
        assert!(natural_bcs.enabled_ij(2, 0, &grid)); // corner
        assert!(natural_bcs.enabled_ij(2, 1, &grid));
        assert!(natural_bcs.enabled_ij(2, 2, &grid)); // corner

        natural_bcs.set(Side::Ymin, |_, _| -20.0);
        assert!(natural_bcs.enabled_ij(0, 0, &grid)); // corner
        assert!(natural_bcs.enabled_ij(1, 0, &grid));
        assert!(natural_bcs.enabled_ij(2, 0, &grid)); // corner

        natural_bcs.set(Side::Ymax, |_, _| 20.0);
        assert!(natural_bcs.enabled_ij(0, 2, &grid)); // corner
        assert!(natural_bcs.enabled_ij(1, 2, &grid));
        assert!(natural_bcs.enabled_ij(2, 2, &grid)); // corner
    }
}
