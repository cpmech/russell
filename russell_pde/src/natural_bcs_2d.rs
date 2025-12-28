use crate::{Grid2d, Side};
use std::sync::Arc;

/// Implements a handler for natural (Neumann) boundary conditions
///
/// This struct helps to manage natural boundary conditions (NBC) for 2D problems.
/// It holds the number of prescribed equations and the number of unknown equations.
///
/// The grid is assumed to be a regular Cartesian grid with `nx` points along x and `ny` points along y.
pub struct NaturalBcs2d<'a> {
    /// Holds the functions to compute natural boundary conditions (NBC)
    ///
    /// The function is `f(x, y) -> value`
    ///
    /// (4) → (xmin, xmax, ymin, ymax); corresponding to the 4 sides
    pub(crate) functions: Vec<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'a>>,

    /// Holds the sides where natural boundary conditions are applied
    pub(crate) sides: [bool; 4], // Xmin, Xmax, Ymin, Ymax
}

impl<'a> NaturalBcs2d<'a> {
    /// Allocates a new instance
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
    pub fn set(&mut self, side: Side, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'a) {
        let index = side as usize;
        self.functions[index] = Arc::new(f);
        self.sides[index] = true;
    }

    /// Indicates whether a natural boundary condition is enabled at the given (i, j) location
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
