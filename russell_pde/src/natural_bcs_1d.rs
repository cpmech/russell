use crate::{Grid1d, Side};
use std::sync::Arc;

/// Implements a handler for natural (Neumann) boundary conditions
///
/// This struct helps to manage natural boundary conditions (NBC) for 1D problems.
/// It holds the number of prescribed equations and the number of unknown equations.
///
/// The grid is assumed to be a regular Cartesian grid with `nx` points along x.
pub struct NaturalBcs1d<'a> {
    /// Holds the functions to compute natural boundary conditions (NBC)
    ///
    /// The function is `f(x) -> value`
    ///
    /// (2) → (xmin, xmax); corresponding to the 2 sides
    pub(crate) functions: Vec<Arc<dyn Fn(f64) -> f64 + Send + Sync + 'a>>,

    /// Holds the sides where natural boundary conditions are applied
    pub(crate) sides: [bool; 2], // Xmin, Xmax
}

impl<'a> NaturalBcs1d<'a> {
    /// Allocates a new instance
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

    /// Indicates whether a natural boundary condition is enabled at the given m location
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
mod tests {}
