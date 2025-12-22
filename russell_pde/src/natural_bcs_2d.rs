use crate::{Grid2d, Side};
use std::collections::HashSet;
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
    sides: [bool; 4], // Xmin, Xmax, Ymin, Ymax

    /// Indicates whether the structure is built and ready to use
    ready: bool,

    /// Maps node to one of the four functions in `functions`
    ///
    /// length = number of nodes with natural boundary conditions
    has_value: HashSet<usize>,
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
            ready: false,
            has_value: HashSet::new(),
        }
    }

    // --------------------------------------------------------
    // setters
    // --------------------------------------------------------

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
        self.ready = false;
    }

    // --------------------------------------------------------
    // crate
    // --------------------------------------------------------

    /// Builds the internal structures
    pub(crate) fn build(&mut self, grid: &Grid2d) {
        assert_eq!(self.ready, false, "can only build once");
        for index in 0..4 {
            if self.sides[index] {
                let side = Side::from_index(index);
                for &m in grid.get_nodes_on_side(side) {
                    self.has_value.insert(m);
                }
            }
        }
        self.ready = true;
    }

    /// Checks if a node has a NBC value
    pub(crate) fn has_value(&self, m: usize) -> bool {
        assert!(self.ready, "build must be called first");
        self.has_value.contains(&m)
    }
}
