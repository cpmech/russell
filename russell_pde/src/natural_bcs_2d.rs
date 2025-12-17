use crate::{Grid2d, Side};
use std::collections::{HashMap, HashSet};
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
    functions: Vec<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'a>>,

    /// Holds the sides where natural boundary conditions are applied
    sides: HashSet<Side>,

    /// Indicates whether the structure is built and ready to use
    ready: bool,

    /// Maps node to one of the four functions in `functions`
    ///
    /// length = number of nodes with natural boundary conditions
    node_to_function: HashMap<usize, Side>,
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
            sides: HashSet::new(),
            ready: false,
            node_to_function: HashMap::new(),
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
    ///       →
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
    ///             ↑
    ///    ┌─────────────────┐
    ///    │                 │
    ///    │                 │
    ///    │                 │
    ///  ← │                 │ →
    ///    │                 │
    ///    │                 │
    ///    │                 │
    ///    └─────────────────┘
    ///             ↓
    /// ```
    pub fn set_flux(&mut self, grid: &Grid2d, side: Side, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'a) {
        let index = side as usize;
        self.functions[index] = Arc::new(f);
        self.sides.insert(side);
        self.ready = false;
    }

    // --------------------------------------------------------
    // crate
    // --------------------------------------------------------

    /// Builds the internal structures
    ///
    /// Returns the list of boundary nodes with NBCs
    pub(crate) fn build(&mut self, grid: &Grid2d) -> Vec<usize> {
        assert_eq!(self.ready, false, "can only build once");
        let nx = grid.nx();
        let ny = grid.ny();
        let mut nodes_set = HashSet::with_capacity(2 * nx + 2 * ny);
        for &side in &self.sides {
            for &m in grid.get_nodes_on_side(side) {
                self.node_to_function.insert(m, side);
                nodes_set.insert(m);
            }
        }
        self.ready = true;
        let mut nodes: Vec<_> = nodes_set.iter().copied().collect();
        nodes.sort();
        nodes
    }

    /// Returns the NBC value
    ///
    /// # Panics
    ///
    /// A panic may occur if the index is out of bounds.
    pub(crate) fn get_value(&self, m: usize, x: f64, y: f64) -> f64 {
        assert!(self.ready, "build must be called first");
        let side = *self.node_to_function.get(&m).unwrap();
        (self.functions[side as usize])(x, y)
    }

    /// Returns the list of nodes on all sides with NBCs
    pub(crate) fn get_nodes(&self) -> Vec<usize> {
        assert!(self.ready, "build must be called first");
        self.node_to_function.keys().copied().collect()
    }
}
