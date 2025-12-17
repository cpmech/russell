use crate::{Grid2d, Side};
use std::collections::HashMap;
use std::sync::Arc;

/// Implements a handler for natural (Neumann) boundary conditions
///
/// This struct helps to manage natural boundary conditions (ebc) for 2D problems.
/// It holds the number of prescribed equations and the number of unknown equations.
///
/// The grid is assumed to be a regular Cartesian grid with `nx` points along x and `ny` points along y.
pub struct NaturalBcs2d<'a> {
    /// Holds the functions to compute natural boundary conditions (ebc)
    ///
    /// The function is `f(x, y) -> ebc`
    ///
    /// (4) → (xmin, xmax, ymin, ymax); corresponding to the 4 sides
    functions: Vec<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'a>>,

    /// Maps node to one of the four functions in `functions`
    ///
    /// length = number of nodes with natural boundary conditions (prescribed nodes)
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
            node_to_function: HashMap::new(),
        }
    }

    /// Sets natural (Neumann) boundary condition
    ///
    /// ```text
    ///  →    →
    ///  ∇ϕ · n̂ = f(x, y)  on side
    ///
    ///       →
    /// where n̂ is the unit outward normal vector on the boundary
    /// ```
    ///
    /// The function is `f(x, y) -> value`
    pub fn set_flux(&mut self, grid: &Grid2d, side: Side, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'a) {
        match side {
            Side::Xmin => {
                self.functions[0] = Arc::new(f);
                grid.for_each_node_xmin(|n| {
                    self.node_to_function.insert(*n, side);
                });
            }
            Side::Xmax => {
                self.functions[1] = Arc::new(f);
                grid.for_each_node_xmax(|n| {
                    self.node_to_function.insert(*n, side);
                });
            }
            Side::Ymin => {
                self.functions[2] = Arc::new(f);
                grid.for_each_node_ymin(|n| {
                    self.node_to_function.insert(*n, side);
                });
            }
            Side::Ymax => {
                self.functions[3] = Arc::new(f);
                grid.for_each_node_ymax(|n| {
                    self.node_to_function.insert(*n, side);
                });
            }
        };
    }

    /// Indicates whether the given node has a natural boundary condition
    pub fn has_value(&self, m: usize) -> bool {
        self.node_to_function.contains_key(&m)
    }

    /// Returns the natural boundary condition for the given node
    ///
    /// Returns `(side, value)`
    ///
    /// # Panics
    ///
    /// A panic may occur if the index is out of bounds.
    pub fn get_value(&self, m: usize, x: f64, y: f64) -> (Side, f64) {
        let side = *self.node_to_function.get(&m).unwrap();
        let index = side as usize;
        let value = (self.functions[index])(x, y);
        (side, value)
    }
}
