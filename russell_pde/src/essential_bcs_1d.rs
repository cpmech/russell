use crate::{Grid1d, Side};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Implements a handler for essential (Dirichlet) boundary conditions
///
/// This struct helps to manage essential boundary conditions (EBC) for 1D problems.
/// It holds the number of prescribed equations and the number of unknown equations.
///
/// The grid is assumed to be a regular Cartesian grid with `nx` points along x.
pub struct EssentialBcs1d<'a> {
    /// Indicates that the boundary is periodic along x (left ϕ values equal right ϕ values)
    ///
    /// If false, the left/right boundaries are zero-flux (Neumann with ∂ϕ/dx = 0)
    periodic_along_x: bool,

    /// Holds the functions to compute essential boundary conditions (EBC)
    ///
    /// The function is `f(x) -> value`
    ///
    /// (2) → (Xmin, Xmax); corresponding to the 2 sides
    functions: Vec<Arc<dyn Fn(f64) -> f64 + Send + Sync + 'a>>,

    /// Holds the sides where essential boundary conditions are applied
    sides: HashSet<Side>,

    /// Indicates whether the structure is built and ready to use
    ready: bool,

    /// Maps node to one of the two functions in `functions`
    ///
    /// length = number of nodes with essential boundary conditions (prescribed nodes)
    node_to_function: HashMap<usize, Side>,
}

impl<'a> EssentialBcs1d<'a> {
    /// Allocates a new instance
    pub fn new() -> Self {
        EssentialBcs1d {
            periodic_along_x: false,
            functions: vec![
                Arc::new(|_| 0.0), // Xmin
                Arc::new(|_| 0.0), // Xmax
            ],
            sides: HashSet::new(),
            ready: false,
            node_to_function: HashMap::new(),
        }
    }

    // --------------------------------------------------------
    // setters
    // --------------------------------------------------------

    /// Sets periodic boundary condition
    ///
    /// **Note:** Any essential boundary condition on the corresponding side will be removed.
    pub fn set_periodic(&mut self, along_x: bool) {
        self.periodic_along_x = along_x;
        self.ready = false;
    }

    /// Sets essential (Dirichlet) boundary condition
    ///
    /// The function is `f(x) -> value`
    ///
    /// **Note:** Any periodic boundary condition on the corresponding side will be removed.
    ///
    /// # Panics
    ///
    /// A panic may occur if an invalid side is provided for a 1D grid. It must be
    /// either `Side::Xmin` or `Side::Xmax`.
    pub fn set(&mut self, side: Side, f: impl Fn(f64) -> f64 + Send + Sync + 'a) {
        self.periodic_along_x = false;
        let index = side as usize;
        self.functions[index] = Arc::new(f);
        self.sides.insert(side);
        self.ready = false;
    }

    /// Sets homogeneous boundary conditions (i.e., zero essential values at the borders)
    ///
    /// **Note:** Periodic boundary conditions will be removed.
    pub fn set_homogeneous(&mut self) {
        self.periodic_along_x = false;
        self.functions = vec![
            Arc::new(|_| 0.0), // Xmin
            Arc::new(|_| 0.0), // Xmax
        ];
        self.sides.insert(Side::Xmin);
        self.sides.insert(Side::Xmax);
        self.ready = false;
    }

    // --------------------------------------------------------
    // crate
    // --------------------------------------------------------

    /// Builds the internal structures
    ///
    /// Returns the list of boundary nodes with EBCs
    pub(crate) fn build(&mut self, grid: &Grid1d) -> Vec<usize> {
        assert_eq!(self.ready, false, "can only build once");
        let mut nodes_set = HashSet::with_capacity(2);
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

    /// Indicates whether the boundary conditions are periodic along x
    pub(crate) fn is_periodic_along_x(&self) -> bool {
        assert!(self.ready, "build must be called first");
        self.periodic_along_x
    }

    /// Returns the EBC value
    ///
    /// # Panics
    ///
    /// A panic may occur if the index is out of bounds.
    pub(crate) fn get_value(&self, m: usize, x: f64) -> f64 {
        assert!(self.ready, "build must be called first");
        let index = *self.node_to_function.get(&m).unwrap() as usize;
        (self.functions[index])(x)
    }

    /// Returns the list of nodes on all sides with EBCs
    pub(crate) fn get_nodes(&self) -> Vec<usize> {
        assert!(self.ready, "build must be called first");
        self.node_to_function.keys().copied().collect()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {}
