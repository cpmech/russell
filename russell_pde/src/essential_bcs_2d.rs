use crate::{Grid2d, Side};
use std::collections::HashMap;
use std::sync::Arc;

/// Implements a handler for essential (Dirichlet) boundary conditions
///
/// This struct helps to manage essential boundary conditions (EBC) for 2D problems.
/// It holds the number of prescribed equations and the number of unknown equations.
///
/// The grid is assumed to be a regular Cartesian grid with `nx` points along x and `ny` points along y.
pub struct EssentialBcs2d<'a> {
    /// Indicates that the boundary is periodic along x (left ϕ values equal right ϕ values)
    ///
    /// If false, the left/right boundaries are zero-flux (Neumann with ∂ϕ/dx = 0)
    periodic_along_x: bool,

    /// Indicates that the boundary is periodic along x (bottom ϕ values equal top ϕ values)
    ///
    /// If false, the bottom/top boundaries are zero-flux (Neumann with ∂ϕ/dx = 0)
    periodic_along_y: bool,

    /// Holds the functions to compute essential boundary conditions (EBC)
    ///
    /// The function is `f(x, y) -> value`
    ///
    /// (4) → (Xmin, Xmax, Ymin, Ymax); corresponding to the 4 sides
    functions: Vec<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'a>>,

    /// Indicates the sides where essential boundary conditions are applied
    pub(crate) sides: [bool; 4], // Xmin, Xmax, Ymin, Ymax

    /// Indicates whether the structure is built and ready to use
    ready: bool,

    /// Maps node to one of the four functions in `functions`
    ///
    /// length = number of nodes with EBCs
    node_to_function: HashMap<usize, Side>,
}

impl<'a> EssentialBcs2d<'a> {
    /// Allocates a new instance
    pub fn new() -> Self {
        EssentialBcs2d {
            periodic_along_x: false,
            periodic_along_y: false,
            functions: vec![
                Arc::new(|_, _| 0.0), // Xmin
                Arc::new(|_, _| 0.0), // Xmax
                Arc::new(|_, _| 0.0), // Ymin
                Arc::new(|_, _| 0.0), // Ymax
            ],
            sides: [false; 4],
            ready: false,
            node_to_function: HashMap::new(),
        }
    }

    // --------------------------------------------------------
    // setters
    // --------------------------------------------------------

    /// Sets periodic boundary condition
    pub fn set_periodic(&mut self, along_x: bool, along_y: bool) {
        self.periodic_along_x = along_x;
        self.periodic_along_y = along_y;
        self.ready = false
    }

    /// Sets essential (Dirichlet) boundary condition
    ///
    /// The function is `f(x, y) -> value`
    ///
    /// **Note:** Any periodic boundary condition on the corresponding side will be removed.
    pub fn set(&mut self, side: Side, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'a) {
        match side {
            Side::Xmin | Side::Xmax => self.periodic_along_x = false,
            Side::Ymin | Side::Ymax => self.periodic_along_y = false,
        };
        let index = side as usize;
        self.functions[index] = Arc::new(f);
        self.sides[index] = true;
        self.ready = false;
    }

    /// Sets homogeneous boundary conditions (i.e., zero essential values at the borders)
    ///
    /// **Note:** Periodic boundary conditions will be removed.
    pub fn set_homogeneous(&mut self) {
        self.periodic_along_x = false;
        self.periodic_along_y = false;
        self.functions = vec![
            Arc::new(|_, _| 0.0), // Xmin
            Arc::new(|_, _| 0.0), // Xmax
            Arc::new(|_, _| 0.0), // Ymin
            Arc::new(|_, _| 0.0), // Ymax
        ];
        self.sides[0] = true;
        self.sides[1] = true;
        self.sides[2] = true;
        self.sides[3] = true;
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
                    self.node_to_function.insert(m, side);
                }
            }
        }
        self.ready = true;
    }

    /// Indicates whether the boundary conditions are periodic along x
    pub(crate) fn is_periodic_along_x(&self) -> bool {
        assert!(self.ready, "build must be called first");
        self.periodic_along_x
    }

    /// Indicates whether the boundary conditions are periodic along y
    pub(crate) fn is_periodic_along_y(&self) -> bool {
        assert!(self.ready, "build must be called first");
        self.periodic_along_y
    }

    /// Returns the EBC value
    ///
    /// # Panics
    ///
    /// A panic may occur if the index is out of bounds.
    pub(crate) fn get_value(&self, m: usize, x: f64, y: f64) -> f64 {
        assert!(self.ready, "build must be called first");
        let index = *self.node_to_function.get(&m).unwrap() as usize;
        (self.functions[index])(x, y)
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
