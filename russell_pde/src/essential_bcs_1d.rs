use crate::Side;
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
    pub(crate) periodic_along_x: bool,

    /// Holds the functions to compute essential boundary conditions (EBC)
    ///
    /// The function is `f(x) -> value`
    ///
    /// (2) → (Xmin, Xmax); corresponding to the 2 sides
    pub(crate) functions: Vec<Arc<dyn Fn(f64) -> f64 + Send + Sync + 'a>>,

    /// Holds the sides where essential boundary conditions are applied
    pub(crate) sides: [bool; 2], // Xmin, Xmax
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
            sides: [false; 2],
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
        self.sides[index] = true;
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
        self.sides[0] = true;
        self.sides[1] = true;
    }
}
