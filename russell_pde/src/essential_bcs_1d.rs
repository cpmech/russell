use crate::StrError;
use crate::{Grid1d, NaturalBcs1d, Side};
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

    /// Makes sure that all sides have either EBC or NBC, but not both
    pub(crate) fn validate(&self, nbcs: &NaturalBcs1d) -> Result<(), StrError> {
        if self.sides[0] && nbcs.sides[0] {
            return Err("Xmin side must not have both EBC and NBC");
        }
        if self.sides[1] && nbcs.sides[1] {
            return Err("Xmax side must not have both EBC and NBC");
        }
        if self.periodic_along_x {
            if nbcs.sides[0] || nbcs.sides[1] {
                return Err("Periodic X does not allow NBC on Xmin or Xmax");
            }
        } else {
            if !self.sides[0] && !nbcs.sides[0] {
                return Err("Xmin side is missing either EBC or NBC");
            }
            if !self.sides[1] && !nbcs.sides[1] {
                return Err("Xmax side is missing either EBC or NBC");
            }
        }
        Ok(())
    }

    /// Returns the nodes with EBCs
    pub(crate) fn get_nodes(&self, grid: &Grid1d) -> Vec<usize> {
        let mut nodes = Vec::new();
        for side in 0..2 {
            if self.sides[side] {
                let m = if side == 0 { 0 } else { grid.nx() - 1 };
                nodes.push(m);
            }
        }
        nodes
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {}
