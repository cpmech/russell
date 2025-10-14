use crate::StrError;

/// Defines a 1D grid
///
/// ## Grid Layout and Indexing
///
/// ```text
///       0───────1───────2───────3───────4
///      i=0     i=1     i=2     i=3     i=4
///                                     nx=5
/// ```
///
/// ## Boundary Node Classification
///
/// The grid automatically identifies boundary nodes:
/// - **xmin edge**: Left boundary (i = 0)
/// - **xmax edge**: Right boundary (i = nx-1)  
///
/// ## Examples
///
/// ```rust
/// use russell_pde::{Grid1d, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // Create uniform grid
///     let grid = Grid1d::new_uniform(0.0, 1.0, 3)?;
///
///     // Grid layout:
///     //   0───1───2
///     // x=0.0 0.5 1.0
///
///     // Print all node coordinates
///     grid.for_each_coord(|m, x| {
///         println!("Node {}: {:.1}", m, x);
///     });
///     // Output:
///     // Node 0: (0.0)
///     // Node 1: (0.5)
///     // Node 2: (1.0)
///
///     // Collect coordinates into a vector
///     let mut coords = Vec::new();
///     grid.for_each_coord(|_m, x| coords.push(x));
///     Ok(())
/// }
/// ```
pub struct Grid1d {
    /// Number of points (≥ 2)
    ///
    /// This represents the number of columns in the grid.
    nx: usize,

    /// Node coordinates
    ///
    /// Length = nx
    coords: Vec<f64>,
}

impl Grid1d {
    /// Creates a new grid
    ///
    /// This constructor allows for non-uniform spacing by providing explicit
    /// coordinate arrays.
    ///
    /// # Arguments
    ///
    /// * `xx` - Array of x-coordinates (must be strictly increasing, length ≥ 2)
    ///
    /// # Returns
    ///
    /// A new `Grid1d` instance with `nx = xx.len()` points.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `xx.len() < 2`
    /// - `xx` array is not strictly increasing
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid1d, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // Non-uniform grid with refined spacing near boundaries
    ///     let xx = &[0.0, 0.1, 0.5, 0.9, 1.0];
    ///     let grid = Grid1d::new(xx)?;
    ///
    ///     // Grid layout:
    ///     //   0───1───2───3───4
    ///     // x=0.0 0.1 0.5 0.9 1.0
    ///
    ///     assert_eq!(grid.size(), 5);
    ///     Ok(())
    /// }
    /// ```
    pub fn new(xx: &[f64]) -> Result<Self, StrError> {
        let nx = xx.len();
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }
        for i in 1..nx {
            if xx[i] <= xx[i - 1] {
                return Err("xx must be strictly increasing");
            }
        }
        Ok(Self {
            nx,
            coords: xx.to_vec(),
        })
    }

    /// Creates a new grid with uniform spacing
    ///
    /// This constructor creates a structured grid with uniform spacing. The spacing
    /// is calculated automatically based on the domain size and number of points.
    ///
    /// # Arguments
    ///
    /// * `xmin` - Minimum x-coordinate (left boundary)
    /// * `xmax` - Maximum x-coordinate (right boundary)
    /// * `nx` - Number of points (≥ 2)
    ///
    /// # Grid Spacing
    ///
    /// The uniform spacing is calculated as:
    /// - `dx = (xmax - xmin) / (nx - 1)`
    ///
    /// # Returns
    ///
    /// A new `Grid1d` instance with uniformly spaced coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `nx < 2`
    /// - `xmax ≤ xmin`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid1d, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // Create uniform grid
    ///     let grid = Grid1d::new_uniform(0.0, 1.0, 5)?;
    ///
    ///     // Grid layout:
    ///     //   0───1───2───3───4
    ///     // x=0.0 0.25 0.5 0.75 1.0
    ///
    ///     // Check corner coordinates
    ///     assert_eq!(grid.coord(0), 0.0); // left
    ///     assert_eq!(grid.coord(4), 1.0); // right
    ///     Ok(())
    /// }
    /// ```
    pub fn new_uniform(xmin: f64, xmax: f64, nx: usize) -> Result<Self, StrError> {
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }
        if xmax <= xmin {
            return Err("xmax must be > xmin");
        }
        let dx = (xmax - xmin) / ((nx - 1) as f64);
        let mut coords = Vec::with_capacity(nx);
        for i in 0..nx {
            let x = xmin + (i as f64) * dx;
            coords.push(x);
        }
        Ok(Self { nx, coords })
    }

    /// Returns the total number of grid points
    ///
    /// This equals `nx`.
    pub fn size(&self) -> usize {
        self.nx
    }

    /// Returns the spacing (dx) if the grid is uniform
    ///
    /// Returns `None` if the grid is non-uniform
    pub fn get_dx(&self) -> Option<f64> {
        let mut dx = f64::NEG_INFINITY;
        for i in 1..self.nx {
            let x = self.coords[i];
            let xl = self.coords[i - 1];
            if dx == f64::NEG_INFINITY {
                dx = x - xl;
                assert!(dx > 0.0);
            } else if f64::abs(x - xl - dx) > 10.0 * f64::EPSILON {
                return None; // non-uniform
            }
        }
        Some(dx)
    }

    /// Returns the x-coordinate of the specified node
    ///
    /// # Arguments
    ///
    /// * `m` - Node index (0 ≤ m < nx)
    ///
    /// # Returns
    ///
    /// The x-coordinate of node `m`.
    ///
    /// # Panics
    ///
    /// Panics if `m` is out of bounds (≥ nx).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid1d, StrError};
    /// fn main() -> Result<(), StrError> {
    ///     let grid = Grid1d::new_uniform(0.0, 2.0, 3)?;
    ///
    ///     // Grid layout:
    ///     //   0───1───2
    ///     // x=0.0 1.0 2.0
    ///
    ///     assert_eq!(grid.coord(0), 0.0); // left
    ///     assert_eq!(grid.coord(2), 2.0); // right
    ///     Ok(())
    /// }
    /// ```
    pub fn coord(&self, m: usize) -> f64 {
        self.coords[m]
    }

    /// Iterates over all grid nodes with their coordinates
    ///
    /// The provided closure is called for each node with arguments `(m, x)`
    /// where `m` is the linear node index and `x` is the coordinate.
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that accepts `(node_index: usize, x: f64)`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid1d, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let grid = Grid1d::new_uniform(0.0, 1.0, 3)?;
    ///
    ///     // Print all node coordinates
    ///     grid.for_each_coord(|m, x| {
    ///         println!("Node {}: {:.2}", m, x);
    ///     });
    ///
    ///     // Collect coordinates into a vector
    ///     let mut coords = Vec::new();
    ///     grid.for_each_coord(|_m, x| coords.push(x));
    ///     Ok(())
    /// }
    /// ```
    pub fn for_each_coord(&self, mut f: impl FnMut(usize, f64)) {
        for (m, x) in self.coords.iter().enumerate() {
            f(m, *x);
        }
    }

    /// Returns the index of the first node (left boundary)
    pub fn node_xmin(&self) -> usize {
        0
    }

    /// Returns the index of the last node (right boundary)
    pub fn node_xmax(&self) -> usize {
        self.nx - 1
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Grid1d;

    #[test]
    fn new_fails_on_invalid_input() {
        assert_eq!(Grid1d::new(&[0.0]).err(), Some("nx must be ≥ 2"));
        assert_eq!(Grid1d::new(&[0.0, 0.0]).err(), Some("xx must be strictly increasing"));
        assert_eq!(Grid1d::new(&[1.0, 0.0]).err(), Some("xx must be strictly increasing"));
    }

    #[test]
    fn new_uniform_fails_on_invalid_input() {
        assert_eq!(Grid1d::new_uniform(0.0, 1.0, 1).err(), Some("nx must be ≥ 2"));
        assert_eq!(Grid1d::new_uniform(1.0, 0.0, 4).err(), Some("xmax must be > xmin"));
        assert_eq!(Grid1d::new_uniform(1.0, 1.0, 4).err(), Some("xmax must be > xmin"));
    }

    #[test]
    fn new_works() {
        let xx = &[-3.0, -2.9, 2.9, 3.0];
        let correct_coords = vec![-3.0, -2.9, 2.9, 3.0];

        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.nx, 4);
        assert_eq!(grid.coords, correct_coords);

        assert_eq!(grid.size(), 4);
        assert_eq!(grid.get_dx(), None); // non-uniform grid

        let mut coords = Vec::new();
        grid.for_each_coord(|_m, x| coords.push(x));
        assert_eq!(coords, correct_coords);
    }

    #[test]
    fn new_uniform_works() {
        let xmin = -3.0;
        let xmax = 3.0;
        let nx = 4;
        // dx = (3.0 - (-3.0)) / (4 - 1) = 6.0 / 3 = 2.0
        let correct_coords = vec![-3.0, -1.0, 1.0, 3.0];

        let grid = Grid1d::new_uniform(xmin, xmax, nx).unwrap();
        assert_eq!(grid.nx, 4);
        assert_eq!(grid.coords, correct_coords);

        assert_eq!(grid.size(), 4);
        assert_eq!(grid.get_dx(), Some(2.0));

        let mut coords = Vec::new();
        grid.for_each_coord(|_m, x| coords.push(x));
        assert_eq!(coords, correct_coords);
    }

    #[test]
    fn coord_works() {
        let grid = Grid1d::new_uniform(0.0, 4.0, 5).unwrap();
        assert_eq!(grid.coord(0), 0.0);
        assert_eq!(grid.coord(1), 1.0);
        assert_eq!(grid.coord(2), 2.0);
        assert_eq!(grid.coord(3), 3.0);
        assert_eq!(grid.coord(4), 4.0);
    }

    #[test]
    fn boundary_nodes_work() {
        let grid = Grid1d::new_uniform(0.0, 10.0, 6).unwrap();
        assert_eq!(grid.node_xmin(), 0);
        assert_eq!(grid.node_xmax(), 5);
    }

    #[test]
    fn get_dx_works_uniform() {
        let grid = Grid1d::new_uniform(0.0, 3.0, 31).unwrap();
        assert_eq!(grid.get_dx(), Some(0.1));
    }

    #[test]
    fn get_dx_captures_non_uniform() {
        // Non-uniform grid
        let xx = &[0.0, 0.1, 0.5, 1.0];
        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.get_dx(), None);
    }

    #[test]
    fn get_dx_uniform_grids() {
        // Test uniform grid with integer spacing
        let grid = Grid1d::new_uniform(0.0, 6.0, 4).unwrap();
        // dx = 6.0 / 3 = 2.0
        assert_eq!(grid.get_dx(), Some(2.0));

        // Test uniform grid with fractional spacing
        let grid = Grid1d::new_uniform(0.0, 1.0, 5).unwrap();
        // dx = 1.0 / 4 = 0.25
        assert_eq!(grid.get_dx(), Some(0.25));

        // Test minimal 2-point uniform grid
        let grid = Grid1d::new_uniform(0.0, 1.0, 2).unwrap();
        // dx = 1.0 / 1 = 1.0
        assert_eq!(grid.get_dx(), Some(1.0));
    }

    #[test]
    fn get_dx_non_uniform_grids() {
        // Non-uniform spacing
        let xx = &[0.0, 0.1, 0.5, 1.0];
        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.get_dx(), None);

        // Logarithmic spacing
        let xx = &[0.1, 1.0, 10.0, 100.0];
        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.get_dx(), None);
    }

    #[test]
    fn get_dx_uniform_from_arrays() {
        // Test that manually created uniform arrays are detected as uniform
        let xx = &[0.0, 1.0, 2.0, 3.0, 4.0]; // uniform: dx = 1.0
        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.get_dx(), Some(1.0));

        // Test with negative coordinates
        let xx = &[-2.0, -1.0, 0.0, 1.0]; // uniform: dx = 1.0
        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.get_dx(), Some(1.0));

        // Test with fractional uniform spacing
        let xx = &[0.0, 0.25, 0.5, 0.75, 1.0]; // uniform: dx = 0.25
        let grid = Grid1d::new(xx).unwrap();
        let result = grid.get_dx().unwrap();
        assert!((result - 0.25).abs() < 1e-15);
    }

    #[test]
    fn get_dx_precision_edge_cases() {
        // Test with very small spacing that might have floating-point precision issues
        let grid = Grid1d::new_uniform(0.0, 1e-6, 3).unwrap();
        let result = grid.get_dx().unwrap();
        assert!((result - 5e-7).abs() < 1e-21); // dx = 1e-6 / 2 = 5e-7

        // Test with very large spacing
        let grid = Grid1d::new_uniform(0.0, 1e6, 3).unwrap();
        let result = grid.get_dx().unwrap();
        assert!((result - 5e5).abs() < 1e-9); // dx = 1e6 / 2 = 5e5

        // Test grid that's almost uniform but has tiny differences beyond epsilon
        let mut xx = vec![0.0, 1.0, 2.0, 3.0];
        xx[2] += 11.0 * f64::EPSILON; // Add small perturbation beyond epsilon
        let grid = Grid1d::new(&xx).unwrap();
        assert_eq!(grid.get_dx(), None); // Should detect as non-uniform

        // Test grid with differences exactly at epsilon boundary
        let mut xx = vec![0.0, 1.0, 2.0, 3.0];
        xx[2] += f64::EPSILON / 2.0; // Add perturbation within epsilon tolerance
        let grid = Grid1d::new(&xx).unwrap();
        assert_eq!(grid.get_dx(), Some(1.0)); // Should still be uniform
    }

    #[test]
    fn get_dx_different_grid_sizes() {
        // Test with different grid dimensions to ensure algorithm works for all sizes

        // 2-point grid
        let grid = Grid1d::new_uniform(0.0, 3.0, 2).unwrap();
        assert_eq!(grid.get_dx(), Some(3.0));

        // 10-point grid
        let grid = Grid1d::new_uniform(0.0, 9.0, 10).unwrap();
        assert_eq!(grid.get_dx(), Some(1.0));

        // Large grid
        let grid = Grid1d::new_uniform(0.0, 1.0, 50).unwrap();
        let result = grid.get_dx().unwrap();
        assert!((result - 1.0 / 49.0).abs() < 1e-15);
    }

    #[test]
    fn get_dx_boundary_coordinates() {
        // Test with coordinates at domain boundaries

        // Grid spanning zero
        let grid = Grid1d::new_uniform(-1.0, 1.0, 3).unwrap();
        assert_eq!(grid.get_dx(), Some(1.0));

        // Grid with very small domain
        let grid = Grid1d::new_uniform(0.0, 1e-10, 2).unwrap();
        let result = grid.get_dx().unwrap();
        assert!((result - 1e-10).abs() < 1e-25);

        // Grid with large coordinates
        let grid = Grid1d::new_uniform(1e6, 1e6 + 4.0, 3).unwrap();
        assert_eq!(grid.get_dx(), Some(2.0));
    }

    #[test]
    fn for_each_coord_works() {
        let grid = Grid1d::new_uniform(0.0, 2.0, 3).unwrap();

        let mut indices = Vec::new();
        let mut coords = Vec::new();

        grid.for_each_coord(|m, x| {
            indices.push(m);
            coords.push(x);
        });

        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(coords, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn for_each_coord_empty_closure() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 2).unwrap();

        // Test that closure can be called without doing anything
        grid.for_each_coord(|_, _| {});
    }

    #[test]
    fn size_method_works() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 5).unwrap();
        assert_eq!(grid.size(), 5);
    }

    #[test]
    fn constructor_with_single_spacing() {
        // Test edge case with minimal grid
        let grid = Grid1d::new_uniform(5.0, 7.0, 2).unwrap();
        assert_eq!(grid.size(), 2);
        assert_eq!(grid.coord(0), 5.0);
        assert_eq!(grid.coord(1), 7.0);
        assert_eq!(grid.get_dx(), Some(2.0));
    }

    #[test]
    fn constructor_with_negative_domain() {
        let grid = Grid1d::new_uniform(-5.0, -1.0, 5).unwrap();
        assert_eq!(grid.coord(0), -5.0);
        assert_eq!(grid.coord(4), -1.0);
        assert_eq!(grid.get_dx(), Some(1.0)); // dx = 4.0 / 4 = 1.0
    }

    #[test]
    fn non_uniform_constructor_various_cases() {
        // Exponential-like spacing
        let xx = &[0.0, 0.1, 0.3, 0.7, 1.5];
        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.size(), 5);
        assert_eq!(grid.get_dx(), None);

        // Reverse exponential-like spacing
        let xx = &[0.0, 0.8, 1.2, 1.4, 1.5];
        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.size(), 5);
        assert_eq!(grid.get_dx(), None);
    }

    #[test]
    fn boundary_node_indices() {
        // Test various grid sizes
        let grid = Grid1d::new_uniform(0.0, 1.0, 2).unwrap();
        assert_eq!(grid.node_xmin(), 0);
        assert_eq!(grid.node_xmax(), 1);

        let grid = Grid1d::new_uniform(0.0, 1.0, 10).unwrap();
        assert_eq!(grid.node_xmin(), 0);
        assert_eq!(grid.node_xmax(), 9);

        let grid = Grid1d::new_uniform(0.0, 1.0, 100).unwrap();
        assert_eq!(grid.node_xmin(), 0);
        assert_eq!(grid.node_xmax(), 99);
    }

    #[test]
    fn documentation_examples_work() {
        // Test the examples from the documentation

        // Example from Grid1d::new
        let xx = &[0.0, 0.1, 0.5, 0.9, 1.0];
        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.size(), 5);

        // Example from Grid1d::new_uniform
        let grid = Grid1d::new_uniform(0.0, 1.0, 5).unwrap();
        assert_eq!(grid.coord(0), 0.0);
        assert_eq!(grid.coord(4), 1.0);

        // Example from for_each_coord
        let grid = Grid1d::new_uniform(0.0, 1.0, 3).unwrap();
        let mut coords = Vec::new();
        grid.for_each_coord(|_m, x| coords.push(x));
        assert_eq!(coords.len(), 3);
    }
}
