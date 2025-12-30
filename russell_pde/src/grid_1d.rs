use crate::Side;
use crate::StrError;
use russell_lab::math::chebyshev_lobatto_points;
use russell_lab::Vector;

/// Defines a one-dimensional grid with node coordinates
///
/// This structure stores the nodes of a 1D computational grid and provides
/// methods for grid construction, coordinate access, and boundary node identification.
///
/// The grid can be uniform (equally-spaced) or non-uniform, and supports
/// both Cartesian coordinates and Chebyshev-Gauss-Lobatto (CGL) points.
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
    /// Number of grid nodes (nx ≥ 2)
    ///
    /// Represents the total count of nodes in the 1D grid.
    /// Valid node indices are 0 ≤ m < nx.
    nx: usize,

    /// Node coordinates in ascending order (len = nx)
    ///
    /// Stores the x-coordinate of each node. The coordinates are strictly
    /// increasing: coords[0] < coords[1] < ... < coords[nx-1].
    coords: Vector,

    /// Indices of nodes on the xmin boundary (len = 1; always = [0])
    ///
    /// Contains the index of the leftmost boundary node.
    /// In 1D, this is always a single-element vector [0].
    /// This data member ensures a consistent interface with higher-dimensional grids.
    nodes_xmin: Vec<usize>,

    /// Indices of nodes on the xmax boundary (len = 1; always = [nx-1])
    ///
    /// Contains the index of the rightmost boundary node.
    /// In 1D, this is always a single-element vector [nx-1].
    /// This data member ensures a consistent interface with higher-dimensional grids.
    nodes_xmax: Vec<usize>,
}

impl Grid1d {
    /// Creates a new 1D grid from arbitrary x-coordinates
    ///
    /// Constructs a grid with specified node coordinates, which must be
    /// strictly increasing. This allows for non-uniform (arbitrary) spacing.
    ///
    /// # Input
    ///
    /// * `xx` - Array of x-coordinates (must be strictly increasing with length ≥ 2)
    ///
    /// # Returns
    ///
    /// Returns a new Grid1d instance with nx = xx.len() points.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * nx < 2 (need at least two nodes)
    /// * Coordinates are not strictly increasing
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
    ///     assert_eq!(grid.nx(), 5);
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
            coords: Vector::from(&xx),
            nodes_xmin: vec![0],
            nodes_xmax: vec![nx - 1],
        })
    }

    /// Creates a new uniformly-spaced (equally-spaced) grid
    ///
    /// Constructs a grid with uniform spacing between consecutive nodes:
    /// dx = (xmax - xmin) / (nx - 1)
    ///
    /// The grid spans from xmin to xmax with nx equally-spaced nodes.
    ///
    /// # Input
    ///
    /// * `xmin` - Coordinate of the leftmost node (at x = xmin)
    /// * `xmax` - Coordinate of the rightmost node (at x = xmax); must satisfy xmax > xmin
    /// * `nx` - Number of nodes (must satisfy nx ≥ 2)
    ///
    /// # Returns
    ///
    /// Returns a new Grid1d instance with uniform spacing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * nx < 2 (need at least two nodes)
    /// * xmax ≤ xmin (invalid domain)
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
        let mut coords = Vector::new(nx);
        for i in 0..nx {
            let x = xmin + (i as f64) * dx;
            coords[i] = x;
        }
        Ok(Self {
            nx,
            coords,
            nodes_xmin: vec![0],
            nodes_xmax: vec![nx - 1],
        })
    }

    /// Creates a new grid using Chebyshev-Gauss-Lobatto (CGL) points
    ///
    /// The Chebyshev-Gauss-Lobatto points are defined in the interval [-1, 1] and are given by:
    ///
    /// ```text
    /// x_j = -cos(j·π / N),  j = 0, 1, ..., N
    /// ```
    ///
    /// where N = nx - 1 is the polynomial degree.
    ///
    /// **Properties:**
    /// * The points include both endpoints: x₀ = -1 and xₙ = 1
    /// * The spacing is denser near the boundaries, with more points clustered at the endpoints
    /// * CGL points are optimal for spectral collocation methods as they minimize
    ///   interpolation errors and the Lebesgue constant
    /// * The point distribution corresponds to the projection of equally-spaced points
    ///   on a semicircle onto the diameter
    ///
    /// CGL points are particularly useful for spectral methods where high accuracy
    /// is needed for smooth solutions, and the clustered boundary points help
    /// resolve boundary layers.
    ///
    /// # Input
    ///
    /// * `nx` - Number of nodes (must satisfy nx ≥ 2)
    ///
    /// # Returns
    ///
    /// Returns a new Grid1d instance with CGL points in [-1, 1].
    ///
    /// # Errors
    ///
    /// Returns an error if nx < 2.
    ///
    /// # Notes
    ///
    /// * For a different domain [a, b], use the transformation: ξ = (2x - b - a) / (b - a)
    /// * The grid is non-uniform with denser spacing at x = ±1
    pub fn new_chebyshev_gauss_lobatto(nx: usize) -> Result<Self, StrError> {
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }
        //        xb + xa + (xb - xa) u
        // x(u) = —————————————————————
        //                 2
        Ok(Self {
            nx,
            coords: chebyshev_lobatto_points(nx - 1),
            nodes_xmin: vec![0],
            nodes_xmax: vec![nx - 1],
        })
    }

    /// Returns the number of nodes in the grid
    ///
    /// # Returns
    ///
    /// The total count of nodes (nx ≥ 2)
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Returns whether a node lies on the xmin (left) boundary
    ///
    /// # Input
    ///
    /// * `m` - Node index to check
    ///
    /// # Returns
    ///
    /// Returns `true` if node m is at x = xmin (i.e., m == 0), `false` otherwise.
    pub fn is_xmin(&self, m: usize) -> bool {
        m == 0
    }

    /// Returns whether a node lies on the xmax (right) boundary
    ///
    /// # Input
    ///
    /// * `m` - Node index to check
    ///
    /// # Returns
    ///
    /// Returns `true` if node m is at x = xmax (i.e., m == nx-1), `false` otherwise.
    pub fn is_xmax(&self, m: usize) -> bool {
        m == self.nx - 1
    }

    /// Returns whether a node lies on any boundary
    ///
    /// In 1D, a node is on the boundary if it is at either endpoint.
    ///
    /// # Input
    ///
    /// * `m` - Node index to check
    ///
    /// # Returns
    ///
    /// Returns `true` if node m is at either x = xmin or x = xmax, `false` otherwise.
    pub fn on_boundary(&self, m: usize) -> bool {
        m == 0 || m == self.nx - 1
    }

    /// Returns the indices of nodes on a specified boundary side
    ///
    /// # Input
    ///
    /// * `side` - The boundary side to query. Valid values are:
    ///   - `Side::Xmin` - Returns nodes at x = xmin (left boundary)
    ///   - `Side::Xmax` - Returns nodes at x = xmax (right boundary)
    ///
    /// # Returns
    ///
    /// A slice containing node indices on the specified side.
    /// In 1D, this is always a single-element slice:
    /// * `[0]` for Xmin
    /// * `[nx-1]` for Xmax
    ///
    /// # Panics
    ///
    /// Panics if side is Ymin, Ymax, Zmin, or Zmax (invalid for 1D grids).
    pub fn get_nodes_on_side(&self, side: Side) -> &[usize] {
        match side {
            Side::Xmin => &self.nodes_xmin,
            Side::Xmax => &self.nodes_xmax,
            _ => panic!("invalid side for 1D grid"),
        }
    }

    /// Returns slices with the indices of nodes on both boundaries
    ///
    /// # Returns
    ///
    /// A tuple `(nodes_xmin, nodes_xmax)` where:
    /// * `nodes_xmin` - Slice of node indices on the xmin (left) boundary: [0]
    /// * `nodes_xmax` - Slice of node indices on the xmax (right) boundary: [nx-1]
    ///
    /// In 1D, each slice contains exactly one node index.
    pub fn get_boundary_nodes(&self) -> (&[usize], &[usize]) {
        (&self.nodes_xmin, &self.nodes_xmax)
    }

    /// Returns the uniform spacing (dx) if the grid has constant spacing
    ///
    /// Checks whether all consecutive node spacings are equal (within numerical tolerance)
    /// and returns the spacing value if uniform.
    ///
    /// # Returns
    ///
    /// * `Some(dx)` - If the grid is uniformly spaced, returns the constant spacing
    /// * `None` - If the grid is non-uniform (varying spacing between nodes)
    ///
    /// # Notes
    ///
    /// * The method uses a tolerance of 10·ε to account for floating-point precision
    /// * For grids created with `new_uniform()`, this always returns `Some(dx)`
    /// * For CGL grids or arbitrary grids, this typically returns `None`
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

    /// Returns the x-coordinate of a specified node
    ///
    /// # Input
    ///
    /// * `m` - Node index (must satisfy 0 ≤ m < nx)
    ///
    /// # Returns
    ///
    /// The x-coordinate of node m.
    ///
    /// # Panics
    ///
    /// Panics if m ≥ nx (index out of bounds).
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
    /// Calls the provided closure for each node in order (from left to right),
    /// passing the node index and its x-coordinate as arguments.
    ///
    /// # Input
    ///
    /// * `f` - Closure that accepts `(m: usize, x: f64)` where:
    ///   - `m` is the node index (0 to nx-1)
    ///   - `x` is the x-coordinate of node m
    ///
    /// # Notes
    ///
    /// * Nodes are visited in ascending order: m = 0, 1, ..., nx-1
    /// * Useful for operations that need both the index and coordinate of each node
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
        for m in 0..self.nx {
            f(m, self.coords[m]);
        }
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
        assert_eq!(grid.coords.as_data(), &correct_coords);

        assert_eq!(grid.nx(), 4);
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
        assert_eq!(grid.coords.as_data(), &correct_coords);

        assert_eq!(grid.nx(), 4);
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
        assert_eq!(grid.is_xmin(0), true);
        assert_eq!(grid.is_xmin(1), false);
        assert_eq!(grid.is_xmax(5), true);
        assert_eq!(grid.is_xmax(4), false);
        assert_eq!(grid.get_boundary_nodes(), (&[0][..], &[5][..]));
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
        assert_eq!(grid.nx(), 5);
    }

    #[test]
    fn constructor_with_single_spacing() {
        // Test edge case with minimal grid
        let grid = Grid1d::new_uniform(5.0, 7.0, 2).unwrap();
        assert_eq!(grid.nx(), 2);
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
        assert_eq!(grid.nx(), 5);
        assert_eq!(grid.get_dx(), None);

        // Reverse exponential-like spacing
        let xx = &[0.0, 0.8, 1.2, 1.4, 1.5];
        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.nx(), 5);
        assert_eq!(grid.get_dx(), None);
    }

    #[test]
    fn boundary_node_indices() {
        // Test various grid sizes
        let grid = Grid1d::new_uniform(0.0, 1.0, 2).unwrap();
        assert_eq!(grid.is_xmin(0), true);
        assert_eq!(grid.is_xmax(1), true);

        let grid = Grid1d::new_uniform(0.0, 1.0, 10).unwrap();
        assert_eq!(grid.is_xmin(0), true);
        assert_eq!(grid.is_xmax(9), true);

        let grid = Grid1d::new_uniform(0.0, 1.0, 100).unwrap();
        assert_eq!(grid.is_xmin(0), true);
        assert_eq!(grid.is_xmax(99), true);
    }

    #[test]
    fn documentation_examples_work() {
        // Test the examples from the documentation

        // Example from Grid1d::new
        let xx = &[0.0, 0.1, 0.5, 0.9, 1.0];
        let grid = Grid1d::new(xx).unwrap();
        assert_eq!(grid.nx(), 5);

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

    #[test]
    fn new_chebyshev_gauss_lobatto_works() {
        let grid = Grid1d::new_chebyshev_gauss_lobatto(3).unwrap();
        assert_eq!(grid.nx(), 3);
        // Points for N=2 (3 points) in [-1, 1]:
        // x_j = -cos(j * pi / 2), j=0,1,2
        // x_0 = -cos(0) = -1
        // x_1 = -cos(pi/2) = 0
        // x_2 = -cos(pi) = 1
        assert!((grid.coord(0) - (-1.0)).abs() < 1e-15);
        assert!((grid.coord(1) - 0.0).abs() < 1e-15);
        assert!((grid.coord(2) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn new_chebyshev_gauss_lobatto_fails_on_invalid_input() {
        assert_eq!(Grid1d::new_chebyshev_gauss_lobatto(1).err(), Some("nx must be ≥ 2"));
    }

    #[test]
    fn on_boundary_works() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 3).unwrap();
        assert!(grid.on_boundary(0));
        assert!(!grid.on_boundary(1));
        assert!(grid.on_boundary(2));
    }

    #[test]
    fn get_nodes_on_side_works() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 3).unwrap();
        assert_eq!(grid.get_nodes_on_side(crate::Side::Xmin), &[0]);
        assert_eq!(grid.get_nodes_on_side(crate::Side::Xmax), &[2]);
    }

    #[test]
    #[should_panic(expected = "invalid side for 1D grid")]
    fn get_nodes_on_side_panics_on_invalid_side() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 3).unwrap();
        grid.get_nodes_on_side(crate::Side::Ymin);
    }
}
