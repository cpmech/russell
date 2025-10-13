use crate::StrError;

/// Defines a 2D Cartesian grid
///
/// This structure represents a structured 2D grid with nodes arranged in a Cartesian coordinate system.
/// The grid can be created with either arbitrary coordinates or uniform spacing.
///
/// ## Grid Layout and Indexing
///
/// The grid uses a **row-major** indexing scheme where nodes are numbered sequentially:
/// - First along the x-direction (i-index)  
/// - Then along the y-direction (j-index)
///
/// ```text
///      i=0     i=1     i=2     i=3     i=4
/// j=2  10──────11──────12──────13──────14  j=2  ny=3
///       │       │       │       │       │
///       │       │       │       │       │
/// j=1   5───────6───────7───────8───────9  j=1
///       │       │       │       │       │
///       │       │       │       │       │
/// j=0   0───────1───────2───────3───────4  j=0
///      i=0     i=1     i=2     i=3     i=4
///                                     nx=5
/// ```
///
/// ## Index Conversion Formulae
///
/// ```text
/// m = i + j × nx    (convert (i,j) to linear index m)
/// i = m % nx        (extract i-index from linear index m)
/// j = m / nx        (extract j-index from linear index m)
/// ```
///
/// Where:
/// - `m` is the linear node index (0, 1, 2, ...)
/// - `i` is the column index (0 ≤ i < nx)
/// - `j` is the row index (0 ≤ j < ny)
/// - `%` is the modulo operator
/// - `/` is integer division
///
/// ## Boundary Node Classification
///
/// The grid automatically identifies boundary nodes:
/// - **xmin edge**: Left boundary (i = 0)
/// - **xmax edge**: Right boundary (i = nx-1)  
/// - **ymin edge**: Bottom boundary (j = 0)
/// - **ymax edge**: Top boundary (j = ny-1)
///
/// Corner nodes belong to multiple boundaries simultaneously.
///
/// ## Examples
///
/// ```rust
/// use russell_pde::{Grid2d, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // Create uniform 3x3 grid on unit square
///     let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3)?;
///
///     // Grid layout:
///     //   6───7───8  (y=1.0)
///     //   │   │   │
///     //   3───4───5  (y=0.5)
///     //   │   │   │
///     //   0───1───2  (y=0.0)
///     // x=0.0 0.5 1.0
///
///     // Print all node coordinates
///     grid.for_each_coord(|m, x, y| {
///         println!("Node {}: ({:.1}, {:.1})", m, x, y);
///     });
///     // Output:
///     // Node 0: (0.0, 0.0)
///     // Node 1: (0.5, 0.0)
///     // Node 2: (1.0, 0.0)
///     // Node 3: (0.0, 0.5)
///     // ...
///
///     // Collect coordinates into a vector
///     let mut coords = Vec::new();
///     grid.for_each_coord(|_m, x, y| coords.push((x, y)));
///     Ok(())
/// }
/// ```
pub struct Grid2d {
    /// Number of points along the x-direction (≥ 2)
    ///
    /// This represents the number of columns in the grid.
    nx: usize,

    /// Number of points along the y-direction (≥ 2)
    ///
    /// This represents the number of rows in the grid.
    ny: usize,

    /// Node coordinates stored as (x, y) pairs
    ///
    /// The coordinates are stored in row-major order, so:
    /// - `coords[0]` to `coords[nx-1]` are the bottom row (j=0)
    /// - `coords[nx]` to `coords[2*nx-1]` are the second row (j=1)
    /// - And so on...
    coords: Vec<(f64, f64)>,

    /// Linear indices of nodes on the left boundary (xmin edge)
    ///
    /// Contains nodes where i = 0: [0, nx, 2*nx, ..., (ny-1)*nx]
    nodes_xmin: Vec<usize>,

    /// Linear indices of nodes on the right boundary (xmax edge)
    ///
    /// Contains nodes where i = nx-1: [nx-1, 2*nx-1, 3*nx-1, ..., ny*nx-1]
    nodes_xmax: Vec<usize>,

    /// Linear indices of nodes on the bottom boundary (ymin edge)
    ///
    /// Contains nodes where j = 0: [0, 1, 2, ..., nx-1]
    nodes_ymin: Vec<usize>,

    /// Linear indices of nodes on the top boundary (ymax edge)
    ///
    /// Contains nodes where j = ny-1: [(ny-1)*nx, (ny-1)*nx+1, ..., ny*nx-1]
    nodes_ymax: Vec<usize>,
}

impl Grid2d {
    /// Creates a new grid with arbitrary coordinate arrays
    ///
    /// This constructor allows for non-uniform spacing by providing explicit
    /// coordinate arrays for both x and y directions.
    ///
    /// # Arguments
    ///
    /// * `xx` - Array of x-coordinates (must be strictly increasing, length ≥ 2)
    /// * `yy` - Array of y-coordinates (must be strictly increasing, length ≥ 2)
    ///
    /// # Returns
    ///
    /// A new `Grid2d` instance with `nx = xx.len()` and `ny = yy.len()` points.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `xx.len() < 2` or `yy.len() < 2`
    /// - `xx` or `yy` arrays are not strictly increasing
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid2d, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // Non-uniform grid with refined spacing near boundaries
    ///     let xx = &[0.0, 0.1, 0.5, 0.9, 1.0];
    ///     let yy = &[0.0, 0.2, 0.8, 1.0];
    ///     let grid = Grid2d::new(xx, yy)?;
    ///
    ///     // Grid layout (5x4):
    ///     //   15──16──17──18──19  (y=1.0)
    ///     //   │   │   │   │   │
    ///     //   10──11──12──13──14  (y=0.8)
    ///     //   │   │   │   │   │
    ///     //   5───6───7───8───9  (y=0.2)
    ///     //   │   │   │   │   │
    ///     //   0───1───2───3───4  (y=0.0)
    ///     // x=0.0 0.1 0.5 0.9 1.0
    ///
    ///     assert_eq!(grid.nx(), 5);
    ///     assert_eq!(grid.ny(), 4);
    ///     assert_eq!(grid.size(), 20);
    ///     Ok(())
    /// }
    /// ```
    pub fn new(xx: &[f64], yy: &[f64]) -> Result<Self, StrError> {
        let nx = xx.len();
        let ny = yy.len();
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }
        if ny < 2 {
            return Err("ny must be ≥ 2");
        }
        let mut coords = Vec::with_capacity(nx * ny);
        for i in 1..nx {
            if xx[i] <= xx[i - 1] {
                return Err("xx must be strictly increasing");
            }
        }
        for j in 0..ny {
            if j > 0 && yy[j] <= yy[j - 1] {
                return Err("yy must be strictly increasing");
            }
            for i in 0..nx {
                coords.push((xx[i], yy[j]));
            }
        }
        Ok(Self {
            nx,
            ny,
            coords,
            nodes_xmin: (0..ny).map(|j| j * nx).collect(),
            nodes_xmax: (0..ny).map(|j| j * nx + (nx - 1)).collect(),
            nodes_ymin: (0..nx).collect(),
            nodes_ymax: (0..nx).map(|i| (ny - 1) * nx + i).collect(),
        })
    }

    /// Creates a new grid with uniform spacing
    ///
    /// This constructor creates a structured grid with uniform spacing in both
    /// x and y directions. The spacing is calculated automatically based on
    /// the domain size and number of points.
    ///
    /// # Arguments
    ///
    /// * `xmin` - Minimum x-coordinate (left boundary)
    /// * `xmax` - Maximum x-coordinate (right boundary)  
    /// * `ymin` - Minimum y-coordinate (bottom boundary)
    /// * `ymax` - Maximum y-coordinate (top boundary)
    /// * `nx` - Number of points along x-direction (≥ 2)
    /// * `ny` - Number of points along y-direction (≥ 2)
    ///
    /// # Grid Spacing
    ///
    /// The uniform spacing is calculated as:
    /// - `dx = (xmax - xmin) / (nx - 1)`
    /// - `dy = (ymax - ymin) / (ny - 1)`
    ///
    /// # Returns
    ///
    /// A new `Grid2d` instance with uniformly spaced coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `nx < 2` or `ny < 2`
    /// - `xmax ≤ xmin` or `ymax ≤ ymin`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid2d, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // Unit square with 5x4 grid
    ///     let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 5, 4)?;
    ///
    ///     // Grid layout:
    ///     //   15──16──17──18──19  (y=1.0)
    ///     //   │   │   │   │   │
    ///     //   10──11──12──13──14  (y=0.67)
    ///     //   │   │   │   │   │
    ///     //   5───6───7───8───9  (y=0.33)
    ///     //   │   │   │   │   │
    ///     //   0───1───2───3───4  (y=0.0)
    ///     // x=0.0 0.25 0.5 0.75 1.0
    ///
    ///     // Check corner coordinates
    ///     assert_eq!(grid.coord(0), (0.0, 0.0));   // bottom-left
    ///     assert_eq!(grid.coord(4), (1.0, 0.0));   // bottom-right  
    ///     assert_eq!(grid.coord(15), (0.0, 1.0));  // top-left
    ///     assert_eq!(grid.coord(19), (1.0, 1.0));  // top-right
    ///     Ok(())
    /// }
    /// ```
    pub fn new_uniform(xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Result<Self, StrError> {
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }
        if ny < 2 {
            return Err("ny must be ≥ 2");
        }
        if xmax <= xmin {
            return Err("xmax must be > xmin");
        }
        if ymax <= ymin {
            return Err("ymax must be > ymin");
        }
        let dx = (xmax - xmin) / ((nx - 1) as f64);
        let dy = (ymax - ymin) / ((ny - 1) as f64);
        let mut coords = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            let y = ymin + (j as f64) * dy;
            for i in 0..nx {
                let x = xmin + (i as f64) * dx;
                coords.push((x, y));
            }
        }
        Ok(Self {
            nx,
            ny,
            coords,
            nodes_xmin: (0..ny).map(|j| j * nx).collect(),
            nodes_xmax: (0..ny).map(|j| j * nx + (nx - 1)).collect(),
            nodes_ymin: (0..nx).collect(),
            nodes_ymax: (0..nx).map(|i| (ny - 1) * nx + i).collect(),
        })
    }

    /// Returns the number of grid points along the x-direction
    ///
    /// This corresponds to the number of columns in the grid.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Returns the number of grid points along the y-direction
    ///
    /// This corresponds to the number of rows in the grid.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Returns the total number of grid points
    ///
    /// This equals `nx × ny`.
    pub fn size(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns the (x, y) coordinates of the specified node
    ///
    /// # Arguments
    ///
    /// * `m` - Linear node index (0 ≤ m < nx×ny)
    ///
    /// # Returns
    ///
    /// A tuple `(x, y)` containing the coordinates of node `m`.
    ///
    /// # Panics
    ///
    /// Panics if `m` is out of bounds (≥ nx×ny).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid2d, StrError};
    /// fn main() -> Result<(), StrError> {
    ///     let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 1.0, 3, 2)?;
    ///
    ///     // Grid layout (3x2):
    ///     //   3───4───5  (y=1.0)
    ///     //   │   │   │
    ///     //   0───1───2  (y=0.0)
    ///     // x=0.0 1.0 2.0
    ///
    ///     assert_eq!(grid.coord(0), (0.0, 0.0)); // bottom-left
    ///     assert_eq!(grid.coord(1), (1.0, 0.0)); // bottom-center  
    ///     assert_eq!(grid.coord(3), (0.0, 1.0)); // top-left
    ///     Ok(())
    /// }
    /// ```
    pub fn coord(&self, m: usize) -> (f64, f64) {
        self.coords[m]
    }

    /// Iterates over all grid nodes with their coordinates
    ///
    /// The provided closure is called for each node with arguments `(m, x, y)`
    /// where `m` is the linear node index and `(x, y)` are the coordinates.
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that accepts `(node_index: usize, x: f64, y: f64)`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid2d, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3)?;
    ///
    ///     // Print all node coordinates
    ///     grid.for_each_coord(|m, x, y| {
    ///         println!("Node {}: ({:.2}, {:.2})", m, x, y);
    ///     });
    ///
    ///     // Collect coordinates into a vector
    ///     let mut coords = Vec::new();
    ///     grid.for_each_coord(|_m, x, y| coords.push((x, y)));
    ///     Ok(())
    /// }
    /// ```
    pub fn for_each_coord(&self, mut f: impl FnMut(usize, f64, f64)) {
        for (m, (x, y)) in self.coords.iter().enumerate() {
            f(m, *x, *y);
        }
    }

    /// Iterates over nodes on the left boundary (xmin edge)
    ///
    /// Processes all nodes where `i = 0` (leftmost column).
    /// The closure receives a reference to each node index.
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that accepts `&node_index: &usize`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid2d, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 2.0, 4, 3)?;
    ///
    ///     // Grid layout (4x3):
    ///     //    8───9──10──11  (y=2.0)
    ///     //    │   │   │   │
    ///     //    4───5───6───7  (y=1.0)
    ///     //    │   │   │   │
    ///     //    0───1───2───3  (y=0.0)
    ///     //   x=0  0.67 1.33 2.0
    ///     ///
    ///     /// Left boundary nodes: 0, 4, 8 (marked with *)
    ///     ///   *8───9──10──11
    ///     ///    │   │   │   │
    ///     ///   *4───5───6───7
    ///     ///    │   │   │   │
    ///     ///   *0───1───2───3
    ///
    ///     // Apply Dirichlet boundary condition on left edge
    ///     grid.for_each_node_xmin(|&node| {
    ///         let (x, y) = grid.coord(node);
    ///         println!("Left boundary node {}: ({}, {})", node, x, y);
    ///         // Set boundary value: u[node] = boundary_function(x, y)
    ///     });
    ///     // Output:
    ///     // Left boundary node 0: (0, 0)
    ///     // Left boundary node 4: (0, 1)
    ///     // Left boundary node 8: (0, 2)
    ///     Ok(())
    /// }
    /// ```
    pub fn for_each_node_xmin<F>(&self, mut f: F)
    where
        F: FnMut(&usize),
    {
        self.nodes_xmin.iter().for_each(|n| f(n));
    }

    /// Iterates over nodes on the right boundary (xmax edge)
    ///
    /// Processes all nodes where `i = nx-1` (rightmost column).
    /// The closure receives a reference to each node index.
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that accepts `&node_index: &usize`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid2d, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 2.0, 4, 3)?;
    ///
    ///     // Right boundary nodes: 3, 7, 11 (marked with *)
    ///     //    8───9──10──*11
    ///     //    │   │   │   │
    ///     //    4───5───6──*7
    ///     //    │   │   │   │
    ///     //    0───1───2──*3
    ///
    ///     grid.for_each_node_xmax(|&node| {
    ///         let (x, y) = grid.coord(node);
    ///         println!("Right boundary node {}: ({}, {})", node, x, y);
    ///     });
    ///     Ok(())
    /// }
    /// ```
    pub fn for_each_node_xmax<F>(&self, mut f: F)
    where
        F: FnMut(&usize),
    {
        self.nodes_xmax.iter().for_each(|n| f(n));
    }

    /// Iterates over nodes on the bottom boundary (ymin edge)
    ///
    /// Processes all nodes where `j = 0` (bottom row).
    /// The closure receives a reference to each node index.
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that accepts `&node_index: &usize`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid2d, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 2.0, 4, 3)?;
    ///
    ///     // Bottom boundary nodes: 0, 1, 2, 3 (marked with *)
    ///     //    8───9──10──11
    ///     //    │   │   │   │
    ///     //    4───5───6───7
    ///     //    │   │   │   │
    ///     //   *0──*1──*2──*3
    ///
    ///     grid.for_each_node_ymin(|&node| {
    ///         let (x, y) = grid.coord(node);
    ///         println!("Bottom boundary node {}: ({}, {})", node, x, y);
    ///     });
    ///     Ok(())
    /// }
    /// ```
    pub fn for_each_node_ymin<F>(&self, mut f: F)
    where
        F: FnMut(&usize),
    {
        self.nodes_ymin.iter().for_each(|n| f(n));
    }

    /// Iterates over nodes on the top boundary (ymax edge)
    ///
    /// Processes all nodes where `j = ny-1` (top row).
    /// The closure receives a reference to each node index.
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that accepts `&node_index: &usize`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use russell_pde::{Grid2d, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 2.0, 4, 3)?;
    ///
    ///     // Top boundary nodes: 8, 9, 10, 11 (marked with *)
    ///     //   *8──*9─*10─*11
    ///     //    │   │   │   │
    ///     //    4───5───6───7
    ///     //    │   │   │   │
    ///     //    0───1───2───3
    ///
    ///     grid.for_each_node_ymax(|&node| {
    ///         let (x, y) = grid.coord(node);
    ///         println!("Top boundary node {}: ({}, {})", node, x, y);
    ///     });
    ///     Ok(())
    /// }
    /// ```
    pub fn for_each_node_ymax<F>(&self, mut f: F)
    where
        F: FnMut(&usize),
    {
        self.nodes_ymax.iter().for_each(|n| f(n));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Grid2d;

    #[test]
    fn new_fails_on_invalid_input() {
        assert_eq!(Grid2d::new(&[0.0], &[0.0, 1.0]).err(), Some("nx must be ≥ 2"));
        assert_eq!(Grid2d::new(&[0.0, 1.0], &[0.0]).err(), Some("ny must be ≥ 2"));
        assert_eq!(
            Grid2d::new(&[0.0, 0.0], &[0.0, 1.0]).err(),
            Some("xx must be strictly increasing")
        );
        assert_eq!(
            Grid2d::new(&[0.0, 1.0], &[0.0, 0.0]).err(),
            Some("yy must be strictly increasing")
        );
    }

    #[test]
    fn new_uniform_fails_on_invalid_input() {
        assert_eq!(
            Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 1, 4).err(),
            Some("nx must be ≥ 2")
        );
        assert_eq!(
            Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 4, 1).err(),
            Some("ny must be ≥ 2")
        );
        assert_eq!(
            Grid2d::new_uniform(1.0, 0.0, 0.0, 1.0, 4, 4).err(),
            Some("xmax must be > xmin")
        );
        assert_eq!(
            Grid2d::new_uniform(0.0, 1.0, 1.0, 0.0, 4, 4).err(),
            Some("ymax must be > ymin")
        );
    }

    #[test]
    fn new_works() {
        //  8  9 10 11
        //  4  5  6  7
        //  0  1  2  3

        let xx = &[-3.0, -1.0, 1.0, 3.0];
        let yy = &[2.0, 5.0, 8.0];
        let correct_coords = &[
            (-3.0, 2.0), // 0
            (-1.0, 2.0), // 1
            (1.0, 2.0),  // 2
            (3.0, 2.0),  // 3
            //
            (-3.0, 5.0), // 4
            (-1.0, 5.0), // 5
            (1.0, 5.0),  // 6
            (3.0, 5.0),  // 7
            //
            (-3.0, 8.0), // 8
            (-1.0, 8.0), // 9
            (1.0, 8.0),  // 10
            (3.0, 8.0),  // 11
        ];

        let grid = Grid2d::new(xx, yy).unwrap();
        assert_eq!(grid.nx, 4);
        assert_eq!(grid.ny, 3);
        assert_eq!(grid.coords, correct_coords);
        assert_eq!(grid.nodes_xmin, &[0, 4, 8]);
        assert_eq!(grid.nodes_xmax, &[3, 7, 11]);
        assert_eq!(grid.nodes_ymin, &[0, 1, 2, 3]);
        assert_eq!(grid.nodes_ymax, &[8, 9, 10, 11]);

        assert_eq!(grid.nx(), 4);
        assert_eq!(grid.ny(), 3);
        assert_eq!(grid.size(), 12);

        let mut coords = Vec::new();
        grid.for_each_coord(|_m, x, y| coords.push((x, y)));
        assert_eq!(coords, correct_coords);
    }

    #[test]
    fn new_uniform_works() {
        //  8  9 10 11
        //  4  5  6  7
        //  0  1  2  3

        let xmin = -3.0;
        let xmax = 3.0;
        let ymin = 2.0;
        let ymax = 8.0;
        let nx = 4;
        let ny = 3;
        // dx = (3.0 - (-3.0)) / (4 - 1) = 6.0 / 3 = 2.0
        // dy = (8.0 - 2.0) / (3 - 1) = 6.0 / 2 = 3.0
        let correct_coords = &[
            (-3.0, 2.0), // 0
            (-1.0, 2.0), // 1
            (1.0, 2.0),  // 2
            (3.0, 2.0),  // 3
            //
            (-3.0, 5.0), // 4
            (-1.0, 5.0), // 5
            (1.0, 5.0),  // 6
            (3.0, 5.0),  // 7
            //
            (-3.0, 8.0), // 8
            (-1.0, 8.0), // 9
            (1.0, 8.0),  // 10
            (3.0, 8.0),  // 11
        ];

        let grid = Grid2d::new_uniform(xmin, xmax, ymin, ymax, nx, ny).unwrap();
        assert_eq!(grid.nx, 4);
        assert_eq!(grid.ny, 3);
        assert_eq!(grid.coords, correct_coords);
        assert_eq!(grid.nodes_xmin, &[0, 4, 8]);
        assert_eq!(grid.nodes_xmax, &[3, 7, 11]);
        assert_eq!(grid.nodes_ymin, &[0, 1, 2, 3]);
        assert_eq!(grid.nodes_ymax, &[8, 9, 10, 11]);

        assert_eq!(grid.nx(), 4);
        assert_eq!(grid.ny(), 3);
        assert_eq!(grid.size(), 12);

        let mut coords = Vec::new();
        grid.for_each_coord(|_m, x, y| coords.push((x, y)));
        assert_eq!(coords, correct_coords);

        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut bottom = Vec::new();
        let mut top = Vec::new();
        let mut xx_min = Vec::new();
        let mut xx_max = Vec::new();
        let mut yy_min = Vec::new();
        let mut yy_max = Vec::new();
        grid.for_each_node_xmin(|n| {
            left.push(*n);
            let (x, _y) = grid.coord(*n);
            xx_min.push(x);
        });
        grid.for_each_node_xmax(|n| {
            right.push(*n);
            let (x, _y) = grid.coord(*n);
            xx_max.push(x);
        });
        grid.for_each_node_ymin(|n| {
            bottom.push(*n);
            let (_x, y) = grid.coord(*n);
            yy_min.push(y);
        });
        grid.for_each_node_ymax(|n| {
            top.push(*n);
            let (_x, y) = grid.coord(*n);
            yy_max.push(y);
        });
        assert_eq!(left, &[0, 4, 8]);
        assert_eq!(right, &[3, 7, 11]);
        assert_eq!(bottom, &[0, 1, 2, 3]);
        assert_eq!(top, &[8, 9, 10, 11]);
        assert_eq!(xx_min, &[-3.0, -3.0, -3.0]);
        assert_eq!(xx_max, &[3.0, 3.0, 3.0]);
        assert_eq!(yy_min, &[2.0, 2.0, 2.0, 2.0]);
        assert_eq!(yy_max, &[8.0, 8.0, 8.0, 8.0]);
    }
}
