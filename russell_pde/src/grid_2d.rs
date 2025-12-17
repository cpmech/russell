use crate::StrError;
use russell_lab::math::chebyshev_lobatto_points;

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
    /// Minimum x-coordinate
    xmin: f64,

    /// Maximum x-coordinate
    xmax: f64,

    /// Minimum y-coordinate
    ymin: f64,

    /// Maximum y-coordinate
    ymax: f64,

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

    /// Indicates if the grid uses Chebyshev-Gauss-Lobatto points
    is_chebyshev_gauss_lobatto: bool,
}

impl Grid2d {
    /// Auxiliary function to allocate the grid
    fn do_allocate(
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        nx: usize,
        ny: usize,
        coords: Vec<(f64, f64)>,
        cgl_grid: bool,
    ) -> Self {
        Self {
            xmin,
            xmax,
            ymin,
            ymax,
            nx,
            ny,
            coords,
            nodes_xmin: (0..ny).map(|j| j * nx).collect(),
            nodes_xmax: (0..ny).map(|j| j * nx + (nx - 1)).collect(),
            nodes_ymin: (0..nx).collect(),
            nodes_ymax: (0..nx).map(|i| (ny - 1) * nx + i).collect(),
            is_chebyshev_gauss_lobatto: cgl_grid,
        }
    }

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
        let mut xmin = xx[0];
        let mut xmax = xx[0];
        let mut ymin = yy[0];
        let mut ymax = yy[0];
        for i in 1..nx {
            if xx[i] <= xx[i - 1] {
                return Err("xx must be strictly increasing");
            }
            if xx[i] < xmin {
                xmin = xx[i];
            }
            if xx[i] > xmax {
                xmax = xx[i];
            }
        }
        for j in 0..ny {
            if j > 0 && yy[j] <= yy[j - 1] {
                return Err("yy must be strictly increasing");
            }
            if yy[j] < ymin {
                ymin = yy[j];
            }
            if yy[j] > ymax {
                ymax = yy[j];
            }
            for i in 0..nx {
                coords.push((xx[i], yy[j]));
            }
        }
        Ok(Grid2d::do_allocate(xmin, xmax, ymin, ymax, nx, ny, coords, false))
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
        Ok(Grid2d::do_allocate(xmin, xmax, ymin, ymax, nx, ny, coords, false))
    }

    /// Creates a new grid using Chebyshev-Gauss-Lobatto points (tensor product)
    pub fn new_chebyshev_gauss_lobatto(
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        nx: usize,
        ny: usize,
    ) -> Result<Self, StrError> {
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
        //        xb + xa + (xb - xa) u
        // x(u) = —————————————————————
        //                 2
        //        yb + ya + (yb - ya) v
        // y(v) = —————————————————————
        //                 2
        let x_plus = (xmax + xmin) / 2.0;
        let x_minus = (xmax - xmin) / 2.0;
        let y_plus = (ymax + ymin) / 2.0;
        let y_minus = (ymax - ymin) / 2.0;
        let uu = chebyshev_lobatto_points(nx - 1);
        let vv = chebyshev_lobatto_points(ny - 1);
        let mut coords = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            let y = y_plus + y_minus * vv[j];
            for i in 0..nx {
                let x = x_plus + x_minus * uu[i];
                coords.push((x, y));
            }
        }
        Ok(Grid2d::do_allocate(xmin, xmax, ymin, ymax, nx, ny, coords, true))
    }

    /// Indicates if the grid uses Chebyshev-Gauss-Lobatto points
    pub fn is_chebyshev_gauss_lobatto(&self) -> bool {
        self.is_chebyshev_gauss_lobatto
    }

    /// Returns the minimum x-coordinate
    pub fn xmin(&self) -> f64 {
        self.xmin
    }

    /// Returns the maximum x-coordinate
    pub fn xmax(&self) -> f64 {
        self.xmax
    }

    /// Returns the minimum y-coordinate
    pub fn ymin(&self) -> f64 {
        self.ymin
    }

    /// Returns the maximum y-coordinate
    pub fn ymax(&self) -> f64 {
        self.ymax
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

    /// Returns the linear node index m for the specified (i, j) indices
    ///
    /// ```text
    /// m = i + j × nx    (convert (i,j) to linear index m)
    /// i = m % nx        (extract i-index from linear index m)
    /// j = m / nx        (extract j-index from linear index m)
    /// ```
    pub fn get_m(&self, i: usize, j: usize) -> usize {
        i + j * self.nx
    }

    /// Returns the (i, j) indices of the specified node m
    ///
    /// ```text
    /// m = i + j × nx    (convert (i,j) to linear index m)
    /// i = m % nx        (extract i-index from linear index m)
    /// j = m / nx        (extract j-index from linear index m)
    /// ```
    pub fn get_ij(&self, m: usize) -> (usize, usize) {
        let i = m % self.nx;
        let j = m / self.nx;
        (i, j)
    }

    pub fn is_xmin(&self, m: usize) -> bool {
        m % self.nx == 0 // i == 0
    }

    pub fn is_xmax(&self, m: usize) -> bool {
        m % self.nx == self.nx - 1 // i == nx - 1
    }

    pub fn is_ymin(&self, m: usize) -> bool {
        m / self.nx == 0 // j == 0
    }

    pub fn is_ymax(&self, m: usize) -> bool {
        m / self.nx == self.ny - 1 // j == ny - 1
    }

    /// Returns the spacing (dx, dy) if the grid is uniform on both directions
    ///
    /// Returns `None` if the grid is non-uniform in any direction.
    pub fn get_dx_dy(&self) -> Option<(f64, f64)> {
        let mut dx = f64::NEG_INFINITY;
        let mut dy = f64::NEG_INFINITY;
        for j in 1..self.ny {
            for i in 1..self.nx {
                let m = i + j * self.nx; // this node
                let l = m - 1; // left node
                let b = m - self.nx; // bottom node
                let (x, y) = self.coords[m];
                let (xl, _) = self.coords[l];
                let (_, yb) = self.coords[b];
                if dx == f64::NEG_INFINITY {
                    dx = x - xl;
                    assert!(dx > 0.0);
                } else if f64::abs(x - xl - dx) > 10.0 * f64::EPSILON {
                    return None; // non-uniform in x
                }
                if dy == f64::NEG_INFINITY {
                    dy = y - yb;
                    assert!(dy > 0.0);
                } else if f64::abs(y - yb - dy) > 10.0 * f64::EPSILON {
                    return None; // non-uniform in y
                }
            }
        }
        Some((dx, dy))
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

    /// Returns the boundary node indices
    ///
    /// Returns `(nodes_xmin, nodes_xmax, nodes_ymin, nodes_ymax)`
    pub fn boundary_nodes(&self) -> (&[usize], &[usize], &[usize], &[usize]) {
        (&self.nodes_xmin, &self.nodes_xmax, &self.nodes_ymin, &self.nodes_ymax)
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
    ///     let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 3.0, 4, 3)?;
    ///
    ///     // Left boundary nodes: 0, 4, 8 (marked with *)
    ///     //   *8───9──10──11
    ///     //    │   │   │   │
    ///     //   *4───5───6───7
    ///     //    │   │   │   │
    ///     //   *0───1───2───3
    ///
    ///     // Verify left boundary nodes
    ///     let mut left_nodes = Vec::new();
    ///     grid.for_each_node_xmin(|&node| {
    ///         let (x, _y) = grid.coord(node);
    ///         assert_eq!(x, 0.0); // All left boundary nodes have x = 0
    ///         left_nodes.push(node);
    ///     });
    ///     assert_eq!(left_nodes, vec![0, 4, 8]);
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
    ///     let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 3.0, 4, 3)?;
    ///
    ///     // Right boundary nodes: 3, 7, 11 (marked with *)
    ///     //    8───9──10──*11
    ///     //    │   │   │   │
    ///     //    4───5───6──*7
    ///     //    │   │   │   │
    ///     //    0───1───2──*3
    ///
    ///     // Verify right boundary nodes
    ///     let mut right_nodes = Vec::new();
    ///     grid.for_each_node_xmax(|&node| {
    ///         let (x, _y) = grid.coord(node);
    ///         assert_eq!(x, 2.0); // All right boundary nodes have x = 2.0
    ///         right_nodes.push(node);
    ///     });
    ///     assert_eq!(right_nodes, vec![3, 7, 11]);
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
    ///     let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 3.0, 4, 3)?;
    ///
    ///     // Bottom boundary nodes: 0, 1, 2, 3 (marked with *)
    ///     //    8───9──10──11
    ///     //    │   │   │   │
    ///     //    4───5───6───7
    ///     //    │   │   │   │
    ///     //   *0──*1──*2──*3
    ///
    ///     // Verify bottom boundary nodes
    ///     let mut bottom_nodes = Vec::new();
    ///     grid.for_each_node_ymin(|&node| {
    ///         let (_x, y) = grid.coord(node);
    ///         assert_eq!(y, 0.0); // All bottom boundary nodes have y = 0.0
    ///         bottom_nodes.push(node);
    ///     });
    ///     assert_eq!(bottom_nodes, vec![0, 1, 2, 3]);
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
    ///     let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 3.0, 4, 3)?;
    ///
    ///     // Top boundary nodes: 8, 9, 10, 11 (marked with *)
    ///     //   *8──*9─*10─*11
    ///     //    │   │   │   │
    ///     //    4───5───6───7
    ///     //    │   │   │   │
    ///     //    0───1───2───3
    ///
    ///     // Verify top boundary nodes
    ///     let mut top_nodes = Vec::new();
    ///     grid.for_each_node_ymax(|&node| {
    ///         let (_x, y) = grid.coord(node);
    ///         assert_eq!(y, 3.0); // All top boundary nodes have y = 3.0
    ///         top_nodes.push(node);
    ///     });
    ///     assert_eq!(top_nodes, vec![8, 9, 10, 11]);
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
    use russell_lab::approx_eq;
    use std::f64::consts::PI;

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
    fn new_chebyshev_gauss_lobatto_fails_on_invalid_input() {
        assert_eq!(
            Grid2d::new_chebyshev_gauss_lobatto(0.0, 1.0, 0.0, 1.0, 1, 4).err(),
            Some("nx must be ≥ 2")
        );
        assert_eq!(
            Grid2d::new_chebyshev_gauss_lobatto(0.0, 1.0, 0.0, 1.0, 4, 1).err(),
            Some("ny must be ≥ 2")
        );
        assert_eq!(
            Grid2d::new_chebyshev_gauss_lobatto(1.0, 0.0, 0.0, 1.0, 4, 4).err(),
            Some("xmax must be > xmin")
        );
        assert_eq!(
            Grid2d::new_chebyshev_gauss_lobatto(0.0, 1.0, 1.0, 0.0, 4, 4).err(),
            Some("ymax must be > ymin")
        );
    }

    #[test]
    fn new_works() {
        //  8  9 10 11
        //  4  5  6  7
        //  0  1  2  3

        let xx = &[-3.0, -2.9, 2.9, 3.0];
        let yy = &[2.0, 5.0, 8.0];
        let correct_coords = &[
            (-3.0, 2.0), // 0
            (-2.9, 2.0), // 1
            (2.9, 2.0),  // 2
            (3.0, 2.0),  // 3
            //
            (-3.0, 5.0), // 4
            (-2.9, 5.0), // 5
            (2.9, 5.0),  // 6
            (3.0, 5.0),  // 7
            //
            (-3.0, 8.0), // 8
            (-2.9, 8.0), // 9
            (2.9, 8.0),  // 10
            (3.0, 8.0),  // 11
        ];

        let grid = Grid2d::new(xx, yy).unwrap();
        assert_eq!(grid.xmin, -3.0);
        assert_eq!(grid.xmax, 3.0);
        assert_eq!(grid.ymin, 2.0);
        assert_eq!(grid.ymax, 8.0);
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
        assert_eq!(grid.get_dx_dy(), None); // non-uniform grid (along x)

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
        assert_eq!(grid.xmin, -3.0);
        assert_eq!(grid.xmax, 3.0);
        assert_eq!(grid.ymin, 2.0);
        assert_eq!(grid.ymax, 8.0);
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
        assert_eq!(grid.get_dx_dy(), Some((2.0, 3.0)));

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

    #[test]
    fn new_chebyshev_gauss_lobatto_works() {
        //  8  9 10 11
        //  4  5  6  7
        //  0  1  2  3

        let xmin = -3.0;
        let xmax = 3.0;
        let ymin = 2.0;
        let ymax = 8.0;
        let nx = 4;
        let ny = 3;

        //           ⎛  j⋅π  ⎞
        // Uⱼ = -cos ⎜ ————— ⎟
        //           ⎝   N   ⎠
        //
        // j = 0 ... N
        let um1 = -f64::cos(PI / 3.0); // j = 1
        let um2 = -f64::cos(2.0 * PI / 3.0); // j = 2
        let vm1 = -f64::cos(PI / 2.0); // j = 1
        let xm1 = (xmax + xmin + (xmax - xmin) * um1) / 2.0;
        let xm2 = (xmax + xmin + (xmax - xmin) * um2) / 2.0;
        let ym1 = (ymax + ymin + (ymax - ymin) * vm1) / 2.0;

        let correct_coords = &[
            (-3.0, 2.0), // 0
            (xm1, 2.0),  // 1
            (xm2, 2.0),  // 2
            (3.0, 2.0),  // 3
            //
            (-3.0, ym1), // 4
            (xm1, ym1),  // 5
            (xm2, ym1),  // 6
            (3.0, ym1),  // 7
            //
            (-3.0, 8.0), // 8
            (xm1, 8.0),  // 9
            (xm2, 8.0),  // 10
            (3.0, 8.0),  // 11
        ];

        let grid = Grid2d::new_chebyshev_gauss_lobatto(xmin, xmax, ymin, ymax, nx, ny).unwrap();

        assert_eq!(grid.xmin, -3.0);
        assert_eq!(grid.xmax, 3.0);
        assert_eq!(grid.ymin, 2.0);
        assert_eq!(grid.ymax, 8.0);
        assert_eq!(grid.nx, 4);
        assert_eq!(grid.ny, 3);
        assert_eq!(grid.nodes_xmin, &[0, 4, 8]);
        assert_eq!(grid.nodes_xmax, &[3, 7, 11]);
        assert_eq!(grid.nodes_ymin, &[0, 1, 2, 3]);
        assert_eq!(grid.nodes_ymax, &[8, 9, 10, 11]);
        for (m, &(x, y)) in correct_coords.iter().enumerate() {
            let (xg, yg) = grid.coord(m);
            approx_eq(x, xg, 1e-15);
            approx_eq(y, yg, 1e-15);
        }
    }

    #[test]
    fn get_dx_dy_works_31x31() {
        let (nx, ny) = (31, 31);
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 3.0, nx, ny).unwrap();
        assert_eq!(grid.get_dx_dy(), Some((0.1, 0.1)));
    }

    #[test]
    fn get_dx_dy_captures_non_uniform_levels() {
        //     8    9   10   11     y=8.0
        //     4    5    6    7     y=5.0
        //     0    1    2    3     y=2.0
        // x=-3.0 -1.0  1.0  3.0
        // dx = 2.0, dy = 3.0
        let mut grid = Grid2d::new_uniform(-3.0, 3.0, 2.0, 8.0, 4, 3).unwrap();
        assert_eq!(grid.nx, 4);
        assert_eq!(grid.ny, 3);
        assert_eq!(grid.get_dx_dy(), Some((2.0, 3.0)));

        // now, we must change the coordinates internally because
        // there is no way to create a non-uniform grid
        assert_eq!(grid.coords[6], (1.0, 5.0)); // before
        grid.coords[6] = (1.1, 5.0); // after (non-uniform in x)
        assert_eq!(grid.get_dx_dy(), None);

        // fixed
        grid.coords[6] = (1.0, 5.0); // after (non-uniform in x)
        assert_eq!(grid.get_dx_dy(), Some((2.0, 3.0)));

        // now, break uniformity in y
        grid.coords[6] = (1.0, 5.1); // after (non-uniform in y)
        assert_eq!(grid.get_dx_dy(), None);
    }

    #[test]
    fn get_dx_dy_uniform_grids() {
        // Test uniform grid with integer spacing
        let grid = Grid2d::new_uniform(0.0, 6.0, 0.0, 4.0, 4, 3).unwrap();
        // dx = 6.0 / 3 = 2.0, dy = 4.0 / 2 = 2.0
        assert_eq!(grid.get_dx_dy(), Some((2.0, 2.0)));

        // Test uniform grid with fractional spacing
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 5, 3).unwrap();
        // dx = 1.0 / 4 = 0.25, dy = 1.0 / 2 = 0.5
        assert_eq!(grid.get_dx_dy(), Some((0.25, 0.5)));

        // Test uniform grid with different aspect ratios
        let grid = Grid2d::new_uniform(-2.0, 2.0, -1.0, 3.0, 3, 5).unwrap();
        // dx = 4.0 / 2 = 2.0, dy = 4.0 / 4 = 1.0
        assert_eq!(grid.get_dx_dy(), Some((2.0, 1.0)));

        // Test minimal 2x2 uniform grid
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 2, 2).unwrap();
        // dx = 1.0 / 1 = 1.0, dy = 1.0 / 1 = 1.0
        assert_eq!(grid.get_dx_dy(), Some((1.0, 1.0)));
    }

    #[test]
    fn get_dx_dy_non_uniform_grids() {
        // Non-uniform in x direction
        let xx = &[0.0, 0.1, 0.5, 1.0]; // non-uniform spacing
        let yy = &[0.0, 0.5, 1.0]; // uniform spacing: dy = 0.5
        let grid = Grid2d::new(xx, yy).unwrap();
        assert_eq!(grid.get_dx_dy(), None); // should return None due to non-uniform x

        // Non-uniform in y direction
        let xx = &[0.0, 1.0, 2.0]; // uniform spacing: dx = 1.0
        let yy = &[0.0, 0.1, 1.0]; // non-uniform spacing
        let grid = Grid2d::new(xx, yy).unwrap();
        assert_eq!(grid.get_dx_dy(), None); // should return None due to non-uniform y

        // Non-uniform in both directions
        let xx = &[0.0, 0.2, 0.7, 1.0]; // non-uniform spacing
        let yy = &[0.0, 0.3, 0.8, 1.0]; // non-uniform spacing
        let grid = Grid2d::new(xx, yy).unwrap();
        assert_eq!(grid.get_dx_dy(), None); // should return None

        // Logarithmic spacing
        let xx = &[0.1, 1.0, 10.0, 100.0]; // logarithmic spacing
        let yy = &[0.01, 0.1, 1.0]; // logarithmic spacing
        let grid = Grid2d::new(xx, yy).unwrap();
        assert_eq!(grid.get_dx_dy(), None);
    }

    #[test]
    fn get_dx_dy_uniform_from_arrays() {
        // Test that manually created uniform arrays are detected as uniform
        let xx = &[0.0, 1.0, 2.0, 3.0, 4.0]; // uniform: dx = 1.0
        let yy = &[0.0, 0.5, 1.0]; // uniform: dy = 0.5
        let grid = Grid2d::new(xx, yy).unwrap();
        assert_eq!(grid.get_dx_dy(), Some((1.0, 0.5)));

        // Test with negative coordinates
        let xx = &[-2.0, -1.0, 0.0, 1.0]; // uniform: dx = 1.0
        let yy = &[-1.0, 1.0, 3.0]; // uniform: dy = 2.0
        let grid = Grid2d::new(xx, yy).unwrap();
        assert_eq!(grid.get_dx_dy(), Some((1.0, 2.0)));

        // Test with fractional uniform spacing
        let xx = &[0.0, 0.25, 0.5, 0.75, 1.0]; // uniform: dx = 0.25
        let yy = &[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]; // uniform: dy = 1/3
        let grid = Grid2d::new(xx, yy).unwrap();
        let result = grid.get_dx_dy().unwrap();
        assert!((result.0 - 0.25).abs() < 1e-15);
        assert!((result.1 - 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn get_dx_dy_precision_edge_cases() {
        // Test with very small spacing that might have floating-point precision issues
        let grid = Grid2d::new_uniform(0.0, 1e-6, 0.0, 1e-6, 3, 3).unwrap();
        let result = grid.get_dx_dy().unwrap();
        assert!((result.0 - 5e-7).abs() < 1e-21); // dx = 1e-6 / 2 = 5e-7
        assert!((result.1 - 5e-7).abs() < 1e-21); // dy = 1e-6 / 2 = 5e-7

        // Test with very large spacing
        let grid = Grid2d::new_uniform(0.0, 1e6, 0.0, 1e6, 3, 3).unwrap();
        let result = grid.get_dx_dy().unwrap();
        assert!((result.0 - 5e5).abs() < 1e-9); // dx = 1e6 / 2 = 5e5
        assert!((result.1 - 5e5).abs() < 1e-9); // dy = 1e6 / 2 = 5e5

        // Test grid that's almost uniform but has tiny differences beyond epsilon
        let mut xx = vec![0.0, 1.0, 2.0, 3.0];
        xx[2] += 11.0 * f64::EPSILON; // Add small perturbation beyond epsilon
        let yy = &[0.0, 1.0, 2.0];
        let grid = Grid2d::new(&xx, yy).unwrap();
        assert_eq!(grid.get_dx_dy(), None); // Should detect as non-uniform

        // Test grid with differences exactly at epsilon boundary
        let mut xx = vec![0.0, 1.0, 2.0, 3.0];
        xx[2] += f64::EPSILON / 2.0; // Add perturbation within epsilon tolerance
        let yy = &[0.0, 1.0, 2.0];
        let grid = Grid2d::new(&xx, yy).unwrap();
        assert_eq!(grid.get_dx_dy(), Some((1.0, 1.0))); // Should still be uniform
    }

    #[test]
    fn get_dx_dy_different_grid_sizes() {
        // Test with different grid dimensions to ensure algorithm works for all sizes

        // 2x2 grid
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 4.0, 2, 2).unwrap();
        assert_eq!(grid.get_dx_dy(), Some((3.0, 4.0)));

        // 10x2 grid (wide)
        let grid = Grid2d::new_uniform(0.0, 9.0, 0.0, 1.0, 10, 2).unwrap();
        assert_eq!(grid.get_dx_dy(), Some((1.0, 1.0)));

        // 2x10 grid (tall)
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 9.0, 2, 10).unwrap();
        assert_eq!(grid.get_dx_dy(), Some((1.0, 1.0)));

        // Large square grid
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 50, 50).unwrap();
        let result = grid.get_dx_dy().unwrap();
        assert!((result.0 - 1.0 / 49.0).abs() < 1e-15);
        assert!((result.1 - 1.0 / 49.0).abs() < 1e-15);
    }

    #[test]
    fn get_dx_dy_boundary_coordinates() {
        // Test with coordinates at domain boundaries

        // Grid spanning zero
        let grid = Grid2d::new_uniform(-1.0, 1.0, -2.0, 2.0, 3, 5).unwrap();
        assert_eq!(grid.get_dx_dy(), Some((1.0, 1.0)));

        // Grid with zero width (should fail in constructor, but testing robustness)
        // This is already tested in constructor tests, but let's test a near-zero case
        let grid = Grid2d::new_uniform(0.0, 1e-10, 0.0, 1e-10, 2, 2).unwrap();
        let result = grid.get_dx_dy().unwrap();
        assert!((result.0 - 1e-10).abs() < 1e-25);
        assert!((result.1 - 1e-10).abs() < 1e-25);

        // Grid with large coordinates
        let grid = Grid2d::new_uniform(1e6, 1e6 + 4.0, 1e9, 1e9 + 6.0, 3, 4).unwrap();
        assert_eq!(grid.get_dx_dy(), Some((2.0, 2.0)));
    }

    #[test]
    fn get_m_and_get_ij_work() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3).unwrap();
        //      i=0     i=1     i=2
        // j=2  6───────7───────8  j=2
        //      │       │       │
        // j=1  3───────4───────5  j=1
        //      │       │       │
        // j=0  0───────1───────2  j=0
        //      i=0     i=1     i=2
        assert_eq!(grid.get_m(0, 0), 0);
        assert_eq!(grid.get_m(1, 0), 1);
        assert_eq!(grid.get_m(2, 0), 2);
        assert_eq!(grid.get_m(0, 1), 3);
        assert_eq!(grid.get_m(1, 1), 4);
        assert_eq!(grid.get_m(2, 1), 5);
        assert_eq!(grid.get_m(0, 2), 6);
        assert_eq!(grid.get_m(1, 2), 7);
        assert_eq!(grid.get_m(2, 2), 8);

        assert_eq!(grid.get_ij(0), (0, 0));
        assert_eq!(grid.get_ij(1), (1, 0));
        assert_eq!(grid.get_ij(2), (2, 0));
        assert_eq!(grid.get_ij(3), (0, 1));
        assert_eq!(grid.get_ij(4), (1, 1));
        assert_eq!(grid.get_ij(5), (2, 1));
        assert_eq!(grid.get_ij(6), (0, 2));
        assert_eq!(grid.get_ij(7), (1, 2));
        assert_eq!(grid.get_ij(8), (2, 2));
    }
}
