use crate::StrError;

/// Defines a 2D Cartesian grid
///
/// A sample grid is illustrated below:
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
/// Thus:
///
/// ```text
/// m = i + j nx
/// i = m % nx
/// j = m / nx
///
/// "%" is the modulo operator
/// "/" is the integer division operator
/// ```
pub struct Grid2d {
    /// Number of points along x (≥ 2)
    nx: usize,

    /// Number of points along y (≥ 2)
    ny: usize,

    /// Node coordinates
    coords: Vec<(f64, f64)>,

    /// Indices of nodes on the xmin edge
    nodes_xmin: Vec<usize>,

    /// Indices of nodes on the xmax edge
    nodes_xmax: Vec<usize>,

    /// Indices of nodes on the ymin edge
    nodes_ymin: Vec<usize>,

    /// Indices of nodes on the ymax edge
    nodes_ymax: Vec<usize>,
}

impl Grid2d {
    /// Allocates a new instance with given coordinates
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

    /// Allocates a new instance with uniform coordinates
    ///
    /// # Arguments
    ///
    /// * `xmin` -- min x coordinate
    /// * `xmax` -- max x coordinate
    /// * `ymin` -- min y coordinate
    /// * `ymax` -- max y coordinate
    /// * `nx` -- number of points along x (≥ 2)
    /// * `ny` -- number of points along y (≥ 2)
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

    /// Returns the number of points along x
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Returns the number of points along y
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Returns the total number of points
    pub fn size(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns the coordinates of a given node
    ///
    /// # Panics
    ///
    /// May panic if `m` is out of bounds
    pub fn coord(&self, m: usize) -> (f64, f64) {
        self.coords[m]
    }

    /// Loops over all coordinates
    ///
    /// The function is `(m, x, y)` where m is the node index and (x, y) are the coordinates
    pub fn for_each_coord(&self, mut f: impl FnMut(usize, f64, f64)) {
        for (m, (x, y)) in self.coords.iter().enumerate() {
            f(m, *x, *y);
        }
    }

    /// Loops over the indices of nodes with xmin
    pub fn for_each_node_xmin<F>(&self, mut f: F)
    where
        F: FnMut(&usize),
    {
        self.nodes_xmin.iter().for_each(|n| f(n));
    }

    /// Loops over the indices of nodes with xmax
    pub fn for_each_node_xmax<F>(&self, mut f: F)
    where
        F: FnMut(&usize),
    {
        self.nodes_xmax.iter().for_each(|n| f(n));
    }

    /// Loops over the indices of nodes with ymin
    pub fn for_each_node_ymin<F>(&self, mut f: F)
    where
        F: FnMut(&usize),
    {
        self.nodes_ymin.iter().for_each(|n| f(n));
    }

    /// Loops over the indices of nodes with ymax
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
