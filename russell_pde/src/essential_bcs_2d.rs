use crate::{Grid2d, Side};
use std::collections::HashMap;
use std::sync::Arc;

/// Implements a handler for essential (Dirichlet) boundary conditions
///
/// This struct helps to manage essential boundary conditions (ebc) for 2D problems.
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

    /// Holds the functions to compute essential boundary conditions (ebc)
    ///
    /// The function is `f(x, y) -> ebc`
    ///
    /// (4) → (xmin, xmax, ymin, ymax); corresponding to the 4 sides
    functions: Vec<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'a>>,

    /// Maps node to one of the four functions in `functions`
    ///
    /// length = number of nodes with essential boundary conditions (prescribed nodes)
    node_to_function: HashMap<usize, usize>,
}

impl<'a> EssentialBcs2d<'a> {
    /// Allocates a new instance
    pub fn new() -> Self {
        EssentialBcs2d {
            periodic_along_x: false,
            periodic_along_y: false,
            functions: vec![
                Arc::new(|_, _| 0.0), // xmin
                Arc::new(|_, _| 0.0), // xmax
                Arc::new(|_, _| 0.0), // ymin
                Arc::new(|_, _| 0.0), // ymax
            ],
            node_to_function: HashMap::new(),
        }
    }

    // --------------------------------------------------------
    // setters
    // --------------------------------------------------------

    /// Sets periodic boundary condition
    ///
    /// **Note:** Any essential boundary condition on the corresponding side will be removed.
    pub fn set_periodic(&mut self, grid: &Grid2d, along_x: bool, along_y: bool) {
        self.periodic_along_x = along_x;
        self.periodic_along_y = along_y;
        if along_x {
            grid.for_each_node_xmin(|n| {
                self.node_to_function.remove(n);
            });
            grid.for_each_node_xmax(|n| {
                self.node_to_function.remove(n);
            });
        }
        if along_y {
            grid.for_each_node_ymin(|n| {
                self.node_to_function.remove(n);
            });
            grid.for_each_node_ymax(|n| {
                self.node_to_function.remove(n);
            });
        }
    }

    /// Sets essential (Dirichlet) boundary condition
    ///
    /// The function is `f(x, y) -> ebc`
    ///
    /// **Note:** Any periodic boundary condition on the corresponding side will be removed.
    pub fn set(&mut self, grid: &Grid2d, side: Side, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'a) {
        match side {
            Side::Xmin => {
                self.periodic_along_x = false;
                self.functions[0] = Arc::new(f);
                grid.for_each_node_xmin(|n| {
                    self.node_to_function.insert(*n, 0);
                });
            }
            Side::Xmax => {
                self.periodic_along_x = false;
                self.functions[1] = Arc::new(f);
                grid.for_each_node_xmax(|n| {
                    self.node_to_function.insert(*n, 1);
                });
            }
            Side::Ymin => {
                self.periodic_along_y = false;
                self.functions[2] = Arc::new(f);
                grid.for_each_node_ymin(|n| {
                    self.node_to_function.insert(*n, 2);
                });
            }
            Side::Ymax => {
                self.periodic_along_y = false;
                self.functions[3] = Arc::new(f);
                grid.for_each_node_ymax(|n| {
                    self.node_to_function.insert(*n, 3);
                });
            }
        };
    }

    /// Sets homogeneous boundary conditions (i.e., zero essential values at the borders)
    ///
    /// **Note:** Periodic boundary conditions will be removed.
    pub fn set_homogeneous(&mut self, grid: &Grid2d) {
        self.periodic_along_x = false;
        self.periodic_along_y = false;
        self.node_to_function.clear();
        self.functions = vec![
            Arc::new(|_, _| 0.0), // xmin
            Arc::new(|_, _| 0.0), // xmax
            Arc::new(|_, _| 0.0), // ymin
            Arc::new(|_, _| 0.0), // ymax
        ];
        grid.for_each_node_xmin(|n| {
            self.node_to_function.insert(*n, 0);
        });
        grid.for_each_node_xmax(|n| {
            self.node_to_function.insert(*n, 1);
        });
        grid.for_each_node_ymin(|n| {
            self.node_to_function.insert(*n, 2);
        });
        grid.for_each_node_ymax(|n| {
            self.node_to_function.insert(*n, 3);
        });
    }

    /// Indicates whether the boundary conditions are periodic along x
    pub fn is_periodic_along_x(&self) -> bool {
        self.periodic_along_x
    }

    /// Indicates whether the boundary conditions are periodic along y
    pub fn is_periodic_along_y(&self) -> bool {
        self.periodic_along_y
    }

    /// Returns the prescribed value for the given node
    ///
    /// # Panics
    ///
    /// A panic may occur if the index is out of bounds.
    pub fn get_prescribed_value(&self, m: usize, x: f64, y: f64) -> f64 {
        let index = self.node_to_function.get(&m).unwrap();
        (self.functions[*index])(x, y)
    }

    /// Returns the list of nodes with essential boundary conditions (prescribed nodes)
    pub fn get_p_list(&self) -> Vec<usize> {
        self.node_to_function.keys().copied().collect()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EssentialBcs2d;
    use crate::{Grid2d, Side};

    const LEF: f64 = 1.0;
    const RIG: f64 = 2.0;
    const BOT: f64 = 3.0;
    const TOP: f64 = 4.0;

    #[test]
    fn new_default_and_set_periodic_work() {
        //  8  9 10 11
        //  4  5  6  7
        //  0  1  2  3
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.periodic_along_y, false);
        assert_eq!(ebcs.functions.len(), 4);
        assert_eq!(ebcs.node_to_function.len(), 0);
        for i in 0..4 {
            assert_eq!((ebcs.functions[i])(0.0, 0.0), 0.0);
        }

        ebcs.set_periodic(&grid, true, false);
        assert_eq!(ebcs.periodic_along_x, true);
        assert_eq!(ebcs.periodic_along_y, false);
        ebcs.set_periodic(&grid, false, true);
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.periodic_along_y, true);
        ebcs.set_periodic(&grid, false, false);
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.periodic_along_y, false);
    }

    #[test]
    fn all_functionality_works() {
        // --- default: no essential boundary conditions ---

        // 12 13 14 15
        //  8  9 10 11
        //  4  5  6  7
        //  0  1  2  3
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 4, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new();
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.periodic_along_y, false);
        assert_eq!(ebcs.functions.len(), 4);
        assert_eq!(ebcs.node_to_function.len(), 0);

        // -- set essential boundary conditions on all sides --

        // 12* 13* 14* 15*
        //  8*  9  10  11*
        //  4*  5   6   7*
        //  0*  1*  2*  3*
        let lef = |_, _| LEF;
        let rig = |_, _| RIG;
        let bot = |_, _| BOT;
        let top = |_, _| TOP;

        ebcs.set(&grid, Side::Xmin, lef);
        let mut res: Vec<_> = ebcs.node_to_function.keys().copied().collect();
        res.sort();
        assert_eq!(res, vec![0, 4, 8, 12]);

        ebcs.set(&grid, Side::Xmax, rig);
        let mut res: Vec<_> = ebcs.node_to_function.keys().copied().collect();
        res.sort();
        assert_eq!(res, vec![0, 3, 4, 7, 8, 11, 12, 15]);

        ebcs.set(&grid, Side::Ymin, bot);
        let mut res: Vec<_> = ebcs.node_to_function.keys().copied().collect();
        res.sort();
        assert_eq!(res, vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 15]);

        ebcs.set(&grid, Side::Ymax, top);
        let mut res: Vec<_> = ebcs.node_to_function.keys().copied().collect();
        res.sort();
        assert_eq!(res, vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]);

        // --- check getters ---

        assert_eq!(ebcs.is_periodic_along_x(), false);
        assert_eq!(ebcs.is_periodic_along_y(), false);
    }

    #[test]
    fn set_homogeneous_works() {
        // 12 13 14 15
        //  8  9 10 11
        //  4  5  6  7
        //  0  1  2  3
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // First set some non-homogeneous conditions
        ebcs.set(&grid, Side::Xmin, |_, _| 10.0);
        ebcs.set_periodic(&grid, true, false);
        assert_eq!(ebcs.periodic_along_x, true);

        // Set homogeneous - should clear periodic and set all boundaries to zero
        ebcs.set_homogeneous(&grid);

        // Check that periodic flags are cleared
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.periodic_along_y, false);

        // Check that all boundary nodes are prescribed
        let mut prescribed_nodes: Vec<_> = ebcs.node_to_function.keys().copied().collect();
        prescribed_nodes.sort();
        assert_eq!(prescribed_nodes, vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]);

        // Check that all functions return 0.0
        for i in 0..4 {
            assert_eq!((ebcs.functions[i])(1.5, 2.5), 0.0);
        }

        // Check prescribed values at boundaries
        assert_eq!(ebcs.get_prescribed_value(0, 0.0, 0.0), 0.0); // corner
        assert_eq!(ebcs.get_prescribed_value(1, 1.0, 0.0), 0.0); // bottom edge
        assert_eq!(ebcs.get_prescribed_value(4, 0.0, 1.0), 0.0); // left edge
        assert_eq!(ebcs.get_prescribed_value(15, 3.0, 3.0), 0.0); // top-right corner
    }

    #[test]
    fn get_prescribed_value_works() {
        // 8  9 10 11
        // 4  5  6  7
        // 0  1  2  3
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 2.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Set boundary conditions with coordinate-dependent functions
        ebcs.set(&grid, Side::Xmin, |x, y| x + y); // left: f(x,y) = x + y
        ebcs.set(&grid, Side::Xmax, |x, y| x * y); // right: f(x,y) = x * y
        ebcs.set(&grid, Side::Ymin, |x, y| x - y); // bottom: f(x,y) = x - y
        ebcs.set(&grid, Side::Ymax, |x, y| 2.0 * x + y); // top: f(x,y) = 2x + y

        // Test left boundary (x = 0)
        assert_eq!(ebcs.get_prescribed_value(0, 0.0, 0.0), 0.0); // 0 + 0 = 0
        assert_eq!(ebcs.get_prescribed_value(4, 0.0, 1.0), 1.0); // 0 + 1 = 1
        assert_eq!(ebcs.get_prescribed_value(8, 0.0, 2.0), 2.0); // 0 + 2 = 2

        // Test right boundary (x = 3)
        assert_eq!(ebcs.get_prescribed_value(3, 3.0, 0.0), 3.0); // overridden by bottom: 3 - 0 = 3
        assert_eq!(ebcs.get_prescribed_value(7, 3.0, 1.0), 3.0); // 3 * 1 = 3
        assert_eq!(ebcs.get_prescribed_value(11, 3.0, 2.0), 8.0); // overridden by top: 2*3 + 2 = 8

        // Test bottom boundary (y = 0)
        assert_eq!(ebcs.get_prescribed_value(1, 1.0, 0.0), 1.0); // 1 - 0 = 1
        assert_eq!(ebcs.get_prescribed_value(2, 2.0, 0.0), 2.0); // 2 - 0 = 2

        // Test top boundary (y = 2)
        assert_eq!(ebcs.get_prescribed_value(9, 1.0, 2.0), 4.0); // 2*1 + 2 = 4
        assert_eq!(ebcs.get_prescribed_value(10, 2.0, 2.0), 6.0); // 2*2 + 2 = 6
    }

    #[test]
    fn get_p_list_works() {
        // 8  9 10 11
        // 4  5  6  7
        // 0  1  2  3
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 2.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Initially no prescribed nodes
        assert_eq!(ebcs.get_p_list().len(), 0);

        // Set left boundary
        ebcs.set(&grid, Side::Xmin, |_, _| 1.0);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0, 4, 8]);

        // Add bottom boundary
        ebcs.set(&grid, Side::Ymin, |_, _| 2.0);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0, 1, 2, 3, 4, 8]);

        // Add all boundaries
        ebcs.set(&grid, Side::Xmax, |_, _| 3.0);
        ebcs.set(&grid, Side::Ymax, |_, _| 4.0);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0, 1, 2, 3, 4, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn periodic_boundary_conditions_work() {
        // 8  9 10 11
        // 4  5  6  7
        // 0  1  2  3
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 2.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Set some essential BCs first
        ebcs.set(&grid, Side::Xmin, |_, _| 1.0);
        ebcs.set(&grid, Side::Ymin, |_, _| 2.0);
        assert_eq!(ebcs.node_to_function.len(), 6); // 0,1,2,3,4,8

        // Set periodic along x - should remove xmin/xmax prescribed nodes
        ebcs.set_periodic(&grid, true, false);
        assert!(ebcs.is_periodic_along_x());
        assert!(!ebcs.is_periodic_along_y());

        // Should have removed left boundary nodes but kept bottom
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![1, 2]); // only bottom boundary with 0 < x < 3 remains

        // Set periodic along y - should remove ymin/ymax prescribed nodes
        ebcs.set_periodic(&grid, false, true);
        assert!(!ebcs.is_periodic_along_x());
        assert!(ebcs.is_periodic_along_y());

        // Should have removed bottom boundary nodes
        assert_eq!(ebcs.get_p_list().len(), 0);

        // Set both periodic
        ebcs.set(&grid, Side::Xmin, |_, _| 1.0); // add some BCs back
        ebcs.set(&grid, Side::Ymax, |_, _| 2.0);
        ebcs.set_periodic(&grid, true, true);
        assert!(ebcs.is_periodic_along_x());
        assert!(ebcs.is_periodic_along_y());
        assert_eq!(ebcs.get_p_list().len(), 0); // all BCs should be removed
    }

    #[test]
    fn essential_bc_overrides_periodic() {
        let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Set periodic along x
        ebcs.set_periodic(&grid, true, false);
        assert!(ebcs.is_periodic_along_x());

        // Set essential BC on left - should disable periodic along x
        ebcs.set(&grid, Side::Xmin, |_, _| 5.0);
        assert!(!ebcs.is_periodic_along_x());

        // Set periodic along y
        ebcs.set_periodic(&grid, false, true);
        assert!(ebcs.is_periodic_along_y());

        // Set essential BC on bottom - should disable periodic along y
        ebcs.set(&grid, Side::Ymin, |_, _| 10.0);
        assert!(!ebcs.is_periodic_along_y());
    }

    #[test]
    fn single_side_boundary_conditions() {
        // Test setting BCs on individual sides
        let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Only left boundary
        ebcs.set(&grid, Side::Xmin, |_, y| y);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0, 3, 6]); // left column

        // Clear and set only right boundary
        ebcs = EssentialBcs2d::new();
        ebcs.set(&grid, Side::Xmax, |_, y| 2.0 * y);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![2, 5, 8]); // right column

        // Clear and set only bottom boundary
        ebcs = EssentialBcs2d::new();
        ebcs.set(&grid, Side::Ymin, |x, _| x);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0, 1, 2]); // bottom row

        // Clear and set only top boundary
        ebcs = EssentialBcs2d::new();
        ebcs.set(&grid, Side::Ymax, |x, _| 3.0 * x);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![6, 7, 8]); // top row
    }

    #[test]
    fn complex_boundary_functions() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Set complex functions on each boundary
        use std::f64::consts::PI;

        ebcs.set(&grid, Side::Xmin, |_x, y| (PI * y).sin()); // sin(πy) on left
        ebcs.set(&grid, Side::Xmax, |_x, y| (PI * y).cos()); // cos(πy) on right
        ebcs.set(&grid, Side::Ymin, |x, _y| x * x); // x² on bottom
        ebcs.set(&grid, Side::Ymax, |x, _y| (x - 0.5).abs()); // |x-0.5| on top

        // Test prescribed values
        assert_eq!(ebcs.get_prescribed_value(0, 0.0, 0.0), 0.0); // sin(0) = 0
        assert_eq!(ebcs.get_prescribed_value(3, 0.0, 0.5), 1.0); // sin(π/2) = 1
        assert_eq!(ebcs.get_prescribed_value(2, 1.0, 0.0), 1.0); // cos(0) = 1
        assert_eq!(ebcs.get_prescribed_value(1, 0.5, 0.0), 0.25); // (0.5)² = 0.25
        assert_eq!(ebcs.get_prescribed_value(7, 0.5, 1.0), 0.0); // |0.5-0.5| = 0
    }

    #[test]
    fn boundary_conditions_on_small_grid() {
        // Test with minimal 2x2 grid
        // 2  3
        // 0  1
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 2, 2).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        ebcs.set_homogeneous(&grid);

        // All nodes should be prescribed (boundary nodes only)
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0, 1, 2, 3]);

        // All values should be zero
        for &node in &p_list {
            let (x, y) = match node {
                0 => (0.0, 0.0),
                1 => (1.0, 0.0),
                2 => (0.0, 1.0),
                3 => (1.0, 1.0),
                _ => unreachable!(),
            };
            assert_eq!(ebcs.get_prescribed_value(node, x, y), 0.0);
        }
    }

    #[test]
    fn boundary_conditions_on_large_grid() {
        // Test with larger grid to ensure scalability
        let grid = Grid2d::new_uniform(0.0, 10.0, 0.0, 8.0, 11, 9).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Set different values on each boundary
        ebcs.set(&grid, Side::Xmin, |_, _| 1.0); // left = 1
        ebcs.set(&grid, Side::Xmax, |_, _| 2.0); // right = 2
        ebcs.set(&grid, Side::Ymin, |_, _| 3.0); // bottom = 3
        ebcs.set(&grid, Side::Ymax, |_, _| 4.0); // top = 4

        let p_list = ebcs.get_p_list();

        // Should have all boundary nodes
        // Left: 9 nodes, Right: 9 nodes, Bottom: 11 nodes, Top: 11 nodes
        // But corners are counted multiple times, so: 9 + 9 + 11 + 11 - 4 = 36
        assert_eq!(p_list.len(), 36);

        // Check that we don't have any interior nodes
        for &node in &p_list {
            let i = node % 11; // x-index
            let j = node / 11; // y-index

            // Should be on at least one boundary
            assert!(i == 0 || i == 10 || j == 0 || j == 8);
        }
    }

    #[test]
    fn updating_boundary_conditions() {
        let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Set initial BC
        ebcs.set(&grid, Side::Xmin, |_, _| 1.0);
        assert_eq!(ebcs.get_prescribed_value(0, 0.0, 0.0), 1.0);

        // Update with new function
        ebcs.set(&grid, Side::Xmin, |_, y| 2.0 * y);
        assert_eq!(ebcs.get_prescribed_value(0, 0.0, 0.0), 0.0); // 2 * 0 = 0
        assert_eq!(ebcs.get_prescribed_value(3, 0.0, 1.0), 2.0); // 2 * 1 = 2

        // Add more boundaries and then clear with homogeneous
        ebcs.set(&grid, Side::Xmax, |_, _| 5.0);
        ebcs.set(&grid, Side::Ymin, |_, _| 10.0);

        ebcs.set_homogeneous(&grid);

        // All should now be zero
        for node in ebcs.get_p_list() {
            let i = node % 3;
            let j = node / 3;
            let x = i as f64;
            let y = j as f64;
            assert_eq!(ebcs.get_prescribed_value(node, x, y), 0.0);
        }
    }

    #[test]
    fn mixed_periodic_and_essential_conditions() {
        let grid = Grid2d::new_uniform(0.0, 4.0, 0.0, 3.0, 5, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Set periodic along x only
        ebcs.set_periodic(&grid, true, false);

        // Set essential BCs on y boundaries (should work with x-periodic)
        ebcs.set(&grid, Side::Ymin, |x, _| x);
        ebcs.set(&grid, Side::Ymax, |x, _| 2.0 * x);

        assert!(ebcs.is_periodic_along_x());
        assert!(!ebcs.is_periodic_along_y());

        // Should only have top and bottom boundary nodes
        let mut p_list = ebcs.get_p_list();
        p_list.sort();

        // Bottom row: 0,1,2,3,4 and top row: 15,16,17,18,19
        assert_eq!(p_list, vec![0, 1, 2, 3, 4, 15, 16, 17, 18, 19]);

        // Test prescribed values
        assert_eq!(ebcs.get_prescribed_value(2, 2.0, 0.0), 2.0); // x on bottom
        assert_eq!(ebcs.get_prescribed_value(17, 2.0, 3.0), 4.0); // 2x on top
    }

    #[test]
    #[should_panic]
    fn get_prescribed_value_panics_on_non_prescribed_node() {
        let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Set only left boundary
        ebcs.set(&grid, Side::Xmin, |_, _| 1.0);

        // Try to get prescribed value for interior node (should panic)
        let _ = ebcs.get_prescribed_value(4, 1.0, 1.0); // interior node
    }

    #[test]
    fn coordinate_dependent_functions_precision() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 101, 101).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        // Set precise mathematical functions
        ebcs.set(&grid, Side::Xmin, |_x, y| y * y * y); // y³ on left
        ebcs.set(&grid, Side::Ymin, |x, _y| (x * std::f64::consts::PI).sin()); // sin(πx) on bottom

        // Test at specific coordinates
        assert_eq!(ebcs.get_prescribed_value(0, 0.0, 0.0), 0.0); // 0³ = 0

        let node_at_quarter_left = 25 * 101; // y = 0.25 on left boundary
        let expected = 0.25_f64.powi(3); // 0.25³
        assert_eq!(ebcs.get_prescribed_value(node_at_quarter_left, 0.0, 0.25), expected);

        let node_at_half_bottom = 50; // x = 0.5 on bottom boundary
        let expected = (0.5 * std::f64::consts::PI).sin(); // sin(π/2) = 1
        assert!((ebcs.get_prescribed_value(node_at_half_bottom, 0.5, 0.0) - expected).abs() < 1e-15);
    }
}
