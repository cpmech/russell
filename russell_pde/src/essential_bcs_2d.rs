use crate::{Grid2d, Side, StrError};
use std::collections::HashMap;
use std::sync::Arc;

/// Implements a handler for essential (Dirichlet) boundary conditions
///
/// This struct helps to manage essential boundary conditions (ebc) for 2D problems.
/// It holds the number of prescribed equations and the number of unknown equations.
///
/// The grid is assumed to be a regular Cartesian grid with `nx` points along x and `ny` points along y.
pub struct EssentialBcs2d<'a> {
    /// Holds the 2D grid
    grid: Grid2d,

    /// Flags equations with prescribed EBC values
    ///
    /// length = total number of nodes in the grid
    is_prescribed: Vec<bool>,

    /// Maps (global) node ID to the prescribed index (local)
    ///
    /// length = total number of nodes in the grid
    index_prescribed: Vec<usize>,

    /// Maps (global) node ID to the unknown index (local)
    ///
    /// length = total number of nodes in the grid
    index_unknown: Vec<usize>,

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

    /// Collects the essential boundary conditions
    ///
    /// Maps node to one of the four functions in `functions`
    ///
    /// length = number of nodes with essential boundary conditions (prescribed nodes)
    node_to_function: HashMap<usize, usize>,

    /// Holds the sorted indices of the equations with essential boundary conditions
    prescribed_sorted: Vec<usize>,

    /// Holds the sorted indices of the equations without essential boundary conditions
    unknown_sorted: Vec<usize>,
}

impl<'a> EssentialBcs2d<'a> {
    /// Allocates a new instance
    pub fn new(grid: Grid2d) -> Self {
        let dim = grid.size();
        EssentialBcs2d {
            grid,
            is_prescribed: vec![false; dim],
            index_prescribed: vec![usize::MAX; dim],
            index_unknown: vec![usize::MAX; dim],
            periodic_along_x: false,
            periodic_along_y: false,
            functions: vec![
                Arc::new(|_, _| 0.0), // xmin
                Arc::new(|_, _| 0.0), // xmax
                Arc::new(|_, _| 0.0), // ymin
                Arc::new(|_, _| 0.0), // ymax
            ],
            node_to_function: HashMap::new(),
            prescribed_sorted: Vec::new(),
            unknown_sorted: (0..dim).collect(), // Initialize with all node indices
        }
    }

    // --------------------------------------------------------
    // setters
    // --------------------------------------------------------

    /// Recomputes the internal arrays after any change
    fn recompute_arrays(&mut self) {
        self.index_prescribed.fill(usize::MAX);
        self.index_unknown.fill(usize::MAX);
        self.unknown_sorted.clear();
        let dim = self.grid.size();
        let mut ip = 0;
        let mut iu = 0;
        for m in 0..dim {
            if self.node_to_function.contains_key(&m) {
                self.index_prescribed[m] = ip;
                self.is_prescribed[m] = true;
                ip += 1;
            } else {
                self.index_unknown[m] = iu;
                self.is_prescribed[m] = false;
                self.unknown_sorted.push(m); // already in sorted order due to loop order
                iu += 1;
            }
        }
        self.prescribed_sorted = self.node_to_function.keys().copied().collect();
        self.prescribed_sorted.sort(); // need this since HashMap keys are unordered
    }

    /// Sets periodic boundary condition
    ///
    /// **Note:** Any essential boundary condition on the corresponding side will be removed.
    pub fn set_periodic(&mut self, along_x: bool, along_y: bool) {
        self.periodic_along_x = along_x;
        self.periodic_along_y = along_y;
        if along_x {
            self.grid.for_each_node_xmin(|n| {
                self.node_to_function.remove(n);
            });
            self.grid.for_each_node_xmax(|n| {
                self.node_to_function.remove(n);
            });
        }
        if along_y {
            self.grid.for_each_node_ymin(|n| {
                self.node_to_function.remove(n);
            });
            self.grid.for_each_node_ymax(|n| {
                self.node_to_function.remove(n);
            });
        }
        self.recompute_arrays();
    }

    /// Sets essential (Dirichlet) boundary condition
    ///
    /// The function is `f(x, y) -> ebc`
    ///
    /// **Note:** Any periodic boundary condition on the corresponding side will be removed.
    pub fn set(&mut self, side: Side, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'a) {
        match side {
            Side::Xmin => {
                self.periodic_along_x = false;
                self.functions[0] = Arc::new(f);
                self.grid.for_each_node_xmin(|n| {
                    self.node_to_function.insert(*n, 0);
                });
            }
            Side::Xmax => {
                self.periodic_along_x = false;
                self.functions[1] = Arc::new(f);
                self.grid.for_each_node_xmax(|n| {
                    self.node_to_function.insert(*n, 1);
                });
            }
            Side::Ymin => {
                self.periodic_along_y = false;
                self.functions[2] = Arc::new(f);
                self.grid.for_each_node_ymin(|n| {
                    self.node_to_function.insert(*n, 2);
                });
            }
            Side::Ymax => {
                self.periodic_along_y = false;
                self.functions[3] = Arc::new(f);
                self.grid.for_each_node_ymax(|n| {
                    self.node_to_function.insert(*n, 3);
                });
            }
        };
        self.recompute_arrays();
    }

    /// Sets homogeneous boundary conditions (i.e., zero essential values at the borders)
    ///
    /// **Note:** Periodic boundary conditions will be removed.
    pub fn set_homogeneous(&mut self) {
        self.periodic_along_x = false;
        self.periodic_along_y = false;
        self.node_to_function.clear();
        self.functions = vec![
            Arc::new(|_, _| 0.0), // xmin
            Arc::new(|_, _| 0.0), // xmax
            Arc::new(|_, _| 0.0), // ymin
            Arc::new(|_, _| 0.0), // ymax
        ];
        self.grid.for_each_node_xmin(|n| {
            self.node_to_function.insert(*n, 0);
        });
        self.grid.for_each_node_xmax(|n| {
            self.node_to_function.insert(*n, 1);
        });
        self.grid.for_each_node_ymin(|n| {
            self.node_to_function.insert(*n, 2);
        });
        self.grid.for_each_node_ymax(|n| {
            self.node_to_function.insert(*n, 3);
        });
        self.recompute_arrays();
    }

    // --------------------------------------------------------
    // getters
    // --------------------------------------------------------

    /// Returns a reference to the grid
    pub fn get_grid(&self) -> &Grid2d {
        &self.grid
    }

    /// Indicates whether the boundary conditions are periodic along x
    pub fn is_periodic_along_x(&self) -> bool {
        self.periodic_along_x
    }

    /// Indicates whether the boundary conditions are periodic along y
    pub fn is_periodic_along_y(&self) -> bool {
        self.periodic_along_y
    }

    /// Indicates whether a node has a prescribed value or not
    pub fn is_prescribed(&self, m: usize) -> bool {
        self.is_prescribed[m]
    }

    /// Returns the local index of the prescribed node
    pub fn get_index_prescribed(&self, m: usize) -> Result<usize, StrError> {
        if self.index_prescribed[m] == usize::MAX {
            Err("node is not a prescribed node")
        } else {
            Ok(self.index_prescribed[m])
        }
    }

    /// Returns the local index of the unknown node
    pub fn get_index_unknown(&self, m: usize) -> Result<usize, StrError> {
        if self.index_unknown[m] == usize::MAX {
            Err("node is not a unknown node")
        } else {
            Ok(self.index_unknown[m])
        }
    }

    /// Returns the total number of equations (nodes)
    ///
    /// This is equal to the total number of nodes in the grid and equal
    /// to `num_prescribed + num_unknown`.
    pub fn num_total(&self) -> usize {
        self.grid.size()
    }

    /// Returns the number of prescribed equations
    ///
    /// The number of prescribed equations is equal to the number of nodes with essential conditions.
    pub fn num_prescribed(&self) -> usize {
        self.node_to_function.len()
    }

    /// Returns the number of unknown equations
    pub fn num_unknown(&self) -> usize {
        self.unknown_sorted.len()
    }

    /// Returns the (sorted) indices of the nodes with prescribed values
    pub fn get_nodes_prescribed(&self) -> &Vec<usize> {
        &self.prescribed_sorted
    }

    /// Returns the (sorted) indices of the nodes with unknown values
    pub fn get_nodes_unknown(&self) -> &Vec<usize> {
        &self.unknown_sorted
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

    /// Applies a function to each prescribed node
    ///
    /// The function is `f(ip, m, x, y, value)` where:
    ///
    /// * `ip`: the index of the prescribed node in the sorted list of prescribed nodes
    /// * `m`: the global index of the node
    /// * `x`: the x-coordinate of the node
    /// * `y`: the y-coordinate of the node
    /// * `value`: the prescribed value at the node
    pub fn for_each_prescribed_node<F>(&self, mut f: F)
    where
        F: FnMut(usize, usize, f64, f64, f64),
    {
        self.prescribed_sorted.iter().enumerate().for_each(|(ip, &m)| {
            let (x, y) = self.grid.coord(m);
            let value = self.get_prescribed_value(m, x, y);
            f(ip, m, x, y, value);
        });
    }

    /// Applies a function to each unknown node
    ///
    /// The function is `f(iu, m, x, y)` where:
    ///
    /// * `iu`: the index of the unknown node in the sorted list of unknown nodes
    /// * `m`: the global index of the node
    /// * `x`: the x-coordinate of the node
    /// * `y`: the y-coordinate of the node
    pub fn for_each_unknown_node<F>(&self, mut f: F)
    where
        F: FnMut(usize, usize, f64, f64),
    {
        self.unknown_sorted.iter().enumerate().for_each(|(iu, &m)| {
            let (x, y) = self.grid.coord(m);
            f(iu, m, x, y);
        });
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
        let mut ebcs = EssentialBcs2d::new(grid);
        assert_eq!(&ebcs.is_prescribed, &vec![false; 12]);
        assert_eq!(&ebcs.index_prescribed, &vec![usize::MAX; 12]);
        assert_eq!(&ebcs.index_unknown, &vec![usize::MAX; 12]);
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.periodic_along_y, false);
        assert_eq!(ebcs.functions.len(), 4);
        assert_eq!(ebcs.node_to_function.len(), 0);
        assert_eq!(ebcs.prescribed_sorted.len(), 0);
        assert_eq!(&ebcs.unknown_sorted, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        for i in 0..4 {
            assert_eq!((ebcs.functions[i])(0.0, 0.0), 0.0);
        }

        ebcs.set_periodic(true, false);
        assert_eq!(ebcs.periodic_along_x, true);
        assert_eq!(ebcs.periodic_along_y, false);
        ebcs.set_periodic(false, true);
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.periodic_along_y, true);
        ebcs.set_periodic(false, false);
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
        let mut ebcs = EssentialBcs2d::new(grid);
        assert_eq!(&ebcs.is_prescribed, &vec![false; 16]);
        assert_eq!(&ebcs.index_prescribed, &vec![usize::MAX; 16]);
        assert_eq!(&ebcs.index_unknown, &vec![usize::MAX; 16]);
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.periodic_along_y, false);
        assert_eq!(ebcs.functions.len(), 4);
        assert_eq!(ebcs.node_to_function.len(), 0);
        assert_eq!(ebcs.prescribed_sorted.len(), 0);
        assert_eq!(
            &ebcs.unknown_sorted,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );

        // -- set essential boundary conditions on all sides --

        // 12* 13* 14* 15*
        //  8*  9  10  11*
        //  4*  5   6   7*
        //  0*  1*  2*  3*
        let lef = |_, _| LEF;
        let rig = |_, _| RIG;
        let bot = |_, _| BOT;
        let top = |_, _| TOP;

        ebcs.set(Side::Xmin, lef);
        assert_eq!(ebcs.prescribed_sorted, vec![0, 4, 8, 12]);
        ebcs.for_each_prescribed_node(|ip, m, _, _, _| {
            assert_eq!(ebcs.index_prescribed[m], ip);
            assert_eq!(ebcs.index_unknown[m], usize::MAX);
        });
        ebcs.for_each_unknown_node(|iu, m, _, _| {
            assert_eq!(ebcs.index_prescribed[m], usize::MAX);
            assert_eq!(ebcs.index_unknown[m], iu);
        });

        ebcs.set(Side::Xmax, rig);
        assert_eq!(ebcs.prescribed_sorted, vec![0, 3, 4, 7, 8, 11, 12, 15]);
        ebcs.for_each_prescribed_node(|ip, m, _, _, _| {
            assert_eq!(ebcs.index_prescribed[m], ip);
            assert_eq!(ebcs.index_unknown[m], usize::MAX);
        });
        ebcs.for_each_unknown_node(|iu, m, _, _| {
            assert_eq!(ebcs.index_prescribed[m], usize::MAX);
            assert_eq!(ebcs.index_unknown[m], iu);
        });

        ebcs.set(Side::Ymin, bot);
        assert_eq!(ebcs.prescribed_sorted, vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 15]);
        ebcs.for_each_prescribed_node(|ip, m, _, _, _| {
            assert_eq!(ebcs.index_prescribed[m], ip);
            assert_eq!(ebcs.index_unknown[m], usize::MAX);
        });
        ebcs.for_each_unknown_node(|iu, m, _, _| {
            assert_eq!(ebcs.index_prescribed[m], usize::MAX);
            assert_eq!(ebcs.index_unknown[m], iu);
        });

        ebcs.set(Side::Ymax, top);
        assert_eq!(ebcs.prescribed_sorted, vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]);
        assert_eq!(
            &ebcs.index_prescribed,
            &vec![
                0,          // 0*
                1,          // 1*
                2,          // 2*
                3,          // 3*
                4,          // 4*
                usize::MAX, // 5
                usize::MAX, // 6
                5,          // 7*
                6,          // 8*
                usize::MAX, // 9
                usize::MAX, // 10
                7,          // 11*
                8,          // 12*
                9,          // 13*
                10,         // 14*
                11,         // 15*
            ]
        );
        assert_eq!(
            &ebcs.index_unknown,
            &vec![
                usize::MAX, // 0*
                usize::MAX, // 1*
                usize::MAX, // 2*
                usize::MAX, // 3*
                usize::MAX, // 4*
                0,          // 5
                1,          // 6
                usize::MAX, // 7*
                usize::MAX, // 8*
                2,          // 9
                3,          // 10
                usize::MAX, // 11*
                usize::MAX, // 12*
                usize::MAX, // 13*
                usize::MAX, // 14*
                usize::MAX, // 15*
            ]
        );

        // --- check getters ---

        assert_eq!(ebcs.get_grid().size(), 16);
        assert_eq!(ebcs.is_periodic_along_x(), false);
        assert_eq!(ebcs.is_periodic_along_y(), false);
        assert_eq!(ebcs.is_prescribed(0), true);
        assert_eq!(ebcs.is_prescribed(15), true);
        assert_eq!(ebcs.is_prescribed(5), false);
        assert_eq!(ebcs.get_index_prescribed(0), Ok(0));
        assert_eq!(ebcs.get_index_prescribed(15), Ok(11));
        assert_eq!(ebcs.get_index_prescribed(5), Err("node is not a prescribed node"));
        assert_eq!(ebcs.get_index_unknown(0), Err("node is not a unknown node"));
        assert_eq!(ebcs.get_index_unknown(5), Ok(0));
        assert_eq!(ebcs.get_index_unknown(10), Ok(3));
        assert_eq!(ebcs.num_prescribed(), 12);
        assert_eq!(ebcs.num_unknown(), 4);
        assert_eq!(
            ebcs.get_nodes_prescribed(),
            &vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]
        );
        assert_eq!(ebcs.get_nodes_unknown(), &vec![5, 6, 9, 10]);
        let mut res = Vec::new();
        ebcs.prescribed_sorted.iter().for_each(|&m| {
            let value = ebcs.get_prescribed_value(m, 0.0, 0.0); // x and y do not matter here
            res.push((m, value));
        });
        assert_eq!(
            res,
            &[
                (0, BOT),  // bottom* and left  (wins*)
                (1, BOT),  // bottom
                (2, BOT),  // bottom
                (3, BOT),  // bottom* and right
                (4, LEF),  // left
                (7, RIG),  // right
                (8, LEF),  // left
                (11, RIG), // right
                (12, TOP), // top* and left
                (13, TOP), // top
                (14, TOP), // top
                (15, TOP), // top* and right
            ]
        );
        assert_eq!(ebcs.num_prescribed(), 12);
        let correct_prescribed = vec![
            true,  // 0
            true,  // 1
            true,  // 2
            true,  // 3
            true,  // 4
            false, // 5
            false, // 6
            true,  // 7
            true,  // 8
            false, // 9
            false, // 10
            true,  // 11
            true,  // 12
            true,  // 13
            true,  // 14
            true,  // 15
        ];
        assert_eq!(&ebcs.is_prescribed, &correct_prescribed);
        let mut res = Vec::new();
        ebcs.for_each_prescribed_node(|ip, m, x, y, value| {
            assert_eq!(ebcs.index_prescribed[m], ip);
            assert_eq!(ebcs.index_unknown[m], usize::MAX);
            res.push((ip, m, x, y, value));
        });
        assert_eq!(
            &res,
            &[
                (0, 0, 0.0, 0.0, BOT),
                (1, 1, 1.0 / 3.0, 0.0, BOT),
                (2, 2, 2.0 / 3.0, 0.0, BOT),
                (3, 3, 1.0, 0.0, BOT),
                (4, 4, 0.0, 1.0 / 3.0, LEF),
                (5, 7, 1.0, 1.0 / 3.0, RIG),
                (6, 8, 0.0, 2.0 / 3.0, LEF),
                (7, 11, 1.0, 2.0 / 3.0, RIG),
                (8, 12, 0.0, 1.0, TOP),
                (9, 13, 1.0 / 3.0, 1.0, TOP),
                (10, 14, 2.0 / 3.0, 1.0, TOP),
                (11, 15, 1.0, 1.0, TOP),
            ]
        );
        let mut res = Vec::new();
        ebcs.for_each_unknown_node(|iu, m, x, y| {
            assert_eq!(ebcs.index_prescribed[m], usize::MAX);
            assert_eq!(ebcs.index_unknown[m], iu);
            res.push((iu, m, x, y));
        });
        assert_eq!(
            &res,
            &[
                (0, 5, 1.0 / 3.0, 1.0 / 3.0),
                (1, 6, 2.0 / 3.0, 1.0 / 3.0),
                (2, 9, 1.0 / 3.0, 2.0 / 3.0),
                (3, 10, 2.0 / 3.0, 2.0 / 3.0),
            ]
        );

        // --- set homogeneous boundary conditions ---

        // 12* 13* 14* 15*
        //  8*  9  10  11*
        //  4*  5   6   7*
        //  0*  1*  2*  3*
        ebcs.set_homogeneous();
        let mut res = Vec::new();
        ebcs.prescribed_sorted.iter().for_each(|&m| {
            let value = ebcs.get_prescribed_value(m, 0.0, 0.0); // x and y do not matter here
            res.push((m, value));
        });
        assert_eq!(
            res,
            &[
                (0, 0.0),
                (1, 0.0),
                (2, 0.0),
                (3, 0.0),
                (4, 0.0),
                (7, 0.0),
                (8, 0.0),
                (11, 0.0),
                (12, 0.0),
                (13, 0.0),
                (14, 0.0),
                (15, 0.0),
            ]
        );
        assert_eq!(&ebcs.is_prescribed, &correct_prescribed);

        // --- set periodic boundary conditions ---

        // 12  13* 14* 15
        //  8   9  10  11
        //  4   5   6   7
        //  0   1*  2*  3
        ebcs.set_periodic(true, false);
        let mut res = Vec::new();
        ebcs.prescribed_sorted.iter().for_each(|&m| {
            let value = ebcs.get_prescribed_value(m, 0.0, 0.0); // x and y do not matter here
            res.push((m, value));
        });
        assert_eq!(res, &[(1, 0.0), (2, 0.0), (13, 0.0), (14, 0.0),]);
        assert_eq!(
            &ebcs.is_prescribed,
            &[
                false, // 0
                true,  // 1
                true,  // 2
                false, // 3
                false, // 4
                false, // 5
                false, // 6
                false, // 7
                false, // 8
                false, // 9
                false, // 10
                false, // 11
                false, // 12
                true,  // 13
                true,  // 14
                false, // 15
            ]
        );
        // 12  13  14  15
        //  8   9  10  11
        //  4   5   6   7
        //  0   1   2   3
        ebcs.set_periodic(true, true);
        assert_eq!(ebcs.prescribed_sorted.len(), 0);
        assert_eq!(&ebcs.is_prescribed, &vec![false; 16]);
    }

    #[test]
    fn new_initializes_correctly() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3).unwrap();
        let ebcs = EssentialBcs2d::new(grid);

        assert_eq!(ebcs.grid.size(), 9);
        assert_eq!(ebcs.is_prescribed.len(), 9);
        assert!(ebcs.is_prescribed.iter().all(|&x| !x)); // all false initially
        assert!(!ebcs.periodic_along_x);
        assert!(!ebcs.periodic_along_y);
        assert_eq!(ebcs.functions.len(), 4);
        assert!(ebcs.node_to_function.is_empty());
        assert!(ebcs.prescribed_sorted.is_empty());
        assert_eq!(ebcs.unknown_sorted, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);

        // Check that default functions return 0.0
        for i in 0..4 {
            assert_eq!((ebcs.functions[i])(123.456, 789.012), 0.0);
        }
    }

    #[test]
    fn minimal_2x2_grid_works() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 2, 2).unwrap();
        let mut ebcs = EssentialBcs2d::new(grid);

        // Grid layout for 2x2:
        //  2──3  (y=1.0)
        //  │  │
        //  0──1  (y=0.0)

        assert_eq!(ebcs.num_unknown(), 4);
        assert_eq!(ebcs.num_prescribed(), 0);

        // Set homogeneous BCs - all nodes should be prescribed since all are on boundary
        ebcs.set_homogeneous();
        assert_eq!(ebcs.num_prescribed(), 4);
        assert_eq!(ebcs.num_unknown(), 0);
        assert_eq!(ebcs.get_nodes_prescribed(), &vec![0, 1, 2, 3]);
        assert!(ebcs.get_nodes_unknown().is_empty());
    }

    #[test]
    fn coordinate_dependent_functions_work() {
        let grid = Grid2d::new_uniform(-1.0, 1.0, -1.0, 1.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new(grid);

        // Grid coordinates:
        //   6(-1,1)  7(0,1)   8(1,1)
        //   3(-1,0)  4(0,0)   5(1,0)
        //   0(-1,-1) 1(0,-1)  2(1,-1)

        // Set coordinate-dependent functions
        let lef = |x, y| x * x + y; // x² + y
        let rig = |x, y| 2.0 * x + y * y; // 2x + y²
        let bot = |x, y| x + 3.0 * y; // x + 3y
        let top = |x, y| x * y; // xy

        ebcs.set(Side::Xmin, lef); // nodes 0, 3, 6
        ebcs.set(Side::Xmax, rig); // nodes 2, 5, 8
        ebcs.set(Side::Ymin, bot); // nodes 0, 1, 2
        ebcs.set(Side::Ymax, top); // nodes 6, 7, 8

        // Test prescribed values at actual coordinates
        // Node 0: (-1, -1) -> bot function wins: -1 + 3*(-1) = -4
        assert_eq!(ebcs.get_prescribed_value(0, -1.0, -1.0), -4.0);

        // Node 1: (0, -1) -> bot function: 0 + 3*(-1) = -3
        assert_eq!(ebcs.get_prescribed_value(1, 0.0, -1.0), -3.0);

        // Node 3: (-1, 0) -> lef function: (-1)² + 0 = 1
        assert_eq!(ebcs.get_prescribed_value(3, -1.0, 0.0), 1.0);

        // Node 5: (1, 0) -> rig function: 2 * 1 + 0² = 2
        assert_eq!(ebcs.get_prescribed_value(5, 1.0, 0.0), 2.0);

        // Node 7: (0, 1) -> top function: 0 * 1 = 0
        assert_eq!(ebcs.get_prescribed_value(7, 0.0, 1.0), 0.0);

        // Node 8: (1, 1) -> top function wins: 1 * 1 = 1
        assert_eq!(ebcs.get_prescribed_value(8, 1.0, 1.0), 1.0);
    }

    #[test]
    fn periodic_boundary_conditions_work() {
        // Grid layout (4x3):
        //    8───9───10──11
        //    │   │   │   │
        //    4───5───6───7
        //    │   │   │   │
        //    0───1───2───3
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new(grid);

        // First set some essential BCs
        let lef = |_, _| LEF;
        let bot = |_, _| BOT;
        assert_eq!(lef(0.0, 0.0), LEF);
        assert_eq!(bot(0.0, 0.0), BOT);
        ebcs.set(Side::Xmin, lef);
        ebcs.set(Side::Ymin, bot);

        assert_eq!(ebcs.get_nodes_prescribed(), &vec![0, 1, 2, 3, 4, 8]);
        assert!(!ebcs.is_periodic_along_x());
        assert!(!ebcs.is_periodic_along_y());

        // Set periodic along x - should remove left/right BCs but keep the middle nodes on the bottom
        ebcs.set_periodic(true, false);
        assert!(ebcs.is_periodic_along_x());
        assert!(!ebcs.is_periodic_along_y());
        assert_eq!(ebcs.get_nodes_prescribed(), &vec![1, 2]); // only MIDDLE bottom remains

        // Set periodic along both - should remove all essential BCs
        ebcs.set_periodic(true, true);
        assert!(ebcs.is_periodic_along_x());
        assert!(ebcs.is_periodic_along_y());
        assert!(ebcs.get_nodes_prescribed().is_empty());
        assert_eq!(ebcs.get_nodes_unknown(), &vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);

        // Set periodic only along y
        ebcs.set_periodic(false, true);
        assert!(!ebcs.is_periodic_along_x());
        assert!(ebcs.is_periodic_along_y());
        assert!(ebcs.get_nodes_prescribed().is_empty());
    }

    #[test]
    fn boundary_condition_overrides_work() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new(grid);

        // Set initial boundary condition
        let func1 = |_, _| 10.0;
        ebcs.set(Side::Xmin, func1);
        assert_eq!(ebcs.get_prescribed_value(0, 0.0, 0.0), 10.0);

        // Override with new function
        let func2 = |x, y| x + y + 20.0;
        ebcs.set(Side::Xmin, func2);
        assert_eq!(ebcs.get_prescribed_value(0, 1.0, 2.0), 23.0);

        // Nodes should remain the same
        assert_eq!(ebcs.get_nodes_prescribed(), &vec![0, 3, 6]);

        // Set periodic to remove essential BC
        ebcs.set_periodic(true, false);
        assert!(ebcs.get_nodes_prescribed().is_empty());

        // Set essential again - should work after periodic
        ebcs.set(Side::Xmin, func1);
        assert_eq!(ebcs.get_nodes_prescribed(), &vec![0, 3, 6]);
        assert!(!ebcs.is_periodic_along_x());
    }

    #[test]
    fn corner_node_priority_works() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new(grid);

        // Grid layout:
        //   6───7───8  (y=1.0)
        //   │   │   │
        //   3───4───5  (y=0.5)
        //   │   │   │
        //   0───1───2  (y=0.0)

        // Set boundaries in specific order to test priority
        let lef = |_, _| LEF; // 1.0
        let rig = |_, _| RIG; // 2.0
        let bot = |_, _| BOT; // 3.0
        let top = |_, _| TOP; // 4.0

        // Set in order: left, right, bottom, top
        ebcs.set(Side::Xmin, lef); // sets nodes 0, 3, 6
        ebcs.set(Side::Xmax, rig); // sets nodes 2, 5, 8
        ebcs.set(Side::Ymin, bot); // sets nodes 0, 1, 2 (0,2 already set)
        ebcs.set(Side::Ymax, top); // sets nodes 6, 7, 8 (6,8 already set)

        // Check corner node values - last set should win
        assert_eq!(ebcs.get_prescribed_value(0, 0.0, 0.0), BOT); // bottom wins over left
        assert_eq!(ebcs.get_prescribed_value(2, 0.0, 0.0), BOT); // bottom wins over right
        assert_eq!(ebcs.get_prescribed_value(6, 0.0, 0.0), TOP); // top wins over left
        assert_eq!(ebcs.get_prescribed_value(8, 0.0, 0.0), TOP); // top wins over right

        // Check non-corner boundary nodes
        assert_eq!(ebcs.get_prescribed_value(1, 0.0, 0.0), BOT); // bottom only
        assert_eq!(ebcs.get_prescribed_value(3, 0.0, 0.0), LEF); // left only
        assert_eq!(ebcs.get_prescribed_value(5, 0.0, 0.0), RIG); // right only
        assert_eq!(ebcs.get_prescribed_value(7, 0.0, 0.0), TOP); // top only
    }

    #[test]
    fn homogeneous_boundary_conditions_work() {
        // Grid layout (4x3):
        //    8───9───10──11
        //    │   │   │   │
        //    4───5───6───7
        //    │   │   │   │
        //    0───1───2───3
        let grid = Grid2d::new_uniform(-1.0, 1.0, -1.0, 1.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new(grid);

        // Set some non-homogeneous BCs first
        let lef = |_, _| LEF;
        assert_eq!(lef(0.0, 0.0), LEF);
        ebcs.set(Side::Xmin, lef);
        ebcs.set_periodic(false, true);

        assert!(ebcs.is_periodic_along_y());
        assert!(!ebcs.get_nodes_prescribed().is_empty());

        // Set homogeneous - should remove all existing BCs and set all boundaries to zero
        ebcs.set_homogeneous();

        assert!(!ebcs.is_periodic_along_x());
        assert!(!ebcs.is_periodic_along_y());

        // All boundary nodes should be prescribed
        let boundary_nodes = vec![0, 1, 2, 3, 4, 7, 8, 9, 10, 11];
        assert_eq!(ebcs.get_nodes_prescribed(), &boundary_nodes);

        // Interior nodes should be unknown
        let interior_nodes = vec![5, 6];
        assert_eq!(ebcs.get_nodes_unknown(), &interior_nodes);

        // All prescribed values should be zero
        for &node in &boundary_nodes {
            let (x, y) = ebcs.get_grid().coord(node);
            assert_eq!(ebcs.get_prescribed_value(node, x, y), 0.0);
        }
    }

    #[test]
    fn complementary_prescribed_unknown_nodes() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 4, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new(grid);

        // Test that prescribed + unknown = all nodes at each stage
        let all_nodes: Vec<usize> = (0..16).collect();

        // Stage 1: No BCs
        let mut combined = ebcs.get_nodes_prescribed().clone();
        combined.extend(ebcs.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Stage 2: Left BC only
        let lef = |_, _| LEF;
        assert_eq!(lef(0.0, 0.0), LEF);
        ebcs.set(Side::Xmin, lef);

        let mut combined = ebcs.get_nodes_prescribed().clone();
        combined.extend(ebcs.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Stage 3: Add more boundaries
        let rig = |_, _| RIG;
        let bot = |_, _| BOT;
        assert_eq!(rig(0.0, 0.0), RIG);
        assert_eq!(bot(0.0, 0.0), BOT);
        ebcs.set(Side::Xmax, rig);
        ebcs.set(Side::Ymin, bot);

        let mut combined = ebcs.get_nodes_prescribed().clone();
        combined.extend(ebcs.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Stage 4: Periodic BC
        ebcs.set_periodic(true, true);

        let mut combined = ebcs.get_nodes_prescribed().clone();
        combined.extend(ebcs.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);
    }

    #[test]
    fn non_uniform_grid_works() {
        // Test with non-uniform grid spacing
        let xx = &[0.0, 0.1, 0.5, 0.9, 1.0];
        let yy = &[0.0, 0.3, 1.0];
        let grid = Grid2d::new(xx, yy).unwrap();
        let mut ebcs = EssentialBcs2d::new(grid);

        // Grid layout (5x3):
        //   10──11──12──13──14  (y=1.0)
        //   │   │   │   │   │
        //   5───6───7───8───9  (y=0.3)
        //   │   │   │   │   │
        //   0───1───2───3───4  (y=0.0)

        assert_eq!(ebcs.grid.size(), 15);

        // Set coordinate-dependent function
        let lef = |x, y| 10.0 * x + y;
        ebcs.set(Side::Xmin, lef); // nodes 0, 5, 10

        // Test prescribed values at actual coordinates
        assert_eq!(ebcs.get_prescribed_value(0, 0.0, 0.0), 0.0); // 10*0.0 + 0.0
        assert_eq!(ebcs.get_prescribed_value(5, 0.0, 0.3), 0.3); // 10*0.0 + 0.3
        assert_eq!(ebcs.get_prescribed_value(10, 0.0, 1.0), 1.0); // 10*0.0 + 1.0

        assert_eq!(ebcs.get_nodes_prescribed(), &vec![0, 5, 10]);
        assert_eq!(ebcs.get_nodes_unknown(), &vec![1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]);
    }

    #[test]
    fn state_consistency_after_operations() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new(grid);

        // Perform a sequence of operations and verify consistency

        // 1. Set left boundary
        let lef = |_, _| LEF;
        assert_eq!(lef(0.0, 0.0), LEF);
        ebcs.set(Side::Xmin, lef);
        assert_eq!(ebcs.num_prescribed() + ebcs.num_unknown(), 9);

        // 2. Add right boundary
        let rig = |_, _| RIG;
        assert_eq!(rig(0.0, 0.0), RIG);
        ebcs.set(Side::Xmax, rig);
        assert_eq!(ebcs.num_prescribed() + ebcs.num_unknown(), 9);

        // 3. Set periodic along x (should remove left/right)
        ebcs.set_periodic(true, false);
        assert_eq!(ebcs.num_prescribed() + ebcs.num_unknown(), 9);
        assert!(ebcs.get_nodes_prescribed().is_empty());

        // 4. Set homogeneous (should add all boundary nodes)
        ebcs.set_homogeneous();
        assert_eq!(ebcs.num_prescribed() + ebcs.num_unknown(), 9);
        assert_eq!(ebcs.num_prescribed(), 8); // all boundary nodes
        assert_eq!(ebcs.num_unknown(), 1); // only center node

        // 5. Set periodic both ways (should remove all)
        ebcs.set_periodic(true, true);
        assert_eq!(ebcs.num_prescribed() + ebcs.num_unknown(), 9);
        assert_eq!(ebcs.num_prescribed(), 0);
        assert_eq!(ebcs.num_unknown(), 9);

        // Verify final state
        assert!(ebcs.is_periodic_along_x());
        assert!(ebcs.is_periodic_along_y());
        assert!(ebcs.node_to_function.is_empty());
        assert!(ebcs.prescribed_sorted.is_empty());
        assert_eq!(ebcs.unknown_sorted, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(ebcs.is_prescribed.iter().all(|&x| !x));
    }
}
