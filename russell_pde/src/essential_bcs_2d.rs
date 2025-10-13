use crate::{Grid2d, Side};
use russell_sparse::CooMatrix;
use russell_sparse::Sym;
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
    grid: &'a Grid2d,

    /// Flags equations with prescribed EBC values
    is_prescribed: Vec<bool>,

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
    /// Maps node ID to one of the four functions in `functions`
    essential: HashMap<usize, usize>,

    /// Holds the sorted indices of the equations with essential boundary conditions
    essential_sorted: Vec<usize>,

    /// Holds the sorted indices of the equations without essential boundary conditions
    unknown_sorted: Vec<usize>,
}

impl<'a> EssentialBcs2d<'a> {
    /// Allocates a new instance
    pub fn new(grid: &'a Grid2d) -> Self {
        let dim = grid.size();
        EssentialBcs2d {
            grid,
            is_prescribed: vec![false; dim],
            periodic_along_x: false,
            periodic_along_y: false,
            functions: vec![
                Arc::new(|_, _| 0.0), // xmin
                Arc::new(|_, _| 0.0), // xmax
                Arc::new(|_, _| 0.0), // ymin
                Arc::new(|_, _| 0.0), // ymax
            ],
            essential: HashMap::new(),
            essential_sorted: Vec::new(),
            unknown_sorted: (0..dim).collect(), // Initialize with all node indices
        }
    }

    // --------------------------------------------------------
    // setters
    // --------------------------------------------------------

    /// Recomputes the prescribed flags array and the essential_sorted array
    fn recompute_arrays(&mut self) {
        self.unknown_sorted.clear();
        let dim = self.grid.size();
        for m in 0..dim {
            if self.essential.contains_key(&m) {
                self.is_prescribed[m] = true;
            } else {
                self.is_prescribed[m] = false;
                self.unknown_sorted.push(m); // already in sorted order due to loop order
            }
        }
        self.essential_sorted = self.essential.keys().copied().collect();
        self.essential_sorted.sort(); // need this since HashMap keys are unordered
    }

    /// Sets periodic boundary condition
    ///
    /// **Note:** Any essential boundary condition on the corresponding side will be removed.
    pub fn set_periodic(&mut self, along_x: bool, along_y: bool) {
        self.periodic_along_x = along_x;
        self.periodic_along_y = along_y;
        if along_x {
            self.grid.for_each_node_xmin(|n| {
                self.essential.remove(n);
            });
            self.grid.for_each_node_xmax(|n| {
                self.essential.remove(n);
            });
        }
        if along_y {
            self.grid.for_each_node_ymin(|n| {
                self.essential.remove(n);
            });
            self.grid.for_each_node_ymax(|n| {
                self.essential.remove(n);
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
                    self.essential.insert(*n, 0);
                });
            }
            Side::Xmax => {
                self.periodic_along_x = false;
                self.functions[1] = Arc::new(f);
                self.grid.for_each_node_xmax(|n| {
                    self.essential.insert(*n, 1);
                });
            }
            Side::Ymin => {
                self.periodic_along_y = false;
                self.functions[2] = Arc::new(f);
                self.grid.for_each_node_ymin(|n| {
                    self.essential.insert(*n, 2);
                });
            }
            Side::Ymax => {
                self.periodic_along_y = false;
                self.functions[3] = Arc::new(f);
                self.grid.for_each_node_ymax(|n| {
                    self.essential.insert(*n, 3);
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
        self.essential.clear();
        self.functions = vec![
            Arc::new(|_, _| 0.0), // xmin
            Arc::new(|_, _| 0.0), // xmax
            Arc::new(|_, _| 0.0), // ymin
            Arc::new(|_, _| 0.0), // ymax
        ];
        self.grid.for_each_node_xmin(|n| {
            self.essential.insert(*n, 0);
        });
        self.grid.for_each_node_xmax(|n| {
            self.essential.insert(*n, 1);
        });
        self.grid.for_each_node_ymin(|n| {
            self.essential.insert(*n, 2);
        });
        self.grid.for_each_node_ymax(|n| {
            self.essential.insert(*n, 3);
        });
        self.recompute_arrays();
    }

    // --------------------------------------------------------
    // getters
    // --------------------------------------------------------

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

    /// Returns the number of prescribed equations
    ///
    /// The number of prescribed equations is equal to the number of nodes with essential conditions.
    pub fn num_prescribed(&self) -> usize {
        self.essential.len()
    }

    /// Returns the number of unknown equations
    pub fn num_unknown(&self) -> usize {
        self.unknown_sorted.len()
    }

    /// Returns the (sorted) indices of the nodes with prescribed values
    pub fn get_nodes_prescribed(&self) -> &Vec<usize> {
        &self.essential_sorted
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
        let index = self.essential.get(&m).unwrap();
        (self.functions[*index])(x, y)
    }

    /// Generates the Lagrange matrix
    ///
    /// Returns the Lagrange matrix `E` for handling essential boundary conditions
    /// with the Lagrange multipliers method (LMM).
    ///
    /// The LMM considers the augmented system of equations:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K  Eᵀ │ │ u │   │ f │
    /// │       │ │   │ = │   │
    /// │ E  0  │ │ w │   │ ū │
    /// └       ┘ └   ┘   └   ┘
    ///     A       x       b
    /// ```
    ///
    /// where `E` is the Lagrange matrix, `u` is the vector of unknowns, `f` is the vector of "forces",
    /// `w` is the vector of Lagrange multipliers, and `ū` is the vector of prescribed essential values.
    pub fn get_lagrange_matrix(&self) -> CooMatrix {
        let np = self.essential.len();
        let dim = self.grid.size();
        let nnz = np;
        let mut ee = CooMatrix::new(np, dim, nnz, Sym::No).unwrap();
        self.essential_sorted.iter().enumerate().for_each(|(ip, &m)| {
            ee.put(ip, m, 1.0).unwrap();
        });
        ee
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
        let mut ebcs = EssentialBcs2d::new(&grid);
        assert_eq!(&ebcs.is_prescribed, &vec![false; 12]);
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.periodic_along_y, false);
        assert_eq!(ebcs.functions.len(), 4);
        assert_eq!(ebcs.essential.len(), 0);
        assert_eq!(ebcs.essential_sorted.len(), 0);
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
    fn new_and_set_work() {
        // 12 13 14 15
        //  8  9 10 11
        //  4  5  6  7
        //  0  1  2  3
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 4, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new(&grid);
        assert_eq!(&ebcs.is_prescribed, &vec![false; 16]);
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.periodic_along_y, false);
        assert_eq!(ebcs.functions.len(), 4);
        assert_eq!(ebcs.essential.len(), 0);
        assert_eq!(ebcs.essential_sorted.len(), 0);
        assert_eq!(
            &ebcs.unknown_sorted,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );

        // 12* 13* 14* 15*
        //  8*  9  10  11*
        //  4*  5   6   7*
        //  0*  1*  2*  3*
        let lef = |_, _| LEF;
        let rig = |_, _| RIG;
        let bot = |_, _| BOT;
        let top = |_, _| TOP;
        ebcs.set(Side::Xmin, lef);
        assert_eq!(ebcs.essential_sorted, vec![0, 4, 8, 12]);
        ebcs.set(Side::Xmax, rig);
        assert_eq!(ebcs.essential_sorted, vec![0, 3, 4, 7, 8, 11, 12, 15]);
        ebcs.set(Side::Ymin, bot);
        assert_eq!(ebcs.essential_sorted, vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 15]);
        ebcs.set(Side::Ymax, top);
        assert_eq!(ebcs.essential_sorted, vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]);

        assert_eq!(ebcs.is_periodic_along_x(), false);
        assert_eq!(ebcs.is_periodic_along_y(), false);
        assert_eq!(ebcs.is_prescribed(0), true);
        assert_eq!(ebcs.is_prescribed(15), true);
        assert_eq!(ebcs.is_prescribed(5), false);
        assert_eq!(ebcs.num_prescribed(), 12);
        assert_eq!(ebcs.num_unknown(), 4);
        assert_eq!(
            ebcs.get_nodes_prescribed(),
            &vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]
        );
        assert_eq!(ebcs.get_nodes_unknown(), &vec![5, 6, 9, 10]);

        let mut res = Vec::new();
        ebcs.essential_sorted.iter().for_each(|&m| {
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

        // 12* 13* 14* 15*
        //  8*  9  10  11*
        //  4*  5   6   7*
        //  0*  1*  2*  3*
        ebcs.set_homogeneous();
        let mut res = Vec::new();
        ebcs.essential_sorted.iter().for_each(|&m| {
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

        // 12  13* 14* 15
        //  8   9  10  11
        //  4   5   6   7
        //  0   1*  2*  3
        ebcs.set_periodic(true, false);
        let mut res = Vec::new();
        ebcs.essential_sorted.iter().for_each(|&m| {
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
        assert_eq!(ebcs.essential_sorted.len(), 0);
        assert_eq!(&ebcs.is_prescribed, &vec![false; 16]);
    }

    #[test]
    fn get_lagrange_matrix_works() {
        // 12* 13  14  15
        //  8*  9  10  11
        //  4*  5   6   7
        //  0*  1   2   3
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 4, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new(&grid);
        const LEF: f64 = 1.0;
        let lef = |_, _| LEF;
        assert_eq!(lef(0.0, 0.0), LEF);
        ebcs.set(Side::Xmin, lef); //  0  4  8  12
        let ee = ebcs.get_lagrange_matrix();
        assert_eq!(
            format!("{}", ee.as_dense()),
            "┌                                 ┐\n\
             │ 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 │\n\
             └                                 ┘"
        );
    }

    #[test]
    fn new_initializes_correctly() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3).unwrap();
        let ebcs = EssentialBcs2d::new(&grid);

        assert_eq!(ebcs.grid.size(), 9);
        assert_eq!(ebcs.is_prescribed.len(), 9);
        assert!(ebcs.is_prescribed.iter().all(|&x| !x)); // all false initially
        assert!(!ebcs.periodic_along_x);
        assert!(!ebcs.periodic_along_y);
        assert_eq!(ebcs.functions.len(), 4);
        assert!(ebcs.essential.is_empty());
        assert!(ebcs.essential_sorted.is_empty());
        assert_eq!(ebcs.unknown_sorted, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);

        // Check that default functions return 0.0
        for i in 0..4 {
            assert_eq!((ebcs.functions[i])(123.456, 789.012), 0.0);
        }
    }

    #[test]
    fn minimal_2x2_grid_works() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 2, 2).unwrap();
        let mut ebcs = EssentialBcs2d::new(&grid);

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
    fn single_boundary_conditions_work() {
        let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 2.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new(&grid);

        // Grid layout (4x3):
        //    8───9───10──11 (y=2.0)
        //    │   │   │   │
        //    4───5───6───7  (y=1.0)
        //    │   │   │   │
        //    0───1───2───3  (y=0.0)

        // Test each boundary individually

        // Left boundary only
        let lef = |_, _| LEF;
        assert_eq!(lef(0.0, 0.0), LEF);
        ebcs.set(Side::Xmin, lef);
        assert_eq!(ebcs.get_nodes_prescribed(), &vec![0, 4, 8]);
        assert_eq!(ebcs.get_nodes_unknown(), &vec![1, 2, 3, 5, 6, 7, 9, 10, 11]);
        assert!(!ebcs.is_periodic_along_x());
        assert!(!ebcs.is_periodic_along_y());

        // Clear and test right boundary only
        let mut ebcs = EssentialBcs2d::new(&grid);
        let rig = |_, _| RIG;
        assert_eq!(rig(0.0, 0.0), RIG);
        ebcs.set(Side::Xmax, rig);
        assert_eq!(ebcs.get_nodes_prescribed(), &vec![3, 7, 11]);
        assert_eq!(ebcs.get_nodes_unknown(), &vec![0, 1, 2, 4, 5, 6, 8, 9, 10]);

        // Clear and test bottom boundary only
        let mut ebcs = EssentialBcs2d::new(&grid);
        let bot = |_, _| BOT;
        assert_eq!(bot(0.0, 0.0), BOT);
        ebcs.set(Side::Ymin, bot);
        assert_eq!(ebcs.get_nodes_prescribed(), &vec![0, 1, 2, 3]);
        assert_eq!(ebcs.get_nodes_unknown(), &vec![4, 5, 6, 7, 8, 9, 10, 11]);

        // Clear and test top boundary only
        let mut ebcs = EssentialBcs2d::new(&grid);
        let top = |_, _| TOP;
        assert_eq!(top(0.0, 0.0), TOP);
        ebcs.set(Side::Ymax, top);
        assert_eq!(ebcs.get_nodes_prescribed(), &vec![8, 9, 10, 11]);
        assert_eq!(ebcs.get_nodes_unknown(), &vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn coordinate_dependent_functions_work() {
        let grid = Grid2d::new_uniform(-1.0, 1.0, -1.0, 1.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new(&grid);

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
        let (x0, y0) = grid.coord(0);
        assert_eq!(ebcs.get_prescribed_value(0, x0, y0), -4.0);

        // Node 1: (0, -1) -> bot function: 0 + 3*(-1) = -3
        let (x1, y1) = grid.coord(1);
        assert_eq!(ebcs.get_prescribed_value(1, x1, y1), -3.0);

        // Node 3: (-1, 0) -> lef function: (-1)² + 0 = 1
        let (x3, y3) = grid.coord(3);
        assert_eq!(ebcs.get_prescribed_value(3, x3, y3), 1.0);

        // Node 5: (1, 0) -> rig function: 2 * 1 + 0² = 2
        let (x5, y5) = grid.coord(5);
        assert_eq!(ebcs.get_prescribed_value(5, x5, y5), 2.0);

        // Node 7: (0, 1) -> top function: 0 * 1 = 0
        let (x7, y7) = grid.coord(7);
        assert_eq!(ebcs.get_prescribed_value(7, x7, y7), 0.0);

        // Node 8: (1, 1) -> top function wins: 1 * 1 = 1
        let (x8, y8) = grid.coord(8);
        assert_eq!(ebcs.get_prescribed_value(8, x8, y8), 1.0);
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
        let mut ebcs = EssentialBcs2d::new(&grid);

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
        let mut ebcs = EssentialBcs2d::new(&grid);

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
        let mut ebcs = EssentialBcs2d::new(&grid);

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
        let mut ebcs = EssentialBcs2d::new(&grid);

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
            let (x, y) = grid.coord(node);
            assert_eq!(ebcs.get_prescribed_value(node, x, y), 0.0);
        }
    }

    #[test]
    fn complementary_prescribed_unknown_nodes() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 4, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new(&grid);

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
        let mut ebcs = EssentialBcs2d::new(&grid);

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
        let mut ebcs = EssentialBcs2d::new(&grid);

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
        assert!(ebcs.essential.is_empty());
        assert!(ebcs.essential_sorted.is_empty());
        assert_eq!(ebcs.unknown_sorted, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(ebcs.is_prescribed.iter().all(|&x| !x));
    }
}
