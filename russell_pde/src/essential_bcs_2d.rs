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
    fn set_periodic_works() {
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
    fn set_works() {
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
}
