use super::Side;
use crate::StrError;
use russell_sparse::{CooMatrix, Sym};
use std::collections::HashMap;
use std::sync::Arc;

/// Implements the Finite Difference (FDM) Laplacian operator in 2D
///
/// Given the (continuum) scalar field ϕ(x, y) and its Laplacian
///
/// ```text
///           ∂²ϕ        ∂²ϕ
/// L{ϕ} = kx ———  +  ky ———
///           ∂x²        ∂y²
/// ```
///
/// we substitute the partial derivatives using central FDM over a rectangular grid.
/// The resulting discrete Laplacian is expressed by the coefficient matrix `A` and the vector `X`:
///
/// ```text
/// D{ϕᵢⱼ} = A ⋅ X
/// ```
///
/// ϕᵢⱼ are the discrete counterpart of ϕ(x, y) over the (nx, ny) grid. However, these
/// values are "sequentially" mapped onto to the vector `X` using the following formula:
///
/// ```text
/// ϕᵢⱼ → Xₘ   with   m = i + j nx
/// ```
///
/// The dimension of the coefficient matrix is `dim = nrow = ncol = nx × ny`.
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
///
/// # Remarks
///
/// * The operator is built with a three-point stencil.
/// * The boundary conditions may be Neumann with zero-flux or periodic.
/// * By default (Neumann BC), the boundary nodes are 'mirrored' yielding a no-flux barrier.
pub struct FdmLaplacian2d<'a> {
    xmin: f64,              // min x coordinate
    ymin: f64,              // min y coordinate
    nx: usize,              // number of points along x (≥ 2)
    ny: usize,              // number of points along y (≥ 2)
    dx: f64,                // grid spacing along x
    dy: f64,                // grid spacing along y
    nodes_xmin: Vec<usize>, // indices of nodes on the xmin edge
    nodes_xmax: Vec<usize>, // indices of nodes on the xmax edge
    nodes_ymin: Vec<usize>, // indices of nodes on the ymin edge
    nodes_ymax: Vec<usize>, // indices of nodes on the ymax edge
    prescribed: Vec<bool>,  // flags equations with prescribed EBC values

    /// Indicates that the boundary is periodic along x (left ϕ values equal right ϕ values)
    ///
    /// If false, the left/right boundaries are zero-flux (Neumann with ∂ϕ/dx = 0)
    periodic_along_x: bool,

    /// Indicates that the boundary is periodic along x (bottom ϕ values equal top ϕ values)
    ///
    /// If false, the bottom/top boundaries are zero-flux (Neumann with ∂ϕ/dx = 0)
    periodic_along_y: bool,

    /// Holds the FDM coefficients (α, β, β, γ, γ)
    ///
    /// These coefficients are applied over the "bandwidth" of the coefficient matrix
    molecule: Vec<f64>,

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

impl<'a> FdmLaplacian2d<'a> {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `kx` -- diffusion parameter x
    /// * `ky` -- diffusion parameter y
    /// * `xmin` -- min x coordinate
    /// * `xmax` -- max x coordinate
    /// * `ymin` -- min y coordinate
    /// * `ymax` -- max y coordinate
    /// * `nx` -- number of points along x (≥ 2)
    /// * `ny` -- number of points along y (≥ 2)
    pub fn new(
        kx: f64,
        ky: f64,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        nx: usize,
        ny: usize,
    ) -> Result<Self, StrError> {
        // check
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }
        if ny < 2 {
            return Err("ny must be ≥ 2");
        }

        // auxiliary data
        let dim = nx * ny;
        let dx = (xmax - xmin) / ((nx - 1) as f64);
        let dy = (ymax - ymin) / ((ny - 1) as f64);
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let alpha = -2.0 * (kx / dx2 + ky / dy2);
        let beta = kx / dx2;
        let gamma = ky / dy2;

        // allocate instance
        Ok(FdmLaplacian2d {
            xmin,
            ymin,
            nx,
            ny,
            dx,
            dy,
            nodes_xmin: (0..dim).step_by(nx).collect(),
            nodes_xmax: ((nx - 1)..dim).step_by(nx).collect(),
            nodes_ymin: (0..nx).collect(),
            nodes_ymax: ((dim - nx)..dim).collect(),
            prescribed: vec![false; dim],
            periodic_along_x: false,
            periodic_along_y: false,
            molecule: vec![alpha, beta, beta, gamma, gamma],
            functions: vec![
                Arc::new(|_, _| 0.0), // xmin
                Arc::new(|_, _| 0.0), // xmax
                Arc::new(|_, _| 0.0), // ymin
                Arc::new(|_, _| 0.0), // ymax
            ],
            essential: HashMap::new(),
            essential_sorted: Vec::new(),
            unknown_sorted: (0..dim).collect(), // Initialize with all node indices
        })
    }

    /// Recomputes the prescribed flags array and the essential_sorted array
    fn recompute_arrays(&mut self) {
        self.unknown_sorted.clear();
        let dim = self.nx * self.ny;
        for m in 0..dim {
            if self.essential.contains_key(&m) {
                self.prescribed[m] = true;
            } else {
                self.prescribed[m] = false;
                self.unknown_sorted.push(m); // already in sorted order due to loop order
            }
        }
        self.essential_sorted = self.essential.keys().copied().collect();
        self.essential_sorted.sort(); // need this since HashMap keys are unordered
    }

    /// Sets periodic boundary condition
    ///
    /// **Note:** Any essential boundary condition on the corresponding side will be removed.
    pub fn set_periodic_boundary_condition(&mut self, along_x: bool, along_y: bool) {
        if along_x {
            self.periodic_along_x = true;
            self.nodes_xmin.iter().for_each(|n| {
                self.essential.remove(n);
            });
            self.nodes_xmax.iter().for_each(|n| {
                self.essential.remove(n);
            });
        }
        if along_y {
            self.periodic_along_y = true;
            self.nodes_ymin.iter().for_each(|n| {
                self.essential.remove(n);
            });
            self.nodes_ymax.iter().for_each(|n| {
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
    pub fn set_essential_boundary_condition(&mut self, side: Side, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'a) {
        match side {
            Side::Xmin => {
                self.periodic_along_x = false;
                self.functions[0] = Arc::new(f);
                self.nodes_xmin.iter().for_each(|n| {
                    self.essential.insert(*n, 0);
                });
            }
            Side::Xmax => {
                self.periodic_along_x = false;
                self.functions[1] = Arc::new(f);
                self.nodes_xmax.iter().for_each(|n| {
                    self.essential.insert(*n, 1);
                });
            }
            Side::Ymin => {
                self.periodic_along_y = false;
                self.functions[2] = Arc::new(f);
                self.nodes_ymin.iter().for_each(|n| {
                    self.essential.insert(*n, 2);
                });
            }
            Side::Ymax => {
                self.periodic_along_y = false;
                self.functions[3] = Arc::new(f);
                self.nodes_ymax.iter().for_each(|n| {
                    self.essential.insert(*n, 3);
                });
            }
        };
        self.recompute_arrays();
    }

    /// Sets homogeneous boundary conditions (i.e., zero essential values at the borders)
    ///
    /// **Note:** Periodic boundary conditions will be removed.
    pub fn set_homogeneous_boundary_conditions(&mut self) {
        self.periodic_along_x = false;
        self.periodic_along_y = false;
        self.essential.clear();
        self.functions = vec![
            Arc::new(|_, _| 0.0), // xmin
            Arc::new(|_, _| 0.0), // xmax
            Arc::new(|_, _| 0.0), // ymin
            Arc::new(|_, _| 0.0), // ymax
        ];
        self.nodes_xmin.iter().for_each(|n| {
            self.essential.insert(*n, 0);
        });
        self.nodes_xmax.iter().for_each(|n| {
            self.essential.insert(*n, 1);
        });
        self.nodes_ymin.iter().for_each(|n| {
            self.essential.insert(*n, 2);
        });
        self.nodes_ymax.iter().for_each(|n| {
            self.essential.insert(*n, 3);
        });
        self.recompute_arrays();
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
    pub fn lagrange_matrix(&self) -> Result<CooMatrix, StrError> {
        let np = self.essential.len(); // number of prescribed equations
        let dim = self.nx * self.ny;
        let nnz = np;
        let mut ee = CooMatrix::new(np, dim, nnz, Sym::No).unwrap();
        self.loop_over_prescribed_values(|ip, m, _val| {
            ee.put(ip, m, 1.0).unwrap();
        });
        Ok(ee)
    }

    /// Computes the (full) coefficient matrix
    pub fn coefficient_matrix(&self) -> Result<CooMatrix, StrError> {
        // count max number of non-zeros
        let dim = self.nx * self.ny;
        let max_nnz_aa = 5 * dim;

        // allocate the matrix
        let mut aa = CooMatrix::new(dim, dim, max_nnz_aa, Sym::No)?;

        // assemble
        for m in 0..dim {
            self.loop_over_bandwidth(m, |n, b| {
                aa.put(m, n, self.molecule[b]).unwrap();
            });
        }
        Ok(aa)
    }

    /// Computes the augmented coefficient matrix for the Lagrange multipliers method
    ///
    /// Returns the augmented matrix `A` for handling essential boundary conditions
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
    pub fn augmented_coefficient_matrix(&self, extra_nnz: usize) -> Result<CooMatrix, StrError> {
        let np = self.essential.len(); // number of prescribed equations
        let dim = self.nx * self.ny;
        let max_nnz_aa = 5 * dim + 2 * np + extra_nnz; // 5 per row + 2 per prescribed equation
        let mut aa = CooMatrix::new(dim + np, dim + np, max_nnz_aa, Sym::No)?;

        // assemble A
        for m in 0..dim {
            self.loop_over_bandwidth(m, |n, b| {
                aa.put(m, n, self.molecule[b]).unwrap();
            });
        }

        // assemble E and Eᵀ
        self.loop_over_prescribed_values(|ip, j, _val| {
            aa.put(dim + ip, j, 1.0).unwrap(); // E
            aa.put(j, dim + ip, 1.0).unwrap(); // Eᵀ
        });
        Ok(aa)
    }

    /// Computes the modified coefficient matrix
    ///
    /// Consider the following partitioning:
    ///
    /// ```text
    /// ┌          ┐ ┌    ┐   ┌    ┐
    /// │ K11  K12 │ │ u1 │   │ f1 │
    /// │          │ │    │ = │    │
    /// │ K21  K22 │ │ u2 │   │ f2 │
    /// └          ┘ └    ┘   └    ┘
    /// ```
    ///
    /// where `1` means *unknown* and `2` means *prescribed*. Thus, `u1` is the sub-vector with
    /// unknown essential values and `u2` is the sub-vector with prescribed essential values.
    ///
    /// Thus:
    ///
    /// ```text
    /// K11 ⋅ u1  +  K12 ⋅ u2  =  f1
    ///
    /// and
    ///
    /// u1 = K11⁻¹ ⋅ (f1 - K12 ⋅ u2)
    /// ```
    ///
    /// Without changing the dimension of the original problem, the **modified**
    /// linear system is:
    ///
    /// ```text
    /// ┌          ┐ ┌    ┐   ┌             ┐
    /// │ K11   0  │ │ u1 │   │ f1 - K12⋅u2 │
    /// │          │ │    │ = │             │
    /// │  0    1  │ │ u2 │   │     u2      │
    /// └          ┘ └    ┘   └             ┘
    ///       A         u            f
    /// ```
    ///
    /// This function also returns the correction matrix, which allows the computation
    /// of the right-hand side vector. For instance:
    ///
    /// ```text
    /// ┌          ┐ ┌    ┐   ┌        ┐
    /// │  0   K12 │ │ .. │   │ K12⋅u2 │
    /// │          │ │    │ = │        │
    /// │  0    0  │ │ u2 │   │   0    │
    /// └          ┘ └    ┘   └        ┘
    ///      C         u           f
    /// ```
    ///
    /// Note that there is no performance loss in using the modified matrix because the sparse
    /// matrix-vector multiplication will execute the same number of computations with a reduced matrix.
    /// Also, the CooMatrix will only hold the non-zero entries, thus, no extra memory is needed.
    ///
    /// # Output
    ///
    /// Returns `(A, C)` where:
    ///
    /// * `A` -- is the modified matrix (dim × dim) with ones placed on the diagonal entries
    ///  corresponding to the prescribed essential values. Also, the entries corresponding to the
    ///  essential values are zeroed.
    /// * `C` -- is the correction matrix (dim × dim) with only the 'unknown rows'
    ///   and the 'prescribed' columns.
    ///
    /// # Warnings
    ///
    /// **Important:** This function must be called after setting the essential boundary conditions.
    pub fn mod_coefficient_matrix(&self) -> Result<(CooMatrix, CooMatrix), StrError> {
        // count max number of non-zeros
        let dim = self.nx * self.ny;
        let np = self.essential.len();
        let mut max_nnz_aa = np; // start with the diagonal 'ones'
        let mut max_nnz_cc = 1; // +1 just for when there are no essential conditions
        for m in 0..dim {
            if !self.prescribed[m] {
                self.loop_over_bandwidth(m, |n, _| {
                    if !self.prescribed[n] {
                        max_nnz_aa += 1;
                    } else {
                        max_nnz_cc += 1;
                    }
                });
            }
        }

        // allocate matrices
        let mut aa = CooMatrix::new(dim, dim, max_nnz_aa, Sym::No)?;
        let mut cc = CooMatrix::new(dim, dim, max_nnz_cc, Sym::No)?;

        // assemble
        for m in 0..dim {
            if !self.prescribed[m] {
                self.loop_over_bandwidth(m, |n, b| {
                    if !self.prescribed[n] {
                        aa.put(m, n, self.molecule[b]).unwrap();
                    } else {
                        cc.put(m, n, self.molecule[b]).unwrap();
                    }
                });
            } else {
                aa.put(m, m, 1.0).unwrap();
            }
        }
        Ok((aa, cc))
    }

    /// Executes a loop over one row of the coefficient matrix 'A' of A ⋅ X = B
    ///
    /// Note that some column indices may appear repeated; e.g. due to the zero-flux boundaries.
    ///
    /// # Input
    ///
    /// * `m` -- the row of the coefficient matrix
    /// * `callback` -- a `function(n, Amn)` where `n` is the column index and
    ///   `Amn` is the m-n-element of the coefficient matrix
    pub fn loop_over_coef_mat_row<F>(&self, m: usize, mut callback: F)
    where
        F: FnMut(usize, f64),
    {
        self.loop_over_bandwidth(m, |n, b| {
            callback(n, self.molecule[b]);
        });
    }

    /// Executes a loop over the prescribed values
    ///
    /// # Input
    ///
    /// * `callback` -- a `function(ip, m, value)` where `ip` is the index of the
    ///   prescribed value/Lagrange multiplier, `m` is the row index in the coefficient
    ///   matrix, and `value` is the prescribed value.
    pub fn loop_over_prescribed_values<F>(&self, mut callback: F)
    where
        F: FnMut(usize, usize, f64),
    {
        let mut ip = 0;
        self.essential_sorted.iter().for_each(|m| {
            let index = self.essential.get(m).unwrap();
            let i = m % self.nx;
            let j = m / self.nx;
            let x = self.xmin + (i as f64) * self.dx;
            let y = self.ymin + (j as f64) * self.dy;
            let value = (self.functions[*index])(x, y);
            callback(ip, *m, value);
            ip += 1;
        });
    }

    /// Executes a loop over the "bandwidth" of the coefficient matrix
    ///
    /// Here, the "bandwidth" means the non-zero values on a row of the coefficient matrix.
    /// This is not the actual bandwidth because the zero elements are ignored. There are
    /// five non-zero values in the "bandwidth" and they correspond to the "molecule" array.
    ///
    /// # Input
    ///
    /// * `m` -- the row index
    /// * `callback` -- a function of `(n, b)` where `n` is the column index and
    ///   `b` is the bandwidth index, i.e., the index in the molecule array.
    fn loop_over_bandwidth<F>(&self, m: usize, mut callback: F)
    where
        F: FnMut(usize, usize),
    {
        // constants for clarity/convenience
        const CUR: usize = 0; // current node
        const LEF: usize = 1; // left node
        const RIG: usize = 2; // right node
        const BOT: usize = 3; // bottom node
        const TOP: usize = 4; // top node
        const INI_X: usize = 0;
        const INI_Y: usize = 0;
        let fin_x = self.nx - 1;
        let fin_y = self.ny - 1;
        let i = m % self.nx;
        let j = m / self.nx;

        // n indices of the non-zero values on the row m of the coefficient matrix
        // (mirror or swap the indices of boundary nodes, as appropriate)
        let mut nn = [0, 0, 0, 0, 0];
        nn[CUR] = m;
        if self.periodic_along_x {
            nn[LEF] = if i != INI_X { m - 1 } else { m + fin_x };
            nn[RIG] = if i != fin_x { m + 1 } else { m - fin_x };
        } else {
            nn[LEF] = if i != INI_X { m - 1 } else { m + 1 };
            nn[RIG] = if i != fin_x { m + 1 } else { m - 1 };
        }
        if self.periodic_along_y {
            nn[BOT] = if j != INI_Y { m - self.nx } else { m + fin_y * self.nx };
            nn[TOP] = if j != fin_y { m + self.nx } else { m - fin_y * self.nx };
        } else {
            nn[BOT] = if j != INI_Y { m - self.nx } else { m + self.nx };
            nn[TOP] = if j != fin_y { m + self.nx } else { m - self.nx };
        }

        // execute callback
        for (b, &n) in nn.iter().enumerate() {
            callback(n, b);
        }
    }

    /// Executes a loop over the grid points
    ///
    /// # Input
    ///
    /// * `callback` -- a function of `(m, x, y)` where `m` is the sequential point number,
    ///   and `(x, y)` are the Cartesian coordinates of the grid point.
    ///
    /// Note that:
    ///
    /// ```text
    /// m = i + j nx
    /// i = m % nx
    /// j = m / nx
    /// ```
    pub fn loop_over_grid_points<F>(&self, mut callback: F)
    where
        F: FnMut(usize, f64, f64),
    {
        let dim = self.nx * self.ny;
        for m in 0..dim {
            let i = m % self.nx;
            let j = m / self.nx;
            let x = self.xmin + (i as f64) * self.dx;
            let y = self.ymin + (j as f64) * self.dy;
            callback(m, x, y)
        }
    }

    /// Returns the dimension of the linear system
    ///
    /// ```text
    /// dim = nx × ny
    /// ```
    pub fn dim(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns the number of prescribed equations
    ///
    /// The number of prescribed equations is equal to the number of nodes with essential conditions.
    pub fn num_prescribed(&self) -> usize {
        self.essential.len()
    }

    /// Returns an access to the array of prescribed flags
    pub fn prescribed_flags(&self) -> &Vec<bool> {
        &self.prescribed
    }

    /// Returns the prescribed value for the given node
    pub fn get_prescribed_value(&self, m: usize) -> Result<f64, StrError> {
        if let Some(index) = self.essential.get(&m) {
            let i = m % self.nx;
            let j = m / self.nx;
            let x = self.xmin + (i as f64) * self.dx;
            let y = self.ymin + (j as f64) * self.dy;
            Ok((self.functions[*index])(x, y))
        } else {
            Err("no prescribed value for the given index")
        }
    }

    /// Returns the (sorted) indices of the nodes with unknown values
    pub fn get_nodes_unknown(&self) -> &Vec<usize> {
        &self.unknown_sorted
    }

    /// Returns the (sorted) indices of the nodes with prescribed values
    pub fn get_nodes_prescribed(&self) -> &Vec<usize> {
        &self.essential_sorted
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{FdmLaplacian2d, Side};
    use russell_lab::{mat_approx_eq, Matrix};

    #[test]
    fn new_works() {
        let lap = FdmLaplacian2d::new(7.0, 8.0, -1.0, 1.0, -3.0, 3.0, 2, 3).unwrap();
        assert_eq!(lap.xmin, -1.0);
        assert_eq!(lap.ymin, -3.0);
        assert_eq!(lap.nx, 2);
        assert_eq!(lap.ny, 3);
        assert_eq!(lap.dx, 2.0);
        assert_eq!(lap.dy, 3.0);
        assert_eq!(lap.nodes_xmin, &[0, 2, 4]);
        assert_eq!(lap.nodes_xmax, &[1, 3, 5]);
        assert_eq!(lap.nodes_ymin, &[0, 1]);
        assert_eq!(lap.nodes_ymax, &[4, 5]);
    }

    #[test]
    fn new_fails_on_invalid_parameters() {
        assert_eq!(
            FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 2).err(),
            Some("nx must be ≥ 2")
        );
        assert_eq!(
            FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 1).err(),
            Some("ny must be ≥ 2")
        );
        assert_eq!(
            FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1).err(),
            Some("nx must be ≥ 2")
        );
    }

    #[test]
    fn set_essential_boundary_condition_works() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        const LEF: f64 = 1.0;
        const RIG: f64 = 2.0;
        const BOT: f64 = 3.0;
        const TOP: f64 = 4.0;
        let lef = |_, _| LEF;
        let rig = |_, _| RIG;
        let bot = |_, _| BOT;
        let top = |_, _| TOP;
        lap.set_essential_boundary_condition(Side::Xmin, lef); //  0*   4   8  12*
        assert_eq!(lap.essential_sorted, vec![0, 4, 8, 12]);
        lap.set_essential_boundary_condition(Side::Xmax, rig); //  3*   7  11  15
        assert_eq!(lap.essential_sorted, vec![0, 3, 4, 7, 8, 11, 12, 15]);
        lap.set_essential_boundary_condition(Side::Ymin, bot); //  0*   1   2   3
        assert_eq!(lap.essential_sorted, vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 15]);
        lap.set_essential_boundary_condition(Side::Ymax, top); // 12*  13  14  15*  (corner*)
        assert_eq!(lap.essential_sorted, vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]);
        assert_eq!(lap.nodes_xmin, &[0, 4, 8, 12]);
        assert_eq!(lap.nodes_xmax, &[3, 7, 11, 15]);
        assert_eq!(lap.nodes_ymin, &[0, 1, 2, 3]);
        assert_eq!(lap.nodes_ymax, &[12, 13, 14, 15]);
        let mut res = Vec::new();
        lap.loop_over_prescribed_values(|_, m, value| res.push((m, value)));
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
        assert_eq!(lap.num_prescribed(), 12);
        assert_eq!(
            lap.prescribed_flags(),
            &vec![
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
            ]
        );
    }

    #[test]
    fn set_homogeneous_boundary_condition_works() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        lap.set_homogeneous_boundary_conditions();
        assert_eq!(lap.nodes_xmin, &[0, 4, 8, 12]);
        assert_eq!(lap.nodes_xmax, &[3, 7, 11, 15]);
        assert_eq!(lap.nodes_ymin, &[0, 1, 2, 3]);
        assert_eq!(lap.nodes_ymax, &[12, 13, 14, 15]);
        let mut res = Vec::new();
        lap.loop_over_prescribed_values(|_, m, value| res.push((m, value)));
        res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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
    }

    #[test]
    fn lagrange_matrix_works() {
        let n = 4;
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, n, n).unwrap();
        const LEF: f64 = 1.0;
        let lef = |_, _| LEF;
        lap.set_essential_boundary_condition(Side::Xmin, lef); //  0*   4   8  12*
        let ee = lap.lagrange_matrix().unwrap();
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
    fn coefficient_matrix_works() {
        let lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let aa = lap.coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 9);
        assert_eq!(lap.num_prescribed(), 0);
        println!("{}", aa.as_dense());
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                            ┐\n\
             │ -4  2  0  2  0  0  0  0  0 │\n\
             │  1 -4  1  0  2  0  0  0  0 │\n\
             │  0  2 -4  0  0  2  0  0  0 │\n\
             │  1  0  0 -4  2  0  1  0  0 │\n\
             │  0  1  0  1 -4  1  0  1  0 │\n\
             │  0  0  1  0  2 -4  0  0  1 │\n\
             │  0  0  0  2  0  0 -4  2  0 │\n\
             │  0  0  0  0  2  0  1 -4  1 │\n\
             │  0  0  0  0  0  2  0  2 -4 │\n\
             └                            ┘"
        );
    }

    #[test]
    fn augmented_coefficient_matrix_works() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        const LEF: f64 = 1.0;
        let lef = |_, _| LEF;
        lap.set_essential_boundary_condition(Side::Xmin, lef); //  0 3 6
        let ee = lap.lagrange_matrix().unwrap();
        println!("{}", ee.as_dense());
        let aa = lap.augmented_coefficient_matrix(2).unwrap();
        println!("{}", aa.as_dense());
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │ -4  2  0  2  0  0  0  0  0  1  0  0 │\n\
             │  1 -4  1  0  2  0  0  0  0  0  0  0 │\n\
             │  0  2 -4  0  0  2  0  0  0  0  0  0 │\n\
             │  1  0  0 -4  2  0  1  0  0  0  1  0 │\n\
             │  0  1  0  1 -4  1  0  1  0  0  0  0 │\n\
             │  0  0  1  0  2 -4  0  0  1  0  0  0 │\n\
             │  0  0  0  2  0  0 -4  2  0  0  0  1 │\n\
             │  0  0  0  0  2  0  1 -4  1  0  0  0 │\n\
             │  0  0  0  0  0  2  0  2 -4  0  0  0 │\n\
             │  1  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  1  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  1  0  0  0  0  0 │\n\
             └                                     ┘"
        );
    }

    #[test]
    fn mod_coefficient_matrix_works() {
        let lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let (aa, _) = lap.mod_coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 9);
        assert_eq!(lap.num_prescribed(), 0);
        let ___ = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
            [-4.0,  2.0,  ___,  2.0,  ___,  ___,  ___,  ___,  ___],
            [ 1.0, -4.0,  1.0,  ___,  2.0,  ___,  ___,  ___,  ___],
            [ ___,  2.0, -4.0,  ___,  ___,  2.0,  ___,  ___,  ___],
            [ 1.0,  ___,  ___, -4.0,  2.0,  ___,  1.0,  ___,  ___],
            [ ___,  1.0,  ___,  1.0, -4.0,  1.0,  ___,  1.0,  ___],
            [ ___,  ___,  1.0,  ___,  2.0, -4.0,  ___,  ___,  1.0],
            [ ___,  ___,  ___,  2.0,  ___,  ___, -4.0,  2.0,  ___],
            [ ___,  ___,  ___,  ___,  2.0,  ___,  1.0, -4.0,  1.0],
            [ ___,  ___,  ___,  ___,  ___,  2.0,  ___,  2.0, -4.0],
        ]);
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
    }

    #[test]
    fn loop_over_molecule_works() {
        // ┌                            ┐
        // │ -4  2  .  2  .  .  .  .  . │  0
        // │  1 -4  1  .  2  .  .  .  . │  1
        // │  .  2 -4  .  .  2  .  .  . │  2
        // │  1  .  . -4  2  .  1  .  . │  3
        // │  .  1  .  1 -4  1  .  1  . │  4
        // │  .  .  1  .  2 -4  .  .  1 │  5
        // │  .  .  .  2  .  . -4  2  . │  6
        // │  .  .  .  .  2  .  1 -4  1 │  7
        // │  .  .  .  .  .  2  .  2 -4 │  8
        // └                            ┘
        //    0  1  2  3  4  5  6  7  8
        let lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let mut row_0 = Vec::new();
        let mut row_4 = Vec::new();
        let mut row_8 = Vec::new();
        lap.loop_over_coef_mat_row(0, |j, aij| row_0.push((j, aij)));
        lap.loop_over_coef_mat_row(4, |j, aij| row_4.push((j, aij)));
        lap.loop_over_coef_mat_row(8, |j, aij| row_8.push((j, aij)));
        assert_eq!(row_0, &[(0, -4.0), (1, 1.0), (1, 1.0), (3, 1.0), (3, 1.0)]);
        assert_eq!(row_4, &[(4, -4.0), (3, 1.0), (5, 1.0), (1, 1.0), (7, 1.0)]);
        assert_eq!(row_8, &[(8, -4.0), (7, 1.0), (7, 1.0), (5, 1.0), (5, 1.0)]);
    }

    #[test]
    fn mod_coefficient_matrix_with_essential_prescribed_works() {
        // The full matrix is:
        // ┌                                                 ┐
        // │ -4  2  .  2  .  .  .  .  .  .  .  .  .  .  .  . │  0 prescribed
        // │  1 -4  1  .  .  2  .  .  .  .  .  .  .  .  .  . │  1 prescribed
        // │  .  1 -4  1  .  .  2  .  .  .  .  .  .  .  .  . │  2 prescribed
        // │  .  .  2 -4  .  .  .  2  .  .  .  .  .  .  .  . │  3 prescribed
        // │  1  .  .  . -4  2  .  .  1  .  .  .  .  .  .  . │  4 prescribed
        // │  .  1  .  .  1 -4  1  .  .  1  .  .  .  .  .  . │  5
        // │  .  .  1  .  .  1 -4  1  .  .  1  .  .  .  .  . │  6
        // │  .  .  .  1  .  .  2 -4  .  .  .  1  .  .  .  . │  7 prescribed
        // │  .  .  .  .  1  .  .  . -4  2  .  .  1  .  .  . │  8 prescribed
        // │  .  .  .  .  .  1  .  .  1 -4  1  .  .  1  .  . │  9
        // │  .  .  .  .  .  .  1  .  .  1 -4  1  .  .  1  . │ 10
        // │  .  .  .  .  .  .  .  1  .  .  2 -4  .  .  .  1 │ 11 prescribed
        // │  .  .  .  .  .  .  .  .  2  .  .  . -4  2  .  . │ 12 prescribed
        // │  .  .  .  .  .  .  .  .  .  2  .  .  1 -4  1  . │ 13 prescribed
        // │  .  .  .  .  .  .  .  .  .  .  2  .  .  1 -4  1 │ 14 prescribed
        // │  .  .  .  .  .  .  .  .  .  .  .  2  .  .  2 -4 │ 15 prescribed
        // └                                                 ┘
        //    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        //    p  p  p  p  p        p  p        p  p  p  p  p
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        lap.set_homogeneous_boundary_conditions();
        let (aa, cc) = lap.mod_coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 16);
        assert_eq!(lap.num_prescribed(), 12);
        const ___: f64 = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
             [ 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  0 prescribed
             [ ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  1 prescribed
             [ ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  2 prescribed
             [ ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  3 prescribed
             [ ___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  4 prescribed
             [ ___, ___, ___, ___, ___,-4.0, 1.0, ___, ___, 1.0, ___, ___, ___, ___, ___, ___], //  5
             [ ___, ___, ___, ___, ___, 1.0,-4.0, ___, ___, ___, 1.0, ___, ___, ___, ___, ___], //  6
             [ ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___], //  7 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___], //  8 prescribed
             [ ___, ___, ___, ___, ___, 1.0, ___, ___, ___,-4.0, 1.0, ___, ___, ___, ___, ___], //  9
             [ ___, ___, ___, ___, ___, ___, 1.0, ___, ___, 1.0,-4.0, ___, ___, ___, ___, ___], // 10
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___], // 11 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___], // 12 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___], // 13 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___], // 14 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0], // 15 prescribed
         ]); //  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
             //  p    p    p    p    p              p    p              p    p    p    p    p
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
        #[rustfmt::skip]
        let cc_correct = Matrix::from(&[
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  0 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  1 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  2 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  3 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  4 prescribed
             [ ___, 1.0, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  5
             [ ___, ___, 1.0, ___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___], //  6
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  7 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  8 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___, 1.0, ___, ___], //  9
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, 1.0, ___], // 10
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], // 11 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], // 12 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], // 13 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], // 14 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], // 15 prescribed
         ]); //  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
             //  p    p    p    p    p              p    p              p    p    p    p    p
        mat_approx_eq(&cc.as_dense(), &cc_correct, 1e-15);
    }

    #[test]
    fn mod_coefficient_matrix_with_periodic_bcs_works() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3, 4).unwrap();
        lap.set_periodic_boundary_condition(true, true);
        let (aa, cc) = lap.mod_coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 12);
        assert_eq!(cc.get_info().2, 0); // nnz
        const ___: f64 = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
             [-4.0, 1.0, 1.0, 1.0, ___, ___, ___, ___, ___, 1.0, ___, ___], //  0 left  bottom
             [ 1.0,-4.0, 1.0, ___, 1.0, ___, ___, ___, ___, ___, 1.0, ___], //  1       bottom
             [ 1.0, 1.0,-4.0, ___, ___, 1.0, ___, ___, ___, ___, ___, 1.0], //  2 right bottom
             [ 1.0, ___, ___,-4.0, 1.0, 1.0, 1.0, ___, ___, ___, ___, ___], //  3 left
             [ ___, 1.0, ___, 1.0,-4.0, 1.0, ___, 1.0, ___, ___, ___, ___], //  4
             [ ___, ___, 1.0, 1.0, 1.0,-4.0, ___, ___, 1.0, ___, ___, ___], //  5 right
             [ ___, ___, ___, 1.0, ___, ___,-4.0, 1.0, 1.0, 1.0, ___, ___], //  6 left
             [ ___, ___, ___, ___, 1.0, ___, 1.0,-4.0, 1.0, ___, 1.0, ___], //  7
             [ ___, ___, ___, ___, ___, 1.0, 1.0, 1.0,-4.0, ___, ___, 1.0], //  8 right
             [ 1.0, ___, ___, ___, ___, ___, 1.0, ___, ___,-4.0, 1.0, 1.0], //  9 left  top
             [ ___, 1.0, ___, ___, ___, ___, ___, 1.0, ___, 1.0,-4.0, 1.0], // 10       top
             [ ___, ___, 1.0, ___, ___, ___, ___, ___, 1.0, 1.0, 1.0,-4.0], // 11 right top
         ]); //  0    1    2    3    4    5    6    7    8    9   10   11
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
    }

    #[test]
    fn get_grid_coordinates_works() {
        let (nx, ny) = (2, 3);
        let lap = FdmLaplacian2d::new(7.0, 8.0, -1.0, 1.0, -3.0, 3.0, nx, ny).unwrap();
        let mut xx = Matrix::new(ny, nx);
        let mut yy = Matrix::new(ny, nx);
        lap.loop_over_grid_points(|m, x, y| {
            let i = m % nx;
            let j = m / nx;
            xx.set(j, i, x);
            yy.set(j, i, y);
        });
        assert_eq!(
            format!("{}", xx),
            "┌       ┐\n\
             │ -1  1 │\n\
             │ -1  1 │\n\
             │ -1  1 │\n\
             └       ┘"
        );
        assert_eq!(
            format!("{}", yy),
            "┌       ┐\n\
             │ -3 -3 │\n\
             │  0  0 │\n\
             │  3  3 │\n\
             └       ┘"
        );
    }

    #[test]
    fn get_prescribed_value_works() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Set different boundary conditions for each side
        const LEF: f64 = 10.0;
        const RIG: f64 = 20.0;
        const BOT: f64 = 30.0;
        const TOP: f64 = 40.0;

        let lef = |_x, y| LEF + y; // 10 + y
        let rig = |_x, y| RIG + y; // 20 + y
        let bot = |x, _y| BOT + x; // 30 + x
        let top = |x, _y| TOP + x; // 40 + x

        lap.set_essential_boundary_condition(Side::Xmin, lef);
        lap.set_essential_boundary_condition(Side::Xmax, rig);
        lap.set_essential_boundary_condition(Side::Ymin, bot);
        lap.set_essential_boundary_condition(Side::Ymax, top);

        // Test prescribed values on different boundaries
        // Grid layout for 4x4:
        //  12  13  14  15  (y=3)
        //   8   9  10  11  (y=2)
        //   4   5   6   7  (y=1)
        //   0   1   2   3  (y=0)
        // (x=0)(x=1)(x=2)(x=3)

        // Bottom boundary (y=0): should use bot function (30 + x)
        assert_eq!(lap.get_prescribed_value(0).unwrap(), 30.0); // x=0, y=0 -> 30 + 0 = 30
        assert_eq!(lap.get_prescribed_value(1).unwrap(), 31.0); // x=1, y=0 -> 30 + 1 = 31
        assert_eq!(lap.get_prescribed_value(2).unwrap(), 32.0); // x=2, y=0 -> 30 + 2 = 32
        assert_eq!(lap.get_prescribed_value(3).unwrap(), 33.0); // x=3, y=0 -> 30 + 3 = 33

        // Top boundary (y=3): should use top function (40 + x)
        assert_eq!(lap.get_prescribed_value(12).unwrap(), 40.0); // x=0, y=3 -> 40 + 0 = 40
        assert_eq!(lap.get_prescribed_value(13).unwrap(), 41.0); // x=1, y=3 -> 40 + 1 = 41
        assert_eq!(lap.get_prescribed_value(14).unwrap(), 42.0); // x=2, y=3 -> 40 + 2 = 42
        assert_eq!(lap.get_prescribed_value(15).unwrap(), 43.0); // x=3, y=3 -> 40 + 3 = 43

        // Left boundary (x=0): should use lef function (10 + y)
        assert_eq!(lap.get_prescribed_value(4).unwrap(), 11.0); // x=0, y=1 -> 10 + 1 = 11
        assert_eq!(lap.get_prescribed_value(8).unwrap(), 12.0); // x=0, y=2 -> 10 + 2 = 12

        // Right boundary (x=3): should use rig function (20 + y)
        assert_eq!(lap.get_prescribed_value(7).unwrap(), 21.0); // x=3, y=1 -> 20 + 1 = 21
        assert_eq!(lap.get_prescribed_value(11).unwrap(), 22.0); // x=3, y=2 -> 20 + 2 = 22
    }

    #[test]
    fn get_prescribed_value_fails_for_unprescribed_node() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Only set boundary condition on one side
        let lef = |_x, _y| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        // Try to get prescribed value for a node that doesn't have one
        assert_eq!(
            lap.get_prescribed_value(5).err(), // node 5 is interior
            Some("no prescribed value for the given index")
        );

        assert_eq!(
            lap.get_prescribed_value(9).err(), // node 9 is interior
            Some("no prescribed value for the given index")
        );
    }

    #[test]
    fn get_prescribed_value_with_homogeneous_bcs() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        lap.set_homogeneous_boundary_conditions();

        // All boundary nodes should have prescribed value of 0.0
        let boundary_nodes = vec![0, 1, 2, 3, 5, 6, 7, 8]; // all except node 4 (center)

        for &node in &boundary_nodes {
            assert_eq!(lap.get_prescribed_value(node).unwrap(), 0.0);
        }

        // Interior node should not have a prescribed value
        assert_eq!(
            lap.get_prescribed_value(4).err(),
            Some("no prescribed value for the given index")
        );
    }

    #[test]
    fn get_prescribed_value_with_coordinate_dependent_function() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();

        // Set a function that depends on both x and y coordinates
        let func = |x, y| x * x + y * y; // x² + y²
        lap.set_essential_boundary_condition(Side::Xmin, func);

        // Grid coordinates for nx=3, ny=3:
        // (0,2) -> node 6  (1,2) -> node 7  (2,2) -> node 8
        // (0,1) -> node 3  (1,1) -> node 4  (2,1) -> node 5
        // (0,0) -> node 0  (1,0) -> node 1  (2,0) -> node 2

        // Left boundary nodes (x=0): should use func(0, y) = 0² + y² = y²
        assert_eq!(lap.get_prescribed_value(0).unwrap(), 0.0); // x=0, y=0 -> 0² + 0² = 0
        assert_eq!(lap.get_prescribed_value(3).unwrap(), 1.0); // x=0, y=1 -> 0² + 1² = 1
        assert_eq!(lap.get_prescribed_value(6).unwrap(), 4.0); // x=0, y=2 -> 0² + 2² = 4
    }

    #[test]
    fn get_nodes_unknown_works() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Initially, no boundary conditions set, so all nodes are unknown
        let all_nodes: Vec<usize> = (0..16).collect();
        assert_eq!(lap.get_nodes_unknown(), &all_nodes);

        // Set boundary condition on left side (nodes 0, 4, 8, 12)
        let lef = |_x, _y| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        // Unknown nodes should be all except left boundary
        let expected_unknown = vec![1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);

        // Set boundary condition on right side (nodes 3, 7, 11, 15)
        let rig = |_x, _y| 20.0;
        lap.set_essential_boundary_condition(Side::Xmax, rig);

        // Unknown nodes should exclude both left and right boundaries
        let expected_unknown = vec![1, 2, 5, 6, 9, 10, 13, 14];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
    }

    #[test]
    fn get_nodes_unknown_with_homogeneous_bcs() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        lap.set_homogeneous_boundary_conditions();

        // Grid layout for 3x3:
        //  6  7  8
        //  3  4  5
        //  0  1  2

        // Only interior nodes should be unknown (node 4 in this case)
        let expected_unknown = vec![4];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
    }

    #[test]
    fn get_nodes_unknown_with_all_boundaries_prescribed() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Set all boundary conditions
        let lef = |_x, _y| 10.0;
        let rig = |_x, _y| 20.0;
        let bot = |_x, _y| 30.0;
        let top = |_x, _y| 40.0;

        lap.set_essential_boundary_condition(Side::Xmin, lef);
        lap.set_essential_boundary_condition(Side::Xmax, rig);
        lap.set_essential_boundary_condition(Side::Ymin, bot);
        lap.set_essential_boundary_condition(Side::Ymax, top);

        // Grid layout for 4x4:
        //  12* 13* 14* 15*  (y=3) - top boundary
        //   8*  9  10  11*  (y=2)
        //   4*  5   6   7*  (y=1)
        //   0*  1*  2*  3*  (y=0) - bottom boundary
        // (x=0)(x=1)(x=2)(x=3)

        // Only interior nodes should be unknown (nodes 5, 6, 9, 10)
        let expected_unknown = vec![5, 6, 9, 10];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
    }

    #[test]
    fn get_nodes_unknown_with_partial_boundaries() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Set only bottom and top boundaries
        let bot = |_x, _y| 30.0;
        let top = |_x, _y| 40.0;

        lap.set_essential_boundary_condition(Side::Ymin, bot);
        lap.set_essential_boundary_condition(Side::Ymax, top);

        // Unknown nodes should exclude bottom (0,1,2,3) and top (12,13,14,15) boundaries
        let expected_unknown = vec![4, 5, 6, 7, 8, 9, 10, 11];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
    }

    #[test]
    fn get_nodes_unknown_with_periodic_boundaries() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();

        // First set some boundary conditions
        let lef = |_x, _y| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        // Verify that some nodes are prescribed
        let expected_unknown = vec![1, 2, 4, 5, 7, 8];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);

        // Set periodic boundary conditions along x
        lap.set_periodic_boundary_condition(true, false);

        // Grid layout for 3x3:
        //  6  7  8
        //  3  4  5
        //  0  1  2

        // With periodic BC along x, all nodes should be unknown since
        // periodic BC removes essential BC on left/right sides
        let expected_unknown = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
    }

    #[test]
    fn get_nodes_unknown_returns_sorted_indices() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Test 1: Initially all nodes are unknown and should be sorted
        let unknown_initial = lap.get_nodes_unknown();
        let mut sorted_initial = unknown_initial.clone();
        sorted_initial.sort();
        assert_eq!(unknown_initial, &sorted_initial);
        assert_eq!(
            unknown_initial,
            &vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );

        // Test 2: After setting some boundary conditions
        let lef = |_x, _y| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef); // prescribes nodes 0, 4, 8, 12

        let unknown_after_bc = lap.get_nodes_unknown();
        let mut sorted_after_bc = unknown_after_bc.clone();
        sorted_after_bc.sort();
        assert_eq!(unknown_after_bc, &sorted_after_bc);

        // Verify that the unknown nodes are indeed sorted
        for i in 1..unknown_after_bc.len() {
            assert!(
                unknown_after_bc[i] > unknown_after_bc[i - 1],
                "Indices should be sorted in ascending order"
            );
        }

        // Test 3: After setting more boundary conditions
        let rig = |_x, _y| 20.0;
        lap.set_essential_boundary_condition(Side::Xmax, rig); // prescribes nodes 3, 7, 11, 15

        let unknown_final = lap.get_nodes_unknown();
        let mut sorted_final = unknown_final.clone();
        sorted_final.sort();
        assert_eq!(unknown_final, &sorted_final);

        // Expected: [1, 2, 5, 6, 9, 10, 13, 14] - should already be sorted
        assert_eq!(unknown_final, &vec![1, 2, 5, 6, 9, 10, 13, 14]);

        // Verify sorting
        for i in 1..unknown_final.len() {
            assert!(
                unknown_final[i] > unknown_final[i - 1],
                "Final indices should be sorted in ascending order"
            );
        }
    }

    #[test]
    fn get_nodes_unknown_empty_when_all_prescribed() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 2).unwrap();

        // For a 2x2 grid, all 4 nodes are on the boundary
        // Grid layout for 2x2:
        //  2  3
        //  0  1

        lap.set_homogeneous_boundary_conditions();

        // All nodes are on boundary, so no unknown nodes
        let expected_unknown: Vec<usize> = vec![];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
        assert!(lap.get_nodes_unknown().is_empty());
    }

    #[test]
    fn get_nodes_prescribed_works() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Initially, no boundary conditions set, so no nodes are prescribed
        let empty_nodes: Vec<usize> = vec![];
        assert_eq!(lap.get_nodes_prescribed(), &empty_nodes);

        // Set boundary condition on left side (nodes 0, 4, 8, 12)
        let lef = |_x, _y| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        // Prescribed nodes should only include left boundary
        let expected_prescribed = vec![0, 4, 8, 12];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);

        // Set boundary condition on right side (nodes 3, 7, 11, 15)
        let rig = |_x, _y| 20.0;
        lap.set_essential_boundary_condition(Side::Xmax, rig);

        // Prescribed nodes should include both left and right boundaries
        let expected_prescribed = vec![0, 3, 4, 7, 8, 11, 12, 15];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
    }

    #[test]
    fn get_nodes_prescribed_with_homogeneous_bcs() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        lap.set_homogeneous_boundary_conditions();

        // Grid layout for 3x3:
        //  6  7  8
        //  3  4  5
        //  0  1  2

        // All boundary nodes should be prescribed (all except node 4)
        let expected_prescribed = vec![0, 1, 2, 3, 5, 6, 7, 8];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
    }

    #[test]
    fn get_nodes_prescribed_with_all_boundaries_prescribed() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Set all boundary conditions
        let lef = |_x, _y| 10.0;
        let rig = |_x, _y| 20.0;
        let bot = |_x, _y| 30.0;
        let top = |_x, _y| 40.0;

        lap.set_essential_boundary_condition(Side::Xmin, lef);
        lap.set_essential_boundary_condition(Side::Xmax, rig);
        lap.set_essential_boundary_condition(Side::Ymin, bot);
        lap.set_essential_boundary_condition(Side::Ymax, top);

        // Grid layout for 4x4:
        //  12* 13* 14* 15*  (y=3) - top boundary
        //   8*  9  10  11*  (y=2)
        //   4*  5   6   7*  (y=1)
        //   0*  1*  2*  3*  (y=0) - bottom boundary
        // (x=0)(x=1)(x=2)(x=3)

        // All boundary nodes should be prescribed
        let expected_prescribed = vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
    }

    #[test]
    fn get_nodes_prescribed_with_periodic_bc() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // First set essential boundary conditions
        let lef = |_x, _y| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        // Verify that some nodes are prescribed
        let expected_prescribed = vec![0, 4, 8, 12];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);

        // Set periodic boundary condition along x (removes essential BCs on left/right)
        lap.set_periodic_boundary_condition(true, false);

        // With periodic BC along x, no nodes should be prescribed since
        // periodic BC removes essential BC on left/right boundaries
        let expected_prescribed: Vec<usize> = vec![];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
        assert!(lap.get_nodes_prescribed().is_empty());
    }

    #[test]
    fn get_nodes_prescribed_returns_sorted_indices() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Test 1: Initially no nodes are prescribed
        let prescribed_initial = lap.get_nodes_prescribed();
        assert!(prescribed_initial.is_empty());

        // Test 2: Set boundaries in different order to test sorting
        let rig = |_x, _y| 20.0;
        lap.set_essential_boundary_condition(Side::Xmax, rig); // prescribes nodes 3, 7, 11, 15

        let prescribed_after_right = lap.get_nodes_prescribed();
        assert_eq!(prescribed_after_right, &vec![3, 7, 11, 15]);

        // Test 3: Add left boundary condition
        let lef = |_x, _y| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef); // prescribes nodes 0, 4, 8, 12

        let prescribed_final = lap.get_nodes_prescribed();
        let mut sorted_final = prescribed_final.clone();
        sorted_final.sort();
        assert_eq!(prescribed_final, &sorted_final);

        // Expected: [0, 3, 4, 7, 8, 11, 12, 15] - should already be sorted
        assert_eq!(prescribed_final, &vec![0, 3, 4, 7, 8, 11, 12, 15]);

        // Verify sorting
        for i in 1..prescribed_final.len() {
            assert!(
                prescribed_final[i] > prescribed_final[i - 1],
                "Indices should be sorted in ascending order"
            );
        }
    }

    #[test]
    fn get_nodes_prescribed_empty_when_none_prescribed() {
        let lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // No boundary conditions set, so no prescribed nodes
        let expected_prescribed: Vec<usize> = vec![];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
        assert!(lap.get_nodes_prescribed().is_empty());
    }

    #[test]
    fn get_nodes_prescribed_all_when_all_prescribed() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 2).unwrap();

        // For a 2x2 grid, all 4 nodes are on the boundary
        // Grid layout for 2x2:
        //  2  3
        //  0  1

        lap.set_homogeneous_boundary_conditions();

        // All nodes are on boundary, so all nodes are prescribed
        let expected_prescribed = vec![0, 1, 2, 3];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
    }

    #[test]
    fn get_nodes_prescribed_with_partial_boundaries() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Set only bottom and top boundaries
        let bot = |_x, _y| 30.0;
        let top = |_x, _y| 40.0;

        lap.set_essential_boundary_condition(Side::Ymin, bot);
        lap.set_essential_boundary_condition(Side::Ymax, top);

        // Prescribed nodes should include bottom (0,1,2,3) and top (12,13,14,15) boundaries
        let expected_prescribed = vec![0, 1, 2, 3, 12, 13, 14, 15];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
    }

    #[test]
    fn get_nodes_prescribed_complementary_to_unknown() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Test that prescribed + unknown = all nodes at each stage
        let all_nodes: Vec<usize> = (0..16).collect();

        // Stage 1: No BCs
        let mut combined = lap.get_nodes_prescribed().clone();
        combined.extend(lap.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Stage 2: Left BC only
        let lef = |_x, _y| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        let mut combined = lap.get_nodes_prescribed().clone();
        combined.extend(lap.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Stage 3: Add right BC
        let rig = |_x, _y| 20.0;
        lap.set_essential_boundary_condition(Side::Xmax, rig);

        let mut combined = lap.get_nodes_prescribed().clone();
        combined.extend(lap.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Stage 4: Add all boundaries
        let bot = |_x, _y| 30.0;
        let top = |_x, _y| 40.0;
        lap.set_essential_boundary_condition(Side::Ymin, bot);
        lap.set_essential_boundary_condition(Side::Ymax, top);

        let mut combined = lap.get_nodes_prescribed().clone();
        combined.extend(lap.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Stage 5: Periodic BC (removes essential BCs)
        lap.set_periodic_boundary_condition(true, true);

        let mut combined = lap.get_nodes_prescribed().clone();
        combined.extend(lap.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Verify no overlap between prescribed and unknown
        let prescribed = lap.get_nodes_prescribed();
        let unknown = lap.get_nodes_unknown();
        for &p_node in prescribed {
            assert!(
                !unknown.contains(&p_node),
                "Node {} should not be in both prescribed and unknown",
                p_node
            );
        }
        for &u_node in unknown {
            assert!(
                !prescribed.contains(&u_node),
                "Node {} should not be in both unknown and prescribed",
                u_node
            );
        }
    }

    #[test]
    fn get_nodes_prescribed_with_corner_overlaps() {
        let mut lap = FdmLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

        // Set boundary conditions that create corner overlaps
        const LEF: f64 = 10.0;
        const RIG: f64 = 20.0;
        const BOT: f64 = 30.0;
        const TOP: f64 = 40.0;

        let lef = |_x, _y| LEF;
        let rig = |_x, _y| RIG;
        let bot = |_x, _y| BOT;
        let top = |_x, _y| TOP;

        // Add boundaries one by one and check prescribed nodes
        lap.set_essential_boundary_condition(Side::Xmin, lef); // nodes 0, 4, 8, 12
        assert_eq!(lap.get_nodes_prescribed(), &vec![0, 4, 8, 12]);

        lap.set_essential_boundary_condition(Side::Xmax, rig); // adds nodes 3, 7, 11, 15
        assert_eq!(lap.get_nodes_prescribed(), &vec![0, 3, 4, 7, 8, 11, 12, 15]);

        lap.set_essential_boundary_condition(Side::Ymin, bot); // adds nodes 1, 2 (0,3 already prescribed)
        assert_eq!(lap.get_nodes_prescribed(), &vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 15]);

        lap.set_essential_boundary_condition(Side::Ymax, top); // adds nodes 13, 14 (12,15 already prescribed)
        assert_eq!(
            lap.get_nodes_prescribed(),
            &vec![0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]
        );

        // Corner nodes (0, 3, 12, 15) should be prescribed only once
        let prescribed = lap.get_nodes_prescribed();
        let mut unique_prescribed = prescribed.clone();
        unique_prescribed.sort();
        unique_prescribed.dedup();
        assert_eq!(
            prescribed.len(),
            unique_prescribed.len(),
            "No duplicate prescribed nodes should exist"
        );
    }
}
