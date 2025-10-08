use super::Side;
use crate::StrError;
use russell_sparse::{CooMatrix, Sym};
use std::collections::HashMap;
use std::sync::Arc;

/// Implements the Finite Difference (FDM) Laplacian operator in 1D
///
/// Given the (continuum) scalar field ϕ(x) and its Laplacian
///
/// ```text
///           ∂²ϕ
/// L{ϕ} = kx ——— + [cf ϕ]
///           ∂x²
///
/// The term in brackets is optional.
/// ```
///
/// we substitute the partial derivatives using central FDM over a one-dimensional grid.
/// The resulting discrete Laplacian is expressed by the coefficient matrix `A` and the vector `X`:
///
/// ```text
/// D{ϕᵢ} = A ⋅ X
/// ```
///
/// ϕᵢ are the discrete counterpart of ϕ(x) over the (nx) grid.
///
/// The dimension of the coefficient matrix is `dim = nx`.
///
/// A sample grid is illustrated below:
///
/// ```text
///  0───────1───────2───────3───────4
/// i=0     i=1     i=2     i=3     i=4
///                                nx=5
/// ```
///
/// # Remarks
///
/// * The operator is built with a three-point stencil.
/// * The boundary conditions may be Neumann with zero-flux or periodic.
/// * By default (Neumann BC), the boundary nodes are 'mirrored' yielding a no-flux barrier.
pub struct FdmLaplacian1d<'a> {
    xmin: f64,             // min x coordinate
    nx: usize,             // number of points along x (≥ 2)
    dx: f64,               // grid spacing along x
    node_xmin: usize,      // index of the node on the xmin "side"
    node_xmax: usize,      // index of the node on the xmax "side"
    prescribed: Vec<bool>, // flags equations with prescribed EBC values

    /// Indicates that the boundary is periodic along x (left ϕ values equal right ϕ values)
    ///
    /// If false, the left/right boundaries are zero-flux (Neumann with ∂ϕ/dx = 0)
    periodic_along_x: bool,

    /// Holds the FDM coefficients (α, β, β)
    ///
    /// These coefficients are applied over the "bandwidth" of the coefficient matrix
    molecule: Vec<f64>,

    /// Holds the functions to compute essential boundary conditions (ebc)
    ///
    /// The function is `f(x) -> ebc`
    ///
    /// (2) → (xmin, xmax); corresponding to the 2 "sides"
    functions: Vec<Arc<dyn Fn(f64) -> f64 + Send + Sync + 'a>>,

    /// Collects the essential boundary conditions
    ///
    /// Maps node ID to one of the four functions in `functions`
    essential: HashMap<usize, usize>,

    /// Holds the sorted indices of the equations with essential boundary conditions
    essential_sorted: Vec<usize>,

    /// Holds the sorted indices of the equations without essential boundary conditions
    unknown_sorted: Vec<usize>,
}

impl<'a> FdmLaplacian1d<'a> {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `kx` -- diffusion parameter x
    /// * `xmin` -- min x coordinate
    /// * `xmax` -- max x coordinate
    /// * `nx` -- number of points along x (≥ 2)
    pub fn new(kx: f64, xmin: f64, xmax: f64, nx: usize, cf: Option<f64>) -> Result<Self, StrError> {
        // check
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }

        // auxiliary data
        let dx = (xmax - xmin) / ((nx - 1) as f64);
        let dx2 = dx * dx;
        let alpha = match cf {
            Some(value) => -2.0 * kx / dx2 + value,
            None => -2.0 * kx / dx2,
        };
        let beta = kx / dx2;

        // allocate instance
        Ok(FdmLaplacian1d {
            xmin,
            nx,
            dx,
            node_xmin: 0,
            node_xmax: nx - 1,
            prescribed: vec![false; nx],
            periodic_along_x: false,
            molecule: vec![alpha, beta, beta],
            functions: vec![
                Arc::new(|_| 0.0), // xmin
                Arc::new(|_| 0.0), // xmax
            ],
            essential: HashMap::new(),
            essential_sorted: Vec::new(),
            unknown_sorted: (0..nx).collect(), // Initialize with all node indices
        })
    }

    /// Recomputes the prescribed flags array and the essential_sorted array
    fn recompute_arrays(&mut self) {
        self.unknown_sorted.clear();
        for m in 0..self.nx {
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
    pub fn set_periodic_boundary_condition(&mut self) {
        self.periodic_along_x = true;
        self.essential.remove(&self.node_xmin);
        self.essential.remove(&self.node_xmax);
        self.recompute_arrays();
    }

    /// Sets essential (Dirichlet) boundary condition (ebc)
    ///
    /// The function is `f(x) -> ebc`
    ///
    /// **Note:** Any periodic boundary condition on the corresponding side will be removed.
    pub fn set_essential_boundary_condition(&mut self, side: Side, f: impl Fn(f64) -> f64 + Send + Sync + 'a) {
        match side {
            Side::Xmin => {
                self.periodic_along_x = false;
                self.functions[0] = Arc::new(f);
                self.essential.insert(self.node_xmin, 0);
            }
            Side::Xmax => {
                self.periodic_along_x = false;
                self.functions[1] = Arc::new(f);
                self.essential.insert(self.node_xmax, 1);
            }
            Side::Ymin => (),
            Side::Ymax => (),
        };
        self.recompute_arrays();
    }

    /// Sets homogeneous boundary conditions (i.e., zero essential values at the borders)
    ///
    /// **Note:** Periodic boundary conditions will be removed.
    pub fn set_homogeneous_boundary_conditions(&mut self) {
        self.periodic_along_x = false;
        self.essential.clear();
        self.functions = vec![
            Arc::new(|_| 0.0), // xmin
            Arc::new(|_| 0.0), // xmax
        ];
        self.essential.insert(self.node_xmin, 0);
        self.essential.insert(self.node_xmax, 0);
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
        let dim = self.nx;
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
        let dim = self.nx;
        let max_nnz_aa = 3 * dim;

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
        let dim = self.nx;
        let max_nnz_aa = 3 * dim + 2 * np + extra_nnz; // 3 per row + 2 per prescribed equation
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
        let dim = self.nx;
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
            let x = self.xmin + (*m as f64) * self.dx;
            let value = (self.functions[*index])(x);
            callback(ip, *m, value);
            ip += 1;
        });
    }

    /// Executes a loop over the "bandwidth" of the coefficient matrix
    ///
    /// Here, the "bandwidth" means the non-zero values on a row of the coefficient matrix.
    /// This is not the actual bandwidth because the zero elements are ignored. There are
    /// three non-zero values in the "bandwidth" and they correspond to the "molecule" array.
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
        const INI_X: usize = 0;
        let fin_x = self.nx - 1;
        let i = m;

        // n indices of the non-zero values on the row m of the coefficient matrix
        // (mirror or swap the indices of boundary nodes, as appropriate)
        let mut nn = [0, 0, 0];
        nn[CUR] = m;
        if self.periodic_along_x {
            nn[LEF] = if i != INI_X { m - 1 } else { m + fin_x };
            nn[RIG] = if i != fin_x { m + 1 } else { m - fin_x };
        } else {
            nn[LEF] = if i != INI_X { m - 1 } else { m + 1 };
            nn[RIG] = if i != fin_x { m + 1 } else { m - 1 };
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
    /// * `callback` -- a function of `(m, x)` where `m` is the sequential point number,
    ///   and `x` is the Cartesian coordinates of the grid point.
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
        F: FnMut(usize, f64),
    {
        let dim = self.nx;
        for m in 0..dim {
            let x = self.xmin + (m as f64) * self.dx;
            callback(m, x)
        }
    }

    /// Returns the grid spacing
    pub fn grid_spacing(&self) -> f64 {
        self.dx
    }

    /// Returns the dimension of the linear system
    ///
    /// ```text
    /// dim = nx
    /// ```
    pub fn dim(&self) -> usize {
        self.nx
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
            let x = self.xmin + (m as f64) * self.dx;
            Ok((self.functions[*index])(x))
        } else {
            Err("no essential boundary condition for the given node")
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
    use super::{FdmLaplacian1d, Side};
    use russell_lab::{mat_approx_eq, Matrix, Vector};

    #[test]
    fn new_works() {
        let lap = FdmLaplacian1d::new(7.0, -1.0, 1.0, 2, None).unwrap();
        assert_eq!(lap.xmin, -1.0);
        assert_eq!(lap.nx, 2);
        assert_eq!(lap.dx, 2.0);
        assert_eq!(lap.node_xmin, 0);
        assert_eq!(lap.node_xmax, 1);
    }

    #[test]
    fn new_fails_on_invalid_parameters() {
        assert_eq!(
            FdmLaplacian1d::new(1.0, 0.0, 1.0, 1, None).err(),
            Some("nx must be ≥ 2")
        );
    }

    #[test]
    fn set_essential_boundary_condition_works() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 3.0, 4, None).unwrap();
        const LEF: f64 = 1.0;
        const RIG: f64 = 2.0;
        let lef = |_| LEF;
        let rig = |_| RIG;
        lap.set_essential_boundary_condition(Side::Xmin, lef);
        assert_eq!(lap.essential_sorted, vec![0]);
        lap.set_essential_boundary_condition(Side::Xmax, rig);
        assert_eq!(lap.essential_sorted, vec![0, 3]);
        assert_eq!(lap.node_xmin, 0);
        assert_eq!(lap.node_xmax, 3);
        let mut res = Vec::new();
        lap.loop_over_prescribed_values(|_, m, value| res.push((m, value)));
        assert_eq!(res, &[(0, LEF), (3, RIG)]);
        assert_eq!(lap.num_prescribed(), 2);
        assert_eq!(
            lap.prescribed_flags(),
            &vec![
                true,  // 0
                false, // 1
                false, // 2
                true,  // 3
            ]
        );
    }

    #[test]
    fn set_homogeneous_boundary_condition_works() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 3.0, 4, None).unwrap();
        lap.set_homogeneous_boundary_conditions();
        assert_eq!(lap.node_xmin, 0);
        assert_eq!(lap.node_xmax, 3);
        let mut res = Vec::new();
        lap.loop_over_prescribed_values(|_, m, value| res.push((m, value)));
        res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(res, &[(0, 0.0), (3, 0.0),]);
    }

    #[test]
    fn lagrange_matrix_works() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 3.0, 5, None).unwrap();
        const LEF: f64 = 1.0;
        const RIG: f64 = 2.0;
        let lef = |_| LEF;
        let rig = |_| RIG;
        lap.set_essential_boundary_condition(Side::Xmin, lef);
        lap.set_essential_boundary_condition(Side::Xmax, rig);
        let ee = lap.lagrange_matrix().unwrap();
        assert_eq!(
            format!("{}", ee.as_dense()),
            "┌           ┐\n\
             │ 1 0 0 0 0 │\n\
             │ 0 0 0 0 1 │\n\
             └           ┘"
        );
    }

    #[test]
    fn coefficient_matrix_works() {
        let lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();
        let kk = lap.coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 5);
        assert_eq!(lap.num_prescribed(), 0);
        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌                ┐\n\
             │ -2  2  0  0  0 │\n\
             │  1 -2  1  0  0 │\n\
             │  0  1 -2  1  0 │\n\
             │  0  0  1 -2  1 │\n\
             │  0  0  0  2 -2 │\n\
             └                ┘"
        );
    }

    #[test]
    fn augmented_coefficient_matrix_works() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();
        const LEF: f64 = 1.0;
        const RIG: f64 = 2.0;
        let lef = |_| LEF;
        let rig = |_| RIG;
        lap.set_essential_boundary_condition(Side::Xmin, lef);
        lap.set_essential_boundary_condition(Side::Xmax, rig);
        let aa = lap.augmented_coefficient_matrix(2).unwrap();
        println!("{}", aa.as_dense());
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                      ┐\n\
             │ -2  2  0  0  0  1  0 │\n\
             │  1 -2  1  0  0  0  0 │\n\
             │  0  1 -2  1  0  0  0 │\n\
             │  0  0  1 -2  1  0  0 │\n\
             │  0  0  0  2 -2  0  1 │\n\
             │  1  0  0  0  0  0  0 │\n\
             │  0  0  0  0  1  0  0 │\n\
             └                      ┘"
        );
    }

    #[test]
    fn mod_coefficient_matrix_works() {
        let lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();
        let (aa, _) = lap.mod_coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 5);
        assert_eq!(lap.num_prescribed(), 0);
        let ___ = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
            [-2.0,  2.0,  ___,  ___,  ___],
            [ 1.0, -2.0,  1.0,  ___,  ___],
            [ ___,  1.0, -2.0,  1.0,  ___],
            [ ___,  ___,  1.0, -2.0,  1.0],
            [ ___,  ___,  ___,  2.0, -2.0],
        ]);
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
    }

    #[test]
    fn mod_coefficient_matrix_works_with_cf() {
        let lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, Some(-1.0)).unwrap();
        let (aa, _) = lap.mod_coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 5);
        assert_eq!(lap.num_prescribed(), 0);
        let ___ = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
            [-3.0,  2.0,  ___,  ___,  ___],
            [ 1.0, -3.0,  1.0,  ___,  ___],
            [ ___,  1.0, -3.0,  1.0,  ___],
            [ ___,  ___,  1.0, -3.0,  1.0],
            [ ___,  ___,  ___,  2.0, -3.0],
        ]);
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
    }

    #[test]
    fn loop_over_molecule_works() {
        // ┌                 ┐
        // │ -2  2  .  .  .  │  0
        // │  1 -2  1  .  .  │  1
        // │  .  1 -2  1  .  │  2
        // │  .  .  1 -2  1  │  3
        // │  .  .  .  2 -2  │  4
        // └                 ┘
        //    0  1  2  3  4
        let lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();
        let mut row_0 = Vec::new();
        let mut row_2 = Vec::new();
        let mut row_4 = Vec::new();
        lap.loop_over_coef_mat_row(0, |j, aij| row_0.push((j, aij)));
        lap.loop_over_coef_mat_row(2, |j, aij| row_2.push((j, aij)));
        lap.loop_over_coef_mat_row(4, |j, aij| row_4.push((j, aij)));
        assert_eq!(row_0, &[(0, -2.0), (1, 1.0), (1, 1.0)]);
        assert_eq!(row_2, &[(2, -2.0), (1, 1.0), (3, 1.0)]);
        assert_eq!(row_4, &[(4, -2.0), (3, 1.0), (3, 1.0)]);
    }

    #[test]
    fn mod_coefficient_matrix_with_essential_prescribed_works() {
        // The full matrix is:
        // ┌                 ┐
        // │ -2  2  .  .  .  │  0 prescribed
        // │  1 -2  1  .  .  │  1
        // │  .  1 -2  1  .  │  2
        // │  .  .  1 -2  1  │  3
        // │  .  .  .  2 -2  │  4 prescribed
        // └                 ┘
        //    0  1  2  3  4
        //    p           p
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();
        lap.set_homogeneous_boundary_conditions();
        let (aa, cc) = lap.mod_coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 5);
        assert_eq!(lap.num_prescribed(), 2);
        const ___: f64 = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
            [ 1.0,  ___,  ___,  ___,  ___], // 0 prescribed
            [ ___, -2.0,  1.0,  ___,  ___], // 1
            [ ___,  1.0, -2.0,  1.0,  ___], // 2
            [ ___,  ___,  1.0, -2.0,  ___], // 3
            [ ___,  ___,  ___,  ___,  1.0], // 4 prescribed
        ]); //  0     1     2     3     4
            //  p                       p
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
        #[rustfmt::skip]
        let cc_correct = Matrix::from(&[
            [ ___,  ___,  ___,  ___,  ___], // 0 prescribed
            [ 1.0,  ___,  ___,  ___,  ___], // 1
            [ ___,  ___,  ___,  ___,  ___], // 2
            [ ___,  ___,  ___,  ___,  1.0], // 3
            [ ___,  ___,  ___,  ___,  ___], // 4 prescribed
        ]); //  0     1     2     3     4
            //  p                       p
        mat_approx_eq(&cc.as_dense(), &cc_correct, 1e-15);
    }

    #[test]
    fn mod_coefficient_matrix_with_periodic_bcs_works() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();
        lap.set_periodic_boundary_condition();
        let (aa, cc) = lap.mod_coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 5);
        assert_eq!(cc.get_info().2, 0); // nnz
        const ___: f64 = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
            [-2.0,  1.0,  ___,  ___,  1.0],
            [ 1.0, -2.0,  1.0,  ___,  ___],
            [ ___,  1.0, -2.0,  1.0,  ___],
            [ ___,  ___,  1.0, -2.0,  1.0],
            [ 1.0,  ___,  ___,  1.0, -2.0],
        ]);
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
    }

    #[test]
    fn get_grid_coordinates_works() {
        let lap = FdmLaplacian1d::new(7.0, -1.0, 1.0, 3, None).unwrap();
        let mut xx = Vector::new(3);
        lap.loop_over_grid_points(|m, x| {
            xx[m] = x;
        });
        assert_eq!(
            format!("{}", xx),
            "┌    ┐\n\
             │ -1 │\n\
             │  0 │\n\
             │  1 │\n\
             └    ┘"
        );
    }

    #[test]
    fn get_prescribed_value_works() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // Set different boundary conditions for each side
        const LEF: f64 = 10.0;
        const RIG: f64 = 20.0;

        let lef = |x| LEF + x; // 10 + x
        let rig = |x| RIG + x; // 20 + x

        lap.set_essential_boundary_condition(Side::Xmin, lef);
        lap.set_essential_boundary_condition(Side::Xmax, rig);

        // Test prescribed values on boundaries
        // Grid points: x=0, x=1, x=2, x=3, x=4
        // Node 0: x=0 -> lef(0) = 10 + 0 = 10
        // Node 4: x=4 -> rig(4) = 20 + 4 = 24
        assert_eq!(lap.get_prescribed_value(0).unwrap(), 10.0);
        assert_eq!(lap.get_prescribed_value(4).unwrap(), 24.0);
    }

    #[test]
    fn get_prescribed_value_fails_for_unprescribed_node() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // Only set boundary condition on one side
        let lef = |_x| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        // Try to get prescribed value for nodes that don't have one
        assert_eq!(
            lap.get_prescribed_value(1).err(),
            Some("no essential boundary condition for the given node")
        );

        assert_eq!(
            lap.get_prescribed_value(2).err(),
            Some("no essential boundary condition for the given node")
        );

        assert_eq!(
            lap.get_prescribed_value(3).err(),
            Some("no essential boundary condition for the given node")
        );

        assert_eq!(
            lap.get_prescribed_value(4).err(),
            Some("no essential boundary condition for the given node")
        );
    }

    #[test]
    fn get_prescribed_value_with_homogeneous_bcs() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 2.0, 3, None).unwrap();
        lap.set_homogeneous_boundary_conditions();

        // Both boundary nodes should have prescribed value of 0.0
        assert_eq!(lap.get_prescribed_value(0).unwrap(), 0.0); // left boundary
        assert_eq!(lap.get_prescribed_value(2).unwrap(), 0.0); // right boundary

        // Interior node should not have a prescribed value
        assert_eq!(
            lap.get_prescribed_value(1).err(),
            Some("no essential boundary condition for the given node")
        );
    }

    #[test]
    fn get_prescribed_value_with_coordinate_dependent_function() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // Set a function that depends on x coordinate
        let func = |x| x * x; // x²
        lap.set_essential_boundary_condition(Side::Xmin, func);
        lap.set_essential_boundary_condition(Side::Xmax, func);

        // Grid points: x=0, x=1, x=2, x=3, x=4
        // Node 0: x=0 -> func(0) = 0² = 0
        // Node 4: x=4 -> func(4) = 4² = 16
        assert_eq!(lap.get_prescribed_value(0).unwrap(), 0.0);
        assert_eq!(lap.get_prescribed_value(4).unwrap(), 16.0);
    }

    #[test]
    fn get_prescribed_value_with_different_functions_per_side() {
        let mut lap = FdmLaplacian1d::new(1.0, -2.0, 2.0, 5, None).unwrap();

        // Set different functions for each side
        let lef_func = |x| 2.0 * x + 1.0; // 2x + 1
        let rig_func = |x| x * x + 3.0; // x² + 3

        lap.set_essential_boundary_condition(Side::Xmin, lef_func);
        lap.set_essential_boundary_condition(Side::Xmax, rig_func);

        // Grid points: x=-2, x=-1, x=0, x=1, x=2 (with dx=1)
        // Node 0: x=-2 -> lef_func(-2) = 2*(-2) + 1 = -3
        // Node 4: x=2  -> rig_func(2)  = 2² + 3 = 7
        assert_eq!(lap.get_prescribed_value(0).unwrap(), -3.0);
        assert_eq!(lap.get_prescribed_value(4).unwrap(), 7.0);
    }

    #[test]
    fn get_prescribed_value_with_single_boundary() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 3.0, 4, None).unwrap();

        // Set only left boundary condition
        let func = |x| 5.0 + 2.0 * x; // 5 + 2x
        lap.set_essential_boundary_condition(Side::Xmin, func);

        // Grid points: x=0, x=1, x=2, x=3
        // Only node 0 should have prescribed value
        assert_eq!(lap.get_prescribed_value(0).unwrap(), 5.0); // 5 + 2*0 = 5

        // Other nodes should not have prescribed values
        for node in 1..4 {
            assert_eq!(
                lap.get_prescribed_value(node).err(),
                Some("no essential boundary condition for the given node")
            );
        }
    }

    #[test]
    fn get_prescribed_value_after_removing_periodic_bc() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // First set periodic boundary condition
        lap.set_periodic_boundary_condition();

        // No nodes should have prescribed values with periodic BC
        for node in 0..5 {
            assert_eq!(
                lap.get_prescribed_value(node).err(),
                Some("no essential boundary condition for the given node")
            );
        }

        // Now set essential boundary condition (this removes periodic BC)
        let func = |x| x + 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, func);

        // Now node 0 should have prescribed value
        assert_eq!(lap.get_prescribed_value(0).unwrap(), 10.0); // 0 + 10 = 10

        // Other nodes still should not have prescribed values
        for node in 1..5 {
            assert_eq!(
                lap.get_prescribed_value(node).err(),
                Some("no essential boundary condition for the given node")
            );
        }
    }

    #[test]
    fn get_nodes_unknown_works() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // Initially, no boundary conditions set, so all nodes are unknown
        let all_nodes: Vec<usize> = (0..5).collect();
        assert_eq!(lap.get_nodes_unknown(), &all_nodes);

        // Set boundary condition on left side (node 0)
        let lef = |_x| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        // Unknown nodes should be all except left boundary
        let expected_unknown = vec![1, 2, 3, 4];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);

        // Set boundary condition on right side (node 4)
        let rig = |_x| 20.0;
        lap.set_essential_boundary_condition(Side::Xmax, rig);

        // Unknown nodes should exclude both left and right boundaries
        let expected_unknown = vec![1, 2, 3];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
    }

    #[test]
    fn get_nodes_unknown_with_homogeneous_bcs() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 2.0, 3, None).unwrap();
        lap.set_homogeneous_boundary_conditions();

        // Grid layout for 3 nodes:
        //  0  1  2
        //  ^     ^
        // left  right
        // boundary

        // Only interior nodes should be unknown (node 1 in this case)
        let expected_unknown = vec![1];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
    }

    #[test]
    fn get_nodes_unknown_with_all_boundaries_prescribed() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // Set both boundary conditions
        let lef = |_x| 10.0;
        let rig = |_x| 20.0;

        lap.set_essential_boundary_condition(Side::Xmin, lef);
        lap.set_essential_boundary_condition(Side::Xmax, rig);

        // Grid layout for 5 nodes:
        //  0*  1   2   3   4*
        //  ^               ^
        // left            right
        // boundary        boundary

        // Only interior nodes should be unknown (nodes 1, 2, 3)
        let expected_unknown = vec![1, 2, 3];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
    }

    #[test]
    fn get_nodes_unknown_with_periodic_bc() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // First set essential boundary conditions
        let lef = |_x| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        // Verify that some nodes are prescribed
        let expected_unknown = vec![1, 2, 3, 4];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);

        // Set periodic boundary condition (removes essential BCs)
        lap.set_periodic_boundary_condition();

        // With periodic BC, all nodes should be unknown since
        // periodic BC removes essential BC on boundaries
        let expected_unknown = vec![0, 1, 2, 3, 4];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
    }

    #[test]
    fn get_nodes_unknown_returns_sorted_indices() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // Test 1: Initially all nodes are unknown and should be sorted
        let unknown_initial = lap.get_nodes_unknown();
        let mut sorted_initial = unknown_initial.clone();
        sorted_initial.sort();
        assert_eq!(unknown_initial, &sorted_initial);
        assert_eq!(unknown_initial, &vec![0, 1, 2, 3, 4]);

        // Test 2: After setting left boundary condition
        let lef = |_x| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef); // prescribes node 0

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

        // Test 3: After setting right boundary condition
        let rig = |_x| 20.0;
        lap.set_essential_boundary_condition(Side::Xmax, rig); // prescribes node 4

        let unknown_final = lap.get_nodes_unknown();
        let mut sorted_final = unknown_final.clone();
        sorted_final.sort();
        assert_eq!(unknown_final, &sorted_final);

        // Expected: [1, 2, 3] - should already be sorted
        assert_eq!(unknown_final, &vec![1, 2, 3]);

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
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 1.0, 2, None).unwrap();

        // For a 2-node grid, both nodes are on the boundary
        // Grid layout for 2 nodes:
        //  0  1
        //  ^  ^
        // left right

        lap.set_homogeneous_boundary_conditions();

        // All nodes are on boundary, so no unknown nodes
        let expected_unknown: Vec<usize> = vec![];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);
        assert!(lap.get_nodes_unknown().is_empty());
    }

    #[test]
    fn get_nodes_unknown_with_single_boundary() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 3.0, 4, None).unwrap();

        // Set only left boundary condition
        let func = |_x| 5.0;
        lap.set_essential_boundary_condition(Side::Xmin, func);

        // Grid points: 0, 1, 2, 3
        // Only node 0 is prescribed
        let expected_unknown = vec![1, 2, 3];
        assert_eq!(lap.get_nodes_unknown(), &expected_unknown);

        // Clear and set only right boundary condition
        let mut lap2 = FdmLaplacian1d::new(1.0, 0.0, 3.0, 4, None).unwrap();
        lap2.set_essential_boundary_condition(Side::Xmax, func);

        // Only node 3 is prescribed
        let expected_unknown2 = vec![0, 1, 2];
        assert_eq!(lap2.get_nodes_unknown(), &expected_unknown2);
    }

    #[test]
    fn get_nodes_prescribed_works() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // Initially, no boundary conditions set, so no nodes are prescribed
        let empty_nodes: Vec<usize> = vec![];
        assert_eq!(lap.get_nodes_prescribed(), &empty_nodes);

        // Set boundary condition on left side (node 0)
        let lef = |_x| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        // Prescribed nodes should only include left boundary
        let expected_prescribed = vec![0];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);

        // Set boundary condition on right side (node 4)
        let rig = |_x| 20.0;
        lap.set_essential_boundary_condition(Side::Xmax, rig);

        // Prescribed nodes should include both left and right boundaries
        let expected_prescribed = vec![0, 4];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
    }

    #[test]
    fn get_nodes_prescribed_with_homogeneous_bcs() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 2.0, 3, None).unwrap();
        lap.set_homogeneous_boundary_conditions();

        // Grid layout for 3 nodes:
        //  0  1  2
        //  ^     ^
        // left  right
        // boundary

        // Both boundary nodes should be prescribed
        let expected_prescribed = vec![0, 2];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
    }

    #[test]
    fn get_nodes_prescribed_with_all_boundaries_prescribed() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // Set both boundary conditions
        let lef = |_x| 10.0;
        let rig = |_x| 20.0;

        lap.set_essential_boundary_condition(Side::Xmin, lef);
        lap.set_essential_boundary_condition(Side::Xmax, rig);

        // Grid layout for 5 nodes:
        //  0*  1   2   3   4*
        //  ^               ^
        // left            right
        // boundary        boundary

        // Only boundary nodes should be prescribed (nodes 0, 4)
        let expected_prescribed = vec![0, 4];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
    }

    #[test]
    fn get_nodes_prescribed_with_periodic_bc() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // First set essential boundary conditions
        let lef = |_x| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        // Verify that some nodes are prescribed
        let expected_prescribed = vec![0];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);

        // Set periodic boundary condition (removes essential BCs)
        lap.set_periodic_boundary_condition();

        // With periodic BC, no nodes should be prescribed since
        // periodic BC removes essential BC on boundaries
        let expected_prescribed: Vec<usize> = vec![];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
        assert!(lap.get_nodes_prescribed().is_empty());
    }

    #[test]
    fn get_nodes_prescribed_returns_sorted_indices() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // Test 1: Initially no nodes are prescribed
        let prescribed_initial = lap.get_nodes_prescribed();
        assert!(prescribed_initial.is_empty());

        // Test 2: Set right boundary first, then left boundary
        let rig = |_x| 20.0;
        lap.set_essential_boundary_condition(Side::Xmax, rig); // prescribes node 4

        let prescribed_after_right = lap.get_nodes_prescribed();
        assert_eq!(prescribed_after_right, &vec![4]);

        // Test 3: Add left boundary condition
        let lef = |_x| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef); // prescribes node 0

        let prescribed_final = lap.get_nodes_prescribed();
        let mut sorted_final = prescribed_final.clone();
        sorted_final.sort();
        assert_eq!(prescribed_final, &sorted_final);

        // Expected: [0, 4] - should already be sorted
        assert_eq!(prescribed_final, &vec![0, 4]);

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
        let lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // No boundary conditions set, so no prescribed nodes
        let expected_prescribed: Vec<usize> = vec![];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
        assert!(lap.get_nodes_prescribed().is_empty());
    }

    #[test]
    fn get_nodes_prescribed_all_when_all_prescribed() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 1.0, 2, None).unwrap();

        // For a 2-node grid, both nodes are on the boundary
        // Grid layout for 2 nodes:
        //  0  1
        //  ^  ^
        // left right

        lap.set_homogeneous_boundary_conditions();

        // All nodes are on boundary, so all nodes are prescribed
        let expected_prescribed = vec![0, 1];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);
    }

    #[test]
    fn get_nodes_prescribed_with_single_boundary() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 3.0, 4, None).unwrap();

        // Set only left boundary condition
        let func = |_x| 5.0;
        lap.set_essential_boundary_condition(Side::Xmin, func);

        // Grid points: 0, 1, 2, 3
        // Only node 0 is prescribed
        let expected_prescribed = vec![0];
        assert_eq!(lap.get_nodes_prescribed(), &expected_prescribed);

        // Clear and set only right boundary condition
        let mut lap2 = FdmLaplacian1d::new(1.0, 0.0, 3.0, 4, None).unwrap();
        lap2.set_essential_boundary_condition(Side::Xmax, func);

        // Only node 3 is prescribed
        let expected_prescribed2 = vec![3];
        assert_eq!(lap2.get_nodes_prescribed(), &expected_prescribed2);
    }

    #[test]
    fn get_nodes_prescribed_complementary_to_unknown() {
        let mut lap = FdmLaplacian1d::new(1.0, 0.0, 4.0, 5, None).unwrap();

        // Test that prescribed + unknown = all nodes at each stage

        // Stage 1: No BCs
        let all_nodes: Vec<usize> = (0..5).collect();
        let mut combined = lap.get_nodes_prescribed().clone();
        combined.extend(lap.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Stage 2: Left BC only
        let lef = |_x| 10.0;
        lap.set_essential_boundary_condition(Side::Xmin, lef);

        let mut combined = lap.get_nodes_prescribed().clone();
        combined.extend(lap.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Stage 3: Both BCs
        let rig = |_x| 20.0;
        lap.set_essential_boundary_condition(Side::Xmax, rig);

        let mut combined = lap.get_nodes_prescribed().clone();
        combined.extend(lap.get_nodes_unknown().iter());
        combined.sort();
        assert_eq!(combined, all_nodes);

        // Stage 4: Periodic BC (removes essential BCs)
        lap.set_periodic_boundary_condition();

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
}
