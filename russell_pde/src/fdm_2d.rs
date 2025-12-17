use crate::{EquationHandler, EssentialBcs2d, Grid2d, StrError};
use russell_lab::Vector;
use russell_sparse::{CooMatrix, Genie, LinSolver, Sym};

// constants for clarity/convenience
const CUR: usize = 0; // current node
const LEF: usize = 1; // left node
const RIG: usize = 2; // right node
const BOT: usize = 3; // bottom node
const TOP: usize = 4; // top node
const INI_X: usize = 0;
const INI_Y: usize = 0;

/// Implements the Finite Difference method (FDM) in 2D
///
/// The FDM can be used to solve the following problem:
///
/// ```text
///     ∂²ϕ      ∂²ϕ
/// -kx ——— - ky ——— = source(x, y)
///     ∂x²      ∂y²
/// ```
///
/// with essential (EBC) and natural (NBC) boundary conditions.
///
/// The method substitutes the partial derivatives using central differences over a linear grid.
/// The resulting discrete problem is expressed by the coefficient matrix `K` and the vector `a`:
///
/// ```text
/// K a = f
/// ```
///
/// ϕᵢⱼ are the discrete counterpart of ϕ(x, y) over the (nx, ny) grid. However, these
/// values are "sequentially" mapped onto to the vector `a` using the following formula:
///
/// ```text
/// ϕᵢⱼ → aₘ   with   m = i + j nx
/// ```
///
/// To account for the EBCs, two approaches are possible:
///
/// 1. Use the system partitioning strategy (SPS)
/// 2. Use the Lagrange multipliers method (LMM)
///
/// ## Approach 1: System partitioning strategy (SPS)
///
/// Consider the following partitioning of the vectors `a` and `f` and the matrix `K`:
///
/// ```text
/// ┌       ┐ ┌   ┐   ┌   ┐
/// │ K̄   Ǩ │ │ ̄a │   │ f̄ │
/// │       │ │   │ = │   │
/// │ Ḵ   ̰K │ │ ǎ │   │ f̌ │
/// └       ┘ └   ┘   └   ┘
///     K       a       f
/// ```
///
/// where `ā` (a-bar) is a reduced vector containing only the unknown values (i.e., non-EBC nodes), and `ǎ` (a-check)
/// is a reduced vector containing only the prescribed values (i.e., EBC nodes). `f̄` and `f̌` are the associated reduced
/// right-hand side vectors. The `K̄` (K-bar) matrix is the reduced discrete Laplacian operator and `Ǩ` (K-check) is a
/// *correction* matrix. The `Ḵ` (K-underline) and `K̰` (K-under-tilde) matrices are often not needed.
///
/// Thus, the linear system to be solved is:
///
/// ```text
/// K̄ ā = f̄ - Ǩ ǎ
/// ```
///
/// ## Approach 2: Lagrange multipliers method (LMM)
///
/// The LMM consists of augmenting the original linear system with additional equations:
///
/// ```text
/// ┌       ┐ ┌   ┐   ┌   ┐
/// │ K  Cᵀ │ │ a │   │ f │
/// │       │ │   │ = │   │
/// │ C  0  │ │ ℓ │   │ ǎ │
/// └       ┘ └   ┘   └   ┘
///     M       A       F
/// ```
///
/// where `ℓ` is the vector of Lagrange multipliers, `C` is the constraints matrix, and `ǎ` is the vector of
/// prescribed values at EBC nodes. The constraints matrix `C` has a row for each EBC (prescribed) node and a column
/// for every node. Each row in `C` has a single `1` at the column corresponding to the EBC node, and `0`s elsewhere.
///
/// The FDM stencil uses the "molecule" {α, β, β, γ, γ} corresponding to the {CUR, LEF, RIG, BOT, TOP} nodes, such that:
///
/// ```text
/// α ϕ_{cur} + β ϕ_{lef} + β ϕ_{rig} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur}
/// ```
///
/// # Natural boundary conditions (NBC)
///
/// In 2D, the flux vector reduces to `w = [wx, wy]ᵀ`, where
///
/// ```text
/// wx = -kx ∂ϕ/∂x
/// wy = -ky ∂ϕ/∂y
/// ```
///
/// The normal vectors at the boundaries are illustrated below:
///
/// ```text
///                          @ Ymax:
///                               ┌    ┐   ┌    ┐
///                               │ wx │   │  0 │
///                          wₙ = │    │ · │    │
///                               │ wy │   │  1 │
///                               └    ┘   └    ┘
///                             = -ky ∂ϕ/∂y
///
///                                   ↑
/// @ Xmin:                  ┌─────────────────┐     @ Xmax:
///      ┌    ┐   ┌    ┐     │                 │          ┌    ┐   ┌    ┐
///      │ wx │   │ -1 │     │                 │          │ wx │   │  1 │
/// wₙ = │    │ · │    │     │                 │     wₙ = │    │ · │    │
///      │ wy │   │  0 │   ← │                 │ →        │ wy │   │  0 │
///      └    ┘   └    ┘     │                 │          └    ┘   └    ┘
///    = kx ∂ϕ/∂x            │                 │        = -kx ∂ϕ/∂x
///                          │                 │
///                          └─────────────────┘
///                                   ↓
///
///                          @ Ymin:
///                               ┌    ┐   ┌    ┐
///                               │ wx │   │  0 │
///                          wₙ = │    │ · │    │
///                               │ wy │   │ -1 │
///                               └    ┘   └    ┘
///                             = ky ∂ϕ/∂y
/// ```
///
/// The natural boundary conditions (NBC) are set by modifying the right-hand side vector.
///
/// The FDM stencil at the left side (Xmin) is:
///
/// ```text
/// α ϕ_{cur} + β ϕ_{ghost} + β ϕ_{rig} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur}
/// ```
///
/// where `α = 2(kx/Δx²+ky/Δy²)`, `β = -kx/Δx²`, and `γ = -ky/Δy²`.
///
/// The central difference formula for the gradient @ Xmin is:
///
/// ```text
/// ∂ϕ │      ϕ_{rig} - ϕ_{ghost}
/// —— │    ≈ —————————————————— := g
/// ∂x │cur          2 Δx
/// ```
///
/// Thus, `ϕ_{ghost} = ϕ_{rig} - 2 g Δx`.
///
/// By substituting `ϕ_{ghost}` in the FDM stencil, we get:
///
/// ```text
/// α ϕ_{cur} + β (ϕ_{rig} - 2 g Δx) + β ϕ_{rig} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur}
/// α ϕ_{cur} + 2 β ϕ_{rig} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur} + 2 β g Δx
/// α ϕ_{cur} + 2 β ϕ_{rig} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur} - 2 (kx/Δx²) g Δx
/// α ϕ_{cur} + 2 β ϕ_{rig} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur} - 2 (kx g) / Δx
/// α ϕ_{cur} + 2 β ϕ_{rig} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur} - 2 wₙ / Δx
///                                                          └───────────┘
///                                                            extra term
/// ```
///
/// Where `wₙ := kx g = q̄` at the left side (Xmin).
///
/// The FDM stencil at the right side (Xmax) is:
///
/// ```text
/// α ϕ_{cur} + β ϕ_{lef} + β ϕ_{ghost} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur}
/// ```
///
/// Analogously, at the right side (Xmax) the gradient is:
///
/// ```text
/// ∂ϕ │        ϕ_{ghost} - ϕ_{lef}
/// —— │      ≈ ——————————————————— := g
/// ∂x │{cur}           2 Δx
/// ```
///
/// Thus, `ϕ_{ghost} = ϕ_{lef} + 2 g Δx`.
///
/// By substituting `ϕ_{ghost}` in the FDM stencil, we get:
///
/// ```text
/// α ϕ_{cur} + β ϕ_{lef} + β (ϕ_{lef} + 2 g Δx) + γ ϕ_{bot} + γ ϕ_{top} = s_{cur}
/// α ϕ_{cur} + 2 β ϕ_{lef} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur} - 2 β g Δx
/// α ϕ_{cur} + 2 β ϕ_{lef} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur} - 2 (-kx/Δx²) g Δx
/// α ϕ_{cur} + 2 β ϕ_{lef} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur} - 2 (-kx g) / Δx
/// α ϕ_{cur} + 2 β ϕ_{lef} + γ ϕ_{bot} + γ ϕ_{top} = s_{cur} - 2 wₙ / Δx
///                                                          └───────────┘
///                                                            extra term
/// ```
///
/// where `wₙ := -kx g = q̄` at the right side (Xmax).
///
/// Similar expressions can be derived for the bottom (Ymin) and top (Ymax) sides.
pub struct Fdm2d<'a> {
    /// Defines the 2D grid
    grid: Grid2d,

    /// Holds a reference to the essential boundary conditions handler
    ebcs: EssentialBcs2d<'a>,

    /// Tool to handle the equation numbers such as unknowns prescribed
    equations: EquationHandler,

    /// Holds the FDM coefficients (α, β, β, γ, γ) corresponding to (CUR, LEF, RIG, BOT, TOP)
    ///
    /// These coefficients are applied over the "bandwidth" of the coefficient matrix
    molecule: Vec<f64>,
}

impl<'a> Fdm2d<'a> {
    /// Allocates a new instance
    ///
    /// # Arguments
    ///
    /// * `grid` -- the 2D grid
    /// * `ebcs` -- the essential boundary conditions handler
    /// * `kx` -- the diffusion coefficient along x
    /// * `ky` -- the diffusion coefficient along y
    pub fn new(grid: Grid2d, mut ebcs: EssentialBcs2d<'a>, kx: f64, ky: f64) -> Result<Self, StrError> {
        // check grid
        let (dx, dy) = match grid.get_dx_dy() {
            Some((dx, dy)) => (dx, dy),
            None => return Err("grid must have uniform spacing"),
        };

        // build EBC data
        ebcs.build(&grid);

        // allocate equations handler
        let neq = grid.size();
        let mut equations = EquationHandler::new(neq);
        equations.recompute(&ebcs.get_nodes());

        // auxiliary variables
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let alpha = 2.0 * (kx / dx2 + ky / dy2);
        let beta = -kx / dx2;
        let gamma = -ky / dy2;

        // done
        Ok(Fdm2d {
            grid,
            ebcs,
            equations,
            molecule: vec![alpha, beta, beta, gamma, gamma],
        })
    }

    /// Solves the Poisson equation in 2D
    ///
    /// ```text
    ///     ∂²ϕ      ∂²ϕ
    /// -kx ——— - ky ——— = source(x, y)
    ///     ∂x²      ∂y²
    /// ```
    pub fn solve<F>(&self, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64, f64) -> f64,
    {
        // assemble the coefficient matrix and the lhs and rhs vectors
        let (kk_bar, kk_check) = self.get_matrices_sps(0, Sym::No);
        let (mut a_bar, a_check, mut f_bar) = self.get_vectors_sps(source);
        let kk_check = kk_check.unwrap();

        // initialize the right-hand side
        kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check).unwrap(); // f̄ -= Ǩ ǎ

        // solve the linear system
        let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
        solver.actual.factorize(&kk_bar, None).unwrap();
        solver.actual.solve(&mut a_bar, &f_bar, false).unwrap();

        // results
        Ok(self.get_joined_vector_sps(&a_bar, &a_check))
    }

    /// Solves the Poisson equation in 2D (Lagrange multipliers method)
    ///
    /// ```text
    ///     ∂²ϕ      ∂²ϕ
    /// -kx ——— - ky ——— = source(x, y)
    ///     ∂x²      ∂y²
    /// ```
    pub fn solve_lmm<F>(&self, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64, f64) -> f64,
    {
        // assemble the coefficient matrix and the lhs and rhs vectors
        let (mm, _) = self.get_matrices_lmm(0, false);
        let (mut aa, ff) = self.get_vectors_lmm(source);

        // solve the linear system
        let mut solver = LinSolver::new(Genie::Umfpack).unwrap();
        solver.actual.factorize(&mm, None).unwrap();
        solver.actual.solve(&mut aa, &ff, false).unwrap();

        // results
        let neq = self.get_dims_lmm().0;
        Ok(Vector::from(&&aa.as_data()[..neq]))
    }

    /// Returns the dimensions for the system partitioning strategy (SPS)
    ///
    /// Returns `(nu, np)` where:
    ///
    /// * `nu` is the number of unknowns
    /// * `np` is the number of prescribed values
    pub fn get_dims_sps(&self) -> (usize, usize) {
        let nu = self.equations.nu();
        let np = self.equations.np();
        (nu, np)
    }

    /// Returns the dimensions for the Lagrange multipliers method (LMM)
    ///
    /// Returns `(neq, nlag, ndim)` where:
    ///
    /// * `neq` is the number of equations = number of unknowns + number of prescribed values.
    /// * `nlag` is the number of Lagrange multipliers = number of prescribed values.
    /// * `ndim` is the system dimension = number of equations + number of Lagrange multipliers,
    pub fn get_dims_lmm(&self) -> (usize, usize, usize) {
        let neq = self.equations.neq();
        let nlag = self.equations.np();
        let ndim = neq + nlag;
        (neq, nlag, ndim)
    }

    /// Access the grid
    pub fn get_grid(&self) -> &Grid2d {
        &self.grid
    }

    /// Access the equation numbering handler
    pub fn get_equations(&self) -> &EquationHandler {
        &self.equations
    }

    /// Returns the matrices for the system partitioning strategy (SPS)
    ///
    /// Returns `(kk_bar, kk_check)` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K̄   Ǩ │ │ ̄a │   │ f̄ │
    /// │       │ │   │ = │   │
    /// │ Ḵ   ̰K │ │ ǎ │   │ f̌ │
    /// └       ┘ └   ┘   └   ┘
    ///     K       a       f
    /// ```
    ///
    /// # Arguments
    ///
    /// * `extra_nnz` -- extra non-zeros to allocate in the K-bar matrix
    /// * `sym_kk_bar` -- symmetry of the K-bar matrix
    ///
    /// Note that the `K` (K-check) matrix is only available if there are essential boundary conditions.
    pub fn get_matrices_sps(&self, extra_nnz: usize, sym_kk_bar: Sym) -> (CooMatrix, Option<CooMatrix>) {
        let nu = self.equations.nu();
        let np = self.equations.np();
        let nnz_kk_bar = 5 * nu + extra_nnz; // 5 is the bandwidth
        let mut kk_bar = CooMatrix::new(nu, nu, nnz_kk_bar, sym_kk_bar).unwrap();
        let mut kk_check = if np == 0 {
            // russell_sparse requires at least a 1x1 matrix with 1 non-zero entry
            CooMatrix::new(1, 1, 1, Sym::No).unwrap()
        } else {
            let nnz_kk_check = 4 * np; // 4 is the max number of neighbors (worst case)
            CooMatrix::new(nu, np, nnz_kk_check, Sym::No).unwrap()
        };
        self.equations.unknown().iter().for_each(|&m| {
            let iu = self.equations.iu(m);
            self.loop_over_bandwidth(m, |b, n| {
                if self.equations.is_prescribed(n) {
                    let jp = self.equations.ip(n);
                    kk_check.put(iu, jp, self.molecule[b]).unwrap();
                } else {
                    let ju = self.equations.iu(n);
                    kk_bar.put(iu, ju, self.molecule[b]).unwrap();
                }
            });
        });
        if np == 0 {
            (kk_bar, None)
        } else {
            (kk_bar, Some(kk_check))
        }
    }

    /// Returns the matrix for the Lagrange multipliers method (LMM)
    ///
    /// Returns `(mm, cc)` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K  Cᵀ │ │ a │   │ f │
    /// │       │ │   │ = │   │
    /// │ C  0  │ │ ℓ │   │ ǎ │
    /// └       ┘ └   ┘   └   ┘
    ///     M       A       F
    /// ```
    ///
    /// # Arguments
    ///
    /// * `extra_nnz` -- extra non-zeros to allocate in the A matrix
    /// * `get_constraints_mat` -- whether to return the constraints matrix or not
    ///
    /// Note: this matrix is not symmetric because of the flipping (mirroring) strategy for boundary nodes.
    pub fn get_matrices_lmm(&self, extra_nnz: usize, get_constraints_mat: bool) -> (CooMatrix, Option<CooMatrix>) {
        // build the augmented matrix
        let (neq, nlag, ndim) = self.get_dims_lmm();
        let nnz = 5 * neq + 2 * nlag + extra_nnz; // 5 is the bandwidth, 2*nlag is for C and Cᵀ
        let mut mm = CooMatrix::new(ndim, ndim, nnz, Sym::No).unwrap();
        for m in 0..neq {
            self.loop_over_bandwidth(m, |b, n| {
                mm.put(m, n, self.molecule[b]).unwrap();
            });
        }

        // assemble C and Cᵀ into M
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            mm.put(neq + ip, m, 1.0).unwrap(); // C
            mm.put(m, neq + ip, 1.0).unwrap(); // Cᵀ
        });

        // build and return the C matrix, if requested and available
        if get_constraints_mat && nlag > 0 {
            let mut cc = CooMatrix::new(nlag, neq, nlag, Sym::No).unwrap();
            self.equations.prescribed().iter().for_each(|&m| {
                let ip = self.equations.ip(m);
                cc.put(ip, m, 1.0).unwrap(); // C
            });
            (mm, Some(cc))
        } else {
            (mm, None)
        }
    }

    /// Returns the vectors for the solution of the system of equations using the system partitioning strategy (SPS)
    ///
    /// Returns `(a_bar, a_check, f_bar)` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K̄   Ǩ │ │ ̄a │   │ f̄ │
    /// │       │ │   │ = │   │
    /// │ Ḵ   ̰K │ │ ǎ │   │ f̌ │
    /// └       ┘ └   ┘   └   ┘
    ///     K       a       f
    /// ```
    ///
    /// The `source` function calculates f(x, y).
    pub fn get_vectors_sps<F>(&self, source: F) -> (Vector, Vector, Vector)
    where
        F: Fn(f64, f64) -> f64,
    {
        let nu = self.equations.nu();
        let np = self.equations.np();
        let a_bar = Vector::new(nu);
        let mut a_check = Vector::new(np);
        let mut f_bar = Vector::new(nu);
        self.equations.unknown().iter().for_each(|&m| {
            let iu = self.equations.iu(m);
            let (x, y) = self.grid.coord(m);
            f_bar[iu] = source(x, y);
        });
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            let (x, y) = self.grid.coord(m);
            let val = self.ebcs.get_value(m, x, y);
            a_check[ip] = val;
        });
        (a_bar, a_check, f_bar)
    }

    /// Joins the a-bar and a-check vectors used in the system partitioning strategy (SPS)
    ///
    /// Returns `a` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K̄   Ǩ │ │ ̄a │   │ f̄ │
    /// │       │ │   │ = │   │
    /// │ Ḵ   ̰K │ │ ǎ │   │ f̌ │
    /// └       ┘ └   ┘   └   ┘
    ///     K       a       f
    /// ```
    pub fn get_joined_vector_sps(&self, a_bar: &Vector, a_check: &Vector) -> Vector {
        let neq = self.equations.neq();
        let mut a = Vector::new(neq);
        self.equations.unknown().iter().for_each(|&m| {
            let iu = self.equations.iu(m);
            a[m] = a_bar[iu];
        });
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            a[m] = a_check[ip];
        });
        a
    }

    /// Returns the vectors for the solution of the system of equations using the Lagrange multipliers method (LMM)
    ///
    /// Returns `(aa, ff)` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K  Cᵀ │ │ a │   │ f │
    /// │       │ │   │ = │   │
    /// │ C  0  │ │ ℓ │   │ ǎ │
    /// └       ┘ └   ┘   └   ┘
    ///     M       A       F
    /// ```
    ///
    /// The `source` function calculates f(x, y).
    pub fn get_vectors_lmm<F>(&self, source: F) -> (Vector, Vector)
    where
        F: Fn(f64, f64) -> f64,
    {
        let (neq, _, ndim) = self.get_dims_lmm();
        let aa = Vector::new(ndim);
        let mut ff = Vector::new(ndim);
        self.grid.for_each_coord(|m, x, y| {
            ff[m] = source(x, y);
        });
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            let (x, y) = self.grid.coord(m);
            let val = self.ebcs.get_value(m, x, y);
            ff[neq + ip] = val;
        });
        (aa, ff)
    }

    /// Executes a loop over one row of the full coefficient matrix K
    ///
    /// **Note**: The ghost boundary indices are flipped to avoid negative indices.
    /// This also allows the setting up of flux boundary conditions. Therefore, some
    /// column indices may appear repeated; e.g. due to the zero-flux boundaries.
    ///
    /// # Input
    ///
    /// * `m` -- the row of the coefficient matrix
    /// * `callback` -- a `function(n, val_mn)` where `n` is the column index and
    ///   `val_mn` is the m-n-element of the coefficient matrix
    pub fn loop_over_full_coef_mat_row<F>(&self, m: usize, mut callback: F)
    where
        F: FnMut(usize, f64),
    {
        self.loop_over_bandwidth(m, |b, n| {
            callback(n, self.molecule[b]);
        });
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
    pub fn for_each_coord<F>(&self, mut callback: F)
    where
        F: FnMut(usize, f64, f64),
    {
        self.grid.for_each_coord(|m, x, y| {
            callback(m, x, y);
        });
    }

    /// Executes a loop over the "bandwidth" of the coefficient matrix
    ///
    /// **Note**: The ghost boundary indices are flipped to avoid negative indices.
    /// This also allows the setting up of flux boundary conditions. Therefore, some
    /// column indices may appear repeated; e.g. due to the zero-flux boundaries.
    ///
    /// Here, the "bandwidth" means the non-zero values on a row of the coefficient matrix.
    /// This is not the actual bandwidth because the zero elements are ignored. There are
    /// five non-zero values in the "bandwidth" and they correspond to the "molecule" array.
    ///
    /// The callback function is `(b, n)` where `b` is the bandwidth index (index in the molecule array),
    /// and `n` is the column index.
    fn loop_over_bandwidth<F>(&self, m: usize, mut callback: F)
    where
        F: FnMut(usize, usize),
    {
        // constants for clarity/convenience
        let nx = self.grid.nx();
        let ny = self.grid.ny();
        let fin_x = nx - 1;
        let fin_y = ny - 1;
        let i = m % nx;
        let j = m / nx;

        // n indices of the non-zero values on the row m of the coefficient matrix
        // (mirror or swap the indices of boundary nodes, as appropriate)
        let mut nn = [0, 0, 0, 0, 0];
        nn[CUR] = m;
        if self.ebcs.is_periodic_along_x() {
            nn[LEF] = if i != INI_X { m - 1 } else { m + fin_x };
            nn[RIG] = if i != fin_x { m + 1 } else { m - fin_x };
        } else {
            nn[LEF] = if i != INI_X { m - 1 } else { m + 1 };
            nn[RIG] = if i != fin_x { m + 1 } else { m - 1 };
        }
        if self.ebcs.is_periodic_along_y() {
            nn[BOT] = if j != INI_Y { m - nx } else { m + fin_y * nx };
            nn[TOP] = if j != fin_y { m + nx } else { m - fin_y * nx };
        } else {
            nn[BOT] = if j != INI_Y { m - nx } else { m + nx };
            nn[TOP] = if j != fin_y { m + nx } else { m - nx };
        }

        // execute callback
        for b in 0..5 {
            callback(b, nn[b]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Fdm2d;
    use crate::{EssentialBcs2d, Grid2d, Side};
    use russell_lab::Matrix;
    use russell_sparse::Sym;

    #[test]
    fn new_captures_errors() {
        let grid = Grid2d::new(&[0.0, 0.1, 0.4], &[0.0, 0.2, 0.5]).unwrap();
        let ebcs = EssentialBcs2d::new();
        let fdm = Fdm2d::new(grid, ebcs, 1.0, 1.0);
        assert_eq!(fdm.err(), Some("grid must have uniform spacing"));
    }

    #[test]
    fn new_works() {
        //  8  9  10  11
        //  4  5   6   7
        //  0  1   2   3
        // dx = 1.0, dy = 1.0
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 2.0, 4, 3).unwrap();
        let ebcs = EssentialBcs2d::new();

        let fdm = Fdm2d::new(grid, ebcs, 100.0, 300.0).unwrap();
        assert_eq!(&fdm.molecule, &[800.0, -100.0, -100.0, -300.0, -300.0]);

        assert_eq!(fdm.get_dims_sps(), (12, 0));
        assert_eq!(fdm.get_dims_lmm(), (12, 0, 12));
        assert_eq!(fdm.get_grid().size(), 12);
        assert_eq!(fdm.get_equations().neq(), 12);
    }

    #[test]
    fn get_matrices_work() {
        //  8*  9  10  11
        //  4*  5   6   7
        //  0*  1   2   3
        // dx = 1.0, dy = 1.0
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 2.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();
        const LEF: f64 = 1.0;
        let lef = |_, _| LEF;
        assert_eq!(lef(0.0, 0.0), LEF);
        ebcs.set(Side::Xmin, lef); //  0  4  8

        let fdm = Fdm2d::new(grid, ebcs, 100.0, 300.0).unwrap();
        let (kk, cc_mat) = fdm.get_matrices_sps(0, Sym::No);
        let (aa, ee_mat) = fdm.get_matrices_lmm(0, true);
        let cc = cc_mat.unwrap();
        let ee = ee_mat.unwrap();

        assert_eq!(fdm.get_dims_sps(), (9, 3));
        assert_eq!(fdm.get_dims_lmm(), (12, 3, 15));

        // The full matrix is:
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11
        // ┌                                                             ┐
        // │  800 -200    .    . -600    .    .    .    .    .    .    . │  0*(p0)
        // │ -100  800 -100    .    . -600    .    .    .    .    .    . │  1→0
        // │    . -100  800 -100    .    . -600    .    .    .    .    . │  2→1
        // │    .    . -200  800    .    .    . -600    .    .    .    . │  3→2
        // │ -300    .    .    .  800 -200    .    . -300    .    .    . │  4*(p1)
        // │    . -300    .    . -100  800 -100    .    . -300    .    . │  5→3
        // │    .    . -300    .    . -100  800 -100    .    . -300    . │  6→4
        // │    .    .    . -300    .    . -200  800    .    .    . -300 │  7→5
        // │    .    .    .    . -600    .    .    .  800 -200    .    . │  8*(p3)
        // │    .    .    .    .    . -600    .    . -100  800 -100    . │  9→6
        // │    .    .    .    .    .    . -600    .    . -100  800 -100 │ 10→7
        // │    .    .    .    .    .    .    . -600    .    . -200  800 │ 11→8
        // └                                                             ┘
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11

        // K =
        //      1    2    3    5    6    7    9   10   11
        // ┌                                              ┐
        // │  800 -100    . -600    .    .    .    .    . │  1→0
        // │ -100  800 -100    . -600    .    .    .    . │  2→1
        // │    . -200  800    .    . -600    .    .    . │  3→2
        // │ -300    .    .  800 -100    . -300    .    . │  5→3
        // │    . -300    . -100  800 -100    . -300    . │  6→4
        // │    .    . -300    . -200  800    .    . -300 │  7→5
        // │    .    .    . -600    .    .  800 -100    . │  9→6
        // │    .    .    .    . -600    . -100  800 -100 │ 10→7
        // │    .    .    .    .    . -600    . -200  800 │ 11→8
        // └                                              ┘
        //      1    2    3    5    6    7    9   10   11
        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌                                              ┐\n\
             │  800 -100    0 -600    0    0    0    0    0 │\n\
             │ -100  800 -100    0 -600    0    0    0    0 │\n\
             │    0 -200  800    0    0 -600    0    0    0 │\n\
             │ -300    0    0  800 -100    0 -300    0    0 │\n\
             │    0 -300    0 -100  800 -100    0 -300    0 │\n\
             │    0    0 -300    0 -200  800    0    0 -300 │\n\
             │    0    0    0 -600    0    0  800 -100    0 │\n\
             │    0    0    0    0 -600    0 -100  800 -100 │\n\
             │    0    0    0    0    0 -600    0 -200  800 │\n\
             └                                              ┘"
        );

        // C =
        //     0*  4*  8*
        // ┌             ┐
        // │ -100   .   . │  1→0
        // │    .   .   . │  2→1
        // │    .   .   . │  3→2
        // │    . -100  . │  5→3
        // │    .   .   . │  6→4
        // │    .   .   . │  7→5
        // │    .   . -100│  9→6
        // │    .   .   . │ 10→7
        // │    .   .   . │ 11→8
        // └             ┘
        //     0*  4*  8*
        assert_eq!(
            format!("{}", cc.as_dense()),
            "┌                ┐\n\
             │ -100    0    0 │\n\
             │    0    0    0 │\n\
             │    0    0    0 │\n\
             │    0 -100    0 │\n\
             │    0    0    0 │\n\
             │    0    0    0 │\n\
             │    0    0 -100 │\n\
             │    0    0    0 │\n\
             │    0    0    0 │\n\
             └                ┘"
        );

        // E =
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11
        // ┌                                                             ┐
        // │    1    .    .    .    .    .    .    .    .    .    .    . │  0*
        // │    .    .    .    .    1    .    .    .    .    .    .    . │  4*
        // │    .    .    .    .    .    .    .    .    1    .    .    . │  8*
        // └                                                             ┘
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11
        assert_eq!(
            format!("{}", ee.as_dense()),
            "┌                         ┐\n\
             │ 1 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 1 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 1 0 0 0 │\n\
             └                         ┘"
        );

        // A =
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11   12w  13w  14w
        // ┌                                                                            ┐
        // │  800 -200    .    . -600    .    .    .    .    .    .    .    1    .    . │  0*
        // │ -100  800 -100    .    . -600    .    .    .    .    .    .    .    .    . │  1
        // │    . -100  800 -100    .    . -600    .    .    .    .    .    .    .    . │  2
        // │    .    . -200  800    .    .    . -600    .    .    .    .    .    .    . │  3
        // │ -300    .    .    .  800 -200    .    . -300    .    .    .    .    1    . │  4*
        // │    . -300    .    . -100  800 -100    .    . -300    .    .    .    .    . │  5
        // │    .    . -300    .    . -100  800 -100    .    . -300    .    .    .    . │  6
        // │    .    .    . -300    .    . -200  800    .    .    . -300    .    .    . │  7
        // │    .    .    .    . -600    .    .    .  800 -200    .    .    .    .    1 │  8*
        // │    .    .    .    .    . -600    .    . -100  800 -100    .    .    .    . │  9
        // │    .    .    .    .    .    . -600    .    . -100  800 -100    .    .    . │ 10
        // │    .    .    .    .    .    .    . -600    .    . -200  800    .    .    . │ 11
        // │    1    .    .    .    .    .    .    .    .    .    .    .    .    .    . │ 12w
        // │    .    .    .    .    1    .    .    .    .    .    .    .    .    .    . │ 13w
        // │    .    .    .    .    .    .    .    .    1    .    .    .    .    .    . │ 14w
        // └                                                                            ┘
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11   12w  13w  14w
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                                                            ┐\n\
             │  800 -200    0    0 -600    0    0    0    0    0    0    0    1    0    0 │\n\
             │ -100  800 -100    0    0 -600    0    0    0    0    0    0    0    0    0 │\n\
             │    0 -100  800 -100    0    0 -600    0    0    0    0    0    0    0    0 │\n\
             │    0    0 -200  800    0    0    0 -600    0    0    0    0    0    0    0 │\n\
             │ -300    0    0    0  800 -200    0    0 -300    0    0    0    0    1    0 │\n\
             │    0 -300    0    0 -100  800 -100    0    0 -300    0    0    0    0    0 │\n\
             │    0    0 -300    0    0 -100  800 -100    0    0 -300    0    0    0    0 │\n\
             │    0    0    0 -300    0    0 -200  800    0    0    0 -300    0    0    0 │\n\
             │    0    0    0    0 -600    0    0    0  800 -200    0    0    0    0    1 │\n\
             │    0    0    0    0    0 -600    0    0 -100  800 -100    0    0    0    0 │\n\
             │    0    0    0    0    0    0 -600    0    0 -100  800 -100    0    0    0 │\n\
             │    0    0    0    0    0    0    0 -600    0    0 -200  800    0    0    0 │\n\
             │    1    0    0    0    0    0    0    0    0    0    0    0    0    0    0 │\n\
             │    0    0    0    0    1    0    0    0    0    0    0    0    0    0    0 │\n\
             │    0    0    0    0    0    0    0    0    1    0    0    0    0    0    0 │\n\
             └                                                                            ┘"
        );
    }

    #[test]
    fn get_matrices_homogeneous_bcs_work() {
        //       8   9  10  11
        //      ---------------
        // 13 | 12* 13* 14* 15* | 14
        //  9 |  8*  9  10  11* | 10
        //  5 |  4*  5   6   7* |  6
        //  1 |  0*  1*  2*  3* |  2
        //      ---------------
        //       4   5   6   7
        // dx = 1.0, dy = 1.0
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_homogeneous();

        let fdm = Fdm2d::new(grid, ebcs, 1.0, 1.0).unwrap();
        let (kk, cc_mat) = fdm.get_matrices_sps(0, Sym::No);
        let (aa, ee_mat) = fdm.get_matrices_lmm(0, true);
        let cc = cc_mat.unwrap();
        let ee = ee_mat.unwrap();

        assert_eq!(fdm.get_dims_sps(), (4, 12));
        assert_eq!(fdm.get_dims_lmm(), (16, 12, 28));

        // The full matrix is:
        //    0* 1* 2* 3* 4* 5  6  7* 8* 9 10 11*12*13*14*15*
        // ┌                                                 ┐
        // │  4 -2  .  . -2  .  .  .  .  .  .  .  .  .  .  . │  0*
        // │ -1  4 -1  .  . -2  .  .  .  .  .  .  .  .  .  . │  1*
        // │  . -1  4 -1  .  . -2  .  .  .  .  .  .  .  .  . │  2*
        // │  .  . -2  4  .  .  . -2  .  .  .  .  .  .  .  . │  3*
        // │ -1  .  .  .  4 -2  .  . -1  .  .  .  .  .  .  . │  4*
        // │  . -1  .  . -1  4 -1  .  . -1  .  .  .  .  .  . │  5
        // │  .  . -1  .  . -1  4 -1  .  . -1  .  .  .  .  . │  6
        // │  .  .  . -1  .  . -2  4  .  .  . -1  .  .  .  . │  7*
        // │  .  .  .  . -1  .  .  .  4 -2  .  . -1  .  .  . │  8*
        // │  .  .  .  .  . -1  .  . -1  4 -1  .  . -1  .  . │  9
        // │  .  .  .  .  .  . -1  .  . -1  4 -1  .  . -1  . │ 10
        // │  .  .  .  .  .  .  . -1  .  . -2  4  .  .  . -1 │ 11*
        // │  .  .  .  .  .  .  .  . -2  .  .  .  4 -2  .  . │ 12*
        // │  .  .  .  .  .  .  .  .  . -2  .  . -1  4 -1  . │ 13*
        // │  .  .  .  .  .  .  .  .  .  . -2  .  . -1  4 -1 │ 14*
        // │  .  .  .  .  .  .  .  .  .  .  . -2  .  . -2  4 │ 15*
        // └                                                 ┘
        //    0* 1* 2* 3* 4* 5  6  7* 8* 9 10 11*12*13*14*15*

        // K =
        //    5  6  9 10 *
        // ┌              ┐
        // │  4 -1 -1  .  │  5
        // │ -1  4  . -1  │  6
        // │ -1  .  4 -1  │  9
        // │  . -1 -1  4  │ 10
        // └              ┘
        //    5  6  9 10 *
        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌             ┐\n\
             │  4 -1 -1  0 │\n\
             │ -1  4  0 -1 │\n\
             │ -1  0  4 -1 │\n\
             │  0 -1 -1  4 │\n\
             └             ┘"
        );

        // C =
        //    0* 1* 2* 3* 4* 7* 8*11*12*13*14*15*
        // ┌                                     ┐
        // │  . -1  .  . -1  .  .  .  .  .  .  . │  5
        // │  .  . -1  .  . -1  .  .  .  .  .  . │  6
        // │  .  .  .  .  .  . -1  .  . -1  .  . │  9
        // │  .  .  .  .  .  .  . -1  .  . -1  . │ 10
        // └                                     ┘
        //    0* 1* 2* 3* 4* 7* 8*11*12*13*14*15*
        //
        assert_eq!(
            format!("{}", cc.as_dense()),
            "┌                                     ┐\n\
             │  0 -1  0  0 -1  0  0  0  0  0  0  0 │\n\
             │  0  0 -1  0  0 -1  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0 -1  0  0 -1  0  0 │\n\
             │  0  0  0  0  0  0  0 -1  0  0 -1  0 │\n\
             └                                     ┘"
        );

        // E =
        //    0* 1* 2* 3* 4* 5  6  7* 8* 9 10 11*12*13*14*15*
        // ┌                                                 ┐
        // │  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │  0*
        // │  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  . │  1*
        // │  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  . │  2*
        // │  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  . │  3*
        // │  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  . │  4*
        // │  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  . │  7*
        // │  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  . │  8*
        // │  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  . │ 11*
        // │  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  . │ 12*
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  1  .  . │ 13*
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  .  1  . │ 14*
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  1 │ 15*
        // └                                                 ┘
        assert_eq!(
            format!("{}", ee.as_dense()),
            "┌                                 ┐\n\
             │ 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 │\n\
             └                                 ┘"
        );

        // A =
        //    0* 1* 2* 3* 4* 5  6  7* 8* 9 10 11*12*13*14*15*16w17w18w19w20w21w22w23w24w25w26w27w
        // ┌                                                                                     ┐
        // │  4 -2  .  . -2  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  . │  0*
        // │ -1  4 -1  .  . -2  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  . │  1*
        // │  . -1  4 -1  .  . -2  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  . │  2*
        // │  .  . -2  4  .  .  . -2  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  . │  3*
        // │ -1  .  .  .  4 -2  .  . -1  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  . │  4*
        // │  . -1  .  . -1  4 -1  .  . -1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │  5
        // │  .  . -1  .  . -1  4 -1  .  . -1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │  6
        // │  .  .  . -1  .  . -2  4  .  .  . -1  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  . │  7*
        // │  .  .  .  . -1  .  .  .  4 -2  .  . -1  .  .  .  .  .  .  .  .  .  1  .  .  .  .  . │  8*
        // │  .  .  .  .  . -1  .  . -1  4 -1  .  . -1  .  .  .  .  .  .  .  .  .  .  .  .  .  . │  9
        // │  .  .  .  .  .  . -1  .  . -1  4 -1  .  . -1  .  .  .  .  .  .  .  .  .  .  .  .  . │ 10
        // │  .  .  .  .  .  .  . -1  .  . -2  4  .  .  . -1  .  .  .  .  .  .  .  1  .  .  .  . │ 11*
        // │  .  .  .  .  .  .  .  . -2  .  .  .  4 -2  .  .  .  .  .  .  .  .  .  .  1  .  .  . │ 12*
        // │  .  .  .  .  .  .  .  .  . -2  .  . -1  4 -1  .  .  .  .  .  .  .  .  .  .  1  .  . │ 13*
        // │  .  .  .  .  .  .  .  .  .  . -2  .  . -1  4 -1  .  .  .  .  .  .  .  .  .  .  1  . │ 14*
        // │  .  .  .  .  .  .  .  .  .  .  . -2  .  . -2  4  .  .  .  .  .  .  .  .  .  .  .  1 │ 15*
        // │  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 16w
        // │  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 17w
        // │  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 18w
        // │  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 19w
        // │  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 20w
        // │  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 21w
        // │  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 22w
        // │  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 23w
        // │  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 24w
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 25w
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  . │ 26w
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  . │ 27w
        // └                                                                                     ┘
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                                                                     ┐\n\
             │  4 -2  0  0 -2  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │ -1  4 -1  0  0 -2  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0 -1  4 -1  0  0 -2  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0 -2  4  0  0  0 -2  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0 │\n\
             │ -1  0  0  0  4 -2  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0 │\n\
             │  0 -1  0  0 -1  4 -1  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0 -1  0  0 -1  4 -1  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0 -1  0  0 -2  4  0  0  0 -1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0 │\n\
             │  0  0  0  0 -1  0  0  0  4 -2  0  0 -1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0 │\n\
             │  0  0  0  0  0 -1  0  0 -1  4 -1  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0 -1  0  0 -1  4 -1  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0 -1  0  0 -2  4  0  0  0 -1  0  0  0  0  0  0  0  1  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0 -2  0  0  0  4 -2  0  0  0  0  0  0  0  0  0  0  1  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0 -2  0  0 -1  4 -1  0  0  0  0  0  0  0  0  0  0  1  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0 -2  0  0 -1  4 -1  0  0  0  0  0  0  0  0  0  0  1  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0 -2  0  0 -2  4  0  0  0  0  0  0  0  0  0  0  0  1 │\n\
             │  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             └                                                                                     ┘"
        );
    }

    #[test]
    fn get_matrices_periodic_bcs_work() {
        //      0  1  2
        //     --------
        // 11 | 9 10 11 | 9
        //  8 | 6  7  8 | 6
        //  5 | 3  4  5 | 3
        //  2 | 0  1  2 | 0
        //      -------
        //      9 10 11

        let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 3.0, 3, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_periodic(true, true);

        let fdm = Fdm2d::new(grid, ebcs, 1.0, 1.0).unwrap();
        let (kk, cc_mat) = fdm.get_matrices_sps(0, Sym::No);
        let (aa, ee_mat) = fdm.get_matrices_lmm(0, true);
        assert!(cc_mat.is_none());
        assert!(ee_mat.is_none());

        assert_eq!(fdm.get_dims_sps(), (12, 0));
        assert_eq!(fdm.get_dims_lmm(), (12, 0, 12));

        // K = A =
        //    0  1  2  3  4  5  6  7  8  9 10 11
        // ┌                                     ┐
        // │  4 -1 -1 -1  .  .  .  .  . -1  .  . │  0
        // │ -1  4 -1  . -1  .  .  .  .  . -1  . │  1
        // │ -1 -1  4  .  . -1  .  .  .  .  . -1 │  2
        // │ -1  .  .  4 -1 -1 -1  .  .  .  .  . │  3
        // │  . -1  . -1  4 -1  . -1  .  .  .  . │  4
        // │  .  . -1 -1 -1  4  .  . -1  .  .  . │  5
        // │  .  .  . -1  .  .  4 -1 -1 -1  .  . │  6
        // │  .  .  .  . -1  . -1  4 -1  . -1  . │  7
        // │  .  .  .  .  . -1 -1 -1  4  .  . -1 │  8
        // │ -1  .  .  .  .  . -1  .  .  4 -1 -1 │  9
        // │  . -1  .  .  .  .  . -1  . -1  4 -1 │ 10
        // │  .  . -1  .  .  .  .  . -1 -1 -1  4 │ 11
        // └                                     ┘
        //    0  1  2  3  4  5  6  7  8  9 10 11

        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌                                     ┐\n\
             │  4 -1 -1 -1  0  0  0  0  0 -1  0  0 │\n\
             │ -1  4 -1  0 -1  0  0  0  0  0 -1  0 │\n\
             │ -1 -1  4  0  0 -1  0  0  0  0  0 -1 │\n\
             │ -1  0  0  4 -1 -1 -1  0  0  0  0  0 │\n\
             │  0 -1  0 -1  4 -1  0 -1  0  0  0  0 │\n\
             │  0  0 -1 -1 -1  4  0  0 -1  0  0  0 │\n\
             │  0  0  0 -1  0  0  4 -1 -1 -1  0  0 │\n\
             │  0  0  0  0 -1  0 -1  4 -1  0 -1  0 │\n\
             │  0  0  0  0  0 -1 -1 -1  4  0  0 -1 │\n\
             │ -1  0  0  0  0  0 -1  0  0  4 -1 -1 │\n\
             │  0 -1  0  0  0  0  0 -1  0 -1  4 -1 │\n\
             │  0  0 -1  0  0  0  0  0 -1 -1 -1  4 │\n\
             └                                     ┘"
        );
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │  4 -1 -1 -1  0  0  0  0  0 -1  0  0 │\n\
             │ -1  4 -1  0 -1  0  0  0  0  0 -1  0 │\n\
             │ -1 -1  4  0  0 -1  0  0  0  0  0 -1 │\n\
             │ -1  0  0  4 -1 -1 -1  0  0  0  0  0 │\n\
             │  0 -1  0 -1  4 -1  0 -1  0  0  0  0 │\n\
             │  0  0 -1 -1 -1  4  0  0 -1  0  0  0 │\n\
             │  0  0  0 -1  0  0  4 -1 -1 -1  0  0 │\n\
             │  0  0  0  0 -1  0 -1  4 -1  0 -1  0 │\n\
             │  0  0  0  0  0 -1 -1 -1  4  0  0 -1 │\n\
             │ -1  0  0  0  0  0 -1  0  0  4 -1 -1 │\n\
             │  0 -1  0  0  0  0  0 -1  0 -1  4 -1 │\n\
             │  0  0 -1  0  0  0  0  0 -1 -1 -1  4 │\n\
             └                                     ┘"
        );
    }

    #[test]
    fn get_vectors_works() {
        let grid = Grid2d::new_uniform(1.0, 4.0, 1.0, 3.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();

        //  8*  9* 10* 11*
        //  4*  5   6   7*
        //  0*  1*  2*  3*
        ebcs.set(Side::Xmin, |x, y| x + y);
        ebcs.set(Side::Xmax, |x, y| x + y);
        ebcs.set(Side::Ymin, |x, y| x + y);
        ebcs.set(Side::Ymax, |x, y| x + y);

        let fdm = Fdm2d::new(grid, ebcs, 1.0, 1.0).unwrap();

        let nu = 2;
        let np = 10;
        let neq = nu + np;

        let (a_bar, a_check, f_bar) = fdm.get_vectors_sps(|_, _| 100.0);
        assert_eq!(a_bar.dim(), nu);
        assert_eq!(a_check.dim(), np);
        assert_eq!(f_bar.dim(), nu);

        assert_eq!(a_bar.as_data(), &[0.0, 0.0]);
        assert_eq!(
            a_check.as_data(),
            &[
                1.0 + 1.0, //  0*
                2.0 + 1.0, //  1*
                3.0 + 1.0, //  2*
                4.0 + 1.0, //  3*
                1.0 + 2.0, //  4*
                // 2.0 + 2.0, //  5
                // 3.0 + 2.0, //  6
                4.0 + 2.0, //  7*
                1.0 + 3.0, //  8*
                2.0 + 3.0, //  9*
                3.0 + 3.0, // 10*
                4.0 + 3.0, // 11*
            ]
        );
        assert_eq!(f_bar.as_data(), &[100.0, 100.0]);

        let a = fdm.get_joined_vector_sps(&a_bar, &a_check);
        assert_eq!(a.dim(), neq);
        assert_eq!(
            a.as_data(),
            &[
                1.0 + 1.0, //  0*
                2.0 + 1.0, //  1*
                3.0 + 1.0, //  2*
                4.0 + 1.0, //  3*
                1.0 + 2.0, //  4*
                0.0,       //  5
                0.0,       //  6
                4.0 + 2.0, //  7*
                1.0 + 3.0, //  8*
                2.0 + 3.0, //  9*
                3.0 + 3.0, // 10*
                4.0 + 3.0, // 11*
            ]
        );

        let (aa, ff) = fdm.get_vectors_lmm(|_, _| 100.0);
        assert_eq!(aa.dim(), neq + np);
        assert_eq!(aa.as_data(), &vec![0.0; neq + np]);
        assert_eq!(
            ff.as_data(),
            &[
                100.0,     // 0
                100.0,     // 1
                100.0,     // 2
                100.0,     // 3
                100.0,     // 4
                100.0,     // 5
                100.0,     // 6
                100.0,     // 7
                100.0,     // 8
                100.0,     // 9
                100.0,     // 10
                100.0,     // 11
                1.0 + 1.0, //  0*
                2.0 + 1.0, //  1*
                3.0 + 1.0, //  2*
                4.0 + 1.0, //  3*
                1.0 + 2.0, //  4*
                4.0 + 2.0, //  7*
                1.0 + 3.0, //  8*
                2.0 + 3.0, //  9*
                3.0 + 3.0, // 10*
                4.0 + 3.0, // 11*
            ]
        );
    }

    #[test]
    fn loop_over_mm_row_works() {
        // ┌                            ┐
        // │  4 -2  . -2  .  .  .  .  . │  0
        // │ -1  4 -1  . -2  .  .  .  . │  1
        // │  . -2  4  .  . -2  .  .  . │  2
        // │ -1  .  .  4 -2  . -1  .  . │  3
        // │  . -1  . -1  4 -1  . -1  . │  4
        // │  .  . -1  . -2  4  .  . -1 │  5
        // │  .  .  . -2  .  .  4 -2  . │  6
        // │  .  .  .  . -2  . -1  4 -1 │  7
        // │  .  .  .  .  . -2  . -2  4 │  8
        // └                            ┘
        //    0  1  2  3  4  5  6  7  8
        let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let ebcs = EssentialBcs2d::new();
        let lap = Fdm2d::new(grid, ebcs, 1.0, 1.0).unwrap();
        let mut row_0 = Vec::new();
        let mut row_4 = Vec::new();
        let mut row_8 = Vec::new();
        lap.loop_over_full_coef_mat_row(0, |j, aij| row_0.push((j, aij)));
        lap.loop_over_full_coef_mat_row(4, |j, aij| row_4.push((j, aij)));
        lap.loop_over_full_coef_mat_row(8, |j, aij| row_8.push((j, aij)));
        assert_eq!(row_0, &[(0, 4.0), (1, -1.0), (1, -1.0), (3, -1.0), (3, -1.0)]);
        assert_eq!(row_4, &[(4, 4.0), (3, -1.0), (5, -1.0), (1, -1.0), (7, -1.0)]);
        assert_eq!(row_8, &[(8, 4.0), (7, -1.0), (7, -1.0), (5, -1.0), (5, -1.0)]);
    }

    #[test]
    fn loop_over_grid_points_works() {
        let (nx, ny) = (2, 3);
        let grid = Grid2d::new_uniform(-1.0, 1.0, -3.0, 3.0, nx, ny).unwrap();
        let ebcs = EssentialBcs2d::new();
        let lap = Fdm2d::new(grid, ebcs, 1.0, 1.0).unwrap();
        let mut xx = Matrix::new(ny, nx);
        let mut yy = Matrix::new(ny, nx);
        lap.for_each_coord(|m, x, y| {
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
}
