use crate::{EquationHandler, EssentialBcs2d, Grid2d, NaturalBcs2d, Side, StrError};
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

/// Implements the Finite Difference Method (FDM) for 2D problems
///
/// This solver handles elliptic partial differential equations in two dimensions,
/// specifically the Poisson and Helmholtz equations with mixed boundary conditions.
///
/// # Problem Formulation
///
/// The FDM solves the following equation:
///
/// ```text
///     ∂²ϕ      ∂²ϕ
/// -kx ——— - ky ——— + α ϕ = source(x, y)
///     ∂x²      ∂y²
/// ```
///
/// where:
/// * `kx`, `ky` are the diffusion coefficients in x and y directions
/// * `α` is the Helmholtz coefficient (α = 0 for Poisson equation)
/// * `source(x, y)` is the source term
/// * `ϕ(x, y)` is the unknown solution
///
/// # Boundary Conditions
///
/// The solver supports:
/// * **Essential (Dirichlet)**: Prescribed values at boundaries
/// * **Natural (Neumann)**: Prescribed flux at boundaries
/// * **Periodic**: Solution wraps around domain boundaries (in x, y, or both)
///
/// # Discretization
///
/// The method uses central differences on a uniform 2D grid, resulting in a discrete system:
///
/// ```text
/// K a = f
/// ```
///
/// where `K` is the coefficient matrix, `a` is the solution vector, and `f` is the right-hand side.
///
/// ## Grid Indexing
///
/// The 2D grid values ϕᵢⱼ are mapped to a 1D solution vector using:
///
/// ```text
/// ϕᵢⱼ → aₘ   with   m = i + j nx
/// ```
///
/// where `i` is the x-index, `j` is the y-index, and `nx` is the number of grid points along x.
///
/// ## FDM Stencil
///
/// The method uses the "molecule" {α, β, β, γ, γ} for interior nodes:
///
/// ```text
/// α ϕ_cur + β ϕ_lef + β ϕ_rig + γ ϕ_bot + γ ϕ_top = s_cur
/// ```
///
/// where:
/// * `α = 2(kx/Δx² + ky/Δy²)`
/// * `β = -kx/Δx²`
/// * `γ = -ky/Δy²`
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
/// ## Left side (Xmin)
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
/// Let us divide the above expression by two so that the symmetry of the coefficient matrix is preserved:
///
/// ```text
/// ½ α ϕ_{cur} + β ϕ_{rig} + ½ γ ϕ_{bot} + ½ γ ϕ_{top} = ½ s_{cur} - wₙ / Δx
/// ```
///
/// ## Right side (Xmax)
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
/// Let us divide the above expression by two so that the symmetry of the coefficient matrix is preserved:
///
/// ```text
/// ½ α ϕ_{cur} + β ϕ_{lef} + ½ γ ϕ_{bot} + ½ γ ϕ_{top} = ½ s_{cur} - wₙ / Δx
/// ```
///
/// Similar expressions can be derived for the bottom (Ymin) and top (Ymax) sides.
///
/// ## Corners
///
/// At a corner (e.g., Xmin and Ymin):
///
/// ```text
/// α ϕ_{cur} + β ϕ_{ghostX} + β ϕ_{rig} + γ ϕ_{ghostY} + γ ϕ_{top} = s_{cur}
/// ```
///
/// By considering `ϕ_{ghostX} = ϕ_{rig} - 2 gx Δx` and `ϕ_{ghostY} = ϕ_{top} - 2 gy Δy`:
///
/// ```text
/// α ϕ_{cur} + β ϕ_{rig} + β ϕ_{rig} + γ ϕ_{top} + γ ϕ_{top} = s_{cur} + 2 β gx Δx + 2 γ gy Δy
/// α ϕ_{cur} + 2 β ϕ_{rig} + 2 γ ϕ_{top} = s_{cur} - 2 (kx/Δx²) gx Δx - 2 (ky/Δy²) gy Δy
/// α ϕ_{cur} + 2 β ϕ_{rig} + 2 γ ϕ_{top} = s_{cur} - 2 (kx gx) / Δx - 2 (ky gy) / Δy
/// α ϕ_{cur} + 2 β ϕ_{rig} + 2 γ ϕ_{top} = s_{cur} - 2 wₙL / Δx - 2 wₙB / Δy
/// ```
///
/// Let us divide the above expression by four so that the symmetry of the coefficient matrix is preserved:
///
/// ```text
/// ¼ α ϕ_{cur} + ½ β ϕ_{rig} + ½ γ ϕ_{top} = ¼ s_{cur} - ½ wₙL / Δx - ½ wₙB / Δy
/// ```
///
/// Similar expressions can be derived for the other corners.
///
/// # Examples
///
/// Solves the Poisson equation in 2D:
///
/// ```text
///   ∂²ϕ   ∂²ϕ
/// - ——— - ——— = 0    on  x ∈ [0, 1] × [0, 1]
///   ∂x²   ∂y²
///
/// ϕ(0, y) = 0
/// ϕ(1, y) = 0
/// ϕ(x, 0) = sin(πx)
/// ϕ(x, 1) = sin(πx) exp(π)
/// ```
///
/// The analytical solution is `ϕ(x, y) = sin(πx) exp(πy)`.
///
/// ```
/// use russell_lab::approx_eq;
/// use russell_lab::math::PI;
/// use russell_pde::{EssentialBcs2d, Fdm2d, Grid2d, NaturalBcs2d, Side, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // grid
///     let (xmin, xmax) = (0.0, 1.0);
///     let (ymin, ymax) = (0.0, 1.0);
///     let (nx, ny) = (8, 8);
///     let grid = Grid2d::new_uniform(xmin, xmax, ymin, ymax, nx, ny)?;
///
///     // Essential BCs
///     let mut ebcs = EssentialBcs2d::new();
///     ebcs.set(Side::Xmin, |_, _| 0.0);
///     ebcs.set(Side::Xmax, |_, _| 0.0);
///     ebcs.set(Side::Ymin, |x, _| f64::sin(PI * x));
///     ebcs.set(Side::Ymax, |x, _| f64::sin(PI * x) * f64::exp(PI));
///
///     // Natural BCs (none)
///     let nbcs = NaturalBcs2d::new();
///
///     // FDM solver
///     let (kx, ky) = (1.0, 1.0);
///     let fdm = Fdm2d::new(grid, ebcs, nbcs, kx, ky)?;
///
///     // Solve system
///     let alpha = 0.0; // 0 => Poisson; otherwise Helmholtz
///     let source = |_, _| 0.0; // no source term
///     let a = fdm.solve_sps(alpha, source)?;
///
///     // check
///     fdm.for_each_coord(|m, x, y| {
///         let analytical = f64::sin(PI * x) * f64::exp(PI * y);
///         approx_eq(a[m], analytical, 1.32e-1);
///     });
///     Ok(())
/// }
/// ```
pub struct Fdm2d<'a> {
    /// The 2D computational grid
    ///
    /// Must be a uniform grid with equally-spaced points in both x and y directions.
    grid: Grid2d,

    /// Essential (Dirichlet) boundary conditions handler
    ///
    /// Manages prescribed solution values at domain boundaries.
    ebcs: EssentialBcs2d<'a>,

    /// Natural (Neumann) boundary conditions handler
    ///
    /// Manages prescribed flux values at domain boundaries.
    nbcs: NaturalBcs2d<'a>,

    /// Equation numbering handler
    ///
    /// Manages the mapping between grid nodes and equation indices,
    /// distinguishing between unknown and prescribed degrees of freedom.
    equations: EquationHandler,

    /// FDM stencil coefficients: [α, β, β, γ, γ]
    ///
    /// Corresponds to current node, left, right, bottom, and top neighbors:
    /// * `molecule[CUR]` = α = 2(kx/Δx² + ky/Δy²)
    /// * `molecule[LEF]` = β = -kx/Δx²
    /// * `molecule[RIG]` = β = -kx/Δx²
    /// * `molecule[BOT]` = γ = -ky/Δy²
    /// * `molecule[TOP]` = γ = -ky/Δy²
    molecule: Vec<f64>,

    /// Grid spacing in x-direction (Δx)
    ///
    /// Distance between consecutive grid points along x (uniform grid).
    dx: f64,

    /// Grid spacing in y-direction (Δy)
    ///
    /// Distance between consecutive grid points along y (uniform grid).
    dy: f64,

    /// Sparse linear solver type
    ///
    /// Determines which solver backend to use (e.g., UMFPACK, MUMPS).
    /// Default: `Genie::Umfpack`
    genie: Genie,

    /// Use symmetric matrix storage
    ///
    /// Default: `true`
    symmetric: bool,
}

impl<'a> Fdm2d<'a> {
    /// Creates a new FDM solver instance for 2D problems
    ///
    /// # Input
    ///
    /// * `grid` - The 2D computational grid (must be uniform in both directions)
    /// * `ebcs` - Essential (Dirichlet) boundary conditions handler
    /// * `nbcs` - Natural (Neumann) boundary conditions handler
    /// * `kx` - Diffusion coefficient along x
    /// * `ky` - Diffusion coefficient along y
    ///
    /// # Returns
    ///
    /// * `Ok(Fdm2d)` - Successfully initialized solver
    /// * `Err` - If the grid is not uniform
    ///
    /// # Notes
    ///
    /// * The grid must have uniform spacing in both x and y directions
    /// * Boundary conditions are validated when solving, not during construction
    /// * Default solver: UMFPACK with symmetric matrices
    /// * Call [EssentialBcs2d::validate()] before solving to check boundary condition consistency
    pub fn new(
        grid: Grid2d,
        ebcs: EssentialBcs2d<'a>,
        nbcs: NaturalBcs2d<'a>,
        kx: f64,
        ky: f64,
    ) -> Result<Self, StrError> {
        // check grid
        let (dx, dy) = match grid.get_dx_dy() {
            Some((dx, dy)) => (dx, dy),
            None => return Err("grid must have uniform spacing"),
        };

        // allocate equations handler
        let neq = grid.size();
        let mut equations = EquationHandler::new(neq);
        equations.recompute(&ebcs.get_nodes(&grid));

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
            nbcs,
            equations,
            molecule: vec![alpha, beta, beta, gamma, gamma],
            dx,
            dy,
            genie: Genie::Umfpack,
            symmetric: true,
        })
    }

    /// Configures the sparse linear solver options
    ///
    /// # Input
    ///
    /// * `genie` - Sparse solver type (e.g., `Genie::Umfpack`, `Genie::Mumps`)
    /// * `symmetric` - Whether to exploit matrix symmetry
    ///   * `true`: Store only lower/upper triangle (saves memory)
    ///   * `false`: Store full matrix
    pub fn set_solver_options(&mut self, genie: Genie, symmetric: bool) {
        self.genie = genie;
        self.symmetric = symmetric;
    }

    /// Solves the Poisson or Helmholtz equation using System Partitioning Strategy (SPS)
    ///
    /// This method partitions the system into unknown and prescribed degrees of freedom,
    /// solving only for the unknowns while enforcing essential boundary conditions.
    ///
    /// # Input
    ///
    /// * `alpha` - Helmholtz coefficient (α)
    ///   * Set to 0.0 for Poisson equation
    /// * `source` - Source term function `f(x, y) -> value`
    ///
    /// # Returns
    ///
    /// * `Ok(Vector)` - Solution vector `ϕ` at all grid points (stored sequentially)
    /// * `Err` - If boundary conditions are inconsistent or solver fails
    ///
    /// # Equation
    ///
    /// Solves:
    ///
    /// ```text
    ///     ∂²ϕ      ∂²ϕ
    /// -kx ——— - ky ——— + α ϕ = source(x, y)
    ///     ∂x²      ∂y²
    /// ```
    ///
    /// # Notes
    ///
    /// * Automatically validates boundary conditions before solving
    /// * More efficient than LMM for problems with many essential BCs
    /// * Returns solution at all grid points (both unknown and prescribed)
    /// * Solution is stored sequentially: index m = i + j*nx
    pub fn solve_sps<F>(&self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64, f64) -> f64,
    {
        // validates the boundary conditions data
        self.ebcs.validate(&self.nbcs)?;

        // assemble the coefficient matrix and the lhs and rhs vectors
        let sym_kk_bar = self.genie.get_sym(self.symmetric);
        let (kk_bar, kk_check) = self.get_matrices_sps(alpha, 0, sym_kk_bar);
        let (mut a_bar, a_check, mut f_bar) = self.get_vectors_sps(source);
        let kk_check = kk_check.unwrap();

        // initialize the right-hand side
        kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check).unwrap(); // f̄ -= Ǩ ǎ

        // solve the linear system
        let mut solver = LinSolver::new(self.genie)?;
        solver.actual.factorize(&kk_bar, None)?;
        solver.actual.solve(&mut a_bar, &f_bar, false)?;

        // results
        Ok(self.get_joined_vector_sps(&a_bar, &a_check))
    }

    /// Solves the Poisson or Helmholtz equation using Lagrange Multipliers Method (LMM)
    ///
    /// This method enforces essential boundary conditions through Lagrange multipliers,
    /// resulting in a larger but simpler system structure.
    ///
    /// # Input
    ///
    /// * `alpha` - Helmholtz coefficient (α)
    ///   * Set to 0.0 for Poisson equation
    /// * `source` - Source term function `f(x, y) -> value`
    ///
    /// # Returns
    ///
    /// * `Ok(Vector)` - Solution vector `ϕ` at all grid points (stored sequentially)
    /// * `Err` - If boundary conditions are inconsistent or solver fails
    ///
    /// # Equation
    ///
    /// Solves:
    ///
    /// ```text
    ///     ∂²ϕ      ∂²ϕ
    /// -kx ——— - ky ——— + α ϕ = source(x, y)
    ///     ∂x²      ∂y²
    /// ```
    ///
    /// # Notes
    ///
    /// * Automatically validates boundary conditions before solving
    /// * Uses augmented system with Lagrange multipliers for essential BCs
    /// * More flexible than SPS but creates larger system
    /// * Suitable when essential BCs need to be modified frequently
    /// * Solution is stored sequentially: index m = i + j*nx
    pub fn solve_lmm<F>(&self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64, f64) -> f64,
    {
        // validates the boundary conditions data
        self.ebcs.validate(&self.nbcs)?;

        // assemble the coefficient matrix and the lhs and rhs vectors
        let sym_mm = self.genie.get_sym(self.symmetric);
        let (mm, _) = self.get_matrices_lmm(alpha, 0, false, sym_mm);
        let (mut aa, ff) = self.get_vectors_lmm(source);

        // solve the linear system
        let mut solver = LinSolver::new(self.genie)?;
        solver.actual.factorize(&mm, None)?;
        solver.actual.solve(&mut aa, &ff, false)?;

        // results
        let neq = self.equations.neq();
        Ok(Vector::from(&&aa.as_data()[..neq]))
    }

    /// Returns the system dimensions for the System Partitioning Strategy (SPS)
    ///
    /// # Returns
    ///
    /// A tuple `(nu, np)` where:
    /// * `nu` - Number of unknown degrees of freedom (to be solved for)
    /// * `np` - Number of prescribed degrees of freedom (essential BCs)
    ///
    /// # Notes
    ///
    /// The total number of equations is `nu + np`, which equals nx*ny (total grid points).
    pub fn get_dims_sps(&self) -> (usize, usize) {
        let nu = self.equations.nu();
        let np = self.equations.np();
        (nu, np)
    }

    /// Returns the system dimensions for the Lagrange Multipliers Method (LMM)
    ///
    /// # Returns
    ///
    /// A tuple `(neq, nlag, ndim)` where:
    /// * `neq` - Number of equations (= unknown DOFs + prescribed DOFs = nx*ny)
    /// * `nlag` - Number of Lagrange multipliers (= prescribed DOFs = essential BCs)
    /// * `ndim` - Total system dimension (= neq + nlag)
    ///
    /// # Notes
    ///
    /// The augmented system has dimension `ndim = neq + nlag`, which is larger than
    /// the original problem size due to the Lagrange multipliers.
    pub fn get_dims_lmm(&self) -> (usize, usize, usize) {
        let neq = self.equations.neq();
        let nlag = self.equations.np();
        let ndim = neq + nlag;
        (neq, nlag, ndim)
    }

    /// Returns a reference to the computational grid
    ///
    /// # Returns
    ///
    /// Reference to the 2D grid used by the solver.
    pub fn get_grid(&self) -> &Grid2d {
        &self.grid
    }

    /// Returns a reference to the equation numbering handler
    ///
    /// # Returns
    ///
    /// Reference to the [`EquationHandler`] that manages the mapping between
    /// grid nodes and equation indices.
    pub fn get_equations(&self) -> &EquationHandler {
        &self.equations
    }

    /// Assembles the coefficient matrices for the System Partitioning Strategy (SPS)
    ///
    /// Returns `(kk_bar, kk_check)` corresponding to the partitioned system:
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
    /// # Input
    ///
    /// * `alpha` - Helmholtz coefficient (α); set to 0.0 for Poisson equation
    /// * `extra_nnz` - Additional non-zero entries to allocate in K̄ matrix
    /// * `sym_kk_bar` - Symmetry type for the K̄ matrix (e.g., `Sym::YesLower`)
    ///
    /// # Returns
    ///
    /// * `kk_bar` - K̄ matrix (unknown-unknown block)
    /// * `kk_check` - Ǩ matrix (unknown-prescribed block), or `None` if no essential BCs
    ///
    /// # Notes
    ///
    /// * The Ǩ matrix is only created when there are essential boundary conditions (np > 0)
    /// * The 2D stencil has up to 5 neighbors per node
    pub fn get_matrices_sps(&self, alpha: f64, extra_nnz: usize, sym_kk_bar: Sym) -> (CooMatrix, Option<CooMatrix>) {
        let nx = self.grid.nx();
        let ny = self.grid.ny();
        let nu = self.equations.nu();
        let np = self.equations.np();
        let band = if sym_kk_bar.triangular() { 3 } else { 5 };
        let nnz_kk_bar = band * nu + extra_nnz;
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
                let mut val = self.molecule[b];
                if m == n {
                    val += alpha;
                }
                let (i, j) = self.grid.get_ij(m);
                if !self.ebcs.periodic_along_x && (i == 0 || i == nx - 1) {
                    val /= 2.0;
                }
                if !self.ebcs.periodic_along_y && (j == 0 || j == ny - 1) {
                    val /= 2.0;
                }
                if self.equations.is_prescribed(n) {
                    let jp = self.equations.ip(n);
                    kk_check.put(iu, jp, val).unwrap();
                } else {
                    let skip = (sym_kk_bar == Sym::YesLower && m < n) || (sym_kk_bar == Sym::YesUpper && m > n);
                    if !skip {
                        let ju = self.equations.iu(n);
                        kk_bar.put(iu, ju, val).unwrap();
                    }
                };
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
    /// * `alpha` -- Helmholtz coefficient (α). Set to 0.0 for the Poisson equation
    /// * `extra_nnz` -- extra non-zeros to allocate in the A matrix
    /// * `get_constraints_mat` -- whether to return the constraints matrix or not
    /// * `sym_mm` -- symmetry of the M matrix
    ///
    /// Note: this matrix is not symmetric because of the flipping (mirroring) strategy for boundary nodes.
    pub fn get_matrices_lmm(
        &self,
        alpha: f64,
        extra_nnz: usize,
        get_constraints_mat: bool,
        sym_mm: Sym,
    ) -> (CooMatrix, Option<CooMatrix>) {
        // build the augmented matrix
        let nx = self.grid.nx();
        let ny = self.grid.ny();
        let (neq, nlag, ndim) = self.get_dims_lmm();
        let band = if sym_mm.triangular() { 3 } else { 5 };
        let nnz = band * neq + 2 * nlag + extra_nnz; // 2*nlag is for C and Cᵀ
        let mut mm = CooMatrix::new(ndim, ndim, nnz, sym_mm).unwrap();
        for m in 0..neq {
            self.loop_over_bandwidth(m, |b, n| {
                if (sym_mm == Sym::YesLower && m < n) || (sym_mm == Sym::YesUpper && m > n) {
                    return;
                }
                let mut val = self.molecule[b];
                if m == n {
                    val += alpha;
                }
                let (i, j) = self.grid.get_ij(m);
                if !self.ebcs.periodic_along_x && (i == 0 || i == nx - 1) {
                    val /= 2.0;
                }
                if !self.ebcs.periodic_along_y && (j == 0 || j == ny - 1) {
                    val /= 2.0;
                }
                mm.put(m, n, val).unwrap();
            });
        }

        // assemble C and Cᵀ into M
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            match sym_mm {
                Sym::YesLower => {
                    mm.put(neq + ip, m, 1.0).unwrap(); // C
                }
                Sym::YesUpper => {
                    mm.put(m, neq + ip, 1.0).unwrap(); // Cᵀ
                }
                Sym::YesFull | Sym::No => {
                    mm.put(neq + ip, m, 1.0).unwrap(); // C
                    mm.put(m, neq + ip, 1.0).unwrap(); // Cᵀ
                }
            }
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
            let mut den = 1.0;
            let cf = if self.grid.is_corner(m) { 0.5 } else { 1.0 };
            if !self.ebcs.periodic_along_x {
                if self.grid.is_xmin(m) {
                    let wn = self.nbcs.functions[0](x, y);
                    f_bar[iu] += -cf * wn / self.dx;
                    den *= 2.0;
                } else if self.grid.is_xmax(m) {
                    let wn = self.nbcs.functions[1](x, y);
                    f_bar[iu] += -cf * wn / self.dx;
                    den *= 2.0;
                }
            }
            if !self.ebcs.periodic_along_y {
                if self.grid.is_ymin(m) {
                    let wn = self.nbcs.functions[2](x, y);
                    f_bar[iu] += -cf * wn / self.dy;
                    den *= 2.0;
                } else if self.grid.is_ymax(m) {
                    let wn = self.nbcs.functions[3](x, y);
                    f_bar[iu] += -cf * wn / self.dy;
                    den *= 2.0;
                }
            }
            f_bar[iu] += source(x, y) / den;
        });
        for index in 0..4 {
            if self.ebcs.sides[index] {
                for &m in self.grid.get_nodes_on_side(Side::from_index(index)) {
                    let ip = self.equations.ip(m);
                    let (x, y) = self.grid.coord(m);
                    let val = self.ebcs.functions[index](x, y);
                    a_check[ip] = val;
                }
            }
        }
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
            let mut den = 1.0;
            let cf = if self.grid.is_corner(m) { 0.5 } else { 1.0 };
            if !self.ebcs.periodic_along_x {
                if self.grid.is_xmin(m) {
                    let wn = self.nbcs.functions[0](x, y);
                    ff[m] += -cf * wn / self.dx;
                    den *= 2.0;
                }
                if self.grid.is_xmax(m) {
                    let wn = self.nbcs.functions[1](x, y);
                    ff[m] += -cf * wn / self.dx;
                    den *= 2.0;
                }
            }
            if !self.ebcs.periodic_along_y {
                if self.grid.is_ymin(m) {
                    let wn = self.nbcs.functions[2](x, y);
                    ff[m] += -cf * wn / self.dy;
                    den *= 2.0;
                }
                if self.grid.is_ymax(m) {
                    let wn = self.nbcs.functions[3](x, y);
                    ff[m] += -cf * wn / self.dy;
                    den *= 2.0;
                }
            }
            ff[m] += source(x, y) / den;
        });
        for index in 0..4 {
            if self.ebcs.sides[index] {
                for &m in self.grid.get_nodes_on_side(Side::from_index(index)) {
                    let ip = self.equations.ip(m);
                    let (x, y) = self.grid.coord(m);
                    let val = self.ebcs.functions[index](x, y);
                    ff[neq + ip] = val;
                }
            }
        }
        (aa, ff)
    }

    /// Executes a loop over the molecule values at row `m` of the coefficient matrix
    ///
    /// **Note**: The ghost boundary indices are flipped to avoid negative indices.
    /// Therefore, some column indices may appear repeated.
    ///
    /// # Input
    ///
    /// * `m` -- the row of the coefficient matrix
    /// * `callback` -- a `function(n, val_mn)` where `n` is the column index and
    ///   `val_mn` is the molecule value at row `m` and column `n`.
    pub fn loop_over_molecule<F>(&self, m: usize, mut callback: F)
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
        if self.ebcs.periodic_along_x {
            nn[LEF] = if i != INI_X { m - 1 } else { m + fin_x };
            nn[RIG] = if i != fin_x { m + 1 } else { m - fin_x };
        } else {
            nn[LEF] = if i != INI_X { m - 1 } else { m + 1 };
            nn[RIG] = if i != fin_x { m + 1 } else { m - 1 };
        }
        if self.ebcs.periodic_along_y {
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
    use crate::{EssentialBcs2d, Grid2d, NaturalBcs2d, Side};
    use russell_lab::Matrix;
    use russell_sparse::Sym;

    fn assert_symmetric(mat: &Matrix) {
        let (nrow, ncol) = mat.dims();
        assert_eq!(nrow, ncol);
        for i in 0..nrow {
            for j in (i + 1)..ncol {
                assert_eq!(mat.get(i, j), mat.get(j, i));
            }
        }
    }

    #[test]
    fn new_captures_errors() {
        let grid = Grid2d::new(&[0.0, 0.1, 0.4], &[0.0, 0.2, 0.5]).unwrap();
        let ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        let fdm = Fdm2d::new(grid, ebcs, nbcs, 1.0, 1.0);
        assert_eq!(fdm.err(), Some("grid must have uniform spacing"));
    }

    #[test]
    fn get_matrices_work() {
        //  8*  9  10  11
        //  4*  5   6   7
        //  0*  1   2   3
        // dx = 1.0, dy = 1.0
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 2.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();
        let mut nbcs = NaturalBcs2d::new();
        const LEF: f64 = 1.0;
        let lef = |_, _| LEF;
        assert_eq!(lef(0.0, 0.0), LEF);
        ebcs.set(Side::Xmin, lef); //  0  4  8
        nbcs.set(Side::Xmax, |_, _| 0.0); //  3  7 11
        nbcs.set(Side::Ymin, |_, _| 0.0); //  0  1  2  3
        nbcs.set(Side::Ymax, |_, _| 0.0); //  8  9 10 11

        let fdm = Fdm2d::new(grid, ebcs, nbcs, 100.0, 300.0).unwrap();
        assert_eq!(fdm.get_dims_sps(), (9, 3));
        assert_eq!(fdm.get_dims_lmm(), (12, 3, 15));

        fdm.loop_over_molecule(0, |n, val_mn| {
            if n == 0 {
                assert_eq!(val_mn, 800.0);
            } else if n == 1 {
                assert_eq!(val_mn, -100.0);
            } else if n == 4 {
                assert_eq!(val_mn, -300.0);
            }
        });

        // The full matrix is:
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11
        // ┌                                                             ┐
        // │  200  -50    .    . -150    .    .    .    .    .    .    . │  0*(p0)   corner
        // │  -50  400  -50    .    . -300    .    .    .    .    .    . │  1→0      B
        // │    .  -50  400  -50    .    . -300    .    .    .    .    . │  2→1      B
        // │    .    .  -50  200    .    .    . -150    .    .    .    . │  3→2      corner
        // │ -150    .    .    .  400 -100    .    . -150    .    .    . │  4*(p1)   L
        // │    . -300    .    . -100  800 -100    .    . -300    .    . │  5→3
        // │    .    . -300    .    . -100  800 -100    .    . -300    . │  6→4
        // │    .    .    . -150    .    . -100  400    .    .    . -150 │  7→5      R
        // │    .    .    .    . -150    .    .    .  200  -50    .    . │  8*(p3)   corner
        // │    .    .    .    .    . -300    .    .  -50  400  -50    . │  9→6      T
        // │    .    .    .    .    .    . -300    .    .  -50  400  -50 │ 10→7      T
        // │    .    .    .    .    .    .    . -150    .    .  -50  200 │ 11→8      corner
        // └                                                             ┘
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11

        for sym_kk_bar in [Sym::No, Sym::YesLower, Sym::YesUpper, Sym::YesFull] {
            let (kk_bar, kk_check) = fdm.get_matrices_sps(0.0, 0, sym_kk_bar);
            let kk_check = kk_check.unwrap();
            let kk_bar_dense = kk_bar.as_dense();
            assert_symmetric(&kk_bar_dense);
            assert_eq!(
                format!("{}", kk_bar_dense),
                "┌                                              ┐\n\
                 │  400  -50    0 -300    0    0    0    0    0 │\n\
                 │  -50  400  -50    0 -300    0    0    0    0 │\n\
                 │    0  -50  200    0    0 -150    0    0    0 │\n\
                 │ -300    0    0  800 -100    0 -300    0    0 │\n\
                 │    0 -300    0 -100  800 -100    0 -300    0 │\n\
                 │    0    0 -150    0 -100  400    0    0 -150 │\n\
                 │    0    0    0 -300    0    0  400  -50    0 │\n\
                 │    0    0    0    0 -300    0  -50  400  -50 │\n\
                 │    0    0    0    0    0 -150    0  -50  200 │\n\
                 └                                              ┘"
            );
            assert_eq!(
                format!("{}", kk_check.as_dense()),
                "┌                ┐\n\
                 │  -50    0    0 │\n\
                 │    0    0    0 │\n\
                 │    0    0    0 │\n\
                 │    0 -100    0 │\n\
                 │    0    0    0 │\n\
                 │    0    0    0 │\n\
                 │    0    0  -50 │\n\
                 │    0    0    0 │\n\
                 │    0    0    0 │\n\
                 └                ┘"
            );
        }

        for sym_mm in [Sym::No, Sym::YesLower, Sym::YesUpper, Sym::YesFull] {
            let (mm, cc) = fdm.get_matrices_lmm(0.0, 0, true, sym_mm);
            let cc = cc.unwrap();
            let mm_dense = mm.as_dense();
            assert_symmetric(&mm_dense);
            assert_eq!(
                format!("{}", cc.as_dense()),
                "┌                         ┐\n\
                 │ 1 0 0 0 0 0 0 0 0 0 0 0 │\n\
                 │ 0 0 0 0 1 0 0 0 0 0 0 0 │\n\
                 │ 0 0 0 0 0 0 0 0 1 0 0 0 │\n\
                 └                         ┘"
            );
            assert_eq!(
                format!("{}", mm_dense),
                "┌                                                                            ┐\n\
                 │  200  -50    0    0 -150    0    0    0    0    0    0    0    1    0    0 │\n\
                 │  -50  400  -50    0    0 -300    0    0    0    0    0    0    0    0    0 │\n\
                 │    0  -50  400  -50    0    0 -300    0    0    0    0    0    0    0    0 │\n\
                 │    0    0  -50  200    0    0    0 -150    0    0    0    0    0    0    0 │\n\
                 │ -150    0    0    0  400 -100    0    0 -150    0    0    0    0    1    0 │\n\
                 │    0 -300    0    0 -100  800 -100    0    0 -300    0    0    0    0    0 │\n\
                 │    0    0 -300    0    0 -100  800 -100    0    0 -300    0    0    0    0 │\n\
                 │    0    0    0 -150    0    0 -100  400    0    0    0 -150    0    0    0 │\n\
                 │    0    0    0    0 -150    0    0    0  200  -50    0    0    0    0    1 │\n\
                 │    0    0    0    0    0 -300    0    0  -50  400  -50    0    0    0    0 │\n\
                 │    0    0    0    0    0    0 -300    0    0  -50  400  -50    0    0    0 │\n\
                 │    0    0    0    0    0    0    0 -150    0    0  -50  200    0    0    0 │\n\
                 │    1    0    0    0    0    0    0    0    0    0    0    0    0    0    0 │\n\
                 │    0    0    0    0    1    0    0    0    0    0    0    0    0    0    0 │\n\
                 │    0    0    0    0    0    0    0    0    1    0    0    0    0    0    0 │\n\
                 └                                                                            ┘"
            );
        }
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
        let nbcs = NaturalBcs2d::new();
        ebcs.set_periodic(true, true);

        let fdm = Fdm2d::new(grid, ebcs, nbcs, 1.0, 1.0).unwrap();
        let (kk, cc_mat) = fdm.get_matrices_sps(0.0, 0, Sym::No);
        let (aa, ee_mat) = fdm.get_matrices_lmm(0.0, 0, true, Sym::No);
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
        let nbcs = NaturalBcs2d::new();

        let fdm = Fdm2d::new(grid, ebcs, nbcs, 1.0, 1.0).unwrap();

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
                25.0,      // 0   corner
                50.0,      // 1   B
                50.0,      // 2   B
                25.0,      // 3   corner
                50.0,      // 4   L
                100.0,     // 5
                100.0,     // 6
                50.0,      // 7   R
                25.0,      // 8   corner
                50.0,      // 9   T
                50.0,      // 10  T
                25.0,      // 11  corner
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
    fn get_grid_and_get_equations_work() {
        let grid = Grid2d::new_uniform(0.0, 1.0, 0.0, 1.0, 3, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        ebcs.set_homogeneous();
        let fdm = Fdm2d::new(grid, ebcs, nbcs, 1.0, 1.0).unwrap();

        assert_eq!(fdm.get_grid().nx(), 3);
        assert_eq!(fdm.get_equations().neq(), 9);
    }
}
