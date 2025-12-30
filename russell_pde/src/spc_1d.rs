use crate::{EquationHandler, EssentialBcs1d, Grid1d, NaturalBcs1d, Side, StrError};
use russell_lab::{InterpLagrange, Vector};
use russell_sparse::{CooMatrix, Genie, LinSolver, Sym};

/// Implements the Spectral Collocation Method (SPC) for 1D problems
///
/// This solver handles elliptic partial differential equations in one dimension
/// using spectral collocation on Chebyshev-Gauss-Lobatto grids, providing
/// high-order accuracy for smooth solutions.
///
/// # Problem Formulation
///
/// The SPC solves the following equation:
///
/// ```text
///     ∂²ϕ
/// -kx ——— + α ϕ = source(x)
///     ∂x²
/// ```
///
/// where:
/// * `kx` is the diffusion coefficient
/// * `α` is the Helmholtz coefficient (α = 0 for Poisson equation)
/// * `source(x)` is the source term
/// * `ϕ(x)` is the unknown solution
///
/// # Boundary Conditions
///
/// The solver supports:
/// * **Essential (Dirichlet)**: Prescribed values at boundaries
/// * **Natural (Neumann)**: Prescribed flux at boundaries
/// * **Note**: Periodic boundary conditions are not supported for spectral collocation
///
/// # Discretization
///
/// The spectral collocation method approximates the Laplacian at grid points xᵢ using:
///
/// ```text
///              ∂²ϕ│
/// (∇²ϕ)ᵢ = -kx ———│   = ∑ⱼ mkx D⁽²⁾ᵢⱼ ϕⱼ
///              ∂x²│xᵢ
///
/// mkx = -kx
/// ```
///
/// where ϕᵢ are the discrete values of ϕ(x) at the nx grid points, and D⁽²⁾
/// is the second derivative matrix for the spectral collocation method.
///
/// The discrete Laplacian operator can be written as:
///
/// ```text
/// (∇²a)ₘ = ∑ₙ Kₘₙ aₙ
/// ```
///
/// # Solution Methods
///
/// Two methods are implemented to handle essential boundary conditions:
///
/// 1. **System Partitioning Strategy (SPS)**: Partitions unknowns and prescribed values
/// 2. **Lagrange Multipliers Method (LMM)**: Uses augmented system with multipliers
///
/// # Grid Properties
///
/// * Uses Chebyshev-Gauss-Lobatto points for optimal spectral accuracy
/// * Grid is clustered near boundaries for better resolution
/// * Maximum polynomial degree: 2048 (limited by interpolator)
///
/// # Examples
///
/// Solves the Poisson equation in 1D:
///
/// ```text
/// -d²ϕ/dx² = 1   on  x ∈ [0, 1]
///
/// ϕ(0) = 0
/// ϕ(1) = 0
/// ```
///
/// The analytical solution is `ϕ(x) = (x - x²) / 2`.
/// ```
/// use russell_lab::approx_eq;
/// use russell_pde::{EssentialBcs1d, NaturalBcs1d, Side, Spc1d, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // Essential BCs
///     let mut ebcs = EssentialBcs1d::new();
///     ebcs.set(Side::Xmin, |_| 0.0);
///     ebcs.set(Side::Xmax, |_| 0.0);
///
///     // Natural BCs (none)
///     let nbcs = NaturalBcs1d::new();
///
///     // SPC solver
///     let xmin = 0.0;
///     let xmax = 1.0;
///     let nx = 4;
///     let kx = 1.0;
///     let spc = Spc1d::new(xmin, xmax, nx, ebcs, nbcs, kx)?;
///
///     // Solve system
///     let alpha = 0.0; // Poisson
///     let source = |_| 1.0;
///     let phi = spc.solve_sps(alpha, source)?;
///
///     // Check
///     spc.for_each_coord(|m, x| {
///         let analytical = x * (1.0 - x) / 2.0;
///         approx_eq(phi[m], analytical, 1e-14);
///     });
///     Ok(())
/// }
/// ```
pub struct Spc1d<'a> {
    /// Minimum x-coordinate of the computational domain
    xmin: f64,

    /// Maximum x-coordinate of the computational domain
    xmax: f64,

    /// The 1D Chebyshev-Gauss-Lobatto grid
    ///
    /// Grid points are clustered near boundaries for better spectral accuracy.
    grid: Grid1d,

    /// Essential (Dirichlet) boundary conditions handler
    ///
    /// Manages prescribed solution values at domain boundaries.
    /// Periodic boundary conditions are not supported.
    ebcs: EssentialBcs1d<'a>,

    /// Natural (Neumann) boundary conditions handler
    ///
    /// Manages prescribed flux values at domain boundaries.
    nbcs: NaturalBcs1d<'a>,

    /// Negative of the diffusion coefficient along x
    ///
    /// Stored as mkx = -kx for direct use in Laplacian assembly.
    mkx: f64,

    /// Equation numbering handler
    ///
    /// Manages the mapping between grid nodes and equation indices,
    /// distinguishing between unknown and prescribed degrees of freedom.
    equations: EquationHandler,

    /// Lagrange polynomial interpolator
    ///
    /// Computes spectral derivative matrices D⁽¹⁾ and D⁽²⁾.
    interp: InterpLagrange,
}

impl<'a> Spc1d<'a> {
    /// Creates a new spectral collocation solver instance for 1D problems
    ///
    /// # Input
    ///
    /// * `xmin` - Minimum x-coordinate of the domain
    /// * `xmax` - Maximum x-coordinate of the domain
    /// * `nx` - Number of grid points (must be ≥ 2, max polynomial degree: 2048)
    /// * `ebcs` - Essential (Dirichlet) boundary conditions handler
    /// * `nbcs` - Natural (Neumann) boundary conditions handler
    /// * `kx` - Diffusion coefficient
    ///
    /// # Returns
    ///
    /// * `Ok(Spc1d)` - Successfully initialized solver
    /// * `Err` - If:
    ///   * `nx` < 2
    ///   * Polynomial degree > 2048
    ///   * Periodic boundary conditions are specified
    ///
    /// # Notes
    ///
    /// * Uses Chebyshev-Gauss-Lobatto grid points for optimal spectral accuracy
    /// * Grid is automatically generated based on nx
    /// * Boundary conditions are validated when solving, not during construction
    /// * Call [EssentialBcs1d::validate()] before solving to check boundary condition consistency
    /// * Periodic boundary conditions are not supported and will return an error
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ebcs: EssentialBcs1d<'a>,
        nbcs: NaturalBcs1d<'a>,
        kx: f64,
    ) -> Result<Self, StrError> {
        // check
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }

        // polynomial degree
        let nn = nx - 1;
        if nn > 2048 {
            return Err("the maximum allowed polynomial degree is 2048");
        }

        // allocate the Chebyshev-Gauss-Lobatto grid
        let grid = Grid1d::new_chebyshev_gauss_lobatto(nx).unwrap();

        // check that the EBCs are not periodic
        if ebcs.periodic_along_x {
            return Err("essential BCs cannot be periodic");
        }

        // allocate equations handler
        let neq = grid.nx();
        let mut equations = EquationHandler::new(neq);
        equations.recompute(&ebcs.get_nodes(&grid));

        // interpolators
        let mut interp_x = InterpLagrange::new(nn, None).unwrap();
        interp_x.calc_dd1_matrix();
        interp_x.calc_dd2_matrix();

        // done
        Ok(Spc1d {
            xmin,
            xmax,
            grid,
            ebcs,
            nbcs,
            mkx: -kx,
            equations,
            interp: interp_x,
        })
    }

    /// Solves the Poisson or Helmholtz equation using System Partitioning Strategy (SPS)
    ///
    /// This method partitions the system into unknown and prescribed degrees of freedom,
    /// solving only for the unknowns while enforcing essential boundary conditions.
    ///
    /// # Input
    ///
    /// * `alpha` - Helmholtz coefficient (α); set to 0.0 for Poisson equation
    /// * `source` - Source term function `f(x) -> value`
    ///
    /// # Returns
    ///
    /// * `Ok(Vector)` - Solution vector `ϕ` at all grid points
    /// * `Err` - If boundary conditions are inconsistent or solver fails
    ///
    /// # Equation
    ///
    /// Solves:
    ///
    /// ```text
    ///     ∂²ϕ
    /// -kx ——— + α ϕ = source(x)
    ///     ∂x²
    /// ```
    ///
    /// # Notes
    ///
    /// * Automatically validates boundary conditions before solving
    /// * Returns solution at all grid points (both unknown and prescribed)
    pub fn solve_sps<F>(&self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64) -> f64,
    {
        // validates the boundary conditions data
        self.ebcs.validate(&self.nbcs)?;

        // assemble the coefficient matrix and the lhs and rhs vectors
        let (kk_bar, kk_check) = self.get_matrices_sps(alpha, 0);
        let (mut a_bar, a_check, mut f_bar) = self.get_vectors_sps(source);

        // initialize the right-hand side
        kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check).unwrap(); // f̄ -= Ǩ ǎ

        // solve the linear system
        let mut solver = LinSolver::new(Genie::Umfpack)?;
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
    /// * `alpha` - Helmholtz coefficient (α); set to 0.0 for Poisson equation
    /// * `source` - Source term function `f(x) -> value`
    ///
    /// # Returns
    ///
    /// * `Ok(Vector)` - Solution vector `ϕ` at all grid points
    /// * `Err` - If boundary conditions are inconsistent or solver fails
    ///
    /// # Equation
    ///
    /// Solves:
    ///
    /// ```text
    ///     ∂²ϕ
    /// -kx ——— + α ϕ = source(x)
    ///     ∂x²
    /// ```
    ///
    /// # Notes
    ///
    /// * Automatically validates boundary conditions before solving
    /// * Returns solution at all grid points (both unknown and prescribed)
    pub fn solve_lmm<F>(&self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64) -> f64,
    {
        // validates the boundary conditions data
        self.ebcs.validate(&self.nbcs)?;

        // assemble the coefficient matrix and the lhs and rhs vectors
        let (mm, _) = self.get_matrices_lmm(alpha, 0, false);
        let (mut aa, ff) = self.get_vectors_lmm(source);

        // solve the linear system
        let mut solver = LinSolver::new(Genie::Umfpack)?;
        solver.actual.factorize(&mm, None)?;
        solver.actual.solve(&mut aa, &ff, false)?;

        // results
        let neq = self.equations.neq();
        Ok(Vector::from(&&aa.as_data()[..neq]))
    }

    /// Calculates the flow vectors at each grid point
    ///
    /// Computes the components of the flow vector field based on the solution.
    ///
    /// # Input
    ///
    /// * `a` - Solution vector (must have dimension equal to number of equations)
    ///
    /// # Returns
    ///
    /// * `Ok(wwx)` - Flow vector x-components (length = neq)
    /// * `Err` - If `a.dim()` does not match the number of equations
    ///
    /// # Flow Vector Definition
    ///
    /// The flow vector is defined by:
    ///
    /// ```text
    /// →         →
    /// w = - ḵ · ∇ϕ
    /// ```
    ///
    /// where `ḵ` is the diffusion coefficient and `∇ϕ` is the gradient of the solution.
    pub fn calculate_flow_vectors(&self, a: &Vector) -> Result<Vec<f64>, StrError> {
        let neq = self.equations.neq();
        if a.dim() != neq {
            return Err("a.dim() must equal the number of equations");
        }
        let d1r = self.interp.get_dd1().unwrap();
        let dr_dx = 2.0 / (self.xmax - self.xmin);
        let mut wwx = vec![0.0; neq];
        for m in 0..neq {
            let mut wx = 0.0;
            for n in 0..neq {
                wx += self.mkx * d1r.get(m, n) * dr_dx * a[n];
            }
            wwx[m] = wx;
        }
        Ok(wwx)
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
    /// The total number of equations is `nu + np`, which equals nx (total grid points).
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
    /// * `neq` - Number of equations (= unknown DOFs + prescribed DOFs = nx)
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
    /// │ K̄   Ǩ │ │ ̄a │   │ f̄ │
    /// │       │ │   │ = │   │
    /// │ Ḵ   ̰K │ │ ǎ │   │ f̌ │
    /// └       ┘ └   ┘   └   ┘
    ///     K       a       f
    /// ```
    ///
    /// # Input
    ///
    /// * `alpha` - Helmholtz coefficient (α); set to 0.0 for Poisson equation
    /// * `extra_nnz` - Additional non-zero entries to allocate in K̄ matrix
    ///
    /// # Returns
    ///
    /// * `kk_bar` - K̄ matrix (unknown-unknown block)
    /// * `kk_check` - Ǩ matrix (unknown-prescribed block)
    ///
    /// # Notes
    ///
    /// * Uses spectral derivative matrices D⁽²⁾ for high-order accuracy
    /// * Matrix is generally dense due to spectral method
    /// * Includes domain mapping scaling from `[-1,1]` to `[xmin,xmax]`
    pub fn get_matrices_sps(&self, alpha: f64, extra_nnz: usize) -> (CooMatrix, CooMatrix) {
        // allocate matrices
        let nu = self.equations.nu();
        let np = self.equations.np();
        let nx = self.grid.nx();
        let nnz_wcs = nx * nx; // worst-case scenario
        let mut kk_bar = CooMatrix::new(nu, nu, nnz_wcs + extra_nnz, Sym::No).unwrap();
        let mut kk_check = CooMatrix::new(nu, np, nnz_wcs, Sym::No).unwrap();

        // spectral derivative matrices
        let d1r = self.interp.get_dd1().unwrap();
        let d2r = self.interp.get_dd2().unwrap();

        // scaling coefficients due to domain mapping (from [-1,1] to [xmin,xmax])
        let dr_dx = 2.0 / (self.xmax - self.xmin);
        let cx = dr_dx * dr_dx;

        // add terms to the coefficient matrix
        for &m in self.equations.unknown() {
            if self.nbcs.enabled_m(m, &self.grid) {
                for n in 0..nx {
                    let mut val = 0.0;
                    if m == 0 {
                        // Xmin
                        val += -self.mkx * d1r.get(m, n) * dr_dx; // -1 due to the normal pointing left
                    }
                    if m == nx - 1 {
                        // Xmax
                        val += self.mkx * d1r.get(m, n) * dr_dx;
                    }
                    self.put_val(&mut kk_bar, &mut kk_check, m, n, val);
                }
            } else {
                for n in 0..nx {
                    let mut val = self.mkx * d2r.get(m, n) * cx;
                    if m == n {
                        val += alpha; // diagonal entries due to α ϕ
                    }
                    self.put_val(&mut kk_bar, &mut kk_check, m, n, val);
                }
            }
        }

        // done
        (kk_bar, kk_check)
    }

    /// Assembles the augmented matrix for the Lagrange Multipliers Method (LMM)
    ///
    /// Returns `(mm, cc)` corresponding to the augmented system:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K  Cᵀ │ │ a │   │ f │
    /// │       │ │   │ = │   │
    /// │ C  0  │ │ ℓ │   │ ǎ │
    /// └       ┘ └   ┘   └   ┘
    ///     M       A       F
    /// ```
    ///
    /// # Input
    ///
    /// * `alpha` - Helmholtz coefficient (α); set to 0.0 for Poisson equation
    /// * `extra_nnz` - Additional non-zero entries to allocate in the matrix
    /// * `get_constraints_mat` - Whether to return the constraints matrix C separately
    ///
    /// # Returns
    ///
    /// * `mm` - Augmented M matrix containing K, C, and Cᵀ blocks
    /// * `cc` - Constraints matrix C, or `None` if not requested
    ///
    /// # Notes
    ///
    /// * Uses spectral derivative matrices for high-order accuracy
    /// * The augmented system size is (neq + nlag) × (neq + nlag)
    /// * Matrix is generally dense due to spectral method
    /// * Includes domain mapping scaling from `[-1,1]` to `[xmin,xmax]`
    pub fn get_matrices_lmm(
        &self,
        alpha: f64,
        extra_nnz: usize,
        get_constraints_mat: bool,
    ) -> (CooMatrix, Option<CooMatrix>) {
        // allocate matrices
        let (neq, nlag, ndim) = self.get_dims_lmm();
        let nx = self.grid.nx();
        let nnz_wcs = nx * nx; // worst-case scenario
        let mut mm = CooMatrix::new(ndim, ndim, nnz_wcs + extra_nnz + 2 * nlag, Sym::No).unwrap();

        // spectral derivative matrices
        let d1r = self.interp.get_dd1().unwrap();
        let d2r = self.interp.get_dd2().unwrap();

        // scaling coefficients due to domain mapping (from [-1,1] to [xmin,xmax])
        let dr_dx = 2.0 / (self.xmax - self.xmin);
        let cx = dr_dx * dr_dx;

        // add terms to the coefficient matrix
        for m in 0..neq {
            if self.nbcs.enabled_m(m, &self.grid) {
                for n in 0..nx {
                    let mut val = 0.0;
                    if m == 0 {
                        // Xmin
                        val += -self.mkx * d1r.get(m, n) * dr_dx; // -1 due to the normal pointing left
                    }
                    if m == nx - 1 {
                        // Xmax
                        val += self.mkx * d1r.get(m, n) * dr_dx;
                    }
                    mm.put(m, n, val).unwrap();
                }
            } else {
                for n in 0..nx {
                    let mut val = self.mkx * d2r.get(m, n) * cx;
                    if m == n {
                        val += alpha; // diagonal entries due to α ϕ
                    }
                    mm.put(m, n, val).unwrap();
                }
            }
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

    /// Assembles the solution vectors for the System Partitioning Strategy (SPS)
    ///
    /// Returns `(a_bar, a_check, f_bar)` corresponding to:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K̄   Ǩ │ │ ̄a │   │ f̄ │
    /// │       │ │   │ = │   │
    /// │ Ḵ   ̰K │ │ ǎ │   │ f̌ │
    /// └       ┘ └   ┘   └   ┘
    ///     K       a       f
    /// ```
    ///
    /// # Input
    ///
    /// * `source` - Source term function `f(x) -> value`
    ///
    /// # Returns
    ///
    /// A tuple `(a_bar, a_check, f_bar)` where:
    /// * `a_bar` - Unknown solution components (initialized to zero)
    /// * `a_check` - Prescribed solution components (from essential BCs)
    /// * `f_bar` - Right-hand side vector for unknowns
    pub fn get_vectors_sps<F>(&self, source: F) -> (Vector, Vector, Vector)
    where
        F: Fn(f64) -> f64,
    {
        let nu = self.equations.nu();
        let np = self.equations.np();
        let a_bar = Vector::new(nu);
        let mut a_check = Vector::new(np);
        let mut f_bar = Vector::new(nu);
        self.equations.unknown().iter().for_each(|&m| {
            let iu = self.equations.iu(m);
            let r = self.grid.coord(m);
            let x = self.map_coord(r);
            if self.grid.on_boundary(m) {
                // In the SPC, on the Neumann boundary, we solve -k∂ϕ/∂n = q̄ which is different than the
                // FDM approach which still solves the original equation -k ∇²ϕ = source(x,y). Therefore,
                // we must NOT add the source term to f̄ in the SPC.
                if self.grid.is_xmin(m) {
                    let wn = self.nbcs.functions[0](x);
                    f_bar[iu] += wn;
                }
                if self.grid.is_xmax(m) {
                    let wn = self.nbcs.functions[1](x);
                    f_bar[iu] += wn;
                }
            } else {
                // Solving the original equation -k ∇²ϕ = source(x,y)
                f_bar[iu] = source(x);
            }
        });
        for index in 0..2 {
            if self.ebcs.sides[index] {
                for &m in self.grid.get_nodes_on_side(Side::from_index(index)) {
                    let ip = self.equations.ip(m);
                    let r = self.grid.coord(m);
                    let x = self.map_coord(r);
                    let val = self.ebcs.functions[index](x);
                    a_check[ip] = val;
                }
            }
        }
        (a_bar, a_check, f_bar)
    }

    /// Joins the partitioned solution vectors from the System Partitioning Strategy (SPS)
    ///
    /// Returns the complete solution vector `a` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K̄   Ǩ │ │ ̄a │   │ f̄ │
    /// │       │ │   │ = │   │
    /// │ Ḵ   ̰K │ │ ǎ │   │ f̌ │
    /// └       ┘ └   ┘   └   ┘
    ///     K       a       f
    /// ```
    ///
    /// # Input
    ///
    /// * `a_bar` - Unknown solution components (length = nu)
    /// * `a_check` - Prescribed solution components (length = np)
    ///
    /// # Returns
    ///
    /// Complete solution vector at all grid points (length = neq = nu + np)
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

    /// Assembles the solution vectors for the Lagrange Multipliers Method (LMM)
    ///
    /// Returns `(aa, ff)` corresponding to the augmented system:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K  Cᵀ │ │ a │   │ f │
    /// │       │ │   │ = │   │
    /// │ C  0  │ │ ℓ │   │ ǎ │
    /// └       ┘ └   ┘   └   ┘
    ///     M       A       F
    /// ```
    ///
    /// # Input
    ///
    /// * `source` - Source term function `f(x) -> value`
    ///
    /// # Returns
    ///
    /// A tuple `(aa, ff)` where:
    /// * `aa` - Augmented solution vector (length = ndim = neq + nlag, initialized to zero)
    /// * `ff` - Augmented right-hand side vector (length = ndim)
    pub fn get_vectors_lmm<F>(&self, source: F) -> (Vector, Vector)
    where
        F: Fn(f64) -> f64,
    {
        let (neq, _, ndim) = self.get_dims_lmm();
        let aa = Vector::new(ndim);
        let mut ff = Vector::new(ndim);
        self.grid.for_each_coord(|m, r| {
            let x = self.map_coord(r);
            if self.grid.on_boundary(m) {
                // In the SPC, on the Neumann boundary, we solve -k∂ϕ/∂n = q̄ which is different than the
                // FDM approach which still solves the original equation -k ∇²ϕ = source(x,y). Therefore,
                // we must NOT add the source term to f̄ in the SPC.
                if self.grid.is_xmin(m) {
                    let wn = self.nbcs.functions[0](x);
                    ff[m] += wn;
                }
                if self.grid.is_xmax(m) {
                    let wn = self.nbcs.functions[1](x);
                    ff[m] += wn;
                }
            } else {
                // Solving the original equation -k ∇²ϕ = source(x,y)
                ff[m] = source(x);
            }
        });
        for index in 0..2 {
            if self.ebcs.sides[index] {
                for &m in self.grid.get_nodes_on_side(Side::from_index(index)) {
                    let ip = self.equations.ip(m);
                    let r = self.grid.coord(m);
                    let x = self.map_coord(r);
                    let val = self.ebcs.functions[index](x);
                    ff[neq + ip] = val;
                }
            }
        }
        (aa, ff)
    }

    /// Executes a loop over the grid points
    ///
    /// # Input
    ///
    /// * `callback` -- a function of `(m, x)` where `m` is the sequential point number,
    ///   and `x` is the Cartesian coordinate of the grid point.
    pub fn for_each_coord<F>(&self, mut callback: F)
    where
        F: FnMut(usize, f64),
    {
        self.grid.for_each_coord(|m, r| {
            let x = self.map_coord(r);
            callback(m, x);
        });
    }

    /// Puts the value into the correct position of the coefficient matrix (SPS)
    fn put_val(&self, kk_bar: &mut CooMatrix, kk_check: &mut CooMatrix, m: usize, n: usize, val: f64) {
        // unknown row
        let row = self.equations.iu(m);
        if !self.equations.is_prescribed(n) {
            // unknown column
            let col = self.equations.iu(n);
            kk_bar.put(row, col, val).unwrap();
        } else {
            // prescribed column
            let col = self.equations.ip(n);
            kk_check.put(row, col, val).unwrap();
        }
    }

    /// Maps the reference coordinates r in [-1,1] to the physical coordinate x
    fn map_coord(&self, r: f64) -> f64 {
        (self.xmax + self.xmin + (self.xmax - self.xmin) * r) / 2.0
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Spc1d;
    use crate::{EssentialBcs1d, NaturalBcs1d, Side};
    use russell_lab::Vector;
    use russell_sparse::Sym;

    #[test]
    fn new_captures_errors() {
        let ebcs = EssentialBcs1d::new();
        let nbcs = NaturalBcs1d::new();
        assert_eq!(Spc1d::new(0.0, 1.0, 1, ebcs, nbcs, 1.0).err(), Some("nx must be ≥ 2"));

        let mut ebcs = EssentialBcs1d::new();
        let mut nbcs = NaturalBcs1d::new();
        ebcs.set(Side::Xmin, |_| 0.0);
        nbcs.set(Side::Xmax, |_| 0.0);
        assert_eq!(
            Spc1d::new(0.0, 1.0, 2050, ebcs, nbcs, 1.0).err(),
            Some("the maximum allowed polynomial degree is 2048")
        );

        let mut ebcs = EssentialBcs1d::new();
        let nbcs = NaturalBcs1d::new();
        ebcs.set_periodic(true);
        assert_eq!(
            Spc1d::new(0.0, 1.0, 3, ebcs, nbcs, 1.0).err(),
            Some("essential BCs cannot be periodic")
        );
    }

    #[test]
    fn calculate_flow_vectors_captures_errors() {
        let mut ebcs = EssentialBcs1d::new();
        let mut nbcs = NaturalBcs1d::new();
        ebcs.set(Side::Xmin, |_| 0.0);
        nbcs.set(Side::Xmax, |_| 0.0);
        let spc = Spc1d::new(0.0, 1.0, 2, ebcs, nbcs, 1.0).unwrap();
        let a = Vector::from(&[0.0]); // wrong size
        assert_eq!(
            spc.calculate_flow_vectors(&a.into()).err(),
            Some("a.dim() must equal the number of equations")
        );
    }

    #[test]
    fn get_dims_sps_and_get_equations_work() {
        let mut ebcs = EssentialBcs1d::new();
        let mut nbcs = NaturalBcs1d::new();
        ebcs.set(Side::Xmin, |_| 0.0);
        nbcs.set(Side::Xmax, |_| 0.0);
        let spc = Spc1d::new(0.0, 1.0, 3, ebcs, nbcs, 1.0).unwrap();

        assert_eq!(spc.get_dims_sps(), (2, 1));
        assert_eq!(spc.get_equations().neq(), 3);
        assert_eq!(spc.get_equations().nu(), 2);
        assert_eq!(spc.get_equations().np(), 1);
    }

    #[test]
    fn get_matrices_works() {
        let mut ebcs = EssentialBcs1d::new();
        let nbcs = NaturalBcs1d::new();
        ebcs.set_homogeneous();
        let spc = Spc1d::new(-1.0, 1.0, 5, ebcs, nbcs, -1.0).unwrap();

        let (nu, np) = (3, 2);
        assert_eq!(spc.get_dims_sps(), (nu, np));
        assert_eq!(spc.get_equations().nu(), nu);
        assert_eq!(spc.get_equations().np(), np);

        let (kk_bar, kk_check) = spc.get_matrices_sps(0.0, 0);
        assert_eq!(kk_bar.get_info(), (nu, nu, 9, Sym::No));
        assert_eq!(kk_check.get_info(), (nu, np, 6, Sym::No));

        let neq = nu + np;
        let nlag = np;
        let ndim = neq + nlag;
        assert_eq!(spc.get_dims_lmm(), (neq, nlag, ndim));

        let nnz = neq * neq + 2 * nlag;
        let (mm, cc) = spc.get_matrices_lmm(0.0, 0, true);
        assert_eq!(mm.get_info(), (ndim, ndim, nnz, Sym::No));
        let cc = cc.unwrap();
        assert_eq!(cc.get_info(), (nlag, neq, nlag, Sym::No));
    }
}
