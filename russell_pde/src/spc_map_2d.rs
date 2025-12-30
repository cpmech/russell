use crate::util::delta;
use crate::{EquationHandler, EssentialBcs2d, Grid2d, Metrics, NaturalBcs2d, Side, StrError, Transfinite2d};
use russell_lab::{vec_copy_scaled, vec_inner, vec_norm, InterpLagrange, Norm, Vector};
use russell_sparse::{CooMatrix, Genie, LinSolver, Sym};

/// Implements the Spectral Collocation Method (SPC) for 2D problems with curvilinear coordinates
///
/// This solver handles elliptic partial differential equations in two dimensions
/// on mapped domains using spectral collocation with Transfinite Mapping, providing
/// high-order accuracy for smooth solutions on complex geometries.
///
/// # Problem Formulation
///
/// The SPC solves the following equation on curvilinear coordinates:
///
/// ```text
/// -k ∇²ϕ + α ϕ = source(x,y)
/// ```
///
/// where the Laplacian in curvilinear coordinates is:
///
/// ```text
///       ∂²ϕ       ∂²ϕ        ∂²ϕ          ∂ϕ      ∂ϕ
/// ∇²ϕ = ——— g¹¹ + ——— g²² + ————— 2 g¹² - —— L¹ - —— L²
///       ∂r²       ∂s²       ∂r ∂s         ∂r      ∂r
///
/// L¹ = Γ¹₁₁ g¹¹ + Γ¹₂₂ g²² + 2 Γ¹₁₂ g¹²
/// L² = Γ²₁₁ g¹¹ + Γ²₂₂ g²² + 2 Γ²₁₂ g¹²
/// ```
///
/// where:
/// * `gᵢⱼ` are the contravariant metric coefficients
/// * `Γᵏᵢⱼ` are the Christoffel symbols of the second kind
/// * `k` is the diffusion coefficient
/// * `α` is the Helmholtz coefficient (α = 0 for Poisson equation)
/// * `source(x,y)` is the source term
/// * `ϕ(x,y)` is the unknown solution
///
/// # Coordinate Mapping
///
/// The reference coordinates (r, s) in [-1, 1] × [-1, 1] are mapped onto the physical
/// coordinates (x, y) using Transfinite Mapping, enabling solution of problems on
/// complex geometries with curved boundaries.
///
/// # Discretization
///
/// The spectral collocation method approximates the Laplacian at grid points using:
///
/// ```text
/// (∇²ϕ)ᵢⱼ = ∑ₖ ∑ₗ (
///       D⁽²⁾ᵢₖ δⱼₗ g¹¹
///     + δᵢₖ D̄⁽²⁾ⱼₗ g²²
///     + Dᵢₖ D̄ⱼₗ 2 g¹²
///     - Dᵢₖ δⱼₗ L¹
///     - δᵢₖ D̄ⱼₗ L²
/// ) ϕₖₗ
/// ```
///
/// where ϕᵢⱼ are the discrete values at grid points (rᵢ, sⱼ) mapped to (x, y).
///
/// ## Grid Indexing
///
/// The 2D grid values ϕᵢⱼ are mapped to a 1D solution vector using:
///
/// ```text
/// ϕᵢⱼ → aₘ   with   m = i + j nr
/// ```
///
/// where `i` is the r-index, `j` is the s-index, and `nr` is the number of grid points along r.
///
/// # Boundary Conditions
///
/// The solver supports:
/// * **Essential (Dirichlet)**: Prescribed values at boundaries (all sides must have essential BCs)
/// * **Natural (Neumann)**: Prescribed flux at boundaries (limited support)
/// * **Note**: Periodic boundary conditions are not supported
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
/// * Grid is clustered near boundaries in reference domain
/// * Maximum polynomial degree: 2048 (limited by interpolator)
///
/// # Examples
///
/// Solves the Poisson equation in 2D:
///
/// ```text
///   ∂²ϕ   ∂²ϕ
/// - ——— - ——— = 0    on  a rotated [0, 1] × [0, 1] domain by α
///   ∂x²   ∂y²
///
/// ϕ(0, y) = 0
/// ϕ(1, y) = 0
/// ϕ(x, 0) = sin(π x / cos(α))
/// ϕ(x, 1) = sin(π (x + sin(α)) / cos(α)) exp(π)
/// ```
///
/// The analytical solution is:
///
/// ```text
/// ϕ(x, y) = sin(π x cos(α) + π y sin(α)) * exp(π y cos(α) - π x sin(α))
/// ```
///
/// ```
/// use russell_lab::math::PI;
/// use russell_lab::{approx_eq, Vector};
/// use russell_pde::{EssentialBcs2d, NaturalBcs2d, Side, SpcMap2d, StrError, TransfiniteSamples};
///
/// fn main() -> Result<(), StrError> {
///     // Polynomial degree and tolerance for error checking
///     let nn = 8;
///     let tol = 1.0e-5;
///     let alpha = PI / 6.0;
///     let (ca, sa) = (f64::cos(alpha), f64::sin(alpha));
///
///     // Transfinite map
///     let xa = &[0.0, 0.0];
///     let xb = &[ca, sa];
///     let xc = &[ca - sa, ca + sa];
///     let xd = &[-sa, ca];
///     let mut map = TransfiniteSamples::quadrilateral_2d(xa, xb, xc, xd);
///
///     // Analytical solution
///     let analytical = move |x: f64, y: f64| f64::sin(PI * x * ca + PI * y * sa) * f64::exp(PI * y * ca - PI * x * sa);
///
///     // Essential boundary conditions
///     let mut ebcs = EssentialBcs2d::new();
///     ebcs.set(Side::Xmin, |_, _| 0.0);
///     ebcs.set(Side::Xmax, |_, _| 0.0);
///     ebcs.set(Side::Ymin, move |x, _| f64::sin(PI * x / ca));
///     ebcs.set(Side::Ymax, move |x, _| f64::sin(PI * (x + sa) / ca) * f64::exp(PI));
///
///     // Natural boundary conditions
///     let nbcs = NaturalBcs2d::new();
///
///     // Allocate the solver
///     let (nr, ns) = (nn + 1, nn + 1);
///     let k = 1.0;
///     let mut spc = SpcMap2d::new(map, nr, ns, ebcs, nbcs, k)?;
///
///     // Solve the problem
///     let a = spc.solve_sps(0.0, |_, _| 0.0)?;
///
///     // check
///     spc.for_each_coord(|m, x, y| {
///         approx_eq(a[m], analytical(x, y), tol);
///     });
///     Ok(())
/// }
/// ```
///
/// ![Results](data/figures/doc_example_spc_map.svg)
pub struct SpcMap2d<'a> {
    /// The 2D Chebyshev-Gauss-Lobatto grid on the reference domain [-1, 1] × [-1, 1]
    ///
    /// Grid points are clustered near boundaries in the reference domain for better spectral accuracy.
    grid: Grid2d,

    /// Essential (Dirichlet) boundary conditions handler
    ///
    /// Manages prescribed solution values at domain boundaries.
    /// All sides must have essential BCs for proper functioning.
    /// Periodic boundary conditions are not supported.
    ebcs: EssentialBcs2d<'a>,

    /// Natural (Neumann) boundary conditions handler
    ///
    /// Manages prescribed flux values at domain boundaries (limited support).
    nbcs: NaturalBcs2d<'a>,

    /// Negative of the diffusion coefficient
    ///
    /// Stored as mk = -k for direct use in Laplacian assembly.
    mk: f64,

    /// Equation numbering handler
    ///
    /// Manages the mapping between grid nodes and equation indices,
    /// distinguishing between unknown and prescribed degrees of freedom.
    equations: EquationHandler,

    /// Lagrange polynomial interpolator along r-direction (reference coordinate)
    ///
    /// Computes spectral derivative matrices D⁽¹⁾ and D⁽²⁾ for r.
    interp_r: InterpLagrange,

    /// Lagrange polynomial interpolator along s-direction (reference coordinate)
    ///
    /// Computes spectral derivative matrices D̄⁽¹⁾ and D̄⁽²⁾ for s.
    interp_s: InterpLagrange,

    /// Metrics calculator for curvilinear coordinates
    ///
    /// Computes contravariant metric coefficients (gᵢⱼ) and Christoffel symbols (Γᵏᵢⱼ)
    /// needed for the Laplacian in curvilinear coordinates.
    metrics: Metrics,

    /// Transfinite mapping from reference to physical domain
    ///
    /// Maps reference coordinates (r, s) ∈ [-1, 1] × [-1, 1] to physical coordinates (x, y).
    map: Transfinite2d,

    /// Temporary storage for physical coordinates (x, y)
    x: Vector,

    /// Temporary storage for derivative of x with respect to r
    dx_dr: Vector,

    /// Temporary storage for derivative of x with respect to s
    dx_ds: Vector,

    /// Temporary storage for second derivative of x with respect to r
    d2x_dr2: Vector,

    /// Temporary storage for second derivative of x with respect to s
    d2x_ds2: Vector,

    /// Temporary storage for mixed second derivative of x with respect to r and s
    d2x_drs: Vector,

    /// Temporary storage for unit normal vector at boundaries
    un: Vector,
}

impl<'a> SpcMap2d<'a> {
    /// Creates a new spectral collocation solver instance for 2D problems with curvilinear coordinates
    ///
    /// # Input
    ///
    /// * `map` - Transfinite mapping from reference to physical domain
    /// * `nr` - Number of grid points along r (must be ≥ 2, max polynomial degree: 2048)
    /// * `ns` - Number of grid points along s (must be ≥ 2, max polynomial degree: 2048)
    /// * `ebcs` - Essential (Dirichlet) boundary conditions handler
    /// * `nbcs` - Natural (Neumann) boundary conditions handler
    /// * `k` - Diffusion coefficient
    ///
    /// # Returns
    ///
    /// * `Ok(SpcMap2d)` - Successfully initialized solver
    /// * `Err` - If:
    ///   * `nr` or `ns` < 2
    ///   * Polynomial degree > 2048
    ///   * Periodic boundary conditions are specified
    ///
    /// # Notes
    ///
    /// * Uses Chebyshev-Gauss-Lobatto grid points in reference domain [-1,1] × [-1,1]
    /// * All sides must have essential boundary conditions
    /// * Natural (Neumann) boundary conditions have limited support
    /// * Grid is automatically generated based on nr and ns
    /// * Boundary conditions are validated when solving, not during construction
    /// * Call [EssentialBcs2d::validate()] before solving to check boundary condition consistency
    /// * Periodic boundary conditions are not supported and will return an error
    pub fn new(
        map: Transfinite2d,
        nr: usize,
        ns: usize,
        ebcs: EssentialBcs2d<'a>,
        nbcs: NaturalBcs2d<'a>,
        k: f64,
    ) -> Result<Self, StrError> {
        // check
        if nr < 2 {
            return Err("nr must be ≥ 2");
        }
        if ns < 2 {
            return Err("ns must be ≥ 2");
        }

        // polynomial degrees
        let nn_r = nr - 1;
        let nn_s = ns - 1;
        if nn_r > 2048 || nn_s > 2048 {
            return Err("the maximum allowed polynomial degree is 2048");
        }

        // allocate the Chebyshev-Gauss-Lobatto grid
        let grid = Grid2d::new_chebyshev_gauss_lobatto(nr, ns).unwrap();

        // check that the EBCs is not periodic
        if ebcs.periodic_along_x || ebcs.periodic_along_y {
            return Err("essential BCs cannot be periodic");
        }

        // allocate equations handler
        let neq = grid.size();
        let mut equations = EquationHandler::new(neq);
        equations.recompute(&ebcs.get_nodes(&grid));

        // interpolators
        let mut interp_r = InterpLagrange::new(nn_r, None).unwrap();
        let mut interp_s = InterpLagrange::new(nn_s, None).unwrap();
        interp_r.calc_dd1_matrix();
        interp_s.calc_dd1_matrix();
        interp_r.calc_dd2_matrix();
        interp_s.calc_dd2_matrix();

        // metrics
        let metrics = Metrics::new(2, false);

        // done
        Ok(SpcMap2d {
            grid,
            ebcs,
            nbcs,
            mk: -k,
            equations,
            interp_r,
            interp_s,
            metrics,
            map,
            x: Vector::new(2),
            dx_dr: Vector::new(2),
            dx_ds: Vector::new(2),
            d2x_dr2: Vector::new(2),
            d2x_ds2: Vector::new(2),
            d2x_drs: Vector::new(2),
            un: Vector::new(2),
        })
    }

    /// Solves the Poisson or Helmholtz equation using System Partitioning Strategy (SPS)
    ///
    /// This method partitions the system into unknown and prescribed degrees of freedom,
    /// solving only for the unknowns while enforcing essential boundary conditions on
    /// the curvilinear domain.
    ///
    /// # Input
    ///
    /// * `alpha` - Helmholtz coefficient (α); set to 0.0 for Poisson equation
    /// * `source` - Source term function `f(x, y) -> value` in physical coordinates
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
    /// -k ∇²ϕ + α ϕ = source(x,y)
    /// ```
    ///
    /// where the Laplacian includes curvilinear metric terms.
    ///
    /// # Notes
    ///
    /// * Automatically validates boundary conditions before solving
    /// * Returns solution at all grid points (both unknown and prescribed)
    /// * Solution is stored sequentially: index m = i + j*nr
    /// * Requires mutable self for metric calculations
    pub fn solve_sps<F>(&mut self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64, f64) -> f64,
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
    /// resulting in a larger but simpler system structure on the curvilinear domain.
    ///
    /// # Input
    ///
    /// * `alpha` - Helmholtz coefficient (α); set to 0.0 for Poisson equation
    /// * `source` - Source term function `f(x, y) -> value` in physical coordinates
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
    /// -k ∇²ϕ + α ϕ = source(x,y)
    /// ```
    ///
    /// where the Laplacian includes curvilinear metric terms.
    ///
    /// # Notes
    ///
    /// * Automatically validates boundary conditions before solving
    /// * Returns solution at all grid points (both unknown and prescribed)
    /// * Solution is stored sequentially: index m = i + j*nr
    /// * Requires mutable self for metric calculations
    pub fn solve_lmm<F>(&mut self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64, f64) -> f64,
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
    /// Returns `(wwx, wwy)` where:
    ///
    /// * `wwx` contains all x components of the flow vectors (len = number of equations = a.dim())
    /// * `wwy` contains all y components of the flow vectors (len = number of equations = a.dim())
    ///
    /// The flow vector is defined by:
    ///
    /// ```text
    /// →         →
    /// w = - ḵ · ∇ϕ
    /// ```
    pub fn calculate_flow_vectors(&mut self, a: &Vector) -> Result<(Vec<f64>, Vec<f64>), StrError> {
        let neq = self.equations.neq();
        if a.dim() != neq {
            return Err("a.dim() must equal the number of equations");
        }
        let mut wwx = vec![0.0; neq];
        let mut wwy = vec![0.0; neq];
        let mut w = Vector::new(2);
        for m in 0..neq {
            self.calculate_metrics(m);
            let (i, j) = self.grid.get_ij(m);
            w.fill(0.0);
            for n in 0..neq {
                let (k, l) = self.grid.get_ij(n);
                let akl = a[n];
                for d in 0..2 {
                    w[d] += self.mk
                        * (self.d1r(i, k) * delta(j, l) * self.metrics.g_ctr[0][d]
                            + delta(i, k) * self.d1s(j, l) * self.metrics.g_ctr[1][d])
                        * akl
                }
            }
            wwx[m] = w[0];
            wwy[m] = w[1];
        }
        Ok((wwx, wwy))
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
    /// The total number of equations is `nu + np`, which equals nr*ns (total grid points).
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
    /// * `neq` - Number of equations (= unknown DOFs + prescribed DOFs = nr*ns)
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

    /// Returns a mutable reference to the transfinite mapping
    ///
    /// # Returns
    ///
    /// Mutable reference to the [`Transfinite2d`] mapping that transforms reference
    /// coordinates (r, s) to physical coordinates (x, y).
    pub fn get_map(&mut self) -> &mut Transfinite2d {
        &mut self.map
    }

    /// Returns the coefficient matrices for the system partitioning strategy (SPS)
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
    /// * `alpha` -- Helmholtz coefficient (α); set to 0.0 for the Poisson equation
    /// * `extra_nnz` -- extra non-zeros to allocate in the K-bar matrix
    pub fn get_matrices_sps(&mut self, alpha: f64, extra_nnz: usize) -> (CooMatrix, CooMatrix) {
        // allocate matrices
        let nu = self.equations.nu();
        let np = self.equations.np();
        let nr = self.grid.nx();
        let ns = self.grid.ny();
        let neq = self.equations.neq();
        let nnz_wcs = nr * nr * ns * ns; // worst-case scenario
        let mut kk_bar = CooMatrix::new(nu, nu, nnz_wcs + extra_nnz, Sym::No).unwrap();
        let mut kk_check = CooMatrix::new(nu, np, nnz_wcs, Sym::No).unwrap();

        // add all terms to the coefficient matrix
        for m in self.equations.unknown().clone() {
            let (i, j) = self.grid.get_ij(m);
            self.calculate_metrics(m);
            let g11 = self.metrics.gg_mat.get(0, 0);
            let g22 = self.metrics.gg_mat.get(1, 1);
            let g12 = self.metrics.gg_mat.get(0, 1);
            let ll1 = self.metrics.ell_coefficient_for_laplacian(0);
            let ll2 = self.metrics.ell_coefficient_for_laplacian(1);
            if self.nbcs.enabled_ij(i, j, &self.grid) {
                for n in 0..neq {
                    let (k, l) = self.grid.get_ij(n);
                    let mut val = 0.0;
                    if i == 0 {
                        // Xmin
                        if j == l {
                            self.calc_unit_normal(Side::Xmin);
                            let alpha = vec_inner(&self.un, &self.metrics.g_ctr[0]);
                            val += self.mk * self.d1r(i, k) * alpha;
                        }
                    }
                    if i == nr - 1 {
                        // Xmax
                        if j == l {
                            self.calc_unit_normal(Side::Xmax);
                            let alpha = vec_inner(&self.un, &self.metrics.g_ctr[0]);
                            val += self.mk * self.d1r(i, k) * alpha;
                        }
                    }
                    if j == 0 {
                        // Ymin
                        if i == k {
                            self.calc_unit_normal(Side::Ymin);
                            let beta = vec_inner(&self.un, &self.metrics.g_ctr[1]);
                            val += self.mk * self.d1s(j, l) * beta;
                        }
                    }
                    if j == ns - 1 {
                        // Ymax
                        if i == k {
                            self.calc_unit_normal(Side::Ymax);
                            let beta = vec_inner(&self.un, &self.metrics.g_ctr[1]);
                            val += self.mk * self.d1s(j, l) * beta;
                        }
                    }
                    self.put_val(&mut kk_bar, &mut kk_check, m, n, val);
                }
            } else {
                for n in 0..neq {
                    let (k, l) = self.grid.get_ij(n);
                    let mut val = 0.0
                        + self.d2r(i, k) * delta(j, l) * g11
                        + delta(i, k) * self.d2s(j, l) * g22
                        + self.d1r(i, k) * self.d1s(j, l) * 2.0 * g12
                        - self.d1r(i, k) * delta(j, l) * ll1
                        - delta(i, k) * self.d1s(j, l) * ll2;
                    val *= self.mk;
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
    /// * `alpha` -- Helmholtz coefficient (α); set to 0.0 for the Poisson equation
    /// * `extra_nnz` -- extra non-zeros to allocate in the A matrix
    /// * `get_constraints_mat` -- whether to return the constraints matrix or not
    pub fn get_matrices_lmm(
        &mut self,
        alpha: f64,
        extra_nnz: usize,
        get_constraints_mat: bool,
    ) -> (CooMatrix, Option<CooMatrix>) {
        // allocate matrices
        let (neq, nlag, ndim) = self.get_dims_lmm();
        let nr = self.grid.nx();
        let ns = self.grid.ny();
        let nnz_wcs = nr * nr * ns * ns; // worst-case scenario
        let mut mm = CooMatrix::new(ndim, ndim, nnz_wcs + extra_nnz + 2 * nlag, Sym::No).unwrap();

        // add all terms to the coefficient matrix
        for m in 0..neq {
            let (i, j) = self.grid.get_ij(m);
            self.calculate_metrics(m);
            let g11 = self.metrics.gg_mat.get(0, 0);
            let g22 = self.metrics.gg_mat.get(1, 1);
            let g12 = self.metrics.gg_mat.get(0, 1);
            let ll1 = self.metrics.ell_coefficient_for_laplacian(0);
            let ll2 = self.metrics.ell_coefficient_for_laplacian(1);
            if self.nbcs.enabled_ij(i, j, &self.grid) {
                for n in 0..neq {
                    let (k, l) = self.grid.get_ij(n);
                    let mut val = 0.0;
                    if i == 0 {
                        // Xmin
                        if j == l {
                            self.calc_unit_normal(Side::Xmin);
                            let alpha = vec_inner(&self.un, &self.metrics.g_ctr[0]);
                            val += self.mk * self.d1r(i, k) * alpha;
                        }
                    }
                    if i == nr - 1 {
                        // Xmax
                        if j == l {
                            self.calc_unit_normal(Side::Xmax);
                            let alpha = vec_inner(&self.un, &self.metrics.g_ctr[0]);
                            val += self.mk * self.d1r(i, k) * alpha;
                        }
                    }
                    if j == 0 {
                        // Ymin
                        if i == k {
                            self.calc_unit_normal(Side::Ymin);
                            let beta = vec_inner(&self.un, &self.metrics.g_ctr[1]);
                            val += self.mk * self.d1s(j, l) * beta;
                        }
                    }
                    if j == ns - 1 {
                        // Ymax
                        if i == k {
                            self.calc_unit_normal(Side::Ymax);
                            let beta = vec_inner(&self.un, &self.metrics.g_ctr[1]);
                            val += self.mk * self.d1s(j, l) * beta;
                        }
                    }
                    mm.put(m, n, val).unwrap();
                }
            } else {
                for n in 0..neq {
                    let (k, l) = self.grid.get_ij(n);
                    let mut val = 0.0
                        + self.d2r(i, k) * delta(j, l) * g11
                        + delta(i, k) * self.d2s(j, l) * g22
                        + self.d1r(i, k) * self.d1s(j, l) * 2.0 * g12
                        - self.d1r(i, k) * delta(j, l) * ll1
                        - delta(i, k) * self.d1s(j, l) * ll2;
                    val *= self.mk;
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
    pub fn get_vectors_sps<F>(&mut self, source: F) -> (Vector, Vector, Vector)
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
            let (r, s) = self.grid.coord(m);
            self.map.point(&mut self.x, r, s);
            if self.grid.on_boundary(m) {
                // In the SPC, on the Neumann boundary, we solve -k∂ϕ/∂n = q̄ which is different than the
                // FDM approach which still solves the original equation -k ∇²ϕ = source(x,y). Therefore,
                // we must NOT add the source term to f̄ in the SPC.
                if self.grid.is_xmin(m) {
                    let wn = self.nbcs.functions[0](self.x[0], self.x[1]);
                    f_bar[iu] += wn;
                }
                if self.grid.is_xmax(m) {
                    let wn = self.nbcs.functions[1](self.x[0], self.x[1]);
                    f_bar[iu] += wn;
                }
                if self.grid.is_ymin(m) {
                    let wn = self.nbcs.functions[2](self.x[0], self.x[1]);
                    f_bar[iu] += wn;
                }
                if self.grid.is_ymax(m) {
                    let wn = self.nbcs.functions[3](self.x[0], self.x[1]);
                    f_bar[iu] += wn;
                }
            } else {
                // Solving the original equation -k ∇²ϕ = source(x,y)
                f_bar[iu] = source(self.x[0], self.x[1]);
            }
        });
        for index in 0..4 {
            if self.ebcs.sides[index] {
                for &m in self.grid.get_nodes_on_side(Side::from_index(index)) {
                    let ip = self.equations.ip(m);
                    let (r, s) = self.grid.coord(m);
                    self.map.point(&mut self.x, r, s);
                    let val = self.ebcs.functions[index](self.x[0], self.x[1]);
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
    pub fn get_vectors_lmm<F>(&mut self, source: F) -> (Vector, Vector)
    where
        F: Fn(f64, f64) -> f64,
    {
        let (neq, _, ndim) = self.get_dims_lmm();
        let aa = Vector::new(ndim);
        let mut ff = Vector::new(ndim);
        self.grid.for_each_coord(|m, r, s| {
            self.map.point(&mut self.x, r, s);
            if self.grid.on_boundary(m) {
                // In the SPC, on the Neumann boundary, we solve -k∂ϕ/∂n = q̄ which is different than the
                // FDM approach which still solves the original equation -k ∇²ϕ = source(x,y). Therefore,
                // we must NOT add the source term to f̄ in the SPC.
                if self.grid.is_xmin(m) {
                    let wn = self.nbcs.functions[0](self.x[0], self.x[1]);
                    ff[m] += wn;
                }
                if self.grid.is_xmax(m) {
                    let wn = self.nbcs.functions[1](self.x[0], self.x[1]);
                    ff[m] += wn;
                }
                if self.grid.is_ymin(m) {
                    let wn = self.nbcs.functions[2](self.x[0], self.x[1]);
                    ff[m] += wn;
                }
                if self.grid.is_ymax(m) {
                    let wn = self.nbcs.functions[3](self.x[0], self.x[1]);
                    ff[m] += wn;
                }
            } else {
                // Solving the original equation -k ∇²ϕ = source(x,y)
                ff[m] = source(self.x[0], self.x[1]);
            }
        });
        for index in 0..4 {
            if self.ebcs.sides[index] {
                for &m in self.grid.get_nodes_on_side(Side::from_index(index)) {
                    let ip = self.equations.ip(m);
                    let (r, s) = self.grid.coord(m);
                    self.map.point(&mut self.x, r, s);
                    let val = self.ebcs.functions[index](self.x[0], self.x[1]);
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
    pub fn for_each_coord<F>(&mut self, mut callback: F)
    where
        F: FnMut(usize, f64, f64),
    {
        self.grid.for_each_coord(|m, r, s| {
            // map coordinates
            self.map.point(&mut self.x, r, s);
            callback(m, self.x[0], self.x[1]);
        });
    }

    /// Calculates the unit normal at boundary
    fn calc_unit_normal(&mut self, side: Side) {
        match side {
            Side::Xmin => vec_copy_scaled(&mut self.un, -1.0, &self.metrics.g_ctr[0]).unwrap(),
            Side::Xmax => vec_copy_scaled(&mut self.un, 1.0, &self.metrics.g_ctr[0]).unwrap(),
            Side::Ymin => vec_copy_scaled(&mut self.un, -1.0, &self.metrics.g_ctr[1]).unwrap(),
            Side::Ymax => vec_copy_scaled(&mut self.un, 1.0, &self.metrics.g_ctr[1]).unwrap(),
        }
        let norm_u = vec_norm(&mut self.un, Norm::Euc);
        self.un[0] /= norm_u;
        self.un[1] /= norm_u;
    }

    /// Puts the value into the correct position of the coefficient matrix
    fn put_val(&mut self, kk_bar: &mut CooMatrix, kk_check: &mut CooMatrix, m: usize, n: usize, val: f64) {
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

    /// Maps the coordinates and calculates the metrics at point m
    fn calculate_metrics(&mut self, m: usize) {
        // map coordinates
        let (r, s) = self.grid.coord(m);
        self.map.point_and_derivs(
            &mut self.x,
            &mut self.dx_dr,
            &mut self.dx_ds,
            Some(&mut self.d2x_dr2),
            Some(&mut self.d2x_ds2),
            Some(&mut self.d2x_drs),
            r,
            s,
        );
        // calculate metrics
        self.metrics
            .calculate_2d(
                &self.dx_dr,
                &self.dx_ds,
                Some(&self.d2x_dr2),
                Some(&self.d2x_ds2),
                Some(&self.d2x_drs),
            )
            .unwrap();
    }

    /// Returns the (i,j) component of the first derivative with respect to r
    #[inline]
    fn d1r(&self, i: usize, j: usize) -> f64 {
        self.interp_r.get_dd1().unwrap().get(i, j)
    }

    /// Returns the (i,j) component of the first derivative with respect to s
    #[inline]
    fn d1s(&self, i: usize, j: usize) -> f64 {
        self.interp_s.get_dd1().unwrap().get(i, j)
    }

    /// Returns the (i,j) component of the second derivative with respect to r
    #[inline]
    fn d2r(&self, i: usize, j: usize) -> f64 {
        self.interp_r.get_dd2().unwrap().get(i, j)
    }

    /// Returns the (i,j) component of the second derivative with respect to s
    #[inline]
    fn d2s(&self, i: usize, j: usize) -> f64 {
        self.interp_s.get_dd2().unwrap().get(i, j)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::SpcMap2d;
    use crate::{EssentialBcs2d, NaturalBcs2d, Side, TransfiniteSamples};
    use russell_lab::{mat_approx_eq, Vector};
    use russell_sparse::Sym;

    #[test]
    fn new_captures_errors() {
        let map = TransfiniteSamples::quadrilateral_2d(&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]);
        let ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        assert_eq!(SpcMap2d::new(map, 1, 2, ebcs, nbcs, 1.0).err(), Some("nr must be ≥ 2"));

        let map = TransfiniteSamples::quadrilateral_2d(&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]);
        let ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        assert_eq!(SpcMap2d::new(map, 2, 1, ebcs, nbcs, 1.0).err(), Some("ns must be ≥ 2"));

        let map = TransfiniteSamples::quadrilateral_2d(&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]);
        let mut ebcs = EssentialBcs2d::new();
        let mut nbcs = NaturalBcs2d::new();
        ebcs.set(Side::Xmin, |_, _| 0.0);
        nbcs.set(Side::Xmax, |_, _| 0.0);
        ebcs.set(Side::Ymin, |_, _| 0.0);
        nbcs.set(Side::Ymax, |_, _| 0.0);
        assert_eq!(
            SpcMap2d::new(map, 2050, 2, ebcs, nbcs, 1.0).err(),
            Some("the maximum allowed polynomial degree is 2048")
        );

        let map = TransfiniteSamples::quadrilateral_2d(&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]);
        let mut ebcs = EssentialBcs2d::new();
        let mut nbcs = NaturalBcs2d::new();
        ebcs.set(Side::Xmin, |_, _| 0.0);
        nbcs.set(Side::Xmax, |_, _| 0.0);
        ebcs.set(Side::Ymin, |_, _| 0.0);
        nbcs.set(Side::Ymax, |_, _| 0.0);
        assert_eq!(
            SpcMap2d::new(map, 2, 2050, ebcs, nbcs, 1.0).err(),
            Some("the maximum allowed polynomial degree is 2048")
        );

        let map = TransfiniteSamples::quadrilateral_2d(&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]);
        let mut ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        ebcs.set_periodic(true, true);
        assert_eq!(
            SpcMap2d::new(map, 3, 3, ebcs, nbcs, 1.0).err(),
            Some("essential BCs cannot be periodic")
        );
    }

    #[test]
    fn calculate_flow_vectors_captures_errors() {
        let map = TransfiniteSamples::quadrilateral_2d(&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]);
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_homogeneous();
        let nbcs = NaturalBcs2d::new();
        let mut spc = SpcMap2d::new(map, 2, 2, ebcs, nbcs, 1.0).unwrap();
        let a = Vector::from(&[0.0]); // wrong size
        assert_eq!(
            spc.calculate_flow_vectors(&a).err(),
            Some("a.dim() must equal the number of equations")
        );
    }

    #[test]
    fn get_matrices_works_1() {
        let map = TransfiniteSamples::quadrilateral_2d(&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]);
        let mut ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        ebcs.set_homogeneous();
        let (nr, ns) = (5, 5);
        let mut spc = SpcMap2d::new(map, nr, ns, ebcs, nbcs, 1.0).unwrap();
        let (kk_bar, kk_check) = spc.get_matrices_sps(0.0, 0);
        let kk_bar_dense = kk_bar.as_dense();

        let (nu, np) = (9, 16);
        assert_eq!(spc.get_dims_sps(), (nu, np));
        assert_eq!(spc.get_equations().nu(), nu);
        assert_eq!(spc.get_equations().np(), np);

        let ___ = 0.0;
        #[rustfmt::skip]
        let correct_kk_bar = &[
            [ 28.0, -6.0,  2.0, -6.0,  ___,  ___,  2.0,  ___,  ___],
            [ -4.0, 20.0, -4.0,  ___, -6.0,  ___,  ___,  2.0,  ___],
            [  2.0, -6.0, 28.0,  ___,  ___, -6.0,  ___,  ___,  2.0],
            [ -4.0, ___,   ___, 20.0, -6.0,  2.0, -4.0,  ___,  ___],
            [  ___, -4.0,  ___, -4.0, 12.0, -4.0,  ___, -4.0,  ___],
            [  ___, ___,  -4.0,  2.0, -6.0, 20.0,  ___,  ___, -4.0],
            [  2.0, ___,   ___, -6.0,  ___,  ___, 28.0, -6.0,  2.0],
            [  ___, 2.0,   ___,  ___, -6.0,  ___, -4.0, 20.0, -4.0],
            [  ___, ___,   2.0,  ___,  ___, -6.0,  2.0, -6.0, 28.0],
        ];
        mat_approx_eq(&kk_bar_dense, correct_kk_bar, 1e-13);

        #[rustfmt::skip]
        let correct_kk_check = &[
            [0.0, -9.242640687119286, 0.0, 0.0, 0.0, -9.242640687119286, -0.7573593128807143, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7573593128807143, 0.0, 0.0, 0.0],
            [0.0, 0.0, -9.242640687119286, 0.0, 0.0, 0.9999999999999998, 0.9999999999999993, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7573593128807143, 0.0, 0.0],
            [0.0, 0.0, 0.0, -9.242640687119286, 0.0, -0.757359312880714, -9.242640687119286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7573593128807143, 0.0],
            [0.0, 0.9999999999999998, 0.0, 0.0, 0.0, 0.0, 0.0, -9.242640687119286, -0.7573593128807143, 0.0, 0.0, 0.0, 0.9999999999999993, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9999999999999998, 0.0, 0.0, 0.0, 0.0, 0.9999999999999998, 0.9999999999999993, 0.0, 0.0, 0.0, 0.0, 0.9999999999999993, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9999999999999998, 0.0, 0.0, 0.0, -0.757359312880714, -9.242640687119286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999999993, 0.0],
            [0.0, -0.757359312880714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.242640687119286, -0.7573593128807143, 0.0, -9.242640687119286, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.757359312880714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999999998, 0.9999999999999993, 0.0, 0.0, -9.242640687119286, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.757359312880714, 0.0, 0.0, 0.0, 0.0, 0.0, -0.757359312880714, -9.242640687119286, 0.0, 0.0, 0.0, -9.242640687119286, 0.0],
        ];
        let kk_check_dense = kk_check.as_dense();
        mat_approx_eq(&kk_check_dense, correct_kk_check, 1e-14);

        let neq = nu + np;
        let nlag = np;
        let ndim = neq + nlag;
        assert_eq!(spc.get_dims_lmm(), (neq, nlag, ndim));

        let nnz = neq * neq + 2 * nlag;
        let (mm, cc) = spc.get_matrices_lmm(0.0, 0, true);
        assert_eq!(mm.get_info(), (ndim, ndim, nnz, Sym::No));
        let cc = cc.unwrap();
        assert_eq!(cc.get_info(), (nlag, neq, nlag, Sym::No));

        // check that get_map works
        let _ = spc.get_map();
    }
}
