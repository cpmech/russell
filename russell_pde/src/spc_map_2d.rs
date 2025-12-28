use crate::util::delta;
use crate::{EquationHandler, EssentialBcs2d, Grid2d, Metrics, NaturalBcs2d, Side, StrError, Transfinite2d};
use russell_lab::{vec_copy_scaled, vec_inner, vec_norm, InterpLagrange, Norm, Vector};
use russell_sparse::{CooMatrix, Genie, LinSolver, Sym};

/// Implements the Spectral Collocation method (SPC) in 2D with Transfinite Mapping for curvilinear coordinates
///
/// The SPC can be used to solve the following problem (Poisson or Helmholtz equation):
///
/// ```text
/// -k ∇²ϕ + α ϕ = source(x,y)
/// ```
///
/// where
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
/// where `gᵢⱼ` are the covariant metric coefficients and `Γᵏᵢⱼ` are the Christoffel
/// symbols of the second kind.
///
/// The reference coordinates (r, s) in [-1, 1] × [-1, 1] are mapped onto the physical
/// coordinates (x, y) using Transfinite Mapping.
///
/// The spectral collocation method approximates the Laplacian at the grid points
/// (xᵢ, yⱼ) using:
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
/// where ϕᵢⱼ are the discrete counterpart of ϕ(x(r,s), y(r,s)) over the (nr, ns) grid.
/// However, these values are "sequentially" mapped onto to the vector `a` using the
/// following formula:
///
/// ```text
/// ϕᵢⱼ → aₘ   with   m = i + j nr
/// ```
///
/// thus, we can write the discrete Laplacian operator as:
///
/// ```text
/// (∇²a)ₘ = ∑ₙ Kₘₙ aₙ
/// ```
///
/// Two methods are implemented to handle the essential boundary conditions:
///
/// 1. System Partitioning Strategy (SPS)
/// 2. Lagrange Multipliers Method (LMM)
pub struct SpcMap2d<'a> {
    /// Defines the 2D grid on the reference domain [-1, 1] × [-1, 1]
    grid: Grid2d,

    /// Holds a reference to the essential boundary conditions handler
    ebcs: EssentialBcs2d<'a>,

    /// Holds a reference to the natural boundary conditions handler
    nbcs: NaturalBcs2d<'a>,

    /// Negative of the diffusion coefficient
    ///
    /// mk = -k
    mk: f64,

    /// Tool to handle the equation numbers such as unknowns prescribed
    equations: EquationHandler,

    /// Polynomial interpolator along r
    interp_r: InterpLagrange,

    /// Polynomial interpolator along s
    interp_s: InterpLagrange,

    /// Base vectors and metrics related to the curvilinear coordinates
    metrics: Metrics,

    /// Transfinite mapping from reference to physical domain (r, s) → (x, y)
    map: Transfinite2d,

    /// Temporary: Physical coordinates
    x: Vector,

    /// Temporary: Derivative of x with respect to r
    dx_dr: Vector,

    /// Temporary: Derivative of x with respect to s
    dx_ds: Vector,

    /// Temporary: Second derivative of x with respect to r
    d2x_dr2: Vector,

    /// Temporary: Second derivative of x with respect to s
    d2x_ds2: Vector,

    /// Temporary: Mixed second derivative of x with respect to r and s
    d2x_drs: Vector,

    /// Temporary: Unit normal vector
    un: Vector,
}

impl<'a> SpcMap2d<'a> {
    /// Allocates a new instance
    ///
    /// **Important**:
    ///
    /// 1. All sides must have essential BCs, i.e., Neumann BCs are **not** allowed.
    ///
    /// # Arguments
    ///
    /// * `map` -- the transfinite mapping from reference to physical domain
    /// * `nr` -- number of grid points along r (for the Chebyshev-Gauss-Lobatto grid)
    /// * `ns` -- number of grid points along s (for the Chebyshev-Gauss-Lobatto grid)
    /// * `ebcs` -- the essential boundary conditions handler
    /// * `nbcs` -- the natural boundary conditions handler
    /// * `k` -- the diffusion coefficient
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

        // validates the boundary conditions data
        ebcs.validate(&nbcs)?;

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

    /// Solves the Poisson or Helmholtz equation using the system partitioning strategy (SPS)
    ///
    /// Returns the solution vector `a`.
    ///
    /// ```text
    /// -k ∇²ϕ + α ϕ = source(x,y)
    /// ```
    pub fn solve_sps<F>(&mut self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64, f64) -> f64,
    {
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

    /// Solves the Poisson or Helmholtz equation using the Lagrange multipliers method (LMM)
    ///
    /// Returns the solution vector `a`.
    ///
    /// ```text
    /// -k ∇²ϕ + α ϕ = source(x,y)
    /// ```
    pub fn solve_lmm<F>(&mut self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64, f64) -> f64,
    {
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

    /// Access the equation numbering handler
    pub fn get_equations(&self) -> &EquationHandler {
        &self.equations
    }

    /// Access the transfinite mapping
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
    /// * `alpha` -- Helmholtz coefficient (α). Set to 0.0 for the Poisson equation
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
                    if i == 0 || i == nr - 1 {
                        // Xmin or Xmax
                        if j == l {
                            self.calc_unit_normal(Side::Xmax);
                            let alpha = vec_inner(&self.un, &self.metrics.g_ctr[0]);
                            val += self.mk * self.d1r(i, k) * alpha;
                        }
                    }
                    if j == 0 || j == ns - 1 {
                        // Ymin or Ymax
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
    /// * `alpha` -- Helmholtz coefficient (α). Set to 0.0 for the Poisson equation
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
                    if i == 0 || i == nr - 1 {
                        // Xmin or Xmax
                        if j == l {
                            self.calc_unit_normal(Side::Xmax);
                            let alpha = vec_inner(&self.un, &self.metrics.g_ctr[0]);
                            val += self.mk * self.d1r(i, k) * alpha;
                        }
                    }
                    if j == 0 || j == ns - 1 {
                        // Ymin or Ymax
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
        // println!("{:.2}", kk_bar_dense);

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
        // println!("{:.3}", kk_check_dense);
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
    }
}
