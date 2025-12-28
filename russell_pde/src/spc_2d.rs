use crate::{EquationHandler, EssentialBcs2d, Grid2d, NaturalBcs2d, Side, StrError};
use russell_lab::{InterpLagrange, Vector};
use russell_sparse::{CooMatrix, Genie, LinSolver, Sym};

/// Implements the Spectral Collocation method (SPC) in 2D
///
/// The SPC can be used to solve the following problem (Poisson or Helmholtz equation):
///
/// ```text
///     ∂²ϕ      ∂²ϕ
/// -kx ——— - ky ——— + α ϕ = source(x, y)
///     ∂x²      ∂y²
/// ```
///
/// with essential (EBC) and natural (NBC) boundary conditions.
///
/// The spectral collocation method approximates the Laplacian at the grid points (xᵢ, yⱼ) using:
///
/// ```text
///               ∂²ϕ│             ∂²ϕ│
/// (∇²ϕ)ᵢⱼ = -kx ———│        - ky ———│        = ∑ₖ ∑ₗ (mkx D⁽²⁾ᵢₖ δⱼₗ + mky δᵢₖ D̄⁽²⁾ⱼₗ) ϕₖₗ
///               ∂x²│(xᵢ,xⱼ)      ∂y²│(xᵢ,xⱼ)
///
/// mkx = -kx, mky = -ky
/// ```
///
/// where ϕᵢⱼ are the discrete counterpart of ϕ(x, y) over the (nx, ny) grid. However, these
/// values are "sequentially" mapped onto to the vector `a` using the following formula:
///
/// ```text
/// ϕᵢⱼ → aₘ   with   m = i + j nx
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
pub struct Spc2d<'a> {
    /// Minimum x-coordinate
    xmin: f64,

    /// Maximum x-coordinate
    xmax: f64,

    /// Minimum y-coordinate
    ymin: f64,

    /// Maximum y-coordinate
    ymax: f64,

    /// Defines the 2D grid
    grid: Grid2d,

    /// Holds a reference to the essential boundary conditions handler
    ebcs: EssentialBcs2d<'a>,

    /// Holds a reference to the natural boundary conditions handler
    nbcs: NaturalBcs2d<'a>,

    /// Negative of the diffusion coefficient along x
    ///
    /// mkx = -kx
    mkx: f64,

    /// Negative of the diffusion coefficient along y
    ///
    /// mky = -ky
    mky: f64,

    /// Tool to handle the equation numbers such as unknowns prescribed
    equations: EquationHandler,

    /// Polynomial interpolator along x
    interp_x: InterpLagrange,

    /// Polynomial interpolator along y
    interp_y: InterpLagrange,
}

impl<'a> Spc2d<'a> {
    /// Allocates a new instance
    ///
    /// # Arguments
    ///
    /// * `nx` -- number of grid points along x (for the Chebyshev-Gauss-Lobatto grid)
    /// * `ny` -- number of grid points along y (for the Chebyshev-Gauss-Lobatto grid)
    /// * `ebcs` -- the essential boundary conditions handler
    /// * `nbcs` -- the natural boundary conditions handler
    /// * `kx` -- the diffusion coefficient along x
    /// * `ky` -- the diffusion coefficient along y
    pub fn new(
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        nx: usize,
        ny: usize,
        ebcs: EssentialBcs2d<'a>,
        nbcs: NaturalBcs2d<'a>,
        kx: f64,
        ky: f64,
    ) -> Result<Self, StrError> {
        // check
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }
        if ny < 2 {
            return Err("ny must be ≥ 2");
        }

        // polynomial degrees
        let nn_x = nx - 1;
        let nn_y = ny - 1;
        if nn_x > 2048 || nn_y > 2048 {
            return Err("the maximum allowed polynomial degree is 2048");
        }

        // allocate the Chebyshev-Gauss-Lobatto grid
        let grid = Grid2d::new_chebyshev_gauss_lobatto(nx, ny).unwrap();

        // validates the boundary conditions data
        ebcs.validate(&nbcs)?;

        // check that the EBCs are not periodic
        if ebcs.periodic_along_x || ebcs.periodic_along_y {
            return Err("essential BCs cannot be periodic");
        }

        // allocate equations handler
        let neq = grid.size();
        let mut equations = EquationHandler::new(neq);
        equations.recompute(&ebcs.get_nodes(&grid));

        // interpolators
        let mut interp_x = InterpLagrange::new(nn_x, None).unwrap();
        let mut interp_y = InterpLagrange::new(nn_y, None).unwrap();
        interp_x.calc_dd1_matrix();
        interp_y.calc_dd1_matrix();
        interp_x.calc_dd2_matrix();
        interp_y.calc_dd2_matrix();

        // done
        Ok(Spc2d {
            xmin,
            xmax,
            ymin,
            ymax,
            grid,
            ebcs,
            nbcs,
            mkx: -kx,
            mky: -ky,
            equations,
            interp_x,
            interp_y,
        })
    }

    /// Solves the Poisson or Helmholtz equation using the system partitioning strategy (SPS)
    ///
    /// Returns the solution vector `a`.
    ///
    /// ```text
    ///     ∂²ϕ      ∂²ϕ
    /// -kx ——— - ky ——— + α ϕ = source(x, y)
    ///     ∂x²      ∂y²
    /// ```
    pub fn solve_sps<F>(&self, alpha: f64, source: F) -> Result<Vector, StrError>
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
    ///     ∂²ϕ      ∂²ϕ
    /// -kx ——— - ky ——— + α ϕ = source(x, y)
    ///     ∂x²      ∂y²
    /// ```
    pub fn solve_lmm<F>(&self, alpha: f64, source: F) -> Result<Vector, StrError>
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
    pub fn calculate_flow_vectors(&self, a: &Vector) -> Result<(Vec<f64>, Vec<f64>), StrError> {
        let neq = self.equations.neq();
        if a.dim() != neq {
            return Err("a.dim() must equal the number of equations");
        }
        let d1r = self.interp_x.get_dd1().unwrap();
        let d1s = self.interp_y.get_dd1().unwrap();
        let dr_dx = 2.0 / (self.xmax - self.xmin);
        let ds_dy = 2.0 / (self.ymax - self.ymin);
        let mut wwx = vec![0.0; neq];
        let mut wwy = vec![0.0; neq];
        let mut w = Vector::new(2);
        for m in 0..neq {
            let (i, j) = self.grid.get_ij(m);
            w.fill(0.0);
            for n in 0..neq {
                let (k, l) = self.grid.get_ij(n);
                let akl = a[n];
                if j == l {
                    w[0] += self.mkx * d1r.get(i, k) * dr_dx * akl;
                }
                if i == k {
                    w[1] += self.mky * d1s.get(j, l) * ds_dy * akl;
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
    pub fn get_matrices_sps(&self, alpha: f64, extra_nnz: usize) -> (CooMatrix, CooMatrix) {
        // allocate matrices
        let nu = self.equations.nu();
        let np = self.equations.np();
        let nx = self.grid.nx();
        let ny = self.grid.ny();
        let neq = self.equations.neq();
        let nnz_wcs = nx * nx * ny * ny; // worst-case scenario
        let mut kk_bar = CooMatrix::new(nu, nu, nnz_wcs + extra_nnz, Sym::No).unwrap();
        let mut kk_check = CooMatrix::new(nu, np, nnz_wcs, Sym::No).unwrap();

        // spectral derivative matrices
        let d1r = self.interp_x.get_dd1().unwrap();
        let d1s = self.interp_y.get_dd1().unwrap();
        let d2r = self.interp_x.get_dd2().unwrap();
        let d2s = self.interp_y.get_dd2().unwrap();

        // scaling coefficients due to domain mapping (from [-1,1]×[-1,1] to [xmin,xmax]×[ymin,ymax])
        let dr_dx = 2.0 / (self.xmax - self.xmin);
        let ds_dy = 2.0 / (self.ymax - self.ymin);
        let cx = dr_dx * dr_dx;
        let cy = ds_dy * ds_dy;

        // add terms to the coefficient matrix
        for &m in self.equations.unknown() {
            let (i, j) = self.grid.get_ij(m);
            if self.nbcs.enabled_ij(i, j, &self.grid) {
                for n in 0..neq {
                    let (k, l) = self.grid.get_ij(n);
                    let mut val = 0.0;
                    if i == 0 {
                        // Xmin
                        if j == l {
                            val += -self.mkx * d1r.get(i, k) * dr_dx; // -1 due to the normal pointing left
                        }
                    }
                    if i == nx - 1 {
                        // Xmax
                        if j == l {
                            val += self.mkx * d1r.get(i, k) * dr_dx;
                        }
                    }
                    if j == 0 {
                        // Ymin
                        if i == k {
                            val += -self.mky * d1s.get(j, l) * ds_dy; // -1 due to the normal pointing down
                        }
                    }
                    if j == ny - 1 {
                        // Ymax
                        if i == k {
                            val += self.mky * d1s.get(j, l) * ds_dy;
                        }
                    }
                    self.put_val(&mut kk_bar, &mut kk_check, m, n, val);
                }
            } else {
                for n in 0..neq {
                    let (k, l) = self.grid.get_ij(n);
                    let mut val = 0.0;
                    if j == l {
                        val += self.mkx * d2r.get(i, k) * cx;
                    }
                    if i == k {
                        val += self.mky * d2s.get(j, l) * cy;
                    }
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
        &self,
        alpha: f64,
        extra_nnz: usize,
        get_constraints_mat: bool,
    ) -> (CooMatrix, Option<CooMatrix>) {
        // allocate matrices
        let (neq, nlag, ndim) = self.get_dims_lmm();
        let nx = self.grid.nx();
        let ny = self.grid.ny();
        let nnz_wcs = nx * nx * ny * ny; // worst-case scenario
        let mut mm = CooMatrix::new(ndim, ndim, nnz_wcs + extra_nnz + 2 * nlag, Sym::No).unwrap();

        // spectral derivative matrices
        let d1r = self.interp_x.get_dd1().unwrap();
        let d1s = self.interp_y.get_dd1().unwrap();
        let d2r = self.interp_x.get_dd2().unwrap();
        let d2s = self.interp_y.get_dd2().unwrap();

        // scaling coefficients due to domain mapping (from [-1,1]×[-1,1] to [xmin,xmax]×[ymin,ymax])
        let dr_dx = 2.0 / (self.xmax - self.xmin);
        let ds_dy = 2.0 / (self.ymax - self.ymin);
        let cx = dr_dx * dr_dx;
        let cy = ds_dy * ds_dy;

        // add terms to the coefficient matrix
        for m in 0..neq {
            let (i, j) = self.grid.get_ij(m);
            if self.nbcs.enabled_ij(i, j, &self.grid) {
                for n in 0..neq {
                    let (k, l) = self.grid.get_ij(n);
                    let mut val = 0.0;
                    if i == 0 {
                        // Xmin
                        if j == l {
                            val += -self.mkx * d1r.get(i, k) * dr_dx; // -1 due to the normal pointing left
                        }
                    }
                    if i == nx - 1 {
                        // Xmax
                        if j == l {
                            val += self.mkx * d1r.get(i, k) * dr_dx;
                        }
                    }
                    if j == 0 {
                        // Ymin
                        if i == k {
                            val += -self.mky * d1s.get(j, l) * ds_dy; // -1 due to the normal pointing down
                        }
                    }
                    if j == ny - 1 {
                        // Ymax
                        if i == k {
                            val += self.mky * d1s.get(j, l) * ds_dy;
                        }
                    }
                    mm.put(m, n, val).unwrap();
                }
            } else {
                for n in 0..neq {
                    let (k, l) = self.grid.get_ij(n);
                    let n = k + l * nx;
                    let mut val = 0.0;
                    if j == l {
                        val += self.mkx * d2r.get(i, k) * cx;
                    }
                    if i == k {
                        val += self.mky * d2s.get(j, l) * cy;
                    }
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
            let (r, s) = self.grid.coord(m);
            let (x, y) = self.map_coord(r, s);
            if self.grid.on_boundary(m) {
                // In the SPC, on the Neumann boundary, we solve -k∂ϕ/∂n = q̄ which is different than the
                // FDM approach which still solves the original equation -k ∇²ϕ = source(x,y). Therefore,
                // we must NOT add the source term to f̄ in the SPC.
                if self.grid.is_xmin(m) {
                    let wn = self.nbcs.functions[0](x, y);
                    f_bar[iu] += wn;
                }
                if self.grid.is_xmax(m) {
                    let wn = self.nbcs.functions[1](x, y);
                    f_bar[iu] += wn;
                }
                if self.grid.is_ymin(m) {
                    let wn = self.nbcs.functions[2](x, y);
                    f_bar[iu] += wn;
                }
                if self.grid.is_ymax(m) {
                    let wn = self.nbcs.functions[3](x, y);
                    f_bar[iu] += wn;
                }
            } else {
                // Solving the original equation -k ∇²ϕ = source(x,y)
                f_bar[iu] = source(x, y);
            }
        });
        for index in 0..4 {
            if self.ebcs.sides[index] {
                for &m in self.grid.get_nodes_on_side(Side::from_index(index)) {
                    let ip = self.equations.ip(m);
                    let (r, s) = self.grid.coord(m);
                    let (x, y) = self.map_coord(r, s);
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
        self.grid.for_each_coord(|m, r, s| {
            let (x, y) = self.map_coord(r, s);
            if self.grid.on_boundary(m) {
                // In the SPC, on the Neumann boundary, we solve -k∂ϕ/∂n = q̄ which is different than the
                // FDM approach which still solves the original equation -k ∇²ϕ = source(x,y). Therefore,
                // we must NOT add the source term to f̄ in the SPC.
                if self.grid.is_xmin(m) {
                    let wn = self.nbcs.functions[0](x, y);
                    ff[m] += wn;
                }
                if self.grid.is_xmax(m) {
                    let wn = self.nbcs.functions[1](x, y);
                    ff[m] += wn;
                }
                if self.grid.is_ymin(m) {
                    let wn = self.nbcs.functions[2](x, y);
                    ff[m] += wn;
                }
                if self.grid.is_ymax(m) {
                    let wn = self.nbcs.functions[3](x, y);
                    ff[m] += wn;
                }
            } else {
                // Solving the original equation -k ∇²ϕ = source(x,y)
                ff[m] = source(x, y);
            }
        });
        for index in 0..4 {
            if self.ebcs.sides[index] {
                for &m in self.grid.get_nodes_on_side(Side::from_index(index)) {
                    let ip = self.equations.ip(m);
                    let (r, s) = self.grid.coord(m);
                    let (x, y) = self.map_coord(r, s);
                    let val = self.ebcs.functions[index](x, y);
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
    pub fn for_each_coord<F>(&self, mut callback: F)
    where
        F: FnMut(usize, f64, f64),
    {
        self.grid.for_each_coord(|m, r, s| {
            let (x, y) = self.map_coord(r, s);
            callback(m, x, y);
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

    /// Maps the reference coordinates (r,s) in [-1,1]×[-1,1] to the physical coordinates (x,y)
    fn map_coord(&self, r: f64, s: f64) -> (f64, f64) {
        let x = (self.xmax + self.xmin + (self.xmax - self.xmin) * r) / 2.0;
        let y = (self.ymax + self.ymin + (self.ymax - self.ymin) * s) / 2.0;
        (x, y)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Spc2d;
    use crate::{EssentialBcs2d, NaturalBcs2d, Side};
    use russell_lab::{mat_approx_eq, Vector};
    use russell_sparse::Sym;

    #[test]
    fn new_captures_errors() {
        let ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        assert_eq!(
            Spc2d::new(0.0, 1.0, 0.0, 1.0, 1, 2, ebcs, nbcs, 1.0, 1.0).err(),
            Some("nx must be ≥ 2")
        );

        let ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        assert_eq!(
            Spc2d::new(0.0, 1.0, 0.0, 1.0, 2, 1, ebcs, nbcs, 1.0, 1.0).err(),
            Some("ny must be ≥ 2")
        );

        let mut ebcs = EssentialBcs2d::new();
        let mut nbcs = NaturalBcs2d::new();
        ebcs.set(Side::Xmin, |_, _| 0.0);
        nbcs.set(Side::Xmax, |_, _| 0.0);
        ebcs.set(Side::Ymin, |_, _| 0.0);
        nbcs.set(Side::Ymax, |_, _| 0.0);
        assert_eq!(
            Spc2d::new(0.0, 1.0, 0.0, 1.0, 2050, 2, ebcs, nbcs, 1.0, 1.0).err(),
            Some("the maximum allowed polynomial degree is 2048")
        );

        let mut ebcs = EssentialBcs2d::new();
        let mut nbcs = NaturalBcs2d::new();
        ebcs.set(Side::Xmin, |_, _| 0.0);
        nbcs.set(Side::Xmax, |_, _| 0.0);
        ebcs.set(Side::Ymin, |_, _| 0.0);
        nbcs.set(Side::Ymax, |_, _| 0.0);
        assert_eq!(
            Spc2d::new(0.0, 1.0, 0.0, 1.0, 2, 2050, ebcs, nbcs, 1.0, 1.0).err(),
            Some("the maximum allowed polynomial degree is 2048")
        );

        let mut ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        ebcs.set_periodic(true, true);
        assert_eq!(
            Spc2d::new(0.0, 1.0, 0.0, 1.0, 3, 3, ebcs, nbcs, 1.0, 1.0).err(),
            Some("essential BCs cannot be periodic")
        );
    }

    #[test]
    fn calculate_flow_vectors_captures_errors() {
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_homogeneous();
        let nbcs = NaturalBcs2d::new();
        let spc = Spc2d::new(-1.0, 1.0, -1.0, 1.0, 2, 2, ebcs, nbcs, 1.0, 1.0).unwrap();
        let a = Vector::from(&[0.0]); // wrong size
        assert_eq!(
            spc.calculate_flow_vectors(&a).err(),
            Some("a.dim() must equal the number of equations")
        );
    }

    #[test]
    fn get_matrices_works_1() {
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_homogeneous();
        let nbcs = NaturalBcs2d::new();
        let (nx, ny) = (5, 5);
        let spc = Spc2d::new(-1.0, 1.0, -1.0, 1.0, nx, ny, ebcs, nbcs, 1.0, 1.0).unwrap();
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
        mat_approx_eq(&kk_bar_dense, correct_kk_bar, 1e-14);

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
