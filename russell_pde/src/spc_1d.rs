use crate::{EquationHandler, EssentialBcs1d, Grid1d, NaturalBcs1d, Side, StrError};
use russell_lab::{InterpLagrange, Vector};
use russell_sparse::{CooMatrix, Genie, LinSolver, Sym};

/// Implements the Spectral Collocation method (SPC) in 1D
///
/// The SPC can be used to solve the following problem (Poisson or Helmholtz equation):
///
/// ```text
///     ∂²ϕ
/// -kx ——— + α ϕ = source(x, y)
///     ∂x²
/// ```
///
/// with essential (EBC) and natural (NBC) boundary conditions.
///
/// The spectral collocation method approximates the Laplacian at the grid points (xᵢ, yⱼ) using:
///
/// ```text
///              ∂²ϕ│
/// (∇²ϕ)ᵢ = -kx ———│   = ∑ⱼ mkx D⁽²⁾ᵢⱼ ϕⱼ
///              ∂x²│xᵢ
///
/// mkx = -kx
/// ```
///
/// where ϕᵢ are the discrete counterpart of ϕ(x) over the (nx) grid.
///
/// We can write the discrete Laplacian operator as:
///
/// ```text
/// (∇²a)ₘ = ∑ₙ Kₘₙ aₙ
/// ```
///
/// Two methods are implemented to handle the essential boundary conditions:
///
/// 1. System Partitioning Strategy (SPS)
/// 2. Lagrange Multipliers Method (LMM)
pub struct Spc1d<'a> {
    /// Minimum x-coordinate
    xmin: f64,

    /// Maximum x-coordinate
    xmax: f64,

    /// Defines the 1D grid
    grid: Grid1d,

    /// Holds a reference to the essential boundary conditions handler
    ebcs: EssentialBcs1d<'a>,

    /// Holds a reference to the natural boundary conditions handler
    nbcs: NaturalBcs1d<'a>,

    /// Negative of the diffusion coefficient along x
    ///
    /// mkx = -kx
    mkx: f64,

    /// Tool to handle the equation numbers such as unknowns prescribed
    equations: EquationHandler,

    /// Polynomial interpolator along x
    interp_x: InterpLagrange,
}

impl<'a> Spc1d<'a> {
    /// Allocates a new instance
    ///
    /// # Arguments
    ///
    /// * `xmin` -- minimum x-coordinate of the domain
    /// * `xmax` -- maximum x-coordinate of the domain
    /// * `nx` -- number of grid points along x (for the Chebyshev-Gauss-Lobatto grid)
    /// * `ebcs` -- the essential boundary conditions handler
    /// * `nbcs` -- the natural boundary conditions handler
    /// * `kx` -- the diffusion coefficient along x
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ebcs: EssentialBcs1d<'a>,
        nbcs: NaturalBcs1d<'a>,
        kx: f64,
    ) -> Result<Self, StrError> {
        // allocate the Chebyshev-Gauss-Lobatto grid
        let grid = Grid1d::new_chebyshev_gauss_lobatto(nx)?;

        // validates the boundary conditions data
        ebcs.validate(&nbcs)?;

        // check that the EBCs are not periodic
        if ebcs.periodic_along_x {
            return Err("essential BCs cannot be periodic");
        }

        // allocate equations handler
        let neq = grid.nx();
        let mut equations = EquationHandler::new(neq);
        equations.recompute(&ebcs.get_nodes(&grid));

        // polynomial degree
        let nn_x = grid.nx() - 1;

        // interpolators
        let mut interp_x = InterpLagrange::new(nn_x, None)?;
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
            interp_x,
        })
    }

    /// Solves the Poisson or Helmholtz equation using the system partitioning strategy (SPS)
    ///
    /// Returns the solution vector `a`.
    ///
    /// ```text
    ///     ∂²ϕ
    /// -kx ——— + α ϕ = source(x)
    ///     ∂x²
    /// ```
    pub fn solve_sps<F>(&self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64) -> f64,
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
    ///     ∂²ϕ
    /// -kx ——— + α ϕ = source(x)
    ///     ∂x²
    /// ```
    pub fn solve_lmm<F>(&self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64) -> f64,
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
        let nnz_wcs = nx * nx; // worst-case scenario
        let mut kk_bar = CooMatrix::new(nu, nu, nnz_wcs + extra_nnz, Sym::No).unwrap();
        let mut kk_check = CooMatrix::new(nu, np, nnz_wcs, Sym::No).unwrap();

        // spectral derivative matrices
        let dd1x = self.interp_x.get_dd1().unwrap();
        let dd2x = self.interp_x.get_dd2().unwrap();

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
                        val += -self.mkx * dd1x.get(m, n) * dr_dx; // -1 due to the normal pointing left
                    }
                    if m == nx - 1 {
                        // Xmax
                        val += self.mkx * dd1x.get(m, n) * dr_dx;
                    }
                    self.put_val(&mut kk_bar, &mut kk_check, m, n, val);
                }
            } else {
                for n in 0..nx {
                    let mut val = self.mkx * dd2x.get(m, n) * cx;
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
        let nnz_wcs = nx * nx; // worst-case scenario
        let mut mm = CooMatrix::new(ndim, ndim, nnz_wcs + extra_nnz + 2 * nlag, Sym::No).unwrap();

        // spectral derivative matrices
        let dd1x = self.interp_x.get_dd1().unwrap();
        let dd2x = self.interp_x.get_dd2().unwrap();

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
                        val += -self.mkx * dd1x.get(m, n) * dr_dx; // -1 due to the normal pointing left
                    }
                    if m == nx - 1 {
                        // Xmax
                        val += self.mkx * dd1x.get(m, n) * dr_dx;
                    }
                    mm.put(m, n, val).unwrap();
                }
            } else {
                for n in 0..nx {
                    let mut val = self.mkx * dd2x.get(m, n) * cx;
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
    /// The `source` function calculates f(x).
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
    /// The `source` function calculates f(x).
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
mod tests {}
