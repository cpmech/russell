use crate::util::validate_bcs_1d;
use crate::{EquationHandler, EssentialBcs1d, Grid1d, NaturalBcs1d, StrError};
use russell_lab::Vector;
use russell_sparse::{CooMatrix, Genie, LinSolver, Sym};

// constants for clarity/convenience
const CUR: usize = 0; // current node
const LEF: usize = 1; // left node
const RIG: usize = 2; // right node
const INI_X: usize = 0;

/// Implements the Finite Difference method (FDM) in 1D
///
/// The FDM can be used to solve the following problem (Poisson or Helmholtz equation):
///
/// ```text
/// Poisson:
///
///     ∂²ϕ
/// -kx ——— + α ϕ = source(x)
///     ∂x²
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
/// ϕₘ are the discrete counterpart of ϕ(x) over the (nx) grid.
///
/// The FDM stencil uses the "molecule" {α, β, β} such that:
///
/// ```text
/// α ϕᵢ + β ϕᵢ₋₁ + β ϕᵢ₊₁ = sᵢ
/// ```
///
/// # Natural boundary conditions (NBC)
///
/// In 1D, the flux vector reduces to `w = [wx, 0]ᵀ`, where
///
/// ```text
/// wx = -kx ∂ϕ/∂x
/// ```
///
/// The normal vectors at the boundaries are illustrated below:
///
/// ```text
/// @ Xmin:                                     @ Xmax:
///      ┌    ┐   ┌    ┐                             ┌    ┐   ┌    ┐
///      │ wx │   │ -1 │    ┌──────────────┐         │ wx │   │  1 │
/// wₙ = │    │ · │    │  ← │              │ →  wₙ = │    │ · │    │
///      │  0 │   │  0 │    └──────────────┘         │  0 │   │  0 │
///      └    ┘   └    ┘                             └    ┘   └    ┘
///    = kx ∂ϕ/∂x                                  = -kx ∂ϕ/∂x
/// ```
///
/// The natural boundary conditions (NBC) are set by modifying the right-hand side vector.
///
/// The FDM stencil at the left side (Xmin) is:
///
/// ```text
/// α ϕ_{0} + β ϕ_{-1} + β ϕ_{1} = s_{0}
/// ```
///
/// where `α = 2kx/Δx²` and `β = -kx/Δx²`.
///
/// The central difference formula for the gradient @ Xmin is:
///
/// ```text
/// ∂ϕ │    ϕ_{1} - ϕ_{-1}
/// —— │  ≈ —————————————— := g
/// ∂x │0        2 Δx
/// ```
///
/// Thus, `ϕ_{-1} = ϕ_{1} - 2 g Δx`.
///
/// By substituting `ϕ_{-1}` in the FDM stencil, we get:
///
/// ```text
/// α ϕ_{0} + β (ϕ_{1} - 2 g Δx) + β ϕ_{1} = s_{0}
/// α ϕ_{0} + 2 β ϕ_{1} = s_{0} + 2 β g Δx
/// α ϕ_{0} + 2 β ϕ_{1} = s_{0} - 2 (kx/Δx²) g Δx
/// α ϕ_{0} + 2 β ϕ_{1} = s_{0} - 2 (kx g) / Δx
/// α ϕ_{0} + 2 β ϕ_{1} = s_{0} - 2 wₙ / Δx
///                            └───────────┘
///                              extra term
/// ```
///
/// Where `wₙ := kx g = q̄` at the left side (Xmin).
///
/// The FDM stencil at the right side (Xmax) is:
///
/// ```text
/// α ϕ_{nx-1} + β ϕ_{nx-2} + β ϕ_{nx} = s_{nx-1}
/// ```
///
/// Analogously, at the right side (Xmax) the gradient is:
///
/// ```text
/// ∂ϕ │         ϕ_{nx} - ϕ_{nx-2}
/// —— │       ≈ ————————————————— := g
/// ∂x │{nx-1}         2 Δx
/// ```
///
/// Thus, `ϕ_{nx} = ϕ_{nx-2} + 2 g Δx`.
///
/// By substituting `ϕ_{nx}` in the FDM stencil, we get:
///
/// ```text
/// α ϕ_{nx-1} + β ϕ_{nx-2} + β (ϕ_{nx-2} + 2 g Δx) = s_{nx-1}
/// α ϕ_{nx-1} + 2 β ϕ_{nx-2} = s_{nx-1} - 2 β g Δx
/// α ϕ_{nx-1} + 2 β ϕ_{nx-2} = s_{nx-1} - 2 (-kx/Δx²) g Δx
/// α ϕ_{nx-1} + 2 β ϕ_{nx-2} = s_{nx-1} - 2 (-kx g) / Δx
/// α ϕ_{nx-1} + 2 β ϕ_{nx-2} = s_{nx-1} - 2 wₙ / Δx
///                                     └───────────┘
///                                       extra term
/// ```
///
/// where `wₙ := -kx g = q̄` at the right side (Xmax).
pub struct Fdm1d<'a> {
    /// Defines the 1D grid
    grid: Grid1d,

    /// Holds the essential boundary conditions handler
    ebcs: EssentialBcs1d<'a>,

    /// Holds the natural boundary conditions handler
    nbcs: NaturalBcs1d<'a>,

    /// Tool to handle the equation numbers such as unknowns prescribed
    equations: EquationHandler,

    /// Holds the FDM coefficients (α, β, β) corresponding to (CUR, LEF, RIG)
    ///
    /// These coefficients are applied over the "bandwidth" of the coefficient matrix
    molecule: Vec<f64>,

    /// Grid spacing (uniform grid)
    dx: f64,
}

impl<'a> Fdm1d<'a> {
    /// Allocates a new instance
    ///
    /// # Arguments
    ///
    /// * `grid` -- the 1D grid
    /// * `ebcs` -- the essential boundary conditions handler
    /// * `nbcs` -- the natural boundary conditions handler
    /// * `kx` -- the diffusion coefficient along x
    ///
    /// **Note:** Zero Natural (Neumann) boundary conditions are assumed for boundaries with no explicit condition set.
    pub fn new(grid: Grid1d, ebcs: EssentialBcs1d<'a>, mut nbcs: NaturalBcs1d<'a>, kx: f64) -> Result<Self, StrError> {
        // check grid
        let dx = match grid.get_dx() {
            Some(dx) => dx,
            None => return Err("grid must have uniform spacing"),
        };

        // build the boundary conditions data
        nbcs.build(&grid);
        validate_bcs_1d(&ebcs, &nbcs)?;

        // allocate equations handler
        let neq = grid.nx();
        let mut equations = EquationHandler::new(neq);

        // set prescribed equations
        let mut p_list = Vec::new();
        for side in 0..2 {
            if ebcs.sides[side] {
                let m = if side == 0 { 0 } else { grid.nx() - 1 };
                p_list.push(m);
            }
        }
        equations.recompute(&p_list);

        // auxiliary variables
        let dx2 = dx * dx;
        let alpha = 2.0 * kx / dx2;
        let beta = -kx / dx2;

        // done
        Ok(Fdm1d {
            grid,
            ebcs,
            nbcs,
            equations,
            molecule: vec![alpha, beta, beta],
            dx,
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
    ///
    /// For the convection problem, the equation can be rearranged as:
    ///
    /// ```text
    ///     ∂²ϕ
    /// -kx ——— + (ϕ - ϕ∞) α = f(x)
    ///     ∂x²
    ///
    ///     ∂²ϕ
    /// -kx ——— + α ϕ = f(x) + α ϕ∞
    ///     ∂x²        └───────────┘
    ///                  source(x)
    /// ```
    pub fn solve_sps<F>(&self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64) -> f64,
    {
        // assemble the coefficient matrix and the lhs and rhs vectors
        let (kk_bar, kk_check) = self.get_matrices_sps(alpha, 0);
        let (mut a_bar, a_check, mut f_bar) = self.get_vectors_sps(source);
        let kk_check = kk_check.unwrap();

        // update the right-hand side with the prescribed values
        kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check)?; // f̄ -= Ǩ ǎ

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
    /// For the convection problem, the equation can be rearranged as:
    ///
    /// ```text
    ///     ∂²ϕ
    /// -kx ——— + (ϕ - ϕ∞) α = f(x)
    ///     ∂x²
    ///
    ///     ∂²ϕ
    /// -kx ——— + α ϕ = f(x) + α ϕ∞
    ///     ∂x²        └───────────┘
    ///                  source(x)
    /// ```
    pub fn solve_lmm<F>(&self, alpha: f64, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64) -> f64,
    {
        // assemble the coefficient matrix and the lhs and rhs vectors
        let neq = self.equations.neq();
        let extra_nnz = neq; // diagonal entries due to ϕ β
        let (mm, _) = self.get_matrices_lmm(alpha, extra_nnz, false);
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

    /// Access the grid
    pub fn get_grid(&self) -> &Grid1d {
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
    /// * `alpha` -- Helmholtz coefficient (α). Set to 0.0 for the Poisson equation
    /// * `extra_nnz` -- extra non-zeros to allocate in the K-bar matrix
    ///
    /// Note that the `K` (K-check) matrix is only available if there are essential boundary conditions.
    pub fn get_matrices_sps(&self, alpha: f64, extra_nnz: usize) -> (CooMatrix, Option<CooMatrix>) {
        let nu = self.equations.nu();
        let np = self.equations.np();
        let nnz_kk_bar = 3 * nu + extra_nnz; // 3 is the bandwidth
        let mut kk_bar = CooMatrix::new(nu, nu, nnz_kk_bar, Sym::No).unwrap();
        let mut kk_check = if np == 0 {
            // russell_sparse requires at least a 1x1 matrix with 1 non-zero entry
            CooMatrix::new(1, 1, 1, Sym::No).unwrap()
        } else {
            let nnz_kk_check = 2 * np; // 4 is the max number of neighbors (worst case)
            CooMatrix::new(nu, np, nnz_kk_check, Sym::No).unwrap()
        };
        self.equations.unknown().iter().for_each(|&m| {
            let iu = self.equations.iu(m);
            self.loop_over_bandwidth(m, |b, n| {
                let mut val = self.molecule[b];
                if m == n {
                    val += alpha; // Helmholtz term
                }
                if self.equations.is_prescribed(n) {
                    let jp = self.equations.ip(n);
                    kk_check.put(iu, jp, val).unwrap();
                } else {
                    let ju = self.equations.iu(n);
                    kk_bar.put(iu, ju, val).unwrap();
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
    /// * `alpha` -- Helmholtz coefficient (α). Set to 0.0 for the Poisson equation
    /// * `extra_nnz` -- extra non-zeros to allocate in the A matrix
    /// * `get_constraints_mat` -- whether to return the constraints matrix or not
    ///
    /// Note: this matrix is not symmetric because of the flipping (mirroring) strategy for boundary nodes.
    pub fn get_matrices_lmm(
        &self,
        alpha: f64,
        extra_nnz: usize,
        get_constraints_mat: bool,
    ) -> (CooMatrix, Option<CooMatrix>) {
        // build the augmented matrix
        let (neq, nlag, ndim) = self.get_dims_lmm();
        let nnz = 3 * neq + 2 * nlag + extra_nnz; // 3 is the bandwidth, 2*nlag is for C and Cᵀ
        let mut mm = CooMatrix::new(ndim, ndim, nnz, Sym::No).unwrap();
        for m in 0..neq {
            self.loop_over_bandwidth(m, |b, n| {
                let mut val = self.molecule[b];
                if m == n {
                    val += alpha;
                }
                mm.put(m, n, val).unwrap();
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
            let x = self.grid.coord(m);
            f_bar[iu] = source(x);
            if self.grid.is_xmin(m) {
                let wn = self.nbcs.functions[0](x);
                f_bar[iu] += -2.0 * wn / self.dx;
            }
            if self.grid.is_xmax(m) {
                let wn = self.nbcs.functions[1](x);
                f_bar[iu] += -2.0 * wn / self.dx;
            }
        });
        for side in 0..2 {
            if self.ebcs.sides[side] {
                let m = if side == 0 { 0 } else { self.grid.nx() - 1 };
                let ip = self.equations.ip(m);
                let x = self.grid.coord(m);
                let val = self.ebcs.functions[side](x);
                a_check[ip] = val;
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
        self.grid.for_each_coord(|m, x| {
            ff[m] = source(x);
            if self.grid.is_xmin(m) {
                let wn = self.nbcs.functions[0](x);
                ff[m] += -2.0 * wn / self.dx;
            }
            if self.grid.is_xmax(m) {
                let wn = self.nbcs.functions[1](x);
                ff[m] += -2.0 * wn / self.dx;
            }
        });
        for side in 0..2 {
            if self.ebcs.sides[side] {
                let m = if side == 0 { 0 } else { self.grid.nx() - 1 };
                let ip = self.equations.ip(m);
                let x = self.grid.coord(m);
                let val = self.ebcs.functions[side](x);
                ff[neq + ip] = val;
            }
        }
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
    /// * `callback` -- a function of `(m, x)` where `m` is the sequential point number,
    ///   and `x` is coordinate.
    pub fn for_each_coord<F>(&self, mut callback: F)
    where
        F: FnMut(usize, f64),
    {
        self.grid.for_each_coord(|m, x| {
            callback(m, x);
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
        let fin_x = self.grid.nx() - 1;

        // n indices of the non-zero values on the row m of the coefficient matrix
        // (mirror or swap the indices of boundary nodes, as appropriate)
        let mut nn = [0, 0, 0];
        nn[CUR] = m;
        if self.ebcs.periodic_along_x {
            nn[LEF] = if m != INI_X { m - 1 } else { m + fin_x };
            nn[RIG] = if m != fin_x { m + 1 } else { m - fin_x };
        } else {
            nn[LEF] = if m != INI_X { m - 1 } else { m + 1 };
            nn[RIG] = if m != fin_x { m + 1 } else { m - 1 };
        }

        // execute callback
        for b in 0..3 {
            callback(b, nn[b]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Fdm1d;
    use crate::{EssentialBcs1d, Grid1d, NaturalBcs1d, Side};

    const LEF: f64 = 1.0;
    const RIG: f64 = 2.0;

    #[test]
    fn new_captures_errors() {
        let grid = Grid1d::new(&[0.0, 0.1, 0.4]).unwrap();
        let ebcs = EssentialBcs1d::new();
        let nbcs = NaturalBcs1d::new();
        let fdm = Fdm1d::new(grid, ebcs, nbcs, 1.0);
        assert_eq!(fdm.err(), Some("grid must have uniform spacing"));
    }

    #[test]
    fn get_matrices_work() {
        //  0*  1   2   3
        // dx = 1.0
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let mut ebcs = EssentialBcs1d::new();
        let mut nbcs = NaturalBcs1d::new();
        const LEF: f64 = 1.0;
        let lef = |_| LEF;
        assert_eq!(lef(0.0), LEF);
        ebcs.set(Side::Xmin, lef); //  0
        nbcs.set(Side::Xmax, |_| 0.0);

        let fdm = Fdm1d::new(grid, ebcs, nbcs, 100.0).unwrap();
        let (kk, cc_mat) = fdm.get_matrices_sps(0.0, 0);
        let (aa, ee_mat) = fdm.get_matrices_lmm(0.0, 0, true);
        let cc = cc_mat.unwrap();
        let ee = ee_mat.unwrap();

        assert_eq!(fdm.get_dims_sps(), (3, 1));
        assert_eq!(fdm.get_dims_lmm(), (4, 1, 5));

        // The full matrix is:
        //      0*   1    2    3
        // ┌                     ┐
        // │  200 -200    .    . │  0*(p0)
        // │ -100  200 -100    . │  1→0
        // │    . -100  200 -100 │  2→1
        // │    .    . -200  200 │  3→2
        // └                     ┘
        //      0*   1    2    3

        // K =
        //      1    2    3
        // ┌                ┐
        // │  200 -100    . │  1→0
        // │ -100  200 -100 │  2→1
        // │    . -200  200 │  3→2
        // └                ┘
        //      1    2    3
        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌                ┐\n\
             │  200 -100    0 │\n\
             │ -100  200 -100 │\n\
             │    0 -200  200 │\n\
             └                ┘"
        );

        // C =
        //     0*
        // ┌     ┐
        // │ -100 │  1→0
        // │    . │  2→1
        // │    . │  3→2
        // └      ┘
        //     0*
        assert_eq!(
            format!("{}", cc.as_dense()),
            "┌      ┐\n\
             │ -100 │\n\
             │    0 │\n\
             │    0 │\n\
             └      ┘"
        );

        // E =
        //      0*   1    2    3
        // ┌                     ┐
        // │    1    .    .    . │  0*
        // └                     ┘
        //      0*   1    2    3
        assert_eq!(
            format!("{}", ee.as_dense()),
            "┌         ┐\n\
             │ 1 0 0 0 │\n\
             └         ┘"
        );

        // A =
        //      0*   1    2    3    4w
        // ┌                          ┐
        // │  200 -200    .    .    1 │  0*
        // │ -100  200 -100    .    . │  1
        // │    . -100  200 -100    . │  2
        // │    .    . -200  200    . │  3
        // │    1    .    .    .    . │  4w
        // └                          ┘
        //      0*   1    2    3    4w
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                          ┐\n\
             │  200 -200    0    0    1 │\n\
             │ -100  200 -100    0    0 │\n\
             │    0 -100  200 -100    0 │\n\
             │    0    0 -200  200    0 │\n\
             │    1    0    0    0    0 │\n\
             └                          ┘"
        );
    }

    #[test]
    fn get_matrices_homogeneous_bcs_work() {
        //  1 |  0*  1   2   3* |  2
        // dx = 1.0
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let mut ebcs = EssentialBcs1d::new();
        let nbcs = NaturalBcs1d::new();
        ebcs.set_homogeneous();

        let fdm = Fdm1d::new(grid, ebcs, nbcs, 1.0).unwrap();
        let (kk, cc_mat) = fdm.get_matrices_sps(0.0, 0);
        let (aa, ee_mat) = fdm.get_matrices_lmm(0.0, 0, true);
        let cc = cc_mat.unwrap();
        let ee = ee_mat.unwrap();

        assert_eq!(fdm.get_dims_sps(), (2, 2));
        assert_eq!(fdm.get_dims_lmm(), (4, 2, 6));

        // The full matrix is:
        //    0* 1  2  3*
        // ┌              ┐
        // │  2 -2  .  .  │  0*
        // │ -1  2 -1  .  │  1
        // │  . -1  2 -1  │  2
        // │  .  . -2  2  │  3*
        // └              ┘
        //    0* 1  2  3*

        // K =
        //    1  2
        // ┌       ┐
        // │  2 -1 │  1
        // │ -1  2 │  2
        // └       ┘
        //    1  2
        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌       ┐\n\
             │  2 -1 │\n\
             │ -1  2 │\n\
             └       ┘"
        );

        // C =
        //    0* 3*
        // ┌        ┐
        // │  1  .  │  1
        // │  .  1  │  2
        // └        ┘
        //    0* 3*
        assert_eq!(
            format!("{}", cc.as_dense()),
            "┌       ┐\n\
             │ -1  0 │\n\
             │  0 -1 │\n\
             └       ┘"
        );

        // E =
        //    0* 1  2  3*
        // ┌             ┐
        // │  1  .  .  . │  0*
        // │  .  .  .  1 │  3*
        // └             ┘
        assert_eq!(
            format!("{}", ee.as_dense()),
            "┌         ┐\n\
             │ 1 0 0 0 │\n\
             │ 0 0 0 1 │\n\
             └         ┘"
        );

        // A =
        //    0* 1  2  3* 4w 5w
        // ┌                   ┐
        // │  2 -2  .  .  1  . │  0*
        // │ -1  2 -1  .  .  . │  1
        // │  . -1  2 -1  .  . │  2
        // │  .  . -2  2  .  1 │  3*
        // │  1  .  .  .  .  . │  3*
        // │  .  .  .  1  .  . │  3*
        // └                   ┘
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                   ┐\n\
             │  2 -2  0  0  1  0 │\n\
             │ -1  2 -1  0  0  0 │\n\
             │  0 -1  2 -1  0  0 │\n\
             │  0  0 -2  2  0  1 │\n\
             │  1  0  0  0  0  0 │\n\
             │  0  0  0  1  0  0 │\n\
             └                   ┘"
        );
    }

    #[test]
    fn get_matrices_periodic_bcs_work() {
        //  3 | 0  1  2  3 | 0
        // dx = 1.0
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let mut ebcs = EssentialBcs1d::new();
        let nbcs = NaturalBcs1d::new();
        ebcs.set_periodic(true);

        let fdm = Fdm1d::new(grid, ebcs, nbcs, 1.0).unwrap();
        let (kk, cc_mat) = fdm.get_matrices_sps(0.0, 0);
        let (aa, ee_mat) = fdm.get_matrices_lmm(0.0, 0, true);
        assert!(cc_mat.is_none());
        assert!(ee_mat.is_none());

        assert_eq!(fdm.get_dims_sps(), (4, 0));
        assert_eq!(fdm.get_dims_lmm(), (4, 0, 4));

        // K = A =
        //    0  1  2  3
        // ┌             ┐
        // │  2 -1  . -1 │  0
        // │ -1  2 -1  . │  1
        // │  . -1  2 -1 │  2
        // │ -1  . -1  2 │  3
        // └             ┘
        //    0  1  2  3

        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌             ┐\n\
             │  2 -1  0 -1 │\n\
             │ -1  2 -1  0 │\n\
             │  0 -1  2 -1 │\n\
             │ -1  0 -1  2 │\n\
             └             ┘"
        );
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌             ┐\n\
             │  2 -1  0 -1 │\n\
             │ -1  2 -1  0 │\n\
             │  0 -1  2 -1 │\n\
             │ -1  0 -1  2 │\n\
             └             ┘"
        );
    }

    #[test]
    fn get_vectors_works() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 5).unwrap();
        let mut ebcs = EssentialBcs1d::new();
        let nbcs = NaturalBcs1d::new();

        //  0*  1   2   3   4*
        ebcs.set(Side::Xmin, |_| LEF);
        ebcs.set(Side::Xmax, |_| RIG);

        let fdm = Fdm1d::new(grid, ebcs, nbcs, 1.0).unwrap();

        let (a_bar, a_check, f_bar) = fdm.get_vectors_sps(|_| 100.0);
        assert_eq!(a_bar.dim(), 3); // nu
        assert_eq!(a_check.dim(), 2); // np
        assert_eq!(f_bar.dim(), 3); // nu
        assert_eq!(a_bar.as_data(), &[0.0, 0.0, 0.0]);
        assert_eq!(a_check.as_data(), &[LEF, RIG]);
        assert_eq!(f_bar.as_data(), &[100.0, 100.0, 100.0]);

        let a = fdm.get_joined_vector_sps(&a_bar, &a_check);
        assert_eq!(a.dim(), 5); // na
        assert_eq!(a.as_data(), &[LEF, 0.0, 0.0, 0.0, RIG]);

        let (aa, ff) = fdm.get_vectors_lmm(|_| 100.0);
        assert_eq!(aa.dim(), 5 + 2); // na + nw
        assert_eq!(ff.dim(), 5 + 2); // na + nw
        assert_eq!(aa.as_data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(ff.as_data(), &[100.0, 100.0, 100.0, 100.0, 100.0, LEF, RIG]);
    }
}
