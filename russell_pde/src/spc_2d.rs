use crate::{EquationHandler, EssentialBcs2d, Grid2d, StrError};
use russell_lab::{InterpLagrange, Vector};
use russell_sparse::{CooMatrix, Sym};

/// Implements the Spectral Collocation method (SPC) in 2D
///
/// The SPC can be used to solve the following problem:
///
/// ```text
/// ∂²ϕ   ∂²ϕ
/// ——— + ——— = source(x, y)
/// ∂x²   ∂y²
/// ```
///
/// with essential (EBC) and natural (NBC) boundary conditions.
///
/// The spectral collocation method approximates the Laplacian at the grid points (xᵢ, yⱼ) using:
///
/// ```text
///              ∂²ϕ│              ∂²ϕ│
/// (∇²ϕ)ᵢⱼ = kx ———│        +  ky ———│        = ∑ₖ ∑ₗ (kx D⁽²⁾ᵢₖ δⱼₗ + ky δᵢₖ D̄⁽²⁾ⱼₗ) ϕₖₗ
///              ∂x²│(xᵢ,xⱼ)       ∂y²│(xᵢ,xⱼ)
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
pub struct Spc2d<'a> {
    /// Defines the 2D grid
    grid: Grid2d,

    /// Holds a reference to the essential boundary conditions handler
    ebcs: EssentialBcs2d<'a>,

    /// Diffusion coefficient along x
    kx: f64,

    /// Diffusion coefficient along y
    ky: f64,

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
    /// **Important**:
    ///
    /// 1. Only Chebyshev-Gauss-Lobatto grids are allowed.
    /// 2. All sides must have essential BCs, i.e., Neumann BCs are **not** allowed.
    ///
    /// # Arguments
    ///
    /// * `grid` -- the 2D grid with Chebyshev-Gauss-Lobatto points
    /// * `ebcs` -- the essential boundary conditions handler
    /// * `kx` -- the diffusion coefficient along x
    /// * `ky` -- the diffusion coefficient along y
    pub fn new(grid: Grid2d, ebcs: EssentialBcs2d<'a>, kx: f64, ky: f64) -> Result<Self, StrError> {
        // check grid
        if !grid.is_chebyshev_gauss_lobatto() {
            return Err("grid must use Chebyshev-Gauss-Lobatto points");
        }

        // check that the EBCs is not periodic
        if ebcs.is_periodic_along_x() || ebcs.is_periodic_along_y() {
            return Err("essential BCs cannot be periodic");
        }

        // check that all sides have essential BCs
        let (nodes_xmin, nodes_xmax, nodes_ymin, nodes_ymax) = grid.boundary_nodes();
        for m in nodes_xmin {
            if !ebcs.has_prescribed_value(*m) {
                return Err("essential BCs must be prescribed along x-min side");
            }
        }
        for m in nodes_xmax {
            if !ebcs.has_prescribed_value(*m) {
                return Err("essential BCs must be prescribed along x-max side");
            }
        }
        for m in nodes_ymin {
            if !ebcs.has_prescribed_value(*m) {
                return Err("essential BCs must be prescribed along y-min side");
            }
        }
        for m in nodes_ymax {
            if !ebcs.has_prescribed_value(*m) {
                return Err("essential BCs must be prescribed along y-max side");
            }
        }

        // allocate equations handler
        let neq = grid.size();
        let mut equations = EquationHandler::new(neq);
        equations.recompute(&ebcs.get_p_list());

        // polynomial degrees
        let nn_x = grid.nx() - 1;
        let nn_y = grid.ny() - 1;

        // interpolators
        let mut interp_x = InterpLagrange::new(nn_x, None)?;
        let mut interp_y = InterpLagrange::new(nn_y, None)?;
        interp_x.calc_dd2_matrix();
        interp_y.calc_dd2_matrix();

        // done
        Ok(Spc2d {
            grid,
            ebcs,
            kx,
            ky,
            equations,
            interp_x,
            interp_y,
        })
    }

    /// Returns the dimensions for the system partitioning strategy (SPS)
    ///
    /// Returns `(nu, np)` where:
    ///
    /// * `nu` is the number of unknowns
    /// * `np` is the number of prescribed values
    pub fn get_dims(&self) -> (usize, usize) {
        let nu = self.equations.nu();
        let np = self.equations.np();
        (nu, np)
    }

    /// Access the equation numbering handler
    pub fn get_equations(&self) -> &EquationHandler {
        &self.equations
    }

    /// Returns the coefficient matrices
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
    pub fn get_matrices(&self) -> (CooMatrix, CooMatrix) {
        // allocate matrices
        let nu = self.equations.nu();
        let np = self.equations.np();
        let nx = self.grid.nx();
        let ny = self.grid.ny();
        let nnz_wcs = nx * nx * ny * ny; // worst-case scenario
        let mut kk_bar = CooMatrix::new(nu, nu, nnz_wcs, Sym::No).unwrap();
        let mut kk_check = CooMatrix::new(nu, np, nnz_wcs, Sym::No).unwrap();

        // spectral derivative matrices
        let dd2x = self.interp_x.get_dd2().unwrap();
        let dd2y = self.interp_y.get_dd2().unwrap();

        // scaling coefficients due to domain mapping (from [-1,1]×[-1,1] to [xmin,xmax]×[ymin,ymax])
        let dr_dx = 2.0 / (self.grid.xmax() - self.grid.xmin());
        let ds_dy = 2.0 / (self.grid.ymax() - self.grid.ymin());
        let cx = dr_dx * dr_dx;
        let cy = ds_dy * ds_dy;

        // noting that:
        // (∇²ϕ)ᵢⱼ := kx ∂²ϕ/∂x²|(xᵢ,xⱼ) + ky ∂²ϕ/∂y²|(xᵢ,yⱼ) = ∑ₖ ∑ₗ (kx D⁽²⁾ᵢₖ δⱼₗ + ky δᵢₖ D̄⁽²⁾ⱼₗ) ϕₖₗ
        // we need to build K such that:
        // K(i,j,k,l) = ∑ₖ ∑ₗ (kx D⁽²⁾ᵢₖ δⱼₗ + ky δᵢₖ D̄⁽²⁾ⱼₗ)
        // By using the `put` command, we are effectively summing the contributions to K

        for i in 0..nx {
            for j in 0..ny {
                let m = i + j * nx;
                if !self.equations.is_prescribed(m) {
                    // unknown rows
                    let row = self.equations.iu(m);
                    for k in 0..nx {
                        for l in 0..ny {
                            let n = k + l * nx;
                            let mut val = 0.0;
                            if j == l {
                                val += self.kx * dd2x.get(i, k) * cx;
                            }
                            if i == k {
                                val += self.ky * dd2y.get(j, l) * cy;
                            }
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
                    }
                }
            }
        }

        // done
        (kk_bar, kk_check)
    }

    /// Returns the vectors for the solution of the system of equations
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
    pub fn get_vectors<F>(&self, source: F) -> (Vector, Vector, Vector)
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
            let val = self.ebcs.get_prescribed_value(m, x, y);
            a_check[ip] = val;
        });
        (a_bar, a_check, f_bar)
    }

    /// Joins the a-bar and a-check vectors
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
    pub fn get_joined_vector(&self, a_bar: &Vector, a_check: &Vector) -> Vector {
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Spc2d;
    use crate::{EssentialBcs2d, Grid2d};
    use russell_lab::mat_approx_eq;

    #[test]
    fn get_matrices_works_1() {
        let grid = Grid2d::new_chebyshev_gauss_lobatto(-1.0, 1.0, -1.0, 1.0, 5, 5).unwrap();
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_homogeneous(&grid);
        let spec = Spc2d::new(grid, ebcs, 1.0, 1.0).unwrap();
        let (kk_bar, kk_check) = spec.get_matrices();
        let kk_bar_dense = kk_bar.as_dense();
        // println!("{:.2}", kk_bar_dense);

        let ___ = 0.0;
        #[rustfmt::skip]
        let correct_kk_bar = &[
            [-28.0,   6.0,  -2.0,   6.0,   ___,   ___,  -2.0,   ___,   ___],
            [  4.0, -20.0,   4.0,   ___,   6.0,   ___,   ___,  -2.0,   ___],
            [ -2.0,   6.0, -28.0,   ___,   ___,   6.0,   ___,   ___,  -2.0],
            [  4.0,   ___,   ___, -20.0,   6.0,  -2.0,   4.0,   ___,   ___],
            [  ___,   4.0,   ___,   4.0, -12.0,   4.0,   ___,   4.0,   ___],
            [  ___,   ___,   4.0,  -2.0,   6.0, -20.0,   ___,   ___,   4.0],
            [ -2.0,   ___,   ___,   6.0,   ___,   ___, -28.0,   6.0,  -2.0],
            [  ___,  -2.0,   ___,   ___,   6.0,   ___,   4.0, -20.0,   4.0],
            [  ___,   ___,  -2.0,   ___,   ___,   6.0,  -2.0,   6.0, -28.0],
        ];
        mat_approx_eq(&kk_bar_dense, correct_kk_bar, 1e-14);

        #[rustfmt::skip]
        let correct_kk_check = &[
            [0.0, 9.242640687119286, 0.0, 0.0, 0.0, 9.242640687119286, 0.7573593128807143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7573593128807143, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.242640687119286, 0.0, 0.0, -0.9999999999999998, -0.9999999999999993, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7573593128807143, 0.0, 0.0],
            [0.0, 0.0, 0.0, 9.242640687119286, 0.0, 0.757359312880714, 9.242640687119286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7573593128807143, 0.0],
            [0.0, -0.9999999999999998, 0.0, 0.0, 0.0, 0.0, 0.0, 9.242640687119286, 0.7573593128807143, 0.0, 0.0, 0.0, -0.9999999999999993, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.9999999999999998, 0.0, 0.0, 0.0, 0.0, -0.9999999999999998, -0.9999999999999993, 0.0, 0.0, 0.0, 0.0, -0.9999999999999993, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.9999999999999998, 0.0, 0.0, 0.0, 0.757359312880714, 9.242640687119286, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9999999999999993, 0.0],
            [0.0, 0.757359312880714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.242640687119286, 0.7573593128807143, 0.0, 9.242640687119286, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.757359312880714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9999999999999998, -0.9999999999999993, 0.0, 0.0, 9.242640687119286, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.757359312880714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.757359312880714, 9.242640687119286, 0.0, 0.0, 0.0, 9.242640687119286, 0.0],
        ];
        let kk_check_dense = kk_check.as_dense();
        // println!("{:.3}", kk_check_dense);
        mat_approx_eq(&kk_check_dense, correct_kk_check, 1e-14);
    }
}
