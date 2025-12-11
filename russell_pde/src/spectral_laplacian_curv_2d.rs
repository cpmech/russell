use crate::{EquationHandler, EssentialBcs2d, Grid2d, Metrics, StrError, Transfinite2d};
use russell_lab::{InterpLagrange, Vector};
use russell_sparse::{CooMatrix, Sym};

/// Kronecker delta function
fn delta(i: usize, j: usize) -> f64 {
    if i == j {
        1.0
    } else {
        0.0
    }
}

/// Approximates the Laplacian operator in 2D using the Spectral Collocation Method
/// with the Curvilinear Coordinates
///
/// Given the (continuum) scalar field ϕ(x, y) and its Laplacian
///
/// ```text
///              ∂²ϕ       ∂²ϕ        ∂²ϕ          ∂ϕ      ∂ϕ
/// L{ϕ} = ∇²ϕ = ——— g¹¹ + ——— g²² + ————— 2 g¹² - —— L¹ - —— L²
///              ∂r²       ∂s²       ∂r ∂s         ∂r      ∂r
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
pub struct SpectralLaplacianCurv2d<'a> {
    /// Defines the 2D grid on the reference domain [-1, 1] × [-1, 1]
    grid: Grid2d,

    /// Holds a reference to the essential boundary conditions handler
    ebcs: EssentialBcs2d<'a>,

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

    x: Vector,
    dx_dr: Vector,
    dx_ds: Vector,
    d2x_dr2: Vector,
    d2x_ds2: Vector,
    d2x_drs: Vector,
}

impl<'a> SpectralLaplacianCurv2d<'a> {
    /// Allocates a new instance
    ///
    /// **Important**:
    ///
    /// 1. All sides must have essential BCs, i.e., Neumann BCs are **not** allowed.
    ///
    /// # Arguments
    ///
    /// * `grid` -- the 2D grid with Chebyshev-Gauss-Lobatto points on the reference square [-1, 1] x [-1, 1]
    /// * `ebcs` -- the essential boundary conditions handler
    /// * `map` -- the transfinite mapping from reference to physical domain
    pub fn new(grid: Grid2d, ebcs: EssentialBcs2d<'a>, map: Transfinite2d) -> Result<Self, StrError> {
        // check grid
        if !grid.is_chebyshev_gauss_lobatto() {
            return Err("grid must use Chebyshev-Gauss-Lobatto points");
        }
        if grid.xmin() != -1.0 || grid.xmax() != 1.0 || grid.ymin() != -1.0 || grid.ymax() != 1.0 {
            return Err("grid must be defined on the reference square [-1, 1] x [-1, 1]");
        }

        // check that the EBCs is not periodic
        if ebcs.is_periodic_along_x() || ebcs.is_periodic_along_y() {
            return Err("essential BCs cannot be periodic");
        }

        // check that all sides have essential BCs
        let (nodes_rmin, nodes_rmax, nodes_smin, nodes_smax) = grid.boundary_nodes();
        for m in nodes_rmin {
            if !ebcs.has_prescribed_value(*m) {
                return Err("essential BCs must be prescribed along r-min side");
            }
        }
        for m in nodes_rmax {
            if !ebcs.has_prescribed_value(*m) {
                return Err("essential BCs must be prescribed along r-max side");
            }
        }
        for m in nodes_smin {
            if !ebcs.has_prescribed_value(*m) {
                return Err("essential BCs must be prescribed along s-min side");
            }
        }
        for m in nodes_smax {
            if !ebcs.has_prescribed_value(*m) {
                return Err("essential BCs must be prescribed along s-max side");
            }
        }

        // allocate equations handler
        let neq = grid.size();
        let mut equations = EquationHandler::new(neq);
        equations.recompute(&ebcs.get_p_list());

        // polynomial degrees
        let nn_r = grid.nx() - 1;
        let nn_s = grid.ny() - 1;

        // interpolators
        let mut interp_r = InterpLagrange::new(nn_r, None)?;
        let mut interp_s = InterpLagrange::new(nn_s, None)?;
        interp_r.calc_dd1_matrix();
        interp_s.calc_dd1_matrix();
        interp_r.calc_dd2_matrix();
        interp_s.calc_dd2_matrix();

        // metrics
        let metrics = Metrics::new(2, false);

        // done
        Ok(SpectralLaplacianCurv2d {
            grid,
            ebcs,
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
    pub fn get_matrices(&mut self) -> (CooMatrix, CooMatrix) {
        // allocate matrices
        let nu = self.equations.nu();
        let np = self.equations.np();
        let nr = self.grid.nx();
        let ns = self.grid.ny();
        let nnz_wcs = nr * nr * ns * ns; // worst-case scenario
        let mut kk_bar = CooMatrix::new(nu, nu, nnz_wcs, Sym::No).unwrap();
        let mut kk_check = CooMatrix::new(nu, np, nnz_wcs, Sym::No).unwrap();

        // spectral derivative matrices
        let dd1r = self.interp_r.get_dd1().unwrap();
        let dd1s = self.interp_s.get_dd1().unwrap();
        let dd2r = self.interp_r.get_dd2().unwrap();
        let dd2s = self.interp_s.get_dd2().unwrap();

        // add all terms to the coefficient matrix
        for i in 0..nr {
            for j in 0..ns {
                let m = i + j * nr;
                if !self.equations.is_prescribed(m) {
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
                    let g11 = self.metrics.gg_mat.get(0, 0);
                    let g22 = self.metrics.gg_mat.get(1, 1);
                    let g12 = self.metrics.gg_mat.get(0, 1);
                    let ll1 = self.metrics.ell_coefficient_for_laplacian(0);
                    let ll2 = self.metrics.ell_coefficient_for_laplacian(1);
                    // unknown rows
                    let row = self.equations.iu(m);
                    for k in 0..nr {
                        for l in 0..ns {
                            let n = k + l * nr;
                            let val = 0.0
                                + dd2r.get(i, k) * delta(j, l) * g11
                                + delta(i, k) * dd2s.get(j, l) * g22
                                + dd1r.get(i, k) * dd1s.get(j, l) * 2.0 * g12
                                - dd1r.get(i, k) * delta(j, l) * ll1
                                - delta(i, k) * dd1s.get(j, l) * ll2;
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
    pub fn get_vectors<F>(&mut self, source: F) -> (Vector, Vector, Vector)
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
            f_bar[iu] = source(self.x[0], self.x[1]);
        });
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            let (r, s) = self.grid.coord(m);
            self.map.point(&mut self.x, r, s);
            let val = self.ebcs.get_prescribed_value(m, self.x[0], self.x[1]);
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::SpectralLaplacianCurv2d;
    use crate::{EssentialBcs2d, Grid2d, TransfiniteSamples};
    use russell_lab::mat_approx_eq;

    #[test]
    fn get_matrices_works_1() {
        let grid = Grid2d::new_chebyshev_gauss_lobatto(-1.0, 1.0, -1.0, 1.0, 5, 5).unwrap();
        let map = TransfiniteSamples::quadrilateral_2d(&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]);
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_homogeneous(&grid);
        let mut spectral = SpectralLaplacianCurv2d::new(grid, ebcs, map).unwrap();
        let (kk_bar, kk_check) = spectral.get_matrices();
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
        mat_approx_eq(&kk_bar_dense, correct_kk_bar, 1e-13);

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
