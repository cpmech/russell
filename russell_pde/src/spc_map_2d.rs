use crate::util::delta;
use crate::{EquationHandler, EssentialBcs2d, Grid2d, Metrics, NaturalBcs2d, StrError, Transfinite2d};
use russell_lab::{vec_copy_scaled, vec_inner, vec_norm, InterpLagrange, Norm, Vector};
use russell_sparse::{CooMatrix, Genie, LinSolver, Sym};

/// Implements the Spectral Collocation method (SPC) in 2D with Transfinite Mapping for curvilinear coordinates
///
/// The SPC can be used to solve the following problem:
///
/// ```text
/// -k ∇²ϕ = source(x,y)
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

    x: Vector,
    dx_dr: Vector,
    dx_ds: Vector,
    d2x_dr2: Vector,
    d2x_ds2: Vector,
    d2x_drs: Vector,
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
    /// * `nr` -- number of grid points along r (for the Chebyshev-Gauss-Lobatto grid)
    /// * `ns` -- number of grid points along s (for the Chebyshev-Gauss-Lobatto grid)
    /// * `ebcs` -- the essential boundary conditions handler
    /// * `nbcs` -- the natural boundary conditions handler
    /// * `k` -- the diffusion coefficient
    /// * `map` -- the transfinite mapping from reference to physical domain
    pub fn new(
        nr: usize,
        ns: usize,
        mut ebcs: EssentialBcs2d<'a>,
        mut nbcs: NaturalBcs2d<'a>,
        k: f64,
        map: Transfinite2d,
    ) -> Result<Self, StrError> {
        // allocate the Chebyshev-Gauss-Lobatto grid
        let grid = Grid2d::new_chebyshev_gauss_lobatto(nr, ns)?;

        // build the boundary conditions data
        ebcs.build(&grid);
        nbcs.build(&grid);

        // check that the EBCs is not periodic
        if ebcs.is_periodic_along_x() || ebcs.is_periodic_along_y() {
            return Err("essential BCs cannot be periodic");
        }

        // allocate equations handler
        let neq = grid.size();
        let mut equations = EquationHandler::new(neq);
        equations.recompute(&ebcs.get_nodes());

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
        })
    }

    /// Solves problem
    pub fn solve<F>(&mut self, source: F) -> Result<Vector, StrError>
    where
        F: Fn(f64, f64) -> f64,
    {
        // assemble the coefficient matrix and the lhs and rhs vectors
        let (kk_bar, kk_check) = self.get_matrices();
        let (mut a_bar, a_check, mut f_bar) = self.get_vectors(source);

        // initialize the right-hand side
        kk_check.mat_vec_mul_update(&mut f_bar, -1.0, &a_check).unwrap(); // f̄ -= Ǩ ǎ

        // solve the linear system
        let mut solver = LinSolver::new(Genie::Umfpack)?;
        solver.actual.factorize(&kk_bar, None)?;
        solver.actual.solve(&mut a_bar, &f_bar, false)?;

        // results
        Ok(self.get_joined_vector(&a_bar, &a_check))
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

    /// Access the transfinite mapping
    pub fn get_map(&mut self) -> &mut Transfinite2d {
        &mut self.map
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

        // add all terms to the coefficient matrix
        for i in 0..nr {
            for j in 0..ns {
                let m = i + j * nr;
                if !self.equations.is_prescribed(m) {
                    self.calculate_metrics(m);
                    let g11 = self.metrics.gg_mat.get(0, 0);
                    let g22 = self.metrics.gg_mat.get(1, 1);
                    let g12 = self.metrics.gg_mat.get(0, 1);
                    let ll1 = self.metrics.ell_coefficient_for_laplacian(0);
                    let ll2 = self.metrics.ell_coefficient_for_laplacian(1);
                    let has_nbc = if i == 0 || i == nr - 1 || j == 0 || j == ns - 1 {
                        self.nbcs.has_value(m)
                    } else {
                        false
                    };
                    if has_nbc {
                        let (ex, ey) = self.grid.outward_unit_normal(m);
                        let mut un = Vector::new(2);
                        if ey == 0.0 {
                            vec_copy_scaled(&mut un, ex, &self.metrics.g_ctr[0]).unwrap();
                        } else {
                            vec_copy_scaled(&mut un, ey, &self.metrics.g_ctr[1]).unwrap();
                        }
                        let norm_u = vec_norm(&un, Norm::Euc);
                        un[0] /= norm_u;
                        un[1] /= norm_u;
                        let alpha = vec_inner(&un, &self.metrics.g_ctr[0]);
                        let beta = vec_inner(&un, &self.metrics.g_ctr[1]);
                        // println!( "m = {}, N={:?}, alpha = {:.8}, beta = {:.8}", m, un.as_data(), alpha, beta);
                        for k in 0..nr {
                            for l in 0..ns {
                                let n = k + l * nr;
                                let val = self.d1r(i, k) * delta(j, l) * alpha + delta(i, k) * self.d1s(j, l) * beta;
                                self.put_val(&mut kk_bar, &mut kk_check, m, n, self.mk * val);
                            }
                        }
                    } else {
                        for k in 0..nr {
                            for l in 0..ns {
                                let n = k + l * nr;
                                let val = 0.0
                                    + self.d2r(i, k) * delta(j, l) * g11
                                    + delta(i, k) * self.d2s(j, l) * g22
                                    + self.d1r(i, k) * self.d1s(j, l) * 2.0 * g12
                                    - self.d1r(i, k) * delta(j, l) * ll1
                                    - delta(i, k) * self.d1s(j, l) * ll2;
                                self.put_val(&mut kk_bar, &mut kk_check, m, n, self.mk * val);
                            }
                        }
                    }
                }
            }
        }

        // done
        (kk_bar, kk_check)
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
        for m in self.nbcs.get_nodes() {
            if !self.equations.is_prescribed(m) {
                let iu = self.equations.iu(m);
                let (x, y) = self.grid.coord(m);
                let q_bar = self.nbcs.get_value(m, x, y);
                f_bar[iu] = q_bar;
            }
        }
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            let (r, s) = self.grid.coord(m);
            self.map.point(&mut self.x, r, s);
            let val = self.ebcs.get_value(m, self.x[0], self.x[1]);
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
    use crate::{EssentialBcs2d, NaturalBcs2d, TransfiniteSamples};
    use russell_lab::mat_approx_eq;

    #[test]
    fn get_matrices_works_1() {
        let map = TransfiniteSamples::quadrilateral_2d(&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]);
        let mut ebcs = EssentialBcs2d::new();
        let nbcs = NaturalBcs2d::new();
        ebcs.set_homogeneous();
        let mut spectral = SpcMap2d::new(5, 5, ebcs, nbcs, 1.0, map).unwrap();
        let (kk_bar, kk_check) = spectral.get_matrices();
        let kk_bar_dense = kk_bar.as_dense();
        // println!("{:.2}", kk_bar_dense);

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
    }
}
