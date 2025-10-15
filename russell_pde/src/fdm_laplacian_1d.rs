use crate::{EquationHandler, EssentialBcs1d, Grid1d, StrError};
use russell_lab::Vector;
use russell_sparse::{CooMatrix, Sym};

// constants for clarity/convenience
const CUR: usize = 0; // current node
const LEF: usize = 1; // left node
const RIG: usize = 2; // right node
const INI_X: usize = 0;

/// Implements the Finite Difference (FDM) Laplacian operator in 1D
///
/// Given the (continuum) scalar field ϕ(x) and its Laplacian
///
/// ```text
///           ∂²ϕ
/// L{ϕ} = kx ———
///           ∂x²
/// ```
///
/// we substitute the partial derivatives using central FDM over a linear grid.
/// The resulting discrete Laplacian is expressed by the coefficient matrix `M` and the vector `a`:
///
/// ```text
/// D{ϕₘ} = M a
/// ```
///
/// ϕₘ are the discrete counterpart of ϕ(x) over the (nx) grid.
///
/// Neglecting the essential boundary conditions (EBCs) for the moment, the discrete Laplacian operator `M`
/// is built with a three-point stencil. For a linear problem with the right-hand side represented by `r`,
/// the resulting linear system is:
///
/// ```text
/// M a = r
/// ```
///
/// However, the above linear system is singular, because the EBCs have not been applied yet. Two approaches
/// are possible to apply the EBCs: (1) use a reduced coefficient matrix `K` and a reduced vector `u` containing
/// only the unknown values; and (2) use the Lagrange multipliers method (LMM).
///
/// ## Approach 1: Reduced system
///
/// Consider the following partitioning of the vector `a` and the matrix `M`:
///
/// ```text
/// ┌       ┐ ┌   ┐   ┌   ┐
/// │ K   C │ │ u │   │ f │
/// │       │ │   │ = │   │
/// │ c   k │ │ p │   │ g │
/// └       ┘ └   ┘   └   ┘
///     M       a       r
/// ```
///
/// where `u` is a reduced vector containing only the unknown values (i.e., non-EBC nodes), and `p` is a reduced
/// vector containing only the prescribed values (i.e., EBC nodes). Likewise, `f` and `g` are the corresponding
/// reduced right-hand side vectors. The `K` matrix is the reduced discrete Laplacian operator and `C` is a
/// *correction* matrix. The `c` and `k` matrices are often not needed.
///
/// Thus, the linear system to be solved is:
///
/// ```text
/// K u + C p = f
/// ```
///
/// ## Approach 2: Lagrange multipliers method (LMM)
///
/// The LMM consists of augmenting the original linear system with additional equations:
///
/// ```text
/// ┌       ┐ ┌   ┐   ┌   ┐
/// │ M  Eᵀ │ │ a │   │ r │
/// │       │ │   │ = │   │
/// │ E  0  │ │ w │   │ ū │
/// └       ┘ └   ┘   └   ┘
///     A       h       b
/// ```
///
/// where `w` is the vector of Lagrange multipliers, `E` is the Lagrange matrix, and `ū` is the vector of
/// prescribed values at EBC nodes. The Lagrange matrix `E` has a row for each EBC (prescribed) node and a column
/// for every node. Each row in `E` has a single `1` at the column corresponding to the EBC node, and `0`s elsewhere.
pub struct FdmLaplacian1d<'a> {
    /// Defines the 1D grid
    grid: Grid1d,

    /// Holds a reference to the essential boundary conditions handler
    ebcs: EssentialBcs1d<'a>,

    /// Tool to handle the equation numbers such as unknowns prescribed
    equations: EquationHandler,

    /// Holds the FDM coefficients (α, β, β, γ, γ) corresponding to (CUR, LEF, RIG, BOT, TOP)
    ///
    /// These coefficients are applied over the "bandwidth" of the coefficient matrix
    molecule: Vec<f64>,
}

impl<'a> FdmLaplacian1d<'a> {
    /// Allocates a new instance
    ///
    /// # Arguments
    ///
    /// * `grid` -- the 1D grid
    /// * `ebcs` -- the essential boundary conditions handler
    /// * `kx` -- the diffusion coefficient along x
    pub fn new(grid: Grid1d, ebcs: EssentialBcs1d<'a>, kx: f64) -> Result<Self, StrError> {
        // check grid
        let dx = match grid.get_dx() {
            Some(dx) => dx,
            None => return Err("grid must have uniform spacing"),
        };

        // allocate equations handler
        let neq = grid.size();
        let mut equations = EquationHandler::new(neq);
        equations.recompute(&ebcs.get_p_list());

        // auxiliary variables
        let dx2 = dx * dx;
        let alpha = -2.0 * kx / dx2;
        let beta = kx / dx2;

        // done
        Ok(FdmLaplacian1d {
            grid,
            ebcs,
            equations,
            molecule: vec![alpha, beta, beta],
        })
    }

    /// Returns the dimension of the various vectors and matrices used in the linear system
    ///
    /// Returns `(nu, np, na, nw, nh)`.
    ///
    /// (1) the reduced system:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K   C │ │ u │   │ f │
    /// │       │ │   │ = │   │
    /// │ c   k │ │ p │   │ g │
    /// └       ┘ └   ┘   └   ┘
    ///     M       a       r
    /// ```
    ///
    /// (2) the Lagrange multipliers method (LMM):
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ M  Eᵀ │ │ a │   │ r │
    /// │       │ │   │ = │   │
    /// │ E  0  │ │ w │   │ ū │
    /// └       ┘ └   ┘   └   ┘
    ///     A       h       b
    /// ```
    ///
    /// The dimension of the various vectors and matrices are:
    ///
    /// * nu = num(unknown)
    /// * np = num(prescribed)
    /// * na = nu + np = size(grid)
    /// * nw = np
    /// * nh = na + nw
    pub fn get_info(&self) -> (usize, usize, usize, usize, usize) {
        let nu = self.equations.nu();
        let np = self.equations.np();
        let na = nu + np;
        let nw = np;
        let nh = na + nw;
        (nu, np, na, nw, nh)
    }

    /// Access the grid
    pub fn get_grid(&self) -> &Grid1d {
        &self.grid
    }

    /// Access the equation numbering handler
    pub fn get_equations(&self) -> &EquationHandler {
        &self.equations
    }

    /// Returns the reduced coefficient matrices
    ///
    /// Returns `K` and `C` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K   C │ │ u │   │ f │
    /// │       │ │   │ = │   │
    /// │ c   k │ │ p │   │ g │
    /// └       ┘ └   ┘   └   ┘
    ///     M       a       r
    /// ```
    ///
    /// Note that the `C` matrix is not available if there are no EBCs.
    pub fn get_kk_and_cc_matrices(&self, extra_nnz: usize, sym_kk: Sym) -> (CooMatrix, Option<CooMatrix>) {
        let nu = self.equations.nu();
        let np = self.equations.np();
        let nnz_kk = 3 * nu + extra_nnz; // 3 is the bandwidth
        let mut kk = CooMatrix::new(nu, nu, nnz_kk, sym_kk).unwrap();
        let mut cc = if np == 0 {
            // russell_sparse requires at least a 1x1 matrix with 1 non-zero entry
            CooMatrix::new(1, 1, 1, Sym::No).unwrap()
        } else {
            let nnz_cc = 2 * np; // 4 is the max number of neighbors (worst case)
            CooMatrix::new(nu, np, nnz_cc, Sym::No).unwrap()
        };
        self.equations.unknown().iter().for_each(|&m| {
            let iu = self.equations.iu(m);
            self.loop_over_bandwidth(m, |b, n| {
                if self.equations.is_prescribed(n) {
                    let jp = self.equations.ip(n);
                    cc.put(iu, jp, self.molecule[b]).unwrap();
                } else {
                    let ju = self.equations.iu(n);
                    kk.put(iu, ju, self.molecule[b]).unwrap();
                }
            });
        });
        if np == 0 {
            (kk, None)
        } else {
            (kk, Some(cc))
        }
    }

    /// Returns the matrix for the Lagrange multipliers method (LMM)
    ///
    /// Returns `A` and `E` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ M  Eᵀ │ │ a │   │ r │
    /// │       │ │   │ = │   │
    /// │ E  0  │ │ w │   │ ū │
    /// └       ┘ └   ┘   └   ┘
    ///     A       h       b
    /// ```
    ///
    /// Note: this matrix is not symmetric because of the flipping (mirroring) strategy for boundary nodes.
    pub fn get_aa_and_ee_matrices(&self, extra_nnz: usize, return_ee_mat: bool) -> (CooMatrix, Option<CooMatrix>) {
        // build the A matrix
        let na = self.equations.neq();
        let np = self.equations.np();
        let naa = na + np;
        let nnz = 3 * na + 2 * np + extra_nnz; // 3 is the bandwidth, 2*np is for E and Eᵀ
        let mut aa = CooMatrix::new(naa, naa, nnz, Sym::No).unwrap();
        for m in 0..na {
            self.loop_over_bandwidth(m, |b, n| {
                aa.put(m, n, self.molecule[b]).unwrap();
            });
        }

        // assemble E and Eᵀ into A
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            aa.put(na + ip, m, 1.0).unwrap(); // E
            aa.put(m, na + ip, 1.0).unwrap(); // Eᵀ
        });

        // build and return the E matrix, if requested and available
        if return_ee_mat && np > 0 {
            let mut ee = CooMatrix::new(np, na, np, Sym::No).unwrap();
            self.equations.prescribed().iter().for_each(|&m| {
                let ip = self.equations.ip(m);
                ee.put(ip, m, 1.0).unwrap(); // E
            });
            (aa, Some(ee))
        } else {
            (aa, None)
        }
    }

    /// Returns the vectors for the solution of the system of equations
    ///
    /// Returns `(u, p, f)` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K   C │ │ u │   │ f │
    /// │       │ │   │ = │   │
    /// │ c   k │ │ p │   │ g │
    /// └       ┘ └   ┘   └   ┘
    /// ```
    ///
    /// Note that:
    ///
    /// ```text
    /// nu = num(unknown)
    /// np = num(prescribed)
    /// nf = nu
    /// ```
    ///
    /// The `source` function calculates f(x).
    pub fn get_vectors<F>(&self, source: F) -> (Vector, Vector, Vector)
    where
        F: Fn(f64) -> f64,
    {
        let nu = self.equations.nu();
        let np = self.equations.np();
        let u = Vector::new(nu);
        let mut p = Vector::new(np);
        let mut f = Vector::new(nu);
        self.equations.unknown().iter().for_each(|&m| {
            let iu = self.equations.iu(m);
            let x = self.grid.coord(m);
            f[iu] = source(x);
        });
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            let x = self.grid.coord(m);
            let u_bar = self.ebcs.get_prescribed_value(m, x);
            p[ip] = u_bar;
        });
        (u, p, f)
    }

    /// Returns the composed solution vector from the unknown and prescribed vectors
    ///
    /// Returns `a = (u, p)` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ K   C │ │ u │   │ f │
    /// │       │ │   │ = │   │
    /// │ c   k │ │ p │   │ g │
    /// └       ┘ └   ┘   └   ┘
    /// ```
    pub fn get_composed_vector(&self, u: &Vector, p: &Vector) -> Vector {
        let na = self.equations.neq();
        let mut a = Vector::new(na);
        self.equations.unknown().iter().for_each(|&m| {
            let iu = self.equations.iu(m);
            a[m] = u[iu];
        });
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            a[m] = p[ip];
        });
        a
    }

    /// Returns the vectors for the solution of the system of equations using the Lagrange multipliers method (LMM)
    ///
    /// Returns `(h, b)` from:
    ///
    /// ```text
    /// ┌       ┐ ┌   ┐   ┌   ┐
    /// │ M  Eᵀ │ │ a │   │ r │
    /// │       │ │   │ = │   │
    /// │ E  0  │ │ w │   │ ū │
    /// └       ┘ └   ┘   └   ┘
    ///     A       h       b
    /// ```
    /// where a = (u, p) and w are the Lagrange multipliers
    ///
    /// Note that:
    ///
    /// ```text
    /// nu = num(unknown)
    /// np = num(prescribed)
    /// na = nu + np = size(grid)
    /// nw = np
    /// nh = na + nw
    /// ```
    ///
    /// The `source` function calculates r(x).
    pub fn get_vectors_lmm<F>(&self, source: F) -> (Vector, Vector)
    where
        F: Fn(f64) -> f64,
    {
        let na = self.equations.neq(); // dimension of a = (u, p)
        let nw = self.equations.np(); // number of Lagrange multipliers
        let nh = na + nw; // dimension of h = (u, p, w)
        let h = Vector::new(nh);
        let mut b = Vector::new(nh);
        self.grid.for_each_coord(|m, x| {
            b[m] = source(x);
        });
        self.equations.prescribed().iter().for_each(|&m| {
            let ip = self.equations.ip(m);
            let x = self.grid.coord(m);
            let u_bar = self.ebcs.get_prescribed_value(m, x);
            b[na + ip] = u_bar;
        });
        (h, b)
    }

    /// Executes a loop over one row of the full coefficient matrix M
    ///
    /// **Note**: The ghost boundary indices are flipped to avoid negative indices.
    /// This also allows the setting up of flux boundary conditions. Therefore, some
    /// column indices may appear repeated; e.g. due to the zero-flux boundaries.
    ///
    /// # Input
    ///
    /// * `m` -- the row of the coefficient matrix
    /// * `callback` -- a `function(n, val_mn)` where `n` is the column index and
    ///   `Mmn` is the m-n-element of the coefficient matrix
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
    pub fn loop_over_grid_points<F>(&self, mut callback: F)
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
        let fin_x = self.grid.node_xmax();

        // n indices of the non-zero values on the row m of the coefficient matrix
        // (mirror or swap the indices of boundary nodes, as appropriate)
        let mut nn = [0, 0, 0];
        nn[CUR] = m;
        if self.ebcs.is_periodic_along_x() {
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
    use super::FdmLaplacian1d;
    use crate::{EssentialBcs1d, Grid1d, Side};
    use russell_lab::Vector;
    use russell_sparse::Sym;

    const LEF: f64 = 1.0;
    const RIG: f64 = 2.0;

    #[test]
    fn new_captures_errors() {
        let grid = Grid1d::new(&[0.0, 0.1, 0.4]).unwrap();
        let ebcs = EssentialBcs1d::new();
        let fdm = FdmLaplacian1d::new(grid, ebcs, 1.0);
        assert_eq!(fdm.err(), Some("grid must have uniform spacing"));
    }

    #[test]
    fn new_works() {
        //  0  1   2   3
        // dx = 1.0
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let ebcs = EssentialBcs1d::new();

        let fdm = FdmLaplacian1d::new(grid, ebcs, 100.0).unwrap();
        assert_eq!(&fdm.molecule, &[-200.0, 100.0, 100.0]);

        assert_eq!(fdm.get_info(), (4, 0, 4, 0, 4));
        assert_eq!(fdm.get_grid().size(), 4);
        assert_eq!(fdm.get_equations().neq(), 4);
    }

    #[test]
    fn get_matrices_work() {
        //  0*  1   2   3
        // dx = 1.0
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let mut ebcs = EssentialBcs1d::new();
        const LEF: f64 = 1.0;
        let lef = |_| LEF;
        assert_eq!(lef(0.0), LEF);
        ebcs.set(&grid, Side::Xmin, lef); //  0

        let fdm = FdmLaplacian1d::new(grid, ebcs, 100.0).unwrap();
        let (kk, cc_mat) = fdm.get_kk_and_cc_matrices(0, Sym::No);
        let (aa, ee_mat) = fdm.get_aa_and_ee_matrices(0, true);
        let cc = cc_mat.unwrap();
        let ee = ee_mat.unwrap();

        let (nu, np, na, nw, nh) = fdm.get_info();
        assert_eq!(nu, 3);
        assert_eq!(np, 1);
        assert_eq!(na, 4);
        assert_eq!(nw, 1);
        assert_eq!(nh, 5);

        // The full matrix is:
        //      0*   1    2    3
        // ┌                     ┐
        // │ -200  200    .    . │  0*(p0)
        // │  100 -200  100    . │  1→0
        // │    .  100 -200  100 │  2→1
        // │    .    .  200 -200 │  3→2
        // └                     ┘
        //      0*   1    2    3

        // K =
        //      1    2    3
        // ┌                ┐
        // │ -200  100    . │  1→0
        // │  100 -200  100 │  2→1
        // │    .  200 -200 │  3→2
        // └                ┘
        //      1    2    3
        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌                ┐\n\
             │ -200  100    0 │\n\
             │  100 -200  100 │\n\
             │    0  200 -200 │\n\
             └                ┘"
        );

        // C =
        //     0*
        // ┌     ┐
        // │ 100 │  1→0
        // │   . │  2→1
        // │   . │  3→2
        // └     ┘
        //     0*
        assert_eq!(
            format!("{}", cc.as_dense()),
            "┌     ┐\n\
             │ 100 │\n\
             │   0 │\n\
             │   0 │\n\
             └     ┘"
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
        // │ -200  200    .    .    1 │  0*
        // │  100 -200  100    .    . │  1
        // │    .  100 -200  100    . │  2
        // │    .    .  200 -200    . │  3
        // │    1    .    .    .    . │  4w
        // └                          ┘
        //      0*   1    2    3    4w
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                          ┐\n\
             │ -200  200    0    0    1 │\n\
             │  100 -200  100    0    0 │\n\
             │    0  100 -200  100    0 │\n\
             │    0    0  200 -200    0 │\n\
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
        ebcs.set_homogeneous(&grid);

        let fdm = FdmLaplacian1d::new(grid, ebcs, 1.0).unwrap();
        let (kk, cc_mat) = fdm.get_kk_and_cc_matrices(0, Sym::No);
        let (aa, ee_mat) = fdm.get_aa_and_ee_matrices(0, true);
        let cc = cc_mat.unwrap();
        let ee = ee_mat.unwrap();

        let (nu, np, na, nw, nh) = fdm.get_info();
        assert_eq!(nu, 2);
        assert_eq!(np, 2);
        assert_eq!(na, 4);
        assert_eq!(nw, 2);
        assert_eq!(nh, 6);

        // The full matrix is:
        //    0* 1  2  3*
        // ┌              ┐
        // │ -2  2  .  .  │  0*
        // │  1 -2  1  .  │  1
        // │  .  1 -2  1  │  2
        // │  .  .  2 -2  │  3*
        // └              ┘
        //    0* 1  2  3*

        // K =
        //    1  2
        // ┌       ┐
        // │ -2  1 │  1
        // │  1 -2 │  2
        // └       ┘
        //    1  2
        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌       ┐\n\
             │ -2  1 │\n\
             │  1 -2 │\n\
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
            "┌     ┐\n\
             │ 1 0 │\n\
             │ 0 1 │\n\
             └     ┘"
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
        // │ -2  2  .  .  1  . │  0*
        // │  1 -2  1  .  .  . │  1
        // │  .  1 -2  1  .  . │  2
        // │  .  .  2 -2  .  1 │  3*
        // │  1  .  .  .  .  . │  3*
        // │  .  .  .  1  .  . │  3*
        // └                   ┘
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                   ┐\n\
             │ -2  2  0  0  1  0 │\n\
             │  1 -2  1  0  0  0 │\n\
             │  0  1 -2  1  0  0 │\n\
             │  0  0  2 -2  0  1 │\n\
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
        ebcs.set_periodic(&grid, true);

        let fdm = FdmLaplacian1d::new(grid, ebcs, 1.0).unwrap();
        let (kk, cc_mat) = fdm.get_kk_and_cc_matrices(0, Sym::No);
        let (aa, ee_mat) = fdm.get_aa_and_ee_matrices(0, true);
        assert!(cc_mat.is_none());
        assert!(ee_mat.is_none());

        let (nu, np, na, nw, nh) = fdm.get_info();
        assert_eq!(nu, 4);
        assert_eq!(np, 0);
        assert_eq!(na, 4);
        assert_eq!(nw, 0);
        assert_eq!(nh, 4);

        // K = A =
        //    0  1  2  3
        // ┌             ┐
        // │ -2  1  .  1 │  0
        // │  1 -2  1  . │  1
        // │  .  1 -2  1 │  2
        // │  1  .  1 -2 │  3
        // └             ┘
        //    0  1  2  3

        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌             ┐\n\
             │ -2  1  0  1 │\n\
             │  1 -2  1  0 │\n\
             │  0  1 -2  1 │\n\
             │  1  0  1 -2 │\n\
             └             ┘"
        );
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌             ┐\n\
             │ -2  1  0  1 │\n\
             │  1 -2  1  0 │\n\
             │  0  1 -2  1 │\n\
             │  1  0  1 -2 │\n\
             └             ┘"
        );
    }

    #[test]
    fn get_vectors_works() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 5).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        //  0*  1   2   3   4*
        ebcs.set(&grid, Side::Xmin, |_| LEF);
        ebcs.set(&grid, Side::Xmax, |_| RIG);

        let fdm = FdmLaplacian1d::new(grid, ebcs, 1.0).unwrap();

        let (u, p, f) = fdm.get_vectors(|_| 100.0);
        assert_eq!(u.dim(), 3); // nu
        assert_eq!(p.dim(), 2); // np
        assert_eq!(f.dim(), 3); // nu
        assert_eq!(u.as_data(), &[0.0, 0.0, 0.0]);
        assert_eq!(p.as_data(), &[LEF, RIG]);
        assert_eq!(f.as_data(), &[100.0, 100.0, 100.0]);

        let a = fdm.get_composed_vector(&u, &p);
        assert_eq!(a.dim(), 5); // na
        assert_eq!(a.as_data(), &[LEF, 0.0, 0.0, 0.0, RIG]);

        let (h, b) = fdm.get_vectors_lmm(|_| 100.0);
        assert_eq!(h.dim(), 5 + 2); // na + nw
        assert_eq!(b.dim(), 5 + 2); // na + nw
        assert_eq!(h.as_data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(b.as_data(), &[100.0, 100.0, 100.0, 100.0, 100.0, LEF, RIG]);
    }

    #[test]
    fn loop_over_mm_row_works() {
        // The full matrix is:
        //    0  1  2  3
        // ┌              ┐
        // │ -2  2  .  .  │  0
        // │  1 -2  1  .  │  1
        // │  .  1 -2  1  │  2
        // │  .  .  2 -2  │  3
        // └              ┘
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let ebcs = EssentialBcs1d::new();
        let lap = FdmLaplacian1d::new(grid, ebcs, 1.0).unwrap();
        let mut row_0 = Vec::new();
        let mut row_1 = Vec::new();
        let mut row_2 = Vec::new();
        let mut row_3 = Vec::new();
        lap.loop_over_full_coef_mat_row(0, |n, val| row_0.push((n, val)));
        lap.loop_over_full_coef_mat_row(1, |n, val| row_1.push((n, val)));
        lap.loop_over_full_coef_mat_row(2, |n, val| row_2.push((n, val)));
        lap.loop_over_full_coef_mat_row(3, |n, val| row_3.push((n, val)));
        assert_eq!(row_0, &[(0, -2.0), (1, 1.0), (1, 1.0)]);
        assert_eq!(row_1, &[(1, -2.0), (0, 1.0), (2, 1.0)]);
        assert_eq!(row_2, &[(2, -2.0), (1, 1.0), (3, 1.0)]);
        assert_eq!(row_3, &[(3, -2.0), (2, 1.0), (2, 1.0)]);
    }

    #[test]
    fn loop_over_grid_points_works() {
        let nx = 3;
        let grid = Grid1d::new_uniform(-1.0, 1.0, nx).unwrap();
        let ebcs = EssentialBcs1d::new();
        let lap = FdmLaplacian1d::new(grid, ebcs, 1.0).unwrap();
        let mut xx = Vector::new(nx);
        lap.loop_over_grid_points(|m, x| {
            xx[m] = x;
        });
        assert_eq!(
            format!("{}", xx),
            "┌    ┐\n\
             │ -1 │\n\
             │  0 │\n\
             │  1 │\n\
             └    ┘"
        );
    }
}
