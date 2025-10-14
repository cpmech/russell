use crate::EssentialBcs2d;
use crate::StrError;
use russell_sparse::{CooMatrix, Sym};

// constants for clarity/convenience
const CUR: usize = 0; // current node
const LEF: usize = 1; // left node
const RIG: usize = 2; // right node
const BOT: usize = 3; // bottom node
const TOP: usize = 4; // top node
const INI_X: usize = 0;
const INI_Y: usize = 0;

/// Implements the Finite Difference (FDM) Laplacian operator in 2D
///
/// Given the (continuum) scalar field ϕ(x, y) and its Laplacian
///
/// ```text
///           ∂²ϕ        ∂²ϕ
/// L{ϕ} = kx ———  +  ky ———
///           ∂x²        ∂y²
/// ```
///
/// we substitute the partial derivatives using central FDM over a rectangular grid.
/// The resulting discrete Laplacian is expressed by the coefficient matrix `M` and the vector `a`:
///
/// ```text
/// D{ϕᵢⱼ} = M a
/// ```
///
/// ϕᵢⱼ are the discrete counterpart of ϕ(x, y) over the (nx, ny) grid. However, these
/// values are "sequentially" mapped onto to the vector `a` using the following formula:
///
/// ```text
/// ϕᵢⱼ → aₘ   with   m = i + j nx
/// ```
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
///     A       x       b
/// ```
///
/// where `w` is the vector of Lagrange multipliers, `E` is the Lagrange matrix, and `ū` is the vector of
/// prescribed values at EBC nodes. The Lagrange matrix `E` has a row for each EBC (prescribed) node and a column
/// for every node. Each row in `E` has a single `1` at the column corresponding to the EBC node, and `0`s elsewhere.
pub struct FdmLaplacian2dNew<'a> {
    /// Holds a reference to the essential boundary conditions handler
    ebcs: &'a EssentialBcs2d<'a>,

    /// Holds the FDM coefficients (α, β, β, γ, γ) corresponding to (CUR, LEF, RIG, BOT, TOP)
    ///
    /// These coefficients are applied over the "bandwidth" of the coefficient matrix
    molecule: Vec<f64>,
}

impl<'a> FdmLaplacian2dNew<'a> {
    /// Allocates a new instance
    pub fn new(ebcs: &'a EssentialBcs2d<'a>, kx: f64, ky: f64) -> Result<Self, StrError> {
        // check grid
        let (dx, dy) = match ebcs.get_grid().get_dx_dy() {
            Some((dx, dy)) => (dx, dy),
            None => return Err("grid must have uniform spacing"),
        };

        // auxiliary variables
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let alpha = -2.0 * (kx / dx2 + ky / dy2);
        let beta = kx / dx2;
        let gamma = ky / dy2;

        // done
        Ok(FdmLaplacian2dNew {
            ebcs,
            molecule: vec![alpha, beta, beta, gamma, gamma],
        })
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
        let nu = self.ebcs.num_unknown();
        let np = self.ebcs.num_prescribed();
        let nnz_kk = 5 * nu + extra_nnz; // 5 is the bandwidth
        let mut kk = CooMatrix::new(nu, nu, nnz_kk, sym_kk).unwrap();
        let mut cc = if np == 0 {
            // russell_sparse requires at least a 1x1 matrix with 1 non-zero entry
            CooMatrix::new(1, 1, 1, Sym::No).unwrap()
        } else {
            let nnz_cc = 4 * np; // 4 is the max number of neighbors (worst case)
            CooMatrix::new(nu, np, nnz_cc, Sym::No).unwrap()
        };
        self.ebcs.for_each_unknown_node(|iu, m, _, _| {
            self.loop_over_bandwidth(m, |b, n| {
                if self.ebcs.is_prescribed(n) {
                    let jp = self.ebcs.get_index_prescribed(n).unwrap();
                    cc.put(iu, jp, self.molecule[b]).unwrap();
                } else {
                    let ju = self.ebcs.get_index_unknown(n).unwrap();
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
    ///     A       x       b
    /// ```
    ///
    /// Note: this matrix is not symmetric because of the flipping (mirroring) strategy for boundary nodes.
    pub fn get_aa_and_ee_matrices(&self, extra_nnz: usize, return_ee_mat: bool) -> (CooMatrix, Option<CooMatrix>) {
        // build the A matrix
        let na = self.ebcs.num_total();
        let np = self.ebcs.num_prescribed();
        let naa = na + np;
        let nnz = 5 * na + 2 * np + extra_nnz; // 5 is the bandwidth, 2*np is for E and Eᵀ
        let mut aa = CooMatrix::new(naa, naa, nnz, Sym::No).unwrap();
        for m in 0..na {
            self.loop_over_bandwidth(m, |b, n| {
                aa.put(m, n, self.molecule[b]).unwrap();
            });
        }

        // assemble E and Eᵀ into A
        self.ebcs.for_each_prescribed_node(|ip, j, _, _, _| {
            aa.put(na + ip, j, 1.0).unwrap(); // E
            aa.put(j, na + ip, 1.0).unwrap(); // Eᵀ
        });

        // build and return the E matrix, if requested and available
        if return_ee_mat && np > 0 {
            let mut ee = CooMatrix::new(np, na, np, Sym::No).unwrap();
            self.ebcs.for_each_prescribed_node(|ip, j, _, _, _| {
                ee.put(ip, j, 1.0).unwrap(); // E
            });
            (aa, Some(ee))
        } else {
            (aa, None)
        }
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
        let nx = self.ebcs.get_grid().nx();
        let ny = self.ebcs.get_grid().ny();
        let fin_x = nx - 1;
        let fin_y = ny - 1;
        let i = m % nx;
        let j = m / nx;

        // n indices of the non-zero values on the row m of the coefficient matrix
        // (mirror or swap the indices of boundary nodes, as appropriate)
        let mut nn = [0, 0, 0, 0, 0];
        nn[CUR] = m;
        if self.ebcs.is_periodic_along_x() {
            nn[LEF] = if i != INI_X { m - 1 } else { m + fin_x };
            nn[RIG] = if i != fin_x { m + 1 } else { m - fin_x };
        } else {
            nn[LEF] = if i != INI_X { m - 1 } else { m + 1 };
            nn[RIG] = if i != fin_x { m + 1 } else { m - 1 };
        }
        if self.ebcs.is_periodic_along_y() {
            nn[BOT] = if j != INI_Y { m - nx } else { m + fin_y * nx };
            nn[TOP] = if j != fin_y { m + nx } else { m - fin_y * nx };
        } else {
            nn[BOT] = if j != INI_Y { m - nx } else { m + nx };
            nn[TOP] = if j != fin_y { m + nx } else { m - nx };
        }

        // execute callback
        for b in 0..5 {
            callback(b, nn[b]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::FdmLaplacian2dNew;
    use crate::{EssentialBcs2d, Grid2d, Side};
    use russell_sparse::Sym;

    #[test]
    fn new_captures_errors() {
        let grid = Grid2d::new(&[0.0, 0.1, 0.4], &[0.0, 0.2, 0.5]).unwrap();
        let ebcs = EssentialBcs2d::new(&grid);
        let fdm = FdmLaplacian2dNew::new(&ebcs, 1.0, 1.0);
        assert_eq!(fdm.err(), Some("grid must have uniform spacing"));
    }

    #[test]
    fn new_works() {
        //  8  9  10  11
        //  4  5   6   7
        //  0  1   2   3
        // dx = 1.0, dy = 1.0
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 2.0, 4, 3).unwrap();
        let ebcs = EssentialBcs2d::new(&grid);

        let fdm = FdmLaplacian2dNew::new(&ebcs, 100.0, 300.0).unwrap();
        assert_eq!(&fdm.molecule, &[-800.0, 100.0, 100.0, 300.0, 300.0]);
    }

    #[test]
    fn get_matrices_work() {
        //  8*  9  10  11
        //  4*  5   6   7
        //  0*  1   2   3
        // dx = 1.0, dy = 1.0
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 2.0, 4, 3).unwrap();
        let mut ebcs = EssentialBcs2d::new(&grid);
        const LEF: f64 = 1.0;
        let lef = |_, _| LEF;
        assert_eq!(lef(0.0, 0.0), LEF);
        ebcs.set(Side::Xmin, lef); //  0  4  8

        let fdm = FdmLaplacian2dNew::new(&ebcs, 100.0, 300.0).unwrap();
        let (kk, cc_mat) = fdm.get_kk_and_cc_matrices(0, Sym::No);
        let (aa, ee_mat) = fdm.get_aa_and_ee_matrices(0, true);
        let cc = cc_mat.unwrap();
        let ee = ee_mat.unwrap();

        // The full matrix is:
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11
        // ┌                                                             ┐
        // │ -800  200    .    .  600    .    .    .    .    .    .    . │  0*(p0)
        // │  100 -800  100    .    .  600    .    .    .    .    .    . │  1→0
        // │    .  100 -800  100    .    .  600    .    .    .    .    . │  2→1
        // │    .    .  200 -800    .    .    .  600    .    .    .    . │  3→2
        // │  300    .    .    . -800  200    .    .  300    .    .    . │  4*(p1)
        // │    .  300    .    .  100 -800  100    .    .  300    .    . │  5→3
        // │    .    .  300    .    .  100 -800  100    .    .  300    . │  6→4
        // │    .    .    .  300    .    .  200 -800    .    .    .  300 │  7→5
        // │    .    .    .    .  600    .    .    . -800  200    .    . │  8*(p3)
        // │    .    .    .    .    .  600    .    .  100 -800  100    . │  9→6
        // │    .    .    .    .    .    .  600    .    .  100 -800  100 │ 10→7
        // │    .    .    .    .    .    .    .  600    .    .  200 -800 │ 11→8
        // └                                                             ┘
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11

        // K =
        //      1    2    3    5    6    7    9   10   11
        // ┌                                              ┐
        // │ -800  100    .  600    .    .    .    .    . │  1→0
        // │  100 -800  100    .  600    .    .    .    . │  2→1
        // │    .  200 -800    .    .  600    .    .    . │  3→2
        // │  300    .    . -800  100    .  300    .    . │  5→3
        // │    .  300    .  100 -800  100    .  300    . │  6→4
        // │    .    .  300    .  200 -800    .    .  300 │  7→5
        // │    .    .    .  600    .    . -800  100    . │  9→6
        // │    .    .    .    .  600    .  100 -800  100 │ 10→7
        // │    .    .    .    .    .  600    .  200 -800 │ 11→8
        // └                                              ┘
        //      1    2    3    5    6    7    9   10   11
        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌                                              ┐\n\
             │ -800  100    0  600    0    0    0    0    0 │\n\
             │  100 -800  100    0  600    0    0    0    0 │\n\
             │    0  200 -800    0    0  600    0    0    0 │\n\
             │  300    0    0 -800  100    0  300    0    0 │\n\
             │    0  300    0  100 -800  100    0  300    0 │\n\
             │    0    0  300    0  200 -800    0    0  300 │\n\
             │    0    0    0  600    0    0 -800  100    0 │\n\
             │    0    0    0    0  600    0  100 -800  100 │\n\
             │    0    0    0    0    0  600    0  200 -800 │\n\
             └                                              ┘"
        );

        // C =
        //     0*  4*  8*
        // ┌             ┐
        // │ 100   .   . │  1→0
        // │   .   .   . │  2→1
        // │   .   .   . │  3→2
        // │   . 100   . │  5→3
        // │   .   .   . │  6→4
        // │   .   .   . │  7→5
        // │   .   . 100 │  9→6
        // │   .   .   . │ 10→7
        // │   .   .   . │ 11→8
        // └             ┘
        //     0*  4*  8*
        assert_eq!(
            format!("{}", cc.as_dense()),
            "┌             ┐\n\
             │ 100   0   0 │\n\
             │   0   0   0 │\n\
             │   0   0   0 │\n\
             │   0 100   0 │\n\
             │   0   0   0 │\n\
             │   0   0   0 │\n\
             │   0   0 100 │\n\
             │   0   0   0 │\n\
             │   0   0   0 │\n\
             └             ┘"
        );

        // E =
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11
        // ┌                                                             ┐
        // │    1    .    .    .    .    .    .    .    .    .    .    . │  0*
        // │    .    .    .    .    1    .    .    .    .    .    .    . │  4*
        // │    .    .    .    .    .    .    .    .    1    .    .    . │  8*
        // └                                                             ┘
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11
        assert_eq!(
            format!("{}", ee.as_dense()),
            "┌                         ┐\n\
             │ 1 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 1 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 1 0 0 0 │\n\
             └                         ┘"
        );

        // A =
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11   12w  13w  14w
        // ┌                                                                            ┐
        // │ -800  200    .    .  600    .    .    .    .    .    .    .    1    .    . │  0*
        // │  100 -800  100    .    .  600    .    .    .    .    .    .    .    .    . │  1
        // │    .  100 -800  100    .    .  600    .    .    .    .    .    .    .    . │  2
        // │    .    .  200 -800    .    .    .  600    .    .    .    .    .    .    . │  3
        // │  300    .    .    . -800  200    .    .  300    .    .    .    .    1    . │  4*
        // │    .  300    .    .  100 -800  100    .    .  300    .    .    .    .    . │  5
        // │    .    .  300    .    .  100 -800  100    .    .  300    .    .    .    . │  6
        // │    .    .    .  300    .    .  200 -800    .    .    .  300    .    .    . │  7
        // │    .    .    .    .  600    .    .    . -800  200    .    .    .    .    1 │  8*
        // │    .    .    .    .    .  600    .    .  100 -800  100    .    .    .    . │  9
        // │    .    .    .    .    .    .  600    .    .  100 -800  100    .    .    . │ 10
        // │    .    .    .    .    .    .    .  600    .    .  200 -800    .    .    . │ 11
        // │    1    .    .    .    .    .    .    .    .    .    .    .    .    .    . │ 12w
        // │    .    .    .    .    1    .    .    .    .    .    .    .    .    .    . │ 13w
        // │    .    .    .    .    .    .    .    .    1    .    .    .    .    .    . │ 14w
        // └                                                                            ┘
        //      0*   1    2    3    4*   5    6    7    8*   9   10   11   12w  13w  14w
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                                                            ┐\n\
             │ -800  200    0    0  600    0    0    0    0    0    0    0    1    0    0 │\n\
             │  100 -800  100    0    0  600    0    0    0    0    0    0    0    0    0 │\n\
             │    0  100 -800  100    0    0  600    0    0    0    0    0    0    0    0 │\n\
             │    0    0  200 -800    0    0    0  600    0    0    0    0    0    0    0 │\n\
             │  300    0    0    0 -800  200    0    0  300    0    0    0    0    1    0 │\n\
             │    0  300    0    0  100 -800  100    0    0  300    0    0    0    0    0 │\n\
             │    0    0  300    0    0  100 -800  100    0    0  300    0    0    0    0 │\n\
             │    0    0    0  300    0    0  200 -800    0    0    0  300    0    0    0 │\n\
             │    0    0    0    0  600    0    0    0 -800  200    0    0    0    0    1 │\n\
             │    0    0    0    0    0  600    0    0  100 -800  100    0    0    0    0 │\n\
             │    0    0    0    0    0    0  600    0    0  100 -800  100    0    0    0 │\n\
             │    0    0    0    0    0    0    0  600    0    0  200 -800    0    0    0 │\n\
             │    1    0    0    0    0    0    0    0    0    0    0    0    0    0    0 │\n\
             │    0    0    0    0    1    0    0    0    0    0    0    0    0    0    0 │\n\
             │    0    0    0    0    0    0    0    0    1    0    0    0    0    0    0 │\n\
             └                                                                            ┘"
        );
    }

    #[test]
    fn get_matrices_homogeneous_bcs_work() {
        //       8   9  10  11
        //      ---------------
        // 13 | 12* 13* 14* 15* | 14
        //  9 |  8*  9  10  11* | 10
        //  5 |  4*  5   6   7* |  6
        //  1 |  0*  1*  2*  3* |  2
        //      ---------------
        //       4   5   6   7
        // dx = 1.0, dy = 1.0
        let grid = Grid2d::new_uniform(0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new(&grid);
        ebcs.set_homogeneous();

        let fdm = FdmLaplacian2dNew::new(&ebcs, 1.0, 1.0).unwrap();
        let (kk, cc_mat) = fdm.get_kk_and_cc_matrices(0, Sym::No);
        let (aa, ee_mat) = fdm.get_aa_and_ee_matrices(0, true);
        let cc = cc_mat.unwrap();
        let ee = ee_mat.unwrap();

        // The full matrix is:
        //    0* 1* 2* 3* 4* 5  6  7* 8* 9 10 11*12*13*14*15*
        // ┌                                                 ┐
        // │ -4  2  .  .  2  .  .  .  .  .  .  .  .  .  .  . │  0*
        // │  1 -4  1  .  .  2  .  .  .  .  .  .  .  .  .  . │  1*
        // │  .  1 -4  1  .  .  2  .  .  .  .  .  .  .  .  . │  2*
        // │  .  .  2 -4  .  .  .  2  .  .  .  .  .  .  .  . │  3*
        // │  1  .  .  . -4  2  .  .  1  .  .  .  .  .  .  . │  4*
        // │  .  1  .  .  1 -4  1  .  .  1  .  .  .  .  .  . │  5
        // │  .  .  1  .  .  1 -4  1  .  .  1  .  .  .  .  . │  6
        // │  .  .  .  1  .  .  2 -4  .  .  .  1  .  .  .  . │  7*
        // │  .  .  .  .  1  .  .  . -4  2  .  .  1  .  .  . │  8*
        // │  .  .  .  .  .  1  .  .  1 -4  1  .  .  1  .  . │  9
        // │  .  .  .  .  .  .  1  .  .  1 -4  1  .  .  1  . │ 10
        // │  .  .  .  .  .  .  .  1  .  .  2 -4  .  .  .  1 │ 11*
        // │  .  .  .  .  .  .  .  .  2  .  .  . -4  2  .  . │ 12*
        // │  .  .  .  .  .  .  .  .  .  2  .  .  1 -4  1  . │ 13*
        // │  .  .  .  .  .  .  .  .  .  .  2  .  .  1 -4  1 │ 14*
        // │  .  .  .  .  .  .  .  .  .  .  .  2  .  .  2 -4 │ 15*
        // └                                                 ┘
        //    0* 1* 2* 3* 4* 5  6  7* 8* 9 10 11*12*13*14*15*

        // K =
        //    5  6  9 10 *
        // ┌              ┐
        // │ -4  1  1  .  │  5
        // │  1 -4  .  1  │  6
        // │  1  . -4  1  │  9
        // │  .  1  1 -4  │ 10
        // └              ┘
        //    5  6  9 10 *
        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌             ┐\n\
             │ -4  1  1  0 │\n\
             │  1 -4  0  1 │\n\
             │  1  0 -4  1 │\n\
             │  0  1  1 -4 │\n\
             └             ┘"
        );

        // C =
        //    0* 1* 2* 3* 4* 7* 8*11*12*13*14*15*
        // ┌                                     ┐
        // │  .  1  .  .  1  .  .  .  .  .  .  . │  5
        // │  .  .  1  .  .  1  .  .  .  .  .  . │  6
        // │  .  .  .  .  .  .  1  .  .  1  .  . │  9
        // │  .  .  .  .  .  .  .  1  .  .  1  . │ 10
        // └                                     ┘
        //    0* 1* 2* 3* 4* 7* 8*11*12*13*14*15*
        //
        assert_eq!(
            format!("{}", cc.as_dense()),
            "┌                         ┐\n\
             │ 0 1 0 0 1 0 0 0 0 0 0 0 │\n\
             │ 0 0 1 0 0 1 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 1 0 0 1 0 0 │\n\
             │ 0 0 0 0 0 0 0 1 0 0 1 0 │\n\
             └                         ┘"
        );

        // E =
        //    0* 1* 2* 3* 4* 5  6  7* 8* 9 10 11*12*13*14*15*
        // ┌                                                 ┐
        // │  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │  0*
        // │  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  . │  1*
        // │  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  . │  2*
        // │  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  . │  3*
        // │  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  . │  4*
        // │  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  . │  7*
        // │  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  . │  8*
        // │  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  . │ 11*
        // │  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  . │ 12*
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  1  .  . │ 13*
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  .  1  . │ 14*
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  1 │ 15*
        // └                                                 ┘
        assert_eq!(
            format!("{}", ee.as_dense()),
            "┌                                 ┐\n\
             │ 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 │\n\
             └                                 ┘"
        );

        // A =
        //    0* 1* 2* 3* 4* 5  6  7* 8* 9 10 11*12*13*14*15*16w17w18w19w20w21w22w23w24w25w26w27w
        // ┌                                                                                     ┐
        // │ -4  2  .  .  2  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  . │  0*
        // │  1 -4  1  .  .  2  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  . │  1*
        // │  .  1 -4  1  .  .  2  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  . │  2*
        // │  .  .  2 -4  .  .  .  2  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  . │  3*
        // │  1  .  .  . -4  2  .  .  1  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  . │  4*
        // │  .  1  .  .  1 -4  1  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │  5
        // │  .  .  1  .  .  1 -4  1  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │  6
        // │  .  .  .  1  .  .  2 -4  .  .  .  1  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  . │  7*
        // │  .  .  .  .  1  .  .  . -4  2  .  .  1  .  .  .  .  .  .  .  .  .  1  .  .  .  .  . │  8*
        // │  .  .  .  .  .  1  .  .  1 -4  1  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  . │  9
        // │  .  .  .  .  .  .  1  .  .  1 -4  1  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  . │ 10
        // │  .  .  .  .  .  .  .  1  .  .  2 -4  .  .  .  1  .  .  .  .  .  .  .  1  .  .  .  . │ 11*
        // │  .  .  .  .  .  .  .  .  2  .  .  . -4  2  .  .  .  .  .  .  .  .  .  .  1  .  .  . │ 12*
        // │  .  .  .  .  .  .  .  .  .  2  .  .  1 -4  1  .  .  .  .  .  .  .  .  .  .  1  .  . │ 13*
        // │  .  .  .  .  .  .  .  .  .  .  2  .  .  1 -4  1  .  .  .  .  .  .  .  .  .  .  1  . │ 14*
        // │  .  .  .  .  .  .  .  .  .  .  .  2  .  .  2 -4  .  .  .  .  .  .  .  .  .  .  .  1 │ 15*
        // │  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 16w
        // │  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 17w
        // │  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 18w
        // │  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 19w
        // │  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 20w
        // │  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 21w
        // │  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 22w
        // │  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 23w
        // │  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 24w
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  .  . │ 25w
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  .  . │ 26w
        // │  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  .  . │ 27w
        // └                                                                                     ┘
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                                                                     ┐\n\
             │ -4  2  0  0  2  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  1 -4  1  0  0  2  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  1 -4  1  0  0  2  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  2 -4  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0 │\n\
             │  1  0  0  0 -4  2  0  0  1  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0 │\n\
             │  0  1  0  0  1 -4  1  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  1  0  0  1 -4  1  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  1  0  0  2 -4  0  0  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0 │\n\
             │  0  0  0  0  1  0  0  0 -4  2  0  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0 │\n\
             │  0  0  0  0  0  1  0  0  1 -4  1  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  1  0  0  1 -4  1  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  1  0  0  2 -4  0  0  0  1  0  0  0  0  0  0  0  1  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  2  0  0  0 -4  2  0  0  0  0  0  0  0  0  0  0  1  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  2  0  0  1 -4  1  0  0  0  0  0  0  0  0  0  0  1  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  2  0  0  1 -4  1  0  0  0  0  0  0  0  0  0  0  1  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  2  0  0  2 -4  0  0  0  0  0  0  0  0  0  0  0  1 │\n\
             │  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             │  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0 │\n\
             └                                                                                     ┘"
        );
    }

    #[test]
    fn get_matrices_periodic_bcs_work() {
        //      0  1  2
        //     --------
        // 11 | 9 10 11 | 9
        //  8 | 6  7  8 | 6
        //  5 | 3  4  5 | 3
        //  2 | 0  1  2 | 0
        //      -------
        //      9 10 11

        let grid = Grid2d::new_uniform(0.0, 2.0, 0.0, 3.0, 3, 4).unwrap();
        let mut ebcs = EssentialBcs2d::new(&grid);
        ebcs.set_periodic(true, true);

        let fdm = FdmLaplacian2dNew::new(&ebcs, 1.0, 1.0).unwrap();
        let (kk, cc_mat) = fdm.get_kk_and_cc_matrices(0, Sym::No);
        let (aa, ee_mat) = fdm.get_aa_and_ee_matrices(0, true);
        assert!(cc_mat.is_none());
        assert!(ee_mat.is_none());

        // K = A =
        //    0  1  2  3  4  5  6  7  8  9 10 11
        // ┌                                     ┐
        // │ -4  1  1  1  .  .  .  .  .  1  .  . │  0
        // │  1 -4  1  .  1  .  .  .  .  .  1  . │  1
        // │  1  1 -4  .  .  1  .  .  .  .  .  1 │  2
        // │  1  .  . -4  1  1  1  .  .  .  .  . │  3
        // │  .  1  .  1 -4  1  .  1  .  .  .  . │  4
        // │  .  .  1  1  1 -4  .  .  1  .  .  . │  5
        // │  .  .  .  1  .  . -4  1  1  1  .  . │  6
        // │  .  .  .  .  1  .  1 -4  1  .  1  . │  7
        // │  .  .  .  .  .  1  1  1 -4  .  .  1 │  8
        // │  1  .  .  .  .  .  1  .  . -4  1  1 │  9
        // │  .  1  .  .  .  .  .  1  .  1 -4  1 │ 10
        // │  .  .  1  .  .  .  .  .  1  1  1 -4 │ 11
        // └                                     ┘
        //    0  1  2  3  4  5  6  7  8  9 10 11

        assert_eq!(
            format!("{}", kk.as_dense()),
            "┌                                     ┐\n\
             │ -4  1  1  1  0  0  0  0  0  1  0  0 │\n\
             │  1 -4  1  0  1  0  0  0  0  0  1  0 │\n\
             │  1  1 -4  0  0  1  0  0  0  0  0  1 │\n\
             │  1  0  0 -4  1  1  1  0  0  0  0  0 │\n\
             │  0  1  0  1 -4  1  0  1  0  0  0  0 │\n\
             │  0  0  1  1  1 -4  0  0  1  0  0  0 │\n\
             │  0  0  0  1  0  0 -4  1  1  1  0  0 │\n\
             │  0  0  0  0  1  0  1 -4  1  0  1  0 │\n\
             │  0  0  0  0  0  1  1  1 -4  0  0  1 │\n\
             │  1  0  0  0  0  0  1  0  0 -4  1  1 │\n\
             │  0  1  0  0  0  0  0  1  0  1 -4  1 │\n\
             │  0  0  1  0  0  0  0  0  1  1  1 -4 │\n\
             └                                     ┘"
        );
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │ -4  1  1  1  0  0  0  0  0  1  0  0 │\n\
             │  1 -4  1  0  1  0  0  0  0  0  1  0 │\n\
             │  1  1 -4  0  0  1  0  0  0  0  0  1 │\n\
             │  1  0  0 -4  1  1  1  0  0  0  0  0 │\n\
             │  0  1  0  1 -4  1  0  1  0  0  0  0 │\n\
             │  0  0  1  1  1 -4  0  0  1  0  0  0 │\n\
             │  0  0  0  1  0  0 -4  1  1  1  0  0 │\n\
             │  0  0  0  0  1  0  1 -4  1  0  1  0 │\n\
             │  0  0  0  0  0  1  1  1 -4  0  0  1 │\n\
             │  1  0  0  0  0  0  1  0  0 -4  1  1 │\n\
             │  0  1  0  0  0  0  0  1  0  1 -4  1 │\n\
             │  0  0  1  0  0  0  0  0  1  1  1 -4 │\n\
             └                                     ┘"
        );
    }
}
