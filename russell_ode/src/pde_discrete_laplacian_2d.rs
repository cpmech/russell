use crate::StrError;
use russell_sparse::{CooMatrix, Sym};
use std::collections::HashMap;

/// Defines a function of space that returns f64 (e.g., to calculate boundary condition values)
pub type FnSpace = fn(x: f64, y: f64) -> f64;

/// Specifies the (boundary) side of a rectangle
pub enum Side {
    Left,
    Right,
    Bottom,
    Top,
}

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
/// The resulting discrete Laplacian is expressed by the coefficient matrix `A` and the vector `X`:
///
/// ```text
/// D{ϕᵢⱼ} = A ⋅ X
/// ```
///
/// ϕᵢⱼ are the discrete counterpart of ϕ(x, y) over the (nx, ny) grid. However, these
/// values are "sequentially" mapped onto to the vector `X` using the following formula:
///
/// ```text
/// ϕᵢⱼ → Xₘ   with   m = i + j nx
/// ```
///
/// **Notes:**
///
/// * The operator is built with a five-point stencil.
/// * The boundary nodes are 'mirrored' yielding a no-flux barrier.
pub struct PdeDiscreteLaplacian2d {
    xmin: f64,          // min x coordinate
    ymin: f64,          // min y coordinate
    nx: usize,          // number of points along x (≥ 2)
    ny: usize,          // number of points along y (≥ 2)
    dx: f64,            // grid spacing along x
    dy: f64,            // grid spacing along y
    left: Vec<usize>,   // indices of nodes on the left edge
    right: Vec<usize>,  // indices of nodes on the right edge
    bottom: Vec<usize>, // indices of nodes on the bottom edge
    top: Vec<usize>,    // indices of nodes on the top edge

    /// Holds the FDM coefficients (α, β, β, γ, γ)
    ///
    /// These coefficients are applied over the "bandwidth" of the coefficient matrix
    molecule: Vec<f64>,

    /// Collects the essential boundary conditions
    /// Maps node => prescribed_value
    essential: HashMap<usize, FnSpace>,
}

impl PdeDiscreteLaplacian2d {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `kx` -- diffusion parameter x
    /// * `ky` -- diffusion parameter y
    /// * `xmin` -- min x coordinate
    /// * `xmax` -- max x coordinate
    /// * `ymin` -- min y coordinate
    /// * `ymax` -- max y coordinate
    /// * `nx` -- number of points along x (≥ 2)
    /// * `ny` -- number of points along y (≥ 2)
    pub fn new(
        kx: f64,
        ky: f64,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        nx: usize,
        ny: usize,
    ) -> Result<Self, StrError> {
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }
        if ny < 2 {
            return Err("ny must be ≥ 2");
        }
        let dim = nx * ny;
        let dx = (xmax - xmin) / ((nx - 1) as f64);
        let dy = (ymax - ymin) / ((ny - 1) as f64);
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let alpha = -2.0 * (kx / dx2 + ky / dy2);
        let beta = kx / dx2;
        let gamma = ky / dy2;
        Ok(PdeDiscreteLaplacian2d {
            xmin,
            ymin,
            nx,
            ny,
            dx,
            dy,
            left: (0..dim).step_by(nx).collect(),
            right: ((nx - 1)..dim).step_by(nx).collect(),
            bottom: (0..nx).collect(),
            top: ((dim - nx)..dim).collect(),
            molecule: vec![alpha, beta, beta, gamma, gamma],
            essential: HashMap::new(),
        })
    }

    /// Sets essential (Dirichlet) boundary condition
    pub fn set_essential_boundary_condition(&mut self, side: Side, value: FnSpace) {
        match side {
            Side::Left => self.left.iter().for_each(|n| {
                self.essential.insert(*n, value);
            }),
            Side::Right => self.right.iter().for_each(|n| {
                self.essential.insert(*n, value);
            }),
            Side::Bottom => self.bottom.iter().for_each(|n| {
                self.essential.insert(*n, value);
            }),
            Side::Top => self.top.iter().for_each(|n| {
                self.essential.insert(*n, value);
            }),
        };
    }

    /// Sets homogeneous boundary conditions (i.e., zero essential values at the borders)
    pub fn set_homogeneous_boundary_conditions(&mut self) {
        self.left.iter().for_each(|n| {
            self.essential.insert(*n, |_, _| 0.0);
        });
        self.right.iter().for_each(|n| {
            self.essential.insert(*n, |_, _| 0.0);
        });
        self.bottom.iter().for_each(|n| {
            self.essential.insert(*n, |_, _| 0.0);
        });
        self.top.iter().for_each(|n| {
            self.essential.insert(*n, |_, _| 0.0);
        });
    }

    /// Computes the coefficient matrix 'A' of A ⋅ X = B
    ///
    /// **Note:** Consider the following partitioning:
    ///
    /// ```text
    /// ┌          ┐ ┌    ┐   ┌    ┐
    /// │ Auu  Aup │ │ Xu │   │ Bu │
    /// │          │ │    │ = │    │
    /// │ Apu  App │ │ Xp │   │ Bp │
    /// └          ┘ └    ┘   └    ┘
    /// ```
    ///
    /// where `u` means *unknown* and `p` means *prescribed*. Thus, `Xu` is the sub-vector with
    /// unknown essential values and `Xp` is the sub-vector with prescribed essential values.
    ///
    /// Thus:
    ///
    /// ```text
    /// Auu ⋅ Xu  +  Aup ⋅ Xp  =  Bu
    /// ```
    ///
    /// To handle the prescribed essential values, we modify the system as follows:
    ///
    /// ```text
    /// ┌          ┐ ┌    ┐   ┌             ┐
    /// │ Auu   0  │ │ Xu │   │ Bu - Aup⋅Xp │
    /// │          │ │    │ = │             │
    /// │  0    1  │ │ Xp │   │     Xp      │
    /// └          ┘ └    ┘   └             ┘
    /// A := augmented(Auu)
    /// ```
    ///
    /// Thus:
    ///
    /// ```text
    /// Xu = Auu⁻¹ ⋅ (Bu - Aup⋅Xp)
    /// Xp = Xp
    /// ```
    ///
    /// Furthermore, we return an augmented 'Aup' matrix (called 'C', correction matrix), such that:
    ///
    /// ```text
    /// ┌          ┐ ┌    ┐   ┌        ┐
    /// │  0   Aup │ │ .. │   │ Aup⋅Xp │
    /// │          │ │    │ = │        │
    /// │  0    0  │ │ Xp │   │   0    │
    /// └          ┘ └    ┘   └        ┘
    /// C := augmented(Aup)
    /// ```
    ///
    /// Note that there is no performance loss in using the augmented matrix because the sparse
    /// matrix-vector multiplication will execute the same number of computations with a reduced matrix.
    /// Also, the CooMatrix will only hold the non-zero entries, thus, no extra memory is wasted.
    ///
    /// # Output
    ///
    /// Returns `(A, C)` where:
    ///
    /// * `A` -- is the augmented 'Auu' matrix (dim × dim) with ones placed on the diagonal entries
    ///  corresponding to the prescribed essential values. Also, the entries corresponding to the
    ///  essential values are zeroed.
    /// * `C` -- is the augmented 'Aup' (correction) matrix (dim × dim) with only the 'unknown rows'
    ///   and the 'prescribed' columns.
    ///
    /// # Warnings
    ///
    /// **Important:** This function must be called after [PdeDiscreteLaplacian2d::set_essential_boundary_condition]
    ///
    /// # Todo
    ///
    /// * Implement the symmetric version for solvers that can handle a triangular matrix storage.
    pub fn coefficient_matrix(&self) -> Result<(CooMatrix, CooMatrix), StrError> {
        // count max number of non-zeros
        let dim = self.nx * self.ny;
        let np = self.essential.len();
        let mut max_nnz_aa = np; // start with the diagonal 'ones'
        let mut max_nnz_cc = 1; // +1 just for when there are no essential conditions
        for i in 0..dim {
            if !self.essential.contains_key(&i) {
                self.loop_over_bandwidth(i, |j, _| {
                    if !self.essential.contains_key(&j) {
                        max_nnz_aa += 1;
                    } else {
                        max_nnz_cc += 1;
                    }
                });
            }
        }

        // allocate matrices
        let mut aa = CooMatrix::new(dim, dim, max_nnz_aa, Sym::No)?;
        let mut cc = CooMatrix::new(dim, dim, max_nnz_cc, Sym::No)?;

        // assemble
        for i in 0..dim {
            if !self.essential.contains_key(&i) {
                self.loop_over_bandwidth(i, |j, b| {
                    if !self.essential.contains_key(&j) {
                        aa.put(i, j, self.molecule[b]).unwrap();
                    } else {
                        cc.put(i, j, self.molecule[b]).unwrap();
                    }
                });
            } else {
                aa.put(i, i, 1.0).unwrap();
            }
        }
        Ok((aa, cc))
    }

    /// Executes a loop over one row of the coefficient matrix 'A' of A ⋅ X = B
    ///
    /// Note that the column indices will appear repeated due to the (zero-flux) boundaries.
    ///
    /// # Input
    ///
    /// * `i` -- the row of the coefficient matrix
    /// * `callback` -- a `function(j, Aij)` where `j` is the column index
    pub fn loop_over_coef_mat_row<F>(&self, i: usize, mut callback: F)
    where
        F: FnMut(usize, f64),
    {
        self.loop_over_bandwidth(i, |j, b| {
            callback(j, self.molecule[b]);
        });
    }

    /// Executes a loop over the prescribed values
    ///
    /// # Input
    ///
    /// * `callback` -- a `function(i, value)` where `i` is the row index
    ///   and `value` is the prescribed value.
    pub fn loop_over_prescribed_values<F>(&self, mut callback: F)
    where
        F: FnMut(usize, f64),
    {
        self.essential.iter().for_each(|(i, value)| {
            let row = i / self.nx;
            let col = i % self.nx;
            let x = self.xmin + (col as f64) * self.dx;
            let y = self.ymin + (row as f64) * self.dy;
            callback(*i, value(x, y));
        });
    }

    /// Executes a loop over the bandwidth of the coefficient matrix
    ///
    /// # Input
    ///
    /// * `i` -- the row index
    /// * `callback` -- a `function(j, b)` where `j` is the column index and
    ///   `b` is the bandwidth index, i.e., the index in the molecule array.
    fn loop_over_bandwidth<F>(&self, i: usize, mut callback: F)
    where
        F: FnMut(usize, usize),
    {
        // row and column
        let row = i / self.nx;
        let col = i % self.nx;

        // j-index of grid nodes (mirror if needed)
        let mut jays = [0, 0, 0, 0, 0];
        jays[0] = i; // current node
        jays[1] = if col == 0 { i + 1 } else { i - 1 }; // left node
        jays[2] = if col == self.nx - 1 { i - 1 } else { i + 1 }; // right node
        jays[3] = if row == 0 { i + self.nx } else { i - self.nx }; // bottom node
        jays[4] = if row == self.ny - 1 { i - self.nx } else { i + self.nx }; // top node

        // execute callback
        for (b, &j) in jays.iter().enumerate() {
            callback(j, b);
        }
    }

    /// Executes a loop over the grid points
    ///
    /// # Input
    ///
    /// * `callback` -- a `function(i, x, y)` where `i` is the point number and
    ///   `(x, y)` are the grid point coordinates.
    ///
    /// The row and column indices of the grid point can be determined with:
    ///
    /// ```text
    /// let row = i / nx;
    /// let col = i % nx;
    /// ```
    pub fn loop_over_grid_points<F>(&self, mut callback: F)
    where
        F: FnMut(usize, f64, f64),
    {
        let dim = self.nx * self.ny;
        for i in 0..dim {
            let row = i / self.nx;
            let col = i % self.nx;
            let x = self.xmin + (col as f64) * self.dx;
            let y = self.ymin + (row as f64) * self.dy;
            callback(i, x, y)
        }
    }

    /// Returns the dimension of the linear system
    ///
    /// ```text
    /// dim =  nx * ny
    /// ```
    pub fn dim(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns the number of prescribed equations
    ///
    /// The number of prescribed equations is equal to the number of nodes with essential conditions.
    pub fn num_prescribed(&self) -> usize {
        self.essential.len()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{PdeDiscreteLaplacian2d, Side};
    use russell_lab::{mat_approx_eq, Matrix};

    #[test]
    fn new_works() {
        let lap = PdeDiscreteLaplacian2d::new(7.0, 8.0, -1.0, 1.0, -3.0, 3.0, 2, 3).unwrap();
        assert_eq!(lap.xmin, -1.0);
        assert_eq!(lap.ymin, -3.0);
        assert_eq!(lap.nx, 2);
        assert_eq!(lap.ny, 3);
        assert_eq!(lap.dx, 2.0);
        assert_eq!(lap.dy, 3.0);
        assert_eq!(lap.left, &[0, 2, 4]);
        assert_eq!(lap.right, &[1, 3, 5]);
        assert_eq!(lap.bottom, &[0, 1]);
        assert_eq!(lap.top, &[4, 5]);
    }

    #[test]
    fn set_essential_boundary_condition_works() {
        let mut lap = PdeDiscreteLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        const LEF: f64 = 1.0;
        const RIG: f64 = 2.0;
        const BOT: f64 = 3.0;
        const TOP: f64 = 4.0;
        let lef = |_, _| LEF;
        let rig = |_, _| RIG;
        let bot = |_, _| BOT;
        let top = |_, _| TOP;
        lap.set_essential_boundary_condition(Side::Left, lef); //    0*   4   8  12*
        lap.set_essential_boundary_condition(Side::Right, rig); //   3*   7  11  15
        lap.set_essential_boundary_condition(Side::Bottom, bot); //  0*   1   2   3
        lap.set_essential_boundary_condition(Side::Top, top); //    12*  13  14  15*  (corner*)
        assert_eq!(lap.left, &[0, 4, 8, 12]);
        assert_eq!(lap.right, &[3, 7, 11, 15]);
        assert_eq!(lap.bottom, &[0, 1, 2, 3]);
        assert_eq!(lap.top, &[12, 13, 14, 15]);
        let mut res = Vec::new();
        lap.loop_over_prescribed_values(|i, value| res.push((i, value)));
        res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(
            res,
            &[
                (0, BOT),  // bottom* and left  (wins*)
                (1, BOT),  // bottom
                (2, BOT),  // bottom
                (3, BOT),  // bottom* and right
                (4, LEF),  // left
                (7, RIG),  // right
                (8, LEF),  // left
                (11, RIG), // right
                (12, TOP), // top* and left
                (13, TOP), // top
                (14, TOP), // top
                (15, TOP), // top* and right
            ]
        );
    }

    #[test]
    fn set_homogeneous_boundary_condition_works() {
        let mut lap = PdeDiscreteLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        lap.set_homogeneous_boundary_conditions();
        assert_eq!(lap.left, &[0, 4, 8, 12]);
        assert_eq!(lap.right, &[3, 7, 11, 15]);
        assert_eq!(lap.bottom, &[0, 1, 2, 3]);
        assert_eq!(lap.top, &[12, 13, 14, 15]);
        let mut res = Vec::new();
        lap.loop_over_prescribed_values(|i, value| res.push((i, value)));
        res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(
            res,
            &[
                (0, 0.0),
                (1, 0.0),
                (2, 0.0),
                (3, 0.0),
                (4, 0.0),
                (7, 0.0),
                (8, 0.0),
                (11, 0.0),
                (12, 0.0),
                (13, 0.0),
                (14, 0.0),
                (15, 0.0),
            ]
        );
    }

    #[test]
    fn coefficient_matrix_works() {
        let lap = PdeDiscreteLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let (aa, _) = lap.coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 9);
        assert_eq!(lap.num_prescribed(), 0);
        let ___ = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
            [-4.0,  2.0,  ___,  2.0,  ___,  ___,  ___,  ___,  ___],
            [ 1.0, -4.0,  1.0,  ___,  2.0,  ___,  ___,  ___,  ___],
            [ ___,  2.0, -4.0,  ___,  ___,  2.0,  ___,  ___,  ___],
            [ 1.0,  ___,  ___, -4.0,  2.0,  ___,  1.0,  ___,  ___],
            [ ___,  1.0,  ___,  1.0, -4.0,  1.0,  ___,  1.0,  ___],
            [ ___,  ___,  1.0,  ___,  2.0, -4.0,  ___,  ___,  1.0],
            [ ___,  ___,  ___,  2.0,  ___,  ___, -4.0,  2.0,  ___],
            [ ___,  ___,  ___,  ___,  2.0,  ___,  1.0, -4.0,  1.0],
            [ ___,  ___,  ___,  ___,  ___,  2.0,  ___,  2.0, -4.0],
        ]);
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
    }

    #[test]
    fn loop_over_molecule_works() {
        // ┌                            ┐
        // │ -4  2  .  2  .  .  .  .  . │  0
        // │  1 -4  1  .  2  .  .  .  . │  1
        // │  .  2 -4  .  .  2  .  .  . │  2
        // │  1  .  . -4  2  .  1  .  . │  3
        // │  .  1  .  1 -4  1  .  1  . │  4
        // │  .  .  1  .  2 -4  .  .  1 │  5
        // │  .  .  .  2  .  . -4  2  . │  6
        // │  .  .  .  .  2  .  1 -4  1 │  7
        // │  .  .  .  .  .  2  .  2 -4 │  8
        // └                            ┘
        //    0  1  2  3  4  5  6  7  8
        let lap = PdeDiscreteLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let mut row_0 = Vec::new();
        let mut row_4 = Vec::new();
        let mut row_8 = Vec::new();
        lap.loop_over_coef_mat_row(0, |j, aij| row_0.push((j, aij)));
        lap.loop_over_coef_mat_row(4, |j, aij| row_4.push((j, aij)));
        lap.loop_over_coef_mat_row(8, |j, aij| row_8.push((j, aij)));
        assert_eq!(row_0, &[(0, -4.0), (1, 1.0), (1, 1.0), (3, 1.0), (3, 1.0)]);
        assert_eq!(row_4, &[(4, -4.0), (3, 1.0), (5, 1.0), (1, 1.0), (7, 1.0)]);
        assert_eq!(row_8, &[(8, -4.0), (7, 1.0), (7, 1.0), (5, 1.0), (5, 1.0)]);
    }

    #[test]
    fn coefficient_matrix_with_essential_prescribed_works() {
        // The full matrix is:
        // ┌                                                 ┐
        // │ -4  2  .  .  2  .  .  .  .  .  .  .  .  .  .  . │  0 prescribed
        // │  1 -4  1  .  .  2  .  .  .  .  .  .  .  .  .  . │  1 prescribed
        // │  .  1 -4  1  .  .  2  .  .  .  .  .  .  .  .  . │  2 prescribed
        // │  .  .  2 -4  .  .  .  2  .  .  .  .  .  .  .  . │  3 prescribed
        // │  1  .  .  . -4  2  .  .  1  .  .  .  .  .  .  . │  4 prescribed
        // │  .  1  .  .  1 -4  1  .  .  1  .  .  .  .  .  . │  5
        // │  .  .  1  .  .  1 -4  1  .  .  1  .  .  .  .  . │  6
        // │  .  .  .  1  .  .  2 -4  .  .  .  1  .  .  .  . │  7 prescribed
        // │  .  .  .  .  1  .  .  . -4  2  .  .  1  .  .  . │  8 prescribed
        // │  .  .  .  .  .  1  .  .  1 -4  1  .  .  1  .  . │  9
        // │  .  .  .  .  .  .  1  .  .  1 -4  1  .  .  1  . │ 10
        // │  .  .  .  .  .  .  .  1  .  .  2 -4  .  .  .  1 │ 11 prescribed
        // │  .  .  .  .  .  .  .  .  2  .  .  . -4  2  .  . │ 12 prescribed
        // │  .  .  .  .  .  .  .  .  .  2  .  .  1 -4  1  . │ 13 prescribed
        // │  .  .  .  .  .  .  .  .  .  .  2  .  .  1 -4  1 │ 14 prescribed
        // │  .  .  .  .  .  .  .  .  .  .  .  2  .  .  2 -4 │ 15 prescribed
        // └                                                 ┘
        //    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        //    p  p  p  p  p        p  p        p  p  p  p  p
        let mut lap = PdeDiscreteLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        lap.set_homogeneous_boundary_conditions();
        let (aa, cc) = lap.coefficient_matrix().unwrap();
        assert_eq!(lap.dim(), 16);
        assert_eq!(lap.num_prescribed(), 12);
        const ___: f64 = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
             [ 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  0 prescribed
             [ ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  1 prescribed
             [ ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  2 prescribed
             [ ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  3 prescribed
             [ ___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  4 prescribed
             [ ___, ___, ___, ___, ___,-4.0, 1.0, ___, ___, 1.0, ___, ___, ___, ___, ___, ___], //  5
             [ ___, ___, ___, ___, ___, 1.0,-4.0, ___, ___, ___, 1.0, ___, ___, ___, ___, ___], //  6
             [ ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___], //  7 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___], //  8 prescribed
             [ ___, ___, ___, ___, ___, 1.0, ___, ___, ___,-4.0, 1.0, ___, ___, ___, ___, ___], //  9
             [ ___, ___, ___, ___, ___, ___, 1.0, ___, ___, 1.0,-4.0, ___, ___, ___, ___, ___], // 10
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___], // 11 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___], // 12 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___], // 13 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___], // 14 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0], // 15 prescribed
         ]); //  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
             //  p    p    p    p    p              p    p              p    p    p    p    p
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
        #[rustfmt::skip]
        let cc_correct = Matrix::from(&[
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  0 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  1 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  2 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  3 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  4 prescribed
             [ ___, 1.0, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  5
             [ ___, ___, 1.0, ___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___], //  6
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  7 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  8 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___, 1.0, ___, ___], //  9
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, 1.0, ___], // 10
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], // 11 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], // 12 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], // 13 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], // 14 prescribed
             [ ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], // 15 prescribed
         ]); //  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
             //  p    p    p    p    p              p    p              p    p    p    p    p
        mat_approx_eq(&cc.as_dense(), &cc_correct, 1e-15);
    }

    #[test]
    fn get_grid_coordinates_works() {
        let (nx, ny) = (2, 3);
        let lap = PdeDiscreteLaplacian2d::new(7.0, 8.0, -1.0, 1.0, -3.0, 3.0, nx, ny).unwrap();
        let mut xx = Matrix::new(ny, nx);
        let mut yy = Matrix::new(ny, nx);
        lap.loop_over_grid_points(|i, x, y| {
            let row = i / nx;
            let col = i % nx;
            xx.set(row, col, x);
            yy.set(row, col, y);
        });
        assert_eq!(
            format!("{}", xx),
            "┌       ┐\n\
             │ -1  1 │\n\
             │ -1  1 │\n\
             │ -1  1 │\n\
             └       ┘"
        );
        assert_eq!(
            format!("{}", yy),
            "┌       ┐\n\
             │ -3 -3 │\n\
             │  0  0 │\n\
             │  3  3 │\n\
             └       ┘"
        );
    }
}
