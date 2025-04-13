use super::Side2d;
use crate::StrError;
use russell_sparse::CooMatrix;

pub trait OperatorTrait<'a>: Send {
    /// Sets periodic boundary condition
    ///
    /// **Note:** Any essential boundary condition on the corresponding side will be removed.
    fn set_periodic_boundary_condition(&mut self, along_x: bool, along_y: bool, along_z: bool);

    /// Sets essential (Dirichlet) boundary condition
    ///
    /// **Note:** Any periodic boundary condition on the corresponding side will be removed.
    fn set_essential_boundary_condition(&mut self, side: Side2d, f: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'a);

    /// Sets homogeneous boundary conditions (i.e., zero essential values at the borders)
    ///
    /// **Note:** Periodic boundary conditions will be removed.
    fn set_homogeneous_boundary_conditions(&mut self);

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
    /// **Important:** This function must be called after setting the essential boundary conditions.
    ///
    /// # Todo
    ///
    /// * Implement the symmetric version for solvers that can handle a triangular matrix storage.
    fn coefficient_matrix(&self) -> Result<(CooMatrix, CooMatrix), StrError>;

    /// Executes a loop over one row of the coefficient matrix 'A' of A ⋅ X = B
    ///
    /// Note that some column indices may appear repeated; e.g. due to the zero-flux boundaries.
    ///
    /// # Input
    ///
    /// * `m` -- the row of the coefficient matrix
    /// * `callback` -- a `function(n, Amn)` where `n` is the column index and
    ///   `Amn` is the m-n-element of the coefficient matrix
    fn loop_over_coef_mat_row(&self, m: usize, callback: impl FnMut(usize, f64));

    /// Executes a loop over the prescribed values
    ///
    /// # Input
    ///
    /// * `callback` -- a `function(m, value)` where `m` is the row index and
    ///   `value` is the prescribed value.
    fn loop_over_prescribed_values(&self, callback: impl FnMut(usize, f64));

    /// Executes a loop over the grid points
    ///
    /// # Input
    ///
    /// * `callback` -- a function of `(m, x, y)` where `m` is the sequential point number,
    ///   and `(x, y)` are the Cartesian coordinates of the grid point.
    fn loop_over_grid_points(&self, callback: impl FnMut(usize, f64, f64, f64));

    /// Returns the dimension of the linear system
    fn dim(&self) -> usize;

    /// Returns the number of prescribed equations
    ///
    /// The number of prescribed equations is equal to the number of nodes with essential conditions.
    fn num_prescribed(&self) -> usize;
}
