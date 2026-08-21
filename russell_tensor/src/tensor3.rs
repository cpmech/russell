use super::{
    IJK_TO_MN_CASE_A, IJK_TO_MN_CASE_B, IJK_TO_MN_SYM_CASE_A, IJK_TO_MN_SYM_CASE_B, MN_TO_IJK_CASE_A, MN_TO_IJK_CASE_B,
    SQRT_2,
};
use crate::{Rep, StrError};
use russell_lab::{AsArray2D, Matrix};
use serde::{Deserialize, Serialize};
use std::cmp;
use std::fmt::{self, Write};

/// Defines a third-order tensor in R³×R³×R³
///
/// The matrix representation of Tensor3 results in a rectangular matrix.
/// Therefore, two cases are considered here:
///
/// Case A: Tensor3 applied to a Tensor1 (vector) yielding a Tensor2
/// Case B: Tensor3 applied to a Tensor2 yielding a Tensor1 (vector)
///
/// Symbolically:
///
/// ```text
/// Case A =>  T = H . u   (Tᵢⱼ = Σ_k Hᵢⱼₖ uₖ)
/// Case B =>  v = M : S   (vᵢ = Σ_j Σ_k Mᵢⱼₖ Sⱼₖ)
/// ```
///
/// where `T` and `S` are second-order tensors, `H` and `M` are third-order tensors
/// and `u` and `v` are first-order tensors (vectors).
///
/// The matrix representations associated with the two cases are
/// (symmetry here means minor-symmetry):
///
/// ```text
/// Rep::General:
///   Case A =>  [T]_(9×1) = [H]_(9×3) * [u]_(3×1)
///   Case B =>  [v]_(3×1) = [M]_(3×9) * [S]_(9×1)
/// ```
///
/// ```text
/// Rep::Symmetric:
///   Case A =>  [T]_(6×1) = [H]_(6×3) * [u]_(3×1)
///   Case B =>  [v]_(3×1) = [M]_(3×6) * [S]_(6×1)
/// ```
///
/// ```text
/// Rep::Symmetric2D:
///   Case A =>  [T]_(4×1) = [H]_(4×3) * [u]_(3×1)
///   Case B =>  [v]_(3×1) = [M]_(3×4) * [S]_(4×1)
/// ```
///
/// Note that the first-order tensors (vectors) are always given by the standard
/// components in 3D. All functions here require vectors such as `[u] = {u0, u1, u2}`.
///
/// # Standard and Kelvin components
///
/// The methods of this struct follow a naming convention that distinguishes
/// between the **standard** (Cartesian) components `Hᵢⱼₖ` and the **Kelvin**
/// components stored internally:
///
/// * Methods dealing with **standard components** carry the `std` qualifier in
///   their names (e.g., [Tensor3::from_std_matrix], [Tensor3::get_std],
///   [Tensor3::as_std_matrix], [Tensor3::sym_set_std]).
/// * Methods dealing directly with the **Kelvin components** carry no qualifier
///   (e.g., [Tensor3::get], [Tensor3::set], [Tensor3::set_tensor],
///   [Tensor3::update]).
///
/// Internally, the components are converted to the Kelvin basis as follows.
///
/// The Kelvin components Ĥijk are calculated from the standard components Hijk
/// using the following expression for Case A:
///
/// ```text
/// Case A:
///        ⎧ Hijk                if i = j
/// Ĥijk = ⎨ (Hijk + Hjik) / √2  if i < j
///        ⎩ (Hjik - Hijk) / √2  if i > j
/// ```
///
/// The Kelvin components Ĥijk are calculated from the standard components Hijk
/// using the following expression for Case B:
///
/// ```text
/// Case B:
///        ⎧ Hijk                if j = k
/// Ĥijk = ⎨ (Hijk + Hikj) / √2  if j < k
///        ⎩ (Hikj - Hijk) / √2  if j > k
/// ```
///
/// In Case A, minor-symmetry means Hijk = Hjik. Then, the mapping simplifies to:
///
/// ```text
/// Case A:
///        ⎧ Hijk     if i = j
/// Ĥijk = ⎨ Hijk √2  if i < j
///        ⎩ 0        if i > j
/// ```
///
/// In Case B, minor-symmetry means Hijk = Hikj. Then, the mapping simplifies to:
///
/// ```text
/// Case B:
///        ⎧ Hijk        if j = k
/// Ĥijk = ⎨ Hijk √2  if j < k
///        ⎩ 0        if j > k
/// ```
///
/// The components are organized in matrices:
/// * For Case A, the order of row indices, pairs (i,j) in (i,j,k), follow the same order used for Tensor2.
/// * For Case B, the order of column indices, pairs (j,k) in (i,j,k), follow the same order as the one for Tensor2.
///
/// The matrices are illustrated as follows.
///
/// [Rep::General]
///
/// ```text
/// Case A:
///      0 0   0 1   0 2
///    -----------------
/// 0 │ Ĥ000  Ĥ001  Ĥ002
/// 1 │ Ĥ110  Ĥ111  Ĥ112
/// 2 │ Ĥ220  Ĥ221  Ĥ222
///   │
/// 3 │ Ĥ010  Ĥ011  Ĥ012
/// 4 │ Ĥ120  Ĥ121  Ĥ122
/// 5 │ Ĥ020  Ĥ021  Ĥ022
///   │
/// 6 │ Ĥ100  Ĥ101  Ĥ102
/// 7 │ Ĥ210  Ĥ211  Ĥ212
/// 8 │ Ĥ200  Ĥ201  Ĥ202
///    -----------------
///      8 0   8 1   8 2
/// ```
///
/// ```text
/// Case B:
///      0 0  0 1  0 2  0 3  0 4  0 5  0 6  0 7  0 8
///    ---------------------------------------------
/// 0 │ Ĥ000 Ĥ011 Ĥ022 Ĥ001 Ĥ012 Ĥ002 Ĥ010 Ĥ021 Ĥ020
/// 1 │ Ĥ100 Ĥ111 Ĥ122 Ĥ101 Ĥ112 Ĥ102 Ĥ110 Ĥ121 Ĥ120
/// 2 │ Ĥ200 Ĥ211 Ĥ222 Ĥ201 Ĥ212 Ĥ202 Ĥ210 Ĥ221 Ĥ220
///    ---------------------------------------------
///      2 0  2 1  2 2  2 3  2 4  2 5  2 6  2 7  2 8
/// ```
///
/// [Rep::Symmetric]
///
/// ```text
/// Case A:
///      0 0      0 1      0 2    
///    --------------------------
/// 0 │ H000     H001     H002
/// 1 │ H110     H111     H112
/// 2 │ H220     H221     H222
///   │
/// 3 │ H010*√2  H011*√2  H012*√2
/// 4 │ H120*√2  H121*√2  H122*√2
/// 5 │ H020*√2  H021*√2  H022*√2
///    --------------------------
///      5 0      5 1      5 2
/// ```
///
/// ```text
/// Case B:
///      0 0  0 1  0 2  0 3     0 4     0 5
///    ---------------------------------------
/// 0 │ H000 H011 H022 H001*√2 H012*√2 H002*√2
/// 1 │ H100 H111 H122 H101*√2 H112*√2 H102*√2
/// 2 │ H200 H211 H222 H201*√2 H212*√2 H202*√2
///    ---------------------------------------
///      2 0  2 1  2 2  2 3     2 4     2 5
/// ```
///
/// [Rep::Symmetric2D]
///
/// ```text
/// Case A:
///      0 0      0 1      0 2    
///    --------------------------
/// 0 │ H000     H001     H002   
/// 1 │ H110     H111     H112   
/// 2 │ H220     H221     H222   
///   │
/// 3 │ H010*√2  H011*√2  H012*√2
///    --------------------------
///      3 0      3 1      3 2
/// ```
///
/// ```text
/// Case B:
///      0 0  0 1  0 2  0 3   
///    -----------------------
/// 0 │ H000 H011 H022 H001*√2
/// 1 │ H100 H111 H122 H101*√2
/// 2 │ H200 H211 H222 H201*√2
///    -----------------------
///      2 0  2 1  2 2  2 3
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tensor3 {
    /// Indicates Case A; otherwise Case B
    case_a: bool,

    /// Holds the actual number of rows of the Kelvin matrix
    ///
    /// Case A:
    /// * General: `nrow = 9`
    /// * Symmetric: `nrow = 6`
    /// * Symmetric2D: `nrow = 4`
    ///
    /// Case B:
    /// * General: `nrow = 3`
    /// * Symmetric: `nrow = 3`
    /// * Symmetric2D: `nrow = 3`
    nrow: usize,

    /// Holds the actual number of columns of the Kelvin matrix
    ///
    /// Case A:
    /// * General: `ncol = 3`
    /// * Symmetric: `ncol = 3`
    /// * Symmetric2D: `ncol = 3`
    ///
    /// Case B:
    /// * General: `ncol = 9`
    /// * Symmetric: `ncol = 6`
    /// * Symmetric2D: `ncol = 4`
    ncol: usize,

    /// Holds the components in Kelvin basis as matrix (heap).
    ///
    /// Heap version => dynamically allocated memory
    #[cfg(feature = "heap")]
    pub(crate) mat: Matrix,

    /// Holds the components in Kelvin basis as matrix (stack).
    ///
    /// Stack version => fixed size memory
    ///
    /// This array may use more data than necessary in symmetric cases
    #[cfg(not(feature = "heap"))]
    pub(crate) mat: [[f64; 9]; 9],

    /// Holds the Rep (representation) enum
    rep: Rep,

    /// Enables the loop-based implementation (instead of the unrolled one)
    ///
    /// **Note:** This field is temporary and will be removed in a future version.
    pub use_loops: bool,
}

impl Tensor3 {
    /// Creates a new (zeroed) Tensor3
    ///
    /// # Input
    ///
    /// * `rep` -- the [Rep] representation
    /// * `case_a` -- Case A instead of Case B
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Rep, StrError, Tensor3};
    ///
    /// fn main() {
    ///     let cc = Tensor3::new(Rep::General, true);
    ///     assert_eq!(cc.dims(), (9, 3));
    ///
    ///     let dd = Tensor3::new(Rep::Symmetric, true);
    ///     assert_eq!(dd.dims(), (6, 3));
    ///
    ///     let ee = Tensor3::new(Rep::Symmetric2D, true);
    ///     assert_eq!(ee.dims(), (4, 3));
    /// }
    /// ```
    pub fn new(rep: Rep, case_a: bool) -> Self {
        let (nrow, ncol) = if case_a { (rep.dim(), 3) } else { (3, rep.dim()) };
        #[cfg(feature = "heap")]
        {
            Tensor3 {
                case_a,
                nrow,
                ncol,
                mat: Matrix::new(nrow, ncol),
                rep,
                use_loops: false,
            }
        }
        #[cfg(not(feature = "heap"))]
        {
            Tensor3 {
                case_a,
                nrow,
                ncol,
                mat: [[0.0; 9]; 9],
                rep,
                use_loops: false,
            }
        }
    }

    /// Allocates a minor-symmetric Tensor3
    pub fn new_sym(two_dim: bool, case_a: bool) -> Self {
        if two_dim {
            Tensor3::new(Rep::Symmetric2D, case_a)
        } else {
            Tensor3::new(Rep::Symmetric, case_a)
        }
    }

    /// Allocates a minor-symmetric Tensor3 given the space dimension
    ///
    /// **Note:** `space_ndim` must be 2 or 3 (only 2 is checked, otherwise 3 is assumed)
    pub fn new_sym_ndim(space_ndim: usize, case_a: bool) -> Self {
        if space_ndim == 2 {
            Tensor3::new(Rep::Symmetric2D, case_a)
        } else {
            Tensor3::new(Rep::Symmetric, case_a)
        }
    }

    /// Returns the representation associated with this Tensor3
    #[inline]
    pub fn rep(&self) -> Rep {
        self.rep
    }

    /// Returns whether the matrix representation adopted corresponds to Case A
    #[inline]
    pub fn is_case_a(&self) -> bool {
        self.case_a
    }

    /// Returns the Kelvin matrix dimension (nrow, ncol)
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    /// Returns the (m,n) component of the Kelvin matrix
    ///
    /// # Input
    ///
    /// Check the range of indices by calling [Tensor3::dims()]
    ///
    /// * `m` -- the row index
    /// * `n` -- the column index
    ///
    /// # Panics
    ///
    /// A panic will occur if the indices are out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Rep, Tensor3};
    ///
    /// let mut dd = Tensor3::new(Rep::General, true);
    /// dd.set(0, 0, 123.0);
    /// assert_eq!(dd.get(0, 0), 123.0);
    /// ```
    #[inline]
    pub fn get(&self, m: usize, n: usize) -> f64 {
        #[cfg(feature = "heap")]
        {
            self.mat.get(m, n)
        }
        #[cfg(not(feature = "heap"))]
        {
            self.mat[m][n]
        }
    }

    /// Sets the (m,n) component of the Kelvin matrix
    ///
    /// # Input
    ///
    /// Check the range of indices by calling [Tensor3::dims()]
    ///
    /// * `m` -- the row index
    /// * `n` -- the column index
    /// * `value` -- the value to set
    ///
    /// # Panics
    ///
    /// A panic will occur if the indices are out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Rep, Tensor3};
    ///
    /// let mut dd = Tensor3::new(Rep::General, true);
    /// dd.set(0, 0, 123.0);
    /// assert_eq!(dd.get(0, 0), 123.0);
    /// ```
    #[inline]
    pub fn set(&mut self, m: usize, n: usize, value: f64) {
        #[cfg(feature = "heap")]
        {
            self.mat.set(m, n, value);
        }
        #[cfg(not(feature = "heap"))]
        {
            self.mat[m][n] = value;
        }
    }

    /// Sets this tensor from a nested array containing the standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard Dijk components with respect to an orthonormal Cartesian basis
    pub fn set_std_array(&mut self, inp: &[[[f64; 3]; 3]; 3]) -> Result<(), StrError> {
        let dim = self.rep.dim();
        if self.case_a {
            if dim == 4 || dim == 6 {
                let max = if dim == 4 { 3 } else { 6 };
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            // check minor-symmetry on (i,j)
                            if i > j {
                                if inp[i][j][k] != inp[j][i][k] {
                                    return Err("the input data does not correspond to a minor-symmetric tensor");
                                }
                            } else {
                                let (m, n) = IJK_TO_MN_CASE_A[i][j][k];
                                if m > max {
                                    if inp[i][j][k] != 0.0 {
                                        return Err(
                                            "the input data does not correspond to a 2D minor-symmetric tensor",
                                        );
                                    }
                                    continue;
                                } else if m < 3 {
                                    self.set(m, n, inp[i][j][k]);
                                } else {
                                    self.set(m, n, SQRT_2 * inp[i][j][k]);
                                }
                            }
                        }
                    }
                }
            } else {
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_A[i][j][k];
                            // ** i == j **
                            if i == j {
                                self.set(m, n, inp[i][j][k]);
                            // ** i < j **
                            } else if i < j {
                                self.set(m, n, (inp[i][j][k] + inp[j][i][k]) / SQRT_2);
                            // ** i > j **
                            } else if i > j {
                                self.set(m, n, (inp[j][i][k] - inp[i][j][k]) / SQRT_2);
                            }
                        }
                    }
                }
            }
        } else {
            if dim == 4 || dim == 6 {
                let max = if dim == 4 { 3 } else { 6 };
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            // check minor-symmetry on (j,k)
                            if j > k {
                                if inp[i][j][k] != inp[i][k][j] {
                                    return Err("the input data does not correspond to a minor-symmetric tensor");
                                }
                            } else {
                                let (m, n) = IJK_TO_MN_CASE_B[i][j][k];
                                if n > max {
                                    if inp[i][j][k] != 0.0 {
                                        return Err(
                                            "the input data does not correspond to a 2D minor-symmetric tensor",
                                        );
                                    }
                                    continue;
                                } else if n < 3 {
                                    self.set(m, n, inp[i][j][k]);
                                } else {
                                    self.set(m, n, SQRT_2 * inp[i][j][k]);
                                }
                            }
                        }
                    }
                }
            } else {
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_B[i][j][k];
                            // ** j == k **
                            if j == k {
                                self.set(m, n, inp[i][j][k]);
                            // ** j < k **
                            } else if j < k {
                                self.set(m, n, (inp[i][j][k] + inp[i][k][j]) / SQRT_2);
                            // ** j > k **
                            } else if j > k {
                                self.set(m, n, (inp[i][k][j] - inp[i][j][k]) / SQRT_2);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Creates a new Tensor3 constructed from a nested array containing the standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard Dijk components with respect to an orthonormal Cartesian basis
    /// * `rep` -- the [Rep] representation
    /// * `case_a` -- Case A instead of Case B
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Rep, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[[0.0; 3]; 3]; 3];
    ///     for i in 0..3 {
    ///         for j in 0..3 {
    ///             for k in 0..3 {
    ///                 inp[i][j][k] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             }
    ///         }
    ///     }
    ///     let dd = Tensor3::from_std_array(&inp, Rep::General, true)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌             ┐\n\
    ///          │ 111 112 113 │\n\
    ///          │ 221 222 223 │\n\
    ///          │ 331 332 333 │\n\
    ///          │ 121 122 123 │\n\
    ///          │ 231 232 233 │\n\
    ///          │ 131 132 133 │\n\
    ///          │ 211 212 213 │\n\
    ///          │ 321 322 323 │\n\
    ///          │ 311 312 313 │\n\
    ///          └             ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn from_std_array(inp: &[[[f64; 3]; 3]; 3], rep: Rep, case_a: bool) -> Result<Self, StrError> {
        let mut res = Tensor3::new(rep, case_a);
        res.set_std_array(inp)?;
        Ok(res)
    }

    /// Sets this tensor from a matrix with standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard matrix of components with respect to an orthonormal Cartesian basis.
    ///   The matrix must be 9x3 for Case A or 3x9 for Case B
    ///   even if it corresponds to a minor-symmetric tensor.
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix has the incorrect dimensions:
    /// * Case A: 9x3
    /// * Case B: 3x9
    pub fn set_std_matrix<'a, S>(&mut self, inp: &'a S) -> Result<(), StrError>
    where
        S: AsArray2D<'a, f64>,
    {
        let dim = self.rep.dim();
        if self.case_a {
            if dim == 4 || dim == 6 {
                let max = if dim == 4 { 3 } else { 6 };
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_A[i][j][k];
                            let (r, s) = IJK_TO_MN_CASE_A[j][i][k];
                            // check minor-symmetry
                            if i > j {
                                if inp.at(m, n) != inp.at(r, s) {
                                    return Err("the input data does not correspond to a minor-symmetric tensor");
                                }
                            } else {
                                if m > max {
                                    if inp.at(m, n) != 0.0 {
                                        return Err(
                                            "the input data does not correspond to a 2D minor-symmetric tensor",
                                        );
                                    }
                                    continue;
                                } else if m < 3 {
                                    self.set(m, n, inp.at(m, n));
                                } else {
                                    self.set(m, n, SQRT_2 * inp.at(m, n));
                                }
                            }
                        }
                    }
                }
            } else {
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_A[i][j][k];
                            // ** i == j **
                            if i == j {
                                self.set(m, n, inp.at(m, n));
                            // ** i < j **
                            } else if i < j {
                                let (r, s) = IJK_TO_MN_CASE_A[j][i][k];
                                self.set(m, n, (inp.at(m, n) + inp.at(r, s)) / SQRT_2);
                            // ** i > j **
                            } else if i > j {
                                let (r, s) = IJK_TO_MN_CASE_A[j][i][k];
                                self.set(m, n, (inp.at(r, s) - inp.at(m, n)) / SQRT_2);
                            }
                        }
                    }
                }
            }
        } else {
            if dim == 4 || dim == 6 {
                let max = if dim == 4 { 3 } else { 6 };
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_B[i][j][k];
                            let (r, s) = IJK_TO_MN_CASE_B[i][k][j];
                            // check minor-symmetry
                            if j > k {
                                if inp.at(m, n) != inp.at(r, s) {
                                    return Err("the input data does not correspond to a minor-symmetric tensor");
                                }
                            } else {
                                if n > max {
                                    if inp.at(m, n) != 0.0 {
                                        return Err(
                                            "the input data does not correspond to a 2D minor-symmetric tensor",
                                        );
                                    }
                                    continue;
                                } else if n < 3 {
                                    self.set(m, n, inp.at(m, n));
                                } else {
                                    self.set(m, n, SQRT_2 * inp.at(m, n));
                                }
                            }
                        }
                    }
                }
            } else {
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_B[i][j][k];
                            // ** j == k **
                            if j == k {
                                self.set(m, n, inp.at(m, n));
                            // ** j < k **
                            } else if j < k {
                                let (r, s) = IJK_TO_MN_CASE_B[i][k][j];
                                self.set(m, n, (inp.at(m, n) + inp.at(r, s)) / SQRT_2);
                            // ** j > k **
                            } else if j > k {
                                let (r, s) = IJK_TO_MN_CASE_B[i][k][j];
                                self.set(m, n, (inp.at(r, s) - inp.at(m, n)) / SQRT_2);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Creates a new Tensor3 constructed from a matrix with standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard matrix of components with respect to an orthonormal Cartesian basis.
    ///   The matrix must be 9x3 for Case A or 3x9 for Case B
    ///   even if it corresponds to a minor-symmetric tensor.
    /// * `rep` -- the [Rep] representation
    /// * `case_a` -- Case A instead of Case B
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix has the incorrect dimensions:
    /// * Case A: 9x3
    /// * Case B: 3x9
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Rep, MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General, true)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌             ┐\n\
    ///          │ 111 112 113 │\n\
    ///          │ 221 222 223 │\n\
    ///          │ 331 332 333 │\n\
    ///          │ 121 122 123 │\n\
    ///          │ 231 232 233 │\n\
    ///          │ 131 132 133 │\n\
    ///          │ 211 212 213 │\n\
    ///          │ 321 322 323 │\n\
    ///          │ 311 312 313 │\n\
    ///          └             ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn from_std_matrix<'a, S>(inp: &'a S, rep: Rep, case_a: bool) -> Result<Self, StrError>
    where
        S: AsArray2D<'a, f64>,
    {
        let mut res = Tensor3::new(rep, case_a);
        res.set_std_matrix(inp)?;
        Ok(res)
    }

    /// Returns the (i,j,k) standard component
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Rep, MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General, true)?;
    ///
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             let val = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             approx_eq(dd.get_std(i,j,k), val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn get_std(&self, i: usize, j: usize, k: usize) -> f64 {
        if self.case_a {
            match self.nrow {
                4 => {
                    let (m, n) = IJK_TO_MN_SYM_CASE_A[i][j][k];
                    if m > 3 {
                        0.0
                    } else if m < 3 {
                        self.get(m, n)
                    } else {
                        self.get(m, n) / SQRT_2
                    }
                }
                6 => {
                    let (m, n) = IJK_TO_MN_SYM_CASE_A[i][j][k];
                    if m < 3 { self.get(m, n) } else { self.get(m, n) / SQRT_2 }
                }
                _ => {
                    let (m, n) = IJK_TO_MN_CASE_A[i][j][k];
                    let val = self.get(m, n);
                    // ** i == j **
                    if i == j {
                        val
                    // ** i < j **
                    } else if i < j {
                        let (r, s) = IJK_TO_MN_CASE_A[j][i][k];
                        let other = self.get(r, s);
                        (val + other) / SQRT_2
                    // ** i > j **
                    } else {
                        let (r, s) = IJK_TO_MN_CASE_A[j][i][k];
                        let other = self.get(r, s);
                        (other - val) / SQRT_2
                    }
                }
            }
        } else {
            match self.ncol {
                4 => {
                    let (m, n) = IJK_TO_MN_SYM_CASE_B[i][j][k];
                    if n > 3 {
                        0.0
                    } else if n < 3 {
                        self.get(m, n)
                    } else {
                        self.get(m, n) / SQRT_2
                    }
                }
                6 => {
                    let (m, n) = IJK_TO_MN_SYM_CASE_B[i][j][k];
                    if n < 3 { self.get(m, n) } else { self.get(m, n) / SQRT_2 }
                }
                _ => {
                    let (m, n) = IJK_TO_MN_CASE_B[i][j][k];
                    let val = self.get(m, n);
                    // ** j == k **
                    if j == k {
                        val
                    // ** j < k **
                    } else if j < k {
                        let (r, s) = IJK_TO_MN_CASE_B[i][k][j];
                        let other = self.get(r, s);
                        (val + other) / SQRT_2
                    // ** j > k **
                    } else {
                        let (r, s) = IJK_TO_MN_CASE_B[i][k][j];
                        let other = self.get(r, s);
                        (other - val) / SQRT_2
                    }
                }
            }
        }
    }

    /// Adds another tensor to this one
    ///
    /// ```text
    /// self += α other
    /// ```
    ///
    /// # Panics
    ///
    /// A panic will occur if the tensors have different [Rep].
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Rep, MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..4 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = 1.0;
    ///         }
    ///     }
    ///
    ///     let mut dd = Tensor3::new(Rep::General, true);
    ///     let ee = Tensor3::from_std_matrix(&inp, Rep::General, true)?;
    ///     dd.update(2.0, &ee);
    ///
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌       ┐\n\
    ///          │ 2 2 2 │\n\
    ///          │ 2 2 2 │\n\
    ///          │ 2 2 2 │\n\
    ///          │ 2 2 2 │\n\
    ///          │ 0 0 0 │\n\
    ///          │ 0 0 0 │\n\
    ///          │ 0 0 0 │\n\
    ///          │ 0 0 0 │\n\
    ///          │ 0 0 0 │\n\
    ///          └       ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn update(&mut self, alpha: f64, other: &Tensor3) {
        assert_eq!(other.rep, self.rep);
        assert_eq!(other.case_a, self.case_a);
        for m in 0..self.nrow {
            for n in 0..self.ncol {
                self.set(m, n, self.get(m, n) + alpha * other.get(m, n));
            }
        }
    }

    /// Returns a 3x3x3 array with the standard components
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Rep, MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General, true)?;
    ///     let arr = dd.as_std_array();
    ///
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             let val = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             approx_eq(arr[i][j][k], val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn as_std_array(&self) -> Vec<Vec<Vec<f64>>> {
        let mut dd = vec![vec![vec![0.0; 3]; 3]; 3];
        self.to_std_array(&mut dd);
        dd
    }

    /// Converts this tensor to a 3x3x3 array with the standard components
    ///
    /// # Panics
    ///
    /// A panic will occur if the array is not 3x3x3, i.e., `vec![vec![vec![0.0; 3]; 3]; 3]`
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Rep, MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General, true)?;
    ///     let mut arr = vec![vec![vec![0.0; 3]; 3]; 3];
    ///     dd.to_std_array(&mut arr);
    ///
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             let val = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             approx_eq(arr[i][j][k], val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn to_std_array(&self, dd: &mut Vec<Vec<Vec<f64>>>) {
        if self.case_a {
            if self.nrow == 9 {
                // General
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            dd[i][j][k] = self.get_std(i, j, k);
                        }
                    }
                }
            } else {
                // Symmetric / Symmetric2D
                for m in 0..self.nrow {
                    for n in 0..self.ncol {
                        let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
                        dd[i][j][k] = self.get_std(i, j, k);
                        if i != j {
                            dd[j][i][k] = dd[i][j][k];
                        }
                    }
                }
            }
        } else {
            if self.ncol == 9 {
                // General
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            dd[i][j][k] = self.get_std(i, j, k);
                        }
                    }
                }
            } else {
                // Symmetric / Symmetric2D
                for m in 0..self.nrow {
                    for n in 0..self.ncol {
                        let (i, j, k) = MN_TO_IJK_CASE_B[m][n];
                        dd[i][j][k] = self.get_std(i, j, k);
                        if j != k {
                            dd[i][k][j] = dd[i][j][k];
                        }
                    }
                }
            }
        }
    }

    /// Returns a matrix with the standard components
    ///
    /// **Note:** The matrix will have the standard components.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Rep, MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General, true)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌             ┐\n\
    ///          │ 111 112 113 │\n\
    ///          │ 221 222 223 │\n\
    ///          │ 331 332 333 │\n\
    ///          │ 121 122 123 │\n\
    ///          │ 231 232 233 │\n\
    ///          │ 131 132 133 │\n\
    ///          │ 211 212 213 │\n\
    ///          │ 321 322 323 │\n\
    ///          │ 311 312 313 │\n\
    ///          └             ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn as_std_matrix(&self) -> Matrix {
        let mut mat = if self.case_a {
            Matrix::new(9, 3)
        } else {
            Matrix::new(3, 9)
        };
        self.to_std_matrix(&mut mat);
        mat
    }

    /// Converts this tensor to a matrix with the standard components
    ///
    /// # Input
    ///
    /// * `mat` -- the resulting matrix
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix has the incorrect dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Matrix;
    /// use russell_tensor::{Rep, MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General, true)?;
    ///     let mut mat = Matrix::new(9, 3);
    ///     dd.to_std_matrix(&mut mat);
    ///     assert_eq!(
    ///         format!("{:.0}", mat),
    ///         "┌             ┐\n\
    ///          │ 111 112 113 │\n\
    ///          │ 221 222 223 │\n\
    ///          │ 331 332 333 │\n\
    ///          │ 121 122 123 │\n\
    ///          │ 231 232 233 │\n\
    ///          │ 131 132 133 │\n\
    ///          │ 211 212 213 │\n\
    ///          │ 321 322 323 │\n\
    ///          │ 311 312 313 │\n\
    ///          └             ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn to_std_matrix(&self, mat: &mut Matrix) {
        if self.case_a {
            assert_eq!(mat.dims(), (9, 3));
            for m in 0..9 {
                for n in 0..3 {
                    let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
                    mat.set(m, n, self.get_std(i, j, k));
                }
            }
        } else {
            assert_eq!(mat.dims(), (3, 9));
            for m in 0..3 {
                for n in 0..9 {
                    let (i, j, k) = MN_TO_IJK_CASE_B[m][n];
                    mat.set(m, n, self.get_std(i, j, k));
                }
            }
        }
    }

    /// Sets the (i,j,k) standard component of a minor-symmetric Tensor3
    ///
    /// # Notes
    ///
    /// 1. The tensor must be symmetric and (i,j) must correspond to the possible
    ///    combination due to the space dimension, otherwise a panic may occur.
    ///
    /// # Panics
    ///
    /// 1. A panic will occur if the tensor is [Rep::General]
    /// 2. A panic will occur if the indices are out of range
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Rep, MN_TO_IJK_CASE_A, Tensor3};
    ///
    /// fn main() {
    ///     let mut dd = Tensor3::new(Rep::Symmetric2D, true);
    ///     for m in 0..4 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             let value = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             dd.sym_set_std(i, j, k, value);
    ///         }
    ///     }
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌             ┐\n\
    ///          │ 111 112 113 │\n\
    ///          │ 221 222 223 │\n\
    ///          │ 331 332 333 │\n\
    ///          │ 121 122 123 │\n\
    ///          │   0   0   0 │\n\
    ///          │   0   0   0 │\n\
    ///          │ 121 122 123 │\n\
    ///          │   0   0   0 │\n\
    ///          │   0   0   0 │\n\
    ///          └             ┘"
    ///     );
    /// }
    /// ```
    pub fn sym_set_std(&mut self, i: usize, j: usize, k: usize, value: f64) {
        assert!(self.rep != Rep::General);
        if self.case_a {
            let (m, n) = IJK_TO_MN_SYM_CASE_A[i][j][k];
            if m < 3 {
                self.set(m, n, value);
            } else {
                self.set(m, n, value * SQRT_2);
            }
        } else {
            let (m, n) = IJK_TO_MN_SYM_CASE_B[i][j][k];
            if n < 3 {
                self.set(m, n, value);
            } else {
                self.set(m, n, value * SQRT_2);
            }
        }
    }

    /// Makes this tensor equal to another tensor, scaled by a factor alpha
    ///
    /// ```text
    /// self := α other
    /// ```
    ///
    /// # Panics
    ///
    /// A panic will occur if the tensors have different [Rep].
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::mat_approx_eq;
    /// use russell_tensor::{Rep, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let data = &[
    ///         [  1.0,  2.0,  3.0],
    ///         [ -1.0, -2.0, -3.0],
    ///         [  2.0,  4.0,  6.0],
    ///         [ 10.0, 20.0, 30.0],
    ///         [  0.0,  0.0,  0.0],
    ///         [  0.0,  0.0,  0.0],
    ///         [ -2.0, -4.0, -6.0],
    ///         [  0.0,  0.0,  0.0],
    ///         [  0.0,  0.0,  0.0],
    ///     ];
    ///     let dd = Tensor3::from_std_matrix(data, Rep::General, true)?;
    ///     let mut ee = Tensor3::new(Rep::General, true);
    ///
    ///     ee.set_tensor(1.0, &dd);
    ///
    ///     mat_approx_eq(&dd.as_std_matrix(), data, 1e-14);
    ///     Ok(())
    /// }
    /// ```
    pub fn set_tensor(&mut self, alpha: f64, other: &Tensor3) {
        assert_eq!(other.rep, self.rep);
        assert_eq!(other.case_a, self.case_a);
        for m in 0..self.nrow {
            for n in 0..self.ncol {
                self.set(m, n, alpha * other.get(m, n));
            }
        }
    }

    /// Returns the permutation (Levi-Civita) tensor
    ///
    /// This function is only available for [Rep::General]
    pub fn constant_permutation(case_a: bool) -> Self {
        let pos_one = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]; // even cyclic permutation
        let neg_one = [(0, 2, 1), (1, 0, 2), (2, 1, 0)]; // odd cyclic permutation
        let mut std_array = [[[0.0; 3]; 3]; 3];
        for (i, j, k) in pos_one {
            std_array[i][j][k] = 1.0;
        }
        for (i, j, k) in neg_one {
            std_array[i][j][k] = -1.0;
        }
        Tensor3::from_std_array(&std_array, Rep::General, case_a).unwrap()
    }
}

impl fmt::Display for Tensor3 {
    /// Generates a string representation of Kelvin matrix associated with this Tensor3
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                let val = self.get(i, j);
                match f.precision() {
                    Some(v) => write!(&mut buf, "{:.1$}", val, v).unwrap(),
                    None => write!(&mut buf, "{}", val).unwrap(),
                }
                width = cmp::max(buf.chars().count(), width);
                buf.clear();
            }
        }
        // draw matrix
        width += 1;
        write!(f, "┌{:1$}┐\n", " ", width * self.ncol + 1).unwrap();
        for i in 0..self.nrow {
            if i > 0 {
                write!(f, " │\n").unwrap();
            }
            for j in 0..self.ncol {
                if j == 0 {
                    write!(f, "│").unwrap();
                }
                let val = self.get(i, j);
                match f.precision() {
                    Some(v) => write!(f, "{:>1$.2$}", val, width, v).unwrap(),
                    None => write!(f, "{:>1$}", val, width).unwrap(),
                }
            }
        }
        write!(f, " │\n").unwrap();
        write!(f, "└{:1$}┘", " ", width * self.ncol + 1).unwrap();
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{MN_TO_IJK_CASE_A, Tensor3};
    use crate::{Rep, SQRT_2, SamplesTensor3};
    use russell_lab::{Matrix, approx_eq, mat_approx_eq};

    #[test]
    fn new_set_and_get_work() {
        // general
        let mut dd = Tensor3::new(Rep::General, true);
        dd.set(0, 0, 123.0);
        assert_eq!(dd.dims(), (9, 3));
        assert_eq!(dd.rep(), Rep::General);
        assert_eq!(dd.get(0, 0), 123.0);

        // symmetric
        let mut dd = Tensor3::new(Rep::Symmetric, true);
        dd.set(0, 0, 123.0);
        assert_eq!(dd.rep(), Rep::Symmetric);
        assert_eq!(dd.dims(), (6, 3));
        assert_eq!(dd.get(0, 0), 123.0);

        let mut dd = Tensor3::new_sym(false, true);
        dd.set(0, 0, 123.0);
        assert_eq!(dd.rep(), Rep::Symmetric);
        assert_eq!(dd.dims(), (6, 3));
        assert_eq!(dd.get(0, 0), 123.0);

        let mut dd = Tensor3::new_sym_ndim(3, true);
        dd.set(0, 0, 123.0);
        assert_eq!(dd.rep(), Rep::Symmetric);
        assert_eq!(dd.dims(), (6, 3));
        assert_eq!(dd.get(0, 0), 123.0);

        // symmetric 2d
        let mut dd = Tensor3::new(Rep::Symmetric2D, true);
        dd.set(0, 0, 123.0);
        assert_eq!(dd.rep(), Rep::Symmetric2D);
        assert_eq!(dd.dims(), (4, 3));
        assert_eq!(dd.get(0, 0), 123.0);

        let mut dd = Tensor3::new_sym(true, true);
        dd.set(0, 0, 123.0);
        assert_eq!(dd.rep(), Rep::Symmetric2D);
        assert_eq!(dd.dims(), (4, 3));
        assert_eq!(dd.get(0, 0), 123.0);

        let mut dd = Tensor3::new_sym_ndim(2, true);
        dd.set(0, 0, 123.0);
        assert_eq!(dd.rep(), Rep::Symmetric2D);
        assert_eq!(dd.dims(), (4, 3));
        assert_eq!(dd.get(0, 0), 123.0);
    }

    #[test]
    fn from_std_array_fails_captures_errors() {
        let res = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1, Rep::Symmetric, true);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );

        let res = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1, Rep::Symmetric2D, true);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn from_std_array_works() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1, Rep::General, true).unwrap();
        for m in 0..9 {
            for n in 0..3 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_A_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 3d
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1, Rep::Symmetric, true).unwrap();
        for m in 0..6 {
            for n in 0..3 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_A_SYM_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 2d
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1, Rep::Symmetric2D, true).unwrap();
        for m in 0..4 {
            for n in 0..3 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_A_SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }
    }

    #[test]
    fn from_std_matrix_fails_captures_errors() {
        let mut inp = [[0.0; 3]; 9];
        inp[3][0] = 1e-15;
        let res = Tensor3::from_std_matrix(&inp, Rep::Symmetric, true);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );

        inp[3][0] = 0.0;
        inp[4][0] = 1.0;
        inp[7][0] = 1.0;
        let res = Tensor3::from_std_matrix(&inp, Rep::Symmetric2D, true);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn get_and_set_work() {
        let mut dd = Tensor3::new(Rep::Symmetric2D, true);
        assert_eq!(dd.get(0, 0), 0.0);
        dd.set(0, 0, 2.0);
        assert_eq!(dd.get(0, 0), 2.0);
    }

    #[test]
    fn from_std_matrix_works() {
        // general
        let dd = Tensor3::from_std_matrix(&SamplesTensor3::CASE_A_SAMPLE1_STD_MATRIX, Rep::General, true).unwrap();
        let (nrow, ncol) = dd.dims();
        for m in 0..nrow {
            for n in 0..ncol {
                approx_eq(dd.get(m, n), SamplesTensor3::CASE_A_SAMPLE1_KELVIN_MATRIX[m][n], 1e-15);
            }
        }

        // symmetric 3D
        let dd =
            Tensor3::from_std_matrix(&SamplesTensor3::CASE_A_SYM_SAMPLE1_STD_MATRIX, Rep::Symmetric, true).unwrap();
        let (nrow, ncol) = dd.dims();
        for m in 0..nrow {
            for n in 0..ncol {
                approx_eq(
                    dd.get(m, n),
                    SamplesTensor3::CASE_A_SYM_SAMPLE1_KELVIN_MATRIX[m][n],
                    1e-14,
                );
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_matrix(
            &SamplesTensor3::CASE_A_SYM_2D_SAMPLE1_STD_MATRIX,
            Rep::Symmetric2D,
            true,
        )
        .unwrap();
        let (nrow, ncol) = dd.dims();
        for m in 0..nrow {
            for n in 0..ncol {
                approx_eq(
                    dd.get(m, n),
                    SamplesTensor3::CASE_A_SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n],
                    1e-14,
                );
            }
        }
    }

    #[test]
    fn get_std_works() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1, Rep::General, true).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::CASE_A_SAMPLE1[i][j][k], 1e-13);
                }
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1, Rep::Symmetric, true).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::CASE_A_SYM_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1, Rep::Symmetric2D, true).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(
                        dd.get_std(i, j, k),
                        SamplesTensor3::CASE_A_SYM_2D_SAMPLE1[i][j][k],
                        1e-14,
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn update_panics_on_incorrect_input() {
        let mut dd = Tensor3::new(Rep::Symmetric2D, true);
        let ee = Tensor3::new(Rep::Symmetric, true);
        dd.update(2.0, &ee);
    }

    #[test]
    fn update_works() {
        let mut dd = Tensor3::new(Rep::Symmetric2D, true);
        let ee = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1, Rep::Symmetric2D, true).unwrap();
        dd.update(2.0, &ee);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(
                        dd.get_std(i, j, k),
                        2.0 * SamplesTensor3::CASE_A_SYM_2D_SAMPLE1[i][j][k],
                        1e-14,
                    );
                }
            }
        }
    }

    #[test]
    fn as_std_array_and_to_std_array_work() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1, Rep::General, true).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_A_SAMPLE1[i][j][k], 1e-13);
                }
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1, Rep::Symmetric, true).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_A_SYM_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1, Rep::Symmetric2D, true).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_A_SYM_2D_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }
    }

    #[test]
    fn as_std_matrix_and_to_std_matrix_work() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1, Rep::General, true).unwrap();
        let mat = dd.as_std_matrix();
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(mat.get(m, n), SamplesTensor3::CASE_A_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1, Rep::Symmetric, true).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (9, 3));
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(
                    mat.get(m, n),
                    SamplesTensor3::CASE_A_SYM_SAMPLE1_STD_MATRIX[m][n],
                    1e-13,
                );
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1, Rep::Symmetric2D, true).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (9, 3));
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(
                    mat.get(m, n),
                    SamplesTensor3::CASE_A_SYM_2D_SAMPLE1_STD_MATRIX[m][n],
                    1e-13,
                );
            }
        }
    }

    #[test]
    fn from_std_array_to_std_matrix_from_std_matrix_work() {
        // General
        #[rustfmt::skip]
        let data = &[
            [
                [ 18.0,  16.0,  14.0],
                [ 36.0,  32.0,  28.0],
                [ 54.0,  48.0,  42.0],
            ],
            [
                [ 72.0,  64.0,  56.0],
                [ 90.0,  80.0,  70.0],
                [108.0,  96.0,  84.0],
            ],
            [
                [126.0, 112.0,  98.0],
                [144.0, 128.0, 112.0],
                [162.0, 144.0, 126.0],
            ],
        ];
        let dd = Tensor3::from_std_array(data, Rep::General, true).unwrap();
        let m1 = dd.as_std_matrix();
        #[rustfmt::skip]
        let correct = &[
            [ 18.0,  16.0,  14.0],
            [ 90.0,  80.0,  70.0],
            [162.0, 144.0, 126.0],
            [ 36.0,  32.0,  28.0],
            [108.0,  96.0,  84.0],
            [ 54.0,  48.0,  42.0],
            [ 72.0,  64.0,  56.0],
            [144.0, 128.0, 112.0],
            [126.0, 112.0,  98.0],
        ];
        mat_approx_eq(&m1, correct, 1e-13);
        let ee = Tensor3::from_std_matrix(correct, Rep::General, true).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, correct, 1e-13);

        // Symmetric 3D
        #[rustfmt::skip]
        let data = &[
            [
                [ 6.0, 10.0, 12.0],
                [24.0, 40.0, 48.0],
                [36.0, 60.0, 72.0],
            ],
            [
                [24.0, 40.0, 48.0],
                [12.0, 20.0, 24.0],
                [30.0, 50.0, 60.0],
            ],
            [
                [36.0, 60.0, 72.0],
                [30.0, 50.0, 60.0],
                [18.0, 30.0, 36.0],
            ],
        ];
        let dd = Tensor3::from_std_array(data, Rep::Symmetric, true).unwrap();
        let m1 = dd.as_std_matrix();
        #[rustfmt::skip]
        let correct = &[
            [ 6.0, 10.0, 12.0],
            [12.0, 20.0, 24.0],
            [18.0, 30.0, 36.0],
            [24.0, 40.0, 48.0],
            [30.0, 50.0, 60.0],
            [36.0, 60.0, 72.0],
            [24.0, 40.0, 48.0],
            [30.0, 50.0, 60.0],
            [36.0, 60.0, 72.0],
        ];
        mat_approx_eq(&m1, correct, 1e-13);
        let ee = Tensor3::from_std_matrix(correct, Rep::Symmetric, true).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, correct, 1e-13);

        // Symmetric 2D
        #[rustfmt::skip]
        let data = &[
            [
                [ 6.0,  8.0, 0.0],
                [24.0, 32.0, 0.0],
                [ 0.0,  0.0, 0.0],
            ],
            [
                [24.0, 32.0, 0.0],
                [12.0, 16.0, 0.0],
                [ 0.0,  0.0, 0.0],
            ],
            [
                [ 0.0,  0.0, 0.0],
                [ 0.0,  0.0, 0.0],
                [18.0, 24.0, 0.0],
            ],
        ];
        let dd = Tensor3::from_std_array(data, Rep::Symmetric2D, true).unwrap();
        let m1 = dd.as_std_matrix();
        #[rustfmt::skip]
        let correct = &[
            [ 6.0,  8.0, 0.0],
            [12.0, 16.0, 0.0],
            [18.0, 24.0, 0.0],
            [24.0, 32.0, 0.0],
            [ 0.0,  0.0, 0.0],
            [ 0.0,  0.0, 0.0],
            [24.0, 32.0, 0.0],
            [ 0.0,  0.0, 0.0],
            [ 0.0,  0.0, 0.0],
        ];
        mat_approx_eq(&m1, correct, 1e-13);
        let ee = Tensor3::from_std_matrix(correct, Rep::Symmetric2D, true).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, correct, 1e-13);
    }

    fn generate_dd() -> Tensor3 {
        let mut dd = Tensor3::new(Rep::Symmetric, true);
        for m in 0..6 {
            for n in 0..3 {
                let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
                let value = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
                dd.sym_set_std(i, j, k, value);
            }
        }
        dd
    }

    #[test]
    #[should_panic(expected = "self.rep != Rep::General")]
    fn sym_set_std_panics_on_non_sym() {
        let mut dd = Tensor3::new(Rep::General, true);
        dd.sym_set_std(0, 0, 0, 0.0);
    }

    #[test]
    #[should_panic(expected = "the len is 3 but the index is 3")]
    fn sym_set_std_panics_on_incorrect_indices() {
        let mut dd = Tensor3::new(Rep::Symmetric2D, true);
        dd.sym_set_std(0, 0, 3, 5.0);
    }

    #[test]
    fn sym_set_std_works() {
        let dd = generate_dd();
        assert_eq!(
            format!("{:.0}", dd.as_std_matrix()),
            "┌             ┐\n\
             │ 111 112 113 │\n\
             │ 221 222 223 │\n\
             │ 331 332 333 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             └             ┘"
        );
    }

    #[test]
    #[should_panic]
    fn set_tensor_panics_on_incorrect_input() {
        let dd = Tensor3::new(Rep::Symmetric, true);
        let mut ee = Tensor3::new(Rep::General, true);
        ee.set_tensor(2.0, &dd);
    }

    #[test]
    fn set_tensor_works() {
        #[rustfmt::skip]
        let dd = Tensor3::from_std_matrix(&[
                [1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0],
                [9.0, 9.0, 9.0],
                [2.0, 2.0, 2.0],
                [6.0, 6.0, 6.0],
                [3.0, 3.0, 3.0],
                [2.0, 2.0, 2.0],
                [6.0, 6.0, 6.0],
                [3.0, 3.0, 3.0],
        ], Rep::General,true).unwrap();
        let mut ee = Tensor3::new(Rep::General, true);
        ee.set_tensor(2.0, &dd);
        #[rustfmt::skip]
        let correct = Matrix::from(&[
            [ 2.0,  2.0,  2.0],
            [10.0, 10.0, 10.0],
            [18.0, 18.0, 18.0],
            [ 4.0,  4.0,  4.0],
            [12.0, 12.0, 12.0],
            [ 6.0,  6.0,  6.0],
            [ 4.0,  4.0,  4.0],
            [12.0, 12.0, 12.0],
            [ 6.0,  6.0,  6.0],
        ]);
        mat_approx_eq(&ee.as_std_matrix(), &correct, 1e-14);
    }

    /// Generates a non-symmetric standard 3x3x3 tensor with distinct components
    fn generate_std_general() -> [[[f64; 3]; 3]; 3] {
        let mut inp = [[[0.0; 3]; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    inp[i][j][k] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
                }
            }
        }
        inp
    }

    /// Generates a standard 3x3x3 tensor that is minor-symmetric in (j,k)
    fn generate_std_sym_case_b() -> [[[f64; 3]; 3]; 3] {
        let mut inp = [[[0.0; 3]; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let (a, b) = if j <= k { (j, k) } else { (k, j) };
                    inp[i][j][k] = (100 * (i + 1) + 10 * (a + 1) + (b + 1)) as f64;
                }
            }
        }
        inp
    }

    /// Generates a standard 3x3x3 tensor that is minor-symmetric in (j,k) and 2D (zero out-of-plane shears)
    fn generate_std_sym_case_b_2d() -> [[[f64; 3]; 3]; 3] {
        let mut inp = [[[0.0; 3]; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let (a, b) = if j <= k { (j, k) } else { (k, j) };
                    // zero out-of-plane shears (0,2) and (1,2)
                    if a == b || (a, b) == (0, 1) {
                        inp[i][j][k] = (100 * (i + 1) + 10 * (a + 1) + (b + 1)) as f64;
                    }
                }
            }
        }
        inp
    }

    #[test]
    fn new_case_b_works() {
        let cc = Tensor3::new(Rep::General, false);
        assert_eq!(cc.dims(), (3, 9));
        let dd = Tensor3::new(Rep::Symmetric, false);
        assert_eq!(dd.dims(), (3, 6));
        let ee = Tensor3::new(Rep::Symmetric2D, false);
        assert_eq!(ee.dims(), (3, 4));
    }

    #[test]
    fn from_std_array_case_b_fails_captures_errors() {
        let res = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1, Rep::Symmetric, false);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );

        let res = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1, Rep::Symmetric2D, false);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn from_std_array_case_b_works() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1, Rep::General, false).unwrap();
        for m in 0..3 {
            for n in 0..9 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_B_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1, Rep::Symmetric, false).unwrap();
        for m in 0..3 {
            for n in 0..6 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_B_SYM_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1, Rep::Symmetric2D, false).unwrap();
        for m in 0..3 {
            for n in 0..4 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_B_SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }
    }

    #[test]
    fn from_std_matrix_case_b_works() {
        // general
        let dd = Tensor3::from_std_matrix(&SamplesTensor3::CASE_B_SAMPLE1_STD_MATRIX, Rep::General, false).unwrap();
        let (nrow, ncol) = dd.dims();
        for m in 0..nrow {
            for n in 0..ncol {
                approx_eq(dd.get(m, n), SamplesTensor3::CASE_B_SAMPLE1_KELVIN_MATRIX[m][n], 1e-15);
            }
        }

        // symmetric 3D
        let dd =
            Tensor3::from_std_matrix(&SamplesTensor3::CASE_B_SYM_SAMPLE1_STD_MATRIX, Rep::Symmetric, false).unwrap();
        let (nrow, ncol) = dd.dims();
        for m in 0..nrow {
            for n in 0..ncol {
                approx_eq(
                    dd.get(m, n),
                    SamplesTensor3::CASE_B_SYM_SAMPLE1_KELVIN_MATRIX[m][n],
                    1e-14,
                );
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_matrix(
            &SamplesTensor3::CASE_B_SYM_2D_SAMPLE1_STD_MATRIX,
            Rep::Symmetric2D,
            false,
        )
        .unwrap();
        let (nrow, ncol) = dd.dims();
        for m in 0..nrow {
            for n in 0..ncol {
                approx_eq(
                    dd.get(m, n),
                    SamplesTensor3::CASE_B_SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n],
                    1e-14,
                );
            }
        }
    }

    #[test]
    fn get_std_case_b_works() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1, Rep::General, false).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::CASE_B_SAMPLE1[i][j][k], 1e-13);
                }
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1, Rep::Symmetric, false).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::CASE_B_SYM_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1, Rep::Symmetric2D, false).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(
                        dd.get_std(i, j, k),
                        SamplesTensor3::CASE_B_SYM_2D_SAMPLE1[i][j][k],
                        1e-14,
                    );
                }
            }
        }
    }

    #[test]
    fn update_case_b_works() {
        let mut dd = Tensor3::new(Rep::Symmetric2D, false);
        let ee = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1, Rep::Symmetric2D, false).unwrap();
        dd.update(2.0, &ee);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(
                        dd.get_std(i, j, k),
                        2.0 * SamplesTensor3::CASE_B_SYM_2D_SAMPLE1[i][j][k],
                        1e-14,
                    );
                }
            }
        }
    }

    #[test]
    fn as_std_array_and_to_std_array_case_b_work() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1, Rep::General, false).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_B_SAMPLE1[i][j][k], 1e-13);
                }
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1, Rep::Symmetric, false).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_B_SYM_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1, Rep::Symmetric2D, false).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_B_SYM_2D_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }
    }

    #[test]
    fn as_std_matrix_and_to_std_matrix_case_b_work() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1, Rep::General, false).unwrap();
        let mat = dd.as_std_matrix();
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(mat.get(m, n), SamplesTensor3::CASE_B_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1, Rep::Symmetric, false).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (3, 9));
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(
                    mat.get(m, n),
                    SamplesTensor3::CASE_B_SYM_SAMPLE1_STD_MATRIX[m][n],
                    1e-13,
                );
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1, Rep::Symmetric2D, false).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (3, 9));
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(
                    mat.get(m, n),
                    SamplesTensor3::CASE_B_SYM_2D_SAMPLE1_STD_MATRIX[m][n],
                    1e-13,
                );
            }
        }
    }

    #[test]
    fn sym_set_std_case_b_works() {
        let mut dd = Tensor3::new(Rep::Symmetric, false);
        let inp = generate_std_sym_case_b();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    if j <= k {
                        dd.sym_set_std(i, j, k, inp[i][j][k]);
                    }
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), inp[i][j][k], 1e-13);
                }
            }
        }
    }

    #[test]
    fn from_std_matrix_case_b_symmetric2d_fails() {
        let inp = generate_std_sym_case_b_2d();
        let dd = Tensor3::from_std_array(&inp, Rep::Symmetric2D, false).unwrap();
        let mut mat = dd.as_std_matrix();
        // corrupt the out-of-plane shear (i,j,k) = (0,0,2) -> (m,n) = (0,5)
        mat.set(0, 5, 5.0);
        let res = Tensor3::from_std_matrix(&mat, Rep::Symmetric2D, false);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn from_std_matrix_case_b_symmetric_fails() {
        let inp = generate_std_sym_case_b();
        let dd = Tensor3::from_std_array(&inp, Rep::Symmetric, false).unwrap();
        let mut mat = dd.as_std_matrix();
        // break minor-symmetry: component (0,0,1) differs from its mirror (0,1,0)
        mat.set(0, 3, mat.get(0, 3) + 1.0);
        let res = Tensor3::from_std_matrix(&mat, Rep::Symmetric, false);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );
    }

    #[test]
    fn set_tensor_and_update_case_b_work() {
        let inp = generate_std_general();
        let dd = Tensor3::from_std_array(&inp, Rep::General, false).unwrap();

        // set_tensor
        let mut ee = Tensor3::new(Rep::General, false);
        ee.set_tensor(2.0, &dd);
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(ee.get(m, n), 2.0 * dd.get(m, n), 1e-13);
            }
        }

        // update
        let mut ff = Tensor3::new(Rep::General, false);
        ff.update(1.0, &dd);
        ff.update(2.0, &dd);
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(ff.get(m, n), 3.0 * dd.get(m, n), 1e-13);
            }
        }
    }

    #[test]
    #[should_panic]
    fn update_case_mismatch_panics() {
        let mut dd = Tensor3::new(Rep::General, true);
        let ee = Tensor3::new(Rep::General, false);
        dd.update(1.0, &ee);
    }

    #[test]
    #[should_panic]
    fn set_tensor_case_mismatch_panics() {
        let dd = Tensor3::new(Rep::General, false);
        let mut ee = Tensor3::new(Rep::General, true);
        ee.set_tensor(1.0, &dd);
    }

    #[test]
    fn clone_and_serialize_work() {
        let dd = generate_dd();
        // clone
        let mut cloned = dd.clone();
        cloned.set(0, 0, 999.0);
        assert_eq!(
            format!("{:.0}", dd.as_std_matrix()),
            "┌             ┐\n\
             │ 111 112 113 │\n\
             │ 221 222 223 │\n\
             │ 331 332 333 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             └             ┘"
        );
        assert_eq!(
            format!("{:.0}", cloned.as_std_matrix()),
            "┌             ┐\n\
             │ 999 112 113 │\n\
             │ 221 222 223 │\n\
             │ 331 332 333 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             └             ┘"
        );
        // serialize
        let json = serde_json::to_string(&dd).unwrap();
        assert!(json.len() > 0);
        // deserialize
        let from_json: Tensor3 = serde_json::from_str(&json).unwrap();
        assert_eq!(
            format!("{:.0}", from_json.as_std_matrix()),
            "┌             ┐\n\
             │ 111 112 113 │\n\
             │ 221 222 223 │\n\
             │ 331 332 333 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             └             ┘"
        );
    }

    #[test]
    fn debug_works() {
        let dd = Tensor3::new(Rep::General, true);
        assert!(format!("{:?}", dd).len() > 0);
    }

    #[test]
    fn constant_permutation_works() {
        let perm_a = Tensor3::constant_permutation(true);
        let expected = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, SQRT_2],
            [SQRT_2, 0.0, 0.0],
            [0.0, -SQRT_2, 0.0],
        ];
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(perm_a.get(m, n), expected[m][n], 1e-15);
            }
        }
        assert_eq!(
            format!("{:.3}", perm_a),
            "┌                      ┐\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  1.414 │\n\
             │  1.414  0.000  0.000 │\n\
             │  0.000 -1.414  0.000 │\n\
             └                      ┘"
        );

        let perm_b = Tensor3::constant_permutation(false);
        let expected = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SQRT_2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -SQRT_2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SQRT_2, 0.0, 0.0],
        ];
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(perm_b.get(m, n), expected[m][n], 1e-15);
            }
        }
    }
}
