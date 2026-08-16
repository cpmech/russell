use super::{IJK_TO_MN, IJK_TO_MN_SYM, MN_TO_IJK, SQRT_2};
use crate::{AsMatrix9x3, Rep, StrError};
use russell_lab::Matrix;
use serde::{Deserialize, Serialize};

/// Implements a third-order tensor, minor-symmetric or not
///
/// # Standard and Kelvin components
///
/// The methods of this struct follow a naming convention that distinguishes
/// between the **standard** (Cartesian) components `Dᵢⱼₖₗ` and the **Kelvin**
/// components stored internally:
///
/// * Methods dealing with **standard components** carry the `std` qualifier in
///   their names (e.g., [`Tensor3::from_std_matrix`], [`Tensor3::get_std`],
///   [`Tensor3::as_std_matrix`], [`Tensor3::sym_set_std`]).
/// * Methods dealing directly with the **Kelvin components** carry no qualifier
///   (e.g., [`Tensor3::matrix`], [`Tensor3::get_mn`], [`Tensor3::set`],
///   [`Tensor3::update`]).
///
/// Internally, the components are converted to the Kelvin basis as follows.
///
/// First, the following mapping to the Kelvin space is considered:
///
/// ```text
/// i=j:  Mijk := Dijk
/// i<j:  Mijk := (Dijk + Djik) / √2
/// i>j:  Mijk := (Djik − Dijk) / √2
/// ```
///
/// [Rep::General]
///
/// Then, the 27 Mijk components of a Tensor3 are organized as follows:
///
/// ```text
///      0 0   0 1   0 2
///    -----------------
/// 0 │ M000  M001  M002
/// 1 │ M110  M111  M112
/// 2 │ M220  M221  M222
///   │
/// 3 │ M010  M011  M012
/// 4 │ M120  M121  M122
/// 5 │ M020  M021  M022
///   │
/// 6 │ M100  M101  M102
/// 7 │ M210  M211  M212
/// 8 │ M200  M201  M202
///    -----------------
///      8 0   8 1   8 2
/// ```
///
/// Note that the order of row indices (pairs (i,j) in (i,j,k)) follow
/// the same order as the one for Tensor2.
///
/// [Rep::Symmetric]
///
/// If the tensor has Dijk = Djik, the mapping simplifies to:
///
/// ```text
/// i=j:  Mijk := Dijk
/// i<j:  Mijk := Dijk * √2
/// i>j:  Mijk := 0
/// ```
///
/// Then, we only need to store 18 components as follows:
///
/// ```text
///      0 0      0 1      0 2    
///    --------------------------
/// 0 │ D000     D001     D002
/// 1 │ D110     D111     D112
/// 2 │ D220     D221     D222
///   │
/// 3 │ D010*√2  D011*√2  D012*√2
/// 4 │ D120*√2  D121*√2  D122*√2
/// 5 │ D020*√2  D021*√2  D022*√2
///    --------------------------
///      5 0      5 1      5 2
/// ```
///
/// [Rep::Symmetric2D]
///
/// In 2D, some components are zero, thus we may store only 12 components:
///
/// ```text
///      0 0      0 1      0 2    
///    --------------------------
/// 0 │ D000     D001     D002   
/// 1 │ D110     D111     D112   
/// 2 │ D220     D221     D222   
///   │
/// 3 │ D010*√2  D011*√2  D012*√2
///    --------------------------
///      3 0      3 1      3 2
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tensor3 {
    /// Holds the actual dimension of the Kelvin matrix
    ///
    /// * General: `dim = 9`
    /// * Symmetric: `dim = 6`
    /// * Symmetric2D: `dim = 4`
    pub(crate) dim: usize,

    /// Holds the components in Kelvin basis as matrix.
    ///
    /// This array may use more data than necessary in symmetric cases
    pub(crate) mat: [[f64; 3]; 9],

    /// Holds the Rep (representation) enum
    pub(crate) rep: Rep,

    /// BENCHMARKING. TODO: REMOVE THIS
    pub use_loops: bool,
}

impl Tensor3 {
    /// Creates a new (zeroed) Tensor3
    ///
    /// # Input
    ///
    /// * `rep` -- the [Rep] representation
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Rep, StrError, Tensor3};
    ///
    /// fn main() {
    ///     let cc = Tensor3::new(Rep::General);
    ///     assert_eq!(cc.dim(), 9);
    ///
    ///     let dd = Tensor3::new(Rep::Symmetric);
    ///     assert_eq!(dd.dim(), 6);
    ///
    ///     let ee = Tensor3::new(Rep::Symmetric2D);
    ///     assert_eq!(ee.dim(), 4);
    /// }
    /// ```
    pub fn new(rep: Rep) -> Self {
        Tensor3 {
            dim: rep.dim(),
            mat: [[0.0; 3]; 9],
            rep,
            use_loops: false,
        }
    }

    /// Allocates a minor-symmetric Tensor3
    pub fn new_sym(two_dim: bool) -> Self {
        if two_dim {
            Tensor3::new(Rep::Symmetric2D)
        } else {
            Tensor3::new(Rep::Symmetric)
        }
    }

    /// Allocates a minor-symmetric Tensor3 given the space dimension
    ///
    /// **Note:** `space_ndim` must be 2 or 3 (only 2 is checked, otherwise 3 is assumed)
    pub fn new_sym_ndim(space_ndim: usize) -> Self {
        if space_ndim == 2 {
            Tensor3::new(Rep::Symmetric2D)
        } else {
            Tensor3::new(Rep::Symmetric)
        }
    }

    /// Returns the representation associated with this Tensor3
    pub fn rep(&self) -> Rep {
        self.rep
    }

    /// Returns the Kelvin matrix dimension (4, 6, or 9)
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns an access to the underlying Kelvin matrix
    ///
    /// # Notes
    ///
    /// The returned slice holds the `dim` active rows of the tensor. Each row
    /// is a fixed `[f64; 9]` array, so use [`Tensor3::dim`] as the column bound
    /// when iterating over the components.
    pub fn matrix(&self) -> &[[f64; 3]] {
        &self.mat[0..self.dim]
    }

    /// Returns a mutable access to the underlying Kelvin matrix
    ///
    /// # Notes
    ///
    /// The returned slice holds the `dim` active rows of the tensor. Each row
    /// is a fixed `[f64; 9]` array, so use [`Tensor3::dim`] as the column bound
    /// when iterating over the components.
    pub fn matrix_mut(&mut self) -> &mut [[f64; 3]] {
        &mut self.mat[0..self.dim]
    }

    /// Returns the (m,n) component of the Kelvin matrix
    ///
    /// # Input
    ///
    /// * `m` -- the row index (must be `< 9`)
    /// * `n` -- the column index (must be `< 3`)
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
    /// let mut dd = Tensor3::new(Rep::General);
    /// dd.set(0, 0, 123.0);
    /// assert_eq!(dd.get_mn(0, 0), 123.0);
    /// ```
    pub fn get_mn(&self, m: usize, n: usize) -> f64 {
        self.mat[m][n]
    }

    /// Sets the (m,n) component of the Kelvin matrix
    ///
    /// # Input
    ///
    /// * `m` -- the row index (must be `< 9`)
    /// * `n` -- the column index (must be `< 3`)
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
    /// let mut dd = Tensor3::new(Rep::General);
    /// dd.set(0, 0, 123.0);
    /// assert_eq!(dd.matrix()[0][0], 123.0);
    /// ```
    pub fn set(&mut self, m: usize, n: usize, value: f64) {
        self.mat[m][n] = value;
    }

    /// Creates a new Tensor3 constructed from a nested array containing the standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard Dijkl components given with
    ///   respect to an orthonormal Cartesian basis
    /// * `rep` -- the [Rep] representation
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
    ///                     inp[i][j][k] = 100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             }
    ///         }
    ///     }
    ///     let dd = Tensor3::from_std_array(&inp, Rep::General)?;
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
    pub fn from_std_array(inp: &[[[f64; 3]; 3]; 3], rep: Rep) -> Result<Self, StrError> {
        let dim = rep.dim();
        let mut mat = [[0.0; 3]; 9];
        if dim == 4 || dim == 6 {
            let max = if dim == 4 { 3 } else { 6 };
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        // check minor-symmetry
                        if i > j {
                            if inp[i][j][k] != inp[j][i][k] {
                                return Err("the input data does not correspond to a minor-symmetric tensor");
                            }
                        } else {
                            let (m, n) = IJK_TO_MN[i][j][k];
                            if m > max {
                                if inp[i][j][k] != 0.0 {
                                    return Err("the input data does not correspond to a 2D minor-symmetric tensor");
                                }
                                continue;
                            } else if m < 3 {
                                mat[m][n] = inp[i][j][k];
                            } else {
                                mat[m][n] = SQRT_2 * inp[i][j][k];
                            }
                        }
                    }
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let (m, n) = IJK_TO_MN[i][j][k];
                        // ** i == j **
                        if i == j {
                            mat[m][n] = inp[i][j][k];
                        // ** i < j **
                        } else if i < j {
                            mat[m][n] = (inp[i][j][k] + inp[j][i][k]) / SQRT_2;
                        // ** i > j **
                        } else if i > j {
                            mat[m][n] = (inp[j][i][k] - inp[i][j][k]) / SQRT_2;
                        }
                    }
                }
            }
        }
        Ok(Tensor3 {
            dim,
            mat,
            rep,
            use_loops: false,
        })
    }

    /// Creates a new Tensor3 constructed from a 9x3 matrix with standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard matrix of components given with
    ///   respect to an orthonormal Cartesian basis. The matrix must be (9,3),
    ///   even if it corresponds to a minor-symmetric tensor.
    /// * `rep` -- the [Rep] representation
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix is not 9x3.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Rep, MN_TO_IJK, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General)?;
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
    pub fn from_std_matrix(inp: &dyn AsMatrix9x3, rep: Rep) -> Result<Self, StrError> {
        let dim = rep.dim();
        let mut mat = [[0.0; 3]; 9];
        if dim == 4 || dim == 6 {
            let max = if dim == 4 { 3 } else { 6 };
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let (m, n) = IJK_TO_MN[i][j][k];
                        let (r, s) = IJK_TO_MN[j][i][k];
                        // check minor-symmetry
                        if i > j {
                            if inp.at(m, n) != inp.at(r, s) {
                                return Err("the input data does not correspond to a minor-symmetric tensor");
                            }
                        } else {
                            if m > max {
                                if inp.at(m, n) != 0.0 {
                                    return Err("the input data does not correspond to a 2D minor-symmetric tensor");
                                }
                                continue;
                            } else if m < 3 {
                                mat[m][n] = inp.at(m, n);
                            } else {
                                mat[m][n] = SQRT_2 * inp.at(m, n);
                            }
                        }
                    }
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let (m, n) = IJK_TO_MN[i][j][k];
                        // ** i == j **
                        if i == j {
                            mat[m][n] = inp.at(m, n);
                        // ** i < j **
                        } else if i < j {
                            let (r, s) = IJK_TO_MN[j][i][k];
                            mat[m][n] = (inp.at(m, n) + inp.at(r, s)) / SQRT_2;
                        // ** i > j **
                        } else if i > j {
                            let (r, s) = IJK_TO_MN[j][i][k];
                            mat[m][n] = (inp.at(r, s) - inp.at(m, n)) / SQRT_2;
                        }
                    }
                }
            }
        }
        Ok(Tensor3 {
            dim,
            mat,
            rep,
            use_loops: false,
        })
    }

    /// Returns the (i,j,k) standard component
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Rep, MN_TO_IJK, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General)?;
    ///
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK[m][n];
    ///             let val = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             approx_eq(dd.get_std(i,j,k), val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn get_std(&self, i: usize, j: usize, k: usize) -> f64 {
        match self.dim {
            4 => {
                let (m, n) = IJK_TO_MN_SYM[i][j][k];
                if m > 3 {
                    0.0
                } else if m < 3 {
                    self.mat[m][n]
                } else {
                    self.mat[m][n] / SQRT_2
                }
            }
            6 => {
                let (m, n) = IJK_TO_MN_SYM[i][j][k];
                if m < 3 { self.mat[m][n] } else { self.mat[m][n] / SQRT_2 }
            }
            _ => {
                let (m, n) = IJK_TO_MN[i][j][k];
                let val = self.mat[m][n];
                // ** i == j **
                if i == j {
                    val
                // ** i < j **
                } else if i < j {
                    let (r, s) = IJK_TO_MN[j][i][k];
                    let down = self.mat[r][s];
                    (val + down) / SQRT_2
                // ** i > j **
                } else {
                    let (r, s) = IJK_TO_MN[j][i][k];
                    let up = self.mat[r][s];
                    (up - val) / SQRT_2
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
    /// use russell_tensor::{Rep, MN_TO_IJK, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..4 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK[m][n];
    ///             inp[m][n] = 1.0;
    ///         }
    ///     }
    ///
    ///     let mut dd = Tensor3::new(Rep::General);
    ///     let ee = Tensor3::from_std_matrix(&inp, Rep::General)?;
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
        for m in 0..self.dim {
            for n in 0..3 {
                self.mat[m][n] += alpha * other.mat[m][n];
            }
        }
    }

    /// Returns a 3x3x3 array with the standard components
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Rep, MN_TO_IJK, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General)?;
    ///     let arr = dd.as_std_array();
    ///
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK[m][n];
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
    /// use russell_tensor::{Rep, MN_TO_IJK, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General)?;
    ///     let mut arr = vec![vec![vec![0.0; 3]; 3]; 3];
    ///     dd.to_std_array(&mut arr);
    ///
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK[m][n];
    ///             let val = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             approx_eq(arr[i][j][k], val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn to_std_array(&self, dd: &mut Vec<Vec<Vec<f64>>>) {
        let dim = self.dim;
        if dim < 9 {
            for m in 0..dim {
                for n in 0..3 {
                    let (i, j, k) = MN_TO_IJK[m][n];
                    dd[i][j][k] = self.get_std(i, j, k);
                    if i != j {
                        dd[j][i][k] = dd[i][j][k];
                    }
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        dd[i][j][k] = self.get_std(i, j, k);
                    }
                }
            }
        }
    }

    /// Returns a 9x3 matrix with the standard components
    ///
    /// **Note:** The matrix will have the standard components and 9x3 dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Rep, MN_TO_IJK, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 9]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General)?;
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
        let mut mat = Matrix::new(9, 3);
        self.to_std_matrix(&mut mat);
        mat
    }

    /// Converts this tensor to a 9x3 matrix with the standard components
    ///
    /// # Input
    ///
    /// * `mat` -- the resulting 9x3 matrix
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix is not 9x3
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Matrix;
    /// use russell_tensor::{Rep, MN_TO_IJK, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor3::from_std_matrix(&inp, Rep::General)?;
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
        assert_eq!(mat.dims(), (9, 3));
        for m in 0..9 {
            for n in 0..3 {
                let (i, j, k) = MN_TO_IJK[m][n];
                mat.set(m, n, self.get_std(i, j, k));
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
    /// use russell_tensor::{Rep, MN_TO_IJK, Tensor3};
    ///
    /// fn main() {
    ///     let mut dd = Tensor3::new(Rep::Symmetric2D);
    ///     for m in 0..4 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJKL[m][n];
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
        let (m, n) = IJK_TO_MN_SYM[i][j][k];
        if m < 3 {
            self.mat[m][n] = value;
        } else {
            self.mat[m][n] = value * SQRT_2;
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
    ///     let dd = Tensor3::from_std_matrix(data, Rep::General)?;
    ///     let mut ee = Tensor3::new(Rep::General);
    ///
    ///     ee.set_tensor(1.0, &dd);
    ///
    ///     mat_approx_eq(&dd.as_std_matrix(), data, 1e-14);
    ///     Ok(())
    /// }
    /// ```
    pub fn set_tensor(&mut self, alpha: f64, other: &Tensor3) {
        assert_eq!(other.rep, self.rep);
        let dim = self.dim;
        for i in 0..dim {
            for j in 0..3 {
                self.mat[i][j] = alpha * other.mat[i][j];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{MN_TO_IJK, Tensor3};
    use crate::{Rep, SamplesTensor3};
    use russell_lab::{Matrix, approx_eq, mat_approx_eq};

    #[test]
    fn new_and_getters_work() {
        // general
        let mut dd = Tensor3::new(Rep::General);
        assert_eq!(dd.dim(), 9);
        assert_eq!(dd.rep(), Rep::General);
        dd.matrix_mut()[0][0] = 1.0;

        // symmetric
        let dd = Tensor3::new(Rep::Symmetric);
        assert_eq!(dd.rep(), Rep::Symmetric);
        assert_eq!(dd.dim(), 6);

        let dd = Tensor3::new_sym(false);
        assert_eq!(dd.rep(), Rep::Symmetric);
        assert_eq!(dd.dim(), 6);

        let dd = Tensor3::new_sym_ndim(3);
        assert_eq!(dd.rep(), Rep::Symmetric);
        assert_eq!(dd.dim(), 6);

        // symmetric 2d
        let dd = Tensor3::new(Rep::Symmetric2D);
        assert_eq!(dd.rep(), Rep::Symmetric2D);
        assert_eq!(dd.dim(), 4);

        let dd = Tensor3::new_sym(true);
        assert_eq!(dd.rep(), Rep::Symmetric2D);
        assert_eq!(dd.dim(), 4);

        let dd = Tensor3::new_sym_ndim(2);
        assert_eq!(dd.rep(), Rep::Symmetric2D);
        assert_eq!(dd.dim(), 4);
    }

    #[test]
    fn from_std_array_fails_captures_errors() {
        let res = Tensor3::from_std_array(&SamplesTensor3::SAMPLE1, Rep::Symmetric);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );

        let res = Tensor3::from_std_array(&SamplesTensor3::SYM_SAMPLE1, Rep::Symmetric2D);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn from_std_array_works() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::SAMPLE1, Rep::General).unwrap();
        for m in 0..9 {
            for n in 0..3 {
                assert_eq!(dd.matrix()[m][n], SamplesTensor3::SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 3d
        let dd = Tensor3::from_std_array(&SamplesTensor3::SYM_SAMPLE1, Rep::Symmetric).unwrap();
        for m in 0..6 {
            for n in 0..3 {
                assert_eq!(dd.matrix()[m][n], SamplesTensor3::SYM_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 2d
        let dd = Tensor3::from_std_array(&SamplesTensor3::SYM_2D_SAMPLE1, Rep::Symmetric2D).unwrap();
        for m in 0..4 {
            for n in 0..3 {
                assert_eq!(dd.matrix()[m][n], SamplesTensor3::SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }
    }

    #[test]
    fn from_std_matrix_fails_captures_errors() {
        let mut inp = [[0.0; 3]; 9];
        inp[3][0] = 1e-15;
        let res = Tensor3::from_std_matrix(&inp, Rep::Symmetric);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );

        inp[3][0] = 0.0;
        inp[4][0] = 1.0;
        inp[7][0] = 1.0;
        let res = Tensor3::from_std_matrix(&inp, Rep::Symmetric2D);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn from_std_matrix_works() {
        // general
        let dd = Tensor3::from_std_matrix(&SamplesTensor3::SAMPLE1_STD_MATRIX, Rep::General).unwrap();
        for m in 0..dd.dim() {
            for n in 0..3 {
                approx_eq(dd.matrix()[m][n], SamplesTensor3::SAMPLE1_KELVIN_MATRIX[m][n], 1e-15);
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_matrix(&SamplesTensor3::SYM_SAMPLE1_STD_MATRIX, Rep::Symmetric).unwrap();
        for m in 0..dd.dim() {
            for n in 0..3 {
                approx_eq(
                    dd.matrix()[m][n],
                    SamplesTensor3::SYM_SAMPLE1_KELVIN_MATRIX[m][n],
                    1e-14,
                );
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_matrix(&SamplesTensor3::SYM_2D_SAMPLE1_STD_MATRIX, Rep::Symmetric2D).unwrap();
        for m in 0..dd.dim() {
            for n in 0..3 {
                approx_eq(
                    dd.matrix()[m][n],
                    SamplesTensor3::SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n],
                    1e-14,
                );
            }
        }
    }

    #[test]
    fn get_std_works() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::SAMPLE1, Rep::General).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::SAMPLE1[i][j][k], 1e-13);
                }
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_array(&SamplesTensor3::SYM_SAMPLE1, Rep::Symmetric).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::SYM_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_array(&SamplesTensor3::SYM_2D_SAMPLE1, Rep::Symmetric2D).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::SYM_2D_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn update_panics_on_incorrect_input() {
        let mut dd = Tensor3::new(Rep::Symmetric2D);
        let ee = Tensor3::new(Rep::Symmetric);
        dd.update(2.0, &ee);
    }

    #[test]
    fn update_works() {
        let mut dd = Tensor3::new(Rep::Symmetric2D);
        let ee = Tensor3::from_std_array(&SamplesTensor3::SYM_2D_SAMPLE1, Rep::Symmetric2D).unwrap();
        dd.update(2.0, &ee);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(
                        dd.get_std(i, j, k),
                        2.0 * SamplesTensor3::SYM_2D_SAMPLE1[i][j][k],
                        1e-14,
                    );
                }
            }
        }
    }

    #[test]
    fn as_std_array_and_to_std_array_work() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::SAMPLE1, Rep::General).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::SAMPLE1[i][j][k], 1e-13);
                }
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_array(&SamplesTensor3::SYM_SAMPLE1, Rep::Symmetric).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::SYM_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_array(&SamplesTensor3::SYM_2D_SAMPLE1, Rep::Symmetric2D).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::SYM_2D_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }
    }

    #[test]
    fn as_std_matrix_and_to_std_matrix_work() {
        // general
        let dd = Tensor3::from_std_array(&SamplesTensor3::SAMPLE1, Rep::General).unwrap();
        let mat = dd.as_std_matrix();
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(mat.get(m, n), SamplesTensor3::SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // symmetric 3D
        let dd = Tensor3::from_std_array(&SamplesTensor3::SYM_SAMPLE1, Rep::Symmetric).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (9, 3));
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(mat.get(m, n), SamplesTensor3::SYM_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // symmetric 2D
        let dd = Tensor3::from_std_array(&SamplesTensor3::SYM_2D_SAMPLE1, Rep::Symmetric2D).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (9, 3));
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(mat.get(m, n), SamplesTensor3::SYM_2D_SAMPLE1_STD_MATRIX[m][n], 1e-13);
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
        let dd = Tensor3::from_std_array(data, Rep::General).unwrap();
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
        let ee = Tensor3::from_std_matrix(correct, Rep::General).unwrap();
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
        let dd = Tensor3::from_std_array(data, Rep::Symmetric).unwrap();
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
        let ee = Tensor3::from_std_matrix(correct, Rep::Symmetric).unwrap();
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
        let dd = Tensor3::from_std_array(data, Rep::Symmetric2D).unwrap();
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
        let ee = Tensor3::from_std_matrix(correct, Rep::Symmetric2D).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, correct, 1e-13);
    }

    fn generate_dd() -> Tensor3 {
        let mut dd = Tensor3::new(Rep::Symmetric);
        for m in 0..6 {
            for n in 0..3 {
                let (i, j, k) = MN_TO_IJK[m][n];
                let value = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
                dd.sym_set_std(i, j, k, value);
            }
        }
        dd
    }

    #[test]
    #[should_panic(expected = "self.rep != Rep::General")]
    fn sym_set_std_panics_on_non_sym() {
        let mut dd = Tensor3::new(Rep::General);
        dd.sym_set_std(0, 0, 0, 0.0);
    }

    #[test]
    #[should_panic(expected = "the len is 3 but the index is 3")]
    fn sym_set_std_panics_on_incorrect_indices() {
        let mut dd = Tensor3::new(Rep::Symmetric2D);
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
        let dd = Tensor3::new(Rep::Symmetric);
        let mut ee = Tensor3::new(Rep::General);
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
        ], Rep::General).unwrap();
        let mut ee = Tensor3::new(Rep::General);
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

    #[test]
    fn clone_and_serialize_work() {
        let dd = generate_dd();
        // clone
        let mut cloned = dd.clone();
        cloned.matrix_mut()[0][0] = 999.0;
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
        let dd = Tensor3::new(Rep::General);
        assert!(format!("{:?}", dd).len() > 0);
    }
}
