use crate::{AsMatrix3x3, Mandel, StrError};
use crate::{IJ_TO_M, IJ_TO_M_SYM, M_TO_IJ, TOL_J2};
use crate::{SQRT_2, SQRT_2_BY_3, SQRT_3, SQRT_3_BY_2, SQRT_6};
use russell_lab::{Matrix, Vector};
use serde::{Deserialize, Serialize};

/// Implements a second-order tensor, symmetric or not
///
/// Internally, the components are converted to the Mandel basis. On the Mandel basis,
/// depending on the symmetry, we may store fewer components. Also, we may store
/// only 4 components of Symmetric 2D tensors.
///
/// **General case:**
///
/// ```text
///                       ┌                ┐
///                    00 │      T00       │ 0
///                    11 │      T11       │ 1
/// ┌             ┐    22 │      T22       │ 2
/// │ T00 T01 T02 │    01 │ (T01+T10) / √2 │ 3
/// │ T10 T11 T12 │ => 12 │ (T12+T21) / √2 │ 4
/// │ T20 T21 T22 │    02 │ (T02+T20) / √2 │ 5
/// └             ┘    10 │ (T01-T10) / √2 │ 6
///                    21 │ (T12-T21) / √2 │ 7
///                    20 │ (T02-T20) / √2 │ 8
///                       └                ┘
/// ```
///
/// **Symmetric 3D:**
///
/// ```text
///                       ┌          ┐
/// ┌             ┐    00 │   T00    │ 0
/// │ T00 T01 T02 │    11 │   T11    │ 1
/// │ T01 T11 T12 │ => 22 │   T22    │ 2
/// │ T02 T12 T22 │    01 │ T01 * √2 │ 3
/// └             ┘    12 │ T12 * √2 │ 4
///                    02 │ T02 * √2 │ 5
///                       └          ┘
/// ```
///
/// **Symmetric 2D:**
///
/// ```text
/// ┌             ┐       ┌          ┐
/// │ T00 T01     │    00 │   T00    │ 0
/// │ T01 T11     │ => 11 │   T11    │ 1
/// │         T22 │    22 │   T22    │ 2
/// └             ┘    01 │ T01 * √2 │ 3
///                       └          ┘
/// ```
///
/// # Notes
///
/// * The tensor is represented as a 9D, 6D or 4D vector and saved as `vec`
/// * You may perform operations on `vec` directly because it is isomorphic with the tensor itself
/// * For example, the norm of the tensor equals `vec.norm()`
/// * However, you must be careful when setting a single component of `vec` directly
///   because you may "break" the Mandel representation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tensor2 {
    /// Holds the components in Mandel basis as a vector.
    ///
    /// * General: `vec.dim = 9`
    /// * Symmetric in 3D: `vec.dim = 6`
    /// * Symmetric in 2D: `vec.dim = 4`
    pub vec: Vector,
}

impl Tensor2 {
    /// Creates a new (zeroed) Tensor2
    ///
    /// # Input
    ///
    /// * `case` -- the [Mandel] case
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, StrError, Tensor2};
    ///
    /// fn main() {
    ///     let a = Tensor2::new(Mandel::General);
    ///     assert_eq!(a.vec.as_data(), &[0.0,0.0,0.0,  0.0,0.0,0.0,  0.0,0.0,0.0]);
    ///
    ///     let b = Tensor2::new(Mandel::Symmetric);
    ///     assert_eq!(b.vec.as_data(), &[0.0,0.0,0.0,  0.0,0.0,0.0]);
    ///
    ///     let c = Tensor2::new(Mandel::Symmetric2D);
    ///     assert_eq!(c.vec.as_data(), &[0.0,0.0,0.0,  0.0]);
    /// }
    /// ```
    pub fn new(case: Mandel) -> Self {
        Tensor2 {
            vec: Vector::new(case.dim()),
        }
    }

    /// Allocates a symmetric Tensor2
    pub fn new_sym(two_dim: bool) -> Self {
        if two_dim {
            Tensor2::new(Mandel::Symmetric2D)
        } else {
            Tensor2::new(Mandel::Symmetric)
        }
    }

    /// Allocates a symmetric Tensor2 given the space dimension
    ///
    /// **Note:** `space_ndim` must be 2 or 3 (only 2 is checked, otherwise 3 is assumed)
    pub fn new_sym_ndim(space_ndim: usize) -> Self {
        if space_ndim == 2 {
            Tensor2::new(Mandel::Symmetric2D)
        } else {
            Tensor2::new(Mandel::Symmetric)
        }
    }

    /// Allocates a diagonal Tensor2 from octahedral invariants
    ///
    /// In matrix form, the diagonal component of the tensor are the principal stresses `(σ1, σ2, σ3)`:
    ///
    /// ```text
    /// ┌          ┐
    /// │ σ1  0  0 │
    /// │  0 σ2  0 │
    /// │  0  0 σ3 │
    /// └          ┘
    /// ```
    ///
    /// # Input
    ///
    /// * `sigma_m` -- mean pressure invariant `σm = ⅓ trace(σ)`
    /// * `sigma_d` -- deviatoric stress (von Mises) invariant `σd = ‖s‖ √3/√2 = √3 × J2`
    /// * `lode` -- Lode invariant `l = cos(3θ) = (3 √3 J3)/(2 pow(J2,1.5))`.
    ///   **Note:** The Lode invariant must be in `-1 ≤ lode ≤ 1`
    /// * `two_dim` -- 2D instead of 3D?
    pub fn new_from_oct_invariants(sigma_m: f64, sigma_d: f64, lode: f64, two_dim: bool) -> Result<Self, StrError> {
        if lode < -1.0 || lode > 1.0 {
            return Err("the following range must be satisfied: -1 ≤ lode ≤ 1");
        }
        let d = SQRT_3 * sigma_m;
        let r = SQRT_2_BY_3 * sigma_d;
        let theta = f64::acos(lode) / 3.0;
        let ss1 = r * f64::cos(theta);
        let ss2 = d;
        let ss3 = r * f64::sin(theta);
        let mut tt = Tensor2::new_sym(two_dim);
        tt.vec[0] = (SQRT_2 * ss1 + ss2) / SQRT_3;
        tt.vec[1] = -ss1 / SQRT_6 + ss2 / SQRT_3 - ss3 / SQRT_2;
        tt.vec[2] = -ss1 / SQRT_6 + ss2 / SQRT_3 + ss3 / SQRT_2;
        Ok(tt)
    }

    /// Returns the Mandel case associated with this Tensor2
    #[inline]
    pub fn case(&self) -> Mandel {
        Mandel::new(self.vec.dim())
    }

    /// Sets the Tensor2 with standard components given in matrix form
    ///
    /// # Input
    ///
    /// * `tt` -- the standard (not Mandel) Tij components given  with respect to an orthonormal Cartesian basis
    ///
    /// # Notes
    ///
    /// * In all cases, even in 2D, the input matrix must be 3×3
    /// * If symmetric, the off-diagonal components must equal each other
    /// * If 2D, `data[1][2]` and `data[0][2]` must be equal to zero
    ///
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, StrError, Tensor2, SQRT_2};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // general
    ///     let mut a = Tensor2::new(Mandel::General);
    ///     a.set_matrix(&[
    ///         [1.0, SQRT_2 * 2.0, SQRT_2 * 3.0],
    ///         [SQRT_2 * 4.0, 5.0, SQRT_2 * 6.0],
    ///         [SQRT_2 * 7.0, SQRT_2 * 8.0, 9.0],
    ///     ])?;
    ///     assert_eq!(
    ///         format!("{:.1}", a.vec),
    ///         "┌      ┐\n\
    ///          │  1.0 │\n\
    ///          │  5.0 │\n\
    ///          │  9.0 │\n\
    ///          │  6.0 │\n\
    ///          │ 14.0 │\n\
    ///          │ 10.0 │\n\
    ///          │ -2.0 │\n\
    ///          │ -2.0 │\n\
    ///          │ -4.0 │\n\
    ///          └      ┘"
    ///     );
    ///
    ///     // symmetric-3D
    ///     let mut b = Tensor2::new(Mandel::Symmetric);
    ///     b.set_matrix(&[
    ///             [1.0, 4.0 / SQRT_2, 6.0 / SQRT_2],
    ///             [4.0 / SQRT_2, 2.0, 5.0 / SQRT_2],
    ///             [6.0 / SQRT_2, 5.0 / SQRT_2, 3.0],
    ///     ])?;
    ///     assert_eq!(
    ///         format!("{:.1}", b.vec),
    ///         "┌     ┐\n\
    ///          │ 1.0 │\n\
    ///          │ 2.0 │\n\
    ///          │ 3.0 │\n\
    ///          │ 4.0 │\n\
    ///          │ 5.0 │\n\
    ///          │ 6.0 │\n\
    ///          └     ┘"
    ///     );
    ///
    ///     // symmetric-2D
    ///     let mut c = Tensor2::new(Mandel::Symmetric2D);
    ///     c.set_matrix(&[
    ///             [       1.0, 4.0/SQRT_2, 0.0],
    ///             [4.0/SQRT_2,        2.0, 0.0],
    ///             [       0.0,        0.0, 3.0],
    ///     ])?;
    ///     assert_eq!(
    ///         format!("{:.1}", c.vec),
    ///         "┌     ┐\n\
    ///          │ 1.0 │\n\
    ///          │ 2.0 │\n\
    ///          │ 3.0 │\n\
    ///          │ 4.0 │\n\
    ///          └     ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn set_matrix(&mut self, tt: &dyn AsMatrix3x3) -> Result<(), StrError> {
        let dim = self.vec.dim();
        if dim == 4 || dim == 6 {
            if tt.at(1, 0) != tt.at(0, 1) || tt.at(2, 1) != tt.at(1, 2) || tt.at(2, 0) != tt.at(0, 2) {
                return Err("cannot set symmetric Tensor2 with non-symmetric data");
            }
            if dim == 4 {
                if tt.at(1, 2) != 0.0 || tt.at(0, 2) != 0.0 {
                    return Err("cannot set Symmetric2D Tensor2 with non-zero off-diagonal data");
                }
            }
        }
        for m in 0..dim {
            let (i, j) = M_TO_IJ[m];
            if i == j {
                self.vec[m] = tt.at(i, j);
            }
            if i < j {
                self.vec[m] = (tt.at(i, j) + tt.at(j, i)) / SQRT_2;
            }
            if i > j {
                self.vec[m] = (tt.at(j, i) - tt.at(i, j)) / SQRT_2;
            }
        }
        Ok(())
    }

    /// Creates a new Tensor2 constructed from a matrix
    ///
    /// # Input
    ///
    /// * `tt` -- the standard (not Mandel) Tij components given  with respect to an orthonormal Cartesian basis
    /// * `case` -- the [Mandel] case
    ///
    /// # Notes
    ///
    /// * In all cases, even in 2D, the input matrix must be 3×3
    /// * If symmetric, the off-diagonal components must equal each other
    /// * If 2D, `data[1][2]` and `data[0][2]` must be equal to zero
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, StrError, Tensor2, SQRT_2};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // general
    ///     let a = Tensor2::from_matrix(
    ///         &[
    ///             [1.0, SQRT_2 * 2.0, SQRT_2 * 3.0],
    ///             [SQRT_2 * 4.0, 5.0, SQRT_2 * 6.0],
    ///             [SQRT_2 * 7.0, SQRT_2 * 8.0, 9.0],
    ///         ],
    ///         Mandel::General,
    ///     )?;
    ///     assert_eq!(
    ///         format!("{:.1}", a.vec),
    ///         "┌      ┐\n\
    ///          │  1.0 │\n\
    ///          │  5.0 │\n\
    ///          │  9.0 │\n\
    ///          │  6.0 │\n\
    ///          │ 14.0 │\n\
    ///          │ 10.0 │\n\
    ///          │ -2.0 │\n\
    ///          │ -2.0 │\n\
    ///          │ -4.0 │\n\
    ///          └      ┘"
    ///     );
    ///
    ///     // symmetric-3D
    ///     let b = Tensor2::from_matrix(
    ///         &[
    ///             [1.0, 4.0 / SQRT_2, 6.0 / SQRT_2],
    ///             [4.0 / SQRT_2, 2.0, 5.0 / SQRT_2],
    ///             [6.0 / SQRT_2, 5.0 / SQRT_2, 3.0],
    ///         ],
    ///         Mandel::Symmetric,
    ///     )?;
    ///     assert_eq!(
    ///         format!("{:.1}", b.vec),
    ///         "┌     ┐\n\
    ///          │ 1.0 │\n\
    ///          │ 2.0 │\n\
    ///          │ 3.0 │\n\
    ///          │ 4.0 │\n\
    ///          │ 5.0 │\n\
    ///          │ 6.0 │\n\
    ///          └     ┘"
    ///     );
    ///
    ///     // symmetric-2D
    ///     let c = Tensor2::from_matrix(
    ///         &[
    ///             [       1.0, 4.0/SQRT_2, 0.0],
    ///             [4.0/SQRT_2,        2.0, 0.0],
    ///             [       0.0,        0.0, 3.0],
    ///         ],
    ///         Mandel::Symmetric2D,
    ///     )?;
    ///     assert_eq!(
    ///         format!("{:.1}", c.vec),
    ///         "┌     ┐\n\
    ///          │ 1.0 │\n\
    ///          │ 2.0 │\n\
    ///          │ 3.0 │\n\
    ///          │ 4.0 │\n\
    ///          └     ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn from_matrix(tt: &dyn AsMatrix3x3, case: Mandel) -> Result<Self, StrError> {
        let mut res = Tensor2::new(case);
        res.set_matrix(tt)?;
        Ok(res)
    }

    /// Returns a new identity tensor
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, Tensor2};
    ///
    /// let ii = Tensor2::identity(Mandel::General);
    ///
    /// assert_eq!(
    ///     format!("{}", ii.vec),
    ///     "┌   ┐\n\
    ///      │ 1 │\n\
    ///      │ 1 │\n\
    ///      │ 1 │\n\
    ///      │ 0 │\n\
    ///      │ 0 │\n\
    ///      │ 0 │\n\
    ///      │ 0 │\n\
    ///      │ 0 │\n\
    ///      │ 0 │\n\
    ///      └   ┘"
    /// );
    /// ```
    pub fn identity(case: Mandel) -> Self {
        let mut res = Tensor2::new(case);
        res.vec[0] = 1.0;
        res.vec[1] = 1.0;
        res.vec[2] = 1.0;
        res
    }

    /// Returns the (i,j) component (standard; not Mandel)
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0,  2.0, 0.0],
    ///         [3.0, -1.0, 5.0],
    ///         [0.0,  4.0, 1.0],
    ///     ], Mandel::General)?;
    ///
    ///     approx_eq(a.get(1,2), 5.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn get(&self, i: usize, j: usize) -> f64 {
        match self.vec.dim() {
            4 => {
                let m = IJ_TO_M_SYM[i][j];
                if m > 3 {
                    0.0
                } else if i == j {
                    self.vec[m]
                } else {
                    self.vec[m] / SQRT_2
                }
            }
            6 => {
                let m = IJ_TO_M_SYM[i][j];
                if i == j {
                    self.vec[m]
                } else {
                    self.vec[m] / SQRT_2
                }
            }
            _ => {
                let m = IJ_TO_M[i][j];
                if i == j {
                    self.vec[m]
                } else if i < j {
                    let n = IJ_TO_M[j][i];
                    (self.vec[m] + self.vec[n]) / SQRT_2
                } else {
                    let n = IJ_TO_M[j][i];
                    (self.vec[n] - self.vec[m]) / SQRT_2
                }
            }
        }
    }

    /// Returns a matrix (standard components; not Mandel) representing this tensor
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0,  1.0, 0.0],
    ///         [1.0, -1.0, 0.0],
    ///         [0.0,  0.0, 1.0],
    ///     ], Mandel::Symmetric2D)?;
    ///     assert_eq!(
    ///         format!("{:.1}", a.to_matrix()),
    ///         "┌                ┐\n\
    ///          │  1.0  1.0  0.0 │\n\
    ///          │  1.0 -1.0  0.0 │\n\
    ///          │  0.0  0.0  1.0 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn to_matrix(&self) -> Matrix {
        let mut tt = Matrix::new(3, 3);
        let dim = self.vec.dim();
        if dim < 9 {
            for m in 0..dim {
                let (i, j) = M_TO_IJ[m];
                tt.set(i, j, self.get(i, j));
                if i != j {
                    tt.set(j, i, tt.get(i, j));
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    tt.set(i, j, self.get(i, j));
                }
            }
        }
        tt
    }

    /// Returns a 2x2 matrix (standard components; not Mandel) representing this tensor (2D)
    ///
    /// This function returns the third diagonal component T22 and the 2x2 matrix
    ///
    /// # Panics
    ///
    /// This function works only if the Tensor is Symmetric2D
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let tt = Tensor2::from_matrix(&[
    ///         [1.0, 2.0, 0.0],
    ///         [2.0, 3.0, 0.0],
    ///         [0.0, 0.0, 4.0],
    ///     ], Mandel::Symmetric2D)?;
    ///     let (t22, res) = tt.to_matrix_2d();
    ///     assert_eq!(t22, 4.0);
    ///     assert_eq!(
    ///         format!("{:.1}", res),
    ///         "┌         ┐\n\
    ///          │ 1.0 2.0 │\n\
    ///          │ 2.0 3.0 │\n\
    ///          └         ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn to_matrix_2d(&self) -> (f64, Matrix) {
        assert_eq!(self.case(), Mandel::Symmetric2D);
        let mut tt = Matrix::new(2, 2);
        tt.set(0, 0, self.get(0, 0));
        tt.set(0, 1, self.get(0, 1));
        tt.set(1, 0, self.get(1, 0));
        tt.set(1, 1, self.get(1, 1));
        (self.get(2, 2), tt)
    }

    /// Returns a General Tensor2 regardless of this tensor's Mandel type
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, Tensor2, StrError, SQRT_2};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let tt = Tensor2::from_matrix(&[
    ///         [1.0,        2.0/SQRT_2, 0.0],
    ///         [2.0/SQRT_2, 3.0,        0.0],
    ///         [0.0,        0.0,        4.0],
    ///     ], Mandel::Symmetric2D)?;
    ///     assert_eq!(
    ///         format!("{:.2}", tt.vec),
    ///         "┌      ┐\n\
    ///          │ 1.00 │\n\
    ///          │ 3.00 │\n\
    ///          │ 4.00 │\n\
    ///          │ 2.00 │\n\
    ///          └      ┘"
    ///     );
    ///
    ///     let tt_gen = tt.to_general();
    ///     assert_eq!(
    ///         format!("{:.2}", tt_gen.vec),
    ///         "┌      ┐\n\
    ///          │ 1.00 │\n\
    ///          │ 3.00 │\n\
    ///          │ 4.00 │\n\
    ///          │ 2.00 │\n\
    ///          │ 0.00 │\n\
    ///          │ 0.00 │\n\
    ///          │ 0.00 │\n\
    ///          │ 0.00 │\n\
    ///          │ 0.00 │\n\
    ///          └      ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn to_general(&self) -> Tensor2 {
        let mut res = Tensor2::new(Mandel::General);
        res.vec[0] = self.vec[0];
        res.vec[1] = self.vec[1];
        res.vec[2] = self.vec[2];
        res.vec[3] = self.vec[3];
        let original_dim = self.vec.dim();
        if original_dim > 4 {
            res.vec[4] = self.vec[4];
            res.vec[5] = self.vec[5];
        }
        if original_dim > 6 {
            res.vec[6] = self.vec[6];
            res.vec[7] = self.vec[7];
            res.vec[8] = self.vec[8];
        }
        res
    }

    /// Set all values to zero
    #[inline]
    pub fn clear(&mut self) {
        self.vec.fill(0.0);
    }

    /// Sets the (i,j) component of a symmetric Tensor2
    ///
    /// ```text
    /// σᵢⱼ = value
    /// ```
    ///
    /// **Note:** Only the diagonal and upper-diagonal components need to be set.
    ///
    /// # Panics
    ///
    /// The tensor must be symmetric and (i,j) must correspond to the possible
    /// combination due to the space dimension, otherwise a panic may occur.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() {
    ///     let mut a = Tensor2::new(Mandel::Symmetric2D);
    ///     a.sym_set(0, 0, 1.0);
    ///     a.sym_set(1, 1, 2.0);
    ///     a.sym_set(2, 2, 3.0);
    ///     a.sym_set(0, 1, 4.0);
    ///     assert_eq!(
    ///         format!("{:.1}", a.to_matrix()),
    ///         "┌             ┐\n\
    ///          │ 1.0 4.0 0.0 │\n\
    ///          │ 4.0 2.0 0.0 │\n\
    ///          │ 0.0 0.0 3.0 │\n\
    ///          └             ┘"
    ///     );
    ///
    ///     let mut b = Tensor2::new(Mandel::Symmetric);
    ///     b.sym_set(0, 0, 1.0);
    ///     b.sym_set(1, 1, 2.0);
    ///     b.sym_set(2, 2, 3.0);
    ///     b.sym_set(0, 1, 4.0);
    ///     b.sym_set(1, 0, 4.0);
    ///     b.sym_set(2, 0, 5.0);
    ///     assert_eq!(
    ///         format!("{:.1}", b.to_matrix()),
    ///         "┌             ┐\n\
    ///          │ 1.0 4.0 5.0 │\n\
    ///          │ 4.0 2.0 0.0 │\n\
    ///          │ 5.0 0.0 3.0 │\n\
    ///          └             ┘"
    ///     );
    /// }
    /// ```
    pub fn sym_set(&mut self, i: usize, j: usize, value: f64) {
        assert!(self.case() != Mandel::General);
        let m = IJ_TO_M_SYM[i][j];
        if i == j {
            self.vec[m] = value;
        } else {
            self.vec[m] = value * SQRT_2;
        }
    }

    /// Updates the (i,j) component of a symmetric Tensor2
    ///
    /// ```text
    /// σᵢⱼ += α·value
    /// ```
    ///
    /// **Note:** Only the diagonal and upper-diagonal components **must** be set.
    ///
    /// # Panics
    ///
    /// The tensor must be Symmetric or Symmetric2D and (i,j) must correspond to the possible
    /// combination due to the space dimension, otherwise a panic may occur.
    ///
    /// This function will panic also if i > j (lower-diagonal)
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut a = Tensor2::from_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [2.0, 5.0, 6.0],
    ///         [3.0, 6.0, 9.0],
    ///     ], Mandel::Symmetric)?;
    ///
    ///     a.sym_add(0, 1, 2.0, 10.0);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", a.to_matrix()),
    ///         "┌                ┐\n\
    ///          │  1.0 22.0  3.0 │\n\
    ///          │ 22.0  5.0  6.0 │\n\
    ///          │  3.0  6.0  9.0 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn sym_add(&mut self, i: usize, j: usize, alpha: f64, value: f64) {
        assert!(self.case() != Mandel::General);
        assert!(i <= j);
        let m = IJ_TO_M_SYM[i][j];
        if i == j {
            self.vec[m] += alpha * value;
        } else {
            self.vec[m] += alpha * value * SQRT_2;
        }
    }

    /// Sets this tensor equal to another one
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut a = Tensor2::from_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ], Mandel::General)?;
    ///     let b = Tensor2::from_matrix(&[
    ///         [10.0, 20.0, 30.0],
    ///         [40.0, 50.0, 60.0],
    ///         [70.0, 80.0, 90.0],
    ///     ], Mandel::General)?;
    ///
    ///     a.mirror(&b);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", a.to_matrix()),
    ///         "┌                ┐\n\
    ///          │ 10.0 20.0 30.0 │\n\
    ///          │ 40.0 50.0 60.0 │\n\
    ///          │ 70.0 80.0 90.0 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn mirror(&mut self, other: &Tensor2) -> Result<(), StrError> {
        let dim = self.vec.dim();
        if other.vec.dim() != dim {
            return Err("tensors are incompatible");
        }
        self.vec[0] = other.vec[0];
        self.vec[1] = other.vec[1];
        self.vec[2] = other.vec[2];
        self.vec[3] = other.vec[3];
        if dim > 4 {
            self.vec[4] = other.vec[4];
            self.vec[5] = other.vec[5];
        }
        if dim > 6 {
            self.vec[6] = other.vec[6];
            self.vec[7] = other.vec[7];
            self.vec[8] = other.vec[8];
        }
        Ok(())
    }

    /// Adds another tensor to this one
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut a = Tensor2::from_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ], Mandel::General)?;
    ///     let b = Tensor2::from_matrix(&[
    ///         [10.0, 20.0, 30.0],
    ///         [40.0, 50.0, 60.0],
    ///         [70.0, 80.0, 90.0],
    ///     ], Mandel::General)?;
    ///
    ///     a.add(2.0, &b);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", a.to_matrix()),
    ///         "┌                   ┐\n\
    ///          │  21.0  42.0  63.0 │\n\
    ///          │  84.0 105.0 126.0 │\n\
    ///          │ 147.0 168.0 189.0 │\n\
    ///          └                   ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn add(&mut self, alpha: f64, other: &Tensor2) -> Result<(), StrError> {
        let dim = self.vec.dim();
        if other.vec.dim() != dim {
            return Err("tensors are incompatible");
        }
        self.vec[0] += alpha * other.vec[0];
        self.vec[1] += alpha * other.vec[1];
        self.vec[2] += alpha * other.vec[2];
        self.vec[3] += alpha * other.vec[3];
        if dim > 4 {
            self.vec[4] += alpha * other.vec[4];
            self.vec[5] += alpha * other.vec[5];
        }
        if dim > 6 {
            self.vec[6] += alpha * other.vec[6];
            self.vec[7] += alpha * other.vec[7];
            self.vec[8] += alpha * other.vec[8];
        }
        Ok(())
    }

    /// Calculates the determinant
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ], Mandel::General)?;
    ///
    ///     approx_eq(a.determinant(), 0.0, 1e-13);
    ///     Ok(())
    /// }
    /// ```
    pub fn determinant(&self) -> f64 {
        let a = &self.vec;
        match self.vec.dim() {
            4 => a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0,
            6 => {
                a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0 - (a[0] * a[4] * a[4]) / 2.0
                    + (a[3] * a[4] * a[5]) / SQRT_2
                    - (a[1] * a[5] * a[5]) / 2.0
            }
            _ => {
                a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0 - (a[0] * a[4] * a[4]) / 2.0
                    + (a[3] * a[4] * a[5]) / SQRT_2
                    - (a[1] * a[5] * a[5]) / 2.0
                    + (a[2] * a[6] * a[6]) / 2.0
                    + (a[5] * a[6] * a[7]) / SQRT_2
                    + (a[0] * a[7] * a[7]) / 2.0
                    - (a[4] * a[6] * a[8]) / SQRT_2
                    - (a[3] * a[7] * a[8]) / SQRT_2
                    + (a[1] * a[8] * a[8]) / 2.0
            }
        }
    }

    /// Returns the transpose tensor
    ///
    /// ```text
    /// Aᵀ = transpose(A)
    ///
    /// [Aᵀ]ᵢⱼ = [A]ⱼᵢ
    /// ```
    ///
    /// ## Input/Output
    ///
    /// * `ai` -- a Tensor2 with matching dimensions to hold the transpose tensor
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::vec_approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.1, 1.2, 1.3],
    ///         [2.1, 2.2, 2.3],
    ///         [3.1, 3.2, 3.3],
    ///     ], Mandel::General)?;
    ///
    ///     let mut at = Tensor2::new(Mandel::General);
    ///     a.transpose(&mut at)?;
    ///
    ///     let at_correct = Tensor2::from_matrix(&[
    ///         [1.1, 2.1, 3.1],
    ///         [1.2, 2.2, 3.2],
    ///         [1.3, 2.3, 3.3],
    ///     ], Mandel::General)?;
    ///     vec_approx_eq(&at.vec.as_data(), &at_correct.vec.as_data(), 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn transpose(&self, at: &mut Tensor2) -> Result<(), StrError> {
        let dim = self.vec.dim();
        if at.vec.dim() != dim {
            return Err("tensors are incompatible");
        }
        at.vec[0] = self.vec[0];
        at.vec[1] = self.vec[1];
        at.vec[2] = self.vec[2];
        at.vec[3] = self.vec[3];
        if dim > 4 {
            at.vec[4] = self.vec[4];
            at.vec[5] = self.vec[5];
        }
        if dim > 6 {
            at.vec[6] = -self.vec[6];
            at.vec[7] = -self.vec[7];
            at.vec[8] = -self.vec[8];
        }
        Ok(())
    }

    /// Calculates the inverse tensor
    ///
    /// ```text
    /// A⁻¹ = inverse(A)
    ///
    /// A · A⁻¹ = I
    /// ```
    ///
    /// ## Input/Output
    ///
    /// * `ai` -- a Tensor2 with matching dimensions to hold the inverse tensor
    /// * `tolerance` -- a tolerance for the determinant such that the inverse is computed only if |det| > tolerance
    ///
    /// ## Output
    ///
    /// * If the determinant is zero, the inverse is not computed and returns `None`
    /// * Otherwise, the inverse is computed and returns the determinant
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::{approx_eq, mat_approx_eq, mat_mat_mul, Matrix};
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [6.0,  1.0,  2.0],
    ///         [3.0, 12.0,  4.0],
    ///         [5.0,  6.0, 15.0],
    ///     ], Mandel::General)?;
    ///
    ///     let mut ai = Tensor2::new(Mandel::General);
    ///
    ///     if let Some(det) = a.inverse(&mut ai, 1e-10)? {
    ///         assert_eq!(det, 827.0);
    ///     } else {
    ///         panic!("determinant is zero");
    ///     }
    ///
    ///     let a_mat = a.to_matrix();
    ///     let ai_mat = ai.to_matrix();
    ///     let mut a_times_ai = Matrix::new(3, 3);
    ///     mat_mat_mul(&mut a_times_ai, 1.0, &a_mat, &ai_mat)?;
    ///
    ///     let ii = Matrix::diagonal(&[1.0, 1.0, 1.0]);
    ///     mat_approx_eq(&a_times_ai, &ii, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn inverse(&self, ai: &mut Tensor2, tolerance: f64) -> Result<Option<f64>, StrError> {
        let dim = self.vec.dim();
        if ai.vec.dim() != dim {
            return Err("tensors are incompatible");
        }
        let a = &self.vec;
        match dim {
            4 => {
                let det = a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0;
                if f64::abs(det) > tolerance {
                    ai.vec[0] = (a[1] * a[2]) / det;
                    ai.vec[1] = (a[0] * a[2]) / det;
                    ai.vec[2] = (a[0] * a[1] - a[3] * a[3] / 2.0) / det;
                    ai.vec[3] = -((a[2] * a[3]) / det);
                    return Ok(Some(det));
                }
            }
            6 => {
                let det = a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0 - (a[0] * a[4] * a[4]) / 2.0
                    + (a[3] * a[4] * a[5]) / SQRT_2
                    - (a[1] * a[5] * a[5]) / 2.0;
                if f64::abs(det) > tolerance {
                    ai.vec[0] = (a[1] * a[2] - a[4] * a[4] / 2.0) / det;
                    ai.vec[1] = (a[0] * a[2] - a[5] * a[5] / 2.0) / det;
                    ai.vec[2] = (a[0] * a[1] - a[3] * a[3] / 2.0) / det;
                    ai.vec[3] = (-2.0 * a[2] * a[3] + SQRT_2 * a[4] * a[5]) / (2.0 * det);
                    ai.vec[4] = (-2.0 * a[0] * a[4] + SQRT_2 * a[3] * a[5]) / (2.0 * det);
                    ai.vec[5] = (SQRT_2 * a[3] * a[4] - 2.0 * a[1] * a[5]) / (2.0 * det);
                    return Ok(Some(det));
                }
            }
            _ => {
                let det = a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0 - (a[0] * a[4] * a[4]) / 2.0
                    + (a[3] * a[4] * a[5]) / SQRT_2
                    - (a[1] * a[5] * a[5]) / 2.0
                    + (a[2] * a[6] * a[6]) / 2.0
                    + (a[5] * a[6] * a[7]) / SQRT_2
                    + (a[0] * a[7] * a[7]) / 2.0
                    - (a[4] * a[6] * a[8]) / SQRT_2
                    - (a[3] * a[7] * a[8]) / SQRT_2
                    + (a[1] * a[8] * a[8]) / 2.0;
                if f64::abs(det) > tolerance {
                    ai.vec[0] = (2.0 * a[1] * a[2] - a[4] * a[4] + a[7] * a[7]) / (2.0 * det);
                    ai.vec[1] = (2.0 * a[0] * a[2] - a[5] * a[5] + a[8] * a[8]) / (2.0 * det);
                    ai.vec[2] = (2.0 * a[0] * a[1] - a[3] * a[3] + a[6] * a[6]) / (2.0 * det);
                    ai.vec[3] = -((SQRT_2 * a[2] * a[3] - a[4] * a[5] + a[7] * a[8]) / (SQRT_2 * det));
                    ai.vec[4] = -((SQRT_2 * a[0] * a[4] - a[3] * a[5] + a[6] * a[8]) / (SQRT_2 * det));
                    ai.vec[5] = (a[3] * a[4] - SQRT_2 * a[1] * a[5] + a[6] * a[7]) / (SQRT_2 * det);
                    ai.vec[6] = -((SQRT_2 * a[2] * a[6] + a[5] * a[7] - a[4] * a[8]) / (SQRT_2 * det));
                    ai.vec[7] = -((a[5] * a[6] + SQRT_2 * a[0] * a[7] - a[3] * a[8]) / (SQRT_2 * det));
                    ai.vec[8] = (a[4] * a[6] + a[3] * a[7] - SQRT_2 * a[1] * a[8]) / (SQRT_2 * det);
                    return Ok(Some(det));
                }
            }
        }
        Ok(None)
    }

    /// Calculates the squared tensor
    ///
    /// ```text
    /// A² = A · A
    /// ```
    ///
    /// ## Input/Output
    ///
    /// * `a2` -- a Tensor2 with matching dimensions to hold the squared tensor
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::vec_approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [10.0, 20.0, 10.0],
    ///         [ 4.0,  5.0,  6.0],
    ///         [ 2.0,  3.0,  5.0],
    ///     ], Mandel::General)?;
    ///
    ///     let mut a2 = Tensor2::new(Mandel::General);
    ///     a.squared(&mut a2)?;
    ///
    ///     let a2_correct = Tensor2::from_matrix(&[
    ///         [200.0, 330.0, 270.0],
    ///         [ 72.0, 123.0, 100.0],
    ///         [ 42.0,  70.0,  63.0],
    ///     ], Mandel::General)?;
    ///     vec_approx_eq(&a2.vec.as_data(), &a2_correct.vec.as_data(), 1e-13);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn squared(&self, a2: &mut Tensor2) -> Result<(), StrError> {
        let dim = self.vec.dim();
        if a2.vec.dim() != dim {
            return Err("tensors are incompatible");
        }
        let a = &self.vec;
        match dim {
            4 => {
                a2.vec[0] = a[0] * a[0] + a[3] * a[3] / 2.0;
                a2.vec[1] = a[1] * a[1] + a[3] * a[3] / 2.0;
                a2.vec[2] = a[2] * a[2];
                a2.vec[3] = (SQRT_2 * a[0] * a[3] + SQRT_2 * a[1] * a[3]) / SQRT_2;
            }
            6 => {
                a2.vec[0] = a[0] * a[0] + a[3] * a[3] / 2.0 + a[5] * a[5] / 2.0;
                a2.vec[1] = a[1] * a[1] + a[3] * a[3] / 2.0 + a[4] * a[4] / 2.0;
                a2.vec[2] = a[2] * a[2] + a[4] * a[4] / 2.0 + a[5] * a[5] / 2.0;
                a2.vec[3] = a[0] * a[3] + a[1] * a[3] + a[4] * a[5] / SQRT_2;
                a2.vec[4] = a[1] * a[4] + a[2] * a[4] + a[3] * a[5] / SQRT_2;
                a2.vec[5] = a[0] * a[5] + a[2] * a[5] + a[3] * a[4] / SQRT_2;
            }
            _ => {
                a2.vec[0] = a[0] * a[0] + ((a[3] - a[6]) * (a[3] + a[6])) / 2.0 + ((a[5] - a[8]) * (a[5] + a[8])) / 2.0;
                a2.vec[1] = a[1] * a[1] + ((a[3] - a[6]) * (a[3] + a[6])) / 2.0 + ((a[4] - a[7]) * (a[4] + a[7])) / 2.0;
                a2.vec[2] = a[2] * a[2] + ((a[4] - a[7]) * (a[4] + a[7])) / 2.0 + ((a[5] - a[8]) * (a[5] + a[8])) / 2.0;
                a2.vec[3] = ((a[0] * (a[3] - a[6])) / SQRT_2
                    + (a[1] * (a[3] - a[6])) / SQRT_2
                    + (a[0] * (a[3] + a[6])) / SQRT_2
                    + (a[1] * (a[3] + a[6])) / SQRT_2
                    + ((a[4] + a[7]) * (a[5] - a[8])) / 2.0
                    + ((a[4] - a[7]) * (a[5] + a[8])) / 2.0)
                    / SQRT_2;
                a2.vec[4] = ((a[1] * (a[4] - a[7])) / SQRT_2
                    + (a[2] * (a[4] - a[7])) / SQRT_2
                    + (a[1] * (a[4] + a[7])) / SQRT_2
                    + (a[2] * (a[4] + a[7])) / SQRT_2
                    + ((a[3] + a[6]) * (a[5] - a[8])) / 2.0
                    + ((a[3] - a[6]) * (a[5] + a[8])) / 2.0)
                    / SQRT_2;
                a2.vec[5] = ((a[0] * (a[5] + a[8])) / SQRT_2
                    + (a[2] * (a[5] + a[8])) / SQRT_2
                    + (a[0] * (a[5] - a[8])) / SQRT_2
                    + (a[2] * (a[5] - a[8])) / SQRT_2
                    + ((a[3] - a[6]) * (a[4] - a[7])) / 2.0
                    + ((a[3] + a[6]) * (a[4] + a[7])) / 2.0)
                    / SQRT_2;
                a2.vec[6] = (-(a[0] * (a[3] - a[6])) / SQRT_2 - (a[1] * (a[3] - a[6])) / SQRT_2
                    + (a[0] * (a[3] + a[6])) / SQRT_2
                    + (a[1] * (a[3] + a[6])) / SQRT_2
                    - ((a[4] + a[7]) * (a[5] - a[8])) / 2.0
                    + ((a[4] - a[7]) * (a[5] + a[8])) / 2.0)
                    / SQRT_2;
                a2.vec[7] = (-(a[1] * (a[4] - a[7])) / SQRT_2 - (a[2] * (a[4] - a[7])) / SQRT_2
                    + (a[1] * (a[4] + a[7])) / SQRT_2
                    + (a[2] * (a[4] + a[7])) / SQRT_2
                    - ((a[3] + a[6]) * (a[5] - a[8])) / 2.0
                    + ((a[3] - a[6]) * (a[5] + a[8])) / 2.0)
                    / SQRT_2;
                a2.vec[8] = (-(a[0] * (a[5] - a[8])) / SQRT_2 - (a[2] * (a[5] - a[8])) / SQRT_2
                    + (a[0] * (a[5] + a[8])) / SQRT_2
                    + (a[2] * (a[5] + a[8])) / SQRT_2
                    - ((a[3] - a[6]) * (a[4] - a[7])) / 2.0
                    + ((a[3] + a[6]) * (a[4] + a[7])) / 2.0)
                    / SQRT_2;
            }
        }
        Ok(())
    }

    /// Calculates the trace
    ///
    /// ```text
    /// tr(σ) = σ:I = Σᵢ σᵢᵢ
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ], Mandel::General)?;
    ///
    ///     approx_eq(a.trace(), 15.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn trace(&self) -> f64 {
        self.vec[0] + self.vec[1] + self.vec[2]
    }

    /// Calculates the Euclidean norm
    ///
    /// ```text
    /// norm(σ) = √(σ:σ)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ], Mandel::General)?;
    ///
    ///     approx_eq(a.norm(), f64::sqrt(285.0), 1e-13);
    ///     Ok(())
    /// }
    /// ```
    pub fn norm(&self) -> f64 {
        let mut sm = self.vec[0] * self.vec[0]
            + self.vec[1] * self.vec[1]
            + self.vec[2] * self.vec[2]
            + self.vec[3] * self.vec[3];
        let dim = self.vec.dim();
        if dim > 4 {
            sm += self.vec[4] * self.vec[4] + self.vec[5] * self.vec[5];
        }
        if dim > 6 {
            sm += self.vec[6] * self.vec[6] + self.vec[7] * self.vec[7] + self.vec[8] * self.vec[8];
        }
        f64::sqrt(sm)
    }

    /// Calculates the deviator tensor
    ///
    /// ```text
    /// dev(σ) = σ - ⅓ tr(σ) I
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ], Mandel::General)?;
    ///
    ///     let mut dev = Tensor2::new(Mandel::General);
    ///     a.deviator(&mut dev).unwrap();
    ///     approx_eq(dev.trace(), 0.0, 1e-15);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", dev.to_matrix()),
    ///         "┌                ┐\n\
    ///          │ -4.0  2.0  3.0 │\n\
    ///          │  4.0  0.0  6.0 │\n\
    ///          │  7.0  8.0  4.0 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn deviator(&self, dev: &mut Tensor2) -> Result<(), StrError> {
        let dim = self.vec.dim();
        if dev.vec.dim() != dim {
            return Err("tensors are incompatible");
        }
        let m = (self.vec[0] + self.vec[1] + self.vec[2]) / 3.0;
        dev.vec[0] = self.vec[0] - m;
        dev.vec[1] = self.vec[1] - m;
        dev.vec[2] = self.vec[2] - m;
        dev.vec[3] = self.vec[3];
        if dim > 4 {
            dev.vec[4] = self.vec[4];
            dev.vec[5] = self.vec[5];
        }
        if dim > 6 {
            dev.vec[6] = self.vec[6];
            dev.vec[7] = self.vec[7];
            dev.vec[8] = self.vec[8];
        }
        Ok(())
    }

    /// Calculates the norm of the deviator tensor
    ///
    /// ```text
    /// norm(dev(σ)) = ‖s‖ = ‖ σ - ⅓ tr(σ) I ‖
    ///
    /// ‖s‖² = ⅓ [(σ₁₁-σ₂₂)² + (σ₂₂-σ₃₃)² + (σ₃₃-σ₁₁)²]
    ///       + σ₁₂² + σ₂₃² + σ₁₃² + σ₂₁² + σ₃₂² + σ₃₁²
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [6.0,  1.0,  2.0],
    ///         [3.0, 12.0,  4.0],
    ///         [5.0,  6.0, 15.0],
    ///     ], Mandel::General)?;
    ///
    ///     let mut dev = Tensor2::new(Mandel::General);
    ///     a.deviator(&mut dev).unwrap();
    ///     approx_eq(dev.trace(), 0.0, 1e-15);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", dev.to_matrix()),
    ///         "┌                ┐\n\
    ///          │ -5.0  1.0  2.0 │\n\
    ///          │  3.0  1.0  4.0 │\n\
    ///          │  5.0  6.0  4.0 │\n\
    ///          └                ┘"
    ///     );
    ///
    ///     approx_eq(dev.norm(), f64::sqrt(133.0), 1e-15);
    ///     approx_eq(a.deviator_norm(), f64::sqrt(133.0), 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn deviator_norm(&self) -> f64 {
        let a = &self.vec;
        let mut sq_norm_s = a[3] * a[3]
            + (a[0] - a[1]) * (a[0] - a[1]) / 3.0
            + (a[1] - a[2]) * (a[1] - a[2]) / 3.0
            + (a[2] - a[0]) * (a[2] - a[0]) / 3.0;
        let dim = a.dim();
        if dim > 4 {
            sq_norm_s += a[4] * a[4] + a[5] * a[5];
        }
        if dim > 6 {
            sq_norm_s += a[6] * a[6] + a[7] * a[7] + a[8] * a[8];
        }
        f64::sqrt(sq_norm_s)
    }

    /// Calculates the determinant of the deviator tensor
    ///
    /// ```text
    /// det( σ - ⅓ tr(σ) I )
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [6.0,  1.0,  2.0],
    ///         [3.0, 12.0,  4.0],
    ///         [5.0,  6.0, 15.0],
    ///     ], Mandel::General)?;
    ///
    ///     let mut dev = Tensor2::new(Mandel::General);
    ///     a.deviator(&mut dev).unwrap();
    ///     approx_eq(dev.trace(), 0.0, 1e-15);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", dev.to_matrix()),
    ///         "┌                ┐\n\
    ///          │ -5.0  1.0  2.0 │\n\
    ///          │  3.0  1.0  4.0 │\n\
    ///          │  5.0  6.0  4.0 │\n\
    ///          └                ┘"
    ///     );
    ///
    ///     approx_eq(dev.determinant(), 134.0, 1e-13);
    ///     approx_eq(a.deviator_determinant(), 134.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn deviator_determinant(&self) -> f64 {
        let a = &self.vec;
        let m = (a[0] + a[1] + a[2]) / 3.0;
        match self.vec.dim() {
            4 => (a[2] - m) * (m * m + a[0] * a[1] - m * (a[0] + a[1]) - a[3] * a[3] / 2.0),
            6 => {
                (2.0 * m * m * (a[0] + a[1] + a[2]) - a[2] * a[3] * a[3] + a[0] * (2.0 * a[1] * a[2] - a[4] * a[4])
                    - 2.0 * m * m * m
                    + SQRT_2 * a[3] * a[4] * a[5]
                    - a[1] * a[5] * a[5]
                    + m * (-2.0 * a[1] * a[2] - 2.0 * a[0] * (a[1] + a[2]) + a[3] * a[3] + a[4] * a[4] + a[5] * a[5]))
                    / 2.0
            }
            _ => {
                (2.0 * (a[2] - m)
                    * (2.0 * m * m + 2.0 * a[0] * a[1] - 2.0 * m * (a[0] + a[1]) - a[3] * a[3] + a[6] * a[6])
                    + SQRT_2 * (a[5] - a[8]) * ((a[3] + a[6]) * (a[4] + a[7]) + SQRT_2 * (m - a[1]) * (a[5] + a[8]))
                    + SQRT_2 * (a[4] - a[7]) * ((a[3] - a[6]) * (a[5] + a[8]) + SQRT_2 * (m - a[0]) * (a[4] + a[7])))
                    / 4.0
            }
        }
    }

    // --- PRINCIPAL INVARIANTS -------------------------------------------------------------------------------------------

    /// Calculates I1, the first principal invariant
    ///
    /// ```text
    /// I1 = trace(σ)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let sig = Tensor2::from_matrix(&[
    ///         [50.0,  30.0,  20.0],
    ///         [30.0, -20.0, -10.0],
    ///         [20.0, -10.0,  10.0],
    ///     ], Mandel::Symmetric)?;
    ///     approx_eq(sig.invariant_ii1(), 40.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn invariant_ii1(&self) -> f64 {
        self.trace()
    }

    /// Calculates I2, the second principal invariant
    ///
    /// ```text
    /// I2 = ½ (trace(σ))² - ½ trace(σ·σ)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let sig = Tensor2::from_matrix(&[
    ///         [50.0,  30.0,  20.0],
    ///         [30.0, -20.0, -10.0],
    ///         [20.0, -10.0,  10.0],
    ///     ], Mandel::Symmetric)?;
    ///     approx_eq(sig.invariant_ii2(), -2100.0, 1e-12);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_ii2(&self) -> f64 {
        let a = &self.vec;
        let mut ii2 = a[0] * a[1] + a[0] * a[2] + a[1] * a[2] - a[3] * a[3] / 2.0;
        let dim = self.vec.dim();
        if dim > 4 {
            ii2 -= (a[4] * a[4] + a[5] * a[5]) / 2.0;
        }
        if dim > 6 {
            ii2 += (a[6] * a[6] + a[7] * a[7] + a[8] * a[8]) / 2.0;
        }
        ii2
    }

    /// Calculates I3, the third principal invariant
    ///
    /// ```text
    /// I3 = determinant(σ)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let sig = Tensor2::from_matrix(&[
    ///         [50.0,  30.0,  20.0],
    ///         [30.0, -20.0, -10.0],
    ///         [20.0, -10.0,  10.0],
    ///     ], Mandel::Symmetric)?;
    ///     approx_eq(sig.invariant_ii3(), -28000.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn invariant_ii3(&self) -> f64 {
        self.determinant()
    }

    /// Calculates J2, the second invariant of the deviatoric tensor corresponding to this tensor
    ///
    /// ```text
    /// s = deviator(σ)
    ///
    /// J2 = -IIₛ = ½ trace(s·s) = ½ s : sᵀ
    /// ```
    ///
    /// **Note:** if the tensor is symmetric, then:
    ///
    /// ```text
    /// J2 = ½ s : sᵀ = ½ s : s = ½ ‖s‖² (symmetric σ and s)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let sig = Tensor2::from_matrix(&[
    ///         [ 2.0, -3.0, 4.0],
    ///         [-3.0, -5.0, 1.0],
    ///         [ 4.0,  1.0, 6.0],
    ///     ], Mandel::Symmetric)?;
    ///     approx_eq(sig.invariant_jj2(), 57.0, 1e-14);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_jj2(&self) -> f64 {
        let a = &self.vec;
        match self.vec.dim() {
            4 => {
                (2.0 * (a[0] * a[0] + a[1] * a[1] - a[1] * a[2] + a[2] * a[2] - a[0] * (a[1] + a[2]))
                    + 3.0 * a[3] * a[3])
                    / 6.0
            }
            6 => {
                (2.0 * (a[0] * a[0] + a[1] * a[1] - a[1] * a[2] + a[2] * a[2] - a[0] * (a[1] + a[2]))
                    + 3.0 * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5]))
                    / 6.0
            }
            _ => {
                (2.0 * (a[0] * a[0] + a[1] * a[1] - a[1] * a[2] + a[2] * a[2] - a[0] * (a[1] + a[2]))
                    + 3.0 * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5] - a[6] * a[6] - a[7] * a[7] - a[8] * a[8]))
                    / 6.0
            }
        }
    }

    /// Calculates J3, the second invariant of the deviatoric tensor corresponding to this tensor
    ///
    /// ```text
    /// s = deviator(σ)
    ///
    /// J3 = IIIₛ = determinant(s)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let sig = Tensor2::from_matrix(&[
    ///         [ 2.0, -3.0, 4.0],
    ///         [-3.0, -5.0, 1.0],
    ///         [ 4.0,  1.0, 6.0],
    ///     ], Mandel::Symmetric)?;
    ///     approx_eq(sig.invariant_jj3(), -4.0, 1e-13);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn invariant_jj3(&self) -> f64 {
        self.deviator_determinant()
    }

    // --- OCTAHEDRAL INVARIANTS ------------------------------------------------------------------------------------------

    /// Returns the mean pressure invariant
    ///
    /// ```text
    /// σm = ⅓ trace(σ)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ], Mandel::Symmetric)?;
    ///     approx_eq(a.invariant_sigma_m(), 2.0 / 3.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn invariant_sigma_m(&self) -> f64 {
        self.trace() / 3.0
    }

    /// Returns the deviatoric stress invariant (von Mises)
    ///
    /// This quantity is also known as the **von Mises** effective invariant
    /// or equivalent stress.
    ///
    /// ```text
    /// σd = norm(dev(σ)) × √3/√2 = ‖s‖ √3/√2 = √3 × J2
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ], Mandel::Symmetric)?;
    ///     approx_eq(a.invariant_sigma_d(), 1.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn invariant_sigma_d(&self) -> f64 {
        self.deviator_norm() * SQRT_3_BY_2
    }

    /// Returns the volumetric strain invariant
    ///
    /// ```text
    /// εv = trace(ε)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ], Mandel::Symmetric)?;
    ///     approx_eq(a.invariant_eps_v(), 2.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn invariant_eps_v(&self) -> f64 {
        self.trace()
    }

    /// Returns the deviatoric strain invariant
    ///
    /// ```text
    /// εd = norm(dev(ε)) × √2/√3
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ], Mandel::Symmetric)?;
    ///     approx_eq(a.invariant_eps_d(), 2.0 / 3.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn invariant_eps_d(&self) -> f64 {
        self.deviator_norm() * SQRT_2_BY_3
    }

    /// Returns the Lode invariant
    ///
    /// ```text
    ///                  3 √3 J3
    /// l = cos(3θ) = ─────────────
    ///               2 pow(J2,1.5)
    /// ```
    ///
    /// # Returns
    ///
    /// If `J2 > TOL_J2`, returns `l`. Otherwise, returns None.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Mandel, Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::from_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ], Mandel::Symmetric)?;
    ///     if let Some(l) = a.invariant_lode() {
    ///         approx_eq(l, -1.0, 1e-15);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_lode(&self) -> Option<f64> {
        let jj2 = self.invariant_jj2();
        if jj2 > TOL_J2 {
            let jj3 = self.invariant_jj3();
            Some(1.5 * SQRT_3 * jj3 / f64::powf(jj2, 1.5))
        } else {
            None
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Tensor2;
    use crate::{Mandel, SampleTensor2, SamplesTensor2, IDENTITY2, SQRT_2, SQRT_3};
    use russell_lab::{approx_eq, mat_approx_eq, mat_mat_mul, math::PI, vec_approx_eq, Matrix};
    use serde::{Deserialize, Serialize};

    #[test]
    fn new_and_case_work() {
        // general
        let tt = Tensor2::new(Mandel::General);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(tt.vec.as_data(), correct);
        assert_eq!(tt.case(), Mandel::General);

        // symmetric 3D
        let tt = Tensor2::new(Mandel::Symmetric);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(tt.vec.as_data(), correct);
        assert_eq!(tt.case(), Mandel::Symmetric);

        let tt = Tensor2::new_sym(false);
        assert_eq!(tt.vec.as_data(), correct);
        assert_eq!(tt.case(), Mandel::Symmetric);

        let tt = Tensor2::new_sym_ndim(3);
        assert_eq!(tt.vec.as_data(), correct);
        assert_eq!(tt.case(), Mandel::Symmetric);

        // symmetric 2D
        let tt = Tensor2::new(Mandel::Symmetric2D);
        let correct = &[0.0, 0.0, 0.0, 0.0];
        assert_eq!(tt.vec.as_data(), correct);
        assert_eq!(tt.case(), Mandel::Symmetric2D);

        let tt = Tensor2::new_sym(true);
        assert_eq!(tt.vec.as_data(), correct);
        assert_eq!(tt.case(), Mandel::Symmetric2D);

        let tt = Tensor2::new_sym_ndim(2);
        assert_eq!(tt.vec.as_data(), correct);
        assert_eq!(tt.case(), Mandel::Symmetric2D);
    }

    #[test]
    fn set_matrix_captures_errors() {
        // symmetric 3D
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std_10 = &[
            [1.0, 4.0, 6.0],
            [4.0+eps, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        #[rustfmt::skip]
        let comps_std_20 = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0+eps, 5.0, 3.0],
        ];
        #[rustfmt::skip]
        let comps_std_21 = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0+eps, 3.0],
        ];
        let mut tt = Tensor2::new(Mandel::Symmetric);
        assert_eq!(
            tt.set_matrix(comps_std_10).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );
        assert_eq!(
            tt.set_matrix(comps_std_20).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );
        assert_eq!(
            tt.set_matrix(comps_std_21).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );

        // symmetric 2D
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std_12 = &[
            [1.0,     4.0, 0.0+eps],
            [4.0,     2.0, 0.0],
            [0.0+eps, 0.0, 3.0],
        ];
        #[rustfmt::skip]
        let comps_std_02 = &[
            [1.0, 4.0,     0.0],
            [4.0, 2.0,     0.0+eps],
            [0.0, 0.0+eps, 3.0],
        ];
        let mut tt = Tensor2::new(Mandel::Symmetric2D);
        assert_eq!(
            tt.set_matrix(comps_std_12).err(),
            Some("cannot set Symmetric2D Tensor2 with non-zero off-diagonal data")
        );
        assert_eq!(
            tt.set_matrix(comps_std_02).err(),
            Some("cannot set Symmetric2D Tensor2 with non-zero off-diagonal data")
        );
    }

    #[test]
    fn set_matrix_works() {
        // general
        let mut tt = Tensor2::new(Mandel::General);
        const NOISE: f64 = 1234.568;
        tt.vec.fill(NOISE);
        tt.set_matrix(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            .unwrap();
        let correct = &[
            1.0,
            5.0,
            9.0,
            6.0 / SQRT_2,
            14.0 / SQRT_2,
            10.0 / SQRT_2,
            -2.0 / SQRT_2,
            -2.0 / SQRT_2,
            -4.0 / SQRT_2,
        ];
        vec_approx_eq(tt.vec.as_data(), correct, 1e-15);

        // general (using nested Vec)
        let mut tt = Tensor2::new(Mandel::General);
        tt.vec.fill(NOISE);
        tt.set_matrix(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]])
            .unwrap();
        let correct = &[
            1.0,
            5.0,
            9.0,
            6.0 / SQRT_2,
            14.0 / SQRT_2,
            10.0 / SQRT_2,
            -2.0 / SQRT_2,
            -2.0 / SQRT_2,
            -4.0 / SQRT_2,
        ];
        vec_approx_eq(tt.vec.as_data(), correct, 1e-15);

        // symmetric 3D
        let mut tt = Tensor2::new(Mandel::Symmetric);
        tt.vec.fill(NOISE);
        tt.set_matrix(&[[1.0, 4.0, 6.0], [4.0, 2.0, 5.0], [6.0, 5.0, 3.0]])
            .unwrap();
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0 * SQRT_2];
        vec_approx_eq(tt.vec.as_data(), correct, 1e-14);

        // symmetric 2D
        let mut tt = Tensor2::new(Mandel::Symmetric2D);
        tt.vec.fill(NOISE);
        tt.set_matrix(&[[1.0, 4.0, 0.0], [4.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
            .unwrap();
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2];
        vec_approx_eq(tt.vec.as_data(), correct, 1e-14);
    }

    #[test]
    fn from_matrix_captures_errors() {
        // symmetric 3D
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std_10 = &[
            [1.0, 4.0, 6.0],
            [4.0+eps, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        #[rustfmt::skip]
        let comps_std_20 = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0+eps, 5.0, 3.0],
        ];
        #[rustfmt::skip]
        let comps_std_21 = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0+eps, 3.0],
        ];
        assert_eq!(
            Tensor2::from_matrix(comps_std_10, Mandel::Symmetric).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );
        assert_eq!(
            Tensor2::from_matrix(comps_std_20, Mandel::Symmetric).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );
        assert_eq!(
            Tensor2::from_matrix(comps_std_21, Mandel::Symmetric).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );

        // symmetric 2D
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std_12 = &[
            [1.0,     4.0, 0.0+eps],
            [4.0,     2.0, 0.0],
            [0.0+eps, 0.0, 3.0],
        ];
        #[rustfmt::skip]
        let comps_std_02 = &[
            [1.0, 4.0,     0.0],
            [4.0, 2.0,     0.0+eps],
            [0.0, 0.0+eps, 3.0],
        ];
        assert_eq!(
            Tensor2::from_matrix(comps_std_12, Mandel::Symmetric2D).err(),
            Some("cannot set Symmetric2D Tensor2 with non-zero off-diagonal data")
        );
        assert_eq!(
            Tensor2::from_matrix(comps_std_02, Mandel::Symmetric2D).err(),
            Some("cannot set Symmetric2D Tensor2 with non-zero off-diagonal data")
        );
    }

    #[test]
    fn from_matrix_works() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        let correct = &[
            1.0,
            5.0,
            9.0,
            6.0 / SQRT_2,
            14.0 / SQRT_2,
            10.0 / SQRT_2,
            -2.0 / SQRT_2,
            -2.0 / SQRT_2,
            -4.0 / SQRT_2,
        ];
        vec_approx_eq(tt.vec.as_data(), correct, 1e-15);

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric).unwrap();
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0 * SQRT_2];
        vec_approx_eq(tt.vec.as_data(), correct, 1e-14);

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2];
        vec_approx_eq(tt.vec.as_data(), correct, 1e-14);
    }

    #[test]
    fn identity_works() {
        // general
        let ii = Tensor2::identity(Mandel::General);
        assert_eq!(ii.vec.as_data(), &IDENTITY2);

        // symmetric
        let ii = Tensor2::identity(Mandel::Symmetric);
        assert_eq!(ii.vec.as_data(), &IDENTITY2[0..6]);

        // symmetric 2d
        let ii = Tensor2::identity(Mandel::Symmetric2D);
        assert_eq!(ii.vec.as_data(), &IDENTITY2[0..4]);
    }

    #[test]
    fn get_works() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(tt.get(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(tt.get(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(tt.get(i, j), comps_std[i][j], 1e-14);
            }
        }
    }

    #[test]
    fn to_matrix_works() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        let res = tt.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric).unwrap();
        let res = tt.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        let res = tt.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get(i, j), comps_std[i][j], 1e-14);
            }
        }
    }

    #[test]
    #[should_panic]
    fn to_matrix_2d_panics_on_3d_case() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric).unwrap();
        tt.to_matrix_2d();
    }

    #[test]
    fn to_matrix_2d_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        let (t22, res) = tt.to_matrix_2d();
        assert_eq!(t22, 3.0);
        assert_eq!(
            format!("{:.1}", res),
            "┌         ┐\n\
             │ 1.0 4.0 │\n\
             │ 4.0 2.0 │\n\
             └         ┘"
        );

        #[rustfmt::skip]
        let data = &[
            [1.0, 2.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let tt = Tensor2::from_matrix(data, Mandel::Symmetric2D).unwrap();
        let (t22, a) = tt.to_matrix_2d();
        assert_eq!(t22, 4.0);
        assert_eq!(
            format!("{:.1}", a),
            "┌         ┐\n\
             │ 1.0 2.0 │\n\
             │ 2.0 3.0 │\n\
             └         ┘"
        );
    }

    #[test]
    fn to_general_works() {
        let tt = Tensor2::from_matrix(
            &[[1.0, 2.0 / SQRT_2, 0.0], [2.0 / SQRT_2, 3.0, 0.0], [0.0, 0.0, 4.0]],
            Mandel::Symmetric2D,
        )
        .unwrap();
        let tt_gen = tt.to_general();
        println!("{:.2}", tt.vec);
        println!("{:.2}", tt_gen.vec);

        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        let res = tt.to_general();
        assert_eq!(res.vec.dim(), 9);
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric).unwrap();
        let res = tt.to_general();
        assert_eq!(res.vec.dim(), 9);
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        let res = tt.to_general();
        assert_eq!(res.vec.dim(), 9);
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get(i, j), comps_std[i][j], 1e-14);
            }
        }
    }

    #[test]
    fn sym_set_works() {
        let mut a = Tensor2::new(Mandel::Symmetric);
        a.sym_set(0, 0, 1.0);
        a.sym_set(1, 1, 2.0);
        a.sym_set(2, 2, 3.0);
        a.sym_set(0, 1, 4.0);
        a.sym_set(1, 0, 4.0);
        a.sym_set(2, 0, 5.0);
        let out = a.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌             ┐\n\
             │ 1.0 4.0 5.0 │\n\
             │ 4.0 2.0 0.0 │\n\
             │ 5.0 0.0 3.0 │\n\
             └             ┘"
        );
    }

    #[test]
    fn clear_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let mut a = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        a.clear();
        assert_eq!(a.vec.as_data(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn sym_add_panics_on_non_sym() {
        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let mut a = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        a.sym_add(0, 0, 1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn sym_add_panics_on_i_greater_than_j() {
        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let mut a = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        a.sym_add(1, 0, 1.0, 1.0);
    }

    #[test]
    fn sym_add_works() {
        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let mut a = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        a.sym_add(0, 0, 10.0, 10.0);
        a.sym_add(1, 1, 10.0, 10.0);
        a.sym_add(2, 2, 10.0, 10.0);
        a.sym_add(0, 1, 10.0, 10.0); // must not do (1,0)
        let out = a.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                   ┐\n\
             │ 101.0 104.0   0.0 │\n\
             │ 104.0 102.0   0.0 │\n\
             │   0.0   0.0 103.0 │\n\
             └                   ┘"
        );

        // // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let mut a = Tensor2::from_matrix(comps_std, Mandel::Symmetric).unwrap();
        a.sym_add(0, 0, 10.0, 10.0);
        a.sym_add(1, 1, 10.0, 10.0);
        a.sym_add(2, 2, 10.0, 10.0);
        a.sym_add(0, 1, 10.0, 10.0); // must nod do (1,0)
        a.sym_add(0, 2, 10.0, 10.0); // must not do (2,0)
        a.sym_add(1, 2, 10.0, 10.0); // must not do (2,1)
        let out = a.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                   ┐\n\
             │ 101.0 104.0 106.0 │\n\
             │ 104.0 102.0 105.0 │\n\
             │ 106.0 105.0 103.0 │\n\
             └                   ┘"
        );
    }

    #[test]
    #[should_panic]
    fn sym_add_panics_on_lower_diagonal() {
        let mut a = Tensor2::new(Mandel::Symmetric2D);
        a.sym_add(1, 0, 1.0, 0.0);
    }

    #[test]
    fn mirror_and_add_capture_errors() {
        let mut a = Tensor2::new(Mandel::General);
        let b = Tensor2::new(Mandel::Symmetric);
        assert_eq!(a.mirror(&b).err(), Some("tensors are incompatible"));
        assert_eq!(a.add(1.0, &b).err(), Some("tensors are incompatible"));
    }

    #[test]
    fn mirror_and_add_work() {
        // general
        let mut a = Tensor2::new(Mandel::General);
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [1.0, 3.0, 1.0], 
            [2.0, 2.0, 2.0], 
            [3.0, 1.0, 3.0],
        ],
        Mandel::General).unwrap();
        let c = Tensor2::from_matrix(
            &[[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]],
            Mandel::General,
        )
        .unwrap();
        a.mirror(&b).unwrap();
        a.add(10.0, &c).unwrap();
        let out = a.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │ 1001.0 1003.0 1001.0 │\n\
             │ 1002.0 1002.0 1002.0 │\n\
             │ 1003.0 1001.0 1003.0 │\n\
             └                      ┘"
        );

        // symmetric 3D
        let mut a = Tensor2::new(Mandel::Symmetric);
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [1.0, 3.0, 1.0], 
            [3.0, 2.0, 2.0], 
            [1.0, 2.0, 3.0],
        ],
        Mandel::Symmetric).unwrap();
        let c = Tensor2::from_matrix(
            &[[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]],
            Mandel::Symmetric,
        )
        .unwrap();
        a.mirror(&b).unwrap();
        a.add(10.0, &c).unwrap();
        let out = a.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │ 1001.0 1003.0 1001.0 │\n\
             │ 1003.0 1002.0 1002.0 │\n\
             │ 1001.0 1002.0 1003.0 │\n\
             └                      ┘"
        );

        // symmetric 2D
        let mut a = Tensor2::new(Mandel::Symmetric2D);
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [1.0, 3.0, 0.0], 
            [3.0, 2.0, 0.0], 
            [0.0, 0.0, 3.0],
        ],
        Mandel::Symmetric2D).unwrap();
        let c = Tensor2::from_matrix(
            &[[100.0, 100.0, 0.0], [100.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            Mandel::Symmetric2D,
        )
        .unwrap();
        a.mirror(&b).unwrap();
        a.add(10.0, &c).unwrap();
        let out = a.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │ 1001.0 1003.0    0.0 │\n\
             │ 1003.0 1002.0    0.0 │\n\
             │    0.0    0.0 1003.0 │\n\
             └                      ┘"
        );
    }

    #[test]
    fn clone_and_serialize_work() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        // clone
        let mut cloned = tt.clone();
        cloned.vec[0] = -1.0;
        assert_eq!(
            format!("{:.1}", tt.to_matrix()),
            "┌             ┐\n\
             │ 1.0 2.0 3.0 │\n\
             │ 4.0 5.0 6.0 │\n\
             │ 7.0 8.0 9.0 │\n\
             └             ┘"
        );
        assert_eq!(
            format!("{:.1}", cloned.to_matrix()),
            "┌                ┐\n\
             │ -1.0  2.0  3.0 │\n\
             │  4.0  5.0  6.0 │\n\
             │  7.0  8.0  9.0 │\n\
             └                ┘"
        );
        // serialize
        let mut serialized = Vec::new();
        let mut serializer = rmp_serde::Serializer::new(&mut serialized);
        tt.serialize(&mut serializer).unwrap();
        assert!(serialized.len() > 0);
        // deserialize
        let mut deserializer = rmp_serde::Deserializer::new(&serialized[..]);
        let ss: Tensor2 = Deserialize::deserialize(&mut deserializer).unwrap();
        assert_eq!(
            format!("{:.1}", ss.to_matrix()),
            "┌             ┐\n\
             │ 1.0 2.0 3.0 │\n\
             │ 4.0 5.0 6.0 │\n\
             │ 7.0 8.0 9.0 │\n\
             └             ┘"
        );
    }

    #[test]
    fn debug_works() {
        let tt = Tensor2::new(Mandel::General);
        assert!(format!("{:?}", tt).len() > 0);
    }

    #[test]
    fn determinant_works() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        approx_eq(tt.determinant(), 0.0, 1e-13);

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric).unwrap();
        approx_eq(tt.determinant(), 101.0, 1e-13);

        // symmetric 3D (another test)
        #[rustfmt::skip]
        let comps_std = &[
            [ 1.0, -3.0, 4.0],
            [-3.0, -6.0, 1.0],
            [ 4.0,  1.0, 5.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric).unwrap();
        approx_eq(tt.determinant(), -4.0, 1e-13);

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        approx_eq(tt.determinant(), -42.0, 1e-13);
    }

    #[test]
    fn transpose_catches_errors() {
        let tt = Tensor2::new(Mandel::General);
        let mut tt2 = Tensor2::new(Mandel::Symmetric);
        assert_eq!(tt.transpose(&mut tt2).err(), Some("tensors are incompatible"));
    }

    fn check_transpose(tt: &Tensor2, tt_tran: &Tensor2) {
        let aa = tt.to_matrix();
        let aa_tran = tt_tran.to_matrix();
        for i in 1..3 {
            for j in 1..3 {
                assert_eq!(aa.get(i, j), aa_tran.get(j, i));
            }
        }
    }

    #[test]
    fn transpose_works() {
        // general
        let s = &SamplesTensor2::TENSOR_T;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::General).unwrap();
        let mut tt2 = Tensor2::new(Mandel::General);
        tt.transpose(&mut tt2).unwrap();
        check_transpose(&tt, &tt2);

        // symmetric 3D
        let s = &SamplesTensor2::TENSOR_U;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::Symmetric).unwrap();
        let mut tt2 = Tensor2::new(Mandel::Symmetric);
        tt.transpose(&mut tt2).unwrap();
        check_transpose(&tt, &tt2);

        // symmetric 2D
        let s = &SamplesTensor2::TENSOR_Y;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::Symmetric2D).unwrap();
        let mut tt2 = Tensor2::new(Mandel::Symmetric2D);
        tt.transpose(&mut tt2).unwrap();
        check_transpose(&tt, &tt2);
    }

    #[test]
    fn inverse_catches_errors() {
        let tt = Tensor2::new(Mandel::General);
        let mut tti = Tensor2::new(Mandel::Symmetric);
        assert_eq!(tt.inverse(&mut tti, 0.0).err(), Some("tensors are incompatible"));
    }

    fn check_inverse(tt: &Tensor2, tti: &Tensor2, tol: f64) {
        let aa = tt.to_matrix();
        let aai = tti.to_matrix();
        let mut ii = Matrix::new(3, 3);
        mat_mat_mul(&mut ii, 1.0, &aa, &aai).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    approx_eq(ii.get(i, j), 1.0, tol);
                } else {
                    approx_eq(ii.get(i, j), 0.0, tol);
                }
            }
        }
    }

    #[test]
    fn inverse_works() {
        // general with zero determinant
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        let mut tti = Tensor2::new(Mandel::General);
        let res = tt.inverse(&mut tti, 1e-10).unwrap();
        assert_eq!(res, None);

        // general with non-zero determinant
        let s = &SamplesTensor2::TENSOR_T;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::General).unwrap();
        let mut tti = Tensor2::new(Mandel::General);
        if let Some(det) = tt.inverse(&mut tti, 1e-10).unwrap() {
            assert_eq!(det, s.determinant);
        } else {
            panic!("zero determinant found");
        }
        check_inverse(&tt, &tti, 1e-15);

        // symmetric 3D with zero determinant
        let s = &SamplesTensor2::TENSOR_X;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::Symmetric).unwrap();
        let mut tti = Tensor2::new(Mandel::Symmetric);
        let res = tt.inverse(&mut tti, 1e-10).unwrap();
        assert_eq!(res, None);

        // symmetric 3D
        let s = &SamplesTensor2::TENSOR_U;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::Symmetric).unwrap();
        let mut tti = Tensor2::new(Mandel::Symmetric);
        if let Some(det) = tt.inverse(&mut tti, 1e-10).unwrap() {
            approx_eq(det, s.determinant, 1e-14);
        } else {
            panic!("zero determinant found");
        }
        check_inverse(&tt, &tti, 1e-13);

        // symmetric 2D with zero determinant
        let s = &SamplesTensor2::TENSOR_X;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::Symmetric2D).unwrap();
        let mut tti = Tensor2::new(Mandel::Symmetric2D);
        let res = tt.inverse(&mut tti, 1e-10).unwrap();
        assert_eq!(res, None);

        // symmetric 2D
        let s = &SamplesTensor2::TENSOR_Y;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::Symmetric2D).unwrap();
        let mut tti = Tensor2::new(Mandel::Symmetric2D);
        if let Some(det) = tt.inverse(&mut tti, 1e-10).unwrap() {
            assert_eq!(det, s.determinant);
        } else {
            panic!("zero determinant found");
        }
        check_inverse(&tt, &tti, 1e-15);
    }

    #[test]
    fn squared_catches_errors() {
        let tt = Tensor2::new(Mandel::General);
        let mut tt2 = Tensor2::new(Mandel::Symmetric);
        assert_eq!(tt.squared(&mut tt2).err(), Some("tensors are incompatible"));
    }

    fn check_squared(tt: &Tensor2, tt2: &Tensor2, tol: f64) {
        let aa = tt.to_matrix();
        let aa2 = tt2.to_matrix();
        let mut aa2_correct = Matrix::new(3, 3);
        mat_mat_mul(&mut aa2_correct, 1.0, &aa, &aa).unwrap();
        mat_approx_eq(&aa2, &aa2_correct, tol);
    }

    #[test]
    fn squared_works() {
        // general
        let s = &SamplesTensor2::TENSOR_T;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::General).unwrap();
        let mut tt2 = Tensor2::new(Mandel::General);
        tt.squared(&mut tt2).unwrap();
        check_squared(&tt, &tt2, 1e-13);

        // symmetric 3D
        let s = &SamplesTensor2::TENSOR_U;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::Symmetric).unwrap();
        let mut tt2 = Tensor2::new(Mandel::Symmetric);
        tt.squared(&mut tt2).unwrap();
        check_squared(&tt, &tt2, 1e-14);

        // symmetric 2D
        let s = &SamplesTensor2::TENSOR_Y;
        let tt = Tensor2::from_matrix(&s.matrix, Mandel::Symmetric2D).unwrap();
        let mut tt2 = Tensor2::new(Mandel::Symmetric2D);
        tt.squared(&mut tt2).unwrap();
        check_squared(&tt, &tt2, 1e-15);
    }

    #[test]
    fn trace_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        approx_eq(tt.trace(), 15.0, 1e-15);
    }

    #[test]
    fn norm_works() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        approx_eq(tt.norm(), f64::sqrt(285.0), 1e-15);

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [ 2.0, -3.0, 4.0],
            [-3.0, -5.0, 1.0],
            [ 4.0,  1.0, 6.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric).unwrap();
        approx_eq(tt.norm(), f64::sqrt(117.0), 1e-15);

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        approx_eq(tt.norm(), f64::sqrt(46.0), 1e-15);
    }

    #[test]
    fn deviator_catches_errors() {
        let tt = Tensor2::new(Mandel::General);
        let mut dev = Tensor2::new(Mandel::Symmetric);
        assert_eq!(tt.deviator(&mut dev).err(), Some("tensors are incompatible"));
    }

    #[test]
    fn deviator_norm_and_determinant_work() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::General).unwrap();
        let mut dev = Tensor2::new(Mandel::General);
        tt.deviator(&mut dev).unwrap();
        approx_eq(dev.trace(), 0.0, 1e-15);
        assert_eq!(
            format!("{:.1}", dev.to_matrix()),
            "┌                ┐\n\
             │ -4.0  2.0  3.0 │\n\
             │  4.0  0.0  6.0 │\n\
             │  7.0  8.0  4.0 │\n\
             └                ┘"
        );
        approx_eq(dev.norm(), tt.deviator_norm(), 1e-15);
        approx_eq(dev.determinant(), tt.deviator_determinant(), 1e-12);

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [ 2.0, -3.0, 4.0],
            [-3.0, -5.0, 1.0],
            [ 4.0,  1.0, 6.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric).unwrap();
        let mut dev = Tensor2::new(Mandel::Symmetric);
        tt.deviator(&mut dev).unwrap();
        approx_eq(dev.trace(), 0.0, 1e-15);
        assert_eq!(
            format!("{:.1}", dev.to_matrix()),
            "┌                ┐\n\
             │  1.0 -3.0  4.0 │\n\
             │ -3.0 -6.0  1.0 │\n\
             │  4.0  1.0  5.0 │\n\
             └                ┘"
        );
        approx_eq(dev.norm(), tt.deviator_norm(), 1e-14);
        approx_eq(dev.determinant(), tt.deviator_determinant(), 1e-15);

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, Mandel::Symmetric2D).unwrap();
        let mut dev = Tensor2::new(Mandel::Symmetric2D);
        tt.deviator(&mut dev).unwrap();
        approx_eq(dev.trace(), 0.0, 1e-15);
        assert_eq!(
            format!("{:.1}", dev.to_matrix()),
            "┌                ┐\n\
             │ -1.0  4.0  0.0 │\n\
             │  4.0  0.0  0.0 │\n\
             │  0.0  0.0  1.0 │\n\
             └                ┘"
        );
        approx_eq(dev.norm(), tt.deviator_norm(), 1e-15);
        approx_eq(dev.determinant(), tt.deviator_determinant(), 1e-15);
    }

    fn check_sample(
        sample: &SampleTensor2,
        case: Mandel,
        tol_norm: f64,
        tol_trace: f64,
        tol_det: f64,
        tol_dev_norm: f64,
        tol_dev_det: f64,
        verbose: bool,
    ) {
        let tt = Tensor2::from_matrix(&sample.matrix, case).unwrap();
        if verbose {
            println!("{}", sample.desc);
            println!("    err(norm) = {:?}", tt.norm() - sample.norm);
            println!("    err(trace) = {:?}", tt.trace() - sample.trace);
            println!("    err(determinant) = {:?}", tt.determinant() - sample.determinant);
            println!(
                "    err(deviator_norm) = {:?}",
                tt.deviator_norm() - sample.deviator_norm
            );
            println!(
                "    err(deviator_determinant) = {:?}",
                tt.deviator_determinant() - sample.deviator_determinant
            );
        }
        approx_eq(tt.norm(), sample.norm, tol_norm);
        approx_eq(tt.trace(), sample.trace, tol_trace);
        approx_eq(tt.determinant(), sample.determinant, tol_det);
        approx_eq(tt.deviator_norm(), sample.deviator_norm, tol_dev_norm);
        approx_eq(tt.deviator_determinant(), sample.deviator_determinant, tol_dev_det);
    }

    #[test]
    #[rustfmt::skip]
    fn properties_are_correct() {
        let verb = false;
        //                                                       norm   trace  det dev_norm dev_det
        check_sample(&SamplesTensor2::TENSOR_O, Mandel::General, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_I, Mandel::General, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_X, Mandel::General, 1e-15, 1e-15, 1e-15, 1e-15, 1e-13, verb);
        check_sample(&SamplesTensor2::TENSOR_Y, Mandel::General, 1e-13, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_Z, Mandel::General, 1e-15, 1e-15, 1e-14, 1e-14, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_U, Mandel::General, 1e-13, 1e-15, 1e-14, 1e-14, 1e-13, verb);
        check_sample(&SamplesTensor2::TENSOR_S, Mandel::General, 1e-13, 1e-15, 1e-14, 1e-15, 1e-13, verb);
        check_sample(&SamplesTensor2::TENSOR_R, Mandel::General, 1e-13, 1e-15, 1e-13, 1e-13, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_T, Mandel::General, 1e-13, 1e-15, 1e-15, 1e-14, 1e-15, verb);

        let verb = false;
        //                                                         norm   trace  det dev_norm dev_det
        check_sample(&SamplesTensor2::TENSOR_O, Mandel::Symmetric, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_I, Mandel::Symmetric, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_X, Mandel::Symmetric, 1e-15, 1e-15, 1e-15, 1e-15, 1e-13, verb);
        check_sample(&SamplesTensor2::TENSOR_Y, Mandel::Symmetric, 1e-13, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_Z, Mandel::Symmetric, 1e-15, 1e-15, 1e-14, 1e-14, 1e-14, verb);
        check_sample(&SamplesTensor2::TENSOR_U, Mandel::Symmetric, 1e-13, 1e-15, 1e-14, 1e-14, 1e-13, verb);
        check_sample(&SamplesTensor2::TENSOR_S, Mandel::Symmetric, 1e-13, 1e-15, 1e-14, 1e-15, 1e-13, verb);

        let verb = false;
        //                                                           norm   trace  det dev_norm dev_det
        check_sample(&SamplesTensor2::TENSOR_O, Mandel::Symmetric2D, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_I, Mandel::Symmetric2D, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_X, Mandel::Symmetric2D, 1e-15, 1e-15, 1e-15, 1e-15, 1e-13, verb);
        check_sample(&SamplesTensor2::TENSOR_Y, Mandel::Symmetric2D, 1e-13, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_sample(&SamplesTensor2::TENSOR_Z, Mandel::Symmetric2D, 1e-15, 1e-15, 1e-14, 1e-14, 1e-14, verb);
    }

    /// --- PRINCIPAL INVARIANTS -------------------------------------------------------------------------------------------

    fn check_iis(sample: &SampleTensor2, case: Mandel, tol_a: f64, tol_b: f64, tol_c: f64, tol_d: f64, verbose: bool) {
        let tt = Tensor2::from_matrix(&sample.matrix, case).unwrap();
        let jj2 = -sample.deviator_second_invariant;
        let jj3 = sample.deviator_determinant;
        if verbose {
            println!("{}", sample.desc);
            println!("    err(I1) = {:?}", f64::abs(tt.invariant_ii1() - sample.trace));
            println!(
                "    err(I2) = {:?}",
                f64::abs(tt.invariant_ii2() - sample.second_invariant)
            );
            println!("    err(I3) = {:?}", f64::abs(tt.invariant_ii3() - sample.determinant));
            println!("    err(J2) = {:?}", f64::abs(tt.invariant_jj2() - jj2));
            println!("    err(J3) = {:?}", f64::abs(tt.invariant_jj3() - jj3));
            if case == Mandel::Symmetric || case == Mandel::Symmetric2D {
                let norm_s = tt.deviator_norm();
                println!("    err(J2 - ½‖s‖²) = {:?}", f64::abs(jj2 - norm_s * norm_s / 2.0));
            }
        }
        approx_eq(tt.invariant_ii1(), sample.trace, tol_a);
        approx_eq(tt.invariant_ii2(), sample.second_invariant, tol_b);
        approx_eq(tt.invariant_ii3(), sample.determinant, tol_b);
        approx_eq(tt.invariant_jj2(), jj2, tol_c);
        approx_eq(tt.invariant_jj3(), jj3, tol_c);
        if case == Mandel::Symmetric || case == Mandel::Symmetric2D {
            let norm_s = tt.deviator_norm();
            approx_eq(jj2, norm_s * norm_s / 2.0, tol_d);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn principal_invariants_are_correct() {
        let verb = false;
        check_iis(&SamplesTensor2::TENSOR_O, Mandel::General, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_I, Mandel::General, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_X, Mandel::General, 1e-15, 1e-15, 1e-13, 1e-13, verb);
        check_iis(&SamplesTensor2::TENSOR_Y, Mandel::General, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_Z, Mandel::General, 1e-15, 1e-14, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_U, Mandel::General, 1e-15, 1e-14, 1e-13, 1e-13, verb);
        check_iis(&SamplesTensor2::TENSOR_S, Mandel::General, 1e-15, 1e-14, 1e-13, 1e-13, verb);
        check_iis(&SamplesTensor2::TENSOR_R, Mandel::General, 1e-15, 1e-13, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_T, Mandel::General, 1e-15, 1e-15, 1e-15, 1e-15, verb);

        let verb = false;
        check_iis(&SamplesTensor2::TENSOR_O, Mandel::Symmetric, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_I, Mandel::Symmetric, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_X, Mandel::Symmetric, 1e-15, 1e-15, 1e-13, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_Y, Mandel::Symmetric, 1e-13, 1e-15, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_Z, Mandel::Symmetric, 1e-15, 1e-14, 1e-14, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_U, Mandel::Symmetric, 1e-15, 1e-14, 1e-13, 1e-13, verb);
        check_iis(&SamplesTensor2::TENSOR_S, Mandel::Symmetric, 1e-15, 1e-14, 1e-13, 1e-14, verb);

        let verb = false;
        check_iis(&SamplesTensor2::TENSOR_O, Mandel::Symmetric2D, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_I, Mandel::Symmetric2D, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_X, Mandel::Symmetric2D, 1e-15, 1e-15, 1e-13, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_Y, Mandel::Symmetric2D, 1e-15, 1e-15, 1e-15, 1e-15, verb);
        check_iis(&SamplesTensor2::TENSOR_Z, Mandel::Symmetric2D, 1e-15, 1e-14, 1e-15, 1e-15, verb);
    }

    /// --- OCTAHEDRAL INVARIANTS ------------------------------------------------------------------------------------------

    fn alpha_deg(l1: f64, l2: f64, l3: f64) -> f64 {
        f64::atan2(2.0 * l1 - l2 - l3, (l3 - l2) * SQRT_3) * 180.0 / PI
    }

    fn check_lode(l: Option<f64>, correct: f64, tol: f64, must_be_none: bool) {
        match l {
            Some(ll) => {
                if must_be_none {
                    panic!("Lode invariant must be None");
                } else {
                    approx_eq(ll, correct, tol);
                }
            }
            None => {
                if !must_be_none {
                    panic!("Lode invariant must not be None");
                }
            }
        }
    }

    #[test]
    fn octahedral_invariants_are_correct() {
        let c = Mandel::Symmetric;
        let sigma_d_1 = SQRT_3 / 2.0; // sqrt(((0.5+0.5)² + (0.5)² + (-0.5)²)/3) * sqrt(3/2)
        let eps_d_1 = 1.0 / SQRT_3; // sqrt(((0.5+0.5)² + (0.5)² + (-0.5)²)/3) * sqrt(2/3)
        let sigma_d_2 = 1.0; // sqrt((1² + 1²)/3)* sqrt(3/2)
        let eps_d_2 = 2.0 / 3.0; // sqrt((1² + 1²)/3)* sqrt(2/3)

        // α = 0
        let (l1, l2, l3) = (0.0, -0.5, 0.5);
        approx_eq(alpha_deg(l1, l2, l3), 0.0, 1e-15);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 0.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = 30
        let (l1, l2, l3) = (1.0, 0.0, 1.0);
        approx_eq(alpha_deg(l1, l2, l3), 30.0, 1e-14);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 2.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 2.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        check_lode(tt.invariant_lode(), -1.0, 1e-15, false);

        // α = 60
        let (l1, l2, l3) = (0.5, -0.5, 0.0);
        approx_eq(alpha_deg(l1, l2, l3), 60.0, 1e-14);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 0.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = 90
        let (l1, l2, l3) = (1.0, 0.0, 0.0);
        approx_eq(alpha_deg(l1, l2, l3), 90.0, 1e-15);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 1.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 1.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        check_lode(tt.invariant_lode(), 1.0, 1e-15, false);

        // α = 120
        let (l1, l2, l3) = (0.5, 0.0, -0.5);
        approx_eq(alpha_deg(l1, l2, l3), 120.0, 1e-13);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 0.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = 150
        let (l1, l2, l3) = (1.0, 1.0, 0.0);
        approx_eq(alpha_deg(l1, l2, l3), 150.0, 1e-13);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 2.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 2.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        check_lode(tt.invariant_lode(), -1.0, 1e-15, false);

        // α = 180
        let (l1, l2, l3) = (0.0, 0.5, -0.5);
        approx_eq(alpha_deg(l1, l2, l3), 180.0, 1e-13);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 0.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = -150
        let (l1, l2, l3) = (0.0, 1.0, 0.0);
        approx_eq(alpha_deg(l1, l2, l3), -150.0, 1e-13);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 1.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 1.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        check_lode(tt.invariant_lode(), 1.0, 1e-15, false);

        // α = -120
        let (l1, l2, l3) = (-0.5, 0.5, 0.0);
        approx_eq(alpha_deg(l1, l2, l3), -120.0, 1e-13);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 0.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = -90
        let (l1, l2, l3) = (0.0, 1.0, 1.0);
        approx_eq(alpha_deg(l1, l2, l3), -90.0, 1e-13);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 2.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 2.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        check_lode(tt.invariant_lode(), -1.0, 1e-15, false);

        // α = -60
        let (l1, l2, l3) = (-0.5, 0.0, 0.5);
        approx_eq(alpha_deg(l1, l2, l3), -60.0, 1e-13);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 0.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = -30
        let (l1, l2, l3) = (0.0, 0.0, 1.0);
        approx_eq(alpha_deg(l1, l2, l3), -30.0, 1e-13);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        approx_eq(tt.invariant_sigma_m(), 1.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), sigma_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 1.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        check_lode(tt.invariant_lode(), 1.0, 1e-15, false);
    }

    #[test]
    fn octahedral_invariants_are_correct_simple() {
        // test from https://soilmodels.com/wp-content/uploads/2020/12/stress_space-2.wgl
        let (l1, l2, l3) = (193.18, 88.3, 18.52);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], Mandel::Symmetric).unwrap();
        approx_eq(tt.invariant_sigma_m(), 100.0, 1e-15);
        approx_eq(tt.invariant_sigma_d(), 152.28, 0.0053);
        let lode = tt.invariant_lode().unwrap();
        let theta = (f64::acos(lode) / 3.0) * 180.0 / PI;
        approx_eq(30.0 - theta, 6.62, 0.0019);
    }

    #[test]
    fn lode_invariant_handles_spacial_cases() {
        let c = Mandel::Symmetric;

        // norm(deviator) = 0  with l = 0
        let (l1, l2, l3) = (2.0, 2.0, 2.0);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        check_lode(tt.invariant_lode(), 0.0, 1e-15, true);

        // norm(deviator) > 1e-15  with l ~ -1 (note how l jumps from 0 to -1 for eps from -1e-5 to -1e-3)
        let (l1, l2, l3) = (2.0, 2.0, 2.0 - 1e-3);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], c).unwrap();
        check_lode(tt.invariant_lode(), -1.0, 1e-7, false);
    }

    #[test]
    fn new_from_oct_invariants_works() {
        let (sigma_m, sigma_d) = (1.0, 3.0);

        let tt = Tensor2::new_from_oct_invariants(sigma_m, sigma_d, 1.0, true).unwrap();
        assert_eq!(tt.vec.dim(), 4);
        approx_eq(tt.vec[0], 3.0, 1e-15);
        approx_eq(tt.vec[1], 0.0, 1e-15);
        approx_eq(tt.vec[2], 0.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        let tt = Tensor2::new_from_oct_invariants(sigma_m, sigma_d, 0.0, true).unwrap();
        assert_eq!(tt.vec.dim(), 4);
        approx_eq(tt.vec[0], 1.0 + SQRT_3, 1e-15);
        approx_eq(tt.vec[1], 1.0 - SQRT_3, 1e-15);
        approx_eq(tt.vec[2], 1.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        let tt = Tensor2::new_from_oct_invariants(sigma_m, sigma_d, -1.0, true).unwrap();
        assert_eq!(tt.vec.dim(), 4);
        approx_eq(tt.vec[0], 2.0, 1e-15);
        approx_eq(tt.vec[1], -1.0, 1e-15);
        approx_eq(tt.vec[2], 2.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // the following data corresponds to sigma_m = 1 and sigma_d = 3
        #[rustfmt::skip]
        let principal_stresses_and_lode = [
            ( 3.0          ,  0.0          ,  0.0          ,  1.0 ),
            ( 0.0          ,  3.0          ,  0.0          ,  1.0 ),
            ( 0.0          ,  0.0          ,  3.0          ,  1.0 ),
            ( 1.0 + SQRT_3 ,  1.0 - SQRT_3 ,  1.0          ,  0.0 ),
            ( 1.0 + SQRT_3 ,  1.0          ,  1.0 - SQRT_3 ,  0.0 ),
            ( 1.0          ,  1.0 + SQRT_3 ,  1.0 - SQRT_3 ,  0.0 ),
            ( 1.0 - SQRT_3 ,  1.0 + SQRT_3 ,  1.0          ,  0.0 ),
            ( 1.0          ,  1.0 - SQRT_3 ,  1.0 + SQRT_3 ,  0.0 ),
            ( 1.0 - SQRT_3 ,  1.0          ,  1.0 + SQRT_3 ,  0.0 ),
            ( 2.0          , -1.0          ,  2.0          , -1.0 ),
            ( 2.0          ,  2.0          , -1.0          , -1.0 ),
            (-1.0          ,  2.0          ,  2.0          , -1.0 ),
        ];
        let mut aux = Tensor2::new_sym(true);
        for (sigma_1, sigma_2, sigma_3, lode_correct) in &principal_stresses_and_lode {
            aux.vec[0] = *sigma_1;
            aux.vec[1] = *sigma_2;
            aux.vec[2] = *sigma_3;
            let sigma_m = aux.invariant_sigma_m();
            let sigma_d = aux.invariant_sigma_d();
            let lode = aux.invariant_lode().unwrap();
            approx_eq(sigma_m, 1.0, 1e-15);
            approx_eq(sigma_d, 3.0, 1e-15);
            approx_eq(lode, *lode_correct, 1e-15);
        }
    }
}
