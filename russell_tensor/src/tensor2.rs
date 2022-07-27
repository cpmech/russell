use super::{mandel_dim, IJ_TO_M, IJ_TO_M_SYM, M_TO_IJ, SQRT_2};
use crate::StrError;
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
/// │ T01 T11 T12 │ => 11 │   T11    │ 1
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
    /// * `symmetric` -- whether this tensor is symmetric or not, i.e., Tij = Tji
    /// * `two_dim` -- 2D instead of 3D. Only used if symmetric == true.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::Tensor2;
    ///
    /// let a = Tensor2::new(false, false);
    /// assert_eq!(a.vec.as_data(), &[0.0,0.0,0.0,  0.0,0.0,0.0,  0.0,0.0,0.0]);
    ///
    /// let b = Tensor2::new(true, false);
    /// assert_eq!(b.vec.as_data(), &[0.0,0.0,0.0,  0.0,0.0,0.0]);
    ///
    /// let c = Tensor2::new(true, true);
    /// assert_eq!(c.vec.as_data(), &[0.0,0.0,0.0,  0.0]);
    /// ```
    pub fn new(symmetric: bool, two_dim: bool) -> Self {
        Tensor2 {
            vec: Vector::new(mandel_dim(symmetric, two_dim)),
        }
    }

    /// Creates a new Tensor2 constructed from a matrix
    ///
    /// # Input
    ///
    /// * `tt` - the standard (not Mandel) Tij components given
    ///          with respect to an orthonormal Cartesian basis
    /// * `symmetric` -- whether this tensor is symmetric or not i.e., Tij = Tji
    /// * `two_dim` -- 2D instead of 3D. Only used if symmetric == true.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_chk::assert_vec_approx_eq;
    /// use russell_tensor::{Tensor2, SQRT_2, StrError};
    ///
    /// # fn main() -> Result<(), StrError> {
    /// // general
    /// let a = Tensor2::from_matrix(&[
    ///     [       1.0, SQRT_2*2.0, SQRT_2*3.0],
    ///     [SQRT_2*4.0,        5.0, SQRT_2*6.0],
    ///     [SQRT_2*7.0, SQRT_2*8.0,        9.0],
    /// ], false, false)?;
    /// let correct = &[1.0,5.0,9.0, 6.0,14.0,10.0, -2.0,-2.0,-4.0];
    /// assert_vec_approx_eq!(a.vec.as_data(), correct, 1e-14);
    ///
    /// // symmetric-3D
    /// let b = Tensor2::from_matrix(&[
    ///     [1.0,        4.0/SQRT_2, 6.0/SQRT_2],
    ///     [4.0/SQRT_2, 2.0,        5.0/SQRT_2],
    ///     [6.0/SQRT_2, 5.0/SQRT_2, 3.0       ],
    /// ], true, false)?;
    /// let correct = &[1.0,2.0,3.0, 4.0,5.0,6.0];
    /// assert_vec_approx_eq!(b.vec.as_data(), correct, 1e-14);
    ///
    /// // symmetric-2D
    /// let c = Tensor2::from_matrix(&[
    ///     [1.0,        4.0/SQRT_2, 0.0],
    ///     [4.0/SQRT_2, 2.0,        0.0],
    ///     [0.0,        0.0,        3.0],
    /// ], true, true)?;
    /// let correct = &[1.0,2.0,3.0, 4.0];
    /// assert_vec_approx_eq!(c.vec.as_data(), correct, 1e-14);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_matrix(tt: &[[f64; 3]; 3], symmetric: bool, two_dim: bool) -> Result<Self, StrError> {
        if symmetric {
            if tt[1][0] != tt[0][1] || tt[2][1] != tt[1][2] || tt[2][0] != tt[0][2] {
                return Err("symmetric Tensor2 does not pass symmetry check");
            }
        }
        if two_dim {
            if tt[1][2] != 0.0 || tt[0][2] != 0.0 {
                return Err("cannot define 2D Tensor2 due to non-zero off-diagonal values");
            }
        }
        let dim = mandel_dim(symmetric, two_dim);
        let mut vec = Vector::new(dim);
        for m in 0..dim {
            let (i, j) = M_TO_IJ[m];
            if i == j {
                vec[m] = tt[i][j];
            }
            if i < j {
                vec[m] = (tt[i][j] + tt[j][i]) / SQRT_2;
            }
            if i > j {
                vec[m] = (tt[j][i] - tt[i][j]) / SQRT_2;
            }
        }
        Ok(Tensor2 { vec })
    }

    /// Returns the (i,j) component (standard; not Mandel)
    ///
    /// # Example
    ///
    /// ```
    /// use russell_chk::assert_approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// # fn main() -> Result<(), StrError> {
    /// let a = Tensor2::from_matrix(&[
    ///     [1.0,  2.0, 0.0],
    ///     [3.0, -1.0, 5.0],
    ///     [0.0,  4.0, 1.0],
    /// ], false, false)?;
    ///
    /// assert_approx_eq!(a.get(1,2), 5.0, 1e-15);
    /// # Ok(())
    /// # }
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
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// # fn main() -> Result<(), StrError> {
    /// let a = Tensor2::from_matrix(&[
    ///     [1.0,  1.0, 0.0],
    ///     [1.0, -1.0, 0.0],
    ///     [0.0,  0.0, 1.0],
    /// ], true, true)?;
    ///
    /// let out = a.to_matrix();
    /// assert_eq!(
    ///     format!("{:.1}", out),
    ///     "┌                ┐\n\
    ///      │  1.0  1.0  0.0 │\n\
    ///      │  1.0 -1.0  0.0 │\n\
    ///      │  0.0  0.0  1.0 │\n\
    ///      └                ┘"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_matrix(&self) -> Matrix {
        let mut tt = Matrix::new(3, 3);
        let dim = self.vec.dim();
        if dim < 9 {
            for m in 0..dim {
                let (i, j) = M_TO_IJ[m];
                tt[i][j] = self.get(i, j);
                if i != j {
                    tt[j][i] = tt[i][j];
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    tt[i][j] = self.get(i, j);
                }
            }
        }
        tt
    }

    /// Sets the (i,j) component of a symmetric Tensor2
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
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// # fn main() -> Result<(), StrError> {
    /// let symmetric = true;
    /// let is_2d = true;
    /// let mut a = Tensor2::new(symmetric, is_2d);
    /// a.sym_set(0, 0, 1.0);
    /// a.sym_set(1, 1, 2.0);
    /// a.sym_set(2, 2, 3.0);
    /// a.sym_set(0, 1, 4.0);
    ///
    /// let out = a.to_matrix();
    /// assert_eq!(
    ///     format!("{:.1}", out),
    ///     "┌             ┐\n\
    ///      │ 1.0 4.0 0.0 │\n\
    ///      │ 4.0 2.0 0.0 │\n\
    ///      │ 0.0 0.0 3.0 │\n\
    ///      └             ┘"
    /// );
    ///
    /// let not_2d = false;
    /// let mut b = Tensor2::new(symmetric, not_2d);
    /// b.sym_set(0, 0, 1.0);
    /// b.sym_set(1, 1, 2.0);
    /// b.sym_set(2, 2, 3.0);
    /// b.sym_set(0, 1, 4.0);
    /// b.sym_set(1, 0, 4.0);
    /// b.sym_set(2, 0, 5.0);
    /// let out = b.to_matrix();
    /// assert_eq!(
    ///     format!("{:.1}", out),
    ///     "┌             ┐\n\
    ///      │ 1.0 4.0 5.0 │\n\
    ///      │ 4.0 2.0 0.0 │\n\
    ///      │ 5.0 0.0 3.0 │\n\
    ///      └             ┘"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn sym_set(&mut self, i: usize, j: usize, value: f64) {
        let m = IJ_TO_M_SYM[i][j];
        if i == j {
            self.vec[m] = value;
        } else {
            self.vec[m] = value * SQRT_2;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Tensor2, SQRT_2};
    use crate::StrError;
    use russell_chk::{assert_approx_eq, assert_vec_approx_eq};
    use serde::{Deserialize, Serialize};

    #[test]
    fn new_works() {
        // general
        let tt = Tensor2::new(false, false);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(tt.vec.as_data(), correct);

        // symmetric 3D
        let tt = Tensor2::new(true, false);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(tt.vec.as_data(), correct);

        // symmetric 2D
        let tt = Tensor2::new(true, true);
        let correct = &[0.0, 0.0, 0.0, 0.0];
        assert_eq!(tt.vec.as_data(), correct);
    }

    #[test]
    fn from_matrix_works() -> Result<(), StrError> {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, false, false)?;
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
        assert_vec_approx_eq!(tt.vec.as_data(), correct, 1e-15);

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, true, false)?;
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0 * SQRT_2];
        assert_vec_approx_eq!(tt.vec.as_data(), correct, 1e-14);

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, true, true)?;
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2];
        assert_vec_approx_eq!(tt.vec.as_data(), correct, 1e-14);
        Ok(())
    }

    #[test]
    fn from_matrix_fails_on_wrong_input() {
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
            Tensor2::from_matrix(comps_std_10, true, false).err(),
            Some("symmetric Tensor2 does not pass symmetry check")
        );
        assert_eq!(
            Tensor2::from_matrix(comps_std_20, true, false).err(),
            Some("symmetric Tensor2 does not pass symmetry check")
        );
        assert_eq!(
            Tensor2::from_matrix(comps_std_21, true, false).err(),
            Some("symmetric Tensor2 does not pass symmetry check")
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
            Tensor2::from_matrix(comps_std_12, true, true).err(),
            Some("cannot define 2D Tensor2 due to non-zero off-diagonal values")
        );
        assert_eq!(
            Tensor2::from_matrix(comps_std_02, true, true).err(),
            Some("cannot define 2D Tensor2 due to non-zero off-diagonal values")
        );
    }

    #[test]
    fn get_works() -> Result<(), StrError> {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, false, false)?;
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(tt.get(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, true, false)?;
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(tt.get(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, true, true)?;
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(tt.get(i, j), comps_std[i][j], 1e-14);
            }
        }
        Ok(())
    }

    #[test]
    fn to_matrix_works() -> Result<(), StrError> {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, false, false)?;
        let res = tt.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(res[i][j], comps_std[i][j], 1e-14);
            }
        }

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, true, false)?;
        let res = tt.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(res[i][j], comps_std[i][j], 1e-14);
            }
        }

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, true, true)?;
        let res = tt.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(res[i][j], comps_std[i][j], 1e-14);
            }
        }
        Ok(())
    }

    #[test]
    fn sym_set_works() -> Result<(), StrError> {
        let mut a = Tensor2::new(true, false);
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
        Ok(())
    }

    #[test]
    fn clone_and_serialize_work() -> Result<(), StrError> {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_matrix(comps_std, false, false)?;
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
        tt.serialize(&mut serializer).map_err(|_| "tensor serialize failed")?;
        assert!(serialized.len() > 0);
        // deserialize
        let mut deserializer = rmp_serde::Deserializer::new(&serialized[..]);
        let ss: Tensor2 = Deserialize::deserialize(&mut deserializer).map_err(|_| "cannot deserialize tensor data")?;
        assert_eq!(
            format!("{:.1}", ss.to_matrix()),
            "┌             ┐\n\
             │ 1.0 2.0 3.0 │\n\
             │ 4.0 5.0 6.0 │\n\
             │ 7.0 8.0 9.0 │\n\
             └             ┘"
        );
        Ok(())
    }

    #[test]
    fn debug_works() -> Result<(), StrError> {
        let tt = Tensor2::new(false, false);
        assert!(format!("{:?}", tt).len() > 0);
        Ok(())
    }
}
