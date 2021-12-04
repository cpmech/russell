use super::{mandel_dim, IJ_TO_M, IJ_TO_M_SYM, M_TO_IJ, SQRT_2};
use crate::StrError;
use russell_lab::{Matrix, Vector};

/// Implements a second-order tensor, symmetric or not
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
    /// ```text
    ///                 ┌    ┐    ┌   ┐
    ///                 │ 00 │    │ 0 │
    ///                 │ 11 │    │ 1 │
    /// ┌          ┐    │ 22 │    │ 2 │
    /// │ 00 01 02 │    │ 01 │    │ 3 │
    /// │ 10 11 12 │ => │ 12 │ => │ 4 │
    /// │ 20 21 22 │    │ 02 │    │ 5 │
    /// └          ┘    │ 10 │    │ 6 │
    ///                 │ 21 │    │ 7 │
    ///                 │ 20 │    │ 8 │
    ///                 └    ┘    └   ┘
    /// ```
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Tensor2, SQRT_2};
    use crate::StrError;
    use russell_chk::{assert_approx_eq, assert_vec_approx_eq};

    #[test]
    fn new_tensor2_works() {
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
}
