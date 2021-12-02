use super::{IJ_TO_I, I_TO_IJ, SQRT_2};
use crate::StrError;
use russell_lab::Vector;
use std::cmp;
use std::fmt::{self, Write};

/// Implements a second-order tensor, symmetric or not
pub struct Tensor2 {
    /// Holds the components in Mandel basis as a 9D vector.
    /// dim = 9 (general), 6 (symmetric), or 4 (symmetric)
    pub(super) vec: Vector,
}

impl Tensor2 {
    /// Returns a new Tensor2, symmetric or not, with 0-valued components
    ///
    /// ```text
    ///                          ┌    ┐
    ///                          | M0 |
    ///                          | M1 |
    ///     ┌             ┐      | M2 |
    ///     | T00 T01 T02 |      | M3 |
    /// T = | T10 T11 T12 |  =>  | M4 |
    ///     | T20 T21 T22 |      | M5 |
    ///     └             ┘      | M6 |
    ///                          | M7 |
    ///                          | M8 |
    ///                          └    ┘
    /// ```
    ///
    /// # Input
    ///
    /// * `symmetric` -- whether this tensor is symmetric or not
    /// * `two_dim` -- 2D instead of 3D; effectively used only if symmetric
    ///                since unsymmetric tensors have always 9 components.
    pub fn new(symmetric: bool, two_dim: bool) -> Self {
        Tensor2 {
            vec: Vector::new(mandel_dim(symmetric, two_dim)),
        }
    }

    /// Returns a new Tensor2 constructed from the "standard" components
    ///
    /// # Input
    ///
    /// * `symmetric` -- whether this tensor is symmetric or not
    /// * `two_dim` -- 2D instead of 3D; effectively used only if symmetric
    ///                since unsymmetric tensors have always 9 components.
    pub fn from_tensor(tt: &[[f64; 3]; 3], symmetric: bool, two_dim: bool) -> Result<Self, StrError> {
        if symmetric {
            if tt[1][0] != tt[0][1] || tt[2][1] != tt[1][2] || tt[2][0] != tt[0][2] {
                return Err("the components of the symmetric second order tensor do not pass symmetry check");
            }
        }
        if two_dim {
            if tt[1][2] != 0.0 || tt[0][2] != 0.0 {
                return Err("the tensor cannot be represented in 2D because of non-zero off-diagonal values");
            }
        }
        let dim = mandel_dim(symmetric, two_dim);
        let mut data = Vector::new(dim);
        for m in 0..dim {
            let (i, j) = I_TO_IJ[m];
            if i == j {
                data[m] = tt[i][j];
            }
            if i < j {
                data[m] = (tt[i][j] + tt[j][i]) / SQRT_2;
            }
            if i > j {
                data[m] = (tt[j][i] - tt[i][j]) / SQRT_2;
            }
        }
        Ok(Tensor2 { vec: data })
    }

    /// Returns the (i,j) component
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let m = IJ_TO_I[i][j];
        match self.vec.dim() {
            4 => {
                if i == j {
                    self.vec[m]
                } else if m < 4 {
                    self.vec[m] / SQRT_2
                } else {
                    0.0
                }
            }
            6 => {
                if i == j {
                    self.vec[m]
                } else if i < j {
                    self.vec[m] / SQRT_2
                } else {
                    0.0
                }
            }
            _ => {
                if i == j {
                    self.vec[m]
                } else if i < j {
                    let n = IJ_TO_I[j][i];
                    (self.vec[m] + self.vec[n]) / SQRT_2
                } else {
                    let n = IJ_TO_I[j][i];
                    (self.vec[n] - self.vec[m]) / SQRT_2
                }
            }
        }
    }

    /// Returns a 2D array with the standard components of this second-order tensor
    pub fn to_tensor(&self) -> Vec<Vec<f64>> {
        let mut tt = vec![vec![0.0; 3]; 3];
        if self.vec.dim() < 9 {
            for m in 0..self.vec.dim() {
                let (i, j) = I_TO_IJ[m];
                tt[i][j] = self.get(i, j);
                if i < j {
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

impl fmt::Display for Tensor2 {
    /// Generates a string representation of the Tensor2
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // convert mandel values to full tensor
        let tt = self.to_tensor();
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..3 {
            for j in 0..3 {
                let val = tt[i][j];
                match f.precision() {
                    Some(v) => write!(&mut buf, "{:.1$}", val, v)?,
                    None => write!(&mut buf, "{}", val)?,
                }
                width = cmp::max(buf.chars().count(), width);
                buf.clear();
            }
        }
        // draw matrix
        width += 1;
        write!(f, "┌{:1$}┐\n", " ", width * 3 + 1)?;
        for i in 0..3 {
            if i > 0 {
                write!(f, " │\n")?;
            }
            for j in 0..3 {
                if j == 0 {
                    write!(f, "│")?;
                }
                let val = tt[i][j];
                match f.precision() {
                    Some(v) => write!(f, "{:>1$.2$}", val, width, v)?,
                    None => write!(f, "{:>1$}", val, width)?,
                }
            }
        }
        write!(f, " │\n")?;
        write!(f, "└{:1$}┘", " ", width * 3 + 1)?;
        Ok(())
    }
}

#[inline]
fn mandel_dim(symmetric: bool, two_dim: bool) -> usize {
    if symmetric {
        if two_dim {
            4
        } else {
            6
        }
    } else {
        9
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
    fn from_tensor_works() -> Result<(), StrError> {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_tensor(comps_std, false, false)?;
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
        let tt = Tensor2::from_tensor(comps_std, true, false)?;
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0 * SQRT_2];
        assert_vec_approx_eq!(tt.vec.as_data(), correct, 1e-14);

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::from_tensor(comps_std, true, true)?;
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2];
        assert_vec_approx_eq!(tt.vec.as_data(), correct, 1e-14);
        Ok(())
    }

    #[test]
    fn from_tensor_fails_on_invalid_data() {
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
            Tensor2::from_tensor(comps_std_10, true, false).err(),
            Some("the components of the symmetric second order tensor do not pass symmetry check")
        );
        assert_eq!(
            Tensor2::from_tensor(comps_std_20, true, false).err(),
            Some("the components of the symmetric second order tensor do not pass symmetry check")
        );
        assert_eq!(
            Tensor2::from_tensor(comps_std_21, true, false).err(),
            Some("the components of the symmetric second order tensor do not pass symmetry check")
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
            Tensor2::from_tensor(comps_std_12, true, true).err(),
            Some("the tensor cannot be represented in 2D because of non-zero off-diagonal values")
        );
        assert_eq!(
            Tensor2::from_tensor(comps_std_02, true, true).err(),
            Some("the tensor cannot be represented in 2D because of non-zero off-diagonal values")
        );
    }

    #[test]
    fn to_tensor_works() -> Result<(), StrError> {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::from_tensor(comps_std, false, false)?;
        let res = tt.to_tensor();
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
        let tt = Tensor2::from_tensor(comps_std, true, false)?;
        let res = tt.to_tensor();
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
        let tt = Tensor2::from_tensor(comps_std, true, true)?;
        let res = tt.to_tensor();
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(res[i][j], comps_std[i][j], 1e-14);
            }
        }
        Ok(())
    }

    #[test]
    fn display_trait_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let t2 = Tensor2::from_tensor(comps_std, false, false)?;
        assert_eq!(
            format!("{:.3}", t2),
            "┌                   ┐\n\
             │ 1.000 2.000 3.000 │\n\
             │ 4.000 5.000 6.000 │\n\
             │ 7.000 8.000 9.000 │\n\
             └                   ┘"
        );
        Ok(())
    }
}
