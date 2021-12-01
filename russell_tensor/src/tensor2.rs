use super::{IJ_TO_I, I_TO_IJ, SQRT_2};
use crate::StrError;
use std::cmp;
use std::fmt::{self, Write};

/// Implements a second-order tensor, symmetric or not
pub struct Tensor2 {
    pub(super) comps_mandel: Vec<f64>, // components in Mandel basis. len = 9 or 6 (symmetric)
    pub(super) symmetric: bool,        // this is a symmetric tensor
}

impl Tensor2 {
    /// Returns a new Tensor2, symmetric or not, with 0-valued components
    pub fn new(symmetric: bool) -> Self {
        let size = if symmetric { 6 } else { 9 };
        Tensor2 {
            comps_mandel: vec![0.0; size],
            symmetric,
        }
    }

    /// Returns a new Tensor2 constructed from the "standard" components
    ///
    /// # Arguments
    ///
    /// * tt - the standard components given with respect to an orthonormal Cartesian basis
    /// * symmetric - this is a symmetric tensor
    ///
    pub fn from_tensor(tt: &[[f64; 3]; 3], symmetric: bool) -> Result<Self, StrError> {
        if symmetric {
            if tt[1][0] != tt[0][1] || tt[2][1] != tt[1][2] || tt[2][0] != tt[0][2] {
                return Err("the components of symmetric second order tensor do not pass symmetry check");
            }
        }
        let size = if symmetric { 6 } else { 9 };
        let mut tt_bar = vec![0.0; size];
        for i in 0..3 {
            let j0 = if symmetric { i } else { 0 };
            for j in j0..3 {
                let a = IJ_TO_I[i][j];
                if i == j {
                    tt_bar[a] = tt[i][j];
                }
                if i < j {
                    tt_bar[a] = (tt[i][j] + tt[j][i]) / SQRT_2;
                }
                if i > j {
                    tt_bar[a] = (tt[j][i] - tt[i][j]) / SQRT_2;
                }
            }
        }
        Ok(Tensor2 {
            comps_mandel: tt_bar,
            symmetric,
        })
    }

    /// Returns a 2D array with the standard components of this second-order tensor
    pub fn to_tensor(&self) -> Vec<Vec<f64>> {
        let mut tt = vec![vec![0.0; 3]; 3];
        if self.symmetric {
            for m in 0..6 {
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

    /// Returns the (i,j) component
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let m = IJ_TO_I[i][j];
        if self.symmetric {
            if i == j {
                self.comps_mandel[m]
            } else {
                self.comps_mandel[m] / SQRT_2
            }
        } else {
            let val = self.comps_mandel[m];
            if i == j {
                val
            } else if i < j {
                let n = IJ_TO_I[j][i];
                let next = self.comps_mandel[n];
                (val + next) / SQRT_2
            } else {
                let n = IJ_TO_I[j][i];
                let next = self.comps_mandel[n];
                (next - val) / SQRT_2
            }
        }
    }
}

impl fmt::Display for Tensor2 {
    /// Generates a string representation of the Matrix
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let comps_std = &[
    ///         [1.0, 4.0, 6.0],
    ///         [7.0, 2.0, 5.0],
    ///         [9.0, 8.0, 3.0],
    ///     ];
    ///     let t2 = Tensor2::from_tensor(comps_std, false)?;
    ///     assert_eq!(
    ///         format!("{:.2}", t2),
    ///         "┌                ┐\n\
    ///          │ 1.00 4.00 6.00 │\n\
    ///          │ 7.00 2.00 5.00 │\n\
    ///          │ 9.00 8.00 3.00 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Tensor2, SQRT_2};
    use crate::StrError;
    use russell_chk::{assert_approx_eq, assert_vec_approx_eq};

    #[test]
    fn new_tensor2_works() {
        let t2 = Tensor2::new(false);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_vec_approx_eq!(t2.comps_mandel, correct, 1e-15);
    }

    #[test]
    fn new_symmetric_tensor2_works() {
        let t2 = Tensor2::new(true);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_vec_approx_eq!(t2.comps_mandel, correct, 1e-15);
    }

    #[test]
    fn from_tensor_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let t2 = Tensor2::from_tensor(comps_std, false)?;
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
        assert_vec_approx_eq!(t2.comps_mandel, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn from_symmetric_tensor_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let t2 = Tensor2::from_tensor(comps_std, true)?;
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0 * SQRT_2];
        assert_vec_approx_eq!(t2.comps_mandel, correct, 1e-14);
        Ok(())
    }

    #[test]
    fn from_symmetric_tensor_fails_on_invalid_data() {
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
            Tensor2::from_tensor(comps_std_10, true).err(),
            Some("the components of symmetric second order tensor do not pass symmetry check")
        );
        assert_eq!(
            Tensor2::from_tensor(comps_std_20, true).err(),
            Some("the components of symmetric second order tensor do not pass symmetry check")
        );
        assert_eq!(
            Tensor2::from_tensor(comps_std_21, true).err(),
            Some("the components of symmetric second order tensor do not pass symmetry check")
        );
    }

    #[test]
    fn to_tensor_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let t2 = Tensor2::from_tensor(comps_std, false)?;
        let res = t2.to_tensor();
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(res[i][j], comps_std[i][j], 1e-14);
            }
        }
        Ok(())
    }

    #[test]
    fn to_tensor_symmetric_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let t2 = Tensor2::from_tensor(comps_std, true)?;
        let res = t2.to_tensor();
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
        let t2 = Tensor2::from_tensor(comps_std, false)?;
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
