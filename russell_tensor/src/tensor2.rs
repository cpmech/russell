use super::*;

/// Implements a second-order tensor, symmetric or not
pub struct Tensor2 {
    comps_mandel: Vec<f64>, // components in Mandel basis. len = 9 or 6 (symmetric)
    symmetric: bool,        // this is a symmetric tensor
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
    /// # Panics
    ///
    /// This method panics symmetric=true but the components are not symmetric.
    ///
    pub fn from_tensor(tt: &[[f64; 3]; 3], symmetric: bool) -> Self {
        if symmetric {
            if tt[1][0] != tt[0][1] || tt[2][1] != tt[1][2] || tt[2][0] != tt[0][2] {
                panic!("the components of symmetric second order tensor do not pass symmetry check");
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
        Tensor2 {
            comps_mandel: tt_bar,
            symmetric,
        }
    }

    /// Returns a 2D array with the standard components of this second-order tensor
    pub fn to_tensor(&self) -> Vec<Vec<f64>> {
        let mut tensor = vec![vec![0.0; 3]; 3];
        let map = if self.symmetric { IJ_TO_I } else { IJ_TO_I };
        if self.symmetric {
            for i in 0..3 {
                for j in 0..3 {
                    if i == j {
                        tensor[i][j] = self.comps_mandel[map[i][j]];
                    }
                    if i < j {
                        tensor[i][j] = (self.comps_mandel[map[i][j]] + 0.0) / SQRT_2
                    }
                    if i > j {
                        tensor[i][j] = (self.comps_mandel[map[j][i]] - 0.0) / SQRT_2
                    }
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    if i == j {
                        tensor[i][j] = self.comps_mandel[map[i][j]];
                    }
                    if i < j {
                        tensor[i][j] = (self.comps_mandel[map[i][j]] + self.comps_mandel[map[j][i]]) / SQRT_2
                    }
                    if i > j {
                        tensor[i][j] = (self.comps_mandel[map[j][i]] - self.comps_mandel[map[i][j]]) / SQRT_2
                    }
                }
            }
        }
        tensor
    }

    /// Inner product (double-dot of tensors)
    pub fn inner(&self, other: &Tensor2) -> f64 {
        let mut res = self.comps_mandel[0] * other.comps_mandel[0]
            + self.comps_mandel[1] * other.comps_mandel[1]
            + self.comps_mandel[2] * other.comps_mandel[2]
            + self.comps_mandel[3] * other.comps_mandel[3]
            + self.comps_mandel[4] * other.comps_mandel[4]
            + self.comps_mandel[5] * other.comps_mandel[5];
        if !self.symmetric && !other.symmetric {
            res += self.comps_mandel[6] * other.comps_mandel[6]
                + self.comps_mandel[7] * other.comps_mandel[7]
                + self.comps_mandel[8] * other.comps_mandel[8];
            // NOTE: if any tensor is unsymmetric, there is no need to augment res
            // because the extra three components are zero
        };
        res
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

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
    fn from_tensor_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let t2 = Tensor2::from_tensor(comps_std, false);
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
    }

    #[test]
    fn from_symmetric_tensor_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let t2 = Tensor2::from_tensor(comps_std, true);
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0 * SQRT_2];
        assert_vec_approx_eq!(t2.comps_mandel, correct, 1e-14);
    }

    #[test]
    #[should_panic(expected = "the components of symmetric second order tensor do not pass symmetry check")]
    fn from_symmetric_tensor_panics_on_invalid_data_10() {
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0+eps, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        Tensor2::from_tensor(comps_std, true);
    }

    #[test]
    #[should_panic(expected = "the components of symmetric second order tensor do not pass symmetry check")]
    fn from_symmetric_tensor_panics_on_invalid_data_21() {
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0+eps, 5.0, 3.0],
        ];
        Tensor2::from_tensor(comps_std, true);
    }

    #[test]
    #[should_panic(expected = "the components of symmetric second order tensor do not pass symmetry check")]
    fn from_symmetric_tensor_panics_on_invalid_data_20() {
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0+eps, 3.0],
        ];
        Tensor2::from_tensor(comps_std, true);
    }

    #[test]
    fn to_tensor_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let t2 = Tensor2::from_tensor(comps_std, false);
        let res = t2.to_tensor();
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(res[i][j], comps_std[i][j], 1e-14);
            }
        }
    }

    #[test]
    fn to_tensor_symmetric_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let t2 = Tensor2::from_tensor(comps_std, true);
        let res = t2.to_tensor();
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(res[i][j], comps_std[i][j], 1e-14);
            }
        }
    }

    #[test]
    fn inner_works() {
        #[rustfmt::skip]
        let a_comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        #[rustfmt::skip]
        let b_comps_std = &[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ];
        let a = Tensor2::from_tensor(a_comps_std, false);
        let b = Tensor2::from_tensor(b_comps_std, false);
        assert_approx_eq!(a.inner(&b), 165.0, 1e-15);
    }

    #[test]
    fn inner_symmetric_works() {
        #[rustfmt::skip]
        let a_comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        #[rustfmt::skip]
        let b_comps_std = &[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ];
        let a = Tensor2::from_tensor(a_comps_std, true);
        let b = Tensor2::from_tensor(b_comps_std, true);
        assert_approx_eq!(a.inner(&b), 162.0, 1e-13);
    }

    #[test]
    fn inner_symmetric_with_unsymmetric_works() {
        #[rustfmt::skip]
        let a_comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        #[rustfmt::skip]
        let b_comps_std = &[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ];
        let a = Tensor2::from_tensor(a_comps_std, true);
        let b = Tensor2::from_tensor(b_comps_std, false);
        assert_approx_eq!(a.inner(&b), 168.0, 1e-13);
    }
}
