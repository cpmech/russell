use super::*;

/// Implements a second order tensor
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
    /// * comps_std - the standard components given with respect to an orthonormal Cartesian basis
    /// * symmetric - this is a symmetric tensor
    ///
    /// # Panics
    ///
    /// This method panics if the tensor is symmetric and the components_std are not.
    pub fn from_tensor(comps_std: &[[f64; 3]; 3], symmetric: bool) -> Self {
        if symmetric {
            if comps_std[1][0] != comps_std[0][1]
                || comps_std[2][1] != comps_std[1][2]
                || comps_std[2][0] != comps_std[0][2]
            {
                panic!("the components of symmetric tensor are invalid for symmetry");
            }
        }
        let size = if symmetric { 6 } else { 9 };
        let mut comps_mandel = vec![0.0; size];
        for i in 0..3 {
            let j0 = if symmetric { i } else { 0 };
            for j in j0..3 {
                let a = IJ_TO_I[i][j];
                if i == j {
                    comps_mandel[a] = comps_std[i][j];
                }
                if i < j {
                    comps_mandel[a] = (comps_std[i][j] + comps_std[j][i]) / SQRT_2;
                }
                if i > j {
                    comps_mandel[a] = (comps_std[j][i] - comps_std[i][j]) / SQRT_2;
                }
            }
        }
        Tensor2 {
            comps_mandel,
            symmetric,
        }
    }

    /// Returns a 2D array with the standard components of this second order tensor
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
                        tensor[i][j] =
                            (self.comps_mandel[map[i][j]] + self.comps_mandel[map[j][i]]) / SQRT_2
                    }
                    if i > j {
                        tensor[i][j] =
                            (self.comps_mandel[map[j][i]] - self.comps_mandel[map[i][j]]) / SQRT_2
                    }
                }
            }
        }
        tensor
    }
}

///////////////////////////////////////////////////////////////////////////////

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
        let symmetric = false;
        let t2 = Tensor2::from_tensor(comps_std, symmetric);
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
        let symmetric = true;
        let t2 = Tensor2::from_tensor(comps_std, symmetric);
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0 * SQRT_2];
        assert_vec_approx_eq!(t2.comps_mandel, correct, 1e-14);
    }

    #[test]
    #[should_panic(expected = "the components of symmetric tensor are invalid for symmetry")]
    fn from_symmetric_tensor_panics_on_invalid_data_10() {
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0+eps, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let symmetric = true;
        Tensor2::from_tensor(comps_std, symmetric);
    }

    #[test]
    #[should_panic(expected = "the components of symmetric tensor are invalid for symmetry")]
    fn from_symmetric_tensor_panics_on_invalid_data_21() {
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0+eps, 5.0, 3.0],
        ];
        let symmetric = true;
        Tensor2::from_tensor(comps_std, symmetric);
    }

    #[test]
    #[should_panic(expected = "the components of symmetric tensor are invalid for symmetry")]
    fn from_symmetric_tensor_panics_on_invalid_data_20() {
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0+eps, 3.0],
        ];
        let symmetric = true;
        Tensor2::from_tensor(comps_std, symmetric);
    }

    #[test]
    fn to_tensor_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let symmetric = false;
        let t2 = Tensor2::from_tensor(comps_std, symmetric);
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
        let symmetric = true;
        let t2 = Tensor2::from_tensor(comps_std, symmetric);
        let res = t2.to_tensor();
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(res[i][j], comps_std[i][j], 1e-14);
            }
        }
    }
}
