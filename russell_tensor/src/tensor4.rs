use super::*;

/// Implements a fourth order-tensor, minor-symmetric or not
pub struct Tensor4 {
    comps_mandel: Vec<f64>, // components in Mandel basis. len = 81 or 36 (minor-symmetric). col-major => Fortran
    minor_symmetric: bool,  // this tensor has minor-symmetry
}

impl Tensor4 {
    /// Returns a new Tensor4, minor-symmetric or not, with 0-valued components
    pub fn new(minor_symmetric: bool) -> Self {
        let size = if minor_symmetric { 36 } else { 81 };
        Tensor4 {
            comps_mandel: vec![0.0; size],
            minor_symmetric,
        }
    }

    /// Returns a new Tensor4 constructed from the "standard" components
    ///
    /// # Arguments
    ///
    /// * dd - the standard D[i][j][k][l] components given with respect to an orthonormal Cartesian basis
    /// * minor_symmetric - this is a minor-symmetric tensor
    ///
    /// # Panics
    ///
    /// This method panics if minor_symmetric=true but the components are not symmetric.
    ///
    pub fn from_tensor(dd: &[[[[f64; 3]; 3]; 3]; 3], minor_symmetric: bool) -> Self {
        let size = if minor_symmetric { 6 } else { 9 };
        let mut dd_bar = vec![0.0; size * size];
        if minor_symmetric {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        for l in 0..3 {
                            // check minor-symmetry
                            if i > j || k > l {
                                if dd[i][j][k][l] != dd[j][i][k][l]
                                    || dd[i][j][k][l] != dd[i][j][l][k]
                                    || dd[i][j][k][l] != dd[j][i][l][k]
                                {
                                    panic!("the components of minor-symmetric tensor do not pass symmetry check");
                                }
                            } else {
                                let (a, b) = IJKL_TO_IJ[i][j][k][l];
                                let p = a + b * size; // col-major
                                if i == j && k == l {
                                    dd_bar[p] = dd[i][j][k][l];
                                }
                                if i == j && k < l {
                                    dd_bar[p] = SQRT_2 * dd[i][j][k][l];
                                }
                                if i < j && k == l {
                                    dd_bar[p] = SQRT_2 * dd[i][j][k][l];
                                }
                                if i < j && k < l {
                                    dd_bar[p] = 2.0 * dd[i][j][k][l];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        for l in 0..3 {
                            let (a, b) = IJKL_TO_IJ[i][j][k][l];
                            let p = a + b * size; // col-major
                            if i == j && k == l {
                                dd_bar[p] = dd[i][j][k][l];
                            }
                            if i == j && k < l {
                                dd_bar[p] = (dd[i][j][k][l] + dd[i][j][l][k]) / SQRT_2;
                            }
                            if i == j && k > l {
                                dd_bar[p] = (dd[i][j][l][k] - dd[i][j][k][l]) / SQRT_2;
                            }
                            if i < j && k == l {
                                dd_bar[p] = (dd[i][j][k][l] + dd[j][i][k][l]) / SQRT_2;
                            }
                            if i < j && k < l {
                                dd_bar[p] = (dd[i][j][k][l] + dd[i][j][l][k] + dd[j][i][k][l] + dd[j][i][l][k]) / 2.0;
                            }
                            if i < j && k > l {
                                dd_bar[p] = (dd[i][j][l][k] - dd[i][j][k][l] + dd[j][i][l][k] - dd[j][i][k][l]) / 2.0;
                            }
                            if i > j && k == l {
                                dd_bar[p] = (dd[j][i][k][l] - dd[i][j][k][l]) / SQRT_2;
                            }
                            if i > j && k < l {
                                dd_bar[p] = (dd[j][i][k][l] + dd[j][i][l][k] - dd[i][j][k][l] - dd[i][j][l][k]) / 2.0;
                            }
                            if i > j && k > l {
                                dd_bar[p] = (dd[j][i][l][k] - dd[j][i][k][l] - dd[i][j][l][k] + dd[i][j][k][l]) / 2.0;
                            }
                        }
                    }
                }
            }
        }
        Tensor4 {
            comps_mandel: dd_bar,
            minor_symmetric,
        }
    }

    /// Returns a deep array with the standard components of this fourth-order tensor
    pub fn to_tensor(&self) -> Vec<Vec<Vec<Vec<f64>>>> {
        let mut dd = vec![vec![vec![vec![0.0; 3]; 3]; 3]; 3];
        if self.minor_symmetric {
            for a in 0..6 {
                let [i, j] = I_TO_IJ[a];
                for b in 0..6 {
                    let [k, l] = I_TO_IJ[b];
                    let p = a + b * 6; // col-major
                    if i == j && k == l {
                        dd[i][j][k][l] = self.comps_mandel[p];
                    }
                    if i == j && k < l {
                        dd[i][j][k][l] = (self.comps_mandel[p] + 0.0) / SQRT_2;
                        dd[i][j][l][k] = dd[i][j][k][l];
                    }
                    if i < j && k == l {
                        dd[i][j][k][l] = (self.comps_mandel[p] + 0.0) / SQRT_2;
                        dd[j][i][k][l] = dd[i][j][k][l];
                    }
                    if i < j && k < l {
                        dd[i][j][k][l] = (self.comps_mandel[p] + 0.0 + 0.0 + 0.0) / 2.0;
                        dd[i][j][l][k] = dd[i][j][k][l];
                        dd[j][i][k][l] = dd[i][j][k][l];
                        dd[j][i][l][k] = dd[i][j][k][l];
                    }
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        for l in 0..3 {
                            let (a, b) = IJKL_TO_IJ[i][j][k][l];
                            let val = self.comps_mandel[a + b * 9];
                            if i == j && k == l {
                                dd[i][j][k][l] = val;
                            }
                            if i == j && k < l {
                                dd[i][j][k][l] = (val + dd[i][j][l][k]) / SQRT_2;
                            }
                            if i == j && k > l {
                                dd[i][j][k][l] = (dd[i][j][l][k] - val) / SQRT_2;
                            }
                            if i < j && k == l {
                                dd[i][j][k][l] = (val + dd[j][i][k][l]) / SQRT_2;
                            }
                            if i < j && k < l {
                                dd[i][j][k][l] = (val + dd[i][j][l][k] + dd[j][i][k][l] + dd[j][i][l][k]) / 2.0;
                            }
                            if i < j && k > l {
                                dd[i][j][k][l] = (dd[i][j][l][k] - val + dd[j][i][l][k] - dd[j][i][k][l]) / 2.0;
                            }
                            if i > j && k == l {
                                dd[i][j][k][l] = (dd[j][i][k][l] - val) / SQRT_2;
                            }
                            if i > j && k < l {
                                dd[i][j][k][l] = (dd[j][i][k][l] + dd[j][i][l][k] - val - dd[i][j][l][k]) / 2.0;
                            }
                            if i > j && k > l {
                                dd[i][j][k][l] = (dd[j][i][l][k] - dd[j][i][k][l] - dd[i][j][l][k] + val) / 2.0;
                            }
                        }
                    }
                }
            }
        }
        dd
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn new_tensor4_works() {
        let t4 = Tensor4::new(false);
        let correct = &[0.0; 81];
        assert_vec_approx_eq!(t4.comps_mandel, correct, 1e-15);
    }

    #[test]
    fn from_tensor_works() {
        let t4 = Tensor4::from_tensor(&Samples::TENSOR4_SAMPLE1, false);
        for a in 0..9 {
            for b in 0..9 {
                let p = a + b * 9; // col-major
                assert_eq!(t4.comps_mandel[p], Samples::TENSOR4_SAMPLE1_MANDEL_MATRIX[a][b]);
            }
        }
    }

    #[test]
    fn from_tensor_sym_works() {
        let t4 = Tensor4::from_tensor(&Samples::TENSOR4_SYM_SAMPLE1, true);
        for a in 0..6 {
            for b in 0..6 {
                let p = a + b * 6; // col-major
                assert_eq!(t4.comps_mandel[p], Samples::TENSOR4_SYM_SAMPLE1_MANDEL_MATRIX[a][b]);
            }
        }
    }

    #[test]
    fn to_tensor_4_works() {
        let t4 = Tensor4::from_tensor(&Samples::TENSOR4_SAMPLE1, false);
        let res = t4.to_tensor();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_eq!(res[i][j][k][l], Samples::TENSOR4_SAMPLE1[i][j][k][l]);
                    }
                }
            }
        }
    }

    #[test]
    fn to_tensor_symmetric_4_works() {
        let t4 = Tensor4::from_tensor(&Samples::TENSOR4_SYM_SAMPLE1, true);
        let res = t4.to_tensor();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_approx_eq!(res[i][j][k][l], Samples::TENSOR4_SYM_SAMPLE1[i][j][k][l], 1e-14);
                    }
                }
            }
        }
    }
}
