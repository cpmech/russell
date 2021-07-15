use super::*;

/// Implements a fourth order tensor
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
                                let a = IJKL_TO_I[i][j][k][l];
                                let b = IJKL_TO_J[i][j][k][l];
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
                            let a = IJKL_TO_I[i][j][k][l];
                            let b = IJKL_TO_J[i][j][k][l];
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
}

///////////////////////////////////////////////////////////////////////////////

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
}
