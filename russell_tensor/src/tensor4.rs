use super::{IJKL_TO_IJ, I_TO_IJ, SQRT_2};

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
    /// * dd - the standard Dijkl components given with respect to an orthonormal Cartesian basis
    /// * minor_symmetric - this is a minor-symmetric tensor
    ///
    pub fn from_tensor(dd: &[[[[f64; 3]; 3]; 3]; 3], minor_symmetric: bool) -> Result<Self, &'static str> {
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
                                    return Err("the components of minor-symmetric tensor do not pass symmetry check");
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
        Ok(Tensor4 {
            comps_mandel: dd_bar,
            minor_symmetric,
        })
    }

    /// Returns a nested array with the standard components of this fourth-order tensor
    pub fn to_tensor(&self) -> Vec<Vec<Vec<Vec<f64>>>> {
        let mut dd = vec![vec![vec![vec![0.0; 3]; 3]; 3]; 3];
        if self.minor_symmetric {
            for m in 0..6 {
                let (i, j) = I_TO_IJ[m];
                for n in 0..6 {
                    let (k, l) = I_TO_IJ[n];
                    let p = m + n * 6; // col-major
                    if i == j && k == l {
                        dd[i][j][k][l] = self.comps_mandel[p];
                    }
                    if i == j && k < l {
                        dd[i][j][k][l] = self.comps_mandel[p] / SQRT_2;
                        dd[i][j][l][k] = dd[i][j][k][l];
                    }
                    if i < j && k == l {
                        dd[i][j][k][l] = self.comps_mandel[p] / SQRT_2;
                        dd[j][i][k][l] = dd[i][j][k][l];
                    }
                    if i < j && k < l {
                        dd[i][j][k][l] = self.comps_mandel[p] / 2.0;
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
                            let (m, n) = IJKL_TO_IJ[i][j][k][l];
                            let val = self.comps_mandel[m + n * 9];
                            if i == j && k == l {
                                dd[i][j][k][l] = val;
                            }
                            if i == j && k < l {
                                let (p, q) = IJKL_TO_IJ[i][j][l][k];
                                let right = self.comps_mandel[p + q * 9];
                                dd[i][j][k][l] = (val + right) / SQRT_2;
                            }
                            if i == j && k > l {
                                let (p, q) = IJKL_TO_IJ[i][j][l][k];
                                let left = self.comps_mandel[p + q * 9];
                                dd[i][j][k][l] = (left - val) / SQRT_2;
                            }
                            if i < j && k == l {
                                let (p, q) = IJKL_TO_IJ[j][i][k][l];
                                let down = self.comps_mandel[p + q * 9];
                                dd[i][j][k][l] = (val + down) / SQRT_2;
                            }
                            if i < j && k < l {
                                let (p, q) = IJKL_TO_IJ[i][j][l][k];
                                let (r, s) = IJKL_TO_IJ[j][i][k][l];
                                let (u, v) = IJKL_TO_IJ[j][i][l][k];
                                let right = self.comps_mandel[p + q * 9];
                                let down = self.comps_mandel[r + s * 9];
                                let diag = self.comps_mandel[u + v * 9];
                                dd[i][j][k][l] = (val + right + down + diag) / 2.0;
                            }
                            if i < j && k > l {
                                let (p, q) = IJKL_TO_IJ[i][j][l][k];
                                let (r, s) = IJKL_TO_IJ[j][i][l][k];
                                let (u, v) = IJKL_TO_IJ[j][i][k][l];
                                let left = self.comps_mandel[p + q * 9];
                                let diag = self.comps_mandel[r + s * 9];
                                let down = self.comps_mandel[u + v * 9];
                                dd[i][j][k][l] = (left - val + diag - down) / 2.0;
                            }
                            if i > j && k == l {
                                let (p, q) = IJKL_TO_IJ[j][i][k][l];
                                let up = self.comps_mandel[p + q * 9];
                                dd[i][j][k][l] = (up - val) / SQRT_2;
                            }
                            if i > j && k < l {
                                let (p, q) = IJKL_TO_IJ[j][i][k][l];
                                let (r, s) = IJKL_TO_IJ[j][i][l][k];
                                let (u, v) = IJKL_TO_IJ[i][j][l][k];
                                let up = self.comps_mandel[p + q * 9];
                                let diag = self.comps_mandel[r + s * 9];
                                let right = self.comps_mandel[u + v * 9];
                                dd[i][j][k][l] = (up + diag - val - right) / 2.0;
                            }
                            if i > j && k > l {
                                let (p, q) = IJKL_TO_IJ[j][i][l][k];
                                let (r, s) = IJKL_TO_IJ[j][i][k][l];
                                let (u, v) = IJKL_TO_IJ[i][j][l][k];
                                let diag = self.comps_mandel[p + q * 9];
                                let up = self.comps_mandel[r + s * 9];
                                let left = self.comps_mandel[u + v * 9];
                                dd[i][j][k][l] = (diag - up - left + val) / 2.0;
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
    use super::Tensor4;
    use crate::Samples;
    use russell_chk::*;

    #[test]
    fn new_tensor4_works() {
        let t4 = Tensor4::new(false);
        let correct = &[0.0; 81];
        assert_vec_approx_eq!(t4.comps_mandel, correct, 1e-15);
    }

    #[test]
    fn from_tensor_works() -> Result<(), &'static str> {
        let t4 = Tensor4::from_tensor(&Samples::TENSOR4_SAMPLE1, false)?;
        for a in 0..9 {
            for b in 0..9 {
                let p = a + b * 9; // col-major
                assert_eq!(t4.comps_mandel[p], Samples::TENSOR4_SAMPLE1_MANDEL_MATRIX[a][b]);
            }
        }
        Ok(())
    }

    #[test]
    fn from_tensor_sym_fails() {
        let minor_symmetric = true; // << ERROR
        let res = Tensor4::from_tensor(&Samples::TENSOR4_SAMPLE1, minor_symmetric);
        assert_eq!(
            res.err(),
            Some("the components of minor-symmetric tensor do not pass symmetry check")
        );
    }

    #[test]
    fn from_tensor_sym_works() -> Result<(), &'static str> {
        let t4 = Tensor4::from_tensor(&Samples::TENSOR4_SYM_SAMPLE1, true)?;
        for a in 0..6 {
            for b in 0..6 {
                let p = a + b * 6; // col-major
                assert_eq!(t4.comps_mandel[p], Samples::TENSOR4_SYM_SAMPLE1_MANDEL_MATRIX[a][b]);
            }
        }
        Ok(())
    }

    #[test]
    fn to_tensor_4_works() -> Result<(), &'static str> {
        let t4 = Tensor4::from_tensor(&Samples::TENSOR4_SAMPLE1, false)?;
        let res = t4.to_tensor();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_approx_eq!(res[i][j][k][l], Samples::TENSOR4_SAMPLE1[i][j][k][l], 1e-13);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn to_tensor_symmetric_4_works() -> Result<(), &'static str> {
        let t4 = Tensor4::from_tensor(&Samples::TENSOR4_SYM_SAMPLE1, true)?;
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
        Ok(())
    }
}
