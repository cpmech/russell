use super::{mandel_dim, IJKL_TO_MN, IJKL_TO_MN_SYM, MN_TO_IJKL, SQRT_2};
use crate::StrError;
use russell_lab::Matrix;

/// Implements a fourth order-tensor, minor-symmetric or not
pub struct Tensor4 {
    /// Holds the components in Mandel basis as matrix.
    /// General: (nrow,ncol) = (9,9)
    /// Minor-symmetric in 3D: (nrow,ncol) = (6,6)
    /// Minor-symmetric in 2D: (nrow,ncol) = (4,4)
    pub mat: Matrix,
}

impl Tensor4 {
    /// Creates a new (zeroed) Tensor4
    ///
    /// The components are saved considering the Mandel basis and organized as follows:
    ///
    /// ```text
    ///      0  0   0  1   0  2    0  3   0  4   0  5    0  6   0  7   0  8
    ///    ----------------------------------------------------------------
    /// 0 | 00_00  00_11  00_22   00_01  00_12  00_02   00_10  00_21  00_20
    /// 1 | 11_00  11_11  11_22   11_01  11_12  11_02   11_10  11_21  11_20
    /// 2 | 22_00  22_11  22_22   22_01  22_12  22_02   22_10  22_21  22_20
    ///   |
    /// 3 | 01_00  01_11  01_22   01_01  01_12  01_02   01_10  01_21  01_20
    /// 4 | 12_00  12_11  12_22   12_01  12_12  12_02   12_10  12_21  12_20
    /// 5 | 02_00  02_11  02_22   02_01  02_12  02_02   02_10  02_21  02_20
    ///   |
    /// 6 | 10_00  10_11  10_22   10_01  10_12  10_02   10_10  10_21  10_20
    /// 7 | 21_00  21_11  21_22   21_01  21_12  21_02   21_10  21_21  21_20
    /// 8 | 20_00  20_11  20_22   20_01  20_12  20_02   20_10  20_21  20_20
    ///    ----------------------------------------------------------------
    ///      8  0   8  1   8  2    8  3   8  4   8  5    8  6   8  7   8  8
    /// ```
    ///
    /// # Input
    ///
    /// * `minor_symmetric` -- whether this tensor is minor symmetric or not,
    ///                        i.e., Dijkl = Djikl = Dijlk = Djilk.
    /// * `two_dim` -- 2D instead of 3D; effectively used only if minor-symmetric
    ///                since general tensors have always (9,9) components.
    pub fn new(minor_symmetric: bool, two_dim: bool) -> Self {
        let dim = mandel_dim(minor_symmetric, two_dim);
        Tensor4 {
            mat: Matrix::new(dim, dim),
        }
    }

    /// Creates a new Tensor4 constructed from a nested array
    ///
    /// # Input
    ///
    /// * `dd` - the standard (not Mandel) Dijkl components given with
    ///          respect to an orthonormal Cartesian basis
    /// * `minor_symmetric` -- whether this tensor is minor symmetric or not,
    ///                        i.e., Dijkl = Djikl = Dijlk = Djilk.
    /// * `two_dim` -- 2D instead of 3D; effectively used only if minor-symmetric
    ///                since general tensors have always (9,9) components.
    pub fn from_array(dd: &[[[[f64; 3]; 3]; 3]; 3], minor_symmetric: bool, two_dim: bool) -> Result<Self, StrError> {
        let dim = mandel_dim(minor_symmetric, two_dim);
        let mut mat = Matrix::new(dim, dim);
        if minor_symmetric {
            let max = if two_dim { 3 } else { 6 };
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
                                    return Err("minor-symmetric Tensor4 does not pass symmetry check");
                                }
                            } else {
                                let (m, n) = IJKL_TO_MN[i][j][k][l];
                                if m > max || n > max {
                                    if dd[i][j][k][l] != 0.0 {
                                        return Err("cannot define 2D Tensor4 due to non-zero values");
                                    }
                                    continue;
                                } else if m < 3 && n < 3 {
                                    mat[m][n] = dd[i][j][k][l];
                                } else if m > 2 && n > 2 {
                                    mat[m][n] = 2.0 * dd[i][j][k][l];
                                } else {
                                    mat[m][n] = SQRT_2 * dd[i][j][k][l];
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
                            let (m, n) = IJKL_TO_MN[i][j][k][l];
                            // ** i == j **
                            // 1
                            if i == j && k == l {
                                mat[m][n] = dd[i][j][k][l];
                            // 2
                            } else if i == j && k < l {
                                mat[m][n] = (dd[i][j][k][l] + dd[i][j][l][k]) / SQRT_2;
                            // 3
                            } else if i == j && k > l {
                                mat[m][n] = (dd[i][j][l][k] - dd[i][j][k][l]) / SQRT_2;
                            // ** i < j **
                            // 4
                            } else if i < j && k == l {
                                mat[m][n] = (dd[i][j][k][l] + dd[j][i][k][l]) / SQRT_2;
                            // 5
                            } else if i < j && k < l {
                                mat[m][n] = (dd[i][j][k][l] + dd[i][j][l][k] + dd[j][i][k][l] + dd[j][i][l][k]) / 2.0;
                            // 6
                            } else if i < j && k > l {
                                mat[m][n] = (dd[i][j][l][k] - dd[i][j][k][l] + dd[j][i][l][k] - dd[j][i][k][l]) / 2.0;
                            // ** i > j **
                            // 7
                            } else if i > j && k == l {
                                mat[m][n] = (dd[j][i][k][l] - dd[i][j][k][l]) / SQRT_2;
                            // 8
                            } else if i > j && k < l {
                                mat[m][n] = (dd[j][i][k][l] + dd[j][i][l][k] - dd[i][j][k][l] - dd[i][j][l][k]) / 2.0;
                            // 9
                            } else if i > j && k > l {
                                mat[m][n] = (dd[j][i][l][k] - dd[j][i][k][l] - dd[i][j][l][k] + dd[i][j][k][l]) / 2.0;
                            }
                        }
                    }
                }
            }
        }
        Ok(Tensor4 { mat })
    }

    /// Creates a new Tensor4 constructed from a matrix with standard components
    ///
    /// # Input
    ///
    /// * `std` - the standard (not Mandel) matrix of components given with
    ///           respect to an orthonormal Cartesian basis. The matrix must be (9,9),
    ///           even if it corresponds to a minor-symmetric tensor.
    pub fn from_matrix(std: &[[f64; 9]; 9], minor_symmetric: bool, two_dim: bool) -> Result<Self, StrError> {
        let dim = mandel_dim(minor_symmetric, two_dim);
        let mut mat = Matrix::new(dim, dim);
        if minor_symmetric {
            let max = if two_dim { 3 } else { 6 };
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        for l in 0..3 {
                            let (m, n) = IJKL_TO_MN[i][j][k][l];
                            let (p, q) = IJKL_TO_MN[i][j][l][k];
                            let (r, s) = IJKL_TO_MN[j][i][k][l];
                            let (u, v) = IJKL_TO_MN[j][i][l][k];
                            // check minor-symmetry
                            if i > j || k > l {
                                if std[m][n] != std[p][q] || std[m][n] != std[r][s] || std[m][n] != std[u][v] {
                                    return Err("minor-symmetric Tensor4 does not pass symmetry check");
                                }
                            } else {
                                if m > max || n > max {
                                    if std[m][n] != 0.0 {
                                        return Err("cannot define 2D Tensor4 due to non-zero values");
                                    }
                                    continue;
                                } else if m < 3 && n < 3 {
                                    mat[m][n] = std[m][n];
                                } else if m > 2 && n > 2 {
                                    mat[m][n] = 2.0 * std[m][n];
                                } else {
                                    mat[m][n] = SQRT_2 * std[m][n];
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
                            let (m, n) = IJKL_TO_MN[i][j][k][l];
                            // ** i == j **
                            // 1
                            if i == j && k == l {
                                mat[m][n] = std[m][n];
                            // 2
                            } else if i == j && k < l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                mat[m][n] = (std[m][n] + std[p][q]) / SQRT_2;
                            // 3
                            } else if i == j && k > l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                mat[m][n] = (std[p][q] - std[m][n]) / SQRT_2;
                            // ** i < j **
                            // 4
                            } else if i < j && k == l {
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                mat[m][n] = (std[m][n] + std[r][s]) / SQRT_2;
                            // 5
                            } else if i < j && k < l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                mat[m][n] = (std[m][n] + std[p][q] + std[r][s] + std[u][v]) / 2.0;
                            // 6
                            } else if i < j && k > l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                mat[m][n] = (std[p][q] - std[m][n] + std[u][v] - std[r][s]) / 2.0;
                            // ** i > j **
                            // 7
                            } else if i > j && k == l {
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                mat[m][n] = (std[r][s] - std[m][n]) / SQRT_2;
                            // 8
                            } else if i > j && k < l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                mat[m][n] = (std[r][s] + std[u][v] - std[m][n] - std[p][q]) / 2.0;
                            // 9
                            } else if i > j && k > l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                mat[m][n] = (std[u][v] - std[r][s] - std[p][q] + std[m][n]) / 2.0;
                            }
                        }
                    }
                }
            }
        }
        Ok(Tensor4 { mat })
    }

    /// Returns the (i,j,k,l) component (standard; not Mandel)
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        match self.mat.dims().0 {
            4 => {
                let (m, n) = IJKL_TO_MN_SYM[i][j][k][l];
                if m > 3 || n > 3 {
                    0.0
                } else if m < 3 && n < 3 {
                    self.mat[m][n]
                } else if m > 2 && n > 2 {
                    self.mat[m][n] / 2.0
                } else {
                    self.mat[m][n] / SQRT_2
                }
            }
            6 => {
                let (m, n) = IJKL_TO_MN_SYM[i][j][k][l];
                if m < 3 && n < 3 {
                    self.mat[m][n]
                } else if m > 2 && n > 2 {
                    self.mat[m][n] / 2.0
                } else {
                    self.mat[m][n] / SQRT_2
                }
            }
            _ => {
                let (m, n) = IJKL_TO_MN[i][j][k][l];
                let val = self.mat[m][n];
                // ** i == j **
                // 1
                if i == j && k == l {
                    val
                // 2
                } else if i == j && k < l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let right = self.mat[p][q];
                    (val + right) / SQRT_2
                // 3
                } else if i == j && k > l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let left = self.mat[p][q];
                    (left - val) / SQRT_2
                // ** i < j **
                // 4
                } else if i < j && k == l {
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let down = self.mat[r][s];
                    (val + down) / SQRT_2
                // 5
                } else if i < j && k < l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let right = self.mat[p][q];
                    let down = self.mat[r][s];
                    let diag = self.mat[u][v];
                    (val + right + down + diag) / 2.0
                // 6
                } else if i < j && k > l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let left = self.mat[p][q];
                    let diag = self.mat[u][v];
                    let down = self.mat[r][s];
                    (left - val + diag - down) / 2.0
                // ** i > j **
                // 7
                } else if i > j && k == l {
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let up = self.mat[r][s];
                    (up - val) / SQRT_2
                // 8
                } else if i > j && k < l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let up = self.mat[r][s];
                    let diag = self.mat[u][v];
                    let right = self.mat[p][q];
                    (up + diag - val - right) / 2.0
                // 9: i > j && k > l
                } else {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let diag = self.mat[u][v];
                    let up = self.mat[r][s];
                    let left = self.mat[p][q];
                    (diag - up - left + val) / 2.0
                }
            }
        }
    }

    /// Returns a nested array (standard components; not Mandel) representing this tensor
    pub fn to_array(&self) -> Vec<Vec<Vec<Vec<f64>>>> {
        let mut dd = vec![vec![vec![vec![0.0; 3]; 3]; 3]; 3];
        let dim = self.mat.dims().0;
        if dim < 9 {
            for m in 0..dim {
                for n in 0..dim {
                    let (i, j, k, l) = MN_TO_IJKL[m][n];
                    dd[i][j][k][l] = self.get(i, j, k, l);
                    if i != j || k != l {
                        dd[j][i][k][l] = dd[i][j][k][l];
                        dd[i][j][l][k] = dd[i][j][k][l];
                        dd[j][i][l][k] = dd[i][j][k][l];
                    }
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        for l in 0..3 {
                            dd[i][j][k][l] = self.get(i, j, k, l);
                        }
                    }
                }
            }
        }
        dd
    }

    /// Returns a matrix (standard components; not Mandel) representing this tensor
    pub fn to_matrix(&self) -> Matrix {
        let mut res = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                res[m][n] = self.get(i, j, k, l);
            }
        }
        res
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Tensor4;
    use crate::{Samples, StrError};
    use russell_chk::{assert_approx_eq, assert_vec_approx_eq};

    #[test]
    fn new_tensor4_works() {
        let dd = Tensor4::new(false, false);
        let correct = &[0.0; 81];
        assert_vec_approx_eq!(dd.mat.as_data(), correct, 1e-15);
    }

    #[test]
    fn from_array_fails_on_wrong_input() {
        let res = Tensor4::from_array(&Samples::TENSOR4_SAMPLE1, true, false);
        assert_eq!(res.err(), Some("minor-symmetric Tensor4 does not pass symmetry check"));

        let res = Tensor4::from_array(&Samples::TENSOR4_SYM_SAMPLE1, true, true);
        assert_eq!(res.err(), Some("cannot define 2D Tensor4 due to non-zero values"));
    }

    #[test]
    fn from_array_works() -> Result<(), StrError> {
        // general
        let dd = Tensor4::from_array(&Samples::TENSOR4_SAMPLE1, false, false)?;
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(dd.mat[m][n], Samples::TENSOR4_SAMPLE1_MANDEL_MATRIX[m][n]);
            }
        }

        // sym-3D
        let dd = Tensor4::from_array(&Samples::TENSOR4_SYM_SAMPLE1, true, false)?;
        for m in 0..6 {
            for n in 0..6 {
                assert_eq!(dd.mat[m][n], Samples::TENSOR4_SYM_SAMPLE1_MANDEL_MATRIX[m][n]);
            }
        }

        // sym-2D
        let dd = Tensor4::from_array(&Samples::TENSOR4_SYM_2D_SAMPLE1, true, true)?;
        for m in 0..4 {
            for n in 0..4 {
                assert_eq!(dd.mat[m][n], Samples::TENSOR4_SYM_2D_SAMPLE1_MANDEL_MATRIX[m][n]);
            }
        }
        Ok(())
    }

    #[test]
    fn from_matrix_fails_on_wrong_input() -> Result<(), StrError> {
        let mut inp = [[0.0; 9]; 9];
        inp[0][3] = 1e-15;
        let res = Tensor4::from_matrix(&inp, true, false);
        assert_eq!(res.err(), Some("minor-symmetric Tensor4 does not pass symmetry check"));

        inp[0][3] = 0.0;
        inp[0][4] = 1.0;
        inp[0][7] = 1.0;
        let res = Tensor4::from_matrix(&inp, true, true);
        assert_eq!(res.err(), Some("cannot define 2D Tensor4 due to non-zero values"));
        Ok(())
    }

    #[test]
    fn from_matrix_works() -> Result<(), StrError> {
        // general
        let dd = Tensor4::from_matrix(&Samples::TENSOR4_SAMPLE1_STD_MATRIX, false, false)?;
        for m in 0..9 {
            for n in 0..9 {
                assert_approx_eq!(dd.mat[m][n], Samples::TENSOR4_SAMPLE1_MANDEL_MATRIX[m][n], 1e-15);
            }
        }

        // sym-3D
        let dd = Tensor4::from_matrix(&Samples::TENSOR4_SYM_SAMPLE1_STD_MATRIX, true, false)?;
        for m in 0..6 {
            for n in 0..6 {
                assert_approx_eq!(dd.mat[m][n], Samples::TENSOR4_SYM_SAMPLE1_MANDEL_MATRIX[m][n], 1e-14);
            }
        }

        // sym-2D
        let dd = Tensor4::from_matrix(&Samples::TENSOR4_SYM_2D_SAMPLE1_STD_MATRIX, true, true)?;
        for m in 0..4 {
            for n in 0..4 {
                assert_approx_eq!(dd.mat[m][n], Samples::TENSOR4_SYM_2D_SAMPLE1_MANDEL_MATRIX[m][n], 1e-14);
            }
        }
        Ok(())
    }

    #[test]
    fn get_works() -> Result<(), StrError> {
        // general
        let dd = Tensor4::from_array(&Samples::TENSOR4_SAMPLE1, false, false)?;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_approx_eq!(dd.get(i, j, k, l), Samples::TENSOR4_SAMPLE1[i][j][k][l], 1e-13);
                    }
                }
            }
        }

        // sym-3D
        let dd = Tensor4::from_array(&Samples::TENSOR4_SYM_SAMPLE1, true, false)?;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_approx_eq!(dd.get(i, j, k, l), Samples::TENSOR4_SYM_SAMPLE1[i][j][k][l], 1e-14);
                    }
                }
            }
        }

        // sym-2D
        let dd = Tensor4::from_array(&Samples::TENSOR4_SYM_2D_SAMPLE1, true, true)?;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_approx_eq!(dd.get(i, j, k, l), Samples::TENSOR4_SYM_2D_SAMPLE1[i][j][k][l], 1e-14);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn to_array_works() -> Result<(), StrError> {
        // general
        let dd = Tensor4::from_array(&Samples::TENSOR4_SAMPLE1, false, false)?;
        let res = dd.to_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_approx_eq!(res[i][j][k][l], Samples::TENSOR4_SAMPLE1[i][j][k][l], 1e-13);
                    }
                }
            }
        }

        // sym-3D
        let dd = Tensor4::from_array(&Samples::TENSOR4_SYM_SAMPLE1, true, false)?;
        let res = dd.to_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_approx_eq!(res[i][j][k][l], Samples::TENSOR4_SYM_SAMPLE1[i][j][k][l], 1e-14);
                    }
                }
            }
        }

        // sym-2D
        let dd = Tensor4::from_array(&Samples::TENSOR4_SYM_2D_SAMPLE1, true, true)?;
        let res = dd.to_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_approx_eq!(res[i][j][k][l], Samples::TENSOR4_SYM_2D_SAMPLE1[i][j][k][l], 1e-14);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn to_matrix_works() -> Result<(), StrError> {
        // general
        let dd = Tensor4::from_array(&Samples::TENSOR4_SAMPLE1, false, false)?;
        let mat = dd.to_matrix();
        for m in 0..9 {
            for n in 0..9 {
                assert_approx_eq!(mat[m][n], &Samples::TENSOR4_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // sym-3D
        let dd = Tensor4::from_array(&Samples::TENSOR4_SYM_SAMPLE1, true, false)?;
        let mat = dd.to_matrix();
        assert_eq!(mat.dims(), (9, 9));
        for m in 0..9 {
            for n in 0..9 {
                assert_approx_eq!(mat[m][n], &Samples::TENSOR4_SYM_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // sym-2D
        let dd = Tensor4::from_array(&Samples::TENSOR4_SYM_2D_SAMPLE1, true, true)?;
        let mat = dd.to_matrix();
        assert_eq!(mat.dims(), (9, 9));
        for m in 0..9 {
            for n in 0..9 {
                assert_approx_eq!(mat[m][n], &Samples::TENSOR4_SYM_2D_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }
        Ok(())
    }
}
