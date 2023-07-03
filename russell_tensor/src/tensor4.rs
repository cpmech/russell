use super::{IJKL_TO_MN, IJKL_TO_MN_SYM, MN_TO_IJKL, SQRT_2};
use crate::{Mandel, StrError, ONE_BY_3, TWO_BY_3};
use russell_lab::Matrix;
use serde::{Deserialize, Serialize};

/// Implements a fourth order-tensor, minor-symmetric or not
///
/// Internally, the components are converted to the Mandel basis. On the Mandel basis,
/// depending on the symmetry, we may store fewer components. Also, we may store
/// only 16 components of Minor-Symmetric 2D tensors.
///
/// First, we consider the following mapping to the Mandel space:
///
/// ```text
/// i=j & k=l:  Mijkl := Dijkl
/// i=j & k<l:  Mijkl := (Dijkl + Dijlk) / √2
/// i=j & k>l:  Mijkl := (Dijlk − Dijkl) / √2
///
/// i<j & k=l:  Mijkl := (Dijkl + Djikl) / √2
/// i<j & k<l:  Mijkl := (Dijkl + Dijlk + Djikl + Djilk) / 2
/// i<j & k>l:  Mijkl := (Dijlk − Dijkl + Djilk − Djikl) / 2
///
/// i>j & k=l:  Mijkl := (Djikl − Dijkl) / √2
/// i>j & k<l:  Mijkl := (Djikl + Djilk − Dijkl − Dijlk) / 2
/// i>j & k>l:  Mijkl := (Djilk − Djikl − Dijlk + Dijkl) / 2
/// ```
///
/// **General case:**
///
/// Then, the 81 Mijkl components of a Tensor4 are organized as follows:
///
/// ```text
///      0 0    0 1    0 2     0 3    0 4    0 5     0 6    0 7    0 8
///    ----------------------------------------------------------------
/// 0 │ M0000  M0011  M0022   M0001  M0012  M0002   M0010  M0021  M0020
/// 1 │ M1100  M1111  M1122   M1101  M1112  M1102   M1110  M1121  M1120
/// 2 │ M2200  M2211  M2222   M2201  M2212  M2202   M2210  M2221  M2220
///   │
/// 3 │ M0100  M0111  M0122   M0101  M0112  M0102   M0110  M0121  M0120
/// 4 │ M1200  M1211  M1222   M1201  M1212  M1202   M1210  M1221  M1220
/// 5 │ M0200  M0211  M0222   M0201  M0212  M0202   M0210  M0221  M0220
///   │
/// 6 │ M1000  M1011  M1022   M1001  M1012  M1002   M1010  M1021  M1020
/// 7 │ M2100  M2111  M2122   M2101  M2112  M2102   M2110  M2121  M2120
/// 8 │ M2000  M2011  M2022   M2001  M2012  M2002   M2010  M2021  M2020
///    ----------------------------------------------------------------
///      8 0    8 1    8 2     8 3    8 4    8 5     8 6    8 7    8 8
/// ```
///
/// Note that the order of row indices (pairs (i,j) in (i,j,k,l)) follow
/// the same order as the one for Tensor2. Likewise, the order of column
/// indices (pairs (k,l) in (i,j,k,l)) follow the same order as for Tensor2.
///
/// **Minor-symmetric 3D:**
///
/// If the tensor has Dijkl = Djikl = Dijlk = Djilk, the mapping simplifies to:
///
/// ```text
/// i=j & k=l:  Mijkl := Dijkl
/// i=j & k<l:  Mijkl := Dijkl * √2
/// i=j & k>l:  Mijkl := 0
///
/// i<j & k=l:  Mijkl := Dijkl * √2
/// i<j & k<l:  Mijkl := Dijkl * 2
/// i<j & k>l:  Mijkl := 0
///
/// i>j & k=l:  Mijkl := 0
/// i>j & k<l:  Mijkl := 0
/// i>j & k>l:  Mijkl := 0
/// ```
///
/// Then, we only need to store 36 components as follows:
///
/// ```text
///      0 0       0 1       0 2        0 3       0 4       0 5
///    ------------------------------------------------------------
/// 0 │ D0000     D0011     D0022      D0001*√2  D0012*√2  D0002*√2
/// 1 │ D1100     D1111     D1122      D1101*√2  D1112*√2  D1102*√2
/// 2 │ D2200     D2211     D2222      D2201*√2  D2212*√2  D2202*√2
///   │
/// 3 │ D0100*√2  D0111*√2  D0122*√2   D0101*2   D0112*2   D0102*2
/// 4 │ D1200*√2  D1211*√2  D1222*√2   D1201*2   D1212*2   D1202*2
/// 5 │ D0200*√2  D0211*√2  D0222*√2   D0201*2   D0212*2   D0202*2
///    ------------------------------------------------------------
///      5 0       5 1       5 2        5 3       5 4       5 5
/// ```
///
/// **Minor-symmetric 2D:**
///
/// In 2D, some components are zero, thus we may store only 16 components:
///
/// ```text
///      0 0       0 1       0 2        0 3    
///    ----------------------------------------
/// 0 │ D0000     D0011     D0022      D0001*√2
/// 1 │ D1100     D1111     D1122      D1101*√2
/// 2 │ D2200     D2211     D2222      D2201*√2
///   │
/// 3 │ D0100*√2  D0111*√2  D0122*√2   D0101*2
///    ----------------------------------------
///      3 0       3 1       3 2        3 3    
/// ```
///
/// # Notes
///
/// * The tensor is represented as a (9D x 9D), (6D x 6D) or (4D x 4D) matrix and saved as `mat`
/// * You may perform operations on `mat` directly because it is isomorphic with the tensor itself
/// * For example, the norm of the tensor equals `mat.norm()`
/// * However, you must be careful when setting a single component of `mat` directly
///   because you may "break" the Mandel representation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tensor4 {
    /// Holds the components in Mandel basis as matrix.
    ///
    /// * General: `(nrow,ncol) = (9,9)`
    /// * Minor-symmetric in 3D: `(nrow,ncol) = (6,6)`
    /// * Minor-symmetric in 2D: `(nrow,ncol) = (4,4)`
    pub mat: Matrix,
}

impl Tensor4 {
    /// Creates a new (zeroed) Tensor4
    ///
    /// # Input
    ///
    /// * `case` -- the [Mandel] case
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, StrError, Tensor4};
    ///
    /// fn main() {
    ///     let cc = Tensor4::new(Mandel::General);
    ///     assert_eq!(cc.mat.dims(), (9,9));
    ///
    ///     let dd = Tensor4::new(Mandel::Symmetric);
    ///     assert_eq!(dd.mat.dims(), (6,6));
    ///
    ///     let ee = Tensor4::new(Mandel::Symmetric2D);
    ///     assert_eq!(ee.mat.dims(), (4,4));
    /// }
    /// ```
    pub fn new(case: Mandel) -> Self {
        let dim = case.dim();
        Tensor4 {
            mat: Matrix::new(dim, dim),
        }
    }

    /// Returns the Mandel case associated with this Tensor4
    #[inline]
    pub fn case(&self) -> Mandel {
        Mandel::new(self.mat.nrow())
    }

    /// Creates a new Tensor4 constructed from a nested array
    ///
    /// # Input
    ///
    /// * `inp` -- the standard (not Mandel) Dijkl components given with
    ///   respect to an orthonormal Cartesian basis
    /// * `case` -- the [Mandel] case
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[[[0.0; 3]; 3]; 3]; 3];
    ///     for i in 0..3 {
    ///         for j in 0..3 {
    ///             for k in 0..3 {
    ///                 for l in 0..3 {
    ///                     inp[i][j][k][l] = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///                 }
    ///             }
    ///         }
    ///     }
    ///     let dd = Tensor4::from_array(&inp, Mandel::General)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.to_matrix()),
    ///         "┌                                              ┐\n\
    ///          │ 1111 1122 1133 1112 1123 1113 1121 1132 1131 │\n\
    ///          │ 2211 2222 2233 2212 2223 2213 2221 2232 2231 │\n\
    ///          │ 3311 3322 3333 3312 3323 3313 3321 3332 3331 │\n\
    ///          │ 1211 1222 1233 1212 1223 1213 1221 1232 1231 │\n\
    ///          │ 2311 2322 2333 2312 2323 2313 2321 2332 2331 │\n\
    ///          │ 1311 1322 1333 1312 1323 1313 1321 1332 1331 │\n\
    ///          │ 2111 2122 2133 2112 2123 2113 2121 2132 2131 │\n\
    ///          │ 3211 3222 3233 3212 3223 3213 3221 3232 3231 │\n\
    ///          │ 3111 3122 3133 3112 3123 3113 3121 3132 3131 │\n\
    ///          └                                              ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn from_array(inp: &[[[[f64; 3]; 3]; 3]; 3], case: Mandel) -> Result<Self, StrError> {
        let dim = case.dim();
        let mut mat = Matrix::new(dim, dim);
        if dim == 4 || dim == 6 {
            let max = if dim == 4 { 3 } else { 6 };
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        for l in 0..3 {
                            // check minor-symmetry
                            if i > j || k > l {
                                if inp[i][j][k][l] != inp[j][i][k][l]
                                    || inp[i][j][k][l] != inp[i][j][l][k]
                                    || inp[i][j][k][l] != inp[j][i][l][k]
                                {
                                    return Err("minor-symmetric Tensor4 does not pass symmetry check");
                                }
                            } else {
                                let (m, n) = IJKL_TO_MN[i][j][k][l];
                                if m > max || n > max {
                                    if inp[i][j][k][l] != 0.0 {
                                        return Err("cannot define 2D Tensor4 due to non-zero values");
                                    }
                                    continue;
                                } else if m < 3 && n < 3 {
                                    mat.set(m, n, inp[i][j][k][l]);
                                } else if m > 2 && n > 2 {
                                    mat.set(m, n, 2.0 * inp[i][j][k][l]);
                                } else {
                                    mat.set(m, n, SQRT_2 * inp[i][j][k][l]);
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
                                mat.set(m, n, inp[i][j][k][l]);
                            // 2
                            } else if i == j && k < l {
                                mat.set(m, n, (inp[i][j][k][l] + inp[i][j][l][k]) / SQRT_2);
                            // 3
                            } else if i == j && k > l {
                                mat.set(m, n, (inp[i][j][l][k] - inp[i][j][k][l]) / SQRT_2);
                            // ** i < j **
                            // 4
                            } else if i < j && k == l {
                                mat.set(m, n, (inp[i][j][k][l] + inp[j][i][k][l]) / SQRT_2);
                            // 5
                            } else if i < j && k < l {
                                mat.set(
                                    m,
                                    n,
                                    (inp[i][j][k][l] + inp[i][j][l][k] + inp[j][i][k][l] + inp[j][i][l][k]) / 2.0,
                                );
                            // 6
                            } else if i < j && k > l {
                                mat.set(
                                    m,
                                    n,
                                    (inp[i][j][l][k] - inp[i][j][k][l] + inp[j][i][l][k] - inp[j][i][k][l]) / 2.0,
                                );
                            // ** i > j **
                            // 7
                            } else if i > j && k == l {
                                mat.set(m, n, (inp[j][i][k][l] - inp[i][j][k][l]) / SQRT_2);
                            // 8
                            } else if i > j && k < l {
                                mat.set(
                                    m,
                                    n,
                                    (inp[j][i][k][l] + inp[j][i][l][k] - inp[i][j][k][l] - inp[i][j][l][k]) / 2.0,
                                );
                            // 9
                            } else if i > j && k > l {
                                mat.set(
                                    m,
                                    n,
                                    (inp[j][i][l][k] - inp[j][i][k][l] - inp[i][j][l][k] + inp[i][j][k][l]) / 2.0,
                                );
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
    /// * `inp` -- the standard (not Mandel) matrix of components given with
    ///   respect to an orthonormal Cartesian basis. The matrix must be (9,9),
    ///   even if it corresponds to a minor-symmetric tensor.
    /// * `case` -- the [Mandel] case
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, MN_TO_IJKL, Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 9]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..9 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             inp[m][n] = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor4::from_matrix(&inp, Mandel::General)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.to_matrix()),
    ///         "┌                                              ┐\n\
    ///          │ 1111 1122 1133 1112 1123 1113 1121 1132 1131 │\n\
    ///          │ 2211 2222 2233 2212 2223 2213 2221 2232 2231 │\n\
    ///          │ 3311 3322 3333 3312 3323 3313 3321 3332 3331 │\n\
    ///          │ 1211 1222 1233 1212 1223 1213 1221 1232 1231 │\n\
    ///          │ 2311 2322 2333 2312 2323 2313 2321 2332 2331 │\n\
    ///          │ 1311 1322 1333 1312 1323 1313 1321 1332 1331 │\n\
    ///          │ 2111 2122 2133 2112 2123 2113 2121 2132 2131 │\n\
    ///          │ 3211 3222 3233 3212 3223 3213 3221 3232 3231 │\n\
    ///          │ 3111 3122 3133 3112 3123 3113 3121 3132 3131 │\n\
    ///          └                                              ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn from_matrix(inp: &[[f64; 9]; 9], case: Mandel) -> Result<Self, StrError> {
        let dim = case.dim();
        let mut mat = Matrix::new(dim, dim);
        if dim == 4 || dim == 6 {
            let max = if dim == 4 { 3 } else { 6 };
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
                                if inp[m][n] != inp[p][q] || inp[m][n] != inp[r][s] || inp[m][n] != inp[u][v] {
                                    return Err("minor-symmetric Tensor4 does not pass symmetry check");
                                }
                            } else {
                                if m > max || n > max {
                                    if inp[m][n] != 0.0 {
                                        return Err("cannot define 2D Tensor4 due to non-zero values");
                                    }
                                    continue;
                                } else if m < 3 && n < 3 {
                                    mat.set(m, n, inp[m][n]);
                                } else if m > 2 && n > 2 {
                                    mat.set(m, n, 2.0 * inp[m][n]);
                                } else {
                                    mat.set(m, n, SQRT_2 * inp[m][n]);
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
                                mat.set(m, n, inp[m][n]);
                            // 2
                            } else if i == j && k < l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                mat.set(m, n, (inp[m][n] + inp[p][q]) / SQRT_2);
                            // 3
                            } else if i == j && k > l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                mat.set(m, n, (inp[p][q] - inp[m][n]) / SQRT_2);
                            // ** i < j **
                            // 4
                            } else if i < j && k == l {
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                mat.set(m, n, (inp[m][n] + inp[r][s]) / SQRT_2);
                            // 5
                            } else if i < j && k < l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                mat.set(m, n, (inp[m][n] + inp[p][q] + inp[r][s] + inp[u][v]) / 2.0);
                            // 6
                            } else if i < j && k > l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                mat.set(m, n, (inp[p][q] - inp[m][n] + inp[u][v] - inp[r][s]) / 2.0);
                            // ** i > j **
                            // 7
                            } else if i > j && k == l {
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                mat.set(m, n, (inp[r][s] - inp[m][n]) / SQRT_2);
                            // 8
                            } else if i > j && k < l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                mat.set(m, n, (inp[r][s] + inp[u][v] - inp[m][n] - inp[p][q]) / 2.0);
                            // 9
                            } else if i > j && k > l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                mat.set(m, n, (inp[u][v] - inp[r][s] - inp[p][q] + inp[m][n]) / 2.0);
                            }
                        }
                    }
                }
            }
        }
        Ok(Tensor4 { mat })
    }

    /// Returns the (i,j,k,l) component (standard; not Mandel)
    ///
    /// # Example
    ///
    /// ```
    /// use russell_chk::approx_eq;
    /// use russell_tensor::{Mandel, MN_TO_IJKL, Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 9]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..9 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             inp[m][n] = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor4::from_matrix(&inp, Mandel::General)?;
    ///
    ///     for m in 0..9 {
    ///         for n in 0..9 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             let val = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///             approx_eq(dd.get(i,j,k,l), val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        match self.mat.dims().0 {
            4 => {
                let (m, n) = IJKL_TO_MN_SYM[i][j][k][l];
                if m > 3 || n > 3 {
                    0.0
                } else if m < 3 && n < 3 {
                    self.mat.get(m, n)
                } else if m > 2 && n > 2 {
                    self.mat.get(m, n) / 2.0
                } else {
                    self.mat.get(m, n) / SQRT_2
                }
            }
            6 => {
                let (m, n) = IJKL_TO_MN_SYM[i][j][k][l];
                if m < 3 && n < 3 {
                    self.mat.get(m, n)
                } else if m > 2 && n > 2 {
                    self.mat.get(m, n) / 2.0
                } else {
                    self.mat.get(m, n) / SQRT_2
                }
            }
            _ => {
                let (m, n) = IJKL_TO_MN[i][j][k][l];
                let val = self.mat.get(m, n);
                // ** i == j **
                // 1
                if i == j && k == l {
                    val
                // 2
                } else if i == j && k < l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let right = self.mat.get(p, q);
                    (val + right) / SQRT_2
                // 3
                } else if i == j && k > l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let left = self.mat.get(p, q);
                    (left - val) / SQRT_2
                // ** i < j **
                // 4
                } else if i < j && k == l {
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let down = self.mat.get(r, s);
                    (val + down) / SQRT_2
                // 5
                } else if i < j && k < l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let right = self.mat.get(p, q);
                    let down = self.mat.get(r, s);
                    let diag = self.mat.get(u, v);
                    (val + right + down + diag) / 2.0
                // 6
                } else if i < j && k > l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let left = self.mat.get(p, q);
                    let diag = self.mat.get(u, v);
                    let down = self.mat.get(r, s);
                    (left - val + diag - down) / 2.0
                // ** i > j **
                // 7
                } else if i > j && k == l {
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let up = self.mat.get(r, s);
                    (up - val) / SQRT_2
                // 8
                } else if i > j && k < l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let up = self.mat.get(r, s);
                    let diag = self.mat.get(u, v);
                    let right = self.mat.get(p, q);
                    (up + diag - val - right) / 2.0
                // 9: i > j && k > l
                } else {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let diag = self.mat.get(u, v);
                    let up = self.mat.get(r, s);
                    let left = self.mat.get(p, q);
                    (diag - up - left + val) / 2.0
                }
            }
        }
    }

    /// Returns a nested array (standard components; not Mandel) representing this tensor
    ///
    /// # Example
    ///
    /// ```
    /// use russell_chk::approx_eq;
    /// use russell_tensor::{Mandel, MN_TO_IJKL, Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 9]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..9 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             inp[m][n] = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor4::from_matrix(&inp, Mandel::General)?;
    ///     let arr = dd.to_array();
    ///
    ///     for m in 0..9 {
    ///         for n in 0..9 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             let val = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///             approx_eq(arr[i][j][k][l], val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, MN_TO_IJKL, Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 9]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..9 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             inp[m][n] = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor4::from_matrix(&inp, Mandel::General)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.to_matrix()),
    ///         "┌                                              ┐\n\
    ///          │ 1111 1122 1133 1112 1123 1113 1121 1132 1131 │\n\
    ///          │ 2211 2222 2233 2212 2223 2213 2221 2232 2231 │\n\
    ///          │ 3311 3322 3333 3312 3323 3313 3321 3332 3331 │\n\
    ///          │ 1211 1222 1233 1212 1223 1213 1221 1232 1231 │\n\
    ///          │ 2311 2322 2333 2312 2323 2313 2321 2332 2331 │\n\
    ///          │ 1311 1322 1333 1312 1323 1313 1321 1332 1331 │\n\
    ///          │ 2111 2122 2133 2112 2123 2113 2121 2132 2131 │\n\
    ///          │ 3211 3222 3233 3212 3223 3213 3221 3232 3231 │\n\
    ///          │ 3111 3122 3133 3112 3123 3113 3121 3132 3131 │\n\
    ///          └                                              ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn to_matrix(&self) -> Matrix {
        let mut res = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                res.set(m, n, self.get(i, j, k, l));
            }
        }
        res
    }

    /// Sets the (i,j,k,l) component of a minor-symmetric Tensor4
    ///
    /// # Panics
    ///
    /// The tensor must be symmetric and (i,j) must correspond to the possible
    /// combination due to the space dimension, otherwise a panic may occur.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_tensor::{Mandel, MN_TO_IJKL, Tensor4};
    ///
    /// fn main() {
    ///     let mut dd = Tensor4::new(Mandel::Symmetric2D);
    ///     for m in 0..4 {
    ///         for n in 0..4 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             let value = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///             dd.sym_set(i, j, k, l, value);
    ///         }
    ///     }
    ///     assert_eq!(
    ///         format!("{:.0}", dd.to_matrix()),
    ///         "┌                                              ┐\n\
    ///          │ 1111 1122 1133 1112    0    0 1112    0    0 │\n\
    ///          │ 2211 2222 2233 2212    0    0 2212    0    0 │\n\
    ///          │ 3311 3322 3333 3312    0    0 3312    0    0 │\n\
    ///          │ 1211 1222 1233 1212    0    0 1212    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │ 1211 1222 1233 1212    0    0 1212    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          └                                              ┘"
    ///     );
    /// }
    /// ```
    pub fn sym_set(&mut self, i: usize, j: usize, k: usize, l: usize, value: f64) {
        let (m, n) = IJKL_TO_MN_SYM[i][j][k][l];
        if m < 3 && n < 3 {
            self.mat.set(m, n, value);
        } else if m > 2 && n > 2 {
            self.mat.set(m, n, value * 2.0);
        } else {
            self.mat.set(m, n, value * SQRT_2);
        }
    }

    /// Returns the fourth-order identity tensor I
    ///
    /// ```text
    /// Definition:
    ///        _
    /// II = I ⊗ I
    /// ```
    ///
    /// ```text
    /// Mandel matrix:
    ///        ┌                     ┐
    ///        │ 1 0 0  0 0 0  0 0 0 │
    ///        │ 0 1 0  0 0 0  0 0 0 │
    ///        │ 0 0 1  0 0 0  0 0 0 │
    ///        │ 0 0 0  1 0 0  0 0 0 │
    /// [II] = │ 0 0 0  0 1 0  0 0 0 │
    ///        │ 0 0 0  0 0 1  0 0 0 │
    ///        │ 0 0 0  0 0 0  1 0 0 │
    ///        │ 0 0 0  0 0 0  0 1 0 │
    ///        │ 0 0 0  0 0 0  0 0 1 │
    ///        └                     ┘
    /// ```
    ///
    /// Note: this tensor cannot be represented in reduced-dimension
    /// because it is not minor-symmetric.
    pub fn constant_ii() -> Self {
        Tensor4 {
            //                       1    2    3    4    5    6    7    8    9
            mat: Matrix::diagonal(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        }
    }

    /// Returns the transposition tensor T
    ///
    /// ```text
    /// Definition:
    ///
    /// TT = I ⊗ I
    ///        ‾
    /// ```
    ///
    /// ```text
    /// Mandel matrix:
    ///        ┌                        ┐
    ///        │ 1 0 0  0 0 0   0  0  0 │
    ///        │ 0 1 0  0 0 0   0  0  0 │
    ///        │ 0 0 1  0 0 0   0  0  0 │
    ///        │ 0 0 0  1 0 0   0  0  0 │
    /// [TT] = │ 0 0 0  0 1 0   0  0  0 │
    ///        │ 0 0 0  0 0 1   0  0  0 │
    ///        │ 0 0 0  0 0 0  -1  0  0 │
    ///        │ 0 0 0  0 0 0   0 -1  0 │
    ///        │ 0 0 0  0 0 0   0  0 -1 │
    ///        └                        ┘
    /// ```
    ///
    /// Note: this tensor cannot be represented in reduced-dimension
    /// because it is not minor-symmetric.
    pub fn constant_tt() -> Self {
        let mut tt = Tensor4 { mat: Matrix::new(9, 9) };
        tt.mat.set(0, 0, 1.0);
        tt.mat.set(1, 1, 1.0);
        tt.mat.set(2, 2, 1.0);
        tt.mat.set(3, 3, 1.0);
        tt.mat.set(4, 4, 1.0);
        tt.mat.set(5, 5, 1.0);
        tt.mat.set(6, 6, -1.0);
        tt.mat.set(7, 7, -1.0);
        tt.mat.set(8, 8, -1.0);
        tt
    }

    /// Returns the trace-projection tensor J
    ///
    /// ```text
    /// Definition:
    ///
    /// JJ = I ⊗ I
    /// ```
    ///
    /// ```text
    /// Mandel matrix:
    ///        ┌                     ┐
    ///        │ 1 1 1  0 0 0  0 0 0 │
    ///        │ 1 1 1  0 0 0  0 0 0 │
    ///        │ 1 1 1  0 0 0  0 0 0 │
    ///        │ 0 0 0  0 0 0  0 0 0 │
    /// [JJ] = │ 0 0 0  0 0 0  0 0 0 │
    ///        │ 0 0 0  0 0 0  0 0 0 │
    ///        │ 0 0 0  0 0 0  0 0 0 │
    ///        │ 0 0 0  0 0 0  0 0 0 │
    ///        │ 0 0 0  0 0 0  0 0 0 │
    ///        └                     ┘
    /// ```
    ///
    /// Note: this tensor can be represented in reduced-dimension.
    pub fn constant_jj(reduced_6x6: bool) -> Self {
        let n = if reduced_6x6 { 6 } else { 9 };
        let mut jj = Tensor4 { mat: Matrix::new(n, n) };
        jj.mat.set(0, 0, 1.0);
        jj.mat.set(0, 1, 1.0);
        jj.mat.set(0, 2, 1.0);
        jj.mat.set(1, 0, 1.0);
        jj.mat.set(1, 1, 1.0);
        jj.mat.set(1, 2, 1.0);
        jj.mat.set(2, 0, 1.0);
        jj.mat.set(2, 1, 1.0);
        jj.mat.set(2, 2, 1.0);
        jj
    }

    /// Returns the isotropic making projector Piso
    ///
    /// ```text
    /// Definition:
    ///
    /// Piso = ⅓ I ⊗ I = ⅓ JJ
    /// ```
    ///
    /// ```text
    /// Mandel matrix:
    ///          ┌                     ┐
    ///          │ ⅓ ⅓ ⅓  0 0 0  0 0 0 │
    ///          │ ⅓ ⅓ ⅓  0 0 0  0 0 0 │
    ///          │ ⅓ ⅓ ⅓  0 0 0  0 0 0 │
    ///          │ 0 0 0  0 0 0  0 0 0 │
    /// [Piso] = │ 0 0 0  0 0 0  0 0 0 │
    ///          │ 0 0 0  0 0 0  0 0 0 │
    ///          │ 0 0 0  0 0 0  0 0 0 │
    ///          │ 0 0 0  0 0 0  0 0 0 │
    ///          │ 0 0 0  0 0 0  0 0 0 │
    ///          └                     ┘
    /// ```
    ///
    /// Note: this tensor can be represented in reduced-dimension.
    pub fn constant_pp_iso(reduced_6x6: bool) -> Self {
        let n = if reduced_6x6 { 6 } else { 9 };
        let mut pp_iso = Tensor4 { mat: Matrix::new(n, n) };
        pp_iso.mat.set(0, 0, ONE_BY_3);
        pp_iso.mat.set(0, 1, ONE_BY_3);
        pp_iso.mat.set(0, 2, ONE_BY_3);
        pp_iso.mat.set(1, 0, ONE_BY_3);
        pp_iso.mat.set(1, 1, ONE_BY_3);
        pp_iso.mat.set(1, 2, ONE_BY_3);
        pp_iso.mat.set(2, 0, ONE_BY_3);
        pp_iso.mat.set(2, 1, ONE_BY_3);
        pp_iso.mat.set(2, 2, ONE_BY_3);
        pp_iso
    }

    /// Returns the symmetric making projector Psym
    ///
    /// ```text
    /// Definition:
    ///             _
    /// Psym = ½ (I ⊗ I + I ⊗ I) = ½ (II + TT) = ½ ssd(I)
    ///                     ‾
    /// ```
    ///
    /// ```text
    /// Mandel matrix:
    ///          ┌                     ┐
    ///          │ 1 0 0  0 0 0  0 0 0 │
    ///          │ 0 1 0  0 0 0  0 0 0 │
    ///          │ 0 0 1  0 0 0  0 0 0 │
    ///          │ 0 0 0  1 0 0  0 0 0 │
    /// [Psym] = │ 0 0 0  0 1 0  0 0 0 │
    ///          │ 0 0 0  0 0 1  0 0 0 │
    ///          │ 0 0 0  0 0 0  0 0 0 │
    ///          │ 0 0 0  0 0 0  0 0 0 │
    ///          │ 0 0 0  0 0 0  0 0 0 │
    ///          └                     ┘
    /// ```
    ///
    /// Note: this tensor can be represented in reduced-dimension.
    pub fn constant_pp_sym(reduced_6x6: bool) -> Self {
        let n = if reduced_6x6 { 6 } else { 9 };
        let mut pp_sym = Tensor4 { mat: Matrix::new(n, n) };
        pp_sym.mat.set(0, 0, 1.0);
        pp_sym.mat.set(1, 1, 1.0);
        pp_sym.mat.set(2, 2, 1.0);
        pp_sym.mat.set(3, 3, 1.0);
        pp_sym.mat.set(4, 4, 1.0);
        pp_sym.mat.set(5, 5, 1.0);
        pp_sym
    }

    /// Returns the skew making projector Pskew
    ///
    /// ```text
    /// Definition:
    ///              _
    /// Pskew = ½ (I ⊗ I - I ⊗ I) = ½ (II - TT)
    ///                      ‾
    /// ```
    ///
    /// ```text
    /// Mandel matrix:
    ///           ┌                     ┐
    ///           │ 0 0 0  0 0 0  0 0 0 │
    ///           │ 0 0 0  0 0 0  0 0 0 │
    ///           │ 0 0 0  0 0 0  0 0 0 │
    ///           │ 0 0 0  0 0 0  0 0 0 │
    /// [Pskew] = │ 0 0 0  0 0 0  0 0 0 │
    ///           │ 0 0 0  0 0 0  0 0 0 │
    ///           │ 0 0 0  0 0 0  1 0 0 │
    ///           │ 0 0 0  0 0 0  0 1 0 │
    ///           │ 0 0 0  0 0 0  0 0 1 │
    ///           └                     ┘
    /// ```
    ///
    /// Note: this tensor cannot be represented in reduced-dimension
    /// because it is not minor-symmetric.
    pub fn constant_pp_skew() -> Self {
        let mut pp_skew = Tensor4 { mat: Matrix::new(9, 9) };
        pp_skew.mat.set(6, 6, 1.0);
        pp_skew.mat.set(7, 7, 1.0);
        pp_skew.mat.set(8, 8, 1.0);
        pp_skew
    }

    /// Returns the deviatoric making projector Pdev
    ///
    /// ```text
    /// Definition:
    ///          _
    /// Pdev = I ⊗ I - ⅓ I ⊗ I = II - Piso
    /// ```
    ///
    /// ```text
    /// Mandel matrix:
    ///          ┌                        ┐
    ///          │  ⅔ -⅓ -⅓  0 0 0  0 0 0 │
    ///          │ -⅓  ⅔ -⅓  0 0 0  0 0 0 │
    ///          │ -⅓ -⅓  ⅔  0 0 0  0 0 0 │
    ///          │  0  0  0  1 0 0  0 0 0 │
    /// [Pdev] = │  0  0  0  0 1 0  0 0 0 │
    ///          │  0  0  0  0 0 1  0 0 0 │
    ///          │  0  0  0  0 0 0  1 0 0 │
    ///          │  0  0  0  0 0 0  0 1 0 │
    ///          │  0  0  0  0 0 0  0 0 1 │
    ///          └                        ┘
    /// ```
    ///
    /// Note: this tensor cannot be represented in reduced-dimension
    /// because it is not minor-symmetric.
    pub fn constant_pp_dev() -> Self {
        let mut pp_dev = Tensor4 { mat: Matrix::new(9, 9) };
        pp_dev.mat.set(0, 0, TWO_BY_3);
        pp_dev.mat.set(0, 1, -ONE_BY_3);
        pp_dev.mat.set(0, 2, -ONE_BY_3);
        pp_dev.mat.set(1, 0, -ONE_BY_3);
        pp_dev.mat.set(1, 1, TWO_BY_3);
        pp_dev.mat.set(1, 2, -ONE_BY_3);
        pp_dev.mat.set(2, 0, -ONE_BY_3);
        pp_dev.mat.set(2, 1, -ONE_BY_3);
        pp_dev.mat.set(2, 2, TWO_BY_3);
        pp_dev.mat.set(3, 3, 1.0);
        pp_dev.mat.set(4, 4, 1.0);
        pp_dev.mat.set(5, 5, 1.0);
        pp_dev.mat.set(6, 6, 1.0);
        pp_dev.mat.set(7, 7, 1.0);
        pp_dev.mat.set(8, 8, 1.0);
        pp_dev
    }

    /// Returns the symmetric-deviatoric making projector Psymdev
    ///
    /// ```text
    /// Definition:
    ///                _
    /// Psymdev = ½ (I ⊗ I + I ⊗ I) - ⅓ I ⊗ I = Psym - Piso
    ///                        ‾
    /// ```
    ///
    /// ```text
    /// Mandel matrix:
    ///             ┌                        ┐
    ///             │  ⅔ -⅓ -⅓  0 0 0  0 0 0 │
    ///             │ -⅓  ⅔ -⅓  0 0 0  0 0 0 │
    ///             │ -⅓ -⅓  ⅔  0 0 0  0 0 0 │
    ///             │  0  0  0  1 0 0  0 0 0 │
    /// [Psymdev] = │  0  0  0  0 1 0  0 0 0 │
    ///             │  0  0  0  0 0 1  0 0 0 │
    ///             │  0  0  0  0 0 0  0 0 0 │
    ///             │  0  0  0  0 0 0  0 0 0 │
    ///             │  0  0  0  0 0 0  0 0 0 │
    ///             └                        ┘
    /// ```
    ///
    /// Note: this tensor can be represented in reduced-dimension.
    pub fn constant_pp_symdev(reduced_6x6: bool) -> Self {
        let n = if reduced_6x6 { 6 } else { 9 };
        let mut pp_symdev = Tensor4 { mat: Matrix::new(n, n) };
        pp_symdev.mat.set(0, 0, TWO_BY_3);
        pp_symdev.mat.set(0, 1, -ONE_BY_3);
        pp_symdev.mat.set(0, 2, -ONE_BY_3);
        pp_symdev.mat.set(1, 0, -ONE_BY_3);
        pp_symdev.mat.set(1, 1, TWO_BY_3);
        pp_symdev.mat.set(1, 2, -ONE_BY_3);
        pp_symdev.mat.set(2, 0, -ONE_BY_3);
        pp_symdev.mat.set(2, 1, -ONE_BY_3);
        pp_symdev.mat.set(2, 2, TWO_BY_3);
        pp_symdev.mat.set(3, 3, 1.0);
        pp_symdev.mat.set(4, 4, 1.0);
        pp_symdev.mat.set(5, 5, 1.0);
        pp_symdev
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Tensor4, MN_TO_IJKL};
    use crate::{Mandel, SamplesTensor4};
    use russell_chk::approx_eq;
    use russell_lab::mat_approx_eq;
    use serde::{Deserialize, Serialize};

    #[test]
    fn new_and_case_work() {
        // general
        let dd = Tensor4::new(Mandel::General);
        assert_eq!(dd.mat.as_data().len(), 81);
        assert_eq!(dd.case(), Mandel::General);

        // symmetric
        let dd = Tensor4::new(Mandel::Symmetric);
        assert_eq!(dd.mat.as_data().len(), 36);
        assert_eq!(dd.case(), Mandel::Symmetric);

        // symmetric 2d
        let dd = Tensor4::new(Mandel::Symmetric2D);
        assert_eq!(dd.mat.as_data().len(), 16);
        assert_eq!(dd.case(), Mandel::Symmetric2D);
    }

    #[test]
    fn from_array_fails_captures_errors() {
        let res = Tensor4::from_array(&SamplesTensor4::SAMPLE1, Mandel::Symmetric);
        assert_eq!(res.err(), Some("minor-symmetric Tensor4 does not pass symmetry check"));

        let res = Tensor4::from_array(&SamplesTensor4::SYM_SAMPLE1, Mandel::Symmetric2D);
        assert_eq!(res.err(), Some("cannot define 2D Tensor4 due to non-zero values"));
    }

    #[test]
    fn from_array_works() {
        // general
        let dd = Tensor4::from_array(&SamplesTensor4::SAMPLE1, Mandel::General).unwrap();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(dd.mat.get(m, n), SamplesTensor4::SAMPLE1_MANDEL_MATRIX[m][n]);
            }
        }

        // symmetric 3d
        let dd = Tensor4::from_array(&SamplesTensor4::SYM_SAMPLE1, Mandel::Symmetric).unwrap();
        for m in 0..6 {
            for n in 0..6 {
                assert_eq!(dd.mat.get(m, n), SamplesTensor4::SYM_SAMPLE1_MANDEL_MATRIX[m][n]);
            }
        }

        // symmetric 2d
        let dd = Tensor4::from_array(&SamplesTensor4::SYM_2D_SAMPLE1, Mandel::Symmetric2D).unwrap();
        for m in 0..4 {
            for n in 0..4 {
                assert_eq!(dd.mat.get(m, n), SamplesTensor4::SYM_2D_SAMPLE1_MANDEL_MATRIX[m][n]);
            }
        }
    }

    #[test]
    fn from_matrix_fails_captures_errors() {
        let mut inp = [[0.0; 9]; 9];
        inp[0][3] = 1e-15;
        let res = Tensor4::from_matrix(&inp, Mandel::Symmetric);
        assert_eq!(res.err(), Some("minor-symmetric Tensor4 does not pass symmetry check"));

        inp[0][3] = 0.0;
        inp[0][4] = 1.0;
        inp[0][7] = 1.0;
        let res = Tensor4::from_matrix(&inp, Mandel::Symmetric2D);
        assert_eq!(res.err(), Some("cannot define 2D Tensor4 due to non-zero values"));
    }

    #[test]
    fn from_matrix_works() {
        // general
        let dd = Tensor4::from_matrix(&SamplesTensor4::SAMPLE1_STD_MATRIX, Mandel::General).unwrap();
        for m in 0..9 {
            for n in 0..9 {
                approx_eq(dd.mat.get(m, n), SamplesTensor4::SAMPLE1_MANDEL_MATRIX[m][n], 1e-15);
            }
        }

        // symmetric 3D
        let dd = Tensor4::from_matrix(&SamplesTensor4::SYM_SAMPLE1_STD_MATRIX, Mandel::Symmetric).unwrap();
        for m in 0..6 {
            for n in 0..6 {
                approx_eq(dd.mat.get(m, n), SamplesTensor4::SYM_SAMPLE1_MANDEL_MATRIX[m][n], 1e-14);
            }
        }

        // symmetric 2D
        let dd = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        for m in 0..4 {
            for n in 0..4 {
                approx_eq(
                    dd.mat.get(m, n),
                    SamplesTensor4::SYM_2D_SAMPLE1_MANDEL_MATRIX[m][n],
                    1e-14,
                );
            }
        }
    }

    #[test]
    fn get_works() {
        // general
        let dd = Tensor4::from_array(&SamplesTensor4::SAMPLE1, Mandel::General).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(dd.get(i, j, k, l), SamplesTensor4::SAMPLE1[i][j][k][l], 1e-13);
                    }
                }
            }
        }

        // symmetric 3D
        let dd = Tensor4::from_array(&SamplesTensor4::SYM_SAMPLE1, Mandel::Symmetric).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(dd.get(i, j, k, l), SamplesTensor4::SYM_SAMPLE1[i][j][k][l], 1e-14);
                    }
                }
            }
        }

        // symmetric 2D
        let dd = Tensor4::from_array(&SamplesTensor4::SYM_2D_SAMPLE1, Mandel::Symmetric2D).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(dd.get(i, j, k, l), SamplesTensor4::SYM_2D_SAMPLE1[i][j][k][l], 1e-14);
                    }
                }
            }
        }
    }

    #[test]
    fn to_array_works() {
        // general
        let dd = Tensor4::from_array(&SamplesTensor4::SAMPLE1, Mandel::General).unwrap();
        let res = dd.to_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(res[i][j][k][l], SamplesTensor4::SAMPLE1[i][j][k][l], 1e-13);
                    }
                }
            }
        }

        // symmetric 3D
        let dd = Tensor4::from_array(&SamplesTensor4::SYM_SAMPLE1, Mandel::Symmetric).unwrap();
        let res = dd.to_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(res[i][j][k][l], SamplesTensor4::SYM_SAMPLE1[i][j][k][l], 1e-14);
                    }
                }
            }
        }

        // symmetric 2D
        let dd = Tensor4::from_array(&SamplesTensor4::SYM_2D_SAMPLE1, Mandel::Symmetric2D).unwrap();
        let res = dd.to_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(res[i][j][k][l], SamplesTensor4::SYM_2D_SAMPLE1[i][j][k][l], 1e-14);
                    }
                }
            }
        }
    }

    #[test]
    fn to_matrix_works() {
        // general
        let dd = Tensor4::from_array(&SamplesTensor4::SAMPLE1, Mandel::General).unwrap();
        let mat = dd.to_matrix();
        for m in 0..9 {
            for n in 0..9 {
                approx_eq(mat.get(m, n), SamplesTensor4::SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // symmetric 3D
        let dd = Tensor4::from_array(&SamplesTensor4::SYM_SAMPLE1, Mandel::Symmetric).unwrap();
        let mat = dd.to_matrix();
        assert_eq!(mat.dims(), (9, 9));
        for m in 0..9 {
            for n in 0..9 {
                approx_eq(mat.get(m, n), SamplesTensor4::SYM_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // symmetric 2D
        let dd = Tensor4::from_array(&SamplesTensor4::SYM_2D_SAMPLE1, Mandel::Symmetric2D).unwrap();
        let mat = dd.to_matrix();
        assert_eq!(mat.dims(), (9, 9));
        for m in 0..9 {
            for n in 0..9 {
                approx_eq(mat.get(m, n), SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }
    }

    #[test]
    fn from_array_to_matrix_from_matrix_work() {
        // General
        let data = &[
            [
                [[18.0, 16.0, 14.0], [12.0, 10.0, 8.0], [6.0, 4.0, 2.0]],
                [[36.0, 32.0, 28.0], [24.0, 20.0, 16.0], [12.0, 8.0, 4.0]],
                [[54.0, 48.0, 42.0], [36.0, 30.0, 24.0], [18.0, 12.0, 6.0]],
            ],
            [
                [[72.0, 64.0, 56.0], [48.0, 40.0, 32.0], [24.0, 16.0, 8.0]],
                [[90.0, 80.0, 70.0], [60.0, 50.0, 40.0], [30.0, 20.0, 10.0]],
                [[108.0, 96.0, 84.0], [72.0, 60.0, 48.0], [36.0, 24.0, 12.0]],
            ],
            [
                [[126.0, 112.0, 98.0], [84.0, 70.0, 56.0], [42.0, 28.0, 14.0]],
                [[144.0, 128.0, 112.0], [96.0, 80.0, 64.0], [48.0, 32.0, 16.0]],
                [[162.0, 144.0, 126.0], [108.0, 90.0, 72.0], [54.0, 36.0, 18.0]],
            ],
        ];
        let dd = Tensor4::from_array(data, Mandel::General).unwrap();
        let m1 = dd.to_matrix();
        let correct = &[
            [18.0, 10.0, 2.0, 16.0, 8.0, 14.0, 12.0, 4.0, 6.0],
            [90.0, 50.0, 10.0, 80.0, 40.0, 70.0, 60.0, 20.0, 30.0],
            [162.0, 90.0, 18.0, 144.0, 72.0, 126.0, 108.0, 36.0, 54.0],
            [36.0, 20.0, 4.0, 32.0, 16.0, 28.0, 24.0, 8.0, 12.0],
            [108.0, 60.0, 12.0, 96.0, 48.0, 84.0, 72.0, 24.0, 36.0],
            [54.0, 30.0, 6.0, 48.0, 24.0, 42.0, 36.0, 12.0, 18.0],
            [72.0, 40.0, 8.0, 64.0, 32.0, 56.0, 48.0, 16.0, 24.0],
            [144.0, 80.0, 16.0, 128.0, 64.0, 112.0, 96.0, 32.0, 48.0],
            [126.0, 70.0, 14.0, 112.0, 56.0, 98.0, 84.0, 28.0, 42.0],
        ];
        mat_approx_eq(&m1, correct, 1e-13);
        let ee = Tensor4::from_matrix(correct, Mandel::General).unwrap();
        let m2 = ee.to_matrix();
        mat_approx_eq(&m2, correct, 1e-13);

        // Symmetric 3D
        let data = &[
            [
                [[6.0, 10.0, 12.0], [10.0, 4.0, 8.0], [12.0, 8.0, 2.0]],
                [[24.0, 40.0, 48.0], [40.0, 16.0, 32.0], [48.0, 32.0, 8.0]],
                [[36.0, 60.0, 72.0], [60.0, 24.0, 48.0], [72.0, 48.0, 12.0]],
            ],
            [
                [[24.0, 40.0, 48.0], [40.0, 16.0, 32.0], [48.0, 32.0, 8.0]],
                [[12.0, 20.0, 24.0], [20.0, 8.0, 16.0], [24.0, 16.0, 4.0]],
                [[30.0, 50.0, 60.0], [50.0, 20.0, 40.0], [60.0, 40.0, 10.0]],
            ],
            [
                [[36.0, 60.0, 72.0], [60.0, 24.0, 48.0], [72.0, 48.0, 12.0]],
                [[30.0, 50.0, 60.0], [50.0, 20.0, 40.0], [60.0, 40.0, 10.0]],
                [[18.0, 30.0, 36.0], [30.0, 12.0, 24.0], [36.0, 24.0, 6.0]],
            ],
        ];
        let dd = Tensor4::from_array(data, Mandel::Symmetric).unwrap();
        let m1 = dd.to_matrix();
        let correct = &[
            [6.0, 4.0, 2.0, 10.0, 8.0, 12.0, 10.0, 8.0, 12.0],
            [12.0, 8.0, 4.0, 20.0, 16.0, 24.0, 20.0, 16.0, 24.0],
            [18.0, 12.0, 6.0, 30.0, 24.0, 36.0, 30.0, 24.0, 36.0],
            [24.0, 16.0, 8.0, 40.0, 32.0, 48.0, 40.0, 32.0, 48.0],
            [30.0, 20.0, 10.0, 50.0, 40.0, 60.0, 50.0, 40.0, 60.0],
            [36.0, 24.0, 12.0, 60.0, 48.0, 72.0, 60.0, 48.0, 72.0],
            [24.0, 16.0, 8.0, 40.0, 32.0, 48.0, 40.0, 32.0, 48.0],
            [30.0, 20.0, 10.0, 50.0, 40.0, 60.0, 50.0, 40.0, 60.0],
            [36.0, 24.0, 12.0, 60.0, 48.0, 72.0, 60.0, 48.0, 72.0],
        ];
        mat_approx_eq(&m1, correct, 1e-13);
        let ee = Tensor4::from_matrix(correct, Mandel::Symmetric).unwrap();
        let m2 = ee.to_matrix();
        mat_approx_eq(&m2, correct, 1e-13);

        // Symmetric 2D
        let data = &[
            [
                [[6.0, 8.0, 0.0], [8.0, 4.0, 0.0], [0.0, 0.0, 2.0]],
                [[24.0, 32.0, 0.0], [32.0, 16.0, 0.0], [0.0, 0.0, 8.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[24.0, 32.0, 0.0], [32.0, 16.0, 0.0], [0.0, 0.0, 8.0]],
                [[12.0, 16.0, 0.0], [16.0, 8.0, 0.0], [0.0, 0.0, 4.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[18.0, 24.0, 0.0], [24.0, 12.0, 0.0], [0.0, 0.0, 6.0]],
            ],
        ];
        let dd = Tensor4::from_array(data, Mandel::Symmetric2D).unwrap();
        let m1 = dd.to_matrix();
        let correct = &[
            [6.0, 4.0, 2.0, 8.0, 0.0, 0.0, 8.0, 0.0, 0.0],
            [12.0, 8.0, 4.0, 16.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [18.0, 12.0, 6.0, 24.0, 0.0, 0.0, 24.0, 0.0, 0.0],
            [24.0, 16.0, 8.0, 32.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [24.0, 16.0, 8.0, 32.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        mat_approx_eq(&m1, correct, 1e-13);
        let ee = Tensor4::from_matrix(correct, Mandel::Symmetric2D).unwrap();
        let m2 = ee.to_matrix();
        mat_approx_eq(&m2, correct, 1e-13);
    }

    fn generate_dd() -> Tensor4 {
        let mut dd = Tensor4::new(Mandel::Symmetric);
        for m in 0..6 {
            for n in 0..6 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                let value = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
                dd.sym_set(i, j, k, l, value);
            }
        }
        dd
    }

    #[test]
    fn sym_set_works() {
        let dd = generate_dd();
        assert_eq!(
            format!("{:.0}", dd.to_matrix()),
            "┌                                              ┐\n\
             │ 1111 1122 1133 1112 1123 1113 1112 1123 1113 │\n\
             │ 2211 2222 2233 2212 2223 2213 2212 2223 2213 │\n\
             │ 3311 3322 3333 3312 3323 3313 3312 3323 3313 │\n\
             │ 1211 1222 1233 1212 1223 1213 1212 1223 1213 │\n\
             │ 2311 2322 2333 2312 2323 2313 2312 2323 2313 │\n\
             │ 1311 1322 1333 1312 1323 1313 1312 1323 1313 │\n\
             │ 1211 1222 1233 1212 1223 1213 1212 1223 1213 │\n\
             │ 2311 2322 2333 2312 2323 2313 2312 2323 2313 │\n\
             │ 1311 1322 1333 1312 1323 1313 1312 1323 1313 │\n\
             └                                              ┘"
        );
    }

    #[test]
    fn clone_and_serialize_work() {
        let dd = generate_dd();
        // clone
        let mut cloned = dd.clone();
        cloned.mat.set(0, 0, 9999.0);
        assert_eq!(
            format!("{:.0}", dd.to_matrix()),
            "┌                                              ┐\n\
             │ 1111 1122 1133 1112 1123 1113 1112 1123 1113 │\n\
             │ 2211 2222 2233 2212 2223 2213 2212 2223 2213 │\n\
             │ 3311 3322 3333 3312 3323 3313 3312 3323 3313 │\n\
             │ 1211 1222 1233 1212 1223 1213 1212 1223 1213 │\n\
             │ 2311 2322 2333 2312 2323 2313 2312 2323 2313 │\n\
             │ 1311 1322 1333 1312 1323 1313 1312 1323 1313 │\n\
             │ 1211 1222 1233 1212 1223 1213 1212 1223 1213 │\n\
             │ 2311 2322 2333 2312 2323 2313 2312 2323 2313 │\n\
             │ 1311 1322 1333 1312 1323 1313 1312 1323 1313 │\n\
             └                                              ┘"
        );
        assert_eq!(
            format!("{:.0}", cloned.to_matrix()),
            "┌                                              ┐\n\
             │ 9999 1122 1133 1112 1123 1113 1112 1123 1113 │\n\
             │ 2211 2222 2233 2212 2223 2213 2212 2223 2213 │\n\
             │ 3311 3322 3333 3312 3323 3313 3312 3323 3313 │\n\
             │ 1211 1222 1233 1212 1223 1213 1212 1223 1213 │\n\
             │ 2311 2322 2333 2312 2323 2313 2312 2323 2313 │\n\
             │ 1311 1322 1333 1312 1323 1313 1312 1323 1313 │\n\
             │ 1211 1222 1233 1212 1223 1213 1212 1223 1213 │\n\
             │ 2311 2322 2333 2312 2323 2313 2312 2323 2313 │\n\
             │ 1311 1322 1333 1312 1323 1313 1312 1323 1313 │\n\
             └                                              ┘"
        );
        // serialize
        let mut serialized = Vec::new();
        let mut serializer = rmp_serde::Serializer::new(&mut serialized);
        dd.serialize(&mut serializer).unwrap();
        assert!(serialized.len() > 0);
        // deserialize
        let mut deserializer = rmp_serde::Deserializer::new(&serialized[..]);
        let ee: Tensor4 = Deserialize::deserialize(&mut deserializer).unwrap();
        assert_eq!(
            format!("{:.0}", ee.to_matrix()),
            "┌                                              ┐\n\
             │ 1111 1122 1133 1112 1123 1113 1112 1123 1113 │\n\
             │ 2211 2222 2233 2212 2223 2213 2212 2223 2213 │\n\
             │ 3311 3322 3333 3312 3323 3313 3312 3323 3313 │\n\
             │ 1211 1222 1233 1212 1223 1213 1212 1223 1213 │\n\
             │ 2311 2322 2333 2312 2323 2313 2312 2323 2313 │\n\
             │ 1311 1322 1333 1312 1323 1313 1312 1323 1313 │\n\
             │ 1211 1222 1233 1212 1223 1213 1212 1223 1213 │\n\
             │ 2311 2322 2333 2312 2323 2313 2312 2323 2313 │\n\
             │ 1311 1322 1333 1312 1323 1313 1312 1323 1313 │\n\
             └                                              ┘"
        );
    }

    #[test]
    fn debug_works() {
        let dd = Tensor4::new(Mandel::General);
        assert!(format!("{:?}", dd).len() > 0);
    }

    #[test]
    fn constant_ii_works() {
        let ii = Tensor4::constant_ii();
        assert_eq!(
            format!("{}", ii.mat),
            "┌                   ┐\n\
             │ 1 0 0 0 0 0 0 0 0 │\n\
             │ 0 1 0 0 0 0 0 0 0 │\n\
             │ 0 0 1 0 0 0 0 0 0 │\n\
             │ 0 0 0 1 0 0 0 0 0 │\n\
             │ 0 0 0 0 1 0 0 0 0 │\n\
             │ 0 0 0 0 0 1 0 0 0 │\n\
             │ 0 0 0 0 0 0 1 0 0 │\n\
             │ 0 0 0 0 0 0 0 1 0 │\n\
             │ 0 0 0 0 0 0 0 0 1 │\n\
             └                   ┘"
        );
    }

    #[test]
    fn constant_tt_works() {
        let tt = Tensor4::constant_tt();
        assert_eq!(
            format!("{}", tt.mat),
            "┌                            ┐\n\
             │  1  0  0  0  0  0  0  0  0 │\n\
             │  0  1  0  0  0  0  0  0  0 │\n\
             │  0  0  1  0  0  0  0  0  0 │\n\
             │  0  0  0  1  0  0  0  0  0 │\n\
             │  0  0  0  0  1  0  0  0  0 │\n\
             │  0  0  0  0  0  1  0  0  0 │\n\
             │  0  0  0  0  0  0 -1  0  0 │\n\
             │  0  0  0  0  0  0  0 -1  0 │\n\
             │  0  0  0  0  0  0  0  0 -1 │\n\
             └                            ┘"
        );
    }

    #[test]
    fn constant_jj_works() {
        let jj = Tensor4::constant_jj(false);
        assert_eq!(
            format!("{}", jj.mat),
            "┌                   ┐\n\
             │ 1 1 1 0 0 0 0 0 0 │\n\
             │ 1 1 1 0 0 0 0 0 0 │\n\
             │ 1 1 1 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             └                   ┘"
        );
        let jj = Tensor4::constant_jj(true);
        assert_eq!(
            format!("{}", jj.mat),
            "┌             ┐\n\
             │ 1 1 1 0 0 0 │\n\
             │ 1 1 1 0 0 0 │\n\
             │ 1 1 1 0 0 0 │\n\
             │ 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 │\n\
             └             ┘"
        );
    }

    #[test]
    fn constant_pp_iso_works() {
        let pp_iso = Tensor4::constant_pp_iso(false);
        assert_eq!(
            format!("{:.3}", pp_iso.mat),
            "┌                                                       ┐\n\
             │ 0.333 0.333 0.333 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             │ 0.333 0.333 0.333 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             │ 0.333 0.333 0.333 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             │ 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             │ 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             │ 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             │ 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             │ 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             │ 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             └                                                       ┘"
        );
        let pp_iso = Tensor4::constant_pp_iso(true);
        assert_eq!(
            format!("{:.3}", pp_iso.mat),
            "┌                                     ┐\n\
             │ 0.333 0.333 0.333 0.000 0.000 0.000 │\n\
             │ 0.333 0.333 0.333 0.000 0.000 0.000 │\n\
             │ 0.333 0.333 0.333 0.000 0.000 0.000 │\n\
             │ 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             │ 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             │ 0.000 0.000 0.000 0.000 0.000 0.000 │\n\
             └                                     ┘"
        );
    }

    #[test]
    fn constant_pp_sym_works() {
        let pp_sym = Tensor4::constant_pp_sym(false);
        assert_eq!(
            format!("{}", pp_sym.mat),
            "┌                   ┐\n\
             │ 1 0 0 0 0 0 0 0 0 │\n\
             │ 0 1 0 0 0 0 0 0 0 │\n\
             │ 0 0 1 0 0 0 0 0 0 │\n\
             │ 0 0 0 1 0 0 0 0 0 │\n\
             │ 0 0 0 0 1 0 0 0 0 │\n\
             │ 0 0 0 0 0 1 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             └                   ┘"
        );
        let pp_sym = Tensor4::constant_pp_sym(true);
        assert_eq!(
            format!("{}", pp_sym.mat),
            "┌             ┐\n\
             │ 1 0 0 0 0 0 │\n\
             │ 0 1 0 0 0 0 │\n\
             │ 0 0 1 0 0 0 │\n\
             │ 0 0 0 1 0 0 │\n\
             │ 0 0 0 0 1 0 │\n\
             │ 0 0 0 0 0 1 │\n\
             └             ┘"
        );
    }

    #[test]
    fn constant_pp_skew_works() {
        let pp_skew = Tensor4::constant_pp_skew();
        assert_eq!(
            format!("{}", pp_skew.mat),
            "┌                   ┐\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 0 0 0 │\n\
             │ 0 0 0 0 0 0 1 0 0 │\n\
             │ 0 0 0 0 0 0 0 1 0 │\n\
             │ 0 0 0 0 0 0 0 0 1 │\n\
             └                   ┘"
        );
    }

    #[test]
    fn constant_pp_dev_works() {
        let pp_dev = Tensor4::constant_pp_dev();
        assert_eq!(
            format!("{:.3}", pp_dev.mat),
            "┌                                                                ┐\n\
             │  0.667 -0.333 -0.333  0.000  0.000  0.000  0.000  0.000  0.000 │\n\
             │ -0.333  0.667 -0.333  0.000  0.000  0.000  0.000  0.000  0.000 │\n\
             │ -0.333 -0.333  0.667  0.000  0.000  0.000  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  1.000  0.000  0.000  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  1.000  0.000  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  0.000  1.000  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  0.000  0.000  1.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000 │\n\
             └                                                                ┘"
        );
    }

    #[test]
    fn constant_pp_symdev_works() {
        let pp_symdev = Tensor4::constant_pp_symdev(false);
        assert_eq!(
            format!("{:.3}", pp_symdev.mat),
            "┌                                                                ┐\n\
             │  0.667 -0.333 -0.333  0.000  0.000  0.000  0.000  0.000  0.000 │\n\
             │ -0.333  0.667 -0.333  0.000  0.000  0.000  0.000  0.000  0.000 │\n\
             │ -0.333 -0.333  0.667  0.000  0.000  0.000  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  1.000  0.000  0.000  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  1.000  0.000  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  0.000  1.000  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 │\n\
             └                                                                ┘"
        );
        let pp_symdev = Tensor4::constant_pp_symdev(true);
        assert_eq!(
            format!("{:.3}", pp_symdev.mat),
            "┌                                           ┐\n\
             │  0.667 -0.333 -0.333  0.000  0.000  0.000 │\n\
             │ -0.333  0.667 -0.333  0.000  0.000  0.000 │\n\
             │ -0.333 -0.333  0.667  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  1.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  1.000  0.000 │\n\
             │  0.000  0.000  0.000  0.000  0.000  1.000 │\n\
             └                                           ┘"
        );
    }
}
