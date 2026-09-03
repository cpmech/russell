use super::{IJKL_TO_MN, IJKL_TO_MN_SYM, MN_TO_IJKL, SQRT_2};
use crate::{ONE_BY_3, StrError, TWO_BY_3};
use russell_lab::{AsArray2D, Matrix, Vector, format_scientific, mat_eigen_sym, mat_eigenvalues};
use serde::{Deserialize, Serialize};
use std::cmp;
use std::fmt::{self, Write};

#[cfg(feature = "heap")]
use russell_lab::{mat_inverse, mat_scale};

#[cfg(not(feature = "heap"))]
use russell_lab::small_mat_inv;

/// Defines a fourth-order tensor in R³×R³×R³×R³
///
/// # Standard and Kelvin-Mandel components
///
/// The methods of this struct follow a naming convention that distinguishes
/// between the **standard** (Cartesian) components `Dᵢⱼₖₗ` and the **Kelvin-Mandel**
/// components stored internally:
///
/// * Methods dealing with **standard components** carry the `std` qualifier in
///   their names (e.g., [Tensor4::from_std_matrix], [Tensor4::get_std],
///   [Tensor4::as_std_matrix], [Tensor4::sym_set_std]).
/// * Methods dealing directly with the **Kelvin-Mandel components** carry no qualifier
///   (e.g., [Tensor4::get], [Tensor4::set], [Tensor4::set_tensor],
///   [Tensor4::update]).
///
/// Internally, the components are converted to the Kelvin-Mandel basis as follows.
///
/// First, the following mapping to the Kelvin-Mandel space is considered:
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
/// N = 9:
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
/// N = 6:
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
/// N = 4:
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
#[derive(Clone, Debug)]
pub struct Tensor4<const N: usize> {
    /// Holds the components in Kelvin-Mandel basis as matrix (heap).
    ///
    /// Heap version => dynamically allocated memory
    #[cfg(feature = "heap")]
    pub(crate) mat: Matrix,

    /// Holds the components in Kelvin-Mandel basis as matrix (stack).
    #[cfg(not(feature = "heap"))]
    pub(crate) mat: [[f64; N]; N],
}

// Manual Serialize/Deserialize implementations: serde only implements the traits
// for concrete array sizes, so the derive fails for the generic `[[f64; N]; N]`.
// Since N is known to be 4, 6, or 9 only, we serialize the components as a sequence.
impl<const N: usize> Serialize for Tensor4<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut data = Vec::with_capacity(N * N);
        for m in 0..N {
            for n in 0..N {
                data.push(self.get(m, n));
            }
        }
        data.serialize(serializer)
    }
}

impl<'de, const N: usize> Deserialize<'de> for Tensor4<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = Vec::<f64>::deserialize(deserializer)?;
        let expected = N * N;
        if data.len() != expected {
            return Err(serde::de::Error::custom(format!(
                "Tensor4 dimension mismatch: expected {} components, got {}",
                expected,
                data.len()
            )));
        }
        let mut dd = Tensor4::new();
        let mut k = 0;
        for m in 0..N {
            for n in 0..N {
                dd.set(m, n, data[k]);
                k += 1;
            }
        }
        Ok(dd)
    }
}

impl<const N: usize> Tensor4<N> {
    const VALIDATE_DIM: () = assert!(N == 4 || N == 6 || N == 9, "Tensor dimension must be 4, 6, or 9");

    /// Creates a new (zeroed) Tensor4
    pub fn new() -> Self {
        let _ = Self::VALIDATE_DIM;

        #[cfg(feature = "heap")]
        {
            Tensor4 { mat: Matrix::new(N, N) }
        }
        #[cfg(not(feature = "heap"))]
        {
            Tensor4 { mat: [[0.0; N]; N] }
        }
    }

    /// Returns the (m,n) component of the Kelvin-Mandel matrix
    ///
    /// # Input
    ///
    /// * `m` -- the row index
    /// * `n` -- the column index
    ///
    /// # Panics
    ///
    /// A panic will occur if the indices are out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor4};
    ///
    /// let mut dd = Tensor4::<9>::new();
    /// dd.set(0, 0, 123.0);
    /// assert_eq!(dd.get(0, 0), 123.0);
    /// ```
    #[inline]
    pub fn get(&self, m: usize, n: usize) -> f64 {
        #[cfg(feature = "heap")]
        {
            self.mat.get(m, n)
        }
        #[cfg(not(feature = "heap"))]
        {
            self.mat[m][n]
        }
    }

    /// Sets the (m,n) component of the Kelvin-Mandel matrix
    ///
    /// # Input
    ///
    /// * `m` -- the row index
    /// * `n` -- the column index
    /// * `value` -- the value to set
    ///
    /// # Panics
    ///
    /// A panic will occur if the indices are out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor4};
    ///
    /// let mut dd = Tensor4::<9>::new();
    /// dd.set(0, 0, 123.0);
    /// assert_eq!(dd.get(0, 0), 123.0);
    /// ```
    #[inline]
    pub fn set(&mut self, m: usize, n: usize, value: f64) {
        #[cfg(feature = "heap")]
        {
            self.mat.set(m, n, value);
        }
        #[cfg(not(feature = "heap"))]
        {
            self.mat[m][n] = value;
        }
    }

    /// Adds a value to the (m,n) component of the Kelvin-Mandel matrix
    ///
    /// # Input
    ///
    /// * `m` -- the row index
    /// * `n` -- the column index
    /// * `value` -- the value to be added
    ///
    /// # Panics
    ///
    /// A panic will occur if the indices are out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor4};
    ///
    /// let mut dd = Tensor4::<9>::new();
    /// dd.set(0, 0, 123.0);
    /// dd.add(0, 0, 321.0);
    /// assert_eq!(dd.get(0, 0), 444.0);
    /// ```
    #[inline]
    pub fn add(&mut self, m: usize, n: usize, value: f64) {
        #[cfg(feature = "heap")]
        {
            self.mat.add(m, n, value);
        }
        #[cfg(not(feature = "heap"))]
        {
            self.mat[m][n] += value;
        }
    }

    /// Sets the Kelvin-Mandel matrix directly
    ///
    /// # Input
    ///
    /// * `inp` -- the Kelvin-Mandel matrix; it must have dimensions equal to `N`
    ///   (9×9 for `N = 9`, 6×6 for `N = 6`, and 4×4 for `N = 4`)
    ///
    /// # Warning
    ///
    /// For `N = 6` and `N = 4`, the input matrix must be symmetric
    /// (i.e., the tensor has minor symmetry). Otherwise, an error is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * the input matrix does not have dimensions equal to `N`
    /// * the input matrix is not symmetric (only for `N = 6` and `N = 4`)
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut dd = Tensor4::<6>::new();
    ///     #[rustfmt::skip]
    ///     let mat = [
    ///         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ///         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ///         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ///     ];
    ///     dd.set_matrix(&mat)?;
    ///     assert_eq!(dd.get(3, 3), 1.0);
    ///     Ok(())
    /// }
    /// ```
    pub fn set_matrix<'a, S>(&mut self, inp: &'a S) -> Result<(), StrError>
    where
        S: AsArray2D<'a, f64>,
    {
        let (m, n) = inp.size();
        if m != N || n != N {
            return Err("the input matrix must have dimensions equal to N");
        }
        // check symmetry (the Kelvin-Mandel matrix of a symmetric tensor must be symmetric)
        if N != 9 {
            for i in 0..N {
                for j in (i + 1)..N {
                    if inp.at(i, j) != inp.at(j, i) {
                        return Err("the input matrix must be symmetric");
                    }
                }
            }
        }
        for i in 0..N {
            for j in 0..N {
                self.set(i, j, inp.at(i, j));
            }
        }
        Ok(())
    }

    /// Sets this tensor from a nested array containing the standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard Dijkl components with respect to a Cartesian system
    pub fn set_std_array(&mut self, inp: &[[[[f64; 3]; 3]; 3]; 3]) -> Result<(), StrError> {
        if N == 4 || N == 6 {
            let max = if N == 4 { 3 } else { 6 };
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
                                    return Err("the input data does not correspond to a minor-symmetric tensor");
                                }
                            } else {
                                let (m, n) = IJKL_TO_MN[i][j][k][l];
                                if m > max || n > max {
                                    if inp[i][j][k][l] != 0.0 {
                                        return Err(
                                            "the input data does not correspond to a 2D minor-symmetric tensor",
                                        );
                                    }
                                    continue;
                                } else if m < 3 && n < 3 {
                                    self.set(m, n, inp[i][j][k][l]);
                                } else if m > 2 && n > 2 {
                                    self.set(m, n, 2.0 * inp[i][j][k][l]);
                                } else {
                                    self.set(m, n, SQRT_2 * inp[i][j][k][l]);
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
                                self.set(m, n, inp[i][j][k][l]);
                            // 2
                            } else if i == j && k < l {
                                self.set(m, n, (inp[i][j][k][l] + inp[i][j][l][k]) / SQRT_2);
                            // 3
                            } else if i == j && k > l {
                                self.set(m, n, (inp[i][j][l][k] - inp[i][j][k][l]) / SQRT_2);
                            // ** i < j **
                            // 4
                            } else if i < j && k == l {
                                self.set(m, n, (inp[i][j][k][l] + inp[j][i][k][l]) / SQRT_2);
                            // 5
                            } else if i < j && k < l {
                                self.set(
                                    m,
                                    n,
                                    (inp[i][j][k][l] + inp[i][j][l][k] + inp[j][i][k][l] + inp[j][i][l][k]) / 2.0,
                                );
                            // 6
                            } else if i < j && k > l {
                                self.set(
                                    m,
                                    n,
                                    (inp[i][j][l][k] - inp[i][j][k][l] + inp[j][i][l][k] - inp[j][i][k][l]) / 2.0,
                                );
                            // ** i > j **
                            // 7
                            } else if i > j && k == l {
                                self.set(m, n, (inp[j][i][k][l] - inp[i][j][k][l]) / SQRT_2);
                            // 8
                            } else if i > j && k < l {
                                self.set(
                                    m,
                                    n,
                                    (inp[j][i][k][l] + inp[j][i][l][k] - inp[i][j][k][l] - inp[i][j][l][k]) / 2.0,
                                );
                            // 9
                            } else if i > j && k > l {
                                self.set(
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
        Ok(())
    }

    /// Creates a new Tensor4 constructed from a nested array containing the standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard Dijkl components with respect to a Cartesian system
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor4, StrError};
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
    ///     let dd = Tensor4::<9>::from_std_array(&inp)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
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
    pub fn from_std_array(inp: &[[[[f64; 3]; 3]; 3]; 3]) -> Result<Self, StrError> {
        let mut res = Tensor4::new();
        res.set_std_array(inp)?;
        Ok(res)
    }

    /// Sets this tensor from a 9x9 matrix with standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard matrix of components with respect to a Cartesian system.
    ///   The matrix must be 9x9, even if it corresponds to a minor-symmetric tensor.
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix is not 9x9.
    pub fn set_std_matrix<'a, S>(&mut self, inp: &'a S) -> Result<(), StrError>
    where
        S: AsArray2D<'a, f64>,
    {
        if N == 4 || N == 6 {
            let max = if N == 4 { 3 } else { 6 };
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
                                if inp.at(m, n) != inp.at(p, q)
                                    || inp.at(m, n) != inp.at(r, s)
                                    || inp.at(m, n) != inp.at(u, v)
                                {
                                    return Err("the input data does not correspond to a minor-symmetric tensor");
                                }
                            } else {
                                if m > max || n > max {
                                    if inp.at(m, n) != 0.0 {
                                        return Err(
                                            "the input data does not correspond to a 2D minor-symmetric tensor",
                                        );
                                    }
                                    continue;
                                } else if m < 3 && n < 3 {
                                    self.set(m, n, inp.at(m, n));
                                } else if m > 2 && n > 2 {
                                    self.set(m, n, 2.0 * inp.at(m, n));
                                } else {
                                    self.set(m, n, SQRT_2 * inp.at(m, n));
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
                                self.set(m, n, inp.at(m, n));
                            // 2
                            } else if i == j && k < l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                self.set(m, n, (inp.at(m, n) + inp.at(p, q)) / SQRT_2);
                            // 3
                            } else if i == j && k > l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                self.set(m, n, (inp.at(p, q) - inp.at(m, n)) / SQRT_2);
                            // ** i < j **
                            // 4
                            } else if i < j && k == l {
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                self.set(m, n, (inp.at(m, n) + inp.at(r, s)) / SQRT_2);
                            // 5
                            } else if i < j && k < l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                self.set(m, n, (inp.at(m, n) + inp.at(p, q) + inp.at(r, s) + inp.at(u, v)) / 2.0);
                            // 6
                            } else if i < j && k > l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                self.set(m, n, (inp.at(p, q) - inp.at(m, n) + inp.at(u, v) - inp.at(r, s)) / 2.0);
                            // ** i > j **
                            // 7
                            } else if i > j && k == l {
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                self.set(m, n, (inp.at(r, s) - inp.at(m, n)) / SQRT_2);
                            // 8
                            } else if i > j && k < l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                self.set(m, n, (inp.at(r, s) + inp.at(u, v) - inp.at(m, n) - inp.at(p, q)) / 2.0);
                            // 9
                            } else if i > j && k > l {
                                let (p, q) = IJKL_TO_MN[i][j][l][k];
                                let (r, s) = IJKL_TO_MN[j][i][k][l];
                                let (u, v) = IJKL_TO_MN[j][i][l][k];
                                self.set(m, n, (inp.at(u, v) - inp.at(r, s) - inp.at(p, q) + inp.at(m, n)) / 2.0);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Creates a new Tensor4 constructed from a 9x9 matrix with standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard matrix of components with respect to a Cartesian system.
    ///   The matrix must be 9x9, even if it corresponds to a minor-symmetric tensor.
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix is not 9x9.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{MN_TO_IJKL, Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 9]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..9 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             inp[m][n] = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor4::<9>::from_std_matrix(&inp)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
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
    pub fn from_std_matrix<'a, S>(inp: &'a S) -> Result<Self, StrError>
    where
        S: AsArray2D<'a, f64>,
    {
        let mut res = Tensor4::new();
        res.set_std_matrix(inp)?;
        Ok(res)
    }

    /// Creates a new Tensor4 from the Kelvin-Mandel matrix directly
    ///
    /// # Input
    ///
    /// * `inp` -- the Kelvin-Mandel matrix; it must have dimensions equal to `N`
    ///   (9×9 for `N = 9`, 6×6 for `N = 6`, and 4×4 for `N = 4`)
    ///
    /// # Warning
    ///
    /// For `N = 6` and `N = 4`, the input matrix must be symmetric
    /// (i.e., the tensor has minor symmetry). Otherwise, an error is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * the input matrix does not have dimensions equal to `N`
    /// * the input matrix is not symmetric (only for `N = 6` and `N = 4`)
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     #[rustfmt::skip]
    ///     let mat = [
    ///         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ///         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ///         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ///     ];
    ///     let dd = Tensor4::<6>::from_matrix(&mat)?;
    ///     assert_eq!(dd.get(3, 3), 1.0);
    ///     Ok(())
    /// }
    /// ```
    pub fn from_matrix<'a, S>(inp: &'a S) -> Result<Self, StrError>
    where
        S: AsArray2D<'a, f64>,
    {
        let mut res = Tensor4::new();
        res.set_matrix(inp)?;
        Ok(res)
    }

    /// Returns the (i,j,k,l) standard component
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{MN_TO_IJKL, Tensor4, StrError};
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
    ///     let dd = Tensor4::<9>::from_std_matrix(&inp)?;
    ///
    ///     for m in 0..9 {
    ///         for n in 0..9 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             let val = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///             approx_eq(dd.get_std(i,j,k,l), val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn get_std(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        match N {
            4 => {
                let (m, n) = IJKL_TO_MN_SYM[i][j][k][l];
                if m > 3 || n > 3 {
                    0.0
                } else if m < 3 && n < 3 {
                    self.get(m, n)
                } else if m > 2 && n > 2 {
                    self.get(m, n) / 2.0
                } else {
                    self.get(m, n) / SQRT_2
                }
            }
            6 => {
                let (m, n) = IJKL_TO_MN_SYM[i][j][k][l];
                if m < 3 && n < 3 {
                    self.get(m, n)
                } else if m > 2 && n > 2 {
                    self.get(m, n) / 2.0
                } else {
                    self.get(m, n) / SQRT_2
                }
            }
            _ => {
                let (m, n) = IJKL_TO_MN[i][j][k][l];
                let val = self.get(m, n);
                // ** i == j **
                // 1
                if i == j && k == l {
                    val
                // 2
                } else if i == j && k < l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let right = self.get(p, q);
                    (val + right) / SQRT_2
                // 3
                } else if i == j && k > l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let left = self.get(p, q);
                    (left - val) / SQRT_2
                // ** i < j **
                // 4
                } else if i < j && k == l {
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let down = self.get(r, s);
                    (val + down) / SQRT_2
                // 5
                } else if i < j && k < l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let right = self.get(p, q);
                    let down = self.get(r, s);
                    let diag = self.get(u, v);
                    (val + right + down + diag) / 2.0
                // 6
                } else if i < j && k > l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let left = self.get(p, q);
                    let diag = self.get(u, v);
                    let down = self.get(r, s);
                    (left - val + diag - down) / 2.0
                // ** i > j **
                // 7
                } else if i > j && k == l {
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let up = self.get(r, s);
                    (up - val) / SQRT_2
                // 8
                } else if i > j && k < l {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let up = self.get(r, s);
                    let diag = self.get(u, v);
                    let right = self.get(p, q);
                    (up + diag - val - right) / 2.0
                // 9: i > j && k > l
                } else {
                    let (p, q) = IJKL_TO_MN[i][j][l][k];
                    let (r, s) = IJKL_TO_MN[j][i][k][l];
                    let (u, v) = IJKL_TO_MN[j][i][l][k];
                    let diag = self.get(u, v);
                    let up = self.get(r, s);
                    let left = self.get(p, q);
                    (diag - up - left + val) / 2.0
                }
            }
        }
    }

    /// Calculates the Euclidean norm
    ///
    /// ```text
    /// norm(D) = √(D:D)
    /// ```
    ///
    /// The norm is computed with the Kelvin-Mandel components, which yields the
    /// same value as the Frobenius norm of the standard components because the
    /// Kelvin-Mandel mapping is norm-preserving.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // the identity tensor II has norm = 3
    ///     let dd = Tensor4::<9>::constant_ii();
    ///     approx_eq(dd.norm(), 3.0, 1e-13);
    ///     Ok(())
    /// }
    /// ```
    pub fn norm(&self) -> f64 {
        let mut sm = 0.0;
        for m in 0..N {
            for n in 0..N {
                let v = self.get(m, n);
                sm += v * v;
            }
        }
        f64::sqrt(sm)
    }

    /// Scales this tensor in-place
    ///
    /// ```text
    /// self := α self
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut dd = Tensor4::<9>::new();
    ///     dd.set(0, 0, 1.0);
    ///     dd.set(1, 1, 2.0);
    ///     dd.set(2, 2, 3.0);
    ///     dd.scale(2.0);
    ///     assert_eq!(dd.get(0, 0), 2.0);
    ///     assert_eq!(dd.get(1, 1), 4.0);
    ///     assert_eq!(dd.get(2, 2), 6.0);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn scale(&mut self, alpha: f64) {
        #[cfg(feature = "heap")]
        {
            mat_scale(&mut self.mat, alpha);
        }
        #[cfg(not(feature = "heap"))]
        {
            for m in 0..N {
                for n in 0..N {
                    self.mat[m][n] *= alpha;
                }
            }
        }
    }

    /// Prints the Kelvin-Mandel matrix in scientific notation
    ///
    /// # Input
    ///
    /// * `label` -- a label (e.g., a description of the tensor)
    /// * `factor` -- a factor to multiply the components before printing (e.g., a unit conversion factor)
    /// * `width` -- the field width used to print each component
    /// * `precision` -- the number of digits after the decimal point
    pub fn print(&self, label: &str, factor: f64, width: usize, precision: usize) {
        println!("{} =", label);
        println!("┌{:1$}┐", " ", N * width + 1);
        for m in 0..N {
            if m > 0 {
                println!(" │");
            }
            for n in 0..N {
                if n == 0 {
                    print!("│");
                }
                let val = self.get(m, n) * factor;
                print!("{:>1$}", format_scientific(val, width, precision), width);
            }
        }
        println!(" │");
        println!("└{:1$}┘", " ", N * width + 1);
    }

    /// Adds another tensor to this one
    ///
    /// ```text
    /// self += α other
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{MN_TO_IJKL, Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 9]; 9];
    ///     for m in 0..4 {
    ///         for n in 0..4 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             inp[m][n] = 1.0;
    ///         }
    ///     }
    ///
    ///     let mut dd = Tensor4::<9>::new();
    ///     let ee = Tensor4::<9>::from_std_matrix(&inp)?;
    ///     dd.update(2.0, &ee);
    ///
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌                   ┐\n\
    ///          │ 2 2 2 2 0 0 0 0 0 │\n\
    ///          │ 2 2 2 2 0 0 0 0 0 │\n\
    ///          │ 2 2 2 2 0 0 0 0 0 │\n\
    ///          │ 2 2 2 2 0 0 0 0 0 │\n\
    ///          │ 0 0 0 0 0 0 0 0 0 │\n\
    ///          │ 0 0 0 0 0 0 0 0 0 │\n\
    ///          │ 0 0 0 0 0 0 0 0 0 │\n\
    ///          │ 0 0 0 0 0 0 0 0 0 │\n\
    ///          │ 0 0 0 0 0 0 0 0 0 │\n\
    ///          └                   ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn update(&mut self, alpha: f64, other: &Tensor4<N>) {
        for m in 0..N {
            for n in 0..N {
                self.set(m, n, self.get(m, n) + alpha * other.get(m, n));
            }
        }
    }

    /// Calculates the inverse of the Kelvin-Mandel matrix
    ///
    /// Note: the inverse Tensor4 can be obtained by inverting the Kelvin-Mandel matrix.
    ///
    /// Returns the determinant of the Kelvin-Mandel matrix and the inverse matrix/tensor in `inv`.
    pub fn inverse(&self, inv: &mut Tensor4<N>) -> Result<f64, StrError> {
        #[cfg(feature = "heap")]
        {
            mat_inverse(&mut inv.mat, &self.mat)
        }
        #[cfg(not(feature = "heap"))]
        {
            small_mat_inv(&mut inv.mat, &self.mat, N)
        }
    }

    /// Returns a 3x3x3x3 array with the standard components
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{MN_TO_IJKL, Tensor4, StrError};
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
    ///     let dd = Tensor4::<9>::from_std_matrix(&inp)?;
    ///     let arr = dd.as_std_array();
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
    pub fn as_std_array(&self) -> Vec<Vec<Vec<Vec<f64>>>> {
        let mut dd = vec![vec![vec![vec![0.0; 3]; 3]; 3]; 3];
        self.to_std_array(&mut dd);
        dd
    }

    /// Converts this tensor to a 3x3x3x3 array with the standard components
    ///
    /// # Panics
    ///
    /// A panic will occur if the array is not 3x3x3x3, i.e., `vec![vec![vec![vec![0.0; 3]; 3]; 3]; 3]`
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{MN_TO_IJKL, Tensor4, StrError};
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
    ///     let dd = Tensor4::<9>::from_std_matrix(&inp)?;
    ///     let mut arr = vec![vec![vec![vec![0.0; 3]; 3]; 3]; 3];
    ///     dd.to_std_array(&mut arr);
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
    pub fn to_std_array(&self, dd: &mut Vec<Vec<Vec<Vec<f64>>>>) {
        let dim = N;
        if dim < 9 {
            for m in 0..dim {
                for n in 0..dim {
                    let (i, j, k, l) = MN_TO_IJKL[m][n];
                    dd[i][j][k][l] = self.get_std(i, j, k, l);
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
                            dd[i][j][k][l] = self.get_std(i, j, k, l);
                        }
                    }
                }
            }
        }
    }

    /// Returns a 9x9 matrix with the standard components
    ///
    /// **Note:** The matrix will have the standard components and 9x9 dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{MN_TO_IJKL, Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 9]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..9 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             inp[m][n] = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor4::<9>::from_std_matrix(&inp)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
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
    pub fn as_std_matrix(&self) -> Matrix {
        let mut mat = Matrix::new(9, 9);
        self.to_std_matrix(&mut mat);
        mat
    }

    /// Calculates the eigenvalues of the Kelvin-Mandel matrix (without eigenvectors)
    ///
    /// # Warning
    ///
    /// The Kelvin-Mandel matrix is implicitly assumed symmetric (i.e., the tensor has
    /// major symmetry). Otherwise, only the lower triangle is used and the results are
    /// wrong without raising an error.
    ///
    /// # Output
    ///
    /// * `l` -- (lambda) will hold the eigenvalues (sorted in ascending order); ndim must equal N
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// 1. the tensor is not symmetric; i.e., N == 9
    /// 2. `l.dim()` is not equal to N
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Vector;
    /// use russell_tensor::{Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut dd = Tensor4::<6>::new();
    ///     dd.set(0, 0, 2.0);
    ///     dd.set(1, 1, 3.0);
    ///     dd.set(2, 2, 5.0);
    ///     dd.set(3, 3, 7.0);
    ///     dd.set(4, 4, 11.0);
    ///     dd.set(5, 5, 13.0);
    ///     let mut l = Vector::new(6);
    ///     dd.eigenvalues_sym(&mut l)?;
    ///     assert_eq!(format!("{:.0}", l), "┌    ┐\n│  2 │\n│  3 │\n│  5 │\n│  7 │\n│ 11 │\n│ 13 │\n└    ┘");
    ///     Ok(())
    /// }
    /// ```
    pub fn eigenvalues_sym(&self, l: &mut Vector) -> Result<(), StrError> {
        if N == 9 {
            return Err("the tensor must be symmetric");
        }
        if l.dim() != N {
            return Err("l.dim must be equal to the tensor dimension");
        }
        let mut a = Matrix::new(N, N);
        for m in 0..N {
            for n in 0..N {
                a.set(m, n, self.get(m, n));
            }
        }
        mat_eigen_sym(l, &mut a, false)?;
        Ok(())
    }

    /// Calculates the eigenvalues of the Kelvin-Mandel matrix (without eigenvectors)
    ///
    /// # Output
    ///
    /// * `l_real` -- will hold the real part of the eigenvalues; ndim must equal N
    /// * `l_imag` -- will hold the imaginary part of the eigenvalues; ndim must equal N
    ///
    /// # Errors
    ///
    /// Returns an error if `l_real.dim()` or `l_imag.dim()` is not equal to N
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Vector;
    /// use russell_tensor::{Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut dd = Tensor4::<9>::new();
    ///     for m in 0..9 {
    ///         dd.set(m, m, (m + 1) as f64);
    ///     }
    ///     let mut l_real = Vector::new(9);
    ///     let mut l_imag = Vector::new(9);
    ///     dd.eigenvalues(&mut l_real, &mut l_imag)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn eigenvalues(&self, l_real: &mut Vector, l_imag: &mut Vector) -> Result<(), StrError> {
        if l_real.dim() != N || l_imag.dim() != N {
            return Err("l_real.dim and l_imag.dim must be equal to the tensor dimension");
        }
        let mut a = Matrix::new(N, N);
        for m in 0..N {
            for n in 0..N {
                a.set(m, n, self.get(m, n));
            }
        }
        mat_eigenvalues(l_real, l_imag, &mut a)?;
        Ok(())
    }

    /// Converts this tensor to a 9x9 matrix with the standard components
    ///
    /// # Input
    ///
    /// * `mat` -- the resulting 9x9 matrix
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix is not 9x9
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Matrix;
    /// use russell_tensor::{MN_TO_IJKL, Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 9]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..9 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             inp[m][n] = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor4::<9>::from_std_matrix(&inp)?;
    ///     let mut mat = Matrix::new(9, 9);
    ///     dd.to_std_matrix(&mut mat);
    ///     assert_eq!(
    ///         format!("{:.0}", mat),
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
    pub fn to_std_matrix(&self, mat: &mut Matrix) {
        assert_eq!(mat.dims(), (9, 9));
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                mat.set(m, n, self.get_std(i, j, k, l));
            }
        }
    }

    /// Sets the (i,j,k,l) standard component of a minor-symmetric Tensor4
    ///
    /// # Notes
    ///
    /// 1. The tensor must be symmetric and (i,j) must correspond to the possible
    ///    combination due to the space dimension, otherwise a panic may occur.
    ///
    /// # Panics
    ///
    /// 1. A panic will occur if the tensor is general, i.e., `N = 9`
    /// 2. A panic will occur if the indices are out of range
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{MN_TO_IJKL, Tensor4};
    ///
    /// fn main() {
    ///     let mut dd = Tensor4::<4>::new();
    ///     for m in 0..4 {
    ///         for n in 0..4 {
    ///             let (i, j, k, l) = MN_TO_IJKL[m][n];
    ///             let value = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
    ///             dd.sym_set_std(i, j, k, l, value);
    ///         }
    ///     }
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
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
    pub fn sym_set_std(&mut self, i: usize, j: usize, k: usize, l: usize, value: f64) {
        assert!(N != 9);
        let (m, n) = IJKL_TO_MN_SYM[i][j][k][l];
        if m < 3 && n < 3 {
            self.set(m, n, value);
        } else if m > 2 && n > 2 {
            self.set(m, n, value * 2.0);
        } else {
            self.set(m, n, value * SQRT_2);
        }
    }

    /// Makes this tensor equal to another tensor, scaled by a factor alpha
    ///
    /// ```text
    /// self := α other
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::mat_approx_eq;
    /// use russell_tensor::{Tensor4, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let data = &[
    ///         [  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0],
    ///         [ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
    ///         [  2.0,  4.0,  6.0,  8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
    ///         [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    ///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    ///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    ///         [ -2.0, -4.0, -6.0, -8.0,-10.0,-12.0,-14.0,-16.0,-18.0],
    ///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    ///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    ///     ];
    ///     let dd = Tensor4::<9>::from_std_matrix(data)?;
    ///     let mut ee = Tensor4::<9>::new();
    ///
    ///     ee.set_tensor(1.0, &dd);
    ///
    ///     mat_approx_eq(&dd.as_std_matrix(), data, 1e-14);
    ///     Ok(())
    /// }
    /// ```
    pub fn set_tensor(&mut self, alpha: f64, other: &Tensor4<N>) {
        for m in 0..N {
            for n in 0..N {
                self.set(m, n, alpha * other.get(m, n));
            }
        }
    }

    //
    // --- constants tensors ---
    //

    /// Returns the fourth-order identity tensor (II)
    ///
    /// **Note:** this tensor cannot be represented in reduced-dimension because it is not minor-symmetric.
    ///
    /// ```text
    /// Definition:
    ///        _
    /// II = I ⊗ I
    /// ```
    ///
    /// ```text
    /// Kelvin-Mandel matrix:
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
    pub fn constant_ii() -> Self {
        assert_eq!(N, 9, "identity tensor requires N = 9");
        let mut ii = Tensor4::<N>::new();
        for m in 0..N {
            ii.set(m, m, 1.0);
        }
        ii
    }

    /// Returns the transposition tensor (TT)
    ///
    /// **Note:** this tensor cannot be represented in reduced-dimension because it is not minor-symmetric.
    ///
    /// ```text
    /// Definition:
    ///
    /// TT = I ⊗ I
    ///        ‾
    /// ```
    ///
    /// ```text
    /// Kelvin-Mandel matrix:
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
    pub fn constant_tt() -> Self {
        assert_eq!(N, 9, "transposition tensor requires N = 9");
        let mut tt = Tensor4::<N>::new();
        tt.set(0, 0, 1.0);
        tt.set(1, 1, 1.0);
        tt.set(2, 2, 1.0);
        tt.set(3, 3, 1.0);
        tt.set(4, 4, 1.0);
        tt.set(5, 5, 1.0);
        tt.set(6, 6, -1.0);
        tt.set(7, 7, -1.0);
        tt.set(8, 8, -1.0);
        tt
    }

    /// Returns the trace-projection tensor (JJ)
    ///
    /// Note: this tensor can be represented in reduced-dimension.
    ///
    /// ```text
    /// Definition:
    ///
    /// JJ = I ⊗ I
    /// ```
    ///
    /// ```text
    /// Kelvin-Mandel matrix:
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
    pub fn constant_jj() -> Self {
        let mut jj = Tensor4::<N>::new();
        jj.set(0, 0, 1.0);
        jj.set(0, 1, 1.0);
        jj.set(0, 2, 1.0);
        jj.set(1, 0, 1.0);
        jj.set(1, 1, 1.0);
        jj.set(1, 2, 1.0);
        jj.set(2, 0, 1.0);
        jj.set(2, 1, 1.0);
        jj.set(2, 2, 1.0);
        jj
    }

    /// Returns the isotropic making projector (Piso)
    ///
    /// Note: this tensor can be represented in reduced-dimension.
    ///
    /// ```text
    /// Definition:
    ///
    /// Piso = ⅓ I ⊗ I = ⅓ JJ
    /// ```
    ///
    /// ```text
    /// Kelvin-Mandel matrix:
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
    pub fn constant_pp_iso() -> Self {
        let mut pp_iso = Tensor4::<N>::new();
        pp_iso.set(0, 0, ONE_BY_3);
        pp_iso.set(0, 1, ONE_BY_3);
        pp_iso.set(0, 2, ONE_BY_3);
        pp_iso.set(1, 0, ONE_BY_3);
        pp_iso.set(1, 1, ONE_BY_3);
        pp_iso.set(1, 2, ONE_BY_3);
        pp_iso.set(2, 0, ONE_BY_3);
        pp_iso.set(2, 1, ONE_BY_3);
        pp_iso.set(2, 2, ONE_BY_3);
        pp_iso
    }

    /// Returns the symmetric making projector (Psym)
    ///
    /// Note: this tensor can be represented in reduced-dimension.
    ///
    /// ```text
    /// Definition:
    ///             _
    /// Psym = ½ (I ⊗ I + I ⊗ I) = ½ (II + TT) = ½ ssd(I)
    ///                     ‾
    /// ```
    ///
    /// ```text
    /// Kelvin-Mandel matrix:
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
    pub fn constant_pp_sym() -> Self {
        assert_ne!(N, 4, "Psym tensor cannot be allocated with N = 4");
        let mut pp_sym = Tensor4::<N>::new();
        pp_sym.set(0, 0, 1.0);
        pp_sym.set(1, 1, 1.0);
        pp_sym.set(2, 2, 1.0);
        pp_sym.set(3, 3, 1.0);
        pp_sym.set(4, 4, 1.0);
        pp_sym.set(5, 5, 1.0);
        pp_sym
    }

    /// Returns the skew making projector Pskew
    ///
    /// **Note:** this tensor cannot be represented in reduced-dimension because it is not minor-symmetric.
    ///
    /// ```text
    /// Definition:
    ///              _
    /// Pskew = ½ (I ⊗ I - I ⊗ I) = ½ (II - TT)
    ///                      ‾
    /// ```
    ///
    /// ```text
    /// Kelvin-Mandel matrix:
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
    pub fn constant_pp_skew() -> Self {
        assert_eq!(N, 9, "Pskew tensor requires N = 9");
        let mut pp_skew = Tensor4::<N>::new();
        pp_skew.set(6, 6, 1.0);
        pp_skew.set(7, 7, 1.0);
        pp_skew.set(8, 8, 1.0);
        pp_skew
    }

    /// Returns the deviatoric making projector Pdev
    ///
    /// **Note:** this tensor cannot be represented in reduced-dimension because it is not minor-symmetric.
    ///
    /// ```text
    /// Definition:
    ///          _
    /// Pdev = I ⊗ I - ⅓ I ⊗ I = II - Piso
    /// ```
    ///
    /// ```text
    /// Kelvin-Mandel matrix:
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
    pub fn constant_pp_dev() -> Self {
        assert_eq!(N, 9, "Pdev tensor requires N = 9");
        let mut pp_dev = Tensor4::<N>::new();
        pp_dev.set(0, 0, TWO_BY_3);
        pp_dev.set(0, 1, -ONE_BY_3);
        pp_dev.set(0, 2, -ONE_BY_3);
        pp_dev.set(1, 0, -ONE_BY_3);
        pp_dev.set(1, 1, TWO_BY_3);
        pp_dev.set(1, 2, -ONE_BY_3);
        pp_dev.set(2, 0, -ONE_BY_3);
        pp_dev.set(2, 1, -ONE_BY_3);
        pp_dev.set(2, 2, TWO_BY_3);
        pp_dev.set(3, 3, 1.0);
        pp_dev.set(4, 4, 1.0);
        pp_dev.set(5, 5, 1.0);
        pp_dev.set(6, 6, 1.0);
        pp_dev.set(7, 7, 1.0);
        pp_dev.set(8, 8, 1.0);
        pp_dev
    }

    /// Returns the symmetric-deviatoric making projector Psymdev
    ///
    /// Note: this tensor can be represented in reduced-dimension, but not with N = 4.
    ///
    /// ```text
    /// Definition:
    ///                _
    /// Psymdev = ½ (I ⊗ I + I ⊗ I) - ⅓ I ⊗ I = Psym - Piso
    ///                        ‾
    /// ```
    ///
    /// ```text
    /// Kelvin-Mandel matrix:
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
    pub fn constant_pp_symdev() -> Self {
        assert_ne!(N, 4, "Psymdev tensor cannot be allocated with N = 4");
        let mut pp_symdev = Tensor4::<N>::new();
        pp_symdev.set(0, 0, TWO_BY_3);
        pp_symdev.set(0, 1, -ONE_BY_3);
        pp_symdev.set(0, 2, -ONE_BY_3);
        pp_symdev.set(1, 0, -ONE_BY_3);
        pp_symdev.set(1, 1, TWO_BY_3);
        pp_symdev.set(1, 2, -ONE_BY_3);
        pp_symdev.set(2, 0, -ONE_BY_3);
        pp_symdev.set(2, 1, -ONE_BY_3);
        pp_symdev.set(2, 2, TWO_BY_3);
        pp_symdev.set(3, 3, 1.0);
        pp_symdev.set(4, 4, 1.0);
        pp_symdev.set(5, 5, 1.0);
        pp_symdev
    }

    /// Sets this tensor equal the symmetric-deviatoric making projector (Psymdev)
    ///
    /// Note: this tensor can be represented in reduced-dimension, but not with N = 4.
    ///
    /// ```text
    /// Definition:
    ///                _
    /// Psymdev = ½ (I ⊗ I + I ⊗ I) - ⅓ I ⊗ I = Psym - Piso
    ///                        ‾
    /// ```
    ///
    /// ```text
    /// Kelvin-Mandel matrix:
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
    pub fn set_pp_symdev(&mut self) {
        assert_ne!(N, 4, "Psymdev tensor cannot be allocated with N = 4");
        for m in 0..N {
            for n in 0..N {
                self.set(m, n, 0.0);
            }
        }
        self.set(0, 0, TWO_BY_3);
        self.set(0, 1, -ONE_BY_3);
        self.set(0, 2, -ONE_BY_3);
        self.set(1, 0, -ONE_BY_3);
        self.set(1, 1, TWO_BY_3);
        self.set(1, 2, -ONE_BY_3);
        self.set(2, 0, -ONE_BY_3);
        self.set(2, 1, -ONE_BY_3);
        self.set(2, 2, TWO_BY_3);
        self.set(3, 3, 1.0);
        self.set(4, 4, 1.0);
        self.set(5, 5, 1.0);
    }
}

impl<const N: usize> fmt::Display for Tensor4<N> {
    /// Generates a string representation of Kelvin-Mandel matrix associated with this Tensor4
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..N {
            for j in 0..N {
                let val = self.get(i, j);
                match f.precision() {
                    Some(v) => write!(&mut buf, "{:.1$}", val, v).unwrap(),
                    None => write!(&mut buf, "{}", val).unwrap(),
                }
                width = cmp::max(buf.chars().count(), width);
                buf.clear();
            }
        }
        // draw matrix
        width += 1;
        write!(f, "┌{:1$}┐\n", " ", width * N + 1).unwrap();
        for i in 0..N {
            if i > 0 {
                write!(f, " │\n").unwrap();
            }
            for j in 0..N {
                if j == 0 {
                    write!(f, "│").unwrap();
                }
                let val = self.get(i, j);
                match f.precision() {
                    Some(v) => write!(f, "{:>1$.2$}", val, width, v).unwrap(),
                    None => write!(f, "{:>1$}", val, width).unwrap(),
                }
            }
        }
        write!(f, " │\n").unwrap();
        write!(f, "└{:1$}┘", " ", width * N + 1).unwrap();
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{MN_TO_IJKL, Tensor4};
    use crate::{IDENTITY4, P_DEV, P_ISO, P_SKEW, P_SYM, P_SYMDEV, TRACE_PROJECTION, TRANSPOSITION};
    use crate::{SQRT_2, SamplesTensor4};
    use russell_lab::{Matrix, Vector, approx_eq, mat_approx_eq, vec_approx_eq};

    // Computes the reference norm from the standard components
    fn norm_from_std_array(arr: &[[[[f64; 3]; 3]; 3]; 3]) -> f64 {
        let mut sm = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        sm += arr[i][j][k][l] * arr[i][j][k][l];
                    }
                }
            }
        }
        f64::sqrt(sm)
    }

    #[test]
    fn norm_works() {
        let dd = Tensor4::<9>::from_std_array(&SamplesTensor4::SAMPLE1).unwrap();
        approx_eq(dd.norm(), norm_from_std_array(&SamplesTensor4::SAMPLE1), 1e-13);
        let dd = Tensor4::<6>::from_std_array(&SamplesTensor4::SYM_SAMPLE1).unwrap();
        approx_eq(dd.norm(), norm_from_std_array(&SamplesTensor4::SYM_SAMPLE1), 1e-13);
        let dd = Tensor4::<4>::from_std_array(&SamplesTensor4::SYM_2D_SAMPLE1).unwrap();
        approx_eq(dd.norm(), norm_from_std_array(&SamplesTensor4::SYM_2D_SAMPLE1), 1e-13);
    }

    #[test]
    fn scale_works() {
        let mut dd = Tensor4::<9>::new();
        dd.set(0, 0, 1.0);
        dd.set(1, 1, 2.0);
        dd.set(2, 2, 3.0);
        dd.scale(2.0);
        assert_eq!(dd.get(0, 0), 2.0);
        assert_eq!(dd.get(1, 1), 4.0);
        assert_eq!(dd.get(2, 2), 6.0);
    }

    #[test]
    fn eigenvalues_sym_works() {
        let mut dd = Tensor4::<6>::new();
        dd.set(0, 0, 2.0);
        dd.set(1, 1, 3.0);
        dd.set(2, 2, 5.0);
        dd.set(3, 3, 7.0);
        dd.set(4, 4, 11.0);
        dd.set(5, 5, 13.0);
        let mut l = Vector::new(6);
        dd.eigenvalues_sym(&mut l).unwrap();
        vec_approx_eq(&l, &[2.0, 3.0, 5.0, 7.0, 11.0, 13.0], 1e-13);
    }

    #[test]
    fn eigenvalues_sym_returns_err() {
        let dd = Tensor4::<9>::new();
        let mut l = Vector::new(9);
        assert_eq!(dd.eigenvalues_sym(&mut l).err(), Some("the tensor must be symmetric"));
        let dd = Tensor4::<6>::new();
        let mut l = Vector::new(4);
        assert_eq!(
            dd.eigenvalues_sym(&mut l).err(),
            Some("l.dim must be equal to the tensor dimension")
        );
    }

    #[test]
    fn eigenvalues_works() {
        let mut dd = Tensor4::<9>::new();
        for m in 0..9 {
            dd.set(m, m, (m + 1) as f64);
        }
        let mut lr = Vector::new(9);
        let mut li = Vector::new(9);
        dd.eigenvalues(&mut lr, &mut li).unwrap();
        // sum of real parts = trace = 1 + 2 + ... + 9 = 45
        let sum: f64 = lr.as_data().iter().sum();
        approx_eq(sum, 45.0, 1e-12);
        // all imaginary parts are zero (diagonal matrix)
        for k in 0..9 {
            approx_eq(li[k], 0.0, 1e-13);
        }
    }

    #[test]
    fn eigenvalues_returns_err() {
        let dd = Tensor4::<9>::new();
        let mut lr = Vector::new(9);
        let mut li = Vector::new(8);
        assert_eq!(
            dd.eigenvalues(&mut lr, &mut li).err(),
            Some("l_real.dim and l_imag.dim must be equal to the tensor dimension")
        );
    }

    // sorts complex eigenvalues (as (real, imag) pairs) in ascending order
    fn sorted_complex(lr: &Vector, li: &Vector) -> Vec<(f64, f64)> {
        let mut v: Vec<(f64, f64)> = (0..lr.dim()).map(|k| (lr[k], li[k])).collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
        v
    }

    // Python reference (numpy + scipy):
    // ```python
    // import numpy as np
    // from scipy import linalg
    // M = np.diag([2.0] * 6) + np.diag([-1.0] * 5, 1) + np.diag([-1.0] * 5, -1)
    // linalg.eigvalsh(M)
    // # -> array([0.1980622641951619, 0.7530203962825329, 1.5549581320873714,
    // #           2.445041867912629, 3.2469796037174664, 3.801937735804839])
    // ```
    #[test]
    fn eigenvalues_sym_works_tridiagonal() {
        // 1D Laplacian (tridiagonal, diagonal = 2, off-diagonal = -1)
        let mut dd = Tensor4::<6>::new();
        for m in 0..6 {
            dd.set(m, m, 2.0);
            if m > 0 {
                dd.set(m, m - 1, -1.0);
                dd.set(m - 1, m, -1.0);
            }
        }
        let mut l = Vector::new(6);
        dd.eigenvalues_sym(&mut l).unwrap();
        #[rustfmt::skip]
        let correct = [
            0.1980622641951619, 0.7530203962825329, 1.5549581320873714,
            2.4450418679126290, 3.2469796037174664, 3.8019377358048390,
        ];
        vec_approx_eq(&l, &correct, 1e-13);
    }

    // Python reference (numpy + scipy):
    // ```python
    // import numpy as np
    // from scipy import linalg
    // M = np.zeros((9, 9))
    // M[0, 1] = -1.0
    // M[1, 0] = 1.0
    // for k in range(2, 9):
    //     M[k, k] = float(k + 1)
    // linalg.eigvals(M)  # -> {-1.j, 1.j, 3., 4., 5., 6., 7., 8., 9.}
    // ```
    #[test]
    fn eigenvalues_works_complex_block() {
        // 9x9 = block-diag( [[0,-1],[1,0]], diag(3..9) ): eigenvalues {i, -i, 3, ..., 9}
        let mut dd = Tensor4::<9>::new();
        dd.set(0, 1, -1.0);
        dd.set(1, 0, 1.0);
        for k in 2..9 {
            dd.set(k, k, (k + 1) as f64);
        }
        let mut lr = Vector::new(9);
        let mut li = Vector::new(9);
        dd.eigenvalues(&mut lr, &mut li).unwrap();
        let got = sorted_complex(&lr, &li);
        let expected = [
            (0.0, -1.0),
            (0.0, 1.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
            (7.0, 0.0),
            (8.0, 0.0),
            (9.0, 0.0),
        ];
        for k in 0..9 {
            approx_eq(got[k].0, expected[k].0, 1e-13);
            approx_eq(got[k].1, expected[k].1, 1e-13);
        }
    }

    #[test]
    fn new_set_and_get_work() {
        // general
        let mut dd = Tensor4::<9>::new();
        dd.set(0, 0, 123.0);
        assert_eq!(dd.get(0, 0), 123.0);

        // symmetric
        let mut dd = Tensor4::<6>::new();
        dd.set(0, 0, 123.0);
        assert_eq!(dd.get(0, 0), 123.0);

        // symmetric 2d
        let mut dd = Tensor4::<4>::new();
        dd.set(0, 0, 123.0);
        assert_eq!(dd.get(0, 0), 123.0);
    }

    #[test]
    fn set_matrix_works() {
        // general (9x9) -- symmetry is not required
        let mut dd = Tensor4::<9>::new();
        let mut mat = [[0.0; 9]; 9];
        for m in 0..9 {
            for n in 0..9 {
                mat[m][n] = (100 * (m + 1) + (n + 1)) as f64;
            }
        }
        dd.set_matrix(&mat).unwrap();
        assert_eq!(dd.get(0, 0), 101.0);
        assert_eq!(dd.get(8, 8), 909.0);

        // symmetric (6x6)
        let mut dd = Tensor4::<6>::new();
        let mut mat = [[0.0; 6]; 6];
        for m in 0..6 {
            for n in 0..6 {
                mat[m][n] = ((m + 1) + (n + 1)) as f64;
            }
        }
        dd.set_matrix(&mat).unwrap();
        assert_eq!(dd.get(0, 0), 2.0);
        assert_eq!(dd.get(5, 5), 12.0);
        assert_eq!(dd.get(0, 1), 3.0);

        // error: wrong dimensions
        let mut dd = Tensor4::<6>::new();
        let mat = [[0.0; 5]; 5];
        assert_eq!(
            dd.set_matrix(&mat).err(),
            Some("the input matrix must have dimensions equal to N")
        );

        // error: not symmetric
        let mut dd = Tensor4::<6>::new();
        #[rustfmt::skip]
        let mat = [
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 7.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 8.0],
        ];
        assert_eq!(dd.set_matrix(&mat).err(), Some("the input matrix must be symmetric"));
    }

    #[test]
    fn from_matrix_works() {
        // general (9x9)
        let mut mat = [[0.0; 9]; 9];
        for m in 0..9 {
            for n in 0..9 {
                mat[m][n] = (100 * (m + 1) + (n + 1)) as f64;
            }
        }
        let dd = Tensor4::<9>::from_matrix(&mat).unwrap();
        assert_eq!(dd.get(0, 0), 101.0);
        assert_eq!(dd.get(8, 8), 909.0);

        // symmetric (6x6)
        let mut mat = [[0.0; 6]; 6];
        for m in 0..6 {
            for n in 0..6 {
                mat[m][n] = ((m + 1) + (n + 1)) as f64;
            }
        }
        let dd = Tensor4::<6>::from_matrix(&mat).unwrap();
        assert_eq!(dd.get(0, 0), 2.0);
        assert_eq!(dd.get(5, 5), 12.0);

        // error: wrong dimensions
        let mat = [[0.0; 5]; 5];
        assert_eq!(
            Tensor4::<6>::from_matrix(&mat).err(),
            Some("the input matrix must have dimensions equal to N")
        );
    }

    #[test]
    fn from_std_array_fails_captures_errors() {
        let res = Tensor4::<6>::from_std_array(&SamplesTensor4::SAMPLE1);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );

        let res = Tensor4::<4>::from_std_array(&SamplesTensor4::SYM_SAMPLE1);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn from_std_array_works() {
        // general
        let dd = Tensor4::<9>::from_std_array(&SamplesTensor4::SAMPLE1).unwrap();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(dd.get(m, n), SamplesTensor4::SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 3d
        let dd = Tensor4::<6>::from_std_array(&SamplesTensor4::SYM_SAMPLE1).unwrap();
        for m in 0..6 {
            for n in 0..6 {
                assert_eq!(dd.get(m, n), SamplesTensor4::SYM_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 2d
        let dd = Tensor4::<4>::from_std_array(&SamplesTensor4::SYM_2D_SAMPLE1).unwrap();
        for m in 0..4 {
            for n in 0..4 {
                assert_eq!(dd.get(m, n), SamplesTensor4::SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }
    }

    #[test]
    fn from_std_matrix_fails_captures_errors() {
        let mut inp = [[0.0; 9]; 9];
        inp[0][3] = 1e-15;
        let res = Tensor4::<6>::from_std_matrix(&inp);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );

        inp[0][3] = 0.0;
        inp[0][4] = 1.0;
        inp[0][7] = 1.0;
        let res = Tensor4::<4>::from_std_matrix(&inp);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn get_and_set_work() {
        let mut dd = Tensor4::<4>::new();
        assert_eq!(dd.get(0, 0), 0.0);
        dd.set(0, 0, 2.0);
        assert_eq!(dd.get(0, 0), 2.0);
    }

    #[test]
    fn from_std_matrix_works() {
        // general
        let dd = Tensor4::<9>::from_std_matrix(&SamplesTensor4::SAMPLE1_STD_MATRIX).unwrap();
        for m in 0..9 {
            for n in 0..9 {
                approx_eq(dd.get(m, n), SamplesTensor4::SAMPLE1_KELVIN_MATRIX[m][n], 1e-15);
            }
        }

        // symmetric 3D
        let dd = Tensor4::<6>::from_std_matrix(&SamplesTensor4::SYM_SAMPLE1_STD_MATRIX).unwrap();
        for m in 0..6 {
            for n in 0..6 {
                approx_eq(dd.get(m, n), SamplesTensor4::SYM_SAMPLE1_KELVIN_MATRIX[m][n], 1e-14);
            }
        }

        // symmetric 2D
        let dd = Tensor4::<4>::from_std_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX).unwrap();
        for m in 0..4 {
            for n in 0..4 {
                approx_eq(dd.get(m, n), SamplesTensor4::SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n], 1e-14);
            }
        }
    }

    #[test]
    fn get_std_works() {
        // general
        let dd = Tensor4::<9>::from_std_array(&SamplesTensor4::SAMPLE1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(dd.get_std(i, j, k, l), SamplesTensor4::SAMPLE1[i][j][k][l], 1e-13);
                    }
                }
            }
        }

        // symmetric 3D
        let dd = Tensor4::<6>::from_std_array(&SamplesTensor4::SYM_SAMPLE1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(dd.get_std(i, j, k, l), SamplesTensor4::SYM_SAMPLE1[i][j][k][l], 1e-14);
                    }
                }
            }
        }

        // symmetric 2D
        let dd = Tensor4::<4>::from_std_array(&SamplesTensor4::SYM_2D_SAMPLE1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(
                            dd.get_std(i, j, k, l),
                            SamplesTensor4::SYM_2D_SAMPLE1[i][j][k][l],
                            1e-14,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn update_works() {
        let mut dd = Tensor4::<4>::new();
        let ee = Tensor4::<4>::from_std_array(&SamplesTensor4::SYM_2D_SAMPLE1).unwrap();
        dd.update(2.0, &ee);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(
                            dd.get_std(i, j, k, l),
                            2.0 * SamplesTensor4::SYM_2D_SAMPLE1[i][j][k][l],
                            1e-14,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn inverse_works() {
        let aa_std = [
            [
                [[1.0, 1.0, 3.0], [2.0, 1.0, 3.0], [3.0, 1.0, 1.0]],
                [[2.0, 3.0, 8.0], [6.0, 3.0, 9.0], [7.0, 3.0, 5.0]],
                [[2.0, 5.0, 13.0], [11.0, 6.0, 17.0], [12.0, 8.0, 14.0]],
            ],
            [
                [[1.0, 2.0, 7.0], [7.0, 7.0, 13.0], [12.0, 11.0, 12.0]],
                [[2.0, 5.0, 13.0], [14.0, 16.0, 27.0], [21.0, 20.0, 21.0]],
                [[3.0, 5.0, 15.0], [13.0, 12.0, 25.0], [27.0, 21.0, 21.0]],
            ],
            [
                [[3.0, 6.0, 17.0], [15.0, 13.0, 30.0], [33.0, 26.0, 29.0]],
                [[3.0, 5.0, 14.0], [12.0, 12.0, 25.0], [30.0, 25.0, 25.0]],
                [[1.0, 3.0, 9.0], [11.0, 16.0, 25.0], [28.0, 34.0, 36.0]],
            ],
        ];
        #[rustfmt::skip]
        let aa_kel_expected = [
            [ 1.0, 1.0, 1.0, 3.0 / SQRT_2, 2.0 * SQRT_2, 3.0 * SQRT_2, -1.0 / SQRT_2, SQRT_2, 0.0],
            [ 2.0, 16.0, 21.0, 19.0 / SQRT_2, 47.0 / SQRT_2, 17.0 * SQRT_2, -9.0 / SQRT_2, 7.0 / SQRT_2, -4.0 * SQRT_2],
            [ 1.0, 16.0, 36.0, 7.0 * SQRT_2, 59.0 / SQRT_2, 37.0 / SQRT_2, -4.0 * SQRT_2, -9.0 / SQRT_2, -19.0 / SQRT_2],
            [ 3.0 / SQRT_2, 5.0 * SQRT_2, 17.0 / SQRT_2, 9.0, 18.0, 17.0, -4.0, 4.0, -2.0],
            [ 3.0 * SQRT_2, 12.0 * SQRT_2, 23.0 * SQRT_2, 17.5, 48.0, 43.0, -7.5, 2.0, -14.0],
            [ 5.0 / SQRT_2, 19.0 / SQRT_2, 43.0 / SQRT_2, 18.5, 40.5, 37.5, -7.5, 6.5, -7.5],
            [ 1.0 / SQRT_2, -2.0 * SQRT_2, -7.0 / SQRT_2, 0.0, -6.0, -2.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, -2.0 * SQRT_2, 0.5, -2.0, -1.0, -0.5, 2.0, 2.0],
            [ -1.0 / SQRT_2, -7.0 / SQRT_2, -15.0 / SQRT_2, -2.5, -15.5, -12.5, 1.5, 2.5, 8.5],
        ];
        #[rustfmt::skip]
        let aa_kel_inv_expected = [
            [ 50.0, 2.0, 3.0, -22.0 * SQRT_2, -2.0 * SQRT_2, 13.0 / SQRT_2, -8.0 * SQRT_2, 8.0 * SQRT_2, 5.0 / SQRT_2],
            [ -141.0, -7.0, 2.0, 107.0 / SQRT_2, 3.0 * SQRT_2, -17.0 * SQRT_2, 87.0 / SQRT_2, 9.0 * SQRT_2, -16.0 * SQRT_2],
            [ -59.0, -3.0, 1.0, 45.0 / SQRT_2, SQRT_2, -7.0 * SQRT_2, 37.0 / SQRT_2, 4.0 * SQRT_2, -7.0 * SQRT_2],
            [ -91.0 / SQRT_2, -4.0 * SQRT_2, 12.0 * SQRT_2, 4.0, 0.5, -5.5, 71.0, 82.5, -24.5],
            [ 125.0 * SQRT_2, 7.0 * SQRT_2, -6.0 * SQRT_2, -84.0, -4.5, 28.0, -93.0, -44.5, 34.0],
            [ -39.0 * SQRT_2, -3.0 / SQRT_2, -3.0 * SQRT_2, 38.5, 2.5, -11.0, 10.5, -18.5, -4.0],
            [-7.0 / SQRT_2, SQRT_2, -6.0 * SQRT_2, 17.0, 1.5, -3.5, -20.0, -40.5, 7.5],
            [9.0 * SQRT_2, SQRT_2, -3.0 * SQRT_2, 1.0, -0.5, 1.0, -16.0, -20.5, 5.0],
            [ 48.0 * SQRT_2, 7.0 / SQRT_2, -8.0 * SQRT_2, -17.5, -0.5, 8.0, -57.5, -55.5, 21.0],
        ];
        let aa = Tensor4::<9>::from_std_array(&aa_std).unwrap();
        for m in 0..9 {
            for n in 0..9 {
                approx_eq(aa.get(m, n), aa_kel_expected[m][n], 1e-14);
            }
        }
        let mut aa_inv = Tensor4::<9>::new();
        let det = aa.inverse(&mut aa_inv).unwrap();
        approx_eq(det, 1.0, 1e-12);
        for m in 0..9 {
            for n in 0..9 {
                approx_eq(aa_inv.get(m, n), aa_kel_inv_expected[m][n], 1e-10);
            }
        }
        // Check Dijpq Dpqkl⁻¹ = δik δjl
        let aa_inv_std = aa_inv.as_std_array();
        let aa_inv_std_expected = [
            [
                [[50.0, -30.0, 9.0], [-14.0, 2.0, 6.0], [4.0, -10.0, 3.0]],
                [[-49.0, 36.0, -13.0], [-15.0, -3.0, 22.0], [4.0, -20.0, 6.0]],
                [[9.0, -13.0, 7.0], [34.0, 2.0, -36.0], [-10.0, 38.0, -11.0]],
            ],
            [
                [[-42.0, 39.0, -17.0], [-52.0, -5.0, 61.0], [15.0, -62.0, 18.0]],
                [[-141.0, 97.0, -33.0], [10.0, -7.0, 12.0], [-1.0, -6.0, 2.0]],
                [[134.0, -96.0, 34.0], [13.0, 8.0, -35.0], [-5.0, 30.0, -9.0]],
            ],
            [
                [[-87.0, 62.0, -22.0], [-6.0, -5.0, 20.0], [3.0, -17.0, 5.0]],
                [[116.0, -81.0, 28.0], [-4.0, 6.0, -14.0], [-1.0, 10.0, -3.0]],
                [[-59.0, 41.0, -14.0], [4.0, -3.0, 5.0], [0.0, -3.0, 1.0]],
            ],
        ];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(aa_inv_std[i][j][k][l], aa_inv_std_expected[i][j][k][l], 1e-10);
                        let mut sum = 0.0;
                        for p in 0..3 {
                            for q in 0..3 {
                                sum += aa_std[i][j][p][q] * aa_inv_std[p][q][k][l];
                            }
                        }
                        if i == k && j == l {
                            approx_eq(sum, 1.0, 1e-11);
                        } else {
                            approx_eq(sum, 0.0, 1e-11);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn as_std_array_and_to_std_array_work() {
        // general
        let dd = Tensor4::<9>::from_std_array(&SamplesTensor4::SAMPLE1).unwrap();
        let res = dd.as_std_array();
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
        let dd = Tensor4::<6>::from_std_array(&SamplesTensor4::SYM_SAMPLE1).unwrap();
        let res = dd.as_std_array();
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
        let dd = Tensor4::<4>::from_std_array(&SamplesTensor4::SYM_2D_SAMPLE1).unwrap();
        let res = dd.as_std_array();
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
    fn as_std_matrix_and_to_std_matrix_work() {
        // general
        let dd = Tensor4::<9>::from_std_array(&SamplesTensor4::SAMPLE1).unwrap();
        let mat = dd.as_std_matrix();
        for m in 0..9 {
            for n in 0..9 {
                approx_eq(mat.get(m, n), SamplesTensor4::SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // symmetric 3D
        let dd = Tensor4::<6>::from_std_array(&SamplesTensor4::SYM_SAMPLE1).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (9, 9));
        for m in 0..9 {
            for n in 0..9 {
                approx_eq(mat.get(m, n), SamplesTensor4::SYM_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // symmetric 2D
        let dd = Tensor4::<4>::from_std_array(&SamplesTensor4::SYM_2D_SAMPLE1).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (9, 9));
        for m in 0..9 {
            for n in 0..9 {
                approx_eq(mat.get(m, n), SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }
    }

    #[test]
    fn from_std_array_to_std_matrix_from_std_matrix_work() {
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
        let dd = Tensor4::<9>::from_std_array(data).unwrap();
        let m1 = dd.as_std_matrix();
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
        let ee = Tensor4::<9>::from_std_matrix(correct).unwrap();
        let m2 = ee.as_std_matrix();
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
        let dd = Tensor4::<6>::from_std_array(data).unwrap();
        let m1 = dd.as_std_matrix();
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
        let ee = Tensor4::<6>::from_std_matrix(correct).unwrap();
        let m2 = ee.as_std_matrix();
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
        let dd = Tensor4::<4>::from_std_array(data).unwrap();
        let m1 = dd.as_std_matrix();
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
        let ee = Tensor4::<4>::from_std_matrix(correct).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, correct, 1e-13);
    }

    #[test]
    #[should_panic(expected = "N != 9")]
    fn sym_set_std_panics_on_non_sym() {
        let mut dd = Tensor4::<9>::new();
        dd.sym_set_std(0, 0, 0, 0, 1.0);
    }

    #[test]
    #[should_panic(expected = "the len is 3 but the index is 3")]
    fn sym_set_std_panics_on_incorrect_indices() {
        let mut dd = Tensor4::<4>::new();
        dd.sym_set_std(0, 0, 0, 3, 5.0);
    }

    #[test]
    fn sym_set_std_works() {
        let mut dd = Tensor4::<6>::new();
        for m in 0..6 {
            for n in 0..6 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                let value = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
                dd.sym_set_std(i, j, k, l, value);
            }
        }
        assert_eq!(
            format!("{:.0}", dd.as_std_matrix()),
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
    fn set_tensor_works() {
        #[rustfmt::skip]
        let dd = Tensor4::<9>::from_std_matrix(&[
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        ]).unwrap();
        let mut ee = Tensor4::<9>::new();
        ee.set_tensor(2.0, &dd);
        #[rustfmt::skip]
        let correct = Matrix::from(&[
            [ 2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0],
            [ 4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0],
            [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            [ 6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0],
            [ 4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0],
            [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            [ 6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0],
        ]);
        mat_approx_eq(&ee.as_std_matrix(), &correct, 1e-14);
    }

    #[test]
    fn clone_and_serialize_work() {
        let mut dd = Tensor4::<6>::new();
        for m in 0..6 {
            for n in 0..6 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                let value = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
                dd.sym_set_std(i, j, k, l, value);
            }
        }
        // clone
        let mut cloned = dd.clone();
        cloned.set(0, 0, 9999.0);
        assert_eq!(
            format!("{:.0}", dd.as_std_matrix()),
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
            format!("{:.0}", cloned.as_std_matrix()),
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
        let json = serde_json::to_string(&dd).unwrap();
        assert!(json.len() > 0);
        // deserialize
        let from_json: Tensor4<6> = serde_json::from_str(&json).unwrap();
        assert_eq!(
            format!("{:.0}", from_json.as_std_matrix()),
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
        let dd = Tensor4::<9>::new();
        assert!(format!("{:?}", dd).len() > 0);
    }

    #[test]
    fn constant_ii_works() {
        let ii = Tensor4::<9>::constant_ii();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(ii.get(m, n), IDENTITY4[m][n]);
            }
        }
        assert_eq!(
            format!("{}", ii),
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
        assert_eq!(
            format!("{:.1}", ii),
            "┌                                     ┐\n\
             │ 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             │ 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             │ 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 │\n\
             └                                     ┘"
        );
    }

    #[test]
    fn constant_tt_works() {
        let tt = Tensor4::<9>::constant_tt();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(tt.get(m, n), TRANSPOSITION[m][n]);
            }
        }
    }

    #[test]
    fn constant_jj_works() {
        let jj = Tensor4::<9>::constant_jj();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(jj.get(m, n), TRACE_PROJECTION[m][n]);
            }
        }
        let jj = Tensor4::<6>::constant_jj();
        for m in 0..6 {
            for n in 0..6 {
                assert_eq!(jj.get(m, n), TRACE_PROJECTION[m][n]);
            }
        }
    }

    #[test]
    fn constant_pp_iso_works() {
        let pp_iso = Tensor4::<9>::constant_pp_iso();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(pp_iso.get(m, n), P_ISO[m][n]);
            }
        }
        let pp_iso = Tensor4::<6>::constant_pp_iso();
        for m in 0..6 {
            for n in 0..6 {
                assert_eq!(pp_iso.get(m, n), P_ISO[m][n]);
            }
        }
    }

    #[test]
    fn constant_pp_sym_works() {
        let pp_sym = Tensor4::<9>::constant_pp_sym();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(pp_sym.get(m, n), P_SYM[m][n]);
            }
        }
        let pp_sym = Tensor4::<6>::constant_pp_sym();
        for m in 0..6 {
            for n in 0..6 {
                assert_eq!(pp_sym.get(m, n), P_SYM[m][n]);
            }
        }
    }

    #[test]
    fn constant_pp_skew_works() {
        let pp_skew = Tensor4::<9>::constant_pp_skew();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(pp_skew.get(m, n), P_SKEW[m][n]);
            }
        }
    }

    #[test]
    fn constant_pp_dev_works() {
        let pp_dev = Tensor4::<9>::constant_pp_dev();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(pp_dev.get(m, n), P_DEV[m][n]);
            }
        }
    }

    #[test]
    fn constant_pp_symdev_works() {
        let pp_symdev = Tensor4::<9>::constant_pp_symdev();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(pp_symdev.get(m, n), P_SYMDEV[m][n]);
            }
        }
        let pp_symdev = Tensor4::<6>::constant_pp_symdev();
        for m in 0..6 {
            for n in 0..6 {
                assert_eq!(pp_symdev.get(m, n), P_SYMDEV[m][n]);
            }
        }
    }

    #[test]
    fn set_pp_symdev_works() {
        let mut pp_symdev = Tensor4::<9>::new();
        pp_symdev.set_pp_symdev();
        for m in 0..9 {
            for n in 0..9 {
                assert_eq!(pp_symdev.get(m, n), P_SYMDEV[m][n]);
            }
        }
        let mut pp_symdev = Tensor4::<6>::new();
        pp_symdev.set_pp_symdev();
        for m in 0..6 {
            for n in 0..6 {
                assert_eq!(pp_symdev.get(m, n), P_SYMDEV[m][n]);
            }
        }
    }
}
