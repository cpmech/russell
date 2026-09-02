use super::{
    IJK_TO_MN_CASE_A, IJK_TO_MN_CASE_B, IJK_TO_MN_SYM_CASE_A, IJK_TO_MN_SYM_CASE_B, MN_TO_IJK_CASE_A, MN_TO_IJK_CASE_B,
    SQRT_2,
};
use crate::StrError;
use russell_lab::{AsArray2D, Matrix};
use serde::{Deserialize, Serialize};
use std::cmp;
use std::fmt::{self, Write};

#[cfg(feature = "heap")]
use russell_lab::mat_scale;

/// Defines a third-order tensor in R³×R³×R³
///
/// The matrix representation of Tensor3 results in a rectangular matrix.
/// Therefore, two matrices with max dimensions DIM×3 or 3×DIM are considered here,
/// where DIM (the leading dimension) is one of 4, 6, or 9. For a third-order tensor
/// with indices ijk, the cases are:
///
/// Case A: ij-pairwise => (ij)k => (m)k => (DIM×3)
/// Case B: jk-pairwise => i(jk) => i(n) => (3×DIM)
///
/// Given u, T, and H as first-, second-, and third-order tensors, the
/// main operations involving a third-order tensor are:
///
/// ```text
/// Case A (ij)k =>  T = H · u   or   u = T : H
/// Case B i(jk) =>  u = H : T   or   T = u · H
/// ```
///
/// In index notation (with i,j,k = 1...3):
///
/// ```text
/// Case A (ij)k =>  Tᵢⱼ = Σ H₍ᵢⱼ₎ₖ uₖ      or  uₖ = Σ Σ Tᵢⱼ H₍ᵢⱼ₎ₖ
///                       k                        i j
/// Case B i(jk) =>  uᵢ = Σ Σ Hᵢ₍ⱼₖ₎ Tⱼₖ  or  Tⱼₖ = Σ uᵢ Hᵢ₍ⱼₖ₎
///                      j k                      i
/// ```
///
/// The matrix representations associated with the two cases are
/// (with m,n = 1...DIM and DIM = {4,6,9}):
///
/// ```text
/// Case A (m)k =>  Tₘ = Σ H₍ₘ₎ₖ uₖ  or  uₖ = Σ Tₘ H₍ₘ₎ₖ
///                      k                    m
/// Case B i(n) =>  uᵢ = Σ Hᵢ₍ₙ₎ Tₙ   or  Tₙ = Σ uᵢ Hᵢ₍ₙ₎
///                     n                    i
/// ```
///
/// Note that the first-order tensors (vectors) are always given by the standard
/// components in 3D. All functions here require vectors such as `[u] = {u0, u1, u2}`.
///
/// # Standard and Kelvin-Mandel components
///
/// The methods of this struct follow a naming convention that distinguishes
/// between the **standard** (Cartesian) components `Hᵢⱼₖ` and the **Kelvin-Mandel**
/// components stored internally:
///
/// * Methods dealing with **standard components** carry the `std` qualifier in
///   their names (e.g., [Tensor3::from_std_matrix], [Tensor3::get_std],
///   [Tensor3::as_std_matrix], [Tensor3::sym_set_std]).
/// * Methods dealing directly with the **Kelvin-Mandel components** carry no qualifier
///   (e.g., [Tensor3::get], [Tensor3::set], [Tensor3::set_tensor],
///   [Tensor3::update]).
///
/// Internally, the components are converted to the Kelvin-Mandel basis as follows.
///
/// The Kelvin-Mandel components Ĥijk are calculated from the standard components Hijk
/// using the following expression for Case A:
///
/// ```text
/// Case A:
///        ⎧ Hijk                if i = j
/// Ĥijk = ⎨ (Hijk + Hjik) / √2  if i < j
///        ⎩ (Hjik - Hijk) / √2  if i > j
/// ```
///
/// The Kelvin-Mandel components Ĥijk are calculated from the standard components Hijk
/// using the following expression for Case B:
///
/// ```text
/// Case B:
///        ⎧ Hijk                if j = k
/// Ĥijk = ⎨ (Hijk + Hikj) / √2  if j < k
///        ⎩ (Hikj - Hijk) / √2  if j > k
/// ```
///
/// In Case A, minor-symmetry means Hijk = Hjik. Then, the mapping simplifies to:
///
/// ```text
/// Case A:
///        ⎧ Hijk     if i = j
/// Ĥijk = ⎨ Hijk √2  if i < j
///        ⎩ 0        if i > j
/// ```
///
/// In Case B, minor-symmetry means Hijk = Hikj. Then, the mapping simplifies to:
///
/// ```text
/// Case B:
///        ⎧ Hijk        if j = k
/// Ĥijk = ⎨ Hijk √2  if j < k
///        ⎩ 0        if j > k
/// ```
///
/// The components are organized in matrices:
/// * For Case A, the order of row indices, pairs (i,j) in (i,j,k), follow the same order used for Tensor2.
/// * For Case B, the order of column indices, pairs (j,k) in (i,j,k), follow the same order as the one for Tensor2.
///
/// The matrices are illustrated as follows.
///
/// max(DIM) = 9:
///
/// ```text
/// Case A:
///      0 0   0 1   0 2
///    -----------------
/// 0 │ Ĥ000  Ĥ001  Ĥ002
/// 1 │ Ĥ110  Ĥ111  Ĥ112
/// 2 │ Ĥ220  Ĥ221  Ĥ222
///   │
/// 3 │ Ĥ010  Ĥ011  Ĥ012
/// 4 │ Ĥ120  Ĥ121  Ĥ122
/// 5 │ Ĥ020  Ĥ021  Ĥ022
///   │
/// 6 │ Ĥ100  Ĥ101  Ĥ102
/// 7 │ Ĥ210  Ĥ211  Ĥ212
/// 8 │ Ĥ200  Ĥ201  Ĥ202
///    -----------------
///      8 0   8 1   8 2
/// ```
///
/// ```text
/// Case B:
///      0 0  0 1  0 2  0 3  0 4  0 5  0 6  0 7  0 8
///    ---------------------------------------------
/// 0 │ Ĥ000 Ĥ011 Ĥ022 Ĥ001 Ĥ012 Ĥ002 Ĥ010 Ĥ021 Ĥ020
/// 1 │ Ĥ100 Ĥ111 Ĥ122 Ĥ101 Ĥ112 Ĥ102 Ĥ110 Ĥ121 Ĥ120
/// 2 │ Ĥ200 Ĥ211 Ĥ222 Ĥ201 Ĥ212 Ĥ202 Ĥ210 Ĥ221 Ĥ220
///    ---------------------------------------------
///      2 0  2 1  2 2  2 3  2 4  2 5  2 6  2 7  2 8
/// ```
///
/// max(DIM) = 6:
///
/// ```text
/// Case A:
///      0 0      0 1      0 2    
///    --------------------------
/// 0 │ H000     H001     H002
/// 1 │ H110     H111     H112
/// 2 │ H220     H221     H222
///   │
/// 3 │ H010*√2  H011*√2  H012*√2
/// 4 │ H120*√2  H121*√2  H122*√2
/// 5 │ H020*√2  H021*√2  H022*√2
///    --------------------------
///      5 0      5 1      5 2
/// ```
///
/// ```text
/// Case B:
///      0 0  0 1  0 2  0 3     0 4     0 5
///    ---------------------------------------
/// 0 │ H000 H011 H022 H001*√2 H012*√2 H002*√2
/// 1 │ H100 H111 H122 H101*√2 H112*√2 H102*√2
/// 2 │ H200 H211 H222 H201*√2 H212*√2 H202*√2
///    ---------------------------------------
///      2 0  2 1  2 2  2 3     2 4     2 5
/// ```
///
/// max(DIM) = 4:
///
/// ```text
/// Case A:
///      0 0      0 1      0 2    
///    --------------------------
/// 0 │ H000     H001     H002   
/// 1 │ H110     H111     H112   
/// 2 │ H220     H221     H222   
///   │
/// 3 │ H010*√2  H011*√2  H012*√2
///    --------------------------
///      3 0      3 1      3 2
/// ```
///
/// ```text
/// Case B:
///      0 0  0 1  0 2  0 3   
///    -----------------------
/// 0 │ H000 H011 H022 H001*√2
/// 1 │ H100 H111 H122 H101*√2
/// 2 │ H200 H211 H222 H201*√2
///    -----------------------
///      2 0  2 1  2 2  2 3
/// ```
#[derive(Clone, Debug)]
pub struct Tensor3<const M: usize, const N: usize> {
    /// Holds the components in Kelvin-Mandel basis as matrix (heap).
    ///
    /// Heap version => dynamically allocated memory
    #[cfg(feature = "heap")]
    pub(crate) mat: Matrix,

    /// Holds the components in Kelvin-Mandel basis as matrix (stack).
    ///
    /// Stack version => fixed size memory
    ///
    /// This array may use more data than necessary in symmetric cases
    #[cfg(not(feature = "heap"))]
    pub(crate) mat: [[f64; N]; M],
}

// Manual Serialize/Deserialize implementations: serde only implements the traits
// for concrete array sizes, so the derive fails for the generic `[[f64; N]; N]`.
// Since N is known to be 4, 6, or 9 only, we serialize `(case_a, components)` where
// the components are the active `nrow x ncol` Kelvin-Mandel block.
impl<const M: usize, const N: usize> Serialize for Tensor3<M, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut data = Vec::with_capacity(M * N);
        for m in 0..M {
            for n in 0..N {
                data.push(self.get(m, n));
            }
        }
        data.serialize(serializer)
    }
}

impl<'de, const M: usize, const N: usize> Deserialize<'de> for Tensor3<M, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = Vec::<f64>::deserialize(deserializer)?;
        let expected = M * N;
        if data.len() != expected {
            return Err(serde::de::Error::custom(format!(
                "Tensor3 dimension mismatch: expected {} components, got {}",
                expected,
                data.len()
            )));
        }
        let mut dd = Tensor3::new();
        let mut k = 0;
        for m in 0..M {
            for n in 0..N {
                dd.set(m, n, data[k]);
                k += 1;
            }
        }
        Ok(dd)
    }
}

impl<const M: usize, const N: usize> Tensor3<M, N> {
    // Case A: M = {4,6,9}, N = 3
    // Case B: M = 3, N = {4,6,9}
    const VALIDATE_DIM: () = assert!(
        ((M == 4 || M == 6 || M == 9) && N == 3) || (M == 3 && (N == 4 || N == 6 || N == 9)),
        "Tensor dimension must be such that (DIM,3) for Case A or (3,DIM) for case B with DIM = 4, 6, or 9."
    );

    /// Creates a new (zeroed) Tensor3
    pub fn new() -> Self {
        let _ = Self::VALIDATE_DIM;

        #[cfg(feature = "heap")]
        {
            Tensor3 { mat: Matrix::new(M, N) }
        }
        #[cfg(not(feature = "heap"))]
        {
            Tensor3 { mat: [[0.0; N]; M] }
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
    /// use russell_tensor::{Tensor3};
    ///
    /// let mut dd = Tensor3::<9, 3>::new();
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
    /// use russell_tensor::{Tensor3};
    ///
    /// let mut dd = Tensor3::<9, 3>::new();
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
    /// use russell_tensor::{Tensor3};
    ///
    /// let mut dd = Tensor3::<9, 3>::new();
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

    /// Sets this tensor from a nested array containing the standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard Dijk components with respect to a Cartesian system
    pub fn set_std_array(&mut self, inp: &[[[f64; 3]; 3]; 3]) -> Result<(), StrError> {
        if M > N {
            // Case A: (M, 3) with M = 4,6,9
            if M == 4 || M == 6 {
                let max = if M == 4 { 3 } else { 6 };
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            // check minor-symmetry on (i,j)
                            if i > j {
                                if inp[i][j][k] != inp[j][i][k] {
                                    return Err("the input data does not correspond to a minor-symmetric tensor");
                                }
                            } else {
                                let (m, n) = IJK_TO_MN_CASE_A[i][j][k];
                                if m > max {
                                    if inp[i][j][k] != 0.0 {
                                        return Err(
                                            "the input data does not correspond to a 2D minor-symmetric tensor",
                                        );
                                    }
                                    continue;
                                } else if m < 3 {
                                    self.set(m, n, inp[i][j][k]);
                                } else {
                                    self.set(m, n, SQRT_2 * inp[i][j][k]);
                                }
                            }
                        }
                    }
                }
            } else {
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_A[i][j][k];
                            // ** i == j **
                            if i == j {
                                self.set(m, n, inp[i][j][k]);
                            // ** i < j **
                            } else if i < j {
                                self.set(m, n, (inp[i][j][k] + inp[j][i][k]) / SQRT_2);
                            // ** i > j **
                            } else if i > j {
                                self.set(m, n, (inp[j][i][k] - inp[i][j][k]) / SQRT_2);
                            }
                        }
                    }
                }
            }
        } else {
            // Case B: (3, N) with N = 4,6,9
            if N == 4 || N == 6 {
                let max = if N == 4 { 3 } else { 6 };
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            // check minor-symmetry on (j,k)
                            if j > k {
                                if inp[i][j][k] != inp[i][k][j] {
                                    return Err("the input data does not correspond to a minor-symmetric tensor");
                                }
                            } else {
                                let (m, n) = IJK_TO_MN_CASE_B[i][j][k];
                                if n > max {
                                    if inp[i][j][k] != 0.0 {
                                        return Err(
                                            "the input data does not correspond to a 2D minor-symmetric tensor",
                                        );
                                    }
                                    continue;
                                } else if n < 3 {
                                    self.set(m, n, inp[i][j][k]);
                                } else {
                                    self.set(m, n, SQRT_2 * inp[i][j][k]);
                                }
                            }
                        }
                    }
                }
            } else {
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_B[i][j][k];
                            // ** j == k **
                            if j == k {
                                self.set(m, n, inp[i][j][k]);
                            // ** j < k **
                            } else if j < k {
                                self.set(m, n, (inp[i][j][k] + inp[i][k][j]) / SQRT_2);
                            // ** j > k **
                            } else if j > k {
                                self.set(m, n, (inp[i][k][j] - inp[i][j][k]) / SQRT_2);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Creates a new Tensor3 constructed from a nested array containing the standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard Dijk components with respect to a Cartesian system
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[[0.0; 3]; 3]; 3];
    ///     for i in 0..3 {
    ///         for j in 0..3 {
    ///             for k in 0..3 {
    ///                 inp[i][j][k] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             }
    ///         }
    ///     }
    ///     let dd = Tensor3::<9, 3>::from_std_array(&inp)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌             ┐\n\
    ///          │ 111 112 113 │\n\
    ///          │ 221 222 223 │\n\
    ///          │ 331 332 333 │\n\
    ///          │ 121 122 123 │\n\
    ///          │ 231 232 233 │\n\
    ///          │ 131 132 133 │\n\
    ///          │ 211 212 213 │\n\
    ///          │ 321 322 323 │\n\
    ///          │ 311 312 313 │\n\
    ///          └             ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn from_std_array(inp: &[[[f64; 3]; 3]; 3]) -> Result<Self, StrError> {
        let mut res = Tensor3::new();
        res.set_std_array(inp)?;
        Ok(res)
    }

    /// Sets this tensor from a matrix with standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard matrix of components with respect to a
    ///   Cartesian system. The matrix must be 9x3 for Case A or
    ///   3x9 for Case B even if it corresponds to a minor-symmetric tensor.
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix has the incorrect dimensions:
    /// * Case A: 9x3 required
    /// * Case B: 3x9 required
    pub fn set_std_matrix<'a, S>(&mut self, inp: &'a S) -> Result<(), StrError>
    where
        S: AsArray2D<'a, f64>,
    {
        if M > N {
            // Case A: (M, 3) with M = 4,6,9
            if M == 4 || M == 6 {
                let max = if M == 4 { 3 } else { 6 };
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_A[i][j][k];
                            let (r, s) = IJK_TO_MN_CASE_A[j][i][k];
                            // check minor-symmetry
                            if i > j {
                                if inp.at(m, n) != inp.at(r, s) {
                                    return Err("the input data does not correspond to a minor-symmetric tensor");
                                }
                            } else {
                                if m > max {
                                    if inp.at(m, n) != 0.0 {
                                        return Err(
                                            "the input data does not correspond to a 2D minor-symmetric tensor",
                                        );
                                    }
                                    continue;
                                } else if m < 3 {
                                    self.set(m, n, inp.at(m, n));
                                } else {
                                    self.set(m, n, SQRT_2 * inp.at(m, n));
                                }
                            }
                        }
                    }
                }
            } else {
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_A[i][j][k];
                            // ** i == j **
                            if i == j {
                                self.set(m, n, inp.at(m, n));
                            // ** i < j **
                            } else if i < j {
                                let (r, s) = IJK_TO_MN_CASE_A[j][i][k];
                                self.set(m, n, (inp.at(m, n) + inp.at(r, s)) / SQRT_2);
                            // ** i > j **
                            } else if i > j {
                                let (r, s) = IJK_TO_MN_CASE_A[j][i][k];
                                self.set(m, n, (inp.at(r, s) - inp.at(m, n)) / SQRT_2);
                            }
                        }
                    }
                }
            }
        } else {
            // Case B: (3, N) with N = 4,6,9
            if N == 4 || N == 6 {
                let max = if N == 4 { 3 } else { 6 };
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_B[i][j][k];
                            let (r, s) = IJK_TO_MN_CASE_B[i][k][j];
                            // check minor-symmetry
                            if j > k {
                                if inp.at(m, n) != inp.at(r, s) {
                                    return Err("the input data does not correspond to a minor-symmetric tensor");
                                }
                            } else {
                                if n > max {
                                    if inp.at(m, n) != 0.0 {
                                        return Err(
                                            "the input data does not correspond to a 2D minor-symmetric tensor",
                                        );
                                    }
                                    continue;
                                } else if n < 3 {
                                    self.set(m, n, inp.at(m, n));
                                } else {
                                    self.set(m, n, SQRT_2 * inp.at(m, n));
                                }
                            }
                        }
                    }
                }
            } else {
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            let (m, n) = IJK_TO_MN_CASE_B[i][j][k];
                            // ** j == k **
                            if j == k {
                                self.set(m, n, inp.at(m, n));
                            // ** j < k **
                            } else if j < k {
                                let (r, s) = IJK_TO_MN_CASE_B[i][k][j];
                                self.set(m, n, (inp.at(m, n) + inp.at(r, s)) / SQRT_2);
                            // ** j > k **
                            } else if j > k {
                                let (r, s) = IJK_TO_MN_CASE_B[i][k][j];
                                self.set(m, n, (inp.at(r, s) - inp.at(m, n)) / SQRT_2);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Creates a new Tensor3 constructed from a matrix with standard components
    ///
    /// # Input
    ///
    /// * `inp` -- the standard matrix of components with respect to a
    ///   Cartesian system. The matrix must be 9x3 for Case A or
    ///   3x9 for Case B even if it corresponds to a minor-symmetric tensor.
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix has the incorrect dimensions:
    /// * Case A: 9x3 required
    /// * Case B: 3x9 required
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor3::<9, 3>::from_std_matrix(&inp)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌             ┐\n\
    ///          │ 111 112 113 │\n\
    ///          │ 221 222 223 │\n\
    ///          │ 331 332 333 │\n\
    ///          │ 121 122 123 │\n\
    ///          │ 231 232 233 │\n\
    ///          │ 131 132 133 │\n\
    ///          │ 211 212 213 │\n\
    ///          │ 321 322 323 │\n\
    ///          │ 311 312 313 │\n\
    ///          └             ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn from_std_matrix<'a, S>(inp: &'a S) -> Result<Self, StrError>
    where
        S: AsArray2D<'a, f64>,
    {
        let mut res = Tensor3::new();
        res.set_std_matrix(inp)?;
        Ok(res)
    }

    /// Returns the (i,j,k) standard component
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor3::<9, 3>::from_std_matrix(&inp)?;
    ///
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             let val = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             approx_eq(dd.get_std(i,j,k), val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn get_std(&self, i: usize, j: usize, k: usize) -> f64 {
        if M > N {
            // Case A: (M, 3) with M = 4,6,9
            match M {
                4 => {
                    let (m, n) = IJK_TO_MN_SYM_CASE_A[i][j][k];
                    if m > 3 {
                        0.0
                    } else if m < 3 {
                        self.get(m, n)
                    } else {
                        self.get(m, n) / SQRT_2
                    }
                }
                6 => {
                    let (m, n) = IJK_TO_MN_SYM_CASE_A[i][j][k];
                    if m < 3 { self.get(m, n) } else { self.get(m, n) / SQRT_2 }
                }
                _ => {
                    let (m, n) = IJK_TO_MN_CASE_A[i][j][k];
                    let val = self.get(m, n);
                    // ** i == j **
                    if i == j {
                        val
                    // ** i < j **
                    } else if i < j {
                        let (r, s) = IJK_TO_MN_CASE_A[j][i][k];
                        let other = self.get(r, s);
                        (val + other) / SQRT_2
                    // ** i > j **
                    } else {
                        let (r, s) = IJK_TO_MN_CASE_A[j][i][k];
                        let other = self.get(r, s);
                        (other - val) / SQRT_2
                    }
                }
            }
        } else {
            // Case B: (3, N) with N = 4,6,9
            match N {
                4 => {
                    let (m, n) = IJK_TO_MN_SYM_CASE_B[i][j][k];
                    if n > 3 {
                        0.0
                    } else if n < 3 {
                        self.get(m, n)
                    } else {
                        self.get(m, n) / SQRT_2
                    }
                }
                6 => {
                    let (m, n) = IJK_TO_MN_SYM_CASE_B[i][j][k];
                    if n < 3 { self.get(m, n) } else { self.get(m, n) / SQRT_2 }
                }
                _ => {
                    let (m, n) = IJK_TO_MN_CASE_B[i][j][k];
                    let val = self.get(m, n);
                    // ** j == k **
                    if j == k {
                        val
                    // ** j < k **
                    } else if j < k {
                        let (r, s) = IJK_TO_MN_CASE_B[i][k][j];
                        let other = self.get(r, s);
                        (val + other) / SQRT_2
                    // ** j > k **
                    } else {
                        let (r, s) = IJK_TO_MN_CASE_B[i][k][j];
                        let other = self.get(r, s);
                        (other - val) / SQRT_2
                    }
                }
            }
        }
    }

    /// Calculates the Euclidean norm
    ///
    /// ```text
    /// norm(H) = √(H:H)
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
    /// use russell_tensor::{Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // the permutation (Levi-Civita) tensor has norm = √6
    ///     let dd = Tensor3::<9, 3>::constant_permutation();
    ///     approx_eq(dd.norm(), f64::sqrt(6.0), 1e-13);
    ///     Ok(())
    /// }
    /// ```
    pub fn norm(&self) -> f64 {
        let mut sm = 0.0;
        for m in 0..M {
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
    /// use russell_tensor::{Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut dd = Tensor3::<9, 3>::new();
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
            for m in 0..M {
                for n in 0..N {
                    self.mat[m][n] *= alpha;
                }
            }
        }
    }

    /// Adds another tensor to this one
    ///
    /// ```text
    /// self += α other
    /// ```
    ///
    /// # Panics
    ///
    /// A panic will occur if the tensors have different `case_a`.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..4 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = 1.0;
    ///         }
    ///     }
    ///
    ///     let mut dd = Tensor3::<9, 3>::new();
    ///     let ee = Tensor3::<9, 3>::from_std_matrix(&inp)?;
    ///     dd.update(2.0, &ee);
    ///
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌       ┐\n\
    ///          │ 2 2 2 │\n\
    ///          │ 2 2 2 │\n\
    ///          │ 2 2 2 │\n\
    ///          │ 2 2 2 │\n\
    ///          │ 0 0 0 │\n\
    ///          │ 0 0 0 │\n\
    ///          │ 0 0 0 │\n\
    ///          │ 0 0 0 │\n\
    ///          │ 0 0 0 │\n\
    ///          └       ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn update(&mut self, alpha: f64, other: &Tensor3<M, N>) {
        for m in 0..M {
            for n in 0..N {
                self.set(m, n, self.get(m, n) + alpha * other.get(m, n));
            }
        }
    }

    /// Returns a 3x3x3 array with the standard components
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor3::<9, 3>::from_std_matrix(&inp)?;
    ///     let arr = dd.as_std_array();
    ///
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             let val = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             approx_eq(arr[i][j][k], val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn as_std_array(&self) -> Vec<Vec<Vec<f64>>> {
        let mut dd = vec![vec![vec![0.0; 3]; 3]; 3];
        self.to_std_array(&mut dd);
        dd
    }

    /// Converts this tensor to a 3x3x3 array with the standard components
    ///
    /// # Panics
    ///
    /// A panic will occur if the array is not 3x3x3, i.e., `vec![vec![vec![0.0; 3]; 3]; 3]`
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///
    ///     let dd = Tensor3::<9, 3>::from_std_matrix(&inp)?;
    ///     let mut arr = vec![vec![vec![0.0; 3]; 3]; 3];
    ///     dd.to_std_array(&mut arr);
    ///
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             let val = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             approx_eq(arr[i][j][k], val, 1e-12);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn to_std_array(&self, dd: &mut Vec<Vec<Vec<f64>>>) {
        if M > N {
            // Case A: (M, 3) with M = 4,6,9
            if M == 9 {
                // General
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            dd[i][j][k] = self.get_std(i, j, k);
                        }
                    }
                }
            } else {
                // Symmetric / Symmetric2D
                for m in 0..M {
                    for n in 0..N {
                        let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
                        dd[i][j][k] = self.get_std(i, j, k);
                        if i != j {
                            dd[j][i][k] = dd[i][j][k];
                        }
                    }
                }
            }
        } else {
            // Case B: (3, N) with N = 4,6,9
            if N == 9 {
                // General
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            dd[i][j][k] = self.get_std(i, j, k);
                        }
                    }
                }
            } else {
                // Symmetric / Symmetric2D
                for m in 0..M {
                    for n in 0..N {
                        let (i, j, k) = MN_TO_IJK_CASE_B[m][n];
                        dd[i][j][k] = self.get_std(i, j, k);
                        if j != k {
                            dd[i][k][j] = dd[i][j][k];
                        }
                    }
                }
            }
        }
    }

    /// Returns a matrix with the standard components
    ///
    /// **Note:** The matrix will have the standard components.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor3::<9, 3>::from_std_matrix(&inp)?;
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌             ┐\n\
    ///          │ 111 112 113 │\n\
    ///          │ 221 222 223 │\n\
    ///          │ 331 332 333 │\n\
    ///          │ 121 122 123 │\n\
    ///          │ 231 232 233 │\n\
    ///          │ 131 132 133 │\n\
    ///          │ 211 212 213 │\n\
    ///          │ 321 322 323 │\n\
    ///          │ 311 312 313 │\n\
    ///          └             ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn as_std_matrix(&self) -> Matrix {
        // Note: always return the general matrix, (9,3) or (3,9)
        let mut mat = if M > N { Matrix::new(9, 3) } else { Matrix::new(3, 9) };
        self.to_std_matrix(&mut mat);
        mat
    }

    /// Converts this tensor to a matrix with the standard components
    ///
    /// # Input
    ///
    /// * `mat` -- the resulting matrix
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix has the incorrect dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Matrix;
    /// use russell_tensor::{MN_TO_IJK_CASE_A, Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut inp = [[0.0; 3]; 9];
    ///     for m in 0..9 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             inp[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///         }
    ///     }
    ///     let dd = Tensor3::<9, 3>::from_std_matrix(&inp)?;
    ///     let mut mat = Matrix::new(9, 3);
    ///     dd.to_std_matrix(&mut mat);
    ///     assert_eq!(
    ///         format!("{:.0}", mat),
    ///         "┌             ┐\n\
    ///          │ 111 112 113 │\n\
    ///          │ 221 222 223 │\n\
    ///          │ 331 332 333 │\n\
    ///          │ 121 122 123 │\n\
    ///          │ 231 232 233 │\n\
    ///          │ 131 132 133 │\n\
    ///          │ 211 212 213 │\n\
    ///          │ 321 322 323 │\n\
    ///          │ 311 312 313 │\n\
    ///          └             ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn to_std_matrix(&self, mat: &mut Matrix) {
        if M > N {
            // Case A: (M, 3) with M = 4,6,9
            assert_eq!(mat.dims(), (9, 3), "Matrix dimensions must be (9, 3), Case A");
            for m in 0..9 {
                for n in 0..3 {
                    let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
                    mat.set(m, n, self.get_std(i, j, k));
                }
            }
        } else {
            // Case B: (3, N) with N = 4,6,9
            assert_eq!(mat.dims(), (3, 9), "Matrix dimensions must be (3, 9), Case B");
            for m in 0..3 {
                for n in 0..9 {
                    let (i, j, k) = MN_TO_IJK_CASE_B[m][n];
                    mat.set(m, n, self.get_std(i, j, k));
                }
            }
        }
    }

    /// Sets the (i,j,k) standard component of a minor-symmetric Tensor3
    ///
    /// # Notes
    ///
    /// 1. The tensor must be symmetric and (i,j) must correspond to the possible
    ///    combination due to the space dimension, otherwise a panic may occur.
    ///
    /// # Panics
    ///
    /// 1. A panic will occur if the tensor is not symmetric; i.e., DIM = 9 instead of 4,6
    /// 2. A panic will occur if the indices are out of range
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{MN_TO_IJK_CASE_A, Tensor3};
    ///
    /// fn main() {
    ///     let mut dd = Tensor3::<4, 3>::new();
    ///     for m in 0..4 {
    ///         for n in 0..3 {
    ///             let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
    ///             let value = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
    ///             dd.sym_set_std(i, j, k, value);
    ///         }
    ///     }
    ///     assert_eq!(
    ///         format!("{:.0}", dd.as_std_matrix()),
    ///         "┌             ┐\n\
    ///          │ 111 112 113 │\n\
    ///          │ 221 222 223 │\n\
    ///          │ 331 332 333 │\n\
    ///          │ 121 122 123 │\n\
    ///          │   0   0   0 │\n\
    ///          │   0   0   0 │\n\
    ///          │ 121 122 123 │\n\
    ///          │   0   0   0 │\n\
    ///          │   0   0   0 │\n\
    ///          └             ┘"
    ///     );
    /// }
    /// ```
    pub fn sym_set_std(&mut self, i: usize, j: usize, k: usize, value: f64) {
        if M > N {
            // Case A: (M, 3) with M = 4,6,9
            assert!(M != 9, "minor-symmetric case A requires M = 4,6");
            let (m, n) = IJK_TO_MN_SYM_CASE_A[i][j][k];
            if m < 3 {
                self.set(m, n, value);
            } else {
                self.set(m, n, value * SQRT_2);
            }
        } else {
            // Case B: (3, N) with N = 4,6,9
            assert!(N != 9, "minor-symmetric case B requires N = 4,6");
            let (m, n) = IJK_TO_MN_SYM_CASE_B[i][j][k];
            if n < 3 {
                self.set(m, n, value);
            } else {
                self.set(m, n, value * SQRT_2);
            }
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
    /// use russell_tensor::{Tensor3, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let data = &[
    ///         [  1.0,  2.0,  3.0],
    ///         [ -1.0, -2.0, -3.0],
    ///         [  2.0,  4.0,  6.0],
    ///         [ 10.0, 20.0, 30.0],
    ///         [  0.0,  0.0,  0.0],
    ///         [  0.0,  0.0,  0.0],
    ///         [ -2.0, -4.0, -6.0],
    ///         [  0.0,  0.0,  0.0],
    ///         [  0.0,  0.0,  0.0],
    ///     ];
    ///     let dd = Tensor3::<9, 3>::from_std_matrix(data)?;
    ///     let mut ee = Tensor3::<9, 3>::new();
    ///
    ///     ee.set_tensor(1.0, &dd);
    ///
    ///     mat_approx_eq(&dd.as_std_matrix(), data, 1e-14);
    ///     Ok(())
    /// }
    /// ```
    pub fn set_tensor(&mut self, alpha: f64, other: &Tensor3<M, N>) {
        for m in 0..M {
            for n in 0..N {
                self.set(m, n, alpha * other.get(m, n));
            }
        }
    }

    /// Returns the permutation (Levi-Civita) tensor
    ///
    /// # Panics
    ///
    /// A panic will occur if `DIM != 9`, i.e., if the tensor is not general
    /// (Case A with `M = 9` or Case B with `N = 9`).
    pub fn constant_permutation() -> Self {
        assert!(
            M == 9 || N == 9,
            "the permutation (Levi-Civita) tensor requires DIM = 9"
        );
        let pos_one = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]; // even cyclic permutation
        let neg_one = [(0, 2, 1), (1, 0, 2), (2, 1, 0)]; // odd cyclic permutation
        let mut std_array = [[[0.0; 3]; 3]; 3];
        for (i, j, k) in pos_one {
            std_array[i][j][k] = 1.0;
        }
        for (i, j, k) in neg_one {
            std_array[i][j][k] = -1.0;
        }
        Tensor3::from_std_array(&std_array).unwrap()
    }
}

impl<const M: usize, const N: usize> fmt::Display for Tensor3<M, N> {
    /// Generates a string representation of Kelvin-Mandel matrix associated with this Tensor3
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..M {
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
        for i in 0..M {
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
    use super::{MN_TO_IJK_CASE_A, Tensor3};
    use crate::{SQRT_2, SamplesTensor3};
    use russell_lab::{Matrix, approx_eq, mat_approx_eq};

    // Computes the reference norm from the standard components
    fn norm_from_std_array(arr: &[[[f64; 3]; 3]; 3]) -> f64 {
        let mut sm = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    sm += arr[i][j][k] * arr[i][j][k];
                }
            }
        }
        f64::sqrt(sm)
    }

    #[test]
    fn norm_works() {
        // Case A
        let dd = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        approx_eq(dd.norm(), norm_from_std_array(&SamplesTensor3::CASE_A_SAMPLE1), 1e-13);
        let dd = Tensor3::<6, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1).unwrap();
        approx_eq(
            dd.norm(),
            norm_from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1),
            1e-13,
        );
        let dd = Tensor3::<4, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1).unwrap();
        approx_eq(
            dd.norm(),
            norm_from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1),
            1e-13,
        );
        // Case B
        let dd = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        approx_eq(dd.norm(), norm_from_std_array(&SamplesTensor3::CASE_B_SAMPLE1), 1e-13);
        let dd = Tensor3::<3, 6>::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1).unwrap();
        approx_eq(
            dd.norm(),
            norm_from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1),
            1e-13,
        );
        let dd = Tensor3::<3, 4>::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1).unwrap();
        approx_eq(
            dd.norm(),
            norm_from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1),
            1e-13,
        );
    }

    #[test]
    fn scale_works() {
        let mut dd = Tensor3::<9, 3>::new();
        dd.set(0, 0, 1.0);
        dd.set(1, 1, 2.0);
        dd.set(2, 2, 3.0);
        dd.scale(2.0);
        assert_eq!(dd.get(0, 0), 2.0);
        assert_eq!(dd.get(1, 1), 4.0);
        assert_eq!(dd.get(2, 2), 6.0);
    }

    #[test]
    fn new_set_and_get_work() {
        // general
        let mut dd = Tensor3::<9, 3>::new();
        dd.set(0, 0, 123.0);
        assert_eq!(dd.get(0, 0), 123.0);

        // symmetric
        let mut dd = Tensor3::<6, 3>::new();
        dd.set(0, 0, 123.0);
        assert_eq!(dd.get(0, 0), 123.0);

        // symmetric 2d
        let mut dd = Tensor3::<4, 3>::new();
        dd.set(0, 0, 123.0);
        assert_eq!(dd.get(0, 0), 123.0);
    }

    #[test]
    fn from_std_array_fails_captures_errors() {
        let res = Tensor3::<6, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );

        let res = Tensor3::<4, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn from_std_array_works() {
        // general
        let dd = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        for m in 0..9 {
            for n in 0..3 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_A_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 3d
        let dd = Tensor3::<6, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1).unwrap();
        for m in 0..6 {
            for n in 0..3 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_A_SYM_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 2d
        let dd = Tensor3::<4, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1).unwrap();
        for m in 0..4 {
            for n in 0..3 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_A_SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }
    }

    #[test]
    fn from_std_matrix_fails_captures_errors() {
        let mut inp = [[0.0; 3]; 9];
        inp[3][0] = 1e-15;
        let res = Tensor3::<6, 3>::from_std_matrix(&inp);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );

        inp[3][0] = 0.0;
        inp[4][0] = 1.0;
        inp[7][0] = 1.0;
        let res = Tensor3::<4, 3>::from_std_matrix(&inp);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn get_and_set_work() {
        let mut dd = Tensor3::<4, 3>::new();
        assert_eq!(dd.get(0, 0), 0.0);
        dd.set(0, 0, 2.0);
        assert_eq!(dd.get(0, 0), 2.0);
    }

    #[test]
    fn from_std_matrix_works() {
        // general
        let dd = Tensor3::<9, 3>::from_std_matrix(&SamplesTensor3::CASE_A_SAMPLE1_STD_MATRIX).unwrap();
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(dd.get(m, n), SamplesTensor3::CASE_A_SAMPLE1_KELVIN_MATRIX[m][n], 1e-15);
            }
        }

        // symmetric 3D
        let dd = Tensor3::<6, 3>::from_std_matrix(&SamplesTensor3::CASE_A_SYM_SAMPLE1_STD_MATRIX).unwrap();
        for m in 0..6 {
            for n in 0..3 {
                approx_eq(
                    dd.get(m, n),
                    SamplesTensor3::CASE_A_SYM_SAMPLE1_KELVIN_MATRIX[m][n],
                    1e-14,
                );
            }
        }

        // symmetric 2D
        let dd = Tensor3::<4, 3>::from_std_matrix(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1_STD_MATRIX).unwrap();
        for m in 0..4 {
            for n in 0..3 {
                approx_eq(
                    dd.get(m, n),
                    SamplesTensor3::CASE_A_SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n],
                    1e-14,
                );
            }
        }
    }

    #[test]
    fn get_std_works() {
        // general
        let dd = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::CASE_A_SAMPLE1[i][j][k], 1e-13);
                }
            }
        }

        // symmetric 3D
        let dd = Tensor3::<6, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::CASE_A_SYM_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }

        // symmetric 2D
        let dd = Tensor3::<4, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(
                        dd.get_std(i, j, k),
                        SamplesTensor3::CASE_A_SYM_2D_SAMPLE1[i][j][k],
                        1e-14,
                    );
                }
            }
        }
    }

    #[test]
    fn update_works() {
        let mut dd = Tensor3::<4, 3>::new();
        let ee = Tensor3::<4, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1).unwrap();
        dd.update(2.0, &ee);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(
                        dd.get_std(i, j, k),
                        2.0 * SamplesTensor3::CASE_A_SYM_2D_SAMPLE1[i][j][k],
                        1e-14,
                    );
                }
            }
        }
    }

    #[test]
    fn as_std_array_and_to_std_array_work() {
        // general
        let dd = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_A_SAMPLE1[i][j][k], 1e-13);
                }
            }
        }

        // symmetric 3D
        let dd = Tensor3::<6, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_A_SYM_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }

        // symmetric 2D
        let dd = Tensor3::<4, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_A_SYM_2D_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }
    }

    #[test]
    fn as_std_matrix_and_to_std_matrix_work() {
        // general
        let dd = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        let mat = dd.as_std_matrix();
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(mat.get(m, n), SamplesTensor3::CASE_A_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // symmetric 3D
        let dd = Tensor3::<6, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (9, 3));
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(
                    mat.get(m, n),
                    SamplesTensor3::CASE_A_SYM_SAMPLE1_STD_MATRIX[m][n],
                    1e-13,
                );
            }
        }

        // symmetric 2D
        let dd = Tensor3::<4, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (9, 3));
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(
                    mat.get(m, n),
                    SamplesTensor3::CASE_A_SYM_2D_SAMPLE1_STD_MATRIX[m][n],
                    1e-13,
                );
            }
        }
    }

    #[test]
    fn from_std_array_to_std_matrix_from_std_matrix_work() {
        // General
        #[rustfmt::skip]
        let data = &[
            [
                [ 18.0,  16.0,  14.0],
                [ 36.0,  32.0,  28.0],
                [ 54.0,  48.0,  42.0],
            ],
            [
                [ 72.0,  64.0,  56.0],
                [ 90.0,  80.0,  70.0],
                [108.0,  96.0,  84.0],
            ],
            [
                [126.0, 112.0,  98.0],
                [144.0, 128.0, 112.0],
                [162.0, 144.0, 126.0],
            ],
        ];
        let dd = Tensor3::<9, 3>::from_std_array(data).unwrap();
        let m1 = dd.as_std_matrix();
        #[rustfmt::skip]
        let correct = &[
            [ 18.0,  16.0,  14.0],
            [ 90.0,  80.0,  70.0],
            [162.0, 144.0, 126.0],
            [ 36.0,  32.0,  28.0],
            [108.0,  96.0,  84.0],
            [ 54.0,  48.0,  42.0],
            [ 72.0,  64.0,  56.0],
            [144.0, 128.0, 112.0],
            [126.0, 112.0,  98.0],
        ];
        mat_approx_eq(&m1, correct, 1e-13);
        let ee = Tensor3::<9, 3>::from_std_matrix(correct).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, correct, 1e-13);

        // Symmetric 3D
        #[rustfmt::skip]
        let data = &[
            [
                [ 6.0, 10.0, 12.0],
                [24.0, 40.0, 48.0],
                [36.0, 60.0, 72.0],
            ],
            [
                [24.0, 40.0, 48.0],
                [12.0, 20.0, 24.0],
                [30.0, 50.0, 60.0],
            ],
            [
                [36.0, 60.0, 72.0],
                [30.0, 50.0, 60.0],
                [18.0, 30.0, 36.0],
            ],
        ];
        let dd = Tensor3::<6, 3>::from_std_array(data).unwrap();
        let m1 = dd.as_std_matrix();
        #[rustfmt::skip]
        let correct = &[
            [ 6.0, 10.0, 12.0],
            [12.0, 20.0, 24.0],
            [18.0, 30.0, 36.0],
            [24.0, 40.0, 48.0],
            [30.0, 50.0, 60.0],
            [36.0, 60.0, 72.0],
            [24.0, 40.0, 48.0],
            [30.0, 50.0, 60.0],
            [36.0, 60.0, 72.0],
        ];
        mat_approx_eq(&m1, correct, 1e-13);
        let ee = Tensor3::<6, 3>::from_std_matrix(correct).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, correct, 1e-13);

        // Symmetric 2D
        #[rustfmt::skip]
        let data = &[
            [
                [ 6.0,  8.0, 0.0],
                [24.0, 32.0, 0.0],
                [ 0.0,  0.0, 0.0],
            ],
            [
                [24.0, 32.0, 0.0],
                [12.0, 16.0, 0.0],
                [ 0.0,  0.0, 0.0],
            ],
            [
                [ 0.0,  0.0, 0.0],
                [ 0.0,  0.0, 0.0],
                [18.0, 24.0, 0.0],
            ],
        ];
        let dd = Tensor3::<4, 3>::from_std_array(data).unwrap();
        let m1 = dd.as_std_matrix();
        #[rustfmt::skip]
        let correct = &[
            [ 6.0,  8.0, 0.0],
            [12.0, 16.0, 0.0],
            [18.0, 24.0, 0.0],
            [24.0, 32.0, 0.0],
            [ 0.0,  0.0, 0.0],
            [ 0.0,  0.0, 0.0],
            [24.0, 32.0, 0.0],
            [ 0.0,  0.0, 0.0],
            [ 0.0,  0.0, 0.0],
        ];
        mat_approx_eq(&m1, correct, 1e-13);
        let ee = Tensor3::<4, 3>::from_std_matrix(correct).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, correct, 1e-13);
    }

    fn generate_dd_sym() -> Tensor3<6, 3> {
        let mut dd = Tensor3::new();
        for m in 0..6 {
            for n in 0..3 {
                let (i, j, k) = MN_TO_IJK_CASE_A[m][n];
                let value = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
                dd.sym_set_std(i, j, k, value);
            }
        }
        dd
    }

    #[test]
    #[should_panic(expected = "minor-symmetric case A requires M = 4,6")]
    fn sym_set_std_panics_on_non_sym() {
        let mut dd = Tensor3::<9, 3>::new();
        dd.sym_set_std(0, 0, 0, 0.0);
    }

    #[test]
    #[should_panic(expected = "the len is 3 but the index is 3")]
    fn sym_set_std_panics_on_incorrect_indices() {
        let mut dd = Tensor3::<4, 3>::new();
        dd.sym_set_std(0, 0, 3, 5.0);
    }

    #[test]
    fn sym_set_std_works() {
        let dd = generate_dd_sym();
        assert_eq!(
            format!("{:.0}", dd.as_std_matrix()),
            "┌             ┐\n\
             │ 111 112 113 │\n\
             │ 221 222 223 │\n\
             │ 331 332 333 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             └             ┘"
        );
    }

    #[test]
    fn set_tensor_works() {
        #[rustfmt::skip]
        let dd = Tensor3::<9,3>::from_std_matrix(&[
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [9.0, 9.0, 9.0],
            [2.0, 2.0, 2.0],
            [6.0, 6.0, 6.0],
            [3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0],
            [6.0, 6.0, 6.0],
            [3.0, 3.0, 3.0],
        ]).unwrap();
        let mut ee = Tensor3::<9, 3>::new();
        ee.set_tensor(2.0, &dd);
        #[rustfmt::skip]
        let correct = Matrix::from(&[
            [ 2.0,  2.0,  2.0],
            [10.0, 10.0, 10.0],
            [18.0, 18.0, 18.0],
            [ 4.0,  4.0,  4.0],
            [12.0, 12.0, 12.0],
            [ 6.0,  6.0,  6.0],
            [ 4.0,  4.0,  4.0],
            [12.0, 12.0, 12.0],
            [ 6.0,  6.0,  6.0],
        ]);
        mat_approx_eq(&ee.as_std_matrix(), &correct, 1e-14);
    }

    /// Generates a non-symmetric standard 3x3x3 tensor with distinct components
    fn generate_std_general() -> [[[f64; 3]; 3]; 3] {
        let mut inp = [[[0.0; 3]; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    inp[i][j][k] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
                }
            }
        }
        inp
    }

    /// Generates a standard 3x3x3 tensor that is minor-symmetric in (j,k)
    fn generate_std_sym_case_b() -> [[[f64; 3]; 3]; 3] {
        let mut inp = [[[0.0; 3]; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let (a, b) = if j <= k { (j, k) } else { (k, j) };
                    inp[i][j][k] = (100 * (i + 1) + 10 * (a + 1) + (b + 1)) as f64;
                }
            }
        }
        inp
    }

    /// Generates a standard 3x3x3 tensor that is minor-symmetric in (j,k) and 2D (zero out-of-plane shears)
    fn generate_std_sym_case_b_2d() -> [[[f64; 3]; 3]; 3] {
        let mut inp = [[[0.0; 3]; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let (a, b) = if j <= k { (j, k) } else { (k, j) };
                    // zero out-of-plane shears (0,2) and (1,2)
                    if a == b || (a, b) == (0, 1) {
                        inp[i][j][k] = (100 * (i + 1) + 10 * (a + 1) + (b + 1)) as f64;
                    }
                }
            }
        }
        inp
    }

    #[test]
    fn new_case_b_works() {
        let _ = Tensor3::<3, 9>::new();
        let _ = Tensor3::<3, 6>::new();
        let _ = Tensor3::<3, 4>::new();
    }

    #[test]
    fn from_std_array_case_b_fails_captures_errors() {
        let res = Tensor3::<3, 6>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );

        let res = Tensor3::<3, 4>::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn from_std_array_case_b_works() {
        // general
        let dd = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        for m in 0..3 {
            for n in 0..9 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_B_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 3D
        let dd = Tensor3::<3, 6>::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1).unwrap();
        for m in 0..3 {
            for n in 0..6 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_B_SYM_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }

        // symmetric 2D
        let dd = Tensor3::<3, 4>::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1).unwrap();
        for m in 0..3 {
            for n in 0..4 {
                assert_eq!(dd.get(m, n), SamplesTensor3::CASE_B_SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n]);
            }
        }
    }

    #[test]
    fn from_std_matrix_case_b_works() {
        // general
        let dd = Tensor3::<3, 9>::from_std_matrix(&SamplesTensor3::CASE_B_SAMPLE1_STD_MATRIX).unwrap();
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(dd.get(m, n), SamplesTensor3::CASE_B_SAMPLE1_KELVIN_MATRIX[m][n], 1e-15);
            }
        }

        // symmetric 3D
        let dd = Tensor3::<3, 6>::from_std_matrix(&SamplesTensor3::CASE_B_SYM_SAMPLE1_STD_MATRIX).unwrap();
        for m in 0..3 {
            for n in 0..6 {
                approx_eq(
                    dd.get(m, n),
                    SamplesTensor3::CASE_B_SYM_SAMPLE1_KELVIN_MATRIX[m][n],
                    1e-14,
                );
            }
        }

        // symmetric 2D
        let dd = Tensor3::<3, 4>::from_std_matrix(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1_STD_MATRIX).unwrap();
        for m in 0..3 {
            for n in 0..4 {
                approx_eq(
                    dd.get(m, n),
                    SamplesTensor3::CASE_B_SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n],
                    1e-14,
                );
            }
        }
    }

    #[test]
    fn get_std_case_b_works() {
        // general
        let dd = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::CASE_B_SAMPLE1[i][j][k], 1e-13);
                }
            }
        }

        // symmetric 3D
        let dd = Tensor3::<3, 6>::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), SamplesTensor3::CASE_B_SYM_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }

        // symmetric 2D
        let dd = Tensor3::<3, 4>::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(
                        dd.get_std(i, j, k),
                        SamplesTensor3::CASE_B_SYM_2D_SAMPLE1[i][j][k],
                        1e-14,
                    );
                }
            }
        }
    }

    #[test]
    fn update_case_b_works() {
        let mut dd = Tensor3::<3, 4>::new();
        let ee = Tensor3::<3, 4>::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1).unwrap();
        dd.update(2.0, &ee);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(
                        dd.get_std(i, j, k),
                        2.0 * SamplesTensor3::CASE_B_SYM_2D_SAMPLE1[i][j][k],
                        1e-14,
                    );
                }
            }
        }
    }

    #[test]
    fn as_std_array_and_to_std_array_case_b_work() {
        // general
        let dd = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_B_SAMPLE1[i][j][k], 1e-13);
                }
            }
        }

        // symmetric 3D
        let dd = Tensor3::<3, 6>::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_B_SYM_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }

        // symmetric 2D
        let dd = Tensor3::<3, 4>::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1).unwrap();
        let res = dd.as_std_array();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(res[i][j][k], SamplesTensor3::CASE_B_SYM_2D_SAMPLE1[i][j][k], 1e-14);
                }
            }
        }
    }

    #[test]
    fn as_std_matrix_and_to_std_matrix_case_b_work() {
        // general
        let dd = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        let mat = dd.as_std_matrix();
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(mat.get(m, n), SamplesTensor3::CASE_B_SAMPLE1_STD_MATRIX[m][n], 1e-13);
            }
        }

        // symmetric 3D
        let dd = Tensor3::<3, 6>::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (3, 9));
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(
                    mat.get(m, n),
                    SamplesTensor3::CASE_B_SYM_SAMPLE1_STD_MATRIX[m][n],
                    1e-13,
                );
            }
        }

        // symmetric 2D
        let dd = Tensor3::<3, 4>::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1).unwrap();
        let mat = dd.as_std_matrix();
        assert_eq!(mat.dims(), (3, 9));
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(
                    mat.get(m, n),
                    SamplesTensor3::CASE_B_SYM_2D_SAMPLE1_STD_MATRIX[m][n],
                    1e-13,
                );
            }
        }
    }

    #[test]
    fn sym_set_std_case_b_works() {
        let mut dd = Tensor3::<3, 6>::new();
        let inp = generate_std_sym_case_b();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    if j <= k {
                        dd.sym_set_std(i, j, k, inp[i][j][k]);
                    }
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    approx_eq(dd.get_std(i, j, k), inp[i][j][k], 1e-13);
                }
            }
        }
    }

    #[test]
    fn from_std_matrix_case_b_symmetric2d_fails() {
        let inp = generate_std_sym_case_b_2d();
        let dd = Tensor3::<3, 4>::from_std_array(&inp).unwrap();
        let mut mat = dd.as_std_matrix();
        // corrupt the out-of-plane shear (i,j,k) = (0,0,2) -> (m,n) = (0,5)
        mat.set(0, 5, 5.0);
        let res = Tensor3::<3, 4>::from_std_matrix(&mat);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a 2D minor-symmetric tensor")
        );
    }

    #[test]
    fn from_std_matrix_case_b_symmetric_fails() {
        let inp = generate_std_sym_case_b();
        let dd = Tensor3::<3, 6>::from_std_array(&inp).unwrap();
        let mut mat = dd.as_std_matrix();
        // break minor-symmetry: component (0,0,1) differs from its mirror (0,1,0)
        mat.set(0, 3, mat.get(0, 3) + 1.0);
        let res = Tensor3::<3, 6>::from_std_matrix(&mat);
        assert_eq!(
            res.err(),
            Some("the input data does not correspond to a minor-symmetric tensor")
        );
    }

    #[test]
    fn set_tensor_and_update_case_b_work() {
        let inp = generate_std_general();
        let dd = Tensor3::<3, 9>::from_std_array(&inp).unwrap();

        // set_tensor
        let mut ee = Tensor3::<3, 9>::new();
        ee.set_tensor(2.0, &dd);
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(ee.get(m, n), 2.0 * dd.get(m, n), 1e-13);
            }
        }

        // update
        let mut ff = Tensor3::<3, 9>::new();
        ff.update(1.0, &dd);
        ff.update(2.0, &dd);
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(ff.get(m, n), 3.0 * dd.get(m, n), 1e-13);
            }
        }
    }

    #[test]
    fn clone_and_serialize_work() {
        let dd = generate_dd_sym();
        // clone
        let mut cloned = dd.clone();
        cloned.set(0, 0, 999.0);
        assert_eq!(
            format!("{:.0}", dd.as_std_matrix()),
            "┌             ┐\n\
             │ 111 112 113 │\n\
             │ 221 222 223 │\n\
             │ 331 332 333 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             └             ┘"
        );
        assert_eq!(
            format!("{:.0}", cloned.as_std_matrix()),
            "┌             ┐\n\
             │ 999 112 113 │\n\
             │ 221 222 223 │\n\
             │ 331 332 333 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             └             ┘"
        );
        // serialize
        let json = serde_json::to_string(&dd).unwrap();
        assert!(json.len() > 0);
        // deserialize
        let from_json: Tensor3<6, 3> = serde_json::from_str(&json).unwrap();
        assert_eq!(
            format!("{:.0}", from_json.as_std_matrix()),
            "┌             ┐\n\
             │ 111 112 113 │\n\
             │ 221 222 223 │\n\
             │ 331 332 333 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             │ 121 122 123 │\n\
             │ 231 232 233 │\n\
             │ 131 132 133 │\n\
             └             ┘"
        );
    }

    #[test]
    fn debug_works() {
        let dd = Tensor3::<4, 3>::new();
        assert!(format!("{:?}", dd).len() > 0);
    }

    #[test]
    fn constant_permutation_works() {
        let perm_a = Tensor3::<9, 3>::constant_permutation();
        let expected = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, SQRT_2],
            [SQRT_2, 0.0, 0.0],
            [0.0, -SQRT_2, 0.0],
        ];
        for m in 0..9 {
            for n in 0..3 {
                approx_eq(perm_a.get(m, n), expected[m][n], 1e-15);
            }
        }
        assert_eq!(
            format!("{:.3}", perm_a),
            "┌                      ┐\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  0.000 │\n\
             │  0.000  0.000  1.414 │\n\
             │  1.414  0.000  0.000 │\n\
             │  0.000 -1.414  0.000 │\n\
             └                      ┘"
        );

        let perm_b = Tensor3::<3, 9>::constant_permutation();
        let expected = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SQRT_2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -SQRT_2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SQRT_2, 0.0, 0.0],
        ];
        for m in 0..3 {
            for n in 0..9 {
                approx_eq(perm_b.get(m, n), expected[m][n], 1e-15);
            }
        }
    }
}
