use crate::{IJ_TO_M, IJ_TO_M_SYM, M_TO_IJ, TOL_J2};
use crate::{SQRT_2, SQRT_2_BY_3, SQRT_3, SQRT_3_BY_2, SQRT_6};
use crate::{StrError, Tensor1};
use russell_lab::math::PI;
use russell_lab::{AsArray2D, Matrix, Vector, mat_eigen_sym, mat_eigenvalues, sort3};
use serde::{Deserialize, Serialize};
use std::cmp;
use std::fmt::{self, Write};

/// Defines a second-order tensor in R³×R³
///
/// # Standard and Kelvin-Mandel components
///
/// The methods of this struct follow a naming convention that distinguishes
/// between the **standard** (Cartesian) components `Tᵢⱼ` and the **Kelvin-Mandel**
/// components stored internally:
///
/// * Methods dealing with **standard components** carry the `std` qualifier in
///   their names (e.g., [Tensor2::set_std_matrix], [Tensor2::get_std],
///   [Tensor2::as_std_matrix], [Tensor2::sym_set_std]).
/// * Methods dealing directly with the **Kelvin-Mandel components** carry no qualifier
///   (e.g., [Tensor2::get], [Tensor2::set], [Tensor2::set_vector],
///   [Tensor2::set_tensor], [Tensor2::update]).
///
/// Internally, the components are converted to the Kelvin-Mandel basis as follows.
///
/// N = 9:
///
/// ```text
///                       ┌                ┐
///                    00 │      T00       │ 0
///                    11 │      T11       │ 1
/// ┌             ┐    22 │      T22       │ 2
/// │ T00 T01 T02 │    01 │ (T01+T10) / √2 │ 3
/// │ T10 T11 T12 │ => 12 │ (T12+T21) / √2 │ 4
/// │ T20 T21 T22 │    02 │ (T02+T20) / √2 │ 5
/// └             ┘    10 │ (T01-T10) / √2 │ 6
///                    21 │ (T12-T21) / √2 │ 7
///                    20 │ (T02-T20) / √2 │ 8
///                       └                ┘
/// ```
///
/// N = 6:
///
/// ```text
///                       ┌          ┐
/// ┌             ┐    00 │   T00    │ 0
/// │ T00 T01 T02 │    11 │   T11    │ 1
/// │ T01 T11 T12 │ => 22 │   T22    │ 2
/// │ T02 T12 T22 │    01 │ T01 * √2 │ 3
/// └             ┘    12 │ T12 * √2 │ 4
///                    02 │ T02 * √2 │ 5
///                       └          ┘
/// ```
///
/// N = 4:
///
/// ```text
/// ┌             ┐       ┌          ┐
/// │ T00 T01     │    00 │   T00    │ 0
/// │ T01 T11     │ => 11 │   T11    │ 1
/// │         T22 │    22 │   T22    │ 2
/// └             ┘    01 │ T01 * √2 │ 3
///                       └          ┘
/// ```
#[derive(Clone, Debug)]
pub struct Tensor2<const N: usize> {
    /// Holds the components in Kelvin-Mandel basis as a vector (heap).
    ///
    /// Heap version => dynamically allocated memory
    #[cfg(feature = "heap")]
    pub(crate) vec: Vector,

    /// Holds the components in Kelvin-Mandel basis as a vector (stack).
    ///
    /// Stack version => fixed size memory
    #[cfg(not(feature = "heap"))]
    pub(crate) vec: [f64; N],
}

// Manual Serialize/Deserialize implementations: serde only implements the traits
// for concrete array sizes, so the derive fails for the generic `[f64; N]`.
// Since N is known to be 4, 6, or 9 only, we serialize the components as a sequence.
impl<const N: usize> Serialize for Tensor2<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.as_data().serialize(serializer)
    }
}

impl<'de, const N: usize> Deserialize<'de> for Tensor2<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec = Vec::<f64>::deserialize(deserializer)?;
        if vec.len() != N {
            return Err(serde::de::Error::custom(format!(
                "Tensor2 dimension mismatch: expected {}, got {}",
                N,
                vec.len()
            )));
        }
        let mut tt = Tensor2::new();
        for (i, value) in vec.iter().enumerate() {
            tt.vec[i] = *value;
        }
        Ok(tt)
    }
}

impl<const N: usize> Tensor2<N> {
    const VALIDATE_DIM: () = assert!(N == 4 || N == 6 || N == 9, "Tensor dimension must be 4, 6, or 9");

    /// Creates a new (zeroed) Tensor2
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{StrError, Tensor2};
    ///
    /// fn main() {
    ///     let a = Tensor2::<9>::new();
    ///     for m in 0..a.dim() {
    ///         assert_eq!(a.get(m), 0.0);
    ///     }
    ///
    ///     let b = Tensor2::<6>::new();
    ///     for m in 0..b.dim() {
    ///         assert_eq!(b.get(m), 0.0);
    ///     }
    ///
    ///     let c = Tensor2::<4>::new();
    ///     for m in 0..c.dim() {
    ///         assert_eq!(c.get(m), 0.0);
    ///     }
    /// }
    /// ```
    pub fn new() -> Self {
        let _ = Self::VALIDATE_DIM;

        #[cfg(feature = "heap")]
        let vec = Vector::new(N);

        #[cfg(not(feature = "heap"))]
        let vec = [0.0; N];

        Tensor2 { vec }
    }

    /// Allocates a diagonal Tensor2 from octahedral components
    ///
    /// # Input
    ///
    /// * `distance` -- distance from the octahedral plane to the origin: `d = (λ1 + λ2 + λ3) / √3`
    /// * `radius` -- radius on the octahedral plane: `r = ‖s‖`
    /// * `lode` -- Lode invariant: `l = cos(3θ) = (3 √3 J3)/(2 pow(J2,1.5))`
    ///   **Note:** The Lode invariant must be in `-1 ≤ lode ≤ 1`
    /// * `two_dim` -- 2D instead of 3D?
    ///
    /// The octahedral components and the invariants are related as follows:
    ///
    /// ```text
    /// σm = d / √3   →  d = σm √3
    /// σd = r √3/√2  →  r = σd √2/√3 = √2 √J2
    /// εv = d √3     →  d = εv / √3
    /// εd = r √2/√3  →  r = εd √3/√2
    /// ```
    ///
    /// In matrix form, the diagonal components of the tensor are the principal values `(λ1, λ2, λ3)`:
    ///
    /// ```text
    /// ┌          ┐
    /// │ λ1  0  0 │
    /// │  0 λ2  0 │
    /// │  0  0 λ3 │
    /// └          ┘
    /// ```
    pub fn new_from_octahedral(distance: f64, radius: f64, lode: f64) -> Result<Self, StrError> {
        if lode < -1.0 || lode > 1.0 {
            return Err("lode invariant must be in -1 ≤ lode ≤ 1");
        }
        let theta = f64::acos(lode) / 3.0;
        let star1 = radius * f64::cos(theta);
        let star2 = distance;
        let star3 = radius * f64::sin(theta);
        let mut tt = Tensor2::new();
        tt.vec[0] = (SQRT_2 * star1 + star2) / SQRT_3;
        tt.vec[1] = -star1 / SQRT_6 + star2 / SQRT_3 - star3 / SQRT_2;
        tt.vec[2] = -star1 / SQRT_6 + star2 / SQRT_3 + star3 / SQRT_2;
        Ok(tt)
    }

    /// Allocates a diagonal Tensor2 from octahedral components (using alpha angle)
    ///
    /// # Input
    ///
    /// * `distance` -- distance from the octahedral plane to the origin: `d = (λ1 + λ2 + λ3) / √3`
    /// * `radius` -- radius on the octahedral plane: `r = ‖s‖`
    /// * `alpha` -- alpha angle in radians from -π to π
    /// * `two_dim` -- 2D instead of 3D?
    ///
    /// The octahedral components and the invariants are related as follows:
    ///
    /// ```text
    /// σm = d / √3   →  d = σm √3
    /// σd = r √3/√2  →  r = σd √2/√3 = √2 √J2
    /// εv = d √3     →  d = εv / √3
    /// εd = r √2/√3  →  r = εd √3/√2
    /// ```
    ///
    /// In matrix form, the diagonal components of the tensor are the principal values `(λ1, λ2, λ3)`:
    ///
    /// ```text
    /// ┌          ┐
    /// │ λ1  0  0 │
    /// │  0 λ2  0 │
    /// │  0  0 λ3 │
    /// └          ┘
    /// ```
    pub fn new_from_octahedral_alpha(distance: f64, radius: f64, alpha: f64) -> Result<Self, StrError> {
        if alpha < -PI || alpha > PI {
            return Err("alpha must be in -π ≤ alpha ≤ π");
        }
        let star1 = radius * f64::sin(alpha);
        let star2 = distance;
        let star3 = radius * f64::cos(alpha);
        let mut tt = Tensor2::new();
        tt.vec[0] = (SQRT_2 * star1 + star2) / SQRT_3;
        tt.vec[1] = -star1 / SQRT_6 + star2 / SQRT_3 - star3 / SQRT_2;
        tt.vec[2] = -star1 / SQRT_6 + star2 / SQRT_3 + star3 / SQRT_2;
        Ok(tt)
    }

    /// Returns the Kelvin-Mandel vector dimension (4, 6, or 9)
    #[inline]
    pub fn dim(&self) -> usize {
        N
    }

    /// Returns the m-component of the Kelvin-Mandel vector
    ///
    /// # Panics
    ///
    /// A panic will occur if the index is out of range.
    #[inline]
    pub fn get(&self, m: usize) -> f64 {
        self.vec[m]
    }

    /// Sets the m-component of the Kelvin-Mandel vector
    ///
    /// # Panics
    ///
    /// A panic will occur if the index is out of range.
    #[inline]
    pub fn set(&mut self, m: usize, value: f64) {
        self.vec[m] = value;
    }

    /// Returns a slice to the Kelvin-Mandel vector data (crate-internal)
    ///
    /// Note: the slice length equals the Kelvin-Mandel vector dimension (4, 6, or 9).
    #[inline]
    pub(crate) fn as_data(&self) -> &[f64] {
        #[cfg(feature = "heap")]
        {
            self.vec.as_data().as_slice()
        }
        #[cfg(not(feature = "heap"))]
        {
            &self.vec[..]
        }
    }

    /// Returns a mutable slice to the Kelvin-Mandel vector data (crate-internal)
    ///
    /// Note: the slice length equals the Kelvin-Mandel vector dimension (4, 6, or 9).
    #[inline]
    pub(crate) fn as_mut_data(&mut self) -> &mut [f64] {
        #[cfg(feature = "heap")]
        {
            self.vec.as_mut_data().as_mut_slice()
        }
        #[cfg(not(feature = "heap"))]
        {
            &mut self.vec[..]
        }
    }

    /// Sets the Tensor2 with standard components given in matrix form
    ///
    /// # Input
    ///
    /// * `tt` -- the standard Tij components given with respect to an orthonormal Cartesian basis
    ///
    /// # Notes
    ///
    /// * In all cases, even in 2D, the input matrix must be 3×3
    /// * If symmetric, the off-diagonal components must equal each other
    /// * If 2D, `data[1][2]` and `data[0][2]` must be equal to zero
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Vector;
    /// use russell_tensor::{StrError, Tensor2, SQRT_2};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // general
    ///     let mut a = Tensor2::<9>::new();
    ///     a.set_std_matrix(&[
    ///         [1.0, SQRT_2 * 2.0, SQRT_2 * 3.0],
    ///         [SQRT_2 * 4.0, 5.0, SQRT_2 * 6.0],
    ///         [SQRT_2 * 7.0, SQRT_2 * 8.0, 9.0],
    ///     ])?;
    ///     assert_eq!(
    ///         format!("{:.1}", a),
    ///         "┌      ┐\n\
    ///          │  1.0 │\n\
    ///          │  5.0 │\n\
    ///          │  9.0 │\n\
    ///          │  6.0 │\n\
    ///          │ 14.0 │\n\
    ///          │ 10.0 │\n\
    ///          │ -2.0 │\n\
    ///          │ -2.0 │\n\
    ///          │ -4.0 │\n\
    ///          └      ┘"
    ///     );
    ///
    ///     // symmetric-3D
    ///     let mut b = Tensor2::<6>::new();
    ///     b.set_std_matrix(&[
    ///             [1.0, 4.0 / SQRT_2, 6.0 / SQRT_2],
    ///             [4.0 / SQRT_2, 2.0, 5.0 / SQRT_2],
    ///             [6.0 / SQRT_2, 5.0 / SQRT_2, 3.0],
    ///     ])?;
    ///     assert_eq!(
    ///         format!("{:.1}", b),
    ///         "┌     ┐\n\
    ///          │ 1.0 │\n\
    ///          │ 2.0 │\n\
    ///          │ 3.0 │\n\
    ///          │ 4.0 │\n\
    ///          │ 5.0 │\n\
    ///          │ 6.0 │\n\
    ///          └     ┘"
    ///     );
    ///
    ///     // symmetric-2D
    ///     let mut c = Tensor2::<4>::new();
    ///     c.set_std_matrix(&[
    ///             [       1.0, 4.0/SQRT_2, 0.0],
    ///             [4.0/SQRT_2,        2.0, 0.0],
    ///             [       0.0,        0.0, 3.0],
    ///     ])?;
    ///     assert_eq!(
    ///         format!("{:.1}", c),
    ///         "┌     ┐\n\
    ///          │ 1.0 │\n\
    ///          │ 2.0 │\n\
    ///          │ 3.0 │\n\
    ///          │ 4.0 │\n\
    ///          └     ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn set_std_matrix<'a, S>(&mut self, tt: &'a S) -> Result<(), StrError>
    where
        S: AsArray2D<'a, f64>,
    {
        if N == 4 || N == 6 {
            if tt.at(1, 0) != tt.at(0, 1) || tt.at(2, 1) != tt.at(1, 2) || tt.at(2, 0) != tt.at(0, 2) {
                return Err("cannot set symmetric Tensor2 with non-symmetric data");
            }
            if N == 4 {
                if tt.at(1, 2) != 0.0 || tt.at(0, 2) != 0.0 {
                    return Err("cannot set Symmetric2D Tensor2 with non-zero off-diagonal data");
                }
            }
        }
        for m in 0..N {
            let (i, j) = M_TO_IJ[m];
            if i == j {
                self.vec[m] = tt.at(i, j);
            }
            if i < j {
                self.vec[m] = (tt.at(i, j) + tt.at(j, i)) / SQRT_2;
            }
            if i > j {
                self.vec[m] = (tt.at(j, i) - tt.at(i, j)) / SQRT_2;
            }
        }
        Ok(())
    }

    /// Creates a new Tensor2 constructed from a 3x3 matrix with standard components
    ///
    /// # Input
    ///
    /// * `tt` -- the standard Tij components with respect to an orthonormal Cartesian basis
    ///
    /// # Notes
    ///
    /// * In all cases, even in 2D, the input matrix must be 3×3
    /// * If symmetric, the off-diagonal components must equal each other
    /// * If 2D, `data[1][2]` and `data[0][2]` must be equal to zero
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix is not 3x3.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Vector;
    /// use russell_tensor::{StrError, Tensor2, SQRT_2};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // general
    ///     let a = Tensor2::<9>::from_std_matrix(
    ///         &[
    ///             [1.0, SQRT_2 * 2.0, SQRT_2 * 3.0],
    ///             [SQRT_2 * 4.0, 5.0, SQRT_2 * 6.0],
    ///             [SQRT_2 * 7.0, SQRT_2 * 8.0, 9.0],
    ///         ]
    ///     )?;
    ///     assert_eq!(
    ///         format!("{:.1}", a),
    ///         "┌      ┐\n\
    ///          │  1.0 │\n\
    ///          │  5.0 │\n\
    ///          │  9.0 │\n\
    ///          │  6.0 │\n\
    ///          │ 14.0 │\n\
    ///          │ 10.0 │\n\
    ///          │ -2.0 │\n\
    ///          │ -2.0 │\n\
    ///          │ -4.0 │\n\
    ///          └      ┘"
    ///     );
    ///
    ///     // symmetric-3D
    ///     let b = Tensor2::<6>::from_std_matrix(
    ///         &[
    ///             [1.0, 4.0 / SQRT_2, 6.0 / SQRT_2],
    ///             [4.0 / SQRT_2, 2.0, 5.0 / SQRT_2],
    ///             [6.0 / SQRT_2, 5.0 / SQRT_2, 3.0],
    ///         ]
    ///     )?;
    ///     assert_eq!(
    ///         format!("{:.1}", b),
    ///         "┌     ┐\n\
    ///          │ 1.0 │\n\
    ///          │ 2.0 │\n\
    ///          │ 3.0 │\n\
    ///          │ 4.0 │\n\
    ///          │ 5.0 │\n\
    ///          │ 6.0 │\n\
    ///          └     ┘"
    ///     );
    ///
    ///     // symmetric-2D
    ///     let c = Tensor2::<4>::from_std_matrix(
    ///         &[
    ///             [       1.0, 4.0/SQRT_2, 0.0],
    ///             [4.0/SQRT_2,        2.0, 0.0],
    ///             [       0.0,        0.0, 3.0],
    ///         ]
    ///     )?;
    ///     assert_eq!(
    ///         format!("{:.1}", c),
    ///         "┌     ┐\n\
    ///          │ 1.0 │\n\
    ///          │ 2.0 │\n\
    ///          │ 3.0 │\n\
    ///          │ 4.0 │\n\
    ///          └     ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn from_std_matrix<'a, S>(tt: &'a S) -> Result<Self, StrError>
    where
        S: AsArray2D<'a, f64>,
    {
        let mut res = Tensor2::new();
        res.set_std_matrix(tt)?;
        Ok(res)
    }

    /// Returns a new identity tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Vector;
    /// use russell_tensor::{Tensor2};
    ///
    /// let ii = Tensor2::<9>::identity();
    ///
    /// assert_eq!(
    ///     format!("{}", ii),
    ///     "┌   ┐\n\
    ///      │ 1 │\n\
    ///      │ 1 │\n\
    ///      │ 1 │\n\
    ///      │ 0 │\n\
    ///      │ 0 │\n\
    ///      │ 0 │\n\
    ///      │ 0 │\n\
    ///      │ 0 │\n\
    ///      │ 0 │\n\
    ///      └   ┘"
    /// );
    /// ```
    pub fn identity() -> Self {
        let mut res = Tensor2::new();
        res.vec[0] = 1.0;
        res.vec[1] = 1.0;
        res.vec[2] = 1.0;
        res
    }

    /// Returns the standard (i,j) component
    ///
    /// **Note:** Returns the standard component (not Kelvin-Mandel).
    ///
    /// # Input
    ///
    /// * `i` -- the first index in `(0, 1, 2)`
    /// * `j` -- the second index in `(0, 1, 2)`
    ///
    /// # Panics
    ///
    /// A panic will occur if the indices are out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0,  2.0, 0.0],
    ///         [3.0, -1.0, 5.0],
    ///         [0.0,  4.0, 1.0],
    ///     ])?;
    ///
    ///     approx_eq(a.get_std(1,2), 5.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn get_std(&self, i: usize, j: usize) -> f64 {
        match N {
            4 => {
                let m = IJ_TO_M_SYM[i][j];
                if m > 3 {
                    0.0
                } else if i == j {
                    self.vec[m]
                } else {
                    self.vec[m] / SQRT_2
                }
            }
            6 => {
                let m = IJ_TO_M_SYM[i][j];
                if i == j { self.vec[m] } else { self.vec[m] / SQRT_2 }
            }
            _ => {
                let m = IJ_TO_M[i][j];
                if i == j {
                    self.vec[m]
                } else if i < j {
                    let n = IJ_TO_M[j][i];
                    (self.vec[m] + self.vec[n]) / SQRT_2
                } else {
                    let n = IJ_TO_M[j][i];
                    (self.vec[n] - self.vec[m]) / SQRT_2
                }
            }
        }
    }

    /// Returns a 3x3 matrix with the standard components
    ///
    /// **Note:** The matrix will have the standard components and 3x3 dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0,  1.0, 0.0],
    ///         [1.0, -1.0, 0.0],
    ///         [0.0,  0.0, 1.0],
    ///     ])?;
    ///     assert_eq!(
    ///         format!("{:.1}", a.as_std_matrix()),
    ///         "┌                ┐\n\
    ///          │  1.0  1.0  0.0 │\n\
    ///          │  1.0 -1.0  0.0 │\n\
    ///          │  0.0  0.0  1.0 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn as_std_matrix(&self) -> Matrix {
        let mut mat = Matrix::new(3, 3);
        self.to_std_matrix(&mut mat);
        mat
    }

    /// Converts this tensor to a 3x3 matrix with the standard components
    ///
    /// # Input
    ///
    /// * `mat` -- the resulting 3x3 matrix
    ///
    /// # Panics
    ///
    /// A panic will occur if the matrix is not 3x3
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Matrix;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0,  1.0, 0.0],
    ///         [1.0, -1.0, 0.0],
    ///         [0.0,  0.0, 1.0],
    ///     ])?;
    ///     let mut mat = Matrix::new(3, 3);
    ///     a.to_std_matrix(&mut mat);
    ///     assert_eq!(
    ///         format!("{:.1}", mat),
    ///         "┌                ┐\n\
    ///          │  1.0  1.0  0.0 │\n\
    ///          │  1.0 -1.0  0.0 │\n\
    ///          │  0.0  0.0  1.0 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn to_std_matrix(&self, mat: &mut Matrix) {
        assert_eq!(mat.dims(), (3, 3));
        if N < 9 {
            for m in 0..N {
                let (i, j) = M_TO_IJ[m];
                mat.set(i, j, self.get_std(i, j));
                if i != j {
                    mat.set(j, i, mat.get(i, j));
                }
            }
        } else {
            for i in 0..3 {
                for j in 0..3 {
                    mat.set(i, j, self.get_std(i, j));
                }
            }
        }
    }

    /// Returns a 2x2 matrix with the standard components
    ///
    /// # Notes
    ///
    /// 1. The matrix will have the standard components and 2x2 dimension
    /// 2. This function returns the third diagonal component T22 and the 2x2 matrix
    ///
    /// # Panics
    ///
    /// A panic will occur if the tensor is not symmetric in 2D
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let tt = Tensor2::<4>::from_std_matrix(&[
    ///         [1.0, 2.0, 0.0],
    ///         [2.0, 3.0, 0.0],
    ///         [0.0, 0.0, 4.0],
    ///     ])?;
    ///     let (t22, res) = tt.as_std_matrix_2d();
    ///     assert_eq!(t22, 4.0);
    ///     assert_eq!(
    ///         format!("{:.1}", res),
    ///         "┌         ┐\n\
    ///          │ 1.0 2.0 │\n\
    ///          │ 2.0 3.0 │\n\
    ///          └         ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn as_std_matrix_2d(&self) -> (f64, Matrix) {
        assert_eq!(N, 4, "the tensor must be symmetric in 2D");
        let mut tt = Matrix::new(2, 2);
        tt.set(0, 0, self.get_std(0, 0));
        tt.set(0, 1, self.get_std(0, 1));
        tt.set(1, 0, self.get_std(1, 0));
        tt.set(1, 1, self.get_std(1, 1));
        (self.get_std(2, 2), tt)
    }

    /// Calculates the eigenvalues of this symmetric tensor (without eigenvectors)
    ///
    /// The eigenvalues correspond to the principal values of the tensor.
    ///
    /// # Output
    ///
    /// * `l` -- (lambda) will hold the eigenvalues (sorted in ascending order);
    ///   it must have dimension 3
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * the tensor is not symmetric
    /// * `l.dim()` is not equal to 3
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Vector;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<6>::from_std_matrix(&[
    ///         [2.0, 0.0, 0.0],
    ///         [0.0, 3.0, 4.0],
    ///         [0.0, 4.0, 9.0],
    ///     ])?;
    ///     let mut l = Vector::new(3);
    ///     a.eigenvalues_sym(&mut l)?;
    ///     assert_eq!(format!("{:.0}", l), "┌    ┐\n│  1 │\n│  2 │\n│ 11 │\n└    ┘");
    ///     Ok(())
    /// }
    /// ```
    pub fn eigenvalues_sym(&self, l: &mut Vector) -> Result<(), StrError> {
        if N == 9 {
            return Err("the tensor must be symmetric");
        }
        if l.dim() != 3 {
            return Err("l.dim() must be equal to 3");
        }
        let mut a = self.as_std_matrix();
        mat_eigen_sym(l, &mut a, false)?;
        Ok(())
    }

    /// Calculates the eigenvalues of this (general) tensor (without eigenvectors)
    ///
    /// # Output
    ///
    /// * `l_real` -- will hold the real part of the eigenvalues; it must have dimension 3
    /// * `l_imag` -- will hold the imaginary part of the eigenvalues; it must have dimension 3
    ///
    /// # Errors
    ///
    /// Returns an error if `l_real.dim()` or `l_imag.dim()` is not equal to 3
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Vector;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [2.0, 0.0, 0.0],
    ///         [0.0, 3.0, 4.0],
    ///         [0.0, 4.0, 9.0],
    ///     ])?;
    ///     let mut l_real = Vector::new(3);
    ///     let mut l_imag = Vector::new(3);
    ///     a.eigenvalues(&mut l_real, &mut l_imag)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn eigenvalues(&self, l_real: &mut Vector, l_imag: &mut Vector) -> Result<(), StrError> {
        if l_real.dim() != 3 || l_imag.dim() != 3 {
            return Err("l_real.dim() and l_imag.dim() must be equal to 3");
        }
        let mut a = self.as_std_matrix();
        mat_eigenvalues(l_real, l_imag, &mut a)?;
        Ok(())
    }

    /// Returns a general Tensor2
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Vector;
    /// use russell_tensor::{Tensor2, StrError, SQRT_2};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let tt = Tensor2::<4>::from_std_matrix(&[
    ///         [1.0,        2.0/SQRT_2, 0.0],
    ///         [2.0/SQRT_2, 3.0,        0.0],
    ///         [0.0,        0.0,        4.0],
    ///     ])?;
    ///     assert_eq!(
    ///         format!("{:.2}", tt),
    ///         "┌      ┐\n\
    ///          │ 1.00 │\n\
    ///          │ 3.00 │\n\
    ///          │ 4.00 │\n\
    ///          │ 2.00 │\n\
    ///          └      ┘"
    ///     );
    ///
    ///     let tt_gen = tt.as_general();
    ///     assert_eq!(
    ///         format!("{:.2}", tt_gen),
    ///         "┌      ┐\n\
    ///          │ 1.00 │\n\
    ///          │ 3.00 │\n\
    ///          │ 4.00 │\n\
    ///          │ 2.00 │\n\
    ///          │ 0.00 │\n\
    ///          │ 0.00 │\n\
    ///          │ 0.00 │\n\
    ///          │ 0.00 │\n\
    ///          │ 0.00 │\n\
    ///          └      ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn as_general(&self) -> Tensor2<9> {
        let mut res = Tensor2::<9>::new();
        res.vec[0] = self.vec[0];
        res.vec[1] = self.vec[1];
        res.vec[2] = self.vec[2];
        res.vec[3] = self.vec[3];
        if N > 4 {
            res.vec[4] = self.vec[4];
            res.vec[5] = self.vec[5];
        }
        if N > 6 {
            res.vec[6] = self.vec[6];
            res.vec[7] = self.vec[7];
            res.vec[8] = self.vec[8];
        }
        res
    }

    /// Returns a symmetric tensor
    ///
    /// # Panics
    ///
    /// A panic will occur if the tensor is not symmetric 2D
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Vector;
    /// use russell_tensor::{Tensor2, StrError, SQRT_2};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let tt = Tensor2::<4>::from_std_matrix(&[
    ///         [1.0,        2.0/SQRT_2, 0.0],
    ///         [2.0/SQRT_2, 3.0,        0.0],
    ///         [0.0,        0.0,        4.0],
    ///     ])?;
    ///     assert_eq!(
    ///         format!("{:.2}", tt),
    ///         "┌      ┐\n\
    ///          │ 1.00 │\n\
    ///          │ 3.00 │\n\
    ///          │ 4.00 │\n\
    ///          │ 2.00 │\n\
    ///          └      ┘"
    ///     );
    ///
    ///     let tt_sym = tt.sym2d_as_symmetric();
    ///     assert_eq!(
    ///         format!("{:.2}", tt_sym),
    ///         "┌      ┐\n\
    ///          │ 1.00 │\n\
    ///          │ 3.00 │\n\
    ///          │ 4.00 │\n\
    ///          │ 2.00 │\n\
    ///          │ 0.00 │\n\
    ///          │ 0.00 │\n\
    ///          └      ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn sym2d_as_symmetric(&self) -> Tensor2<6> {
        assert_eq!(N, 4, "the tensor must be symmetric in 2D");
        let mut res = Tensor2::<6>::new();
        res.vec[0] = self.vec[0];
        res.vec[1] = self.vec[1];
        res.vec[2] = self.vec[2];
        res.vec[3] = self.vec[3];
        res
    }

    /// Set all values to zero
    pub fn clear(&mut self) {
        self.vec.fill(0.0);
    }

    /// Sets the (i,j) standard component of a symmetric Tensor2
    ///
    /// ```text
    /// σᵢⱼ = value
    /// ```
    ///
    /// # Notes
    ///
    /// 1. Only the diagonal and upper-diagonal components need to be set.
    /// 2. The tensor must be symmetric and (i,j) must correspond to the possible
    ///    combination due to the space dimension, otherwise a panic may occur.
    ///
    /// # Panics
    ///
    /// A panic will occur if the indices are out of range
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() {
    ///     let mut a = Tensor2::<4>::new();
    ///     a.sym_set_std(0, 0, 1.0);
    ///     a.sym_set_std(1, 1, 2.0);
    ///     a.sym_set_std(2, 2, 3.0);
    ///     a.sym_set_std(0, 1, 4.0);
    ///     assert_eq!(
    ///         format!("{:.1}", a.as_std_matrix()),
    ///         "┌             ┐\n\
    ///          │ 1.0 4.0 0.0 │\n\
    ///          │ 4.0 2.0 0.0 │\n\
    ///          │ 0.0 0.0 3.0 │\n\
    ///          └             ┘"
    ///     );
    ///
    ///     let mut b = Tensor2::<6>::new();
    ///     b.sym_set_std(0, 0, 1.0);
    ///     b.sym_set_std(1, 1, 2.0);
    ///     b.sym_set_std(2, 2, 3.0);
    ///     b.sym_set_std(0, 1, 4.0);
    ///     b.sym_set_std(1, 0, 4.0);
    ///     b.sym_set_std(2, 0, 5.0);
    ///     assert_eq!(
    ///         format!("{:.1}", b.as_std_matrix()),
    ///         "┌             ┐\n\
    ///          │ 1.0 4.0 5.0 │\n\
    ///          │ 4.0 2.0 0.0 │\n\
    ///          │ 5.0 0.0 3.0 │\n\
    ///          └             ┘"
    ///     );
    /// }
    /// ```
    pub fn sym_set_std(&mut self, i: usize, j: usize, value: f64) {
        let m = IJ_TO_M_SYM[i][j];
        if i == j {
            self.vec[m] = value;
        } else {
            self.vec[m] = value * SQRT_2;
        }
    }

    /// Updates the (i,j) standard component of a symmetric Tensor2
    ///
    /// ```text
    /// σᵢⱼ += α value
    /// ```
    ///
    /// # Notes
    ///
    /// 1. Only the diagonal and upper-diagonal components need to be handled.
    /// 2. The tensor must be symmetric and (i,j) must correspond to the possible
    ///    combination due to the space dimension, otherwise a panic may occur.
    ///
    /// # Panics
    ///
    /// 1. A panic will occur if the indices are out of range
    /// 2. A panic will occur if `i > j` (lower-diagonal)
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [2.0, 5.0, 6.0],
    ///         [3.0, 6.0, 9.0],
    ///     ])?;
    ///
    ///     a.sym_add_std(0, 1, 2.0, 10.0);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", a.as_std_matrix()),
    ///         "┌                ┐\n\
    ///          │  1.0 22.0  3.0 │\n\
    ///          │ 22.0  5.0  6.0 │\n\
    ///          │  3.0  6.0  9.0 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn sym_add_std(&mut self, i: usize, j: usize, alpha: f64, value: f64) {
        assert!(i <= j);
        let m = IJ_TO_M_SYM[i][j];
        if i == j {
            self.vec[m] += alpha * value;
        } else {
            self.vec[m] += alpha * value * SQRT_2;
        }
    }

    /// Sets the Kelvin-Mandel vector of this tensor as a scalar multiple of another Kelvin-Mandel vector
    ///
    /// ```text
    /// self := α other
    /// ```
    ///
    /// # Panics
    ///
    /// A panic will occur if the other tensor has an incorrect dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Vector;
    /// use russell_tensor::{Tensor2, StrError, SQRT_2};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ])?;
    ///     let v_kelvin = &Vector::from(&[
    ///         1.0,
    ///         5.0,
    ///         9.0,
    ///         6.0 / SQRT_2,
    ///         14.0 / SQRT_2,
    ///         10.0 / SQRT_2,
    ///         -2.0 / SQRT_2,
    ///         -2.0 / SQRT_2,
    ///         -4.0 / SQRT_2,
    ///     ]);
    ///
    ///     a.set_vector(2.0, v_kelvin.as_data());
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", a.as_std_matrix()),
    ///         "┌                ┐\n\
    ///          │  2.0  4.0  6.0 │\n\
    ///          │  8.0 10.0 12.0 │\n\
    ///          │ 14.0 16.0 18.0 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn set_vector(&mut self, alpha: f64, other: &[f64]) {
        self.vec[0] = alpha * other[0];
        self.vec[1] = alpha * other[1];
        self.vec[2] = alpha * other[2];
        self.vec[3] = alpha * other[3];
        if N > 4 {
            self.vec[4] = alpha * other[4];
            self.vec[5] = alpha * other[5];
        }
        if N > 6 {
            self.vec[6] = alpha * other[6];
            self.vec[7] = alpha * other[7];
            self.vec[8] = alpha * other[8];
        }
    }

    /// Makes this tensor equal to another tensor
    ///
    /// ```text
    /// self := α other
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ])?;
    ///     let b = Tensor2::<9>::from_std_matrix(&[
    ///         [10.0, 20.0, 30.0],
    ///         [40.0, 50.0, 60.0],
    ///         [70.0, 80.0, 90.0],
    ///     ])?;
    ///
    ///     a.set_tensor(2.0, &b);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", a.as_std_matrix()),
    ///         "┌                   ┐\n\
    ///          │  20.0  40.0  60.0 │\n\
    ///          │  80.0 100.0 120.0 │\n\
    ///          │ 140.0 160.0 180.0 │\n\
    ///          └                   ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn set_tensor(&mut self, alpha: f64, other: &Tensor2<N>) {
        self.vec[0] = alpha * other.vec[0];
        self.vec[1] = alpha * other.vec[1];
        self.vec[2] = alpha * other.vec[2];
        self.vec[3] = alpha * other.vec[3];
        if N > 4 {
            self.vec[4] = alpha * other.vec[4];
            self.vec[5] = alpha * other.vec[5];
        }
        if N > 6 {
            self.vec[6] = alpha * other.vec[6];
            self.vec[7] = alpha * other.vec[7];
            self.vec[8] = alpha * other.vec[8];
        }
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
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ])?;
    ///     let b = Tensor2::<9>::from_std_matrix(&[
    ///         [10.0, 20.0, 30.0],
    ///         [40.0, 50.0, 60.0],
    ///         [70.0, 80.0, 90.0],
    ///     ])?;
    ///
    ///     a.update(2.0, &b);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", a.as_std_matrix()),
    ///         "┌                   ┐\n\
    ///          │  21.0  42.0  63.0 │\n\
    ///          │  84.0 105.0 126.0 │\n\
    ///          │ 147.0 168.0 189.0 │\n\
    ///          └                   ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn update(&mut self, alpha: f64, other: &Tensor2<N>) {
        self.vec[0] += alpha * other.vec[0];
        self.vec[1] += alpha * other.vec[1];
        self.vec[2] += alpha * other.vec[2];
        self.vec[3] += alpha * other.vec[3];
        if N > 4 {
            self.vec[4] += alpha * other.vec[4];
            self.vec[5] += alpha * other.vec[5];
        }
        if N > 6 {
            self.vec[6] += alpha * other.vec[6];
            self.vec[7] += alpha * other.vec[7];
            self.vec[8] += alpha * other.vec[8];
        }
    }

    /// Calculates the determinant
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ])?;
    ///
    ///     approx_eq(a.determinant(), 0.0, 1e-13);
    ///     Ok(())
    /// }
    /// ```
    pub fn determinant(&self) -> f64 {
        let a = &self.vec;
        match N {
            4 => a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0,
            6 => {
                a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0 - (a[0] * a[4] * a[4]) / 2.0
                    + (a[3] * a[4] * a[5]) / SQRT_2
                    - (a[1] * a[5] * a[5]) / 2.0
            }
            _ => {
                a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0 - (a[0] * a[4] * a[4]) / 2.0
                    + (a[3] * a[4] * a[5]) / SQRT_2
                    - (a[1] * a[5] * a[5]) / 2.0
                    + (a[2] * a[6] * a[6]) / 2.0
                    + (a[5] * a[6] * a[7]) / SQRT_2
                    + (a[0] * a[7] * a[7]) / 2.0
                    - (a[4] * a[6] * a[8]) / SQRT_2
                    - (a[3] * a[7] * a[8]) / SQRT_2
                    + (a[1] * a[8] * a[8]) / 2.0
            }
        }
    }

    /// Returns the transpose tensor
    ///
    /// ```text
    /// Aᵀ = transpose(A)
    ///
    /// [Aᵀ]ᵢⱼ = [A]ⱼᵢ
    /// ```
    ///
    /// # Output
    ///
    /// * `at` -- a Tensor2 to hold the transpose tensor.
    ///
    /// # Panics
    ///
    /// A panic will occur if `at` has a different [Rep].
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::mat_approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.1, 1.2, 1.3],
    ///         [2.1, 2.2, 2.3],
    ///         [3.1, 3.2, 3.3],
    ///     ])?;
    ///
    ///     let mut at = Tensor2::<9>::new();
    ///     a.transpose(&mut at);
    ///
    ///     let at_correct = Tensor2::<9>::from_std_matrix(&[
    ///         [1.1, 2.1, 3.1],
    ///         [1.2, 2.2, 3.2],
    ///         [1.3, 2.3, 3.3],
    ///     ])?;
    ///     mat_approx_eq(&at.as_std_matrix(), &at_correct.as_std_matrix(), 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn transpose(&self, at: &mut Tensor2<N>) {
        self.transpose_slice(at.as_mut_data());
    }

    /// Returns the transpose tensor components in a caller-provided array (crate-internal)
    ///
    /// Mirrors [transpose] but returns the components instead of writing to a [Tensor2].
    #[inline]
    pub(crate) fn transpose_slice(&self, at: &mut [f64]) {
        // The transpose is given by:
        // [a0, a1, a2, a3, a4, a5, -a6, -a7, -a8]
        at[0] = self.vec[0];
        at[1] = self.vec[1];
        at[2] = self.vec[2];
        at[3] = self.vec[3];
        if N > 4 {
            at[4] = self.vec[4];
            at[5] = self.vec[5];
        }
        if N > 6 {
            at[6] = -self.vec[6];
            at[7] = -self.vec[7];
            at[8] = -self.vec[8];
        }
    }

    /// Calculates the inverse tensor
    ///
    /// ```text
    /// A⁻¹ = inverse(A)
    ///
    /// A · A⁻¹ = I
    /// ```
    ///
    /// # Output
    ///
    /// * `ai` -- a Tensor2 to hold the inverse tensor.
    ///
    /// # Input
    ///
    /// * `tolerance` -- a tolerance for the determinant such that the inverse is computed only if |det| > tolerance
    ///
    /// # Returns
    ///
    /// * If the determinant is zero, the inverse is not computed and returns `None`
    /// * Otherwise, the inverse is computed and returns the determinant
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{approx_eq, mat_approx_eq, mat_mat_mul, Matrix};
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [6.0,  1.0,  2.0],
    ///         [3.0, 12.0,  4.0],
    ///         [5.0,  6.0, 15.0],
    ///     ])?;
    ///
    ///     let mut ai = Tensor2::<9>::new();
    ///
    ///     if let Some(det) = a.inverse(&mut ai, 1e-10) {
    ///         assert_eq!(det, 827.0);
    ///     } else {
    ///         panic!("determinant is zero");
    ///     }
    ///
    ///     let a_mat = a.as_std_matrix();
    ///     let ai_mat = ai.as_std_matrix();
    ///     let mut a_times_ai = Matrix::new(3, 3);
    ///     mat_mat_mul(&mut a_times_ai, 1.0, &a_mat, &ai_mat, 0.0)?;
    ///
    ///     let ii = Matrix::diagonal(&[1.0, 1.0, 1.0]);
    ///     mat_approx_eq(&a_times_ai, &ii, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn inverse(&self, ai: &mut Tensor2<N>, tolerance: f64) -> Option<f64> {
        let a = &self.vec;
        match N {
            4 => {
                let det = a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0;
                if f64::abs(det) > tolerance {
                    ai.vec[0] = (a[1] * a[2]) / det;
                    ai.vec[1] = (a[0] * a[2]) / det;
                    ai.vec[2] = (a[0] * a[1] - a[3] * a[3] / 2.0) / det;
                    ai.vec[3] = -((a[2] * a[3]) / det);
                    return Some(det);
                }
            }
            6 => {
                let det = a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0 - (a[0] * a[4] * a[4]) / 2.0
                    + (a[3] * a[4] * a[5]) / SQRT_2
                    - (a[1] * a[5] * a[5]) / 2.0;
                if f64::abs(det) > tolerance {
                    ai.vec[0] = (a[1] * a[2] - a[4] * a[4] / 2.0) / det;
                    ai.vec[1] = (a[0] * a[2] - a[5] * a[5] / 2.0) / det;
                    ai.vec[2] = (a[0] * a[1] - a[3] * a[3] / 2.0) / det;
                    ai.vec[3] = (-2.0 * a[2] * a[3] + SQRT_2 * a[4] * a[5]) / (2.0 * det);
                    ai.vec[4] = (-2.0 * a[0] * a[4] + SQRT_2 * a[3] * a[5]) / (2.0 * det);
                    ai.vec[5] = (SQRT_2 * a[3] * a[4] - 2.0 * a[1] * a[5]) / (2.0 * det);
                    return Some(det);
                }
            }
            _ => {
                let det = a[0] * a[1] * a[2] - (a[2] * a[3] * a[3]) / 2.0 - (a[0] * a[4] * a[4]) / 2.0
                    + (a[3] * a[4] * a[5]) / SQRT_2
                    - (a[1] * a[5] * a[5]) / 2.0
                    + (a[2] * a[6] * a[6]) / 2.0
                    + (a[5] * a[6] * a[7]) / SQRT_2
                    + (a[0] * a[7] * a[7]) / 2.0
                    - (a[4] * a[6] * a[8]) / SQRT_2
                    - (a[3] * a[7] * a[8]) / SQRT_2
                    + (a[1] * a[8] * a[8]) / 2.0;
                if f64::abs(det) > tolerance {
                    ai.vec[0] = (2.0 * a[1] * a[2] - a[4] * a[4] + a[7] * a[7]) / (2.0 * det);
                    ai.vec[1] = (2.0 * a[0] * a[2] - a[5] * a[5] + a[8] * a[8]) / (2.0 * det);
                    ai.vec[2] = (2.0 * a[0] * a[1] - a[3] * a[3] + a[6] * a[6]) / (2.0 * det);
                    ai.vec[3] = -((SQRT_2 * a[2] * a[3] - a[4] * a[5] + a[7] * a[8]) / (SQRT_2 * det));
                    ai.vec[4] = -((SQRT_2 * a[0] * a[4] - a[3] * a[5] + a[6] * a[8]) / (SQRT_2 * det));
                    ai.vec[5] = (a[3] * a[4] - SQRT_2 * a[1] * a[5] + a[6] * a[7]) / (SQRT_2 * det);
                    ai.vec[6] = -((SQRT_2 * a[2] * a[6] + a[5] * a[7] - a[4] * a[8]) / (SQRT_2 * det));
                    ai.vec[7] = -((a[5] * a[6] + SQRT_2 * a[0] * a[7] - a[3] * a[8]) / (SQRT_2 * det));
                    ai.vec[8] = (a[4] * a[6] + a[3] * a[7] - SQRT_2 * a[1] * a[8]) / (SQRT_2 * det);
                    return Some(det);
                }
            }
        }
        None
    }

    /// Calculates the squared tensor
    ///
    /// ```text
    /// A² = A · A
    /// ```
    ///
    /// # Output
    ///
    /// * `a2` -- a Tensor2 to hold the squared tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::mat_approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [10.0, 20.0, 10.0],
    ///         [ 4.0,  5.0,  6.0],
    ///         [ 2.0,  3.0,  5.0],
    ///     ])?;
    ///
    ///     let mut a2 = Tensor2::<9>::new();
    ///     a.squared(&mut a2);
    ///
    ///     let a2_correct = Tensor2::<9>::from_std_matrix(&[
    ///         [200.0, 330.0, 270.0],
    ///         [ 72.0, 123.0, 100.0],
    ///         [ 42.0,  70.0,  63.0],
    ///     ])?;
    ///     mat_approx_eq(&a2.as_std_matrix(), &a2_correct.as_std_matrix(), 1e-12);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn squared(&self, a2: &mut Tensor2<N>) {
        let a = &self.vec;
        match N {
            4 => {
                a2.vec[0] = a[0] * a[0] + a[3] * a[3] / 2.0;
                a2.vec[1] = a[1] * a[1] + a[3] * a[3] / 2.0;
                a2.vec[2] = a[2] * a[2];
                a2.vec[3] = (SQRT_2 * a[0] * a[3] + SQRT_2 * a[1] * a[3]) / SQRT_2;
            }
            6 => {
                a2.vec[0] = a[0] * a[0] + a[3] * a[3] / 2.0 + a[5] * a[5] / 2.0;
                a2.vec[1] = a[1] * a[1] + a[3] * a[3] / 2.0 + a[4] * a[4] / 2.0;
                a2.vec[2] = a[2] * a[2] + a[4] * a[4] / 2.0 + a[5] * a[5] / 2.0;
                a2.vec[3] = a[0] * a[3] + a[1] * a[3] + a[4] * a[5] / SQRT_2;
                a2.vec[4] = a[1] * a[4] + a[2] * a[4] + a[3] * a[5] / SQRT_2;
                a2.vec[5] = a[0] * a[5] + a[2] * a[5] + a[3] * a[4] / SQRT_2;
            }
            _ => {
                a2.vec[0] = a[0] * a[0] + ((a[3] - a[6]) * (a[3] + a[6])) / 2.0 + ((a[5] - a[8]) * (a[5] + a[8])) / 2.0;
                a2.vec[1] = a[1] * a[1] + ((a[3] - a[6]) * (a[3] + a[6])) / 2.0 + ((a[4] - a[7]) * (a[4] + a[7])) / 2.0;
                a2.vec[2] = a[2] * a[2] + ((a[4] - a[7]) * (a[4] + a[7])) / 2.0 + ((a[5] - a[8]) * (a[5] + a[8])) / 2.0;
                a2.vec[3] = ((a[0] * (a[3] - a[6])) / SQRT_2
                    + (a[1] * (a[3] - a[6])) / SQRT_2
                    + (a[0] * (a[3] + a[6])) / SQRT_2
                    + (a[1] * (a[3] + a[6])) / SQRT_2
                    + ((a[4] + a[7]) * (a[5] - a[8])) / 2.0
                    + ((a[4] - a[7]) * (a[5] + a[8])) / 2.0)
                    / SQRT_2;
                a2.vec[4] = ((a[1] * (a[4] - a[7])) / SQRT_2
                    + (a[2] * (a[4] - a[7])) / SQRT_2
                    + (a[1] * (a[4] + a[7])) / SQRT_2
                    + (a[2] * (a[4] + a[7])) / SQRT_2
                    + ((a[3] + a[6]) * (a[5] - a[8])) / 2.0
                    + ((a[3] - a[6]) * (a[5] + a[8])) / 2.0)
                    / SQRT_2;
                a2.vec[5] = ((a[0] * (a[5] + a[8])) / SQRT_2
                    + (a[2] * (a[5] + a[8])) / SQRT_2
                    + (a[0] * (a[5] - a[8])) / SQRT_2
                    + (a[2] * (a[5] - a[8])) / SQRT_2
                    + ((a[3] - a[6]) * (a[4] - a[7])) / 2.0
                    + ((a[3] + a[6]) * (a[4] + a[7])) / 2.0)
                    / SQRT_2;
                a2.vec[6] = (-(a[0] * (a[3] - a[6])) / SQRT_2 - (a[1] * (a[3] - a[6])) / SQRT_2
                    + (a[0] * (a[3] + a[6])) / SQRT_2
                    + (a[1] * (a[3] + a[6])) / SQRT_2
                    - ((a[4] + a[7]) * (a[5] - a[8])) / 2.0
                    + ((a[4] - a[7]) * (a[5] + a[8])) / 2.0)
                    / SQRT_2;
                a2.vec[7] = (-(a[1] * (a[4] - a[7])) / SQRT_2 - (a[2] * (a[4] - a[7])) / SQRT_2
                    + (a[1] * (a[4] + a[7])) / SQRT_2
                    + (a[2] * (a[4] + a[7])) / SQRT_2
                    - ((a[3] + a[6]) * (a[5] - a[8])) / 2.0
                    + ((a[3] - a[6]) * (a[5] + a[8])) / 2.0)
                    / SQRT_2;
                a2.vec[8] = (-(a[0] * (a[5] - a[8])) / SQRT_2 - (a[2] * (a[5] - a[8])) / SQRT_2
                    + (a[0] * (a[5] + a[8])) / SQRT_2
                    + (a[2] * (a[5] + a[8])) / SQRT_2
                    - ((a[3] - a[6]) * (a[4] - a[7])) / 2.0
                    + ((a[3] + a[6]) * (a[4] + a[7])) / 2.0)
                    / SQRT_2;
            }
        }
    }

    /// Calculates the trace
    ///
    /// ```text
    /// tr(σ) = σ:I = Σᵢ σᵢᵢ
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ])?;
    ///
    ///     approx_eq(a.trace(), 15.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn trace(&self) -> f64 {
        self.vec[0] + self.vec[1] + self.vec[2]
    }

    /// Calculates the Euclidean norm
    ///
    /// ```text
    /// norm(σ) = √(σ:σ)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ])?;
    ///
    ///     approx_eq(a.norm(), f64::sqrt(285.0), 1e-13);
    ///     Ok(())
    /// }
    /// ```
    pub fn norm(&self) -> f64 {
        let mut sm = self.vec[0] * self.vec[0]
            + self.vec[1] * self.vec[1]
            + self.vec[2] * self.vec[2]
            + self.vec[3] * self.vec[3];
        if N > 4 {
            sm += self.vec[4] * self.vec[4] + self.vec[5] * self.vec[5];
        }
        if N > 6 {
            sm += self.vec[6] * self.vec[6] + self.vec[7] * self.vec[7] + self.vec[8] * self.vec[8];
        }
        f64::sqrt(sm)
    }

    /// Calculates the deviator tensor
    ///
    /// ```text
    /// dev(σ) = σ - ⅓ tr(σ) I
    /// ```
    ///
    /// # Output
    ///
    /// * `dev` -- a Tensor2 to hold the deviator tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 2.0, 3.0],
    ///         [4.0, 5.0, 6.0],
    ///         [7.0, 8.0, 9.0],
    ///     ])?;
    ///
    ///     let mut dev = Tensor2::<9>::new();
    ///     a.deviator(&mut dev);
    ///     approx_eq(dev.trace(), 0.0, 1e-15);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", dev.as_std_matrix()),
    ///         "┌                ┐\n\
    ///          │ -4.0  2.0  3.0 │\n\
    ///          │  4.0  0.0  6.0 │\n\
    ///          │  7.0  8.0  4.0 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn deviator(&self, dev: &mut Tensor2<N>) {
        self.deviator_slice(dev.as_mut_data());
    }

    /// Returns the deviator tensor components in a stack-allocated array (crate-internal)
    ///
    /// Mirrors [deviator] but returns the components instead of writing to a [Tensor2].
    #[inline]
    pub(crate) fn deviator_slice(&self, dev: &mut [f64]) {
        let m = (self.vec[0] + self.vec[1] + self.vec[2]) / 3.0;
        dev[0] = self.vec[0] - m;
        dev[1] = self.vec[1] - m;
        dev[2] = self.vec[2] - m;
        dev[3] = self.vec[3];
        if N > 4 {
            dev[4] = self.vec[4];
            dev[5] = self.vec[5];
        }
        if N > 6 {
            dev[6] = self.vec[6];
            dev[7] = self.vec[7];
            dev[8] = self.vec[8];
        }
        let new_trace_s = dev[0] + dev[1] + dev[2];
        if f64::abs(new_trace_s) > 1e-10 {
            // fix error due to large magnitudes
            let mut v = (f64::abs(self.vec[0]), f64::abs(self.vec[1]), f64::abs(self.vec[2]));
            sort3(&mut v);
            let d = f64::max(1.0, v.2);
            let m = (self.vec[0] / d + self.vec[1] / d + self.vec[2] / d) / 3.0;
            dev[0] = (self.vec[0] / d - m) * d;
            dev[1] = (self.vec[1] / d - m) * d;
            dev[2] = (self.vec[2] / d - m) * d;
        }
    }

    /// Calculates the norm of the deviator tensor
    ///
    /// ```text
    /// norm(dev(σ)) = ‖s‖ = ‖ σ - ⅓ tr(σ) I ‖
    ///
    /// ‖s‖² = ⅓ [(σ₁₁-σ₂₂)² + (σ₂₂-σ₃₃)² + (σ₃₃-σ₁₁)²]
    ///       + σ₁₂² + σ₂₃² + σ₁₃² + σ₂₁² + σ₃₂² + σ₃₁²
    /// ```
    ///
    /// Also the radius in the octahedral plane is:
    ///
    /// ```text
    /// r = ‖s‖
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [6.0,  1.0,  2.0],
    ///         [3.0, 12.0,  4.0],
    ///         [5.0,  6.0, 15.0],
    ///     ])?;
    ///
    ///     let mut dev = Tensor2::<9>::new();
    ///     a.deviator(&mut dev);
    ///     approx_eq(dev.trace(), 0.0, 1e-15);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", dev.as_std_matrix()),
    ///         "┌                ┐\n\
    ///          │ -5.0  1.0  2.0 │\n\
    ///          │  3.0  1.0  4.0 │\n\
    ///          │  5.0  6.0  4.0 │\n\
    ///          └                ┘"
    ///     );
    ///
    ///     approx_eq(dev.norm(), f64::sqrt(133.0), 1e-15);
    ///     approx_eq(a.deviator_norm(), f64::sqrt(133.0), 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn deviator_norm(&self) -> f64 {
        let a = &self.vec;
        let mut sq_norm_s = a[3] * a[3]
            + (a[0] - a[1]) * (a[0] - a[1]) / 3.0
            + (a[1] - a[2]) * (a[1] - a[2]) / 3.0
            + (a[2] - a[0]) * (a[2] - a[0]) / 3.0;
        if N > 4 {
            sq_norm_s += a[4] * a[4] + a[5] * a[5];
        }
        if N > 6 {
            sq_norm_s += a[6] * a[6] + a[7] * a[7] + a[8] * a[8];
        }
        f64::sqrt(sq_norm_s)
    }

    /// Calculates the determinant of the deviator tensor
    ///
    /// ```text
    /// det( σ - ⅓ tr(σ) I )
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [6.0,  1.0,  2.0],
    ///         [3.0, 12.0,  4.0],
    ///         [5.0,  6.0, 15.0],
    ///     ])?;
    ///
    ///     let mut dev = Tensor2::<9>::new();
    ///     a.deviator(&mut dev);
    ///     approx_eq(dev.trace(), 0.0, 1e-15);
    ///
    ///     assert_eq!(
    ///         format!("{:.1}", dev.as_std_matrix()),
    ///         "┌                ┐\n\
    ///          │ -5.0  1.0  2.0 │\n\
    ///          │  3.0  1.0  4.0 │\n\
    ///          │  5.0  6.0  4.0 │\n\
    ///          └                ┘"
    ///     );
    ///
    ///     approx_eq(dev.determinant(), 134.0, 1e-13);
    ///     approx_eq(a.deviator_determinant(), 134.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn deviator_determinant(&self) -> f64 {
        let a = &self.vec;
        let m = (a[0] + a[1] + a[2]) / 3.0;
        match N {
            4 => (a[2] - m) * (m * m + a[0] * a[1] - m * (a[0] + a[1]) - a[3] * a[3] / 2.0),
            6 => {
                (2.0 * m * m * (a[0] + a[1] + a[2]) - a[2] * a[3] * a[3] + a[0] * (2.0 * a[1] * a[2] - a[4] * a[4])
                    - 2.0 * m * m * m
                    + SQRT_2 * a[3] * a[4] * a[5]
                    - a[1] * a[5] * a[5]
                    + m * (-2.0 * a[1] * a[2] - 2.0 * a[0] * (a[1] + a[2]) + a[3] * a[3] + a[4] * a[4] + a[5] * a[5]))
                    / 2.0
            }
            _ => {
                (2.0 * (a[2] - m)
                    * (2.0 * m * m + 2.0 * a[0] * a[1] - 2.0 * m * (a[0] + a[1]) - a[3] * a[3] + a[6] * a[6])
                    + SQRT_2 * (a[5] - a[8]) * ((a[3] + a[6]) * (a[4] + a[7]) + SQRT_2 * (m - a[1]) * (a[5] + a[8]))
                    + SQRT_2 * (a[4] - a[7]) * ((a[3] - a[6]) * (a[5] + a[8]) + SQRT_2 * (m - a[0]) * (a[4] + a[7])))
                    / 4.0
            }
        }
    }

    /// Decomposes this tensor into symmetric and skew-symmetric parts
    ///
    /// * A symmetric Tensor2 is defined by Sᵀ = S
    /// * A skew-symmetric Tensor2 is defined by Wᵀ = -W
    ///
    /// For this tensor (A):
    ///
    /// ```text
    /// S := sym(A) = (A + Aᵀ) / 2
    /// W := skw(A) = (A - Aᵀ) / 2
    /// ```
    pub fn decompose(&self, sym: &mut Tensor2<N>, skw: &mut Tensor2<N>) {
        if N == 9 {
            // The symmetric part is given by:
            // [a0, a1, a2, a3, a4, a5, 0, 0, 0]
            // The skew-symmetric part is given by:
            // [0, 0, 0, 0, 0, 0, a6, a7, a8]
            for m in 0..6 {
                sym.vec[m] = self.vec[m];
                skw.vec[m] = 0.0;
            }
            skw.set(6, self.vec[6]);
            skw.set(7, self.vec[7]);
            skw.set(8, self.vec[8]);
        } else {
            // There is only symmetric part
            for m in 0..N {
                sym.vec[m] = self.vec[m];
                skw.vec[m] = 0.0;
            }
        }
    }

    /// Calculates the axial vector omega associated with the skew-symmetric part of this tensor
    ///
    /// The axial vector omega satisfies the following equation:
    ///
    /// ```text
    /// skw . u = omega × u
    ///
    /// For all u (vector) in R3
    /// ```
    ///
    /// The axial vector is given by the following (standard) components
    /// of the skew-symmetric part of the tensor:
    ///
    /// ```text
    /// omega = [−skw_12, skw_02, −skw_01]
    /// ```
    pub fn axial_vector(&self, omega: &mut Tensor1) {
        if N == 9 {
            // The skew-symmetric part is given by:
            // skw_kelvin = [  0,  0,  0,   0,  0,  0,  a6, a7, a8]
            //                00  11  22   01  12  02   10  21  20
            // Converted back to standard basis, it is given by:
            //           [   0     a6/√2   a8/√2]
            // skw_std = [-a6/√2     0     a7/√2]
            //           [-a8/√2  -a7/√2     0  ]
            omega.set(0, -self.vec[7] / SQRT_2); // convert back to standard basis
            omega.set(1, self.vec[8] / SQRT_2); // convert back to standard basis
            omega.set(2, -self.vec[6] / SQRT_2); // convert back to standard basis
        } else {
            omega.set(0, 0.0);
            omega.set(1, 0.0);
            omega.set(2, 0.0);
        }
    }

    // --- PRINCIPAL INVARIANTS -------------------------------------------------------------------------------------------

    /// Calculates I1, the first principal invariant
    ///
    /// ```text
    /// I1 = trace(σ)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let sig = Tensor2::<9>::from_std_matrix(&[
    ///         [50.0,  30.0,  20.0],
    ///         [30.0, -20.0, -10.0],
    ///         [20.0, -10.0,  10.0],
    ///     ])?;
    ///     approx_eq(sig.invariant_ii1(), 40.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_ii1(&self) -> f64 {
        self.trace()
    }

    /// Calculates I2, the second principal invariant
    ///
    /// ```text
    /// I2 = ½ (trace(σ))² - ½ trace(σ·σ)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let sig = Tensor2::<9>::from_std_matrix(&[
    ///         [50.0,  30.0,  20.0],
    ///         [30.0, -20.0, -10.0],
    ///         [20.0, -10.0,  10.0],
    ///     ])?;
    ///     approx_eq(sig.invariant_ii2(), -2100.0, 1e-12);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_ii2(&self) -> f64 {
        let a = &self.vec;
        let mut ii2 = a[0] * a[1] + a[0] * a[2] + a[1] * a[2] - a[3] * a[3] / 2.0;
        if N > 4 {
            ii2 -= (a[4] * a[4] + a[5] * a[5]) / 2.0;
        }
        if N > 6 {
            ii2 += (a[6] * a[6] + a[7] * a[7] + a[8] * a[8]) / 2.0;
        }
        ii2
    }

    /// Calculates I3, the third principal invariant
    ///
    /// ```text
    /// I3 = determinant(σ)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let sig = Tensor2::<9>::from_std_matrix(&[
    ///         [50.0,  30.0,  20.0],
    ///         [30.0, -20.0, -10.0],
    ///         [20.0, -10.0,  10.0],
    ///     ])?;
    ///     approx_eq(sig.invariant_ii3(), -28000.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_ii3(&self) -> f64 {
        self.determinant()
    }

    /// Calculates J2, the second invariant of the deviatoric tensor corresponding to this tensor
    ///
    /// ```text
    /// s = deviator(σ)
    ///
    /// J2 = -IIₛ = ½ trace(s·s) = ½ s : sᵀ
    /// ```
    ///
    /// **Note:** if the tensor is symmetric, then:
    ///
    /// ```text
    /// J2 = ½ s : sᵀ = ½ s : s = ½ ‖s‖² (symmetric σ and s)
    /// ```
    ///
    /// Thus:
    ///
    /// ```text
    /// J2 = ½ r²
    /// ```
    ///
    /// where `r = ‖s‖` is the radius on the octahedral plane.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let sig = Tensor2::<9>::from_std_matrix(&[
    ///         [ 2.0, -3.0, 4.0],
    ///         [-3.0, -5.0, 1.0],
    ///         [ 4.0,  1.0, 6.0],
    ///     ])?;
    ///     approx_eq(sig.invariant_jj2(), 57.0, 1e-14);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_jj2(&self) -> f64 {
        let a = &self.vec;
        match N {
            4 => {
                (2.0 * (a[0] * a[0] + a[1] * a[1] - a[1] * a[2] + a[2] * a[2] - a[0] * (a[1] + a[2]))
                    + 3.0 * a[3] * a[3])
                    / 6.0
            }
            6 => {
                (2.0 * (a[0] * a[0] + a[1] * a[1] - a[1] * a[2] + a[2] * a[2] - a[0] * (a[1] + a[2]))
                    + 3.0 * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5]))
                    / 6.0
            }
            _ => {
                (2.0 * (a[0] * a[0] + a[1] * a[1] - a[1] * a[2] + a[2] * a[2] - a[0] * (a[1] + a[2]))
                    + 3.0 * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5] - a[6] * a[6] - a[7] * a[7] - a[8] * a[8]))
                    / 6.0
            }
        }
    }

    /// Calculates J3, the second invariant of the deviatoric tensor corresponding to this tensor
    ///
    /// ```text
    /// s = deviator(σ)
    ///
    /// J3 = IIIₛ = determinant(s)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let sig = Tensor2::<9>::from_std_matrix(&[
    ///         [ 2.0, -3.0, 4.0],
    ///         [-3.0, -5.0, 1.0],
    ///         [ 4.0,  1.0, 6.0],
    ///     ])?;
    ///     approx_eq(sig.invariant_jj3(), -4.0, 1e-13);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_jj3(&self) -> f64 {
        self.deviator_determinant()
    }

    // --- OCTAHEDRAL INVARIANTS ------------------------------------------------------------------------------------------

    /// Returns the isomorphic mean pressure invariant (distance to octahedral plane)
    ///
    /// ```text
    /// σs = d = trace(σ) / √3
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{approx_eq, math::SQRT_3};
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ])?;
    ///     approx_eq(a.invariant_sigma_s(), 2.0 / SQRT_3, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_sigma_s(&self) -> f64 {
        self.trace() / SQRT_3
    }

    /// Returns the isomorphic deviatoric stress invariant (radius on octahedral plane)
    ///
    /// ```text
    /// σt = r = ‖s‖ = √(2 J2)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{approx_eq, math::SQRT_2_BY_3};
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ])?;
    ///     approx_eq(a.invariant_sigma_t(), SQRT_2_BY_3, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_sigma_t(&self) -> f64 {
        self.deviator_norm()
    }

    /// Returns the mean pressure invariant
    ///
    /// ```text
    /// p = ⅓ trace(σ) = d / √3
    /// ```
    ///
    /// where `d = trace(σ) / √3` is the distance from the octahedral plane to the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ])?;
    ///     approx_eq(a.invariant_p(), 2.0 / 3.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_p(&self) -> f64 {
        self.trace() / 3.0
    }

    /// Returns the deviatoric stress invariant (von Mises)
    ///
    /// This quantity is also known as the **von Mises** effective invariant
    /// or equivalent stress.
    ///
    /// ```text
    /// q = ‖s‖ √3/√2 = r √3/√2 = √3 √J2
    /// ```
    ///
    /// where `r = ‖s‖` is the radius on the octahedral plane.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ])?;
    ///     approx_eq(a.invariant_q(), 1.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_q(&self) -> f64 {
        self.deviator_norm() * SQRT_3_BY_2
    }

    /// Returns the isomorphic mean strain invariant (distance to octahedral plane)
    ///
    /// ```text
    /// εs = d = trace(ε) / √3
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{approx_eq, math::SQRT_3};
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ])?;
    ///     approx_eq(a.invariant_eps_s(), 2.0 / SQRT_3, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_eps_s(&self) -> f64 {
        self.trace() / SQRT_3
    }

    /// Returns the isomorphic deviatoric strain invariant (radius on octahedral plane)
    ///
    /// ```text
    /// εt = r = ‖e‖
    ///
    /// e = deviator(ε)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{approx_eq, math::SQRT_2_BY_3};
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ])?;
    ///     approx_eq(a.invariant_eps_t(), SQRT_2_BY_3, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_eps_t(&self) -> f64 {
        self.deviator_norm()
    }

    /// Returns the volumetric strain invariant
    ///
    /// ```text
    /// εv = trace(ε) = d √3
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ])?;
    ///     approx_eq(a.invariant_eps_v(), 2.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_eps_v(&self) -> f64 {
        self.trace()
    }

    /// Returns the deviatoric strain invariant
    ///
    /// ```text
    /// εd = norm(dev(ε)) × √2/√3 = r √2/√3
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ])?;
    ///     approx_eq(a.invariant_eps_d(), 2.0 / 3.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_eps_d(&self) -> f64 {
        self.deviator_norm() * SQRT_2_BY_3
    }

    /// Returns the Lode invariant
    ///
    /// ```text
    ///                  3 √3 J3
    /// l = cos(3θ) = ─────────────
    ///               2 pow(J2,1.5)
    /// ```
    ///
    /// # Returns
    ///
    /// If `J2 > TOL_J2`, returns `l`. Otherwise, returns None.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::approx_eq;
    /// use russell_tensor::{Tensor2, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let a = Tensor2::<9>::from_std_matrix(&[
    ///         [1.0, 0.0, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///         [0.0, 0.0, 1.0],
    ///     ])?;
    ///     if let Some(l) = a.invariant_lode() {
    ///         approx_eq(l, -1.0, 1e-15);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn invariant_lode(&self) -> Option<f64> {
        let jj2 = self.invariant_jj2();
        if jj2 > TOL_J2 {
            let jj3 = self.invariant_jj3();
            Some(1.5 * SQRT_3 * jj3 / f64::powf(jj2, 1.5))
        } else {
            None
        }
    }

    /// Calculates the octahedral invariants
    ///
    /// # Input
    ///
    /// Returns `(distance, radius, lode)` where:
    ///
    /// * `distance` -- distance `d` from the octahedral plane to the origin
    /// * `radius` -- radius `r` on the octahedral plane
    /// * `lode` -- Lode invariant `l` in `-1 ≤ lode ≤ 1`
    ///
    /// # Returns
    ///
    /// If `J2 > TOL_J2`, returns `l`. Otherwise, returns None.
    ///
    /// # Definitions
    ///
    /// ```text
    /// d = trace(T) / √3 = σs
    /// r = ‖dev(T)‖ = σt
    /// l = cos(3θ) = (3 √3 J3)/(2 pow(J2,1.5))
    /// ```
    pub fn invariants_octahedral(&self) -> (f64, f64, Option<f64>) {
        let distance = self.invariant_ii1() / SQRT_3;
        let radius = self.deviator_norm();
        let lode = self.invariant_lode();
        (distance, radius, lode)
    }
}

impl<const N: usize> fmt::Display for Tensor2<N> {
    /// Generates a string representation of the Kelvin-Mandel vector associated with this Tensor2
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for m in 0..N {
            let val = self.get(m);
            match f.precision() {
                Some(v) => write!(&mut buf, "{:.1$}", val, v).unwrap(),
                None => write!(&mut buf, "{}", val).unwrap(),
            }
            width = cmp::max(buf.chars().count(), width);
            buf.clear();
        }
        // draw vector
        width += 1;
        write!(f, "┌{:1$}┐\n", " ", width + 1).unwrap();
        for m in 0..N {
            if m > 0 {
                write!(f, " │\n").unwrap();
            }
            write!(f, "│").unwrap();
            let val = self.get(m);
            match f.precision() {
                Some(v) => write!(f, "{:>1$.2$}", val, width, v).unwrap(),
                None => write!(f, "{:>1$}", val, width).unwrap(),
            }
        }
        write!(f, " │\n").unwrap();
        write!(f, "└{:1$}┘", " ", width + 1).unwrap();
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Tensor2;
    use crate::{IDENTITY2, SQRT_2, SQRT_2_BY_3, SQRT_3, SQRT_3_BY_2, SQRT_6};
    use crate::{SampleTensor2, SamplesTensor2, Tensor1};
    use russell_lab::{Matrix, Vector, approx_eq, mat_approx_eq, mat_mat_mul, math::PI, vec_approx_eq};

    fn kelvin_vector<const N: usize>(tt: &Tensor2<N>) -> Vec<f64> {
        let mut v = vec![0.0; N];
        for m in 0..N {
            v[m] = tt.get(m);
        }
        v
    }

    #[test]
    fn new_set_and_get_work() {
        // general
        let mut tt = Tensor2::<9>::new();
        tt.set(0, 123.0);
        assert_eq!(tt.dim(), 9);
        assert_eq!(tt.get(0), 123.0);

        // symmetric 3D
        let mut tt = Tensor2::<6>::new();
        tt.set(0, 123.0);
        assert_eq!(tt.dim(), 6);
        assert_eq!(tt.get(0), 123.0);

        let mut tt = Tensor2::<6>::new();
        tt.set(0, 123.0);
        assert_eq!(tt.dim(), 6);
        assert_eq!(tt.get(0), 123.0);

        let mut tt = Tensor2::<6>::new();
        tt.set(0, 123.0);
        assert_eq!(tt.dim(), 6);
        assert_eq!(tt.get(0), 123.0);

        // symmetric 2D
        let mut tt = Tensor2::<4>::new();
        tt.set(0, 123.0);
        assert_eq!(tt.dim(), 4);
        assert_eq!(tt.get(0), 123.0);

        let mut tt = Tensor2::<4>::new();
        tt.set(0, 123.0);
        assert_eq!(tt.dim(), 4);
        assert_eq!(tt.get(0), 123.0);

        let mut tt = Tensor2::<4>::new();
        tt.set(0, 123.0);
        assert_eq!(tt.dim(), 4);
        assert_eq!(tt.get(0), 123.0);
    }

    #[test]
    fn set_std_matrix_captures_errors() {
        // symmetric 3D
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
        let mut tt = Tensor2::<6>::new();
        assert_eq!(
            tt.set_std_matrix(comps_std_10).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );
        assert_eq!(
            tt.set_std_matrix(comps_std_20).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );
        assert_eq!(
            tt.set_std_matrix(comps_std_21).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );

        // symmetric 2D
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std_12 = &[
            [1.0,     4.0, 0.0+eps],
            [4.0,     2.0, 0.0],
            [0.0+eps, 0.0, 3.0],
        ];
        #[rustfmt::skip]
        let comps_std_02 = &[
            [1.0, 4.0,     0.0],
            [4.0, 2.0,     0.0+eps],
            [0.0, 0.0+eps, 3.0],
        ];
        let mut tt = Tensor2::<4>::new();
        assert_eq!(
            tt.set_std_matrix(comps_std_12).err(),
            Some("cannot set Symmetric2D Tensor2 with non-zero off-diagonal data")
        );
        assert_eq!(
            tt.set_std_matrix(comps_std_02).err(),
            Some("cannot set Symmetric2D Tensor2 with non-zero off-diagonal data")
        );
    }

    #[test]
    fn set_std_matrix_works() {
        // general
        let mut tt = Tensor2::<9>::new();
        const NOISE: f64 = 1234.568;
        tt.vec.fill(NOISE);
        tt.set_std_matrix(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            .unwrap();
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
        for m in 0..tt.dim() {
            approx_eq(tt.get(m), correct[m], 1e-15);
        }

        // general (using nested Vec)
        let mut tt = Tensor2::<9>::new();
        tt.vec.fill(NOISE);
        tt.set_std_matrix(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]])
            .unwrap();
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
        for m in 0..tt.dim() {
            approx_eq(tt.get(m), correct[m], 1e-15);
        }

        // symmetric 3D
        let mut tt = Tensor2::<6>::new();
        tt.vec.fill(NOISE);
        tt.set_std_matrix(&[[1.0, 4.0, 6.0], [4.0, 2.0, 5.0], [6.0, 5.0, 3.0]])
            .unwrap();
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0 * SQRT_2];
        for m in 0..tt.dim() {
            approx_eq(tt.get(m), correct[m], 1e-14);
        }

        // symmetric 2D
        let mut tt = Tensor2::<4>::new();
        tt.vec.fill(NOISE);
        tt.set_std_matrix(&[[1.0, 4.0, 0.0], [4.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
            .unwrap();
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2];
        for m in 0..tt.dim() {
            approx_eq(tt.get(m), correct[m], 1e-14);
        }
    }

    #[test]
    fn from_std_matrix_captures_errors() {
        // symmetric 3D
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
            Tensor2::<6>::from_std_matrix(comps_std_10).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );
        assert_eq!(
            Tensor2::<6>::from_std_matrix(comps_std_20).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );
        assert_eq!(
            Tensor2::<6>::from_std_matrix(comps_std_21).err(),
            Some("cannot set symmetric Tensor2 with non-symmetric data")
        );

        // symmetric 2D
        let eps = 1e-15;
        #[rustfmt::skip]
        let comps_std_12 = &[
            [1.0,     4.0, 0.0+eps],
            [4.0,     2.0, 0.0],
            [0.0+eps, 0.0, 3.0],
        ];
        #[rustfmt::skip]
        let comps_std_02 = &[
            [1.0, 4.0,     0.0],
            [4.0, 2.0,     0.0+eps],
            [0.0, 0.0+eps, 3.0],
        ];
        assert_eq!(
            Tensor2::<4>::from_std_matrix(comps_std_12).err(),
            Some("cannot set Symmetric2D Tensor2 with non-zero off-diagonal data")
        );
        assert_eq!(
            Tensor2::<4>::from_std_matrix(comps_std_02).err(),
            Some("cannot set Symmetric2D Tensor2 with non-zero off-diagonal data")
        );
    }

    #[test]
    fn from_std_matrix_works() {
        // general -- example 1
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
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
        for m in 0..tt.dim() {
            approx_eq(tt.get(m), correct[m], 1e-14);
        }

        // general -- example 2
        let tt = Tensor2::<9>::from_std_matrix(&[
            [1.0, SQRT_2 * 2.0, SQRT_2 * 3.0],
            [SQRT_2 * 4.0, 5.0, SQRT_2 * 6.0],
            [SQRT_2 * 7.0, SQRT_2 * 8.0, 9.0],
        ])
        .unwrap();
        assert_eq!(
            format!("{:.1}", tt),
            "┌      ┐\n\
             │  1.0 │\n\
             │  5.0 │\n\
             │  9.0 │\n\
             │  6.0 │\n\
             │ 14.0 │\n\
             │ 10.0 │\n\
             │ -2.0 │\n\
             │ -2.0 │\n\
             │ -4.0 │\n\
             └      ┘"
        );

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0 * SQRT_2];
        for m in 0..tt.dim() {
            approx_eq(tt.get(m), correct[m], 1e-14);
        }

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        let correct = &[1.0, 2.0, 3.0, 4.0 * SQRT_2];
        for m in 0..tt.dim() {
            approx_eq(tt.get(m), correct[m], 1e-14);
        }
    }

    #[test]
    fn identity_works() {
        // general
        let ii = Tensor2::<9>::identity();
        for m in 0..ii.dim() {
            assert_eq!(ii.get(m), IDENTITY2[m]);
        }

        // symmetric
        let ii = Tensor2::<6>::identity();
        for m in 0..ii.dim() {
            assert_eq!(ii.get(m), IDENTITY2[m]);
        }

        // symmetric 2d
        let ii = Tensor2::<4>::identity();
        for m in 0..ii.dim() {
            assert_eq!(ii.get(m), IDENTITY2[m]);
        }
    }

    #[test]
    #[should_panic(expected = "the len is 3 but the index is 3")]
    fn get_std_panics_on_incorrect_input() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        tt.get_std(3, 3);
    }

    #[test]
    fn get_std_works() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(tt.get_std(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(tt.get_std(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(tt.get_std(i, j), comps_std[i][j], 1e-14);
            }
        }
    }

    #[test]
    #[should_panic]
    fn to_std_matrix_panics_on_incorrect_input() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        let mut mat = Matrix::new(2, 2);
        tt.to_std_matrix(&mut mat);
    }

    #[test]
    fn as_std_matrix_and_to_std_matrix_work() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        let res = tt.as_std_matrix();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        let res = tt.as_std_matrix();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        let res = tt.as_std_matrix();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get(i, j), comps_std[i][j], 1e-14);
            }
        }
    }

    #[test]
    fn from_std_matrix_to_std_matrix_from_std_matrix_work() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        let m1 = tt.as_std_matrix();
        mat_approx_eq(&m1, comps_std, 1e-13);
        let ee = Tensor2::<9>::from_std_matrix(&m1).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, comps_std, 1e-13);

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        let m1 = tt.as_std_matrix();
        mat_approx_eq(&m1, comps_std, 1e-13);
        let ee = Tensor2::<6>::from_std_matrix(&m1).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, comps_std, 1e-13);

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        let m1 = tt.as_std_matrix();
        mat_approx_eq(&m1, comps_std, 1e-13);
        let ee = Tensor2::<4>::from_std_matrix(&m1).unwrap();
        let m2 = ee.as_std_matrix();
        mat_approx_eq(&m2, comps_std, 1e-13);
    }

    #[test]
    #[should_panic]
    fn as_std_matrix_2d_panics_on_3d() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        tt.as_std_matrix_2d();
    }

    #[test]
    fn as_std_matrix_2d_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        let (t22, res) = tt.as_std_matrix_2d();
        assert_eq!(t22, 3.0);
        assert_eq!(
            format!("{:.1}", res),
            "┌         ┐\n\
             │ 1.0 4.0 │\n\
             │ 4.0 2.0 │\n\
             └         ┘"
        );

        #[rustfmt::skip]
        let data = &[
            [1.0, 2.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let tt = Tensor2::<4>::from_std_matrix(data).unwrap();
        let (t22, a) = tt.as_std_matrix_2d();
        assert_eq!(t22, 4.0);
        assert_eq!(
            format!("{:.1}", a),
            "┌         ┐\n\
             │ 1.0 2.0 │\n\
             │ 2.0 3.0 │\n\
             └         ┘"
        );
    }

    #[test]
    fn as_general_works() {
        let tt = Tensor2::<4>::from_std_matrix(&[[1.0, 2.0 / SQRT_2, 0.0], [2.0 / SQRT_2, 3.0, 0.0], [0.0, 0.0, 4.0]])
            .unwrap();
        let tt_gen = tt.as_general();
        assert_eq!(format!("{:.2?}", kelvin_vector(&tt)), "[1.00, 3.00, 4.00, 2.00]");
        assert_eq!(
            format!("{:.2?}", kelvin_vector(&tt_gen)),
            "[1.00, 3.00, 4.00, 2.00, 0.00, 0.00, 0.00, 0.00, 0.00]"
        );

        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        let res = tt.as_general();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get_std(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        let res = tt.as_general();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get_std(i, j), comps_std[i][j], 1e-14);
            }
        }

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        let res = tt.as_general();
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(res.get_std(i, j), comps_std[i][j], 1e-14);
            }
        }
    }

    #[test]
    #[should_panic]
    fn sym2d_as_symmetric_panics_on_non_sym2d() {
        let tt = Tensor2::<6>::new();
        tt.sym2d_as_symmetric();
    }

    #[test]
    fn sym2d_as_symmetric_works() {
        let tt = Tensor2::<4>::from_std_matrix(&[[1.0, 2.0 / SQRT_2, 0.0], [2.0 / SQRT_2, 3.0, 0.0], [0.0, 0.0, 4.0]])
            .unwrap();
        let tt_sym = tt.sym2d_as_symmetric();
        assert_eq!(format!("{:.2?}", kelvin_vector(&tt)), "[1.00, 3.00, 4.00, 2.00]");
        assert_eq!(
            format!("{:.2?}", kelvin_vector(&tt_sym)),
            "[1.00, 3.00, 4.00, 2.00, 0.00, 0.00]"
        );
    }

    #[test]
    #[should_panic(expected = "the len is 3 but the index is 3")]
    fn sym_set_std_panics_on_incorrect_indices() {
        let mut a = Tensor2::<6>::new();
        a.sym_set_std(3, 3, 3.0);
    }

    #[test]
    fn sym_set_std_works() {
        let mut a = Tensor2::<6>::new();
        a.sym_set_std(0, 0, 1.0);
        a.sym_set_std(1, 1, 2.0);
        a.sym_set_std(2, 2, 3.0);
        a.sym_set_std(0, 1, 4.0);
        a.sym_set_std(1, 0, 4.0);
        a.sym_set_std(2, 0, 5.0);
        let out = a.as_std_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌             ┐\n\
             │ 1.0 4.0 5.0 │\n\
             │ 4.0 2.0 0.0 │\n\
             │ 5.0 0.0 3.0 │\n\
             └             ┘"
        );
    }

    #[test]
    fn clear_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let mut a = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        a.clear();
        for m in 0..4 {
            assert_eq!(a.get(m), 0.0);
        }
    }

    #[test]
    #[should_panic(expected = "the len is 3 but the index is 3")]
    fn sym_add_std_panics_on_incorrect_indices() {
        let mut a = Tensor2::<6>::new();
        a.sym_add_std(3, 3, 5.0, 6.0);
    }

    #[test]
    #[should_panic(expected = "i <= j")]
    fn sym_add_std_panics_on_lower_diagonal() {
        let mut a = Tensor2::<4>::new();
        a.sym_add_std(1, 0, 5.0, 6.0);
    }

    #[test]
    fn sym_add_std_works() {
        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let mut a = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        a.sym_add_std(0, 0, 10.0, 10.0);
        a.sym_add_std(1, 1, 10.0, 10.0);
        a.sym_add_std(2, 2, 10.0, 10.0);
        a.sym_add_std(0, 1, 10.0, 10.0); // must not do (1,0)
        let out = a.as_std_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                   ┐\n\
             │ 101.0 104.0   0.0 │\n\
             │ 104.0 102.0   0.0 │\n\
             │   0.0   0.0 103.0 │\n\
             └                   ┘"
        );

        // // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let mut a = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        a.sym_add_std(0, 0, 10.0, 10.0);
        a.sym_add_std(1, 1, 10.0, 10.0);
        a.sym_add_std(2, 2, 10.0, 10.0);
        a.sym_add_std(0, 1, 10.0, 10.0); // must nod do (1,0)
        a.sym_add_std(0, 2, 10.0, 10.0); // must not do (2,0)
        a.sym_add_std(1, 2, 10.0, 10.0); // must not do (2,1)
        let out = a.as_std_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                   ┐\n\
             │ 101.0 104.0 106.0 │\n\
             │ 104.0 102.0 105.0 │\n\
             │ 106.0 105.0 103.0 │\n\
             └                   ┘"
        );
    }

    #[test]
    #[should_panic]
    fn set_vector_panics_on_incorrect_input() {
        let mut a = Tensor2::<4>::new();
        let b = [1.0];
        a.set_vector(2.0, &b);
    }

    #[test]
    fn set_vector_works() {
        // general
        let mut tt = Tensor2::<9>::new();
        const NOISE: f64 = 1234.568;
        tt.vec.fill(NOISE);
        tt.set_vector(
            2.0,
            &[
                1.0,
                5.0,
                9.0,
                6.0 / SQRT_2,
                14.0 / SQRT_2,
                10.0 / SQRT_2,
                -2.0 / SQRT_2,
                -2.0 / SQRT_2,
                -4.0 / SQRT_2,
            ],
        );
        let correct = &[[2.0, 4.0, 6.0], [8.0, 10.0, 12.0], [14.0, 16.0, 18.0]];
        mat_approx_eq(&tt.as_std_matrix(), correct, 1e-14);

        // symmetric 3D
        let mut tt = Tensor2::<6>::new();
        tt.vec.fill(NOISE);
        tt.set_vector(2.0, &[1.0, 2.0, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0 * SQRT_2]);
        let correct = &[[2.0, 8.0, 12.0], [8.0, 4.0, 10.0], [12.0, 10.0, 6.0]];
        mat_approx_eq(&tt.as_std_matrix(), correct, 1e-14);

        // symmetric 2D
        let mut tt = Tensor2::<4>::new();
        tt.vec.fill(NOISE);
        tt.set_vector(2.0, &[1.0, 2.0, 3.0, 4.0 * SQRT_2]);
        let correct = &[[2.0, 8.0, 0.0], [8.0, 4.0, 0.0], [0.0, 0.0, 6.0]];
        mat_approx_eq(&tt.as_std_matrix(), correct, 1e-14);
    }

    #[test]
    fn set_tensor_and_update_work() {
        // general
        let mut a = Tensor2::<9>::new();
        #[rustfmt::skip]
        let b = Tensor2::<9>::from_std_matrix(&[
            [1.0, 3.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 1.0, 3.0],
        ],
        ).unwrap();
        let c = Tensor2::<9>::from_std_matrix(&[[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]])
            .unwrap();
        a.set_tensor(2.0, &b);
        a.update(10.0, &c);
        let out = a.as_std_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │ 1002.0 1006.0 1002.0 │\n\
             │ 1004.0 1004.0 1004.0 │\n\
             │ 1006.0 1002.0 1006.0 │\n\
             └                      ┘"
        );

        // symmetric 3D
        let mut a = Tensor2::<6>::new();
        #[rustfmt::skip]
        let b = Tensor2::<6>::from_std_matrix(&[
            [1.0, 3.0, 1.0],
            [3.0, 2.0, 2.0],
            [1.0, 2.0, 3.0],
        ],
        ).unwrap();
        let c = Tensor2::<6>::from_std_matrix(&[[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]])
            .unwrap();
        a.set_tensor(2.0, &b);
        a.update(10.0, &c);
        let out = a.as_std_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │ 1002.0 1006.0 1002.0 │\n\
             │ 1006.0 1004.0 1004.0 │\n\
             │ 1002.0 1004.0 1006.0 │\n\
             └                      ┘"
        );

        // symmetric 2D
        let mut a = Tensor2::<4>::new();
        #[rustfmt::skip]
        let b = Tensor2::<4>::from_std_matrix(&[
            [1.0, 3.0, 0.0],
            [3.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        ).unwrap();
        let c = Tensor2::<4>::from_std_matrix(&[[100.0, 100.0, 0.0], [100.0, 100.0, 0.0], [0.0, 0.0, 100.0]]).unwrap();
        a.set_tensor(2.0, &b);
        a.update(10.0, &c);
        let out = a.as_std_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │ 1002.0 1006.0    0.0 │\n\
             │ 1006.0 1004.0    0.0 │\n\
             │    0.0    0.0 1006.0 │\n\
             └                      ┘"
        );
    }

    #[test]
    fn clone_and_serialize_work() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        // clone
        let mut cloned = tt.clone();
        cloned.vec[0] = -1.0;
        assert_eq!(
            format!("{:.1}", tt.as_std_matrix()),
            "┌             ┐\n\
             │ 1.0 2.0 3.0 │\n\
             │ 4.0 5.0 6.0 │\n\
             │ 7.0 8.0 9.0 │\n\
             └             ┘"
        );
        assert_eq!(
            format!("{:.1}", cloned.as_std_matrix()),
            "┌                ┐\n\
             │ -1.0  2.0  3.0 │\n\
             │  4.0  5.0  6.0 │\n\
             │  7.0  8.0  9.0 │\n\
             └                ┘"
        );
        // serialize
        let json = serde_json::to_string(&tt).unwrap();
        assert!(json.len() > 0);
        // deserialize
        let from_json: Tensor2<9> = serde_json::from_str(&json).unwrap();
        assert_eq!(
            format!("{:.1}", from_json.as_std_matrix()),
            "┌             ┐\n\
             │ 1.0 2.0 3.0 │\n\
             │ 4.0 5.0 6.0 │\n\
             │ 7.0 8.0 9.0 │\n\
             └             ┘"
        );
    }

    #[test]
    fn debug_works() {
        let tt = Tensor2::<9>::new();
        assert!(format!("{:?}", tt).len() > 0);
    }

    #[test]
    fn determinant_works() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        approx_eq(tt.determinant(), 0.0, 1e-13);

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        approx_eq(tt.determinant(), 101.0, 1e-13);

        // symmetric 3D (another test)
        #[rustfmt::skip]
        let comps_std = &[
            [ 1.0, -3.0, 4.0],
            [-3.0, -6.0, 1.0],
            [ 4.0,  1.0, 5.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        approx_eq(tt.determinant(), -4.0, 1e-13);

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        approx_eq(tt.determinant(), -42.0, 1e-13);
    }

    fn check_transpose<const N: usize>(tt: &Tensor2<N>, tt_tran: &Tensor2<N>) {
        let aa = tt.as_std_matrix();
        let aa_tran = tt_tran.as_std_matrix();
        for i in 1..3 {
            for j in 1..3 {
                assert_eq!(aa.get(i, j), aa_tran.get(j, i));
            }
        }
    }

    #[test]
    fn transpose_works() {
        // general
        let s = &SamplesTensor2::TENSOR_T;
        let tt = Tensor2::<9>::from_std_matrix(&s.matrix).unwrap();
        let mut tt_tra = Tensor2::<9>::new();
        tt.transpose(&mut tt_tra);
        check_transpose(&tt, &tt_tra);

        // symmetric 3D
        let s = &SamplesTensor2::TENSOR_U;
        let tt = Tensor2::<6>::from_std_matrix(&s.matrix).unwrap();
        let mut tt_tra = Tensor2::<6>::new();
        tt.transpose(&mut tt_tra);
        check_transpose(&tt, &tt_tra);

        // symmetric 2D
        let s = &SamplesTensor2::TENSOR_Y;
        let tt = Tensor2::<4>::from_std_matrix(&s.matrix).unwrap();
        let mut tt_tra = Tensor2::<4>::new();
        tt.transpose(&mut tt_tra);
        check_transpose(&tt, &tt_tra);
    }

    fn check_inverse<const N: usize>(tt: &Tensor2<N>, tti: &Tensor2<N>, tol: f64) {
        let aa = tt.as_std_matrix();
        let aai = tti.as_std_matrix();
        let mut ii = Matrix::new(3, 3);
        mat_mat_mul(&mut ii, 1.0, &aa, &aai, 0.0).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    approx_eq(ii.get(i, j), 1.0, tol);
                } else {
                    approx_eq(ii.get(i, j), 0.0, tol);
                }
            }
        }
    }

    #[test]
    fn inverse_works() {
        // general with zero determinant
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        let mut tti = Tensor2::<9>::new();
        let res = tt.inverse(&mut tti, 1e-10);
        assert_eq!(res, None);

        // general with non-zero determinant
        let s = &SamplesTensor2::TENSOR_T;
        let tt = Tensor2::<9>::from_std_matrix(&s.matrix).unwrap();
        let mut tti = Tensor2::<9>::new();
        let det = tt.inverse(&mut tti, 1e-10).unwrap();
        assert_eq!(det, s.determinant);
        check_inverse(&tt, &tti, 1e-15);

        // symmetric 3D with zero determinant
        let s = &SamplesTensor2::TENSOR_X;
        let tt = Tensor2::<6>::from_std_matrix(&s.matrix).unwrap();
        let mut tti = Tensor2::<6>::new();
        let res = tt.inverse(&mut tti, 1e-10);
        assert_eq!(res, None);

        // symmetric 3D
        let s = &SamplesTensor2::TENSOR_U;
        let tt = Tensor2::<6>::from_std_matrix(&s.matrix).unwrap();
        let mut tti = Tensor2::<6>::new();
        let det = tt.inverse(&mut tti, 1e-10).unwrap();
        approx_eq(det, s.determinant, 1e-14);
        check_inverse(&tt, &tti, 1e-13);

        // symmetric 2D with zero determinant
        let s = &SamplesTensor2::TENSOR_X;
        let tt = Tensor2::<4>::from_std_matrix(&s.matrix).unwrap();
        let mut tti = Tensor2::<4>::new();
        let res = tt.inverse(&mut tti, 1e-10);
        assert_eq!(res, None);

        // symmetric 2D
        let s = &SamplesTensor2::TENSOR_Y;
        let tt = Tensor2::<4>::from_std_matrix(&s.matrix).unwrap();
        let mut tti = Tensor2::<4>::new();
        let det = tt.inverse(&mut tti, 1e-10).unwrap();
        assert_eq!(det, s.determinant);
        check_inverse(&tt, &tti, 1e-15);
    }

    fn check_squared<const N: usize>(tt: &Tensor2<N>, tt2: &Tensor2<N>, tol: f64) {
        let aa = tt.as_std_matrix();
        let aa2 = tt2.as_std_matrix();
        let mut aa2_correct = Matrix::new(3, 3);
        mat_mat_mul(&mut aa2_correct, 1.0, &aa, &aa, 0.0).unwrap();
        mat_approx_eq(&aa2, &aa2_correct, tol);
    }

    #[test]
    fn squared_works() {
        // general
        let s = &SamplesTensor2::TENSOR_T;
        let tt = Tensor2::<9>::from_std_matrix(&s.matrix).unwrap();
        let mut tt2 = Tensor2::<9>::new();
        tt.squared(&mut tt2);
        check_squared(&tt, &tt2, 1e-13);

        // symmetric 3D
        let s = &SamplesTensor2::TENSOR_U;
        let tt = Tensor2::<6>::from_std_matrix(&s.matrix).unwrap();
        let mut tt2 = Tensor2::<6>::new();
        tt.squared(&mut tt2);
        check_squared(&tt, &tt2, 1e-14);

        // symmetric 2D
        let s = &SamplesTensor2::TENSOR_Y;
        let tt = Tensor2::<4>::from_std_matrix(&s.matrix).unwrap();
        let mut tt2 = Tensor2::<4>::new();
        tt.squared(&mut tt2);
        check_squared(&tt, &tt2, 1e-15);
    }

    #[test]
    fn trace_works() {
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        approx_eq(tt.trace(), 15.0, 1e-15);
    }

    #[test]
    fn eigenvalues_sym_works() {
        #[rustfmt::skip]
        let a = Tensor2::<6>::from_std_matrix(&[
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 4.0],
            [0.0, 4.0, 9.0],
        ]).unwrap();
        let mut l = Vector::new(3);
        a.eigenvalues_sym(&mut l).unwrap();
        vec_approx_eq(&l, &[1.0, 2.0, 11.0], 1e-13);
    }

    #[test]
    fn eigenvalues_sym_returns_err() {
        let a = Tensor2::<9>::new();
        let mut l = Vector::new(3);
        assert_eq!(a.eigenvalues_sym(&mut l).err(), Some("the tensor must be symmetric"));
        let a = Tensor2::<6>::new();
        let mut l = Vector::new(2);
        assert_eq!(a.eigenvalues_sym(&mut l).err(), Some("l.dim() must be equal to 3"));
    }

    #[test]
    fn eigenvalues_works() {
        // rotation about e3 by 90 degrees: eigenvalues {i, -i, 2}
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 2.0],
        ]).unwrap();
        let mut lr = Vector::new(3);
        let mut li = Vector::new(3);
        a.eigenvalues(&mut lr, &mut li).unwrap();
        let close = |x: f64, y: f64| (x - y).abs() < 1e-13;
        let mut has_i = false;
        let mut has_minus_i = false;
        let mut has_2 = false;
        for k in 0..3 {
            let r = lr[k];
            let im = li[k];
            if close(r, 0.0) && close(im, 1.0) {
                has_i = true;
            }
            if close(r, 0.0) && close(im, -1.0) {
                has_minus_i = true;
            }
            if close(r, 2.0) && close(im, 0.0) {
                has_2 = true;
            }
        }
        assert!(has_i && has_minus_i && has_2);
    }

    #[test]
    fn eigenvalues_returns_err() {
        let a = Tensor2::<9>::new();
        let mut lr = Vector::new(3);
        let mut li = Vector::new(2);
        assert_eq!(
            a.eigenvalues(&mut lr, &mut li).err(),
            Some("l_real.dim() and l_imag.dim() must be equal to 3")
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
    // A = np.array([[7.0, -2.0, 0.0], [-2.0, 6.0, -2.0], [0.0, -2.0, 5.0]])
    // linalg.eigvalsh(A)  # -> array([3., 6., 9.])
    // ```
    #[test]
    fn eigenvalues_sym_works_non_diagonal() {
        // non-diagonal symmetric matrix: eigenvalues [3, 6, 9]
        #[rustfmt::skip]
        let a = Tensor2::<6>::from_std_matrix(&[
            [7.0, -2.0,  0.0],
            [-2.0, 6.0, -2.0],
            [0.0, -2.0,  5.0],
        ]).unwrap();
        let mut l = Vector::new(3);
        a.eigenvalues_sym(&mut l).unwrap();
        vec_approx_eq(&l, &[3.0, 6.0, 9.0], 1e-13);
    }

    // Python reference (numpy + scipy):
    // ```python
    // import numpy as np
    // from scipy import linalg
    // C = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])
    // linalg.eigvalsh(C)  # -> array([0., 3., 3.])  (3 has multiplicity 2)
    // ```
    #[test]
    fn eigenvalues_sym_works_repeated() {
        // non-diagonal symmetric matrix: eigenvalues [0, 3, 3] (3 has multiplicity 2)
        #[rustfmt::skip]
        let a = Tensor2::<6>::from_std_matrix(&[
            [2.0, -1.0, -1.0],
            [-1.0, 2.0, -1.0],
            [-1.0, -1.0, 2.0],
        ]).unwrap();
        let mut l = Vector::new(3);
        a.eigenvalues_sym(&mut l).unwrap();
        vec_approx_eq(&l, &[0.0, 3.0, 3.0], 1e-13);
    }

    // Python reference (numpy + scipy):
    // ```python
    // import numpy as np
    // from scipy import linalg
    // B = np.array([[2.0, -1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]])
    // linalg.eigvals(B)  # -> array([2.-1.j, 2.+1.j, 5.+0.j])
    // ```
    #[test]
    fn eigenvalues_works_complex_pair() {
        // general matrix: eigenvalues {2+i, 2-i, 5}
        #[rustfmt::skip]
        let a = Tensor2::<9>::from_std_matrix(&[
            [2.0, -1.0, 0.0],
            [1.0,  2.0, 0.0],
            [0.0,  0.0, 5.0],
        ]).unwrap();
        let mut lr = Vector::new(3);
        let mut li = Vector::new(3);
        a.eigenvalues(&mut lr, &mut li).unwrap();
        let got = sorted_complex(&lr, &li);
        let expected = [(2.0, -1.0), (2.0, 1.0), (5.0, 0.0)];
        for k in 0..3 {
            approx_eq(got[k].0, expected[k].0, 1e-13);
            approx_eq(got[k].1, expected[k].1, 1e-13);
        }
    }

    #[test]
    fn norm_works() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        approx_eq(tt.norm(), f64::sqrt(285.0), 1e-15);

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [ 2.0, -3.0, 4.0],
            [-3.0, -5.0, 1.0],
            [ 4.0,  1.0, 6.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        approx_eq(tt.norm(), f64::sqrt(117.0), 1e-15);

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        approx_eq(tt.norm(), f64::sqrt(46.0), 1e-15);
    }

    #[test]
    fn deviator_norm_and_determinant_work() {
        // general
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let tt = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        let mut dev = Tensor2::<9>::new();
        tt.deviator(&mut dev);
        approx_eq(dev.trace(), 0.0, 1e-15);
        assert_eq!(
            format!("{:.1}", dev.as_std_matrix()),
            "┌                ┐\n\
             │ -4.0  2.0  3.0 │\n\
             │  4.0  0.0  6.0 │\n\
             │  7.0  8.0  4.0 │\n\
             └                ┘"
        );
        approx_eq(dev.norm(), tt.deviator_norm(), 1e-15);
        approx_eq(dev.determinant(), tt.deviator_determinant(), 1e-12);

        // symmetric 3D
        #[rustfmt::skip]
        let comps_std = &[
            [ 2.0, -3.0, 4.0],
            [-3.0, -5.0, 1.0],
            [ 4.0,  1.0, 6.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        let mut dev = Tensor2::<6>::new();
        tt.deviator(&mut dev);
        approx_eq(dev.trace(), 0.0, 1e-15);
        assert_eq!(
            format!("{:.1}", dev.as_std_matrix()),
            "┌                ┐\n\
             │  1.0 -3.0  4.0 │\n\
             │ -3.0 -6.0  1.0 │\n\
             │  4.0  1.0  5.0 │\n\
             └                ┘"
        );
        approx_eq(dev.norm(), tt.deviator_norm(), 1e-14);
        approx_eq(dev.determinant(), tt.deviator_determinant(), 1e-15);

        // symmetric 2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let tt = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        let mut dev = Tensor2::<4>::new();
        tt.deviator(&mut dev);
        approx_eq(dev.trace(), 0.0, 1e-15);
        assert_eq!(
            format!("{:.1}", dev.as_std_matrix()),
            "┌                ┐\n\
             │ -1.0  4.0  0.0 │\n\
             │  4.0  0.0  0.0 │\n\
             │  0.0  0.0  1.0 │\n\
             └                ┘"
        );
        approx_eq(dev.norm(), tt.deviator_norm(), 1e-15);
        approx_eq(dev.determinant(), tt.deviator_determinant(), 1e-15);
    }

    #[test]
    fn decompose_works() {
        // General -- Example 1
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let ten = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        let mut sym = Tensor2::<9>::new();
        let mut skw = Tensor2::<9>::new();
        ten.decompose(&mut sym, &mut skw);
        let sym_mat = sym.as_std_matrix();
        let skw_mat = skw.as_std_matrix();
        let sym_correct = [[1.0, 3.0, 5.0], [3.0, 5.0, 7.0], [5.0, 7.0, 9.0]];
        let skw_correct = [[0.0, -1.0, -2.0], [1.0, 0.0, -1.0], [2.0, 1.0, 0.0]];
        mat_approx_eq(&sym_mat, &sym_correct, 1e-15);
        mat_approx_eq(&skw_mat, &skw_correct, 1e-15);
        assert_eq!(skw.trace(), 0.0);

        // General -- Example 2
        #[rustfmt::skip]
        let comps_std = &[
            [4.0, 2.0, 2.0],
            [6.0, 2.0, 4.0],
            [8.0, 4.0, 2.0],
        ];
        let ten = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        let mut sym = Tensor2::<9>::new();
        let mut skw = Tensor2::<9>::new();
        ten.decompose(&mut sym, &mut skw);
        let sym_mat = sym.as_std_matrix();
        let skw_mat = skw.as_std_matrix();
        let sym_correct = [[4.0, 4.0, 5.0], [4.0, 2.0, 4.0], [5.0, 4.0, 2.0]];
        let skw_correct = [[0.0, -2.0, -3.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        mat_approx_eq(&sym_mat, &sym_correct, 1e-15);
        mat_approx_eq(&skw_mat, &skw_correct, 1e-15);
        assert_eq!(skw.trace(), 0.0);

        // Symmetric
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ];
        let ten = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        let mut sym = Tensor2::<6>::new();
        let mut skw = Tensor2::<6>::new();
        ten.decompose(&mut sym, &mut skw);
        for m in 0..ten.dim() {
            assert_eq!(sym.get(m), ten.get(m));
            assert_eq!(skw.get(m), 0.0);
        }

        // Symmetric2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ];
        let ten = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        let mut sym = Tensor2::<4>::new();
        let mut skw = Tensor2::<4>::new();
        ten.decompose(&mut sym, &mut skw);
        for m in 0..ten.dim() {
            assert_eq!(sym.get(m), ten.get(m));
            assert_eq!(skw.get(m), 0.0);
        }
    }

    #[test]
    fn axial_vector_works() {
        // General -- Example 1
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let ten = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        let mut omega = Tensor1::new();
        ten.axial_vector(&mut omega);
        approx_eq(omega.get(0), 1.0, 1e-15);
        approx_eq(omega.get(1), -2.0, 1e-15);
        approx_eq(omega.get(2), 1.0, 1e-15);

        // General -- Example 2
        #[rustfmt::skip]
        let comps_std = &[
            [4.0, 2.0, 2.0],
            [6.0, 2.0, 4.0],
            [8.0, 4.0, 2.0],
        ];
        let ten = Tensor2::<9>::from_std_matrix(comps_std).unwrap();
        let mut omega = Tensor1::new();
        ten.axial_vector(&mut omega);
        approx_eq(omega.get(0), 0.0, 1e-15);
        approx_eq(omega.get(1), -3.0, 1e-15);
        approx_eq(omega.get(2), 2.0, 1e-15);

        // Symmetric
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ];
        let ten = Tensor2::<6>::from_std_matrix(comps_std).unwrap();
        let mut omega = Tensor1::new();
        ten.axial_vector(&mut omega);
        assert_eq!(omega.get(0), 0.0);
        assert_eq!(omega.get(1), 0.0);
        assert_eq!(omega.get(2), 0.0);

        // Symmetric2D
        #[rustfmt::skip]
        let comps_std = &[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ];
        let ten = Tensor2::<4>::from_std_matrix(comps_std).unwrap();
        let mut omega = Tensor1::new();
        ten.axial_vector(&mut omega);
        assert_eq!(omega.get(0), 0.0);
        assert_eq!(omega.get(1), 0.0);
        assert_eq!(omega.get(2), 0.0);
    }

    fn check_sample<const N: usize>(
        sample: &SampleTensor2,
        tol_norm: f64,
        tol_trace: f64,
        tol_det: f64,
        tol_dev_norm: f64,
        tol_dev_det: f64,
    ) {
        let tt = Tensor2::<N>::from_std_matrix(&sample.matrix).unwrap();
        approx_eq(tt.norm(), sample.norm, tol_norm);
        approx_eq(tt.trace(), sample.trace, tol_trace);
        approx_eq(tt.determinant(), sample.determinant, tol_det);
        approx_eq(tt.deviator_norm(), sample.deviator_norm, tol_dev_norm);
        approx_eq(tt.deviator_determinant(), sample.deviator_determinant, tol_dev_det);
    }

    #[test]
    #[rustfmt::skip]
    fn properties_are_correct() {
        // General
        //                                          norm   trace  det dev_norm dev_det
        check_sample::<9>(&SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15);
        check_sample::<9>(&SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15);
        check_sample::<9>(&SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15, 1e-15, 1e-13);
        check_sample::<9>(&SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-15, 1e-15, 1e-15);
        check_sample::<9>(&SamplesTensor2::TENSOR_Z, 1e-15, 1e-15, 1e-14, 1e-14, 1e-15);
        check_sample::<9>(&SamplesTensor2::TENSOR_U, 1e-13, 1e-15, 1e-14, 1e-14, 1e-13);
        check_sample::<9>(&SamplesTensor2::TENSOR_S, 1e-13, 1e-15, 1e-14, 1e-15, 1e-13);
        check_sample::<9>(&SamplesTensor2::TENSOR_R, 1e-13, 1e-15, 1e-13, 1e-13, 1e-15);
        check_sample::<9>(&SamplesTensor2::TENSOR_T, 1e-13, 1e-15, 1e-15, 1e-14, 1e-15);
        // Symmetric
        //                                          norm   trace  det dev_norm dev_det
        check_sample::<6>(&SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15);
        check_sample::<6>(&SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15);
        check_sample::<6>(&SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15, 1e-15, 1e-13);
        check_sample::<6>(&SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-15, 1e-15, 1e-15);
        check_sample::<6>(&SamplesTensor2::TENSOR_Z, 1e-15, 1e-15, 1e-14, 1e-14, 1e-14);
        check_sample::<6>(&SamplesTensor2::TENSOR_U, 1e-13, 1e-15, 1e-14, 1e-14, 1e-13);
        check_sample::<6>(&SamplesTensor2::TENSOR_S, 1e-13, 1e-15, 1e-14, 1e-15, 1e-13);
        // Symmetric 2D
        //                                                           norm   trace  det dev_norm dev_det
        check_sample::<4>(&SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15);
        check_sample::<4>(&SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15);
        check_sample::<4>(&SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15, 1e-15, 1e-13);
        check_sample::<4>(&SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-15, 1e-15, 1e-15);
        check_sample::<4>(&SamplesTensor2::TENSOR_Z, 1e-15, 1e-15, 1e-14, 1e-14, 1e-14);
    }

    // --- PRINCIPAL INVARIANTS -------------------------------------------------------------------------------------------

    fn check_iis<const N: usize>(sample: &SampleTensor2, tol_a: f64, tol_b: f64, tol_c: f64, tol_d: f64) {
        let tt = Tensor2::<N>::from_std_matrix(&sample.matrix).unwrap();
        let jj2 = -sample.deviator_second_invariant;
        let jj3 = sample.deviator_determinant;
        approx_eq(tt.invariant_ii1(), sample.trace, tol_a);
        approx_eq(tt.invariant_ii2(), sample.second_invariant, tol_b);
        approx_eq(tt.invariant_ii3(), sample.determinant, tol_b);
        approx_eq(tt.invariant_jj2(), jj2, tol_c);
        approx_eq(tt.invariant_jj3(), jj3, tol_c);
        if N == 4 || N == 6 {
            let norm_s = tt.deviator_norm();
            approx_eq(jj2, norm_s * norm_s / 2.0, tol_d);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn principal_invariants_are_correct() {
        // General
        check_iis::<9>(&SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15, 1e-15);
        check_iis::<9>(&SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15, 1e-15);
        check_iis::<9>(&SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-13, 1e-13);
        check_iis::<9>(&SamplesTensor2::TENSOR_Y, 1e-15, 1e-15, 1e-15, 1e-15);
        check_iis::<9>(&SamplesTensor2::TENSOR_Z, 1e-15, 1e-14, 1e-14, 1e-15);
        check_iis::<9>(&SamplesTensor2::TENSOR_U, 1e-15, 1e-14, 1e-13, 1e-13);
        check_iis::<9>(&SamplesTensor2::TENSOR_S, 1e-15, 1e-14, 1e-13, 1e-13);
        check_iis::<9>(&SamplesTensor2::TENSOR_R, 1e-15, 1e-13, 1e-15, 1e-15);
        check_iis::<9>(&SamplesTensor2::TENSOR_T, 1e-15, 1e-15, 1e-15, 1e-15);
        // Symmetric
        check_iis::<6>(&SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15, 1e-15);
        check_iis::<6>(&SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15, 1e-15);
        check_iis::<6>(&SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-13, 1e-15);
        check_iis::<6>(&SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-15, 1e-15);
        check_iis::<6>(&SamplesTensor2::TENSOR_Z, 1e-15, 1e-14, 1e-14, 1e-15);
        check_iis::<6>(&SamplesTensor2::TENSOR_U, 1e-15, 1e-14, 1e-13, 1e-13);
        check_iis::<6>(&SamplesTensor2::TENSOR_S, 1e-15, 1e-14, 1e-13, 1e-14);
        // Symmetric 2D
        check_iis::<4>(&SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15, 1e-15);
        check_iis::<4>(&SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15, 1e-15);
        check_iis::<4>(&SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-13, 1e-15);
        check_iis::<4>(&SamplesTensor2::TENSOR_Y, 1e-15, 1e-15, 1e-15, 1e-15);
        check_iis::<4>(&SamplesTensor2::TENSOR_Z, 1e-15, 1e-14, 1e-15, 1e-15);
    }

    /// --- OCTAHEDRAL INVARIANTS ------------------------------------------------------------------------------------------

    fn alpha_deg(l1: f64, l2: f64, l3: f64) -> f64 {
        f64::atan2(2.0 * l1 - l2 - l3, (l3 - l2) * SQRT_3) * 180.0 / PI
    }

    fn check_lode(l: Option<f64>, correct: f64, tol: f64, must_be_none: bool) {
        if must_be_none {
            assert!(l.is_none());
        } else {
            let lode = l.unwrap();
            approx_eq(lode, correct, tol);
        }
    }

    #[test]
    fn octahedral_invariants_are_correct() {
        let q_1 = SQRT_3 / 2.0; // sqrt(((0.5+0.5)² + (0.5)² + (-0.5)²)/3) * sqrt(3/2)
        let eps_d_1 = 1.0 / SQRT_3; // sqrt(((0.5+0.5)² + (0.5)² + (-0.5)²)/3) * sqrt(2/3)
        let q_2 = 1.0; // sqrt((1² + 1²)/3)* sqrt(3/2)
        let eps_d_2 = 2.0 / 3.0; // sqrt((1² + 1²)/3)* sqrt(2/3)

        // α = 0
        let (l1, l2, l3) = (0.0, -0.5, 0.5);
        approx_eq(alpha_deg(l1, l2, l3), 0.0, 1e-15);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 0.0, 1e-15);
        approx_eq(tt.invariant_q(), q_1, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = 30
        let (l1, l2, l3) = (1.0, 0.0, 1.0);
        approx_eq(alpha_deg(l1, l2, l3), 30.0, 1e-14);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 2.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_q(), q_2, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 2.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), -1.0, 1e-15, false);

        // α = 60
        let (l1, l2, l3) = (0.5, -0.5, 0.0);
        approx_eq(alpha_deg(l1, l2, l3), 60.0, 1e-14);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 0.0, 1e-15);
        approx_eq(tt.invariant_q(), q_1, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = 90
        let (l1, l2, l3) = (1.0, 0.0, 0.0);
        approx_eq(alpha_deg(l1, l2, l3), 90.0, 1e-15);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 1.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_q(), q_2, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 1.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), 1.0, 1e-15, false);

        // α = 120
        let (l1, l2, l3) = (0.5, 0.0, -0.5);
        approx_eq(alpha_deg(l1, l2, l3), 120.0, 1e-13);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 0.0, 1e-15);
        approx_eq(tt.invariant_q(), q_1, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = 150
        let (l1, l2, l3) = (1.0, 1.0, 0.0);
        approx_eq(alpha_deg(l1, l2, l3), 150.0, 1e-13);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 2.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_q(), q_2, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 2.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), -1.0, 1e-15, false);

        // α = 180
        let (l1, l2, l3) = (0.0, 0.5, -0.5);
        approx_eq(alpha_deg(l1, l2, l3), 180.0, 1e-13);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 0.0, 1e-15);
        approx_eq(tt.invariant_q(), q_1, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = -150
        let (l1, l2, l3) = (0.0, 1.0, 0.0);
        approx_eq(alpha_deg(l1, l2, l3), -150.0, 1e-13);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 1.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_q(), q_2, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 1.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), 1.0, 1e-15, false);

        // α = -120
        let (l1, l2, l3) = (-0.5, 0.5, 0.0);
        approx_eq(alpha_deg(l1, l2, l3), -120.0, 1e-13);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 0.0, 1e-15);
        approx_eq(tt.invariant_q(), q_1, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = -90
        let (l1, l2, l3) = (0.0, 1.0, 1.0);
        approx_eq(alpha_deg(l1, l2, l3), -90.0, 1e-13);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 2.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_q(), q_2, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 2.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), -1.0, 1e-15, false);

        // α = -60
        let (l1, l2, l3) = (-0.5, 0.0, 0.5);
        approx_eq(alpha_deg(l1, l2, l3), -60.0, 1e-13);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 0.0, 1e-15);
        approx_eq(tt.invariant_q(), q_1, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 0.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_1, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), 0.0, 1e-15, false);

        // α = -30
        let (l1, l2, l3) = (0.0, 0.0, 1.0);
        approx_eq(alpha_deg(l1, l2, l3), -30.0, 1e-13);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 1.0 / 3.0, 1e-15);
        approx_eq(tt.invariant_q(), q_2, 1e-15);
        approx_eq(tt.invariant_p(), tt.invariant_sigma_s() / SQRT_3, 1e-15);
        approx_eq(tt.invariant_q(), tt.invariant_sigma_t() * SQRT_3_BY_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), 1.0, 1e-15);
        approx_eq(tt.invariant_eps_d(), eps_d_2, 1e-15);
        approx_eq(tt.invariant_eps_v(), tt.invariant_eps_s() * SQRT_3, 1e-15);
        approx_eq(tt.invariant_eps_d(), tt.invariant_eps_t() * SQRT_2_BY_3, 1e-15);
        check_lode(tt.invariant_lode(), 1.0, 1e-15, false);
    }

    #[test]
    fn octahedral_invariants_are_correct_simple() {
        // test from https://soilmodels.com/wp-content/uploads/2020/12/stress_space-2.wgl
        let (l1, l2, l3) = (193.18, 88.3, 18.52);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        approx_eq(tt.invariant_p(), 100.0, 1e-15);
        approx_eq(tt.invariant_q(), 152.28, 0.0053);
        let lode = tt.invariant_lode().unwrap();
        let theta = (f64::acos(lode) / 3.0) * 180.0 / PI;
        approx_eq(30.0 - theta, 6.62, 0.0019);
    }

    #[test]
    fn lode_invariant_handles_special_cases() {
        // norm(deviator) = 0  with l = 0
        let (l1, l2, l3) = (2.0, 2.0, 2.0);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        check_lode(tt.invariant_lode(), 0.0, 1e-15, true);

        // norm(deviator) > 1e-15  with l ~ -1 (note how l jumps from 0 to -1 for eps from -1e-5 to -1e-3)
        let (l1, l2, l3) = (2.0, 2.0, 2.0 - 1e-3);
        let tt = Tensor2::<6>::from_std_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]]).unwrap();
        check_lode(tt.invariant_lode(), -1.0, 1e-7, false);
    }

    #[test]
    fn invariants_octahedral_works() {
        // the following data corresponds to p = 1 and q = 3
        #[rustfmt::skip]
        let principal_stresses_and_lode = [
            ( 3.0          ,  0.0          ,  0.0          ,  1.0 ),
            ( 0.0          ,  3.0          ,  0.0          ,  1.0 ),
            ( 0.0          ,  0.0          ,  3.0          ,  1.0 ),
            ( 1.0 + SQRT_3 ,  1.0 - SQRT_3 ,  1.0          ,  0.0 ),
            ( 1.0 + SQRT_3 ,  1.0          ,  1.0 - SQRT_3 ,  0.0 ),
            ( 1.0          ,  1.0 + SQRT_3 ,  1.0 - SQRT_3 ,  0.0 ),
            ( 1.0 - SQRT_3 ,  1.0 + SQRT_3 ,  1.0          ,  0.0 ),
            ( 1.0          ,  1.0 - SQRT_3 ,  1.0 + SQRT_3 ,  0.0 ),
            ( 1.0 - SQRT_3 ,  1.0          ,  1.0 + SQRT_3 ,  0.0 ),
            ( 2.0          , -1.0          ,  2.0          , -1.0 ),
            ( 2.0          ,  2.0          , -1.0          , -1.0 ),
            (-1.0          ,  2.0          ,  2.0          , -1.0 ),
        ];
        let mut aux = Tensor2::<4>::new();
        for (sigma_1, sigma_2, sigma_3, lode_correct) in &principal_stresses_and_lode {
            aux.vec[0] = *sigma_1;
            aux.vec[1] = *sigma_2;
            aux.vec[2] = *sigma_3;
            let (d, r, l) = aux.invariants_octahedral();
            approx_eq(d / SQRT_3, 1.0, 1e-15);
            approx_eq(r * SQRT_3_BY_2, 3.0, 1e-15);
            approx_eq(l.unwrap(), *lode_correct, 1e-15);
        }
    }

    #[test]
    fn new_from_octahedral_works() {
        assert_eq!(
            Tensor2::<4>::new_from_octahedral(0.0, 0.0, -2.0).err(),
            Some("lode invariant must be in -1 ≤ lode ≤ 1")
        );

        let (p, q) = (1.0, 3.0);
        let (distance, radius) = (p * SQRT_3, q * SQRT_2_BY_3);

        let t1 = Tensor2::<4>::new_from_octahedral(distance, radius, 1.0).unwrap();
        let t2 = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, PI / 2.0).unwrap();
        approx_eq(t1.vec[0], 3.0, 1e-15);
        approx_eq(t1.vec[1], 0.0, 1e-15);
        approx_eq(t1.vec[2], 0.0, 1e-15);
        assert_eq!(t1.vec[3], 0.0);
        for m in 0..t1.dim() {
            approx_eq(t1.get(m), t2.get(m), 1e-15);
        }
        approx_eq(t1.invariant_sigma_s(), distance, 1e-15);
        approx_eq(t1.invariant_sigma_t(), radius, 1e-15);
        approx_eq(t2.invariant_sigma_s(), distance, 1e-15);
        approx_eq(t2.invariant_sigma_t(), radius, 1e-15);

        let t1 = Tensor2::<4>::new_from_octahedral(distance, radius, 0.0).unwrap();
        let t2 = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, PI / 3.0).unwrap();
        approx_eq(t1.vec[0], 1.0 + SQRT_3, 1e-15);
        approx_eq(t1.vec[1], 1.0 - SQRT_3, 1e-15);
        approx_eq(t1.vec[2], 1.0, 1e-15);
        assert_eq!(t1.vec[3], 0.0);
        for m in 0..t1.dim() {
            approx_eq(t1.get(m), t2.get(m), 1e-15);
        }
        approx_eq(t1.invariant_sigma_s(), distance, 1e-15);
        approx_eq(t1.invariant_sigma_t(), radius, 1e-15);
        approx_eq(t2.invariant_sigma_s(), distance, 1e-15);
        approx_eq(t2.invariant_sigma_t(), radius, 1e-15);

        let t1 = Tensor2::<4>::new_from_octahedral(distance, radius, -1.0).unwrap();
        let t2 = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, PI / 6.0).unwrap();
        approx_eq(t1.vec[0], 2.0, 1e-15);
        approx_eq(t1.vec[1], -1.0, 1e-15);
        approx_eq(t1.vec[2], 2.0, 1e-15);
        assert_eq!(t1.vec[3], 0.0);
        for m in 0..t1.dim() {
            approx_eq(t1.get(m), t2.get(m), 1e-15);
        }
        approx_eq(t1.invariant_sigma_s(), distance, 1e-15);
        approx_eq(t1.invariant_sigma_t(), radius, 1e-15);
        approx_eq(t2.invariant_sigma_s(), distance, 1e-15);
        approx_eq(t2.invariant_sigma_t(), radius, 1e-15);
    }

    #[test]
    fn new_from_octahedral_alpha_works() {
        assert_eq!(
            Tensor2::<4>::new_from_octahedral_alpha(0.0, 0.0, -2.0 * PI).err(),
            Some("alpha must be in -π ≤ alpha ≤ π")
        );

        let (distance, radius) = (SQRT_3, SQRT_6);

        // 0 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, 0.0).unwrap();
        approx_eq(tt.vec[0], 1.0, 1e-15);
        approx_eq(tt.vec[1], 1.0 - SQRT_3, 1e-15);
        approx_eq(tt.vec[2], 1.0 + SQRT_3, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // 30 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, PI / 6.0).unwrap();
        approx_eq(tt.vec[0], 2.0, 1e-15);
        approx_eq(tt.vec[1], -1.0, 1e-15);
        approx_eq(tt.vec[2], 2.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // 60 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, PI / 3.0).unwrap();
        approx_eq(tt.vec[0], 1.0 + SQRT_3, 1e-15);
        approx_eq(tt.vec[1], 1.0 - SQRT_3, 1e-15);
        approx_eq(tt.vec[2], 1.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // 90 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, PI / 2.0).unwrap();
        approx_eq(tt.vec[0], 3.0, 1e-15);
        approx_eq(tt.vec[1], 0.0, 1e-15);
        approx_eq(tt.vec[2], 0.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // 120 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, 2.0 * PI / 3.0).unwrap();
        approx_eq(tt.vec[0], 1.0 + SQRT_3, 1e-15);
        approx_eq(tt.vec[1], 1.0, 1e-15);
        approx_eq(tt.vec[2], 1.0 - SQRT_3, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // 150 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, 5.0 * PI / 6.0).unwrap();
        approx_eq(tt.vec[0], 2.0, 1e-15);
        approx_eq(tt.vec[1], 2.0, 1e-15);
        approx_eq(tt.vec[2], -1.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // 180 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, PI).unwrap();
        approx_eq(tt.vec[0], 1.0, 1e-15);
        approx_eq(tt.vec[1], 1.0 + SQRT_3, 1e-15);
        approx_eq(tt.vec[2], 1.0 - SQRT_3, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // -180 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, -PI).unwrap();
        approx_eq(tt.vec[0], 1.0, 1e-15);
        approx_eq(tt.vec[1], 1.0 + SQRT_3, 1e-15);
        approx_eq(tt.vec[2], 1.0 - SQRT_3, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // -150 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, -5.0 * PI / 6.0).unwrap();
        approx_eq(tt.vec[0], 0.0, 1e-15);
        approx_eq(tt.vec[1], 3.0, 1e-15);
        approx_eq(tt.vec[2], 0.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // -120 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, -2.0 * PI / 3.0).unwrap();
        approx_eq(tt.vec[0], 1.0 - SQRT_3, 1e-15);
        approx_eq(tt.vec[1], 1.0 + SQRT_3, 1e-15);
        approx_eq(tt.vec[2], 1.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // -90 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, -PI / 2.0).unwrap();
        approx_eq(tt.vec[0], -1.0, 1e-15);
        approx_eq(tt.vec[1], 2.0, 1e-15);
        approx_eq(tt.vec[2], 2.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // -60 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, -PI / 3.0).unwrap();
        approx_eq(tt.vec[0], 1.0 - SQRT_3, 1e-15);
        approx_eq(tt.vec[1], 1.0, 1e-15);
        approx_eq(tt.vec[2], 1.0 + SQRT_3, 1e-15);
        assert_eq!(tt.vec[3], 0.0);

        // -30 degrees
        let tt = Tensor2::<4>::new_from_octahedral_alpha(distance, radius, -PI / 6.0).unwrap();
        approx_eq(tt.vec[0], 0.0, 1e-15);
        approx_eq(tt.vec[1], 0.0, 1e-15);
        approx_eq(tt.vec[2], 3.0, 1e-15);
        assert_eq!(tt.vec[3], 0.0);
    }

    #[test]
    fn deviator_with_large_numbers_works() {
        let tt = Tensor2::<4>::from_std_matrix(&[
            [-531906.3158661836, -459.8093541033259, 0.0],
            [-459.8093541033259, -531567.8289754189, 0.0],
            [0.0, 0.0, -531737.0724207585],
        ])
        .unwrap();
        let mut ss = Tensor2::<4>::new();
        tt.deviator(&mut ss);
        approx_eq(ss.trace(), 0.0, 1e-14);
    }
}
