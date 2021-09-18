use crate::{AsArray2D, EnumMatrixNorm};
use russell_openblas::*;
use std::cmp;
use std::fmt::{self, Write};
use std::ops::{Index, IndexMut};

/// Holds matrix components and associated functions
///
/// # Remarks
///
/// * Matrix implements the Index traits (mutable or not), thus, we can
///   access components by indices
/// * For faster computations, we recommend using the set of functions that
///   operate on Vectors and Matrices; e.g., `add_matrices`, `cholesky_factor`,
///   `eigen_decomp`, `inverse`, `pseudo_inverse`, `sv_decomp`, `mat_vec_mul`,
///   `sv_decomp`, and others.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// // import
/// use russell_lab::{Matrix, inverse, mat_mat_mul};
///
/// // create new matrix filled with ones
/// let mut a = Matrix::filled(2, 2, 1.0);
///
/// // change off-diagonal component
/// a[0][1] *= -1.0;
///
/// // check
/// assert_eq!(
///     format!("{}", a),
///     "┌       ┐\n\
///      │  1 -1 │\n\
///      │  1  1 │\n\
///      └       ┘"
/// );
///
/// // compute the inverse matrix `ai`
/// let (m, n) = a.dims();
/// let mut ai = Matrix::new(m, n);
/// let det = inverse(&mut ai, &a)?;
///
/// // check the determinant
/// assert_eq!(det, 2.0);
///
/// // check the inverse matrix
/// assert_eq!(
///     format!("{}", ai),
///     "┌           ┐\n\
///      │  0.5  0.5 │\n\
///      │ -0.5  0.5 │\n\
///      └           ┘"
/// );
///
/// // multiply the matrix by its inverse
/// let mut aia = Matrix::new(m, n);
/// mat_mat_mul(&mut aia, 1.0, &ai, &a)?;
///
/// // check the results
/// assert_eq!(
///     format!("{}", aia),
///     "┌     ┐\n\
///      │ 1 0 │\n\
///      │ 0 1 │\n\
///      └     ┘"
/// );
///
/// // create an identity matrix and check again
/// let ii = Matrix::identity(m);
/// assert_eq!(aia.as_data(), ii.as_data());
/// # Ok(())
/// # }
/// ```
pub struct Matrix {
    nrow: usize,    // number of rows
    ncol: usize,    // number of columns
    data: Vec<f64>, // row-major
}

// # Note
//
// Data is stored in row-major format as below
//
// ```text
//       _      _
//      |  0  1  |
//  a = |  2  3  |           a.data = [0, 1, 2, 3, 4, 5]
//      |_ 4  5 _|(m x n)
//
//  a.data[i * n + j] = a[i][j]
// ```
//

impl Matrix {
    /// Creates new (nrow x ncol) Matrix filled with zeros
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let a = Matrix::new(3, 3);
    /// let correct = "┌       ┐\n\
    ///                │ 0 0 0 │\n\
    ///                │ 0 0 0 │\n\
    ///                │ 0 0 0 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    pub fn new(nrow: usize, ncol: usize) -> Self {
        Matrix {
            nrow,
            ncol,
            data: vec![0.0; nrow * ncol],
        }
    }

    /// Creates new identity (square) matrix
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let identity = Matrix::identity(3);
    /// let correct = "┌       ┐\n\
    ///                │ 1 0 0 │\n\
    ///                │ 0 1 0 │\n\
    ///                │ 0 0 1 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", identity), correct);
    /// ```
    pub fn identity(m: usize) -> Self {
        let mut matrix = Matrix {
            nrow: m,
            ncol: m,
            data: vec![0.0; m * m],
        };
        for i in 0..m {
            matrix.data[i * m + i] = 1.0;
        }
        matrix
    }

    /// Creates new matrix completely filled with the same value
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let a = Matrix::filled(2, 3, 4.0);
    /// let correct = "┌       ┐\n\
    ///                │ 4 4 4 │\n\
    ///                │ 4 4 4 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    pub fn filled(m: usize, n: usize, value: f64) -> Self {
        Matrix {
            nrow: m,
            ncol: n,
            data: vec![value; m * n],
        }
    }

    /// Creates new matrix from given data
    ///
    /// # Notes
    ///
    /// * For variable-length rows, the number of columns is defined by the first row
    /// * The next rows must have at least the same number of columns as the first row
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    ///
    /// // heap-allocated 2D array (vector of vectors)
    /// const IGNORED: f64 = 123.456;
    /// let a_data = vec![
    ///     vec![1.0, 2.0],
    ///     vec![3.0, 4.0, IGNORED, IGNORED, IGNORED],
    ///     vec![5.0, 6.0],
    /// ];
    /// let a = Matrix::from(&a_data);
    /// assert_eq!(
    ///     format!("{}", &a),
    ///     "┌     ┐\n\
    ///      │ 1 2 │\n\
    ///      │ 3 4 │\n\
    ///      │ 5 6 │\n\
    ///      └     ┘"
    /// );
    ///
    /// // heap-allocated 2D array (aka slice of slices)
    /// let b_data: &[&[f64]] = &[
    ///     &[10.0, 20.0],
    ///     &[30.0, 40.0, IGNORED],
    ///     &[50.0, 60.0, IGNORED, IGNORED],
    /// ];
    /// let b = Matrix::from(&b_data);
    /// assert_eq!(
    ///     format!("{}", &b),
    ///     "┌       ┐\n\
    ///      │ 10 20 │\n\
    ///      │ 30 40 │\n\
    ///      │ 50 60 │\n\
    ///      └       ┘"
    /// );
    ///
    /// // stack-allocated (fixed-size) 2D array
    /// let c_data = [
    ///     [100.0, 200.0],
    ///     [300.0, 400.0],
    ///     [500.0, 600.0],
    /// ];
    /// let c = Matrix::from(&c_data);
    /// assert_eq!(
    ///     format!("{}", &c),
    ///     "┌         ┐\n\
    ///      │ 100 200 │\n\
    ///      │ 300 400 │\n\
    ///      │ 500 600 │\n\
    ///      └         ┘"
    /// );
    /// ```
    pub fn from<'a, T, U>(array: &'a T) -> Self
    where
        T: AsArray2D<'a, U>,
        U: 'a + Into<f64>,
    {
        let (mut nrow, ncol) = array.size();
        if ncol == 0 {
            nrow = 0
        }
        let mut matrix = Matrix {
            nrow,
            ncol,
            data: vec![0.0; nrow * ncol],
        };
        for i in 0..nrow {
            for j in 0..ncol {
                matrix.data[i * ncol + j] = array.at(i, j).into();
            }
        }
        matrix
    }

    /// Creates new diagonal matrix with given diagonal data
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let a = Matrix::diagonal(&[1.0, 2.0, 3.0]);
    /// let correct = "┌       ┐\n\
    ///                │ 1 0 0 │\n\
    ///                │ 0 2 0 │\n\
    ///                │ 0 0 3 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    pub fn diagonal(data: &[f64]) -> Self {
        let nrow = data.len();
        let ncol = nrow;
        let mut matrix = Matrix {
            nrow,
            ncol,
            data: vec![0.0; nrow * ncol],
        };
        for i in 0..nrow {
            matrix.data[i * ncol + i] = data[i];
        }
        matrix
    }

    /// Returns the number of rows
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let a = Matrix::new(4, 3);
    /// assert_eq!(a.nrow(), 4);
    /// ```
    #[inline]
    pub fn nrow(&self) -> usize {
        self.nrow
    }

    /// Returns the number of columns
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let a = Matrix::new(4, 3);
    /// assert_eq!(a.ncol(), 3);
    /// ```
    #[inline]
    pub fn ncol(&self) -> usize {
        self.ncol
    }

    /// Returns the dimensions (nrow, ncol) of this matrix
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let a = Matrix::new(4, 3);
    /// assert_eq!(a.dims(), (4, 3));
    /// ```
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    /// Scales this matrix
    ///
    /// ```text
    /// a := alpha * a
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut a = Matrix::from(&[
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    /// ]);
    /// a.scale(0.5);
    /// let correct = "┌             ┐\n\
    ///                │ 0.5   1 1.5 │\n\
    ///                │   2 2.5   3 │\n\
    ///                └             ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    pub fn scale(&mut self, alpha: f64) {
        let n: i32 = to_i32(self.data.len());
        dscal(n, alpha, &mut self.data, 1);
    }

    /// Fills this matrix with a given value
    ///
    /// ```text
    /// u[i][j] := value
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut a = Matrix::new(2, 2);
    /// a.fill(8.8);
    /// let correct = "┌         ┐\n\
    ///                │ 8.8 8.8 │\n\
    ///                │ 8.8 8.8 │\n\
    ///                └         ┘";
    /// assert_eq!(format!("{}", a), correct);
    pub fn fill(&mut self, value: f64) {
        self.data.iter_mut().map(|x| *x = value).count();
    }

    /// Returns an access to the underlying data
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let a = Matrix::from(&[[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(a.as_data(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn as_data(&self) -> &Vec<f64> {
        &self.data
    }

    /// Returns a mutable access to the underlying data
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut a = Matrix::from(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let data = a.as_mut_data();
    /// data[1] = 2.2;
    /// assert_eq!(data, &[1.0, 2.2, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn as_mut_data(&mut self) -> &mut Vec<f64> {
        &mut self.data
    }

    /// Returns the (i,j) component
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let a = Matrix::from(&[
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    /// ]);
    /// assert_eq!(a.get(1,1), 4.0);
    /// ```
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        self.data[i * self.ncol + j]
    }

    /// Change the (i,j) component
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut a = Matrix::from(&[
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    /// ]);
    /// a.set(1, 1, -4.0);
    /// let correct = "┌       ┐\n\
    ///                │  1  2 │\n\
    ///                │  3 -4 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        self.data[i * self.ncol + j] = value;
    }

    /// Returns a copy of this matrix
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut a = Matrix::from(&[
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    /// ]);
    /// let a_copy = a.get_copy();
    /// a.set(0, 0, 5.0);
    /// let a_correct = "┌     ┐\n\
    ///                  │ 5 2 │\n\
    ///                  │ 3 4 │\n\
    ///                  └     ┘";
    /// let a_copy_correct = "┌     ┐\n\
    ///                       │ 1 2 │\n\
    ///                       │ 3 4 │\n\
    ///                       └     ┘";
    /// assert_eq!(format!("{}", a), a_correct);
    /// assert_eq!(format!("{}", a_copy), a_copy_correct);
    /// ```
    pub fn get_copy(&self) -> Self {
        Matrix {
            nrow: self.nrow,
            ncol: self.ncol,
            data: self.data.to_vec(),
        }
    }

    /// Returns the matrix norm
    ///
    /// Computes one of:
    ///
    /// ```text
    /// One:  1-norm
    ///
    ///       ‖a‖_1 = max_j ( Σ_i |aᵢⱼ| )
    ///
    /// Inf:  inf-norm
    ///
    ///       ‖a‖_∞ = max_i ( Σ_j |aᵢⱼ| )
    ///
    /// Fro:  Frobenius-norm (2-norm)
    ///
    ///       ‖a‖_F = sqrt(Σ_i Σ_j aᵢⱼ⋅aᵢⱼ) == ‖a‖_2
    ///
    /// Max: max-norm
    ///
    ///      ‖a‖_max = max_ij ( |aᵢⱼ| )
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let a = Matrix::from(&[
    ///     [-2.0,  2.0],
    ///     [ 1.0, -4.0],
    /// ]);
    /// assert_eq!(a.norm(EnumMatrixNorm::One), 6.0);
    /// assert_eq!(a.norm(EnumMatrixNorm::Inf), 5.0);
    /// assert_eq!(a.norm(EnumMatrixNorm::Fro), 5.0);
    /// assert_eq!(a.norm(EnumMatrixNorm::Max), 4.0);
    /// ```
    pub fn norm(&self, kind: EnumMatrixNorm) -> f64 {
        let norm = match kind {
            EnumMatrixNorm::One => b'1',
            EnumMatrixNorm::Inf => b'I',
            EnumMatrixNorm::Fro => b'F',
            EnumMatrixNorm::Max => b'M',
        };
        let (m, n) = (to_i32(self.nrow), to_i32(self.ncol));
        dlange(norm, m, n, &self.data)
    }
}

impl fmt::Display for Matrix {
    /// Generates a string representation of the Matrix
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::Matrix;
    /// let a = Matrix::from(&[
    ///     [1.0, 0.0, -1.0,   8.0],
    ///     [4.0, 3.0,  1.0, -4.04],
    /// ]);
    /// assert_eq!(
    ///     format!("{}", a),
    ///     "┌                         ┐\n\
    ///      │     1     0    -1     8 │\n\
    ///      │     4     3     1 -4.04 │\n\
    ///      └                         ┘"
    /// );
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // handle empty matrix
        if self.nrow == 0 || self.ncol == 0 {
            write!(f, "[]")?;
            return Ok(());
        }
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                let val = self[i][j];
                match f.precision() {
                    Some(v) => write!(&mut buf, "{:.1$}", val, v)?,
                    None => write!(&mut buf, "{}", val)?,
                }
                width = cmp::max(buf.chars().count(), width);
                buf.clear();
            }
        }
        // draw matrix
        width += 1;
        write!(f, "┌{:1$}┐\n", " ", width * self.ncol + 1)?;
        for i in 0..self.nrow {
            if i > 0 {
                write!(f, " │\n")?;
            }
            for j in 0..self.ncol {
                if j == 0 {
                    write!(f, "│")?;
                }
                let val = self[i][j];
                match f.precision() {
                    Some(v) => write!(f, "{:>1$.2$}", val, width, v)?,
                    None => write!(f, "{:>1$}", val, width)?,
                }
            }
        }
        write!(f, " │\n")?;
        write!(f, "└{:1$}┘", " ", width * self.ncol + 1)?;
        Ok(())
    }
}

/// Allows to access Matrix components using indices
///
/// # Example
///
/// ```
/// use russell_lab::Matrix;
/// let a = Matrix::from(&[
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
/// ]);
/// assert_eq!(a[0][0], 1.0);
/// assert_eq!(a[0][1], 2.0);
/// assert_eq!(a[0][2], 3.0);
/// assert_eq!(a[1][0], 4.0);
/// assert_eq!(a[1][1], 5.0);
/// assert_eq!(a[1][2], 6.0);
/// ```
impl Index<usize> for Matrix {
    type Output = [f64];
    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[(i * self.ncol)..((i + 1) * self.ncol)]
    }
}

/// Allows to change Matrix components using indices
///
/// # Example
///
/// ```
/// use russell_lab::Matrix;
/// let mut a = Matrix::from(&[
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
/// ]);
/// a[0][0] -= 1.0;
/// a[0][1] += 1.0;
/// a[0][2] -= 1.0;
/// a[1][0] += 1.0;
/// a[1][1] -= 1.0;
/// a[1][2] += 1.0;
/// assert_eq!(a[0][0], 0.0);
/// assert_eq!(a[0][1], 3.0);
/// assert_eq!(a[0][2], 2.0);
/// assert_eq!(a[1][0], 5.0);
/// assert_eq!(a[1][1], 4.0);
/// assert_eq!(a[1][2], 7.0);
/// ```
impl IndexMut<usize> for Matrix {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[(i * self.ncol)..((i + 1) * self.ncol)]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn new_works() {
        let u = Matrix::new(3, 3);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }

    #[test]
    fn identity_works() {
        let identity = Matrix::identity(2);
        assert_eq!(identity.data, &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn filled_works() {
        let a = Matrix::filled(2, 2, 3.0);
        assert_eq!(a.data, &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn from_works() {
        // heap-allocated 2D array (vector of vectors)
        const IGNORED: f64 = 123.456;
        let a_data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, IGNORED, IGNORED, IGNORED],
            vec![5.0, 6.0],
        ];
        let a = Matrix::from(&a_data);
        assert_eq!(a.data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // heap-allocated 2D array (aka slice of slices)
        #[rustfmt::skip]
        let b_data: &[&[f64]] = &[
            &[10.0, 20.0],
            &[30.0, 40.0, IGNORED],
            &[50.0, 60.0, IGNORED, IGNORED],
        ];
        let b = Matrix::from(&b_data);
        assert_eq!(b.data, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        // stack-allocated (fixed-size) 2D array
        #[rustfmt::skip]
        let c_data = [
            [100.0, 200.0],
            [300.0, 400.0],
            [500.0, 600.0],
        ];
        let c = Matrix::from(&c_data);
        assert_eq!(c.data, &[100.0, 200.0, 300.0, 400.0, 500.0, 600.0]);
    }

    #[test]
    fn from_0_works() {
        let a_data: &[&[f64]] = &[&[]];
        let a = Matrix::from(&a_data);
        assert_eq!(a.nrow, 0);
        assert_eq!(a.ncol, 0);
        assert_eq!(a.data.len(), 0);
    }

    #[test]
    fn diagonal_works() {
        let a = Matrix::diagonal(&[-8.0, 2.0, 1.0]);
        assert_eq!(a.nrow, 3);
        assert_eq!(a.ncol, 3);
        assert_eq!(a.data, [-8.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn nrow_works() {
        let a = Matrix::new(4, 3);
        assert_eq!(a.nrow(), 4);
    }

    #[test]
    fn ncol_works() {
        let a = Matrix::new(4, 3);
        assert_eq!(a.ncol(), 3);
    }

    #[test]
    fn dims_works() {
        let a = Matrix::new(5, 4);
        assert_eq!(a.dims(), (5, 4));
    }

    #[test]
    fn display_works() -> Result<(), &'static str> {
        let a_0x0 = Matrix::new(0, 0);
        let a_0x1 = Matrix::new(0, 1);
        let a_1x0 = Matrix::new(1, 0);
        println!("{:?}", a_0x0.data);
        println!("{:?}", a_0x1.data);
        println!("{:?}", a_1x0.data);
        println!("{}", a_0x0);
        println!("{}", a_0x1);
        println!("{}", a_1x0);
        assert_eq!(format!("{}", a_0x0), "[]");
        assert_eq!(format!("{}", a_0x1), "[]");
        assert_eq!(format!("{}", a_1x0), "[]");
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        assert_eq!(
            format!("{}", a),
            "┌       ┐\n\
             │ 1 2 3 │\n\
             │ 4 5 6 │\n\
             │ 7 8 9 │\n\
             └       ┘"
        );
        Ok(())
    }

    #[test]
    fn display_precision_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [1.0111111, 2.02222222, 3.033333],
            [4.0444444, 5.05555555, 6.066666],
            [7.0777777, 8.08888888, 9.099999],
        ]);
        let correct: &str = "┌                ┐\n\
                             │ 1.01 2.02 3.03 │\n\
                             │ 4.04 5.06 6.07 │\n\
                             │ 7.08 8.09 9.10 │\n\
                             └                ┘";
        assert_eq!(format!("{:.2}", a), correct);
        Ok(())
    }

    #[test]
    fn scale_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            [ 6.0,  9.0,  12.0],
            [-6.0, -9.0, -12.0],
        ]);
        a.scale(1.0 / 3.0);
        #[rustfmt::skip]
        let correct = [
             2.0,  3.0,  4.0,
            -2.0, -3.0, -4.0,
        ];
        assert_vec_approx_eq!(a.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn fill_works() {
        let mut a = Matrix::new(2, 2);
        a.fill(7.7);
        let correct = &[7.7, 7.7, 7.7, 7.7];
        assert_vec_approx_eq!(a.data, correct, 1e-15);
    }

    #[test]
    #[should_panic]
    fn get_panics_on_wrong_indices() {
        let a = Matrix::new(1, 1);
        a.get(1, 0);
    }

    #[test]
    fn get_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [1.0, 2.0],
            [3.0, 4.0],
        ]);
        assert_eq!(a.get(0, 0), 1.0);
        assert_eq!(a.get(0, 1), 2.0);
        assert_eq!(a.get(1, 0), 3.0);
        assert_eq!(a.get(1, 1), 4.0);
    }

    #[test]
    #[should_panic]
    fn set_panics_on_wrong_indices() {
        let mut a = Matrix::new(1, 1);
        a.set(1, 0, 0.0);
    }

    #[test]
    fn set_works() {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            [1.0, 2.0],
            [3.0, 4.0],
        ]);
        a.set(0, 0, -1.0);
        a.set(0, 1, -2.0);
        a.set(1, 0, -3.0);
        a.set(1, 1, -4.0);
        assert_eq!(a.data, &[-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn get_copy_works() {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            [1.0, 2.0],
            [3.0, 4.0],
        ]);
        let a_copy = a.get_copy();
        a.set(0, 0, 0.11);
        a.set(0, 1, 0.22);
        a.set(1, 0, 0.33);
        a.set(1, 1, 0.44);
        assert_eq!(a.data, &[0.11, 0.22, 0.33, 0.44]);
        assert_eq!(a_copy.data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn norm_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [ 5.0, -4.0, 2.0],
            [-1.0,  2.0, 3.0],
            [-2.0,  1.0, 0.0],
        ]);
        assert_eq!(a.norm(EnumMatrixNorm::One), 8.0);
        assert_eq!(a.norm(EnumMatrixNorm::Inf), 11.0);
        assert_eq!(a.norm(EnumMatrixNorm::Fro), 8.0);
        assert_eq!(a.norm(EnumMatrixNorm::Max), 5.0);
    }
}
