use crate::AsArray2D;
use num_traits::Num;
use std::cmp;
use std::fmt::{self, Write};
use std::ops::{Index, IndexMut};

/// Holds matrix components and associated functions
///
/// # Remarks
///
/// * GenericMatrix implements the Index traits (mutable or not), thus, we can
///   access components by indices
/// * GenericMatrix has also methods to access the underlying data (mutable or not);
///   e.g., using `as_data()` and `as_mut_data()`.
/// * Internally, the data is stored in the [**row-major** order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)
/// * For faster computations, we recommend using the set of functions that
///   operate on Vectors and Matrices; e.g., `add_matrices`, `cholesky_factor`,
///   `eigen_decomp`, `inverse`, `pseudo_inverse`, `sv_decomp`, `mat_vec_mul`,
///   `sv_decomp`, and others.
///
/// # Example
///
/// ```
/// use russell_lab::{inverse, mat_mat_mul, GenericMatrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // create new matrix filled with ones
///     let mut a = GenericMatrix::<f64>::filled(2, 2, 1.0);
///
///     // change off-diagonal component
///     a[0][1] *= -1.0;
///
///     // check
///     assert_eq!(
///         format!("{}", a),
///         "┌       ┐\n\
///          │  1 -1 │\n\
///          │  1  1 │\n\
///          └       ┘"
///     );
///
///     // compute the inverse matrix `ai`
///     let (m, n) = a.dims();
///     let mut ai = GenericMatrix::<f64>::new(m, n);
///     let det = inverse(&mut ai, &a)?;
///
///     // check the determinant
///     assert_eq!(det, 2.0);
///
///     // check the inverse matrix
///     assert_eq!(
///         format!("{}", ai),
///         "┌           ┐\n\
///          │  0.5  0.5 │\n\
///          │ -0.5  0.5 │\n\
///          └           ┘"
///     );
///
///     // multiply the matrix by its inverse
///     let mut aia = GenericMatrix::<f64>::new(m, n);
///     mat_mat_mul(&mut aia, 1.0, &ai, &a)?;
///
///     // check the results
///     assert_eq!(
///         format!("{}", aia),
///         "┌     ┐\n\
///          │ 1 0 │\n\
///          │ 0 1 │\n\
///          └     ┘"
///     );
///
///     // create an identity matrix and check again
///     let ii = GenericMatrix::<f64>::identity(m);
///     assert_eq!(aia.as_data(), ii.as_data());
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct GenericMatrix<T>
where
    T: Num,
{
    nrow: usize,  // number of rows
    ncol: usize,  // number of columns
    data: Vec<T>, // row-major
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

impl<T> GenericMatrix<T>
where
    T: Num + Copy,
{
    /// Creates new (nrow x ncol) GenericMatrix filled with zeros
    ///
    /// # Example
    ///
    /// ```
    /// # use russell_lab::GenericMatrix;
    /// let a = GenericMatrix::<f64>::new(3, 3);
    /// let correct = "┌       ┐\n\
    ///                │ 0 0 0 │\n\
    ///                │ 0 0 0 │\n\
    ///                │ 0 0 0 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    pub fn new(nrow: usize, ncol: usize) -> Self {
        GenericMatrix {
            nrow,
            ncol,
            data: vec![T::zero(); nrow * ncol],
        }
    }

    /// Creates new identity (square) matrix
    ///
    /// # Example
    ///
    /// ```
    /// # use russell_lab::GenericMatrix;
    /// let identity = GenericMatrix::<f64>::identity(3);
    /// let correct = "┌       ┐\n\
    ///                │ 1 0 0 │\n\
    ///                │ 0 1 0 │\n\
    ///                │ 0 0 1 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", identity), correct);
    /// ```
    pub fn identity(m: usize) -> Self {
        let mut matrix = GenericMatrix {
            nrow: m,
            ncol: m,
            data: vec![T::zero(); m * m],
        };
        let one = T::one();
        for i in 0..m {
            matrix.data[i * m + i] = one;
        }
        matrix
    }

    /// Creates new matrix completely filled with the same value
    ///
    /// # Example
    ///
    /// ```
    /// # use russell_lab::GenericMatrix;
    /// let a = GenericMatrix::<f64>::filled(2, 3, 4.0);
    /// let correct = "┌       ┐\n\
    ///                │ 4 4 4 │\n\
    ///                │ 4 4 4 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    pub fn filled(m: usize, n: usize, value: T) -> Self {
        GenericMatrix {
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
    /// # use russell_lab::GenericMatrix;
    /// // heap-allocated 2D array (vector of vectors)
    /// const IGNORED: f64 = 123.456;
    /// let a_data = vec![
    ///     vec![1.0, 2.0],
    ///     vec![3.0, 4.0, IGNORED, IGNORED, IGNORED],
    ///     vec![5.0, 6.0],
    /// ];
    /// let a = GenericMatrix::<f64>::from(&a_data);
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
    /// let b = GenericMatrix::<f64>::from(&b_data);
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
    /// let c = GenericMatrix::<f64>::from(&c_data);
    /// assert_eq!(
    ///     format!("{}", &c),
    ///     "┌         ┐\n\
    ///      │ 100 200 │\n\
    ///      │ 300 400 │\n\
    ///      │ 500 600 │\n\
    ///      └         ┘"
    /// );
    /// ```
    pub fn from<'a, S, U>(array: &'a S) -> Self
    where
        S: AsArray2D<'a, U>,
        U: 'a + Into<T>,
    {
        let (mut nrow, ncol) = array.size();
        if ncol == 0 {
            nrow = 0
        }
        let mut matrix = GenericMatrix {
            nrow,
            ncol,
            data: vec![T::zero(); nrow * ncol],
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
    /// # use russell_lab::GenericMatrix;
    /// let a = GenericMatrix::<f64>::diagonal(&[1.0, 2.0, 3.0]);
    /// let correct = "┌       ┐\n\
    ///                │ 1 0 0 │\n\
    ///                │ 0 2 0 │\n\
    ///                │ 0 0 3 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    pub fn diagonal(data: &[T]) -> Self {
        let nrow = data.len();
        let ncol = nrow;
        let mut matrix = GenericMatrix {
            nrow,
            ncol,
            data: vec![T::zero(); nrow * ncol],
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
    /// # use russell_lab::GenericMatrix;
    /// let a = GenericMatrix::<f64>::new(4, 3);
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
    /// # use russell_lab::GenericMatrix;
    /// let a = GenericMatrix::<f64>::new(4, 3);
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
    /// # use russell_lab::GenericMatrix;
    /// let a = GenericMatrix::<f64>::new(4, 3);
    /// assert_eq!(a.dims(), (4, 3));
    /// ```
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
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
    /// # use russell_lab::GenericMatrix;
    /// let mut a = GenericMatrix::<f64>::new(2, 2);
    /// a.fill(8.8);
    /// let correct = "┌         ┐\n\
    ///                │ 8.8 8.8 │\n\
    ///                │ 8.8 8.8 │\n\
    ///                └         ┘";
    /// assert_eq!(format!("{}", a), correct);
    pub fn fill(&mut self, value: T) {
        self.data.iter_mut().map(|x| *x = value).count();
    }

    /// Returns an access to the underlying data
    ///
    /// # Note
    ///
    /// * Internally, the data is stored in the [**row-major** order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)
    ///
    /// # Example
    ///
    /// ```
    /// # use russell_lab::GenericMatrix;
    /// let a = GenericMatrix::<f64>::from(&[[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(a.as_data(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn as_data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a mutable access to the underlying data
    ///
    /// # Note
    ///
    /// * Internally, the data is stored in the [**row-major** order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)
    ///
    /// # Example
    ///
    /// ```
    /// # use russell_lab::GenericMatrix;
    /// let mut a = GenericMatrix::<f64>::from(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let data = a.as_mut_data();
    /// data[1] = 2.2;
    /// assert_eq!(data, &[1.0, 2.2, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn as_mut_data(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Returns the (i,j) component
    ///
    /// # Example
    ///
    /// ```
    /// # use russell_lab::GenericMatrix;
    /// let a = GenericMatrix::<f64>::from(&[
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    /// ]);
    /// assert_eq!(a.get(1,1), 4.0);
    /// ```
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        self.data[i * self.ncol + j]
    }

    /// Change the (i,j) component
    ///
    /// # Example
    ///
    /// ```
    /// # use russell_lab::GenericMatrix;
    /// let mut a = GenericMatrix::<f64>::from(&[
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
    pub fn set(&mut self, i: usize, j: usize, value: T) {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        self.data[i * self.ncol + j] = value;
    }

    /// Returns a copy of this matrix
    ///
    /// ```
    /// # use russell_lab::GenericMatrix;
    /// let mut a = GenericMatrix::<f64>::from(&[
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
        GenericMatrix {
            nrow: self.nrow,
            ncol: self.ncol,
            data: self.data.to_vec(),
        }
    }
}

impl<T> fmt::Display for GenericMatrix<T>
where
    T: Num + Copy + fmt::Display,
{
    /// Generates a string representation of the GenericMatrix
    ///
    /// # Example
    ///
    /// ```
    /// # use russell_lab::GenericMatrix;
    /// let a = GenericMatrix::<f64>::from(&[
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

/// Allows to access GenericMatrix components using indices
///
/// # Example
///
/// ```
/// # use russell_lab::GenericMatrix;
/// let a = GenericMatrix::<f64>::from(&[
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
impl<T> Index<usize> for GenericMatrix<T>
where
    T: Num + Copy,
{
    type Output = [T];
    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[(i * self.ncol)..((i + 1) * self.ncol)]
    }
}

/// Allows to change GenericMatrix components using indices
///
/// # Example
///
/// ```
/// # use russell_lab::GenericMatrix;
/// let mut a = GenericMatrix::<f64>::from(&[
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
impl<T> IndexMut<usize> for GenericMatrix<T>
where
    T: Num + Copy,
{
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[(i * self.ncol)..((i + 1) * self.ncol)]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::GenericMatrix;
    use crate::StrError;
    use russell_chk::assert_vec_approx_eq;

    #[test]
    fn new_works() {
        let u = GenericMatrix::<f64>::new(3, 3);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }

    #[test]
    fn identity_works() {
        let identity = GenericMatrix::<f64>::identity(2);
        assert_eq!(identity.data, &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn filled_works() {
        let a = GenericMatrix::<f64>::filled(2, 2, 3.0);
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
        let a = GenericMatrix::<f64>::from(&a_data);
        assert_eq!(a.data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // heap-allocated 2D array (aka slice of slices)
        #[rustfmt::skip]
        let b_data: &[&[f64]] = &[
            &[10.0, 20.0],
            &[30.0, 40.0, IGNORED],
            &[50.0, 60.0, IGNORED, IGNORED],
        ];
        let b = GenericMatrix::<f64>::from(&b_data);
        assert_eq!(b.data, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        // stack-allocated (fixed-size) 2D array
        #[rustfmt::skip]
        let c_data = [
            [100.0, 200.0],
            [300.0, 400.0],
            [500.0, 600.0],
        ];
        let c = GenericMatrix::<f64>::from(&c_data);
        assert_eq!(c.data, &[100.0, 200.0, 300.0, 400.0, 500.0, 600.0]);
    }

    #[test]
    fn from_0_works() {
        let a_data: &[&[f64]] = &[&[]];
        let a = GenericMatrix::<f64>::from(&a_data);
        assert_eq!(a.nrow, 0);
        assert_eq!(a.ncol, 0);
        assert_eq!(a.data.len(), 0);
    }

    #[test]
    fn diagonal_works() {
        let a = GenericMatrix::<f64>::diagonal(&[-8.0, 2.0, 1.0]);
        assert_eq!(a.nrow, 3);
        assert_eq!(a.ncol, 3);
        assert_eq!(a.data, [-8.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn nrow_works() {
        let a = GenericMatrix::<f64>::new(4, 3);
        assert_eq!(a.nrow(), 4);
    }

    #[test]
    fn ncol_works() {
        let a = GenericMatrix::<f64>::new(4, 3);
        assert_eq!(a.ncol(), 3);
    }

    #[test]
    fn dims_works() {
        let a = GenericMatrix::<f64>::new(5, 4);
        assert_eq!(a.dims(), (5, 4));
    }

    #[test]
    fn display_works() -> Result<(), StrError> {
        let a_0x0 = GenericMatrix::<f64>::new(0, 0);
        let a_0x1 = GenericMatrix::<f64>::new(0, 1);
        let a_1x0 = GenericMatrix::<f64>::new(1, 0);
        assert_eq!(format!("{}", a_0x0), "[]");
        assert_eq!(format!("{}", a_0x1), "[]");
        assert_eq!(format!("{}", a_1x0), "[]");
        #[rustfmt::skip]
        let a = GenericMatrix::<f64>::from(&[
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
    fn display_precision_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let a = GenericMatrix::<f64>::from(&[
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
    fn debug_works() {
        let a = GenericMatrix::<f64>::new(1, 1);
        assert_eq!(format!("{:?}", a), "GenericMatrix { nrow: 1, ncol: 1, data: [0.0] }");
    }

    #[test]
    fn fill_works() {
        let mut a = GenericMatrix::<f64>::new(2, 2);
        a.fill(7.7);
        let correct = &[7.7, 7.7, 7.7, 7.7];
        assert_vec_approx_eq!(a.data, correct, 1e-15);
    }

    #[test]
    #[should_panic]
    fn get_panics_on_wrong_indices() {
        let a = GenericMatrix::<f64>::new(1, 1);
        a.get(1, 0);
    }

    #[test]
    fn get_works() {
        #[rustfmt::skip]
        let a = GenericMatrix::<f64>::from(&[
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
        let mut a = GenericMatrix::<f64>::new(1, 1);
        a.set(1, 0, 0.0);
    }

    #[test]
    fn set_works() {
        #[rustfmt::skip]
        let mut a = GenericMatrix::<f64>::from(&[
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
        let mut a = GenericMatrix::<f64>::from(&[
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
}
