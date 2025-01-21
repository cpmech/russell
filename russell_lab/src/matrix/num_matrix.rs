use crate::{AsArray2D, StrError};
use num_traits::Num;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::cmp;
use std::ffi::OsStr;
use std::fmt::{self, Write};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::{AddAssign, MulAssign};
use std::path::Path;

/// Implements a matrix with numeric components for linear algebra
///
/// # Remarks
///
/// Internally, the data is stored in the **col-major** order.
///
/// ```text
///     ┌     ┐  row_major = {0, 3,
///     │ 0 3 │               1, 4,
/// A = │ 1 4 │               2, 5};
///     │ 2 5 │
///     └     ┘  col_major = {0, 1, 2,
///     (m × n)               3, 4, 5}
///
/// Aᵢⱼ = col_major[i + j·m] = row_major[i·n + j]
///         ↑
/// COL-MAJOR IS ADOPTED HERE
/// ```
///
/// The main reason to use the **col-major** representation is to make the code work
/// better with BLAS/LAPACK written in Fortran. Although those libraries have functions
/// to handle row-major data, they usually add an overhead due to temporary memory
/// allocation and copies, including transposing matrices. Moreover, the row-major
/// versions of some BLAS/LAPACK libraries produce incorrect results (notably the DSYEV).
///
/// Unfortunately, because of the col-major representation, it is no longer possible
/// to use the Index trait to mimic the access of matrix components in the form of a\[i\]\[j\].
/// Therefore, only the `get(i, j)` and `set(i, j, value)` functions are implemented.
///
/// # Examples
///
/// ## Initialization, setting values and printing
///
/// ```
/// use russell_lab::{NumMatrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // create new matrix filled with ones
///     let mut a = NumMatrix::<f64>::filled(2, 2, 1.0);
///
///     // change off-diagonal component
///     a.mul(0, 1, -1.0);
///
///     // check
///     assert_eq!(
///         format!("{}", a),
///         "┌       ┐\n\
///          │  1 -1 │\n\
///          │  1  1 │\n\
///          └       ┘"
///     );
///     Ok(())
/// }
/// ```
///
/// ## Inverse and matrix multiplication
///
/// ```
/// use russell_lab::{mat_inverse, mat_mat_mul, NumMatrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // create new matrix filled with ones
///     let mut a = NumMatrix::<f64>::filled(2, 2, 1.0);
///
///     // change off-diagonal component
///     a.mul(0, 1, -1.0);
///
///     // compute the inverse matrix `ai`
///     let (m, n) = a.dims();
///     let mut ai = NumMatrix::<f64>::new(m, n);
///     let det = mat_inverse(&mut ai, &a)?;
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
///     let mut aia = NumMatrix::<f64>::new(m, n);
///     mat_mat_mul(&mut aia, 1.0, &ai, &a, 0.0)?;
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
///     let ii = NumMatrix::<f64>::identity(m);
///     assert_eq!(aia.as_data(), ii.as_data());
///     Ok(())
/// }
/// ```
///
/// ## Copying a matrix (cloning)
///
/// ```
/// use russell_lab::{NumMatrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // new matrix
///     let a = NumMatrix::<f64>::from(&[
///         [1.0, 2.0],
///         [3.0, 4.0],
///     ]);
///
///     // clone
///     let mut b = a.clone();
///
///     // change clone
///     b.set(0, 0, 5.0);
///
///     // check that clone is correct
///     assert_eq!(
///         format!("{}", b),
///         "┌     ┐\n\
///          │ 5 2 │\n\
///          │ 3 4 │\n\
///          └     ┘"
///     );
///
///     // check that the changed matrix is unmodified
///     assert_eq!(
///         format!("{}", a),
///         "┌     ┐\n\
///          │ 1 2 │\n\
///          │ 3 4 │\n\
///          └     ┘"
///     );
///     Ok(())
/// }
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NumMatrix<T>
where
    T: AddAssign + MulAssign + Num + Copy + DeserializeOwned + Serialize,
{
    nrow: usize, // number of rows
    ncol: usize, // number of columns
    #[serde(bound(deserialize = "Vec<T>: Deserialize<'de>"))]
    data: Vec<T>, // col-major
}

impl<T> NumMatrix<T>
where
    T: AddAssign + MulAssign + Num + Copy + DeserializeOwned + Serialize,
{
    /// Creates new (nrow x ncol) NumMatrix filled with zeros
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::new(3, 3);
    /// let correct = "┌       ┐\n\
    ///                │ 0 0 0 │\n\
    ///                │ 0 0 0 │\n\
    ///                │ 0 0 0 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    pub fn new(nrow: usize, ncol: usize) -> Self {
        NumMatrix {
            nrow,
            ncol,
            data: vec![T::zero(); nrow * ncol],
        }
    }

    /// Creates new identity (square) matrix
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let identity = NumMatrix::<f64>::identity(3);
    /// let correct = "┌       ┐\n\
    ///                │ 1 0 0 │\n\
    ///                │ 0 1 0 │\n\
    ///                │ 0 0 1 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", identity), correct);
    /// ```
    pub fn identity(m: usize) -> Self {
        let mut matrix = NumMatrix {
            nrow: m,
            ncol: m,
            data: vec![T::zero(); m * m],
        };
        let one = T::one();
        for i in 0..m {
            matrix.data[i + i * m] = one;
        }
        matrix
    }

    /// Creates new matrix completely filled with the same value
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::filled(2, 3, 4.0);
    /// let correct = "┌       ┐\n\
    ///                │ 4 4 4 │\n\
    ///                │ 4 4 4 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    pub fn filled(m: usize, n: usize, value: T) -> Self {
        NumMatrix {
            nrow: m,
            ncol: n,
            data: vec![value; m * n],
        }
    }

    /// Returns a new matrix that is initialized from a callback function (map)
    ///
    /// The function maps the indices to the value, e.g., `|i, j| (i + j) as f64`
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let u = NumMatrix::<f64>::initialized(3, 2, |i, j| (i + j) as f64);
    /// assert_eq!(
    ///     format!("{}", u),
    ///     "┌     ┐\n\
    ///      │ 0 1 │\n\
    ///      │ 1 2 │\n\
    ///      │ 2 3 │\n\
    ///      └     ┘"
    /// );
    /// ```
    pub fn initialized<F>(nrow: usize, ncol: usize, mut function: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let data: Vec<T> = (0..(nrow * ncol))
            .into_iter()
            .map(|p| function(p % nrow, p / nrow))
            .collect();
        NumMatrix { nrow, ncol, data }
    }

    /// Creates new matrix from given data
    ///
    /// # Notes
    ///
    /// * For variable-length rows, the number of columns is defined by the first row
    /// * The next rows must have at least the same number of columns as the first row
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// // heap-allocated 2D array (vector of vectors)
    /// const IGNORED: f64 = 123.456;
    /// let a_data = vec![
    ///     vec![1.0, 2.0],
    ///     vec![3.0, 4.0, IGNORED, IGNORED, IGNORED],
    ///     vec![5.0, 6.0],
    /// ];
    /// let a = NumMatrix::<f64>::from(&a_data);
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
    /// let b = NumMatrix::<f64>::from(&b_data);
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
    /// let c = NumMatrix::<f64>::from(&c_data);
    /// assert_eq!(
    ///     format!("{}", &c),
    ///     "┌         ┐\n\
    ///      │ 100 200 │\n\
    ///      │ 300 400 │\n\
    ///      │ 500 600 │\n\
    ///      └         ┘"
    /// );
    /// ```
    pub fn from<'a, S, U>(data: &'a S) -> Self
    where
        S: AsArray2D<'a, U>,
        U: 'a + Into<T>,
    {
        let (mut nrow, ncol) = data.size();
        if ncol == 0 {
            nrow = 0
        }
        let mut matrix = NumMatrix {
            nrow,
            ncol,
            data: vec![T::zero(); nrow * ncol],
        };
        for i in 0..nrow {
            for j in 0..ncol {
                matrix.data[i + j * nrow] = data.at(i, j).into();
            }
        }
        matrix
    }

    /// Creates new matrix from given data, taking the lower triangle (and diagonal) only
    ///
    /// # Notes
    ///
    /// * For variable-length rows, the number of columns is defined by the first row
    /// * The next rows must have at least the same number of columns as the first row
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{NumMatrix, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     const IGNORED: f64 = 123.456;
    ///     let data = [
    ///         [1.0, 2.0, 3.0],
    ///         [2.0, 4.0, 5.0],
    ///         [3.0, 5.0, 6.0],
    ///     ];
    ///     let a = NumMatrix::<f64>::from_lower(&data).unwrap();
    ///     assert_eq!(
    ///         format!("{}", &a),
    ///         "┌       ┐\n\
    ///          │ 1 0 0 │\n\
    ///          │ 2 4 0 │\n\
    ///          │ 3 5 6 │\n\
    ///          └       ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn from_lower<'a, S, U>(data: &'a S) -> Result<Self, StrError>
    where
        S: AsArray2D<'a, U>,
        U: 'a + Into<T>,
    {
        let (mut nrow, ncol) = data.size();
        if ncol == 0 {
            nrow = 0
        }
        if nrow != ncol {
            return Err("the matrix must be square to set lower triangle");
        }
        let mut matrix = NumMatrix {
            nrow,
            ncol,
            data: vec![T::zero(); nrow * ncol],
        };
        for i in 0..nrow {
            for j in 0..(i + 1) {
                matrix.data[i + j * nrow] = data.at(i, j).into();
            }
        }
        Ok(matrix)
    }

    /// Creates new matrix from given data, taking the upper triangle (and diagonal) only
    ///
    /// # Notes
    ///
    /// * For variable-length rows, the number of columns is defined by the first row
    /// * The next rows must have at least the same number of columns as the first row
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{NumMatrix, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     const IGNORED: f64 = 123.456;
    ///     let data = [
    ///         [1.0, 2.0, 3.0],
    ///         [2.0, 4.0, 5.0],
    ///         [3.0, 5.0, 6.0],
    ///     ];
    ///     let a = NumMatrix::<f64>::from_upper(&data).unwrap();
    ///     assert_eq!(
    ///         format!("{}", &a),
    ///         "┌       ┐\n\
    ///          │ 1 2 3 │\n\
    ///          │ 0 4 5 │\n\
    ///          │ 0 0 6 │\n\
    ///          └       ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn from_upper<'a, S, U>(data: &'a S) -> Result<Self, StrError>
    where
        S: AsArray2D<'a, U>,
        U: 'a + Into<T>,
    {
        let (mut nrow, ncol) = data.size();
        if ncol == 0 {
            nrow = 0
        }
        if nrow != ncol {
            return Err("the matrix must be square to set upper triangle");
        }
        let mut matrix = NumMatrix {
            nrow,
            ncol,
            data: vec![T::zero(); nrow * ncol],
        };
        for i in 0..nrow {
            for j in i..nrow {
                matrix.data[i + j * nrow] = data.at(i, j).into();
            }
        }
        Ok(matrix)
    }

    /// Creates new diagonal matrix with given diagonal data
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::diagonal(&[1.0, 2.0, 3.0]);
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
        let mut matrix = NumMatrix {
            nrow,
            ncol,
            data: vec![T::zero(); nrow * ncol],
        };
        for i in 0..nrow {
            matrix.data[i + i * nrow] = data[i];
        }
        matrix
    }

    /// Creates matrix from text file
    ///
    /// # Input
    ///
    /// * `full_path` -- may be a String, &str, or Path
    ///
    /// # Examples
    ///
    /// The code below will read the following file:
    ///
    /// ```text
    ///  ## hash indicates comments
    ///
    ///  1   4  # the first column defines the number of rows
    ///   2   5  
    ///     3   6   # more comments here
    ///   
    ///  ## empty lines are ok  
    ///        ## misalignment is fine
    /// ```
    ///
    /// ```
    /// use russell_lab::{NumMatrix, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let mut a = NumMatrix::<f64>::from_text_file("./data/matrices/example.txt")?;
    ///     assert_eq!(
    ///         format!("{}", a),
    ///         "┌     ┐\n\
    ///          │ 1 4 │\n\
    ///          │ 2 5 │\n\
    ///          │ 3 6 │\n\
    ///          └     ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Notes
    ///
    /// * Comments start with the hash character '#'
    /// * Lines starting with '#' or empty lines are ignored
    /// * The end of the row (line) may contain comments too and will cause to stop reading data,
    ///   thus, the '#' marker in a row (line) must be at the end of the line.
    pub fn from_text_file<P>(full_path: &P) -> Result<Self, StrError>
    where
        P: AsRef<OsStr> + ?Sized,
    {
        // read file
        let path = Path::new(full_path).to_path_buf();
        let input = File::open(path).map_err(|_| "cannot open file")?;
        let buffered = BufReader::new(input);
        let mut lines_iter = buffered.lines();

        // parse rows, ignoring comments and empty lines
        let mut current_row_index = 0;
        let mut number_of_columns = 0;
        let mut data_row_major = Vec::<T>::new();
        loop {
            match lines_iter.next() {
                Some(v) => {
                    // extract line
                    let line = v.unwrap(); // must panic because no error expected here

                    // ignore comments or empty lines
                    let maybe_data = line.trim_start().trim_end_matches("\n");
                    if maybe_data.starts_with("#") || maybe_data == "" {
                        continue; // nothing to parse
                    }

                    // remove whitespace
                    let mut row_values = maybe_data.split_whitespace();

                    // loop over columns
                    let mut column_index = 0;
                    loop {
                        match row_values.next() {
                            Some(s) => {
                                if s.starts_with("#") {
                                    break; // ignore comments at the end of the row
                                }
                                data_row_major.push(T::from_str_radix(s, 10).map_err(|_| "cannot parse value")?);
                                column_index += 1;
                            }
                            None => break,
                        }
                    }

                    // set or check the number of columns
                    if current_row_index == 0 {
                        number_of_columns = column_index; // the first row determines the number of columns
                    } else {
                        if column_index != number_of_columns {
                            return Err("column data is missing");
                        }
                    }
                    current_row_index += 1;
                }
                None => break,
            }
        }
        let (nrow, ncol) = (current_row_index, number_of_columns);
        let mut data = vec![T::zero(); nrow * ncol];
        for i in 0..current_row_index {
            for j in 0..number_of_columns {
                data[i + j * nrow] = data_row_major[i * ncol + j];
            }
        }
        Ok(NumMatrix { nrow, ncol, data })
    }

    /// Returns the number of rows
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::new(4, 3);
    /// assert_eq!(a.nrow(), 4);
    /// ```
    #[inline]
    pub fn nrow(&self) -> usize {
        self.nrow
    }

    /// Returns the number of columns
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::new(4, 3);
    /// assert_eq!(a.ncol(), 3);
    /// ```
    #[inline]
    pub fn ncol(&self) -> usize {
        self.ncol
    }

    /// Returns the dimensions (nrow, ncol) of this matrix
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::new(4, 3);
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
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let mut a = NumMatrix::<f64>::new(2, 2);
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
    /// * Internally, the data is stored in the **col-major** order
    ///
    /// ```text
    ///     ┌     ┐  row_major = {0, 3,
    ///     │ 0 3 │               1, 4,
    /// A = │ 1 4 │               2, 5};
    ///     │ 2 5 │
    ///     └     ┘  col_major = {0, 1, 2,
    ///     (m × n)               3, 4, 5}
    ///
    /// Aᵢⱼ = col_major[i + j·m] = row_major[i·n + j]
    ///         ↑
    /// COL-MAJOR IS ADOPTED HERE
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::from(&[[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(a.as_data(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    #[inline]
    pub fn as_data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a mutable access to the underlying data
    ///
    /// # Note
    ///
    /// * Internally, the data is stored in the **col-major** order
    ///
    /// ```text
    ///     ┌     ┐  row_major = {0, 3,
    ///     │ 0 3 │               1, 4,
    /// A = │ 1 4 │               2, 5};
    ///     │ 2 5 │
    ///     └     ┘  col_major = {0, 1, 2,
    ///     (m × n)               3, 4, 5}
    ///
    /// Aᵢⱼ = col_major[i + j·m] = row_major[i·n + j]
    ///         ↑
    /// COL-MAJOR IS ADOPTED HERE
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let mut a = NumMatrix::<f64>::from(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let data = a.as_mut_data();
    /// data[1] = 2.2;
    /// assert_eq!(data, &[1.0, 2.2, 2.0, 4.0]);
    /// ```
    #[inline]
    pub fn as_mut_data(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Returns the (i,j) component
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::from(&[
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    /// ]);
    /// assert_eq!(a.get(1,1), 4.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function may panic if the indices are out-of-bounds.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        self.data[i + j * self.nrow]
    }

    /// Change the (i,j) component
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let mut a = NumMatrix::<f64>::from(&[
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
    ///
    /// # Panics
    ///
    /// This function may panic if the indices are out-of-bounds.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: T) {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        self.data[i + j * self.nrow] = value;
    }

    /// Adds a value to the (i,j) component
    ///
    /// ```text
    /// aᵢⱼ += value
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let mut a = NumMatrix::<f64>::from(&[
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    /// ]);
    /// a.add(1, 1, -4.0);
    /// let correct = "┌     ┐\n\
    ///                │ 1 2 │\n\
    ///                │ 3 0 │\n\
    ///                └     ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    ///
    /// # Panics
    ///
    /// This function may panic if the indices are out-of-bounds.
    #[inline]
    pub fn add(&mut self, i: usize, j: usize, value: T) {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        self.data[i + j * self.nrow] += value;
    }

    /// Multiply a value to the (i,j) component
    ///
    /// ```text
    /// aᵢⱼ *= value
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let mut a = NumMatrix::<f64>::from(&[
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    /// ]);
    /// a.mul(1, 1, -4.0);
    /// let correct = "┌         ┐\n\
    ///                │   1   2 │\n\
    ///                │   3 -16 │\n\
    ///                └         ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// ```
    ///
    /// # Panics
    ///
    /// This function may panic if the indices are out-of-bounds.
    #[inline]
    pub fn mul(&mut self, i: usize, j: usize, value: T) {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        self.data[i + j * self.nrow] *= value;
    }

    /// Extracts a row given its index
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::from(&[
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    ///     [5.0, 6.0],
    ///     [7.0, 8.0],
    /// ]);
    /// let first_row = a.extract_row(0);
    /// let second_row = a.extract_row(1);
    /// assert_eq!(first_row, [1.0, 2.0]);
    /// assert_eq!(second_row, [3.0, 4.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function may panic if the row index is out-of-bounds.
    pub fn extract_row(&self, i: usize) -> Vec<T> {
        assert!(i < self.nrow);
        let mut res = vec![T::zero(); self.ncol];
        for j in 0..self.ncol {
            res[j] = self.data[i + j * self.nrow];
        }
        res
    }

    /// Extracts a column given its index
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::from(&[
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    ///     [5.0, 6.0],
    ///     [7.0, 8.0],
    /// ]);
    /// let first_column = a.extract_column(0);
    /// let second_column = a.extract_column(1);
    /// assert_eq!(first_column, [1.0, 3.0, 5.0, 7.0]);
    /// assert_eq!(second_column, [2.0, 4.0, 6.0, 8.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function may panic if the column index is out-of-bounds.
    pub fn extract_column(&self, j: usize) -> Vec<T> {
        assert!(j < self.ncol);
        let mut res = vec![T::zero(); self.nrow];
        for i in 0..self.nrow {
            res[i] = self.data[i + j * self.nrow];
        }
        res
    }

    /// Returns this matrix transposed
    ///
    /// Returns `Aᵀ`
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::from(&[
    ///     [1.0, 2.0, 3.0, 4.0],
    ///     [5.0, 6.0, 7.0, 8.0],
    /// ]);
    /// let tt = a.transposed();
    /// assert_eq!(
    ///     format!("{}", tt),
    ///     "┌     ┐\n\
    ///      │ 1 5 │\n\
    ///      │ 2 6 │\n\
    ///      │ 3 7 │\n\
    ///      │ 4 8 │\n\
    ///      └     ┘"
    /// );
    /// ```
    pub fn transposed(&self) -> Self {
        let (m, n) = (self.nrow, self.ncol);
        let mut tt = NumMatrix::new(n, m);
        for i in 0..m {
            for j in 0..n {
                tt.data[j + i * n] = self.data[i + j * m];
            }
        }
        tt
    }
}

impl<T> fmt::Display for NumMatrix<T>
where
    T: AddAssign + MulAssign + Num + Copy + DeserializeOwned + Serialize + fmt::Display,
{
    /// Generates a string representation of the NumMatrix
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumMatrix;
    /// let a = NumMatrix::<f64>::from(&[
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
            write!(f, "[]").unwrap();
            return Ok(());
        }
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..self.nrow {
            for j in 0..self.ncol {
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
        write!(f, "┌{:1$}┐\n", " ", width * self.ncol + 1).unwrap();
        for i in 0..self.nrow {
            if i > 0 {
                write!(f, " │\n").unwrap();
            }
            for j in 0..self.ncol {
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
        write!(f, "└{:1$}┘", " ", width * self.ncol + 1).unwrap();
        Ok(())
    }
}

/// Allows accessing NumMatrix as an Array2D
impl<'a, T: 'a> AsArray2D<'a, T> for NumMatrix<T>
where
    T: AddAssign + MulAssign + Num + Copy + DeserializeOwned + Serialize,
{
    #[inline]
    fn size(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }
    #[inline]
    fn at(&self, i: usize, j: usize) -> T {
        self.get(i, j)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NumMatrix;
    use crate::AsArray2D;

    #[test]
    fn new_works() {
        let u = NumMatrix::<f64>::new(3, 3);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(u.data, correct);
    }

    #[test]
    fn identity_works() {
        let identity = NumMatrix::<f64>::identity(2);
        assert_eq!(identity.data, &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn filled_works() {
        let a = NumMatrix::<f64>::filled(2, 2, 3.0);
        assert_eq!(a.data, &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn initialized_works() {
        // 0 0 0
        // 1 1 1
        let a = NumMatrix::<usize>::initialized(2, 3, |i, _| i);
        assert_eq!(a.data, &[0, 1, 0, 1, 0, 1]);

        // 0 1 2
        // 0 1 2
        let a = NumMatrix::<usize>::initialized(2, 3, |_, j| j);
        assert_eq!(a.data, &[0, 0, 1, 1, 2, 2]);

        // 11 12 13 14
        // 21 22 23 24
        let a = NumMatrix::<f64>::initialized(2, 4, |i, j| (10 * (i + 1) + (j + 1)) as f64);
        assert_eq!(a.data, &[11.0, 21.0, 12.0, 22.0, 13.0, 23.0, 14.0, 24.0])
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
        let a = NumMatrix::<f64>::from(&a_data);
        assert_eq!(a.data, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);

        // heap-allocated 2D array (aka slice of slices)
        #[rustfmt::skip]
        let b_data: &[&[f64]] = &[
            &[10.0, 20.0],
            &[30.0, 40.0, IGNORED],
            &[50.0, 60.0, IGNORED, IGNORED],
        ];
        let b = NumMatrix::<f64>::from(&b_data);
        assert_eq!(b.data, &[10.0, 30.0, 50.0, 20.0, 40.0, 60.0]);

        // stack-allocated (fixed-size) 2D array
        #[rustfmt::skip]
        let c_data = [
            [100.0, 200.0],
            [300.0, 400.0],
            [500.0, 600.0],
        ];
        let c = NumMatrix::<f64>::from(&c_data);
        assert_eq!(c.data, &[100.0, 300.0, 500.0, 200.0, 400.0, 600.0]);
    }

    #[test]
    fn from_lower_works() {
        // heap-allocated 2D array (vector of vectors)
        const IGNORED: f64 = 123.456;
        let a_data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, IGNORED, IGNORED, IGNORED],
            vec![5.0, 6.0],
        ];
        assert_eq!(
            NumMatrix::<f64>::from_lower(&a_data).err(),
            Some("the matrix must be square to set lower triangle")
        );
        let a_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 5.0, IGNORED, IGNORED, IGNORED],
            vec![3.0, 5.0, 6.0],
        ];
        let a = NumMatrix::<f64>::from_lower(&a_data).unwrap();
        assert_eq!(
            a.data,
            &[
                1.0, 2.0, 3.0, // first column
                0.0, 4.0, 5.0, // second column
                0.0, 0.0, 6.0, // third column
            ]
        );
        // empty matrix
        let a_data: [[f64; 0]; 1] = [[]];
        let a = NumMatrix::<f64>::from_lower(&a_data).unwrap();
        assert_eq!(a.dims(), (0, 0));
        assert_eq!(a.data.len(), 0);
    }

    #[test]
    fn from_upper_works() {
        // heap-allocated 2D array (vector of vectors)
        const IGNORED: f64 = 123.456;
        let a_data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, IGNORED, IGNORED, IGNORED],
            vec![5.0, 6.0],
        ];
        assert_eq!(
            NumMatrix::<f64>::from_upper(&a_data).err(),
            Some("the matrix must be square to set upper triangle")
        );
        let a_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 5.0, IGNORED, IGNORED, IGNORED],
            vec![3.0, 5.0, 6.0],
        ];
        let a = NumMatrix::<f64>::from_upper(&a_data).unwrap();
        assert_eq!(
            a.data,
            &[
                1.0, 0.0, 0.0, // first column
                2.0, 4.0, 0.0, // second column
                3.0, 5.0, 6.0, // third column
            ]
        );
        // empty matrix
        let a_data: [[f64; 0]; 1] = [[]];
        let a = NumMatrix::<f64>::from_upper(&a_data).unwrap();
        assert_eq!(a.dims(), (0, 0));
        assert_eq!(a.data.len(), 0);
    }

    #[test]
    fn from_0_works() {
        let a_data: &[&[f64]] = &[&[]];
        let a = NumMatrix::<f64>::from(&a_data);
        assert_eq!(a.nrow, 0);
        assert_eq!(a.ncol, 0);
        assert_eq!(a.data.len(), 0);
    }

    #[test]
    fn diagonal_works() {
        let a = NumMatrix::<f64>::diagonal(&[-8.0, 2.0, 1.0]);
        assert_eq!(a.nrow, 3);
        assert_eq!(a.ncol, 3);
        assert_eq!(a.data, [-8.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn from_text_file_handles_problems() {
        assert_eq!(NumMatrix::<f64>::from_text_file("").err(), Some("cannot open file"),);
        assert_eq!(
            NumMatrix::<f64>::from_text_file("not-found").err(),
            Some("cannot open file"),
        );
        assert_eq!(
            NumMatrix::<f64>::from_text_file("./data/matrices/bad_missing_data.txt").err(),
            Some("column data is missing"),
        );
        assert_eq!(
            NumMatrix::<f64>::from_text_file("./data/matrices/bad_wrong_data.txt").err(),
            Some("cannot parse value"),
        );
    }

    #[test]
    fn from_text_file_works() {
        let a = NumMatrix::<f64>::from_text_file("./data/matrices/ok_empty_file.txt").unwrap();
        assert_eq!(a.nrow, 0);
        assert_eq!(a.ncol, 0);
        assert_eq!(a.data.len(), 0);

        let a = NumMatrix::<f64>::from_text_file("./data/matrices/ok_no_data.txt").unwrap();
        assert_eq!(a.nrow, 0);
        assert_eq!(a.ncol, 0);
        assert_eq!(a.data.len(), 0);
        assert_eq!(format!("{}", a), "[]");

        let a = NumMatrix::<f64>::from_text_file("./data/matrices/ok_single_value.txt").unwrap();
        assert_eq!(a.nrow, 1);
        assert_eq!(a.ncol, 1);
        assert_eq!(a.data.len(), 1);
        assert_eq!(
            format!("{}", a),
            "┌   ┐\n\
             │ 1 │\n\
             └   ┘"
        );

        let a = NumMatrix::<f64>::from_text_file("./data/matrices/ok1.txt").unwrap();
        assert_eq!(a.nrow, 3);
        assert_eq!(a.ncol, 3);
        assert_eq!(a.data.len(), 9);
        assert_eq!(
            format!("{}", a),
            "┌       ┐\n\
             │ 1 2 3 │\n\
             │ 4 5 6 │\n\
             │ 7 8 9 │\n\
             └       ┘"
        );
    }

    #[test]
    fn nrow_works() {
        let a = NumMatrix::<f64>::new(4, 3);
        assert_eq!(a.nrow(), 4);
    }

    #[test]
    fn ncol_works() {
        let a = NumMatrix::<f64>::new(4, 3);
        assert_eq!(a.ncol(), 3);
    }

    #[test]
    fn dims_works() {
        let a = NumMatrix::<f64>::new(5, 4);
        assert_eq!(a.dims(), (5, 4));
    }

    #[test]
    fn display_works() {
        let a_0x0 = NumMatrix::<f64>::new(0, 0);
        let a_0x1 = NumMatrix::<f64>::new(0, 1);
        let a_1x0 = NumMatrix::<f64>::new(1, 0);
        assert_eq!(format!("{}", a_0x0), "[]");
        assert_eq!(format!("{}", a_0x1), "[]");
        assert_eq!(format!("{}", a_1x0), "[]");
        #[rustfmt::skip]
        let a = NumMatrix::<f64>::from(&[
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
    }

    #[test]
    fn display_precision_works() {
        #[rustfmt::skip]
        let a = NumMatrix::<f64>::from(&[
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
    }

    #[test]
    fn debug_works() {
        let a = NumMatrix::<f64>::new(1, 1);
        assert_eq!(format!("{:?}", a), "NumMatrix { nrow: 1, ncol: 1, data: [0.0] }");
    }

    #[test]
    fn fill_works() {
        let mut a = NumMatrix::<f64>::new(2, 2);
        a.fill(7.7);
        let correct = &[7.7, 7.7, 7.7, 7.7];
        assert_eq!(a.data, correct);
    }

    #[test]
    #[should_panic]
    fn get_panics_on_wrong_indices() {
        let a = NumMatrix::<f64>::new(1, 1);
        a.get(1, 0);
    }

    #[test]
    fn get_works() {
        #[rustfmt::skip]
        let a = NumMatrix::<f64>::from(&[
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
        let mut a = NumMatrix::<f64>::new(1, 1);
        a.set(1, 0, 0.0);
    }

    #[test]
    fn set_works() {
        #[rustfmt::skip]
        let mut a = NumMatrix::<f64>::from(&[
            [1.0, 2.0],
            [3.0, 4.0],
        ]);
        a.set(0, 0, -1.0);
        a.set(0, 1, -2.0);
        a.set(1, 0, -3.0);
        a.set(1, 1, -4.0);
        assert_eq!(a.data, &[-1.0, -3.0, -2.0, -4.0]);
    }

    #[test]
    #[should_panic]
    fn add_panics_on_wrong_indices() {
        let mut a = NumMatrix::<f64>::new(1, 1);
        a.add(1, 0, 0.0);
    }

    #[test]
    fn add_works() {
        #[rustfmt::skip]
        let mut a = NumMatrix::<f64>::from(&[
            [1.0, 2.0],
            [3.0, 4.0],
        ]);
        a.add(0, 0, 1.0);
        a.add(0, 1, 2.0);
        a.add(1, 0, 3.0);
        a.add(1, 1, 4.0);
        assert_eq!(a.data, &[2.0, 6.0, 4.0, 8.0]);
    }

    #[test]
    #[should_panic]
    fn mul_panics_on_wrong_indices() {
        let mut a = NumMatrix::<f64>::new(1, 1);
        a.mul(1, 0, 0.0);
    }

    #[test]
    fn mul_works() {
        #[rustfmt::skip]
        let mut a = NumMatrix::<f64>::from(&[
            [1.0, 2.0],
            [3.0, 4.0],
        ]);
        a.mul(0, 0, -1.0);
        a.mul(0, 1, -2.0);
        a.mul(1, 0, -3.0);
        a.mul(1, 1, -4.0);
        assert_eq!(a.data, &[-1.0, -9.0, -4.0, -16.0]);
    }

    #[test]
    fn extract_row_works() {
        #[rustfmt::skip]
        let a = NumMatrix::<f64>::from(&[
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0],
            [4.0, 8.0],
        ]);
        let first_row = a.extract_row(0);
        let second_row = a.extract_row(1);
        assert_eq!(first_row, [1.0, 5.0]);
        assert_eq!(second_row, [2.0, 6.0]);
    }

    #[test]
    fn extract_column_works() {
        #[rustfmt::skip]
        let a = NumMatrix::<f64>::from(&[
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0],
            [4.0, 8.0],
        ]);
        let first_column = a.extract_column(0);
        let second_column = a.extract_column(1);
        assert_eq!(first_column, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(second_column, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn transposed_works() {
        // nrow < ncol
        #[rustfmt::skip]
        let a = NumMatrix::<f64>::from(&[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]);
        let tt = a.transposed();
        assert_eq!(
            format!("{}", tt),
            "┌     ┐\n\
             │ 1 5 │\n\
             │ 2 6 │\n\
             │ 3 7 │\n\
             │ 4 8 │\n\
             └     ┘"
        );
        // nrow = ncol
        #[rustfmt::skip]
        let a = NumMatrix::<f64>::from(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        let tt = a.transposed();
        assert_eq!(
            format!("{}", tt),
            "┌       ┐\n\
             │ 1 4 7 │\n\
             │ 2 5 8 │\n\
             │ 3 6 9 │\n\
             └       ┘"
        );
        // nrow > ncol
        #[rustfmt::skip]
        let a = NumMatrix::<f64>::from(&[
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0],
            [4.0, 8.0],
        ]);
        let tt = a.transposed();
        assert_eq!(
            format!("{}", tt),
            "┌         ┐\n\
             │ 1 2 3 4 │\n\
             │ 5 6 7 8 │\n\
             └         ┘"
        );
    }

    #[test]
    fn clone_and_serialize_work() {
        #[rustfmt::skip]
        let mut a = NumMatrix::<f64>::from(&[
            [1.0, 2.0],
            [3.0, 4.0],
        ]);
        let a_copy = a.clone();
        a.set(0, 0, 0.11);
        a.set(0, 1, 0.22);
        a.set(1, 0, 0.33);
        a.set(1, 1, 0.44);
        assert_eq!(a.data, &[0.11, 0.33, 0.22, 0.44]);
        assert_eq!(a_copy.data, &[1.0, 3.0, 2.0, 4.0]);

        #[rustfmt::skip]
        let a = NumMatrix::<f64>::from(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);

        // clone
        let mut cloned = a.clone();
        cloned.set(0, 0, -1.0);
        assert_eq!(
            format!("{}", a),
            "┌       ┐\n\
             │ 1 2 3 │\n\
             │ 4 5 6 │\n\
             │ 7 8 9 │\n\
             └       ┘"
        );
        assert_eq!(
            format!("{}", cloned),
            "┌          ┐\n\
             │ -1  2  3 │\n\
             │  4  5  6 │\n\
             │  7  8  9 │\n\
             └          ┘"
        );

        // serialize to json
        let json = serde_json::to_string(&a).unwrap();
        assert_eq!(
            json,
            r#"{"nrow":3,"ncol":3,"data":[1.0,4.0,7.0,2.0,5.0,8.0,3.0,6.0,9.0]}"#
        );

        // deserialize from json
        let from_json: NumMatrix<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(
            format!("{}", from_json),
            "┌       ┐\n\
             │ 1 2 3 │\n\
             │ 4 5 6 │\n\
             │ 7 8 9 │\n\
             └       ┘"
        );
    }

    fn array_2d_test<'a, T, U>(array: &'a T) -> String
    where
        T: AsArray2D<'a, U>,
        U: 'a + std::fmt::Debug,
    {
        format!("size = {:?}", array.size()).to_string()
    }

    #[test]
    fn as_array_2d_works() {
        let u = NumMatrix::<i32>::from(&[[1, 2], [3, 4]]);
        assert_eq!(array_2d_test(&u), "size = (2, 2)");
    }
}
