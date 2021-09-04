use crate::EnumMatrixNorm;
use russell_openblas::*;
use std::cmp;
use std::fmt::{self, Write};

/// Holds matrix components and associated functions
pub struct Matrix {
    pub(crate) nrow: usize,    // number of rows
    pub(crate) ncol: usize,    // number of columns
    pub(crate) data: Vec<f64>, // col-major => Fortran
}

// # Note
//
// Data is stored in col-major format
//
// Example of col-major data:
//
// ```text
//       _      _
//      |  0  3  |
//  A = |  1  4  |            ⇒     a = [0, 1, 2, 3, 4, 5]
//      |_ 2  5 _|(m x n)
//
//  a[i+j*m] = A[i][j]
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
            matrix.data[i + i * m] = 1.0;
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
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_lab::*;
    /// let a = Matrix::from(&[
    ///     &[1.0, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    ///     &[7.0, 8.0, 9.0],
    /// ])?;
    /// let correct = "┌       ┐\n\
    ///                │ 1 2 3 │\n\
    ///                │ 4 5 6 │\n\
    ///                │ 7 8 9 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from(data: &[&[f64]]) -> Result<Self, &'static str> {
        let nrow = data.len();
        let ncol = data[0].len();
        let mut matrix = Matrix {
            nrow,
            ncol,
            data: vec![0.0; nrow * ncol],
        };
        for i in 0..nrow {
            if data[i].len() != ncol {
                return Err("all rows must have the same number of columns");
            }
            for j in 0..ncol {
                matrix.data[i + j * nrow] = data[i][j];
            }
        }
        Ok(matrix)
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
            matrix.data[i + i * nrow] = data[i];
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
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_lab::*;
    /// let mut a = Matrix::from(&[
    ///     &[1.0, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    /// ])?;
    /// a.scale(0.5);
    /// let correct = "┌             ┐\n\
    ///                │ 0.5   1 1.5 │\n\
    ///                │   2 2.5   3 │\n\
    ///                └             ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// # Ok(())
    /// # }
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

    /// Returns the (i,j) component
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_lab::*;
    /// let a = Matrix::from(&[
    ///     &[1.0, 2.0],
    ///     &[3.0, 4.0],
    /// ])?;
    /// assert_eq!(a.get(1,1), 4.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        self.data[i + j * self.nrow]
    }

    /// Change the (i,j) component
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_lab::*;
    /// let mut a = Matrix::from(&[
    ///     &[1.0, 2.0],
    ///     &[3.0, 4.0],
    /// ])?;
    /// a.set(1, 1, -4.0);
    /// let correct = "┌       ┐\n\
    ///                │  1  2 │\n\
    ///                │  3 -4 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        self.data[i + j * self.nrow] = value;
    }

    /// Executes the += operation on the (i,j) component
    ///
    /// ```text
    /// a_ij += value
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_lab::*;
    /// let mut a = Matrix::from(&[
    ///     &[1.0, 2.0],
    ///     &[3.0, 4.0],
    /// ])?;
    /// a.plus_equal(1, 1, 0.44);
    /// let correct = "┌           ┐\n\
    ///                │ 1.00 2.00 │\n\
    ///                │ 3.00 4.44 │\n\
    ///                └           ┘";
    /// assert_eq!(format!("{:.2}", a), correct);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn plus_equal(&mut self, i: usize, j: usize, value: f64) {
        self.data[i + j * self.nrow] += value;
    }

    /// Returns a copy of this matrix
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_lab::*;
    /// let mut a = Matrix::from(&[
    ///     &[1.0, 2.0],
    ///     &[3.0, 4.0],
    /// ])?;
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
    /// # Ok(())
    /// # }
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
    /// ‖a‖_1 = max_j ( Σ_i |aij| )
    ///
    /// ‖a‖_∞ = max_i ( Σ_j |aij| )
    ///
    /// ‖a‖_F = sqrt(Σ_i Σ_j aij⋅aij) == ‖a‖_2
    ///
    /// ‖a‖_max = max_ij ( |aij| )
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_lab::*;
    /// let a = Matrix::from(&[
    ///     &[-2.0,  2.0],
    ///     &[ 1.0, -4.0],
    /// ])?;
    /// assert_eq!(a.norm(EnumMatrixNorm::One), 6.0);
    /// assert_eq!(a.norm(EnumMatrixNorm::Inf), 5.0);
    /// assert_eq!(a.norm(EnumMatrixNorm::Fro), 5.0);
    /// assert_eq!(a.norm(EnumMatrixNorm::Max), 4.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm(&self, kind: EnumMatrixNorm) -> f64 {
        let norm = match kind {
            EnumMatrixNorm::One => b'1',
            EnumMatrixNorm::Inf => b'I',
            EnumMatrixNorm::Fro => b'F',
            EnumMatrixNorm::Max => b'M',
        };
        let (m, n) = (to_i32(self.nrow), to_i32(self.ncol));
        let lda = m;
        dlange(norm, m, n, &self.data, lda)
    }
}

impl fmt::Display for Matrix {
    /// Implements the Display trait
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                let val = self.data[i + j * self.nrow];
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
                let val = self.data[i + j * self.nrow];
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
    fn from_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
        ])?;
        let correct = &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        assert_vec_approx_eq!(a.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn from_0_works() -> Result<(), &'static str> {
        let data: &[&[f64]] = &[&[]];
        let a = Matrix::from(data)?;
        assert_eq!(a.data.len(), 0);
        Ok(())
    }

    #[test]
    fn from_fails_on_wrong_columns() {
        #[rustfmt::skip]
        let res = Matrix::from(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0],
            &[7.0, 8.0, 8.0],
        ]);
        assert_eq!(res.err(), Some("all rows must have the same number of columns"));
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
    fn display_trait_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
        ])?;
        let correct = "┌       ┐\n\
                            │ 1 2 3 │\n\
                            │ 4 5 6 │\n\
                            │ 7 8 9 │\n\
                            └       ┘";
        assert_eq!(format!("{}", a), correct);
        Ok(())
    }

    #[test]
    fn display_trait_precision_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[1.0111111, 2.02222222, 3.033333],
            &[4.0444444, 5.05555555, 6.066666],
            &[7.0777777, 8.08888888, 9.099999],
        ])?;
        let correct = "┌                ┐\n\
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
            &[ 6.0,  9.0,  12.0],
            &[-6.0, -9.0, -12.0],
        ])?;
        a.scale(1.0 / 3.0);
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[ 2.0,  3.0,  4.0],
            &[-2.0, -3.0, -4.0],
        ])?;
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
    fn get_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
        ])?;
        assert_eq!(a.get(0, 0), 1.0);
        assert_eq!(a.get(0, 1), 2.0);
        assert_eq!(a.get(1, 0), 3.0);
        assert_eq!(a.get(1, 1), 4.0);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn set_panics_on_wrong_indices() {
        let mut a = Matrix::new(1, 1);
        a.set(1, 0, 0.0);
    }

    #[test]
    fn set_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
        ])?;
        a.set(0, 0, -1.0);
        a.set(0, 1, -2.0);
        a.set(1, 0, -3.0);
        a.set(1, 1, -4.0);
        assert_eq!(a.data, &[-1.0, -3.0, -2.0, -4.0]);
        Ok(())
    }

    #[test]
    fn plus_equal_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
        ])?;
        a.plus_equal(0, 0, 0.11);
        a.plus_equal(0, 1, 0.22);
        a.plus_equal(1, 0, 0.33);
        a.plus_equal(1, 1, 0.44);
        assert_eq!(a.data, &[1.11, 3.33, 2.22, 4.44]);
        Ok(())
    }

    #[test]
    fn get_copy_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
        ])?;
        let a_copy = a.get_copy();
        a.set(0, 0, 0.11);
        a.set(0, 1, 0.22);
        a.set(1, 0, 0.33);
        a.set(1, 1, 0.44);
        assert_eq!(a.data, &[0.11, 0.33, 0.22, 0.44]);
        assert_eq!(a_copy.data, &[1.0, 3.0, 2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn norm_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[ 5.0, -4.0, 2.0],
            &[-1.0,  2.0, 3.0],
            &[-2.0,  1.0, 0.0],
        ])?;
        assert_eq!(a.norm(EnumMatrixNorm::One), 8.0);
        assert_eq!(a.norm(EnumMatrixNorm::Inf), 11.0);
        assert_eq!(a.norm(EnumMatrixNorm::Fro), 8.0);
        assert_eq!(a.norm(EnumMatrixNorm::Max), 5.0);
        Ok(())
    }
}
