use std::cmp;
use std::fmt::{self, Write};

pub struct Matrix {
    pub(super) nrow: usize,    // number of rows
    pub(super) ncol: usize,    // number of columns
    pub(super) data: Vec<f64>, // col-major => Fortran
}

/// Holds matrix components
///
/// # Note
///
/// Data is stored in col-major format
///
/// Example of col-major data:
///
/// ```text
///       _      _
///      |  0  3  |
///  A = |  1  4  |            ⇒     a = [0, 1, 2, 3, 4, 5]
///      |_ 2  5 _|(m x n)
///
///  a[i+j*m] = A[i][j]
/// ```
///
impl Matrix {
    /// Creates new (nrow x ncol) Matrix filled with zeros
    pub fn new(nrow: usize, ncol: usize) -> Self {
        Matrix {
            nrow,
            ncol,
            data: vec![0.0; nrow * ncol],
        }
    }

    /// Creates new matrix from given data
    ///
    /// # Panics
    ///
    /// This function panics if there are rows with different number of columns
    ///
    pub fn from(data: &[&[f64]]) -> Self {
        let nrow = data.len();
        if nrow == 0 {
            return Matrix {
                nrow: 0,
                ncol: 0,
                data: Vec::new(),
            };
        }
        let ncol = data[0].len();
        let mut matrix = Matrix {
            nrow,
            ncol,
            data: vec![0.0; nrow * ncol],
        };
        for i in 0..nrow {
            if data[i].len() != ncol {
                panic!("all rows must have the same number of columns");
            }
            for j in 0..ncol {
                matrix.data[i + j * nrow] = data[i][j];
            }
        }
        matrix
    }

    /// Returns the number of rows
    pub fn nrow(&self) -> usize {
        self.nrow
    }

    /// Returns the number of columns
    pub fn ncol(&self) -> usize {
        self.ncol
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
                write!(&mut buf, "{}", val)?;
                width = cmp::max(buf.chars().count(), width);
                buf.clear();
            }
        }
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
                write!(f, "{:>1$}", val, width)?;
            }
        }
        write!(f, " │\n")?;
        write!(f, "└{:1$}┘", " ", width * self.ncol + 1)?;
        Ok(())
    }
}

mod mat_mat_mul;
pub use crate::matrix::mat_mat_mul::*;

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
    fn from_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
        ]);
        let correct = &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        assert_vec_approx_eq!(a.data, correct, 1e-15);
    }

    #[test]
    #[should_panic(expected = "all rows must have the same number of columns")]
    fn from_panics_on_wrong_columns() {
        #[rustfmt::skip]
         Matrix::from(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0],
            &[7.0, 8.0, 8.0],
        ]);
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
    fn display_trait_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
        ]);
        let correct = "┌       ┐\n\
                            │ 1 2 3 │\n\
                            │ 4 5 6 │\n\
                            │ 7 8 9 │\n\
                            └       ┘";
        assert_eq!(format!("{}", a), correct);
    }
}
