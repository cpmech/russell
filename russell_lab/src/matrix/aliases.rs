use crate::NumMatrix;
use num_complex::Complex64;

/// Matrix is an alias to NumMatrix<f64> and is used in most functions that call OpenBLAS
pub type Matrix = NumMatrix<f64>;

/// ComplexMatrix is an alias to NumMatrix<Complex64> and is used in most functions that call OpenBLAS
pub type ComplexMatrix = NumMatrix<Complex64>;
