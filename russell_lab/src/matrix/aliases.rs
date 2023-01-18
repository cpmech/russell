use crate::NumMatrix;
use num_complex::Complex64;

/// Matrix is an alias to NumMatrix&lt;f64&gt; and is used in most functions that call OpenBLAS
pub type Matrix = NumMatrix<f64>;

/// ComplexMatrix is an alias to NumMatrix&lt;Complex64&gt; and is used in most functions that call OpenBLAS
pub type ComplexMatrix = NumMatrix<Complex64>;
