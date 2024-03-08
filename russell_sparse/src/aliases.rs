use crate::{NumCooMatrix, NumCscMatrix, NumCsrMatrix, NumSparseMatrix};
use num_complex::Complex64;

/// Defines an alias to NumCooMatrix with f64
pub type CooMatrix = NumCooMatrix<f64>;

/// Defines an alias to NumCscMatrix with f64
pub type CscMatrix = NumCscMatrix<f64>;

/// Defines an alias to NumCsrMatrix with f64
pub type CsrMatrix = NumCsrMatrix<f64>;

/// Defines an alias to NumSparseMatrix with f64
pub type SparseMatrix = NumSparseMatrix<f64>;

/// Defines an alias to NumCooMatrix with Complex64
pub type ComplexCooMatrix = NumCooMatrix<Complex64>;

/// Defines an alias to NumCscMatrix with Complex64
pub type ComplexCscMatrix = NumCscMatrix<Complex64>;

/// Defines an alias to NumCsrMatrix with Complex64
pub type ComplexCsrMatrix = NumCsrMatrix<Complex64>;

/// Defines an alias to NumSparseMatrix with Complex64
pub type ComplexSparseMatrix = NumSparseMatrix<Complex64>;
