use crate::NumVector;
use num_complex::Complex64;

/// Vector is an alias to NumVector<f64> and is used in most functions that call OpenBLAS
pub type Vector = NumVector<f64>;

/// ComplexVector is an alias to NumVector<Complex64> and is used in most functions that call OpenBLAS
pub type ComplexVector = NumVector<Complex64>;
