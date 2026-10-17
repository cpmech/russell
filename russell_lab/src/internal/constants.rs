/// Defines the vector size to decide when to use the native Rust code or BLAS
pub(crate) const MAX_DIM_FOR_NATIVE_BLAS: usize = 16;

// -------------------------------------------------------------------------------------------
// IMPORTANT: The constants below must match the corresponding C-code constants in constants.h

// Represents the type of boolean flags interchanged with the C-code.
// Only the C shim uses these three; a native build has no C layer, so they are
// gated rather than left to warn as unused.
// The `oracle` feature is included because its C shim (and the parity tests
// that use it) is the same kind of C-facing layer.
#[cfg(any(feature = "oblas", feature = "mkl", feature = "oracle"))]
pub(crate) type CcBool = i32;

// Boolean flags
pub(crate) const C_TRUE: i32 = 1;
pub(crate) const C_FALSE: i32 = 0;

// Norm codes
pub(crate) const NORM_EUC: isize = 0;
pub(crate) const NORM_FRO: isize = 1;
pub(crate) const NORM_INF: isize = 2;
pub(crate) const NORM_MAX: isize = 3;
pub(crate) const NORM_ONE: isize = 4;

// SVD codes
#[cfg(any(feature = "oblas", feature = "mkl"))]
pub(crate) const SVD_CODE_A: i32 = 0;
//pub(crate) const SVD_CODE_S: i32 = 1;
//pub(crate) const SVD_CODE_O: i32 = 2;
//pub(crate) const SVD_CODE_N: i32 = 3;

// From: /usr/include/x86_64-linux-gnu/cblas.h
// From: /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h
#[cfg(any(feature = "oblas", feature = "mkl"))]
pub(crate) const CBLAS_COL_MAJOR: i32 = 102;
pub(crate) const CBLAS_NO_TRANS: i32 = 111;
pub(crate) const CBLAS_TRANS: i32 = 112;
pub(crate) const CBLAS_CONJ_TRANS: i32 = 113;
pub(crate) const CBLAS_UPPER: i32 = 121;
pub(crate) const CBLAS_LOWER: i32 = 122;

// -------------------------------------------------------------------------------------------
