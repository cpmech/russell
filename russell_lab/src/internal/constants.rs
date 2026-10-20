/// Defines the vector size to decide when to use the native Rust code or BLAS
pub(crate) const MAX_DIM_FOR_NATIVE_BLAS: usize = 16;

// -------------------------------------------------------------------------------------------
// IMPORTANT: The constants below must match the corresponding C-code constants in constants.h

// Represents the type of boolean flags interchanged with the C-code.
// Only the C shim uses these three; a native build has no C layer, so they are
// gated rather than left to warn as unused.
// The `dev_lapack` feature is included because its C shim (and the parity tests
// that use it) is the same kind of C-facing layer.
#[cfg(any(feature = "open_blas", feature = "intel_mkl", feature = "dev_lapack"))]
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
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
pub(crate) const SVD_CODE_A: i32 = 0;
//pub(crate) const SVD_CODE_S: i32 = 1;
//pub(crate) const SVD_CODE_O: i32 = 2;
//pub(crate) const SVD_CODE_N: i32 = 3;

// The CBLAS enum values that the per-call-site `extern "C"` declarations used to
// carry (`CblasColMajor`, `CblasNoTrans`, `CblasTrans`, `CblasUpper`,
// `CblasLower`) are no longer named on the Rust side: `blas_dispatch` passes the
// orientation and the triangle as `bool`s and `c_code/interface_blas.c` is the one
// place that translates them into the library's spelling. Their values, from
// `/usr/include/x86_64-linux-gnu/cblas.h` and `mkl_cblas.h`, are ColMajor 102,
// NoTrans 111, Trans 112, ConjTrans 113, Upper 121 and Lower 122, should a future
// shim entry need them.

// -------------------------------------------------------------------------------------------
