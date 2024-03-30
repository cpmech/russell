/// Defines the vector size to decide when to use the native Rust code or BLAS
pub(crate) const MAX_DIM_FOR_NATIVE_BLAS: usize = 16;

// -------------------------------------------------------------------------------------------
// IMPORTANT: The constants below must match the corresponding C-code constants in constants.h

// Represents the type of boolean flags interchanged with the C-code
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
pub(crate) const SVD_CODE_A: i32 = 0;
//pub(crate) const SVD_CODE_S: i32 = 1;
//pub(crate) const SVD_CODE_O: i32 = 2;
//pub(crate) const SVD_CODE_N: i32 = 3;

// From: /usr/include/x86_64-linux-gnu/cblas.h
// From: /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h
pub(crate) const CBLAS_COL_MAJOR: i32 = 102;
pub(crate) const CBLAS_NO_TRANS: i32 = 111;
pub(crate) const CBLAS_TRANS: i32 = 112;
pub(crate) const CBLAS_CONJ_TRANS: i32 = 113;
pub(crate) const CBLAS_UPPER: i32 = 121;
pub(crate) const CBLAS_LOWER: i32 = 122;

// Make sure that these constants match the c-code constants
pub(crate) const SUCCESSFUL_EXIT: i32 = 0;
// pub(crate) const UNKNOWN_ERROR: i32 = 1;

// -------------------------------------------------------------------------------------------
