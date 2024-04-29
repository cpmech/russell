// Make sure that these constants match the c-code constants
pub(crate) const SUCCESSFUL_EXIT: i32 = 0;
pub(crate) const ERROR_NULL_POINTER: i32 = 100000;
pub(crate) const ERROR_MALLOC: i32 = 200000;
pub(crate) const ERROR_VERSION: i32 = 300000;
pub(crate) const ERROR_NOT_AVAILABLE: i32 = 400000;
pub(crate) const ERROR_NEED_INITIALIZATION: i32 = 500000;
pub(crate) const ERROR_NEED_FACTORIZATION: i32 = 600000;
pub(crate) const ERROR_ALREADY_INITIALIZED: i32 = 700000;
pub(crate) const ERROR_MPI_INIT_FAILED: i32 = 800000;

/// Represents the type of boolean flags interchanged with the C-code
pub(crate) type CcBool = i32;

/// Converts usize to i32
///
/// # Panics
///
/// Will panic if usize is too large to be an i32
#[inline]
pub(crate) fn to_i32(num: usize) -> i32 {
    i32::try_from(num).unwrap()
}
