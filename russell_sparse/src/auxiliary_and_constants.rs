#![allow(unused)]

use crate::StrError;

pub(crate) const SUCCESSFUL_EXIT: i32 = 0;
pub(crate) const NEED_FACTORIZATION: i32 = 100;
pub(crate) const NULL_POINTER_ERROR: i32 = 100000;
pub(crate) const MALLOC_ERROR: i32 = 200000;
pub(crate) const VERSION_ERROR: i32 = 300000;
pub(crate) const NOT_AVAILABLE: i32 = 400000;

/// Represents the type of boolean flags interchanged with the C-code
pub(crate) type CcBool = i32;

/// Converts usize to i32
#[inline]
pub(crate) fn to_i32(num: usize) -> Result<i32, StrError> {
    i32::try_from(num).map_err(|_| "cannot downcast usize to i32")
}
