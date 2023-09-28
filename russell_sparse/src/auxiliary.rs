use crate::StrError;

/// Converts usize to i32
#[inline]
pub(crate) fn to_i32(num: usize) -> Result<i32, StrError> {
    i32::try_from(num).map_err(|_| "cannot downcast usize to i32")
}
