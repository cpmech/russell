/// Converts usize to i32
///
/// # Panics
///
/// Will panic if usize is too large to be an i32
#[inline]
pub(crate) fn to_i32(num: usize) -> i32 {
    i32::try_from(num).unwrap()
}
