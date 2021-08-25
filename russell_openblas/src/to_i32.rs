/// Converts number to i32
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate russell_openblas;
/// # fn main() -> Result<(), &'static str> {
/// use std::convert::TryFrom;
/// let m = 3_usize;
/// let x = vec![0.0; m];
/// let m_i32 = to_i32!(x.len())?;
/// # Ok(())
/// # }
/// ```
///
/// # Note
///
/// Remember to import:
///
/// ```text
/// use std::convert::TryFrom;
/// ```
#[macro_export]
macro_rules! to_i32 {
    ($x:expr) => {
        i32::try_from($x).map_err(|_| "cannot convert to i32")
    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;

    #[test]
    fn usize_to_i32_works() -> Result<(), &'static str> {
        let m = 2_usize;
        let x = vec![0.0; m];
        let m_i32 = to_i32!(x.len())?;
        assert_eq!(m_i32, 2_i32);
        Ok(())
    }
}
