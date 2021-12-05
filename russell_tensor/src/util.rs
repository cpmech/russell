/// Returns the dimension of a vector corresponding to a Tensor2 in the Mandel system
///
/// # Input
///
/// * `symmetric` -- the Tensor2 is symmetric, i.e., Tij = Tji
/// * `two_dim` -- 2D instead of 3D. Only used if symmetric == true.
///
/// # Example
///
/// ```
/// use russell_tensor::mandel_dim;
/// assert_eq!(mandel_dim(false, false), 9);
/// assert_eq!(mandel_dim(true, false), 6);
/// assert_eq!(mandel_dim(true, true), 4);
/// ```
#[inline]
pub fn mandel_dim(symmetric: bool, two_dim: bool) -> usize {
    if symmetric {
        if two_dim {
            4
        } else {
            6
        }
    } else {
        9
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::mandel_dim;

    #[test]
    fn mandel_dim_works() {
        assert_eq!(mandel_dim(false, false), 9);
        assert_eq!(mandel_dim(true, false), 6);
        assert_eq!(mandel_dim(true, true), 4);
    }
}
