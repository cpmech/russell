#[inline]
pub(crate) fn mandel_dim(symmetric: bool, two_dim: bool) -> usize {
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
