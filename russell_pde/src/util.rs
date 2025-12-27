/// Kronecker delta function
#[inline]
pub(crate) fn delta(i: usize, j: usize) -> f64 {
    if i == j {
        1.0
    } else {
        0.0
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::delta;

    #[test]
    fn delta_works() {
        assert_eq!(delta(0, 0), 1.0);
        assert_eq!(delta(1, 1), 1.0);
        assert_eq!(delta(10, 10), 1.0);
        assert_eq!(delta(0, 1), 0.0);
        assert_eq!(delta(1, 0), 0.0);
        assert_eq!(delta(1, 2), 0.0);
    }
}
