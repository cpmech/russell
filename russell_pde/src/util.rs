/// Kronecker delta function
#[inline]
pub(crate) fn delta(i: usize, j: usize) -> f64 {
    if i == j {
        1.0
    } else {
        0.0
    }
}
