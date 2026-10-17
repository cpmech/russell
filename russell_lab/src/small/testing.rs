//! Helpers shared by the tests of the small-matrix routines.

/// Checks `a · ai = I` to `tol`, entry by entry.
#[cfg(test)]
pub(crate) fn check_inverse<const N: usize>(a: &[[f64; N]; N], ai: &[[f64; N]; N], tol: f64) {
    for i in 0..N {
        for j in 0..N {
            let mut sum = 0.0;
            for k in 0..N {
                sum += a[i][k] * ai[k][j];
            }
            let correct = if i == j { 1.0 } else { 0.0 };
            assert!((sum - correct).abs() <= tol, "a*ai[{i}][{j}] = {sum}");
        }
    }
}
