/// Returns evenly spaced numbers over a specified closed interval
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let x = linspace(2.0, 3.0, 5);
/// assert_eq!(x, &[2.0, 2.25, 2.5, 2.75, 3.0]);
/// ```
pub fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![start];
    }
    let step = (stop - start) / ((count - 1) as f64);
    let mut res = vec![start; count];
    for i in 1..count {
        res[i] = start + (i as f64) * step;
    }
    res[count - 1] = stop;
    res
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn linspace_works() {
        let x = linspace(0.0, 1.0, 11);
        let x_correct = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        assert_vec_approx_eq!(x, x_correct, 1e-15);
    }
}
