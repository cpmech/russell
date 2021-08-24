use super::*;

/// Returns evenly spaced numbers over a specified closed interval
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let x = linspace(2.0, 3.0, 5);
/// let correct = "┌      ┐\n\
///                │    2 │\n\
///                │ 2.25 │\n\
///                │  2.5 │\n\
///                │ 2.75 │\n\
///                │    3 │\n\
///                └      ┘";
/// assert_eq!(format!("{}", x), correct);
/// ```
pub fn linspace(start: f64, stop: f64, count: usize) -> Vector {
    let mut res = Vector::new(count);
    if count == 0 {
        return res;
    }
    res.data[0] = start;
    if count == 1 {
        return res;
    }
    res.data[count - 1] = stop;
    if count == 2 {
        return res;
    }
    let step = (stop - start) / ((count - 1) as f64);
    for i in 1..count {
        res.data[i] = start + (i as f64) * step;
    }
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
        let correct = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        assert_vec_approx_eq!(x.data, correct, 1e-15);
    }

    #[test]
    fn linspace_0_works() {
        let x = linspace(2.0, 3.0, 0);
        assert_eq!(x.data.len(), 0);
    }

    #[test]
    fn linspace_1_works() {
        let x = linspace(2.0, 3.0, 1);
        assert_eq!(x.data.len(), 1);
        assert_eq!(x.data[0], 2.0);
    }

    #[test]
    fn linspace_2_works() {
        let x = linspace(2.0, 3.0, 2);
        assert_eq!(x.data.len(), 2);
        assert_eq!(x.data[0], 2.0);
        assert_eq!(x.data[1], 3.0);
    }
}
