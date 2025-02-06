use std::mem;

/// Sorts 2 values in ascending order
///
/// # Examples
///
/// ```
/// use russell_lab::sort2;
///
/// let mut numbers = (3.0, 2.0);
/// sort2(&mut numbers);
/// assert_eq!(numbers, (2.0, 3.0));
/// ```
pub fn sort2<T>(x: &mut (T, T))
where
    T: PartialOrd,
{
    if x.1 < x.0 {
        mem::swap(&mut x.1, &mut x.0);
    }
}

/// Sorts 3 values in ascending order
///
/// # Examples
///
/// ```
/// use russell_lab::sort3;
///
/// let mut numbers = (1.0, 3.0, 2.0);
/// sort3(&mut numbers);
/// assert_eq!(numbers, (1.0, 2.0, 3.0));
/// ```
pub fn sort3<T>(x: &mut (T, T, T))
where
    T: PartialOrd,
{
    if x.1 < x.0 {
        mem::swap(&mut x.1, &mut x.0);
    }
    if x.2 < x.1 {
        mem::swap(&mut x.2, &mut x.1);
    }
    if x.1 < x.0 {
        mem::swap(&mut x.1, &mut x.0);
    }
}

/// Sorts 4 values in ascending order
///
/// # Examples
///
/// ```
/// use russell_lab::sort4;
///
/// let mut numbers = (1.0, 3.0, 2.0, 0.0);
/// sort4(&mut numbers);
/// assert_eq!(numbers, (0.0, 1.0, 2.0, 3.0));
/// ```
pub fn sort4<T>(x: &mut (T, T, T, T))
where
    T: PartialOrd,
{
    if x.1 < x.0 {
        mem::swap(&mut x.0, &mut x.1);
    }
    if x.2 < x.1 {
        mem::swap(&mut x.1, &mut x.2);
    }
    if x.3 < x.2 {
        mem::swap(&mut x.2, &mut x.3);
    }
    if x.1 < x.0 {
        mem::swap(&mut x.0, &mut x.1);
    }
    if x.2 < x.1 {
        mem::swap(&mut x.1, &mut x.2);
    }
    if x.1 < x.0 {
        mem::swap(&mut x.0, &mut x.1);
    }
}

/// Returns the indices that would sort a f64 array in ascending order
///
/// **Warning:** The NaN values are considered equal in this function, i.e., they
/// are not handled well enough and **care should be taken** if NaNs are expected.
///
/// # Arguments
///
/// * `data` -- vector to sort
///
/// # Returns
///
/// A vector of indices that would sort the input vector
///
/// # Examples
///
/// ```
/// use russell_lab::argsort_f64;
///
/// let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0];
/// assert_eq!(argsort_f64(&data), &[1, 3, 6, 0, 9, 2, 4, 8, 7, 5]);
/// ```
pub fn argsort_f64(data: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.sort_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap_or(std::cmp::Ordering::Equal));
    indices
}

/// Returns the indices that would sort two f64 arrays in ascending order
///
/// The function sorts primarily by `x`, then by `y`.
/// Both input vectors must have the same length.
///
/// **Warning:** The NaN values are considered equal in this function, i.e., they
/// are not handled well enough and **care should be taken** if NaNs are expected.
///
/// # Arguments
///
/// * `x` -- first vector to sort by
/// * `y` -- second vector to sort by when `x` values are equal
///
/// # Returns
///
/// A vector of indices that would sort the input vectors
///
/// # Panics
///
/// Panics if the input vectors have different lengths
///
/// # Examples
///
/// ```
/// use russell_lab::argsort2_f64;
///
/// let x = vec![3.0, 1.0, 2.0, 1.0];
/// let y = vec![1.0, 2.0, 1.0, 3.0];
///
/// let sorted_indices = argsort2_f64(&x, &y);
/// assert_eq!(sorted_indices, vec![1, 3, 2, 0]);
/// ```
pub fn argsort2_f64(x: &[f64], y: &[f64]) -> Vec<usize> {
    assert_eq!(x.len(), y.len(), "vectors must have equal length");

    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.sort_by(|&i, &j| {
        x[i].partial_cmp(&x[j])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(y[i].partial_cmp(&y[j]).unwrap_or(std::cmp::Ordering::Equal))
    });
    indices
}

/// Returns the indices that would sort three f64 arrays in ascending order
///
/// The function sorts primarily by `x`, then by `y`, and finally by `z`.
/// All input vectors must have the same length.
///
/// **Warning:** The NaN values are considered equal in this function, i.e., they
/// are not handled well enough and **care should be taken** if NaNs are expected.
///
/// # Arguments
///
/// * `x` -- first vector to sort by
/// * `y` -- second vector to sort by when `x` values are equal
/// * `z` -- third vector to sort by when both `x` and `y` values are equal
///
/// # Returns
///
/// A vector of indices that would sort the input vectors
///
/// # Panics
///
/// Panics if the input vectors have different lengths
///
/// # Examples
///
/// ```
/// use russell_lab::argsort3_f64;
///
/// let x = vec![3.0, 1.0, 2.0];
/// let y = vec![1.0, 2.0, 1.0];
/// let z = vec![5.0, 4.0, 3.0];
///
/// let sorted_indices = argsort3_f64(&x, &y, &z);
/// assert_eq!(sorted_indices, vec![1, 2, 0]);
/// ```
pub fn argsort3_f64(x: &[f64], y: &[f64], z: &[f64]) -> Vec<usize> {
    assert!(
        x.len() == y.len() && y.len() == z.len(),
        "vectors must have equal length"
    );

    let mut indices: Vec<usize> = (0..x.len()).collect();

    indices.sort_by(|&a, &b| {
        x[a].partial_cmp(&x[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal))
            .then(z[a].partial_cmp(&z[b]).unwrap_or(std::cmp::Ordering::Equal))
    });
    indices
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{argsort2_f64, argsort3_f64, argsort_f64, sort2, sort3, sort4};

    #[test]
    fn sort2_works() {
        let mut x = (1, 2);
        sort2(&mut x);
        assert_eq!(x, (1, 2));

        let mut x = (2, 1);
        sort2(&mut x);
        assert_eq!(x, (1, 2));

        let mut x = (1.0, 2.0);
        sort2(&mut x);
        assert_eq!(x, (1.0, 2.0));

        let mut x = (2.0, 1.0);
        sort2(&mut x);
        assert_eq!(x, (1.0, 2.0));
    }

    #[test]
    fn sort3_works() {
        let mut x = (1, 2, 3);
        sort3(&mut x);
        assert_eq!(x, (1, 2, 3));

        let mut x = (1, 3, 2);
        sort3(&mut x);
        assert_eq!(x, (1, 2, 3));

        let mut x = (3, 2, 1);
        sort3(&mut x);
        assert_eq!(x, (1, 2, 3));
    }

    #[test]
    fn sort4_works() {
        let mut x = (1, 2, 3, 4);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (2, 1, 3, 4);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (2, 3, 1, 4);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (2, 3, 4, 1);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (1, 3, 2, 4);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (1, 3, 4, 2);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (3, 1, 2, 4);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (4, 1, 2, 3);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (1, 4, 2, 3);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (4, 3, 2, 1);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));
    }

    #[test]
    fn argsort_f64_works() {
        let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0];
        assert_eq!(argsort_f64(&data), &[1, 3, 6, 0, 9, 2, 4, 8, 7, 5]);

        let data = [3.0, 1.0, 8.0, -5.0];
        assert_eq!(argsort_f64(&data), &[3, 1, 0, 2]);
    }

    #[test]
    fn argsort2_f64_works_basic() {
        let x = vec![3.0, 1.0, 2.0, 1.0];
        let y = vec![1.0, 2.0, 1.0, 3.0];

        let sorted_indices = argsort2_f64(&x, &y);
        assert_eq!(sorted_indices, vec![1, 3, 2, 0]);
    }

    #[test]
    fn argsort2_f64_works_equal_x() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![3.0, 2.0, 1.0];

        let sorted_indices = argsort2_f64(&x, &y);
        assert_eq!(sorted_indices, vec![2, 1, 0]);
    }

    #[test]
    fn argsort2_f64_works_single_element() {
        let x = vec![1.0];
        let y = vec![2.0];

        let sorted_indices = argsort2_f64(&x, &y);
        assert_eq!(sorted_indices, vec![0]);
    }

    #[test]
    #[should_panic(expected = "vectors must have equal length")]
    fn argsort2_f64_works_different_lengths() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0];

        argsort2_f64(&x, &y);
    }

    #[test]
    fn argsort3_f64_works_basic() {
        let x = vec![3.0, 1.0, 2.0, 1.0];
        let y = vec![1.0, 2.0, 1.0, 3.0];
        let z = vec![5.0, 4.0, 3.0, 2.0];

        let sorted_indices = argsort3_f64(&x, &y, &z);
        assert_eq!(sorted_indices, vec![1, 3, 2, 0]);
    }

    #[test]
    fn argsort3_f64_works_equal_x() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![3.0, 2.0, 1.0];
        let z = vec![1.0, 2.0, 3.0];

        let sorted_indices = argsort3_f64(&x, &y, &z);
        assert_eq!(sorted_indices, vec![2, 1, 0]);
    }

    #[test]
    fn argsort3_f64_works_equal_x_y() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![2.0, 2.0, 2.0];
        let z = vec![3.0, 1.0, 2.0];

        let sorted_indices = argsort3_f64(&x, &y, &z);
        assert_eq!(sorted_indices, vec![1, 2, 0]);
    }

    #[test]
    fn argsort3_f64_works_single_element() {
        let x = vec![1.0];
        let y = vec![2.0];
        let z = vec![3.0];

        let sorted_indices = argsort3_f64(&x, &y, &z);
        assert_eq!(sorted_indices, vec![0]);
    }

    #[test]
    #[should_panic(expected = "vectors must have equal length")]
    fn argsort3_f64_works_different_lengths() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0];
        let z = vec![1.0];

        argsort3_f64(&x, &y, &z);
    }
}
