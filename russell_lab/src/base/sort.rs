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

/// Returns the indices that would sort an array of f64
///
/// **Note:** NaN values are handled such that they will be placed at the end of the array.
pub fn argsort_f64(data: &[f64]) -> Vec<usize> {
    // create a vector of (index, value) pairs
    let mut indexed_data: Vec<_> = data.iter().enumerate().collect();

    // sort the pairs by the value, handling NaN values
    indexed_data.sort_by(|a, b| {
        match (a.1.is_nan(), b.1.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,    // both are NaN
            (true, false) => std::cmp::Ordering::Greater, // place NaN after
            (false, true) => std::cmp::Ordering::Less,    // place NaN before
            _ => a.1.partial_cmp(b.1).unwrap(),           // compare normally
        }
    });

    // extract and return the indices from the sorted pairs
    let sorted_indices = indexed_data.iter().map(|(idx, _)| *idx).collect();
    sorted_indices
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{argsort_f64, sort2, sort3, sort4};

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

        let data = [3.0, 1.0, f64::NAN, 8.0, -5.0];
        assert_eq!(argsort_f64(&data), &[4, 1, 0, 3, 2]);
    }
}
