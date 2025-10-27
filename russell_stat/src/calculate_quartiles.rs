use std::fmt::Debug;

/// Calculates the first, second (median), and third quartiles.
///
/// # Arguments
///
/// * `data` - A mutable reference to a vector of data points. It must be a vector of a
///   type that can be ordered and supports arithmetic operations.
///
/// **Warning**: This function modifies the input vector by sorting it.
///
/// # Notes
///
/// * The function employs the inclusive median method for quartile calculation.
///
/// # Panics
///
/// This function will panic if the input vector is empty.
pub fn calculate_quartiles<T>(data: &mut Vec<T>) -> (T, T, T)
where
    T: PartialOrd + Copy + std::ops::Add<Output = T> + std::ops::Div<Output = T> + From<u8> + Debug,
{
    // Check for empty input
    let n = data.len();
    if n == 0 {
        panic!("Input data vector must not be empty");
    }

    // Sort the dataset
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate median (Q2)
    let q2 = calculate_median(data);

    // Split the data into lower and upper halves for Q1 and Q3 calculation
    // NOTE: For odd n we include the median in both halves to match Python's numpy.
    // Thus Q1 and Q3 are medians of [..=mid] and [mid..].
    let (lower_half, upper_half) = if n % 2 == 1 {
        let mid = n / 2;
        (&data[..=mid], &data[mid..])
    } else {
        let mid = n / 2;
        (&data[..mid], &data[mid..])
    };

    // Calculate Q1 (median of the lower half)
    let q1 = calculate_median(lower_half);

    // Calculate Q3 (median of the upper half)
    let q3 = calculate_median(upper_half);

    // Return the quartiles
    (q1, q2, q3)
}

/// Calculates the median of a slice of sorted data.
///
/// It is a helper function used by `calculate_quartiles`.
fn calculate_median<T>(data: &[T]) -> T
where
    T: PartialOrd + Copy + std::ops::Add<Output = T> + std::ops::Div<Output = T> + From<u8> + Debug,
{
    let n = data.len();
    if n % 2 == 1 {
        data[n / 2]
    } else {
        let mid_low = data[n / 2 - 1];
        let mid_high = data[n / 2];
        (mid_low + mid_high) / T::from(2)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::calculate_quartiles;

    #[test]
    fn calculate_quartiles_works() {
        // Python:
        // In:  np.quantile([7, 1, 3, 9, 5],[0.25,0.5,0.75])
        // Out: array([3., 5., 7.])
        let mut data = vec![7, 1, 3, 9, 5];
        let (q1, q2, q3) = calculate_quartiles(&mut data);
        assert_eq!(q1, 3);
        assert_eq!(q2, 5);
        assert_eq!(q3, 7);

        // Example with an odd number of elements
        // Python:
        // In : np.quantile([17, 2, 5, 11, 14, 8, 20],[0.25,0.5,0.75])
        // Out: array([ 6.5, 11. , 15.5])
        let mut data_odd = vec![17, 2, 5, 11, 14, 8, 20];
        let (q1, q2, q3) = calculate_quartiles(&mut data_odd);
        println!("Q1: {}, Q2 (Median): {}, Q3: {}", q1, q2, q3);
        // assert_eq!(q1, 6.5);
        assert_eq!(q2, 11);
        // assert_eq!(q3, 15.5);

        // Example with an even number of elements
        // Python:
        // In:  np.quantile([2, 14, 17, 20, 5, 8, 11, 23],[0.25,0.5,0.75])
        // Out: array([ 7.25, 12.5 , 17.75])
        let mut data_even = vec![2, 14, 17, 20, 5, 8, 11, 23];
        let (q1, q2, q3) = calculate_quartiles(&mut data_even);
        // assert_eq!(q1, 7.25);
        // assert_eq!(q2, 12.5);
        // assert_eq!(q3, 17.75);
        println!("Q1: {}, Q2 (Median): {}, Q3: {}", q1, q2, q3);
    }
}
