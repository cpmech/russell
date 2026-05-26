use super::quartiles;
use num_traits::{Num, NumCast};

/// Calculates the inter-quartile range (IQR) of a dataset.
///
/// The inter-quartile range is a measure of statistical dispersion, defined as the
/// difference between the third quartile (Q3) and the first quartile (Q1): IQR = Q3 - Q1.
/// It represents the middle 50% of the data and is robust to outliers.
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
/// * The function uses the "linear" method of NumPy's `quantile` function.
///
/// # Panics
///
/// This function will panic if the input vector is empty.
///
/// # Warnings
///
/// This function does not check for NaNs or Infinities in the input data.
///
/// # Examples
///
/// ```
/// use russell_stat::inter_quartile_range;
///
/// // Calculate IQR for a simple dataset
/// let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let iqr = inter_quartile_range(&mut data);
/// assert_eq!(iqr, 4.5);
///
/// // IQR is robust to outliers
/// let mut data_with_outlier = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
/// let iqr_outlier = inter_quartile_range(&mut data_with_outlier);
/// assert_eq!(iqr_outlier, 4.5); // Same IQR despite the outlier!
///
/// // IQR for a dataset with all identical values is 0
/// let mut uniform_data = vec![5, 5, 5, 5, 5];
/// let iqr_uniform = inter_quartile_range(&mut uniform_data);
/// assert_eq!(iqr_uniform, 0.0);
/// ```
pub fn inter_quartile_range<T>(data: &mut [T]) -> f64
where
    T: Num + NumCast + Copy + PartialOrd,
{
    let (q1, _, q3) = quartiles(data);
    q3 - q1
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::inter_quartile_range;

    #[test]
    fn calculate_inter_quartile_range_works() {
        // Python:
        // scipy.stats.iqr([10.0, 7.0, 4.0])
        assert_eq!(inter_quartile_range(&mut [10.0, 7.0, 4.0]), 3.0);

        // Python:
        // scipy.stats.iqr([7, 1, 3, 9, 5])
        assert_eq!(inter_quartile_range(&mut [7, 1, 3, 9, 5]), 4.0);

        // Python:
        // scipy.stats.iqr([2, 4, 18, 26, 3, 3, 7, 5, 5, 12])
        let data = &mut [2, 4, 18, 26, 3, 3, 7, 5, 5, 12];
        assert_eq!(inter_quartile_range(data), 7.5);
    }

    #[test]
    #[should_panic(expected = "Input data slice must not be empty")]
    fn inter_quartile_range_panics_on_empty_input() {
        let mut data: Vec<i32> = vec![];
        inter_quartile_range(&mut data);
    }

    #[test]
    fn inter_quartile_range_single_element() {
        // With only one element, IQR should be 0
        let mut data = vec![42];
        assert_eq!(inter_quartile_range(&mut data), 0.0);
    }

    #[test]
    fn inter_quartile_range_two_elements() {
        // Python: scipy.stats.iqr([10, 20])
        // Q1=12.5, Q3=17.5, IQR=5.0
        let mut data = vec![10, 20];
        assert_eq!(inter_quartile_range(&mut data), 5.0);
    }

    #[test]
    fn inter_quartile_range_three_elements() {
        // Python: scipy.stats.iqr([1, 2, 3])
        // Q1=1.5, Q3=2.5, IQR=1.0
        let mut data = vec![1, 2, 3];
        assert_eq!(inter_quartile_range(&mut data), 1.0);
    }

    #[test]
    fn inter_quartile_range_four_elements() {
        // Python: scipy.stats.iqr([1, 2, 3, 4])
        // Q1=1.75, Q3=3.25, IQR=1.5
        let mut data = vec![1, 2, 3, 4];
        assert_eq!(inter_quartile_range(&mut data), 1.5);
    }

    #[test]
    fn inter_quartile_range_with_floats() {
        // Python: scipy.stats.iqr([1.5, 2.7, 3.2, 4.8, 5.1])
        // Q1=2.7, Q3=4.8, IQR=2.1
        let mut data = vec![1.5, 2.7, 3.2, 4.8, 5.1];
        let iqr = inter_quartile_range(&mut data);
        assert!((iqr - 2.1).abs() < 1e-14);
    }

    #[test]
    fn inter_quartile_range_with_negative_values() {
        // Python: scipy.stats.iqr([-10, -5, 0, 5, 10])
        // Q1=-5.0, Q3=5.0, IQR=10.0
        let mut data = vec![-10, -5, 0, 5, 10];
        assert_eq!(inter_quartile_range(&mut data), 10.0);
    }

    #[test]
    fn inter_quartile_range_with_duplicates() {
        // Python: scipy.stats.iqr([1, 2, 2, 2, 3, 4, 5])
        // Q1=2.0, Q3=3.5, IQR=1.5
        let mut data = vec![1, 2, 2, 2, 3, 4, 5];
        assert_eq!(inter_quartile_range(&mut data), 1.5);
    }

    #[test]
    fn inter_quartile_range_all_same_values() {
        // When all values are the same, IQR should be 0
        let mut data = vec![5, 5, 5, 5, 5];
        assert_eq!(inter_quartile_range(&mut data), 0.0);
    }

    #[test]
    fn inter_quartile_range_unsorted_input() {
        // Test that the function correctly sorts unsorted data
        // Python: scipy.stats.iqr([100, 1, 50, 25, 75])
        // Q1=25.0, Q3=75.0, IQR=50.0
        let mut data = vec![100, 1, 50, 25, 75];
        assert_eq!(inter_quartile_range(&mut data), 50.0);
    }

    #[test]
    fn inter_quartile_range_large_dataset() {
        // Test with a larger dataset
        // Python: scipy.stats.iqr(list(range(1, 101)))
        // Q1=25.75, Q3=75.25, IQR=49.5
        let mut data: Vec<i32> = (1..=100).collect();
        assert_eq!(inter_quartile_range(&mut data), 49.5);
    }

    #[test]
    fn inter_quartile_range_odd_length() {
        // Python: scipy.stats.iqr([17, 2, 5, 11, 14, 8, 20])
        // Q1=6.5, Q3=15.5, IQR=9.0
        let mut data = vec![17, 2, 5, 11, 14, 8, 20];
        assert_eq!(inter_quartile_range(&mut data), 9.0);
    }

    #[test]
    fn inter_quartile_range_even_length() {
        // Python: scipy.stats.iqr([2, 14, 17, 20, 5, 8, 11, 23])
        // Q1=7.25, Q3=17.75, IQR=10.5
        let mut data = vec![2, 14, 17, 20, 5, 8, 11, 23];
        assert_eq!(inter_quartile_range(&mut data), 10.5);
    }

    #[test]
    fn inter_quartile_range_with_outliers() {
        // Dataset with clear outliers
        // Python: scipy.stats.iqr([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        // Q1=3.25, Q3=7.75, IQR=4.5
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
        assert_eq!(inter_quartile_range(&mut data), 4.5);
    }

    #[test]
    fn inter_quartile_range_verifies_sorting() {
        // Verify that the data is actually sorted after the function call
        let mut data = vec![5, 2, 8, 1, 9, 3, 7];
        let _ = inter_quartile_range(&mut data);
        assert_eq!(data, vec![1, 2, 3, 5, 7, 8, 9]);
    }
}
