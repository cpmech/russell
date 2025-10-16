use num_traits::Num;

/// Finds the indices of the minimum and maximum values in a slice of generic numbers
///
/// Returns `(index_min, index_max)` where:
/// - `index_min` is the index of the first occurrence of the minimum value
/// - `index_max` is the index of the first occurrence of the maximum value
///
/// # Arguments
///
/// * `x` - A slice of numeric values that implement `Num + PartialOrd`
///
/// # Returns
///
/// A tuple `(usize, usize)` containing the indices of the minimum and maximum values.
///
/// # Special Cases
///
/// - **Empty slice**: Returns `(usize::MAX, usize::MAX)` as invalid indices
/// - **Single element**: Returns `(0, 0)` since the only element is both min and max
/// - **Multiple occurrences**: Returns the index of the **first** occurrence of min/max values
/// - **Equal elements**: If all elements are equal, returns `(0, 0)`
///
/// # Examples
///
/// ```
/// use russell_lab::find_min_max;
///
/// let values = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
/// let (min_idx, max_idx) = find_min_max(&values);
///
/// assert_eq!(min_idx, 1);  // First occurrence of minimum value 1
/// assert_eq!(max_idx, 5);  // First occurrence of maximum value 9
/// assert_eq!(values[min_idx], 1);
/// assert_eq!(values[max_idx], 9);
/// ```
pub fn find_min_max<T>(x: &[T]) -> (usize, usize)
where
    T: Num + PartialOrd,
{
    let n = x.len();
    match n {
        0 => (usize::MAX, usize::MAX),
        1 => (0, 0),
        _ => {
            let mut index_min = 0;
            let mut index_max = 0;
            for i in 1..n {
                if x[i] < x[index_min] {
                    index_min = i;
                } else if x[i] > x[index_max] {
                    index_max = i;
                }
            }
            (index_min, index_max)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::find_min_max;

    #[test]
    fn find_min_max_works() {
        //                                min         max
        //                              0  1  2  3  4  5  6  7  8  9 10
        let (min, max) = find_min_max(&[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]);
        assert_eq!(min, 1);
        assert_eq!(max, 5);
    }

    #[test]
    fn find_min_max_empty_slice() {
        let empty: &[i32] = &[];
        let (min, max) = find_min_max(empty);
        assert_eq!(min, usize::MAX);
        assert_eq!(max, usize::MAX);
    }

    #[test]
    fn find_min_max_single_element() {
        let (min, max) = find_min_max(&[42]);
        assert_eq!(min, 0);
        assert_eq!(max, 0);
    }

    #[test]
    fn find_min_max_two_elements_ascending() {
        let (min, max) = find_min_max(&[1, 5]);
        assert_eq!(min, 0);
        assert_eq!(max, 1);
    }

    #[test]
    fn find_min_max_two_elements_descending() {
        let (min, max) = find_min_max(&[8, 3]);
        assert_eq!(min, 1);
        assert_eq!(max, 0);
    }

    #[test]
    fn find_min_max_two_equal_elements() {
        let (min, max) = find_min_max(&[7, 7]);
        assert_eq!(min, 0);
        assert_eq!(max, 0);
    }

    #[test]
    fn find_min_max_all_equal_elements() {
        let (min, max) = find_min_max(&[5, 5, 5, 5, 5]);
        assert_eq!(min, 0);
        assert_eq!(max, 0);
    }

    #[test]
    fn find_min_max_negative_numbers() {
        let (min, max) = find_min_max(&[-3.0, -1.0, -4.0, -1.0, -5.0]);
        assert_eq!(min, 4); // -5 at index 4
        assert_eq!(max, 1); // -1 at index 1 (first occurrence)
    }

    #[test]
    fn find_min_max_mixed_positive_negative() {
        let (min, max) = find_min_max(&[-2, 5, -8, 10, 1]);
        assert_eq!(min, 2); // -8 at index 2
        assert_eq!(max, 3); // 10 at index 3
    }

    #[test]
    fn find_min_max_zeros() {
        let (min, max) = find_min_max(&[0, 0, 0]);
        assert_eq!(min, 0);
        assert_eq!(max, 0);
    }

    #[test]
    fn find_min_max_with_zeros_and_others() {
        let (min, max) = find_min_max(&[3, 0, -1, 2, 0]);
        assert_eq!(min, 2); // -1 at index 2
        assert_eq!(max, 0); // 3 at index 0

        let (min, max) = find_min_max(&[f64::EPSILON, f64::EPSILON, f64::EPSILON]);
        assert_eq!(min, 0);
        assert_eq!(max, 0);

        let (min, max) = find_min_max(&[f64::EPSILON, 0.0, f64::EPSILON]);
        assert_eq!(min, 1); // 0 > EPSILON
        assert_eq!(max, 0);
    }

    #[test]
    fn find_min_max_first_occurrence_priority() {
        // When there are multiple occurrences of min/max, first should be returned
        let (min, max) = find_min_max(&[2, 1, 3, 1, 3, 2]);
        assert_eq!(min, 1); // first occurrence of minimum value 1
        assert_eq!(max, 2); // first occurrence of maximum value 3
    }

    #[test]
    fn find_min_max_min_at_end() {
        let (min, max) = find_min_max(&[5, 4, 3, 2, 1]);
        assert_eq!(min, 4); // minimum 1 at last index
        assert_eq!(max, 0); // maximum 5 at first index
    }

    #[test]
    fn find_min_max_max_at_end() {
        let (min, max) = find_min_max(&[1, 2, 3, 4, 5]);
        assert_eq!(min, 0); // minimum 1 at first index
        assert_eq!(max, 4); // maximum 5 at last index
    }

    #[test]
    fn find_min_max_alternating() {
        let (min, max) = find_min_max(&[1, 10, 2, 9, 3, 8]);
        assert_eq!(min, 0); // minimum 1 at index 0
        assert_eq!(max, 1); // maximum 10 at index 1
    }

    #[test]
    fn find_min_max_floating_point() {
        let (min, max) = find_min_max(&[3.14, 2.71, 1.41, 2.71, 3.14]);
        assert_eq!(min, 2); // minimum 1.41 at index 2
        assert_eq!(max, 0); // maximum 3.14 at index 0 (first occurrence)
    }

    #[test]
    fn find_min_max_very_small_floats() {
        let (min, max) = find_min_max(&[1e-10, 1e-15, 1e-5, 1e-20]);
        assert_eq!(min, 3); // minimum 1e-20 at index 3
        assert_eq!(max, 2); // maximum 1e-5 at index 2
    }

    #[test]
    fn find_min_max_large_numbers() {
        let values = [1_000_000, 999_999, 1_000_001, 999_998];
        let (min, max) = find_min_max(&values);
        assert_eq!(min, 3); // 999_998 at index 3
        assert_eq!(max, 2); // 1_000_001 at index 2
    }

    #[test]
    fn find_min_max_u8_type() {
        let (min, max) = find_min_max(&[100u8, 50u8, 200u8, 25u8]);
        assert_eq!(min, 3); // 25 at index 3
        assert_eq!(max, 2); // 200 at index 2
    }

    #[test]
    fn find_min_max_i64_type() {
        let values: &[i64] = &[i64::MAX, i64::MIN, 0, -1, 1];
        let (min, max) = find_min_max(values);
        assert_eq!(min, 1); // i64::MIN at index 1
        assert_eq!(max, 0); // i64::MAX at index 0
    }

    #[test]
    fn find_min_max_same_min_max_at_ends() {
        // Case where min and max are at the first and last positions
        let (min, max) = find_min_max(&[1, 5, 3, 7, 2, 10]);
        assert_eq!(min, 0); // minimum 1 at index 0
        assert_eq!(max, 5); // maximum 10 at index 5
    }

    #[test]
    fn find_min_max_plateau_pattern() {
        // Test with plateau patterns
        let (min, max) = find_min_max(&[1, 1, 1, 5, 5, 5, 2, 2, 2]);
        assert_eq!(min, 0); // first occurrence of minimum 1
        assert_eq!(max, 3); // first occurrence of maximum 5
    }

    #[test]
    fn find_min_max_mountain_pattern() {
        // Test with mountain-like pattern
        let (min, max) = find_min_max(&[1, 2, 3, 4, 5, 4, 3, 2, 1]);
        assert_eq!(min, 0); // first occurrence of minimum 1
        assert_eq!(max, 4); // maximum 5 at index 4
    }

    #[test]
    fn find_min_max_valley_pattern() {
        // Test with valley-like pattern
        let (min, max) = find_min_max(&[5, 4, 3, 2, 1, 2, 3, 4, 5]);
        assert_eq!(min, 4); // minimum 1 at index 4
        assert_eq!(max, 0); // first occurrence of maximum 5
    }

    #[test]
    fn find_min_max_repeated_extremes() {
        // Multiple occurrences of both min and max values
        let (min, max) = find_min_max(&[1, 5, 1, 3, 5, 1, 5]);
        assert_eq!(min, 0); // first occurrence of min value 1
        assert_eq!(max, 1); // first occurrence of max value 5
    }
}
