use super::find_min_max;
use num_traits::Num;

/// Finds the local valleys and peaks of a sequence of numbers (return indices)
///
/// Returns `(Vec<usize>, Vec<usize>, index_min, index_max)` containing the indices of
/// the valleys and peaks, and the indices of the global minimum and maximum.
///
/// This function searches for the valleys and peaks in a sequence of numbers
/// and returns their indices. A peak is defined as a point that is greater than its
/// immediate neighbors, while a valley is a point that is less than its immediate
/// neighbors.
///
/// No peaks or valleys are returned if the sequence is strictly increasing or
/// strictly decreasing.
///
/// In the case of a plateau (a flat region), the last point of the plateau is
/// considered a peak if it is greater than the point before the plateau, and a
/// valley if it is less than the point before the plateau.
///
/// **Important:** If the slice is empty, this function returns `usize::MAX` to indicate
/// that there is no global minimum or maximum. In this case, it returns
/// `([], [], usize::MAX, usize::MAX)`.
///
/// # Examples
///
/// ```
/// use russell_lab::base::find_valleys_and_peaks;
///
/// // 500                         *
/// //                            / \
/// // 400                   *   /   *
/// //                      / \ /
/// // 300       *─*─*     /   *
/// //          /     \   /
/// // 200     /       *─*
/// //        /
/// // 100 *─*
///
/// let x = [100, 100, 300, 300, 300, 200, 200, 400, 300, 500, 400];
///
/// let (ii_valleys, ii_peaks, i_min, i_max) = find_valleys_and_peaks(&x);
/// assert_eq!(ii_valleys, [6, 8]);
/// assert_eq!(ii_peaks, [4, 7, 9]);
/// assert_eq!(i_min, 0);
/// assert_eq!(i_max, 9);
/// ```
///
pub fn find_valleys_and_peaks<T>(x: &[T]) -> (Vec<usize>, Vec<usize>, usize, usize)
where
    T: Num + PartialOrd,
{
    let n = x.len();
    match n {
        0 | 1 | 2 => {
            let (i_min, i_max) = find_min_max(x);
            (Vec::new(), Vec::new(), i_min, i_max)
        }
        3 => {
            let (i_min, i_max) = find_min_max(x);
            if x[0] > x[1] && x[1] < x[2] {
                // valley
                (vec![1], Vec::new(), i_min, i_max)
            } else if x[0] < x[1] && x[1] > x[2] {
                // peak
                (Vec::new(), vec![1], i_min, i_max)
            } else {
                // flat, increasing, or decreasing
                (Vec::new(), Vec::new(), i_min, i_max)
            }
        }
        _ => {
            let mut ii_valleys = Vec::new();
            let mut ii_peaks = Vec::new();
            let mut i_min = 0;
            let mut i_max = 0;
            let mut going_down = false;
            let mut going_up = false;
            for i in 1..n {
                if going_down {
                    if x[i] > x[i - 1] {
                        // change detected
                        going_down = false;
                        going_up = true;
                        ii_valleys.push(i - 1); // the previous value was a local min
                    }
                } else if going_up {
                    if x[i] < x[i - 1] {
                        // change detected
                        going_down = true;
                        going_up = false;
                        ii_peaks.push(i - 1); // the previous value was a local max
                    }
                } else {
                    // check if not on a plateau anymore
                    if x[i] < x[i - 1] {
                        going_down = true;
                    } else if x[i] > x[i - 1] {
                        going_up = true;
                    }
                }
                if x[i] < x[i_min] {
                    i_min = i;
                } else if x[i] > x[i_max] {
                    i_max = i;
                }
            }
            (ii_valleys, ii_peaks, i_min, i_max)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::find_valleys_and_peaks;

    #[test]
    fn find_valleys_and_peaks_works() {
        // 500                         *
        //                            / \
        // 400                   *   /   *
        //                      / \ /
        // 300       *─*─*     /   *
        //          /     \   /
        // 200     /       *─*
        //        /
        // 100 *─*
        let x = [100, 100, 300, 300, 300, 200, 200, 400, 300, 500, 400];

        let (ii_valleys, ii_peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(ii_valleys, [6, 8]);
        assert_eq!(ii_peaks, [4, 7, 9]);
        assert_eq!(i_min, 0);
        assert_eq!(i_max, 9);
    }

    #[test]
    fn find_valleys_and_peaks_empty_slice() {
        let empty: &[i32] = &[];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(empty);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, usize::MAX);
        assert_eq!(i_max, usize::MAX);
    }

    #[test]
    fn find_valleys_and_peaks_single_element() {
        let x = [42];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 0);
        assert_eq!(i_max, 0);
    }

    #[test]
    fn find_valleys_and_peaks_two_elements() {
        // Ascending
        let x = [1, 5];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 0);
        assert_eq!(i_max, 1);

        // Descending
        let x = [8, 3];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 1);
        assert_eq!(i_max, 0);

        // Equal
        let x = [7, 7];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 0);
        assert_eq!(i_max, 0);
    }

    #[test]
    fn find_valleys_and_peaks_three_elements() {
        // Valley pattern: high-low-high
        let x = [5, 2, 8];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![1]);
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 1);
        assert_eq!(i_max, 2);

        // Peak pattern: low-high-low
        let x = [3, 9, 1];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, vec![1]);
        assert_eq!(i_min, 2);
        assert_eq!(i_max, 1);

        // Strictly increasing
        let x = [1, 3, 5];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 0);
        assert_eq!(i_max, 2);

        // Strictly decreasing
        let x = [9, 5, 2];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 2);
        assert_eq!(i_max, 0);

        // All equal
        let x = [4, 4, 4];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 0);
        assert_eq!(i_max, 0);
    }

    #[test]
    fn find_valleys_and_peaks_strictly_increasing() {
        let x = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 0);
        assert_eq!(i_max, 8);
    }

    #[test]
    fn find_valleys_and_peaks_strictly_decreasing() {
        let x = [9, 8, 7, 6, 5, 4, 3, 2, 1];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 8);
        assert_eq!(i_max, 0);
    }

    #[test]
    fn find_valleys_and_peaks_all_equal() {
        let x = [5, 5, 5, 5, 5, 5];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 0);
        assert_eq!(i_max, 0);
    }

    #[test]
    fn find_valleys_and_peaks_simple_mountain() {
        // Single peak
        // 5     *
        // 4    / \
        // 3   /   *
        // 2  *     \
        // 1 *       *
        let x = [1, 2, 5, 3, 1];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, Vec::<usize>::new());
        assert_eq!(peaks, vec![2]);
        assert_eq!(i_min, 0); // first occurrence of min value 1
        assert_eq!(i_max, 2);
    }

    #[test]
    fn find_valleys_and_peaks_simple_valley() {
        // Single valley
        // 5 *
        // 4  \     *
        // 3   *   /
        // 2    \ *
        // 1     *
        let x = [5, 3, 1, 2, 4];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![2]);
        assert_eq!(peaks, Vec::<usize>::new());
        assert_eq!(i_min, 2);
        assert_eq!(i_max, 0);
    }

    #[test]
    fn find_valleys_and_peaks_alternating() {
        // Alternating pattern
        // 8              *
        // 7             / \
        // 6            /   \     *
        // 5     *     /     \   / \
        // 4    / \   /       \ /   \
        // 3   /   \ /         *     \
        // 2  /     *                 \
        // 1 *                         *
        let x = [1, 5, 2, 8, 3, 6, 1];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![2, 4]);
        assert_eq!(peaks, vec![1, 3, 5]);
        assert_eq!(i_min, 0); // first occurrence of min value 1
        assert_eq!(i_max, 3);
    }

    #[test]
    fn find_valleys_and_peaks_multiple_peaks_valleys() {
        // More complex pattern with multiple peaks and valleys
        // 5            *
        // 4           / \   *
        // 3    *     /   \ / \   *
        // 2 * / \   /     *   \ /
        // 1  *   \ /           *
        // 0       *
        let x = [2, 1, 3, 0, 5, 2, 4, 1, 3];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![1, 3, 5, 7]);
        assert_eq!(peaks, vec![2, 4, 6]);
        assert_eq!(i_min, 3); // global minimum 0 at index 3
        assert_eq!(i_max, 4); // global maximum 5 at index 4
    }

    #[test]
    fn find_valleys_and_peaks_plateau_peaks() {
        // Plateaus that become peaks
        // 3   *─*─*
        // 2  /     \ *─*─*
        // 1 *       *     \
        // 0                *
        let x = [1, 3, 3, 3, 1, 2, 2, 2, 0];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![4]);
        assert_eq!(peaks, vec![3, 7]);
        assert_eq!(i_min, 8); // minimum 0 at index 8
        assert_eq!(i_max, 1); // first occurrence of maximum 3
    }

    #[test]
    fn find_valleys_and_peaks_plateau_valleys() {
        // Plateaus that become valleys
        // 5 *
        // 4  \       *
        // 3   \     / \       *
        // 2    *─*─*   \     /
        // 1             *─*─*
        let x = [5, 2, 2, 2, 4, 1, 1, 1, 3];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![3, 7]);
        assert_eq!(peaks, vec![4]);
        assert_eq!(i_min, 5); // first occurrence of minimum 1
        assert_eq!(i_max, 0);
    }

    #[test]
    fn find_valleys_and_peaks_long_plateaus() {
        // Very long plateaus
        // 5     *─*─*─*─*─*
        // 4    /           \
        // 3   /             \   *─*─*─*
        // 2  /               \ /       \
        // 1 *                 *         \
        // 0                              *
        let x = [1, 5, 5, 5, 5, 5, 5, 1, 3, 3, 3, 3, 0];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![7]);
        assert_eq!(peaks, vec![6, 11]); // end of plateaus
        assert_eq!(i_min, 12); // minimum 0 at index 12
        assert_eq!(i_max, 1); // first occurrence of maximum 5
    }

    #[test]
    fn find_valleys_and_peaks_floating_point() {
        // 4.1          *
        // 3.5     *   / \
        // 2.9    / \ /   \   *
        // 2.2   /   *     \ /
        // 1.8  /           *
        // 1.0 *
        let x = [1.0, 3.5, 2.2, 4.1, 1.8, 2.9];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![2, 4]);
        assert_eq!(peaks, vec![1, 3]);
        assert_eq!(i_min, 0);
        assert_eq!(i_max, 3);
    }

    #[test]
    fn find_valleys_and_peaks_negative_numbers() {
        // -1 *                   *
        // -2  \     *           /
        // -3   \   / \         *
        // -4    \ /   \       /
        // -5     *     \     /
        // -6            \   /
        // -7             \ /
        // -8              *
        let x = [-1, -5, -2, -8, -3, -1];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![1, 3]);
        assert_eq!(peaks, vec![2]);
        assert_eq!(i_min, 3); // minimum -8 at index 3
        assert_eq!(i_max, 0); // first occurrence of maximum -1
    }

    #[test]
    fn find_valleys_and_peaks_mixed_sign() {
        //  3                    *
        //  2             *     /
        //  1    *       / \   /
        //  0   / \     /   \ /
        // -1  /   \   /     *
        // -2 *     \ /
        // -3        *
        let x = [-2, 1, -3, 2, -1, 3];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![2, 4]);
        assert_eq!(peaks, vec![1, 3]);
        assert_eq!(i_min, 2); // minimum -3 at index 2
        assert_eq!(i_max, 5); // maximum 3 at index 5
    }

    #[test]
    fn find_valleys_and_peaks_zeros() {
        // 3               *
        //                / \
        // 2     *       /   \
        //      / \     /     \
        // 1   /   \   /       \   *
        //    /     \ /         \ / \
        // 0 *       *           *   *
        let x = [0, 2, 0, 3, 0, 1, 0];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![2, 4]);
        assert_eq!(peaks, vec![1, 3, 5]);
        assert_eq!(i_min, 0); // first occurrence of minimum 0
        assert_eq!(i_max, 3); // maximum 3 at index 3
    }

    #[test]
    fn find_valleys_and_peaks_u8_type() {
        // 80           *
        // 60          / \   *
        // 50     *   /   \ / \
        // 30    / \ /     *   \
        // 20   /   *           \
        // 15  /                 *
        // 10 *
        let x: Vec<u8> = vec![10, 50, 20, 80, 30, 60, 15];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![2, 4]);
        assert_eq!(peaks, vec![1, 3, 5]);
        assert_eq!(i_min, 0); // minimum 10 at index 0
        assert_eq!(i_max, 3); // maximum 80 at index 3
    }

    #[test]
    fn find_valleys_and_peaks_large_numbers() {
        // 3000        *
        // 2500       / \     *
        // 2000    * /   \   /
        // 1500   / *     \ /
        // 1200  /         *
        // 1000 *
        let x = [1000, 2000, 1500, 3000, 1200, 2500];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![2, 4]);
        assert_eq!(peaks, vec![1, 3]);
        assert_eq!(i_min, 0); // minimum 1000 at index 0
        assert_eq!(i_max, 3); // maximum 3000 at index 3
    }

    #[test]
    fn find_valleys_and_peaks_very_small_differences() {
        // Test with very small floating point differences
        // 1.0003         *
        //               / \
        // 1.0002   *   /   \
        //         / \ /     \
        // 1.0001 *   *       *
        let x = [1.0001, 1.0002, 1.0001, 1.0003, 1.0001];
        let (valleys, peaks, i_min, i_max) = find_valleys_and_peaks(&x);
        assert_eq!(valleys, vec![2]);
        assert_eq!(peaks, vec![1, 3]);
        assert_eq!(i_min, 0); // first occurrence of minimum 1.0001
        assert_eq!(i_max, 3); // maximum 1.0003 at index 3
    }
}
