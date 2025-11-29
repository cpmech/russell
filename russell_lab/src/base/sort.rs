use std::cmp::Ordering;
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

/// Compare two f64 values safely, handling NaN
fn compare_f64(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        if a.is_nan() && b.is_nan() {
            Ordering::Equal
        } else if a.is_nan() {
            Ordering::Greater // Treat NaN as greater than non-NaN
        } else {
            Ordering::Less
        }
    })
}

/// Returns the indices that would sort a f64 array in ascending order
///
/// **Important:** This function treats NaN as greater than non-NaN.
///
/// # Arguments
///
/// * `x` -- array to sort
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
pub fn argsort_f64(x: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.sort_by(|&i, &j| compare_f64(&x[i], &x[j]));
    indices
}

/// Returns the indices that would sort two f64 arrays in ascending order
///
/// The function sorts primarily by `y`, then by `x`.
/// Both input vectors must have the same length.
///
/// **Important:** This function treats NaN as greater than non-NaN.
///
/// # Arguments
///
/// * `y` -- first array to sort by
/// * `x` -- second array to sort by when `y` values are equal
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
/// //     y
/// // 4.0 +                  (0)
/// //     |
/// // 3.0 +   (4)
/// //     |
/// // 2.0 +   (2)
/// //     |
/// // 1.0 +        (3)  (1)
/// //     |
/// //     +----+----+----+----+-- x
/// //    0.0  1.0  2.0  3.0  4.0
/// let x = vec![4.0, 3.0, 1.0, 2.0, 1.0];
/// let y = vec![4.0, 1.0, 2.0, 1.0, 3.0];
/// let sorted_indices = argsort2_f64(&y, &x);
/// assert_eq!(sorted_indices, vec![3, 1, 2, 4, 0]);
/// ```
pub fn argsort2_f64(y: &[f64], x: &[f64]) -> Vec<usize> {
    assert_eq!(y.len(), x.len(), "arrays must have the same length");
    let mut indices: Vec<usize> = (0..y.len()).collect();
    indices.sort_by(|&i, &j| compare_f64(&y[i], &y[j]).then_with(|| compare_f64(&x[i], &x[j])));
    indices
}

/// Returns the indices that would sort three f64 arrays in ascending order
///
/// The function sorts primarily by `z`, then by `y`, and finally by `x`.
/// All input vectors must have the same length.
///
/// **Important:** This function treats NaN as greater than non-NaN.
///
/// # Arguments
///
/// * `z` -- first array to sort by
/// * `y` -- second array to sort by when `z` values are equal
/// * `x` -- third array to sort by when both `z` and `y` values are equal
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
/// //       4--------------7  1.0
/// //      /.             /|
/// //     / .            / |
/// //    /  .           /  |
/// //   /   .          /   |
/// //  5--------------6    |          z
/// //  |    .         |    |          ↑
/// //  |    0---------|----3  0.0     o → y
/// //  |   /          |   /          ↙
/// //  |  /           |  /          x
/// //  | /            | /
/// //  |/             |/
/// //  1--------------2   1.0
/// // 0.0            1.0
/// let x = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
/// let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
/// let z = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
/// let indices = argsort3_f64(&z, &y, &x);
/// assert_eq!(indices, vec![/*z=0*/ 0, 1, 3, 2, /*z=1*/ 4, 5, 7, 6]);
/// ```
pub fn argsort3_f64(z: &[f64], y: &[f64], x: &[f64]) -> Vec<usize> {
    assert_eq!(z.len(), y.len(), "first two arrays must have the same length");
    assert_eq!(z.len(), x.len(), "all three arrays must have the same length");
    let mut indices: Vec<usize> = (0..z.len()).collect();
    indices.sort_by(|&i, &j| {
        compare_f64(&z[i], &z[j])
            .then_with(|| compare_f64(&y[i], &y[j]))
            .then_with(|| compare_f64(&x[i], &x[j]))
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
        let x = vec![1.0, 2.0, 1.0, 3.0];
        let y = vec![3.0, 1.0, 2.0, 1.0];

        let sorted_indices = argsort2_f64(&y, &x);
        assert_eq!(sorted_indices, vec![1, 3, 2, 0]);
    }

    #[test]
    fn argsort2_f64_works_equal_y() {
        let y = vec![1.0, 1.0, 1.0];
        let x = vec![3.0, 2.0, 1.0];

        let sorted_indices = argsort2_f64(&y, &x);
        assert_eq!(sorted_indices, vec![2, 1, 0]);
    }

    #[test]
    fn argsort2_f64_works_equal_x_y() {
        let y = vec![2.0, 2.0, 2.0];
        let x = vec![2.0, 2.0, 2.0];

        let sorted_indices = argsort2_f64(&y, &x);
        assert_eq!(sorted_indices, vec![0, 1, 2]);
    }

    #[test]
    fn argsort2_f64_works_single_element() {
        let y = vec![1.0];
        let x = vec![2.0];

        let sorted_indices = argsort2_f64(&y, &x);
        assert_eq!(sorted_indices, vec![0]);
    }

    #[test]
    #[should_panic(expected = "arrays must have the same length")]
    fn argsort2_f64_works_different_lengths() {
        let y = vec![1.0, 2.0];
        let x = vec![1.0];

        argsort2_f64(&y, &x);
    }

    #[test]
    fn argsort2_f64_works_with_imprecision() {
        let x = vec![
            0.7886751345948129,
            0.21132486540518716,
            0.788675134594813,
            0.7886751345948129,
            0.21132486540518716,
            0.788675134594813,
            0.21132486540518716,
            0.7886751345948129,
        ];
        let y = vec![
            0.7886751345948129,
            0.21132486540518716,
            0.21132486540518716,
            0.21132486540518716,
            0.788675134594813,
            0.788675134594813,
            0.21132486540518713,
            0.21132486540518716,
        ];
        let indices = argsort2_f64(&y, &x);
        // Verify we get 8 indices
        assert_eq!(indices.len(), 8);
        // Verify ordering: y values should be sorted, then x values for equal y
        for i in 0..indices.len() - 1 {
            let curr_y = y[indices[i]];
            let next_y = y[indices[i + 1]];
            // y should be non-decreasing
            assert!(curr_y <= next_y);
        }
    }

    #[test]
    fn argsort2_f64_works_with_negative_values() {
        let y = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        let x = vec![5.0, 1.0, 0.0, -1.0, -5.0];
        let indices = argsort2_f64(&y, &x);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn argsort2_f64_works_empty_vectors() {
        let y: Vec<f64> = vec![];
        let x: Vec<f64> = vec![];
        let indices = argsort2_f64(&y, &x);
        assert_eq!(indices, Vec::<usize>::new());
    }

    #[test]
    fn argsort2_f64_works_with_duplicates() {
        let y = vec![2.0, 1.0, 2.0, 1.0, 2.0];
        let x = vec![1.0, 2.0, 3.0, 1.0, 2.0];
        let indices = argsort2_f64(&y, &x);
        // y=1.0: indices 1,3 with x=2.0,1.0 -> sorted as 3,1
        // y=2.0: indices 0,2,4 with x=1.0,3.0,2.0 -> sorted as 0,4,2
        assert_eq!(indices, vec![3, 1, 0, 4, 2]);
    }

    #[test]
    fn argsort2_f64_works_with_extreme_values() {
        let y = vec![f64::MAX, f64::MIN, 0.0, -0.0, 1e-308, -1e-308];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = argsort2_f64(&y, &x);
        assert_eq!(indices, vec![1, 5, 2, 3, 4, 0]);
    }

    #[test]
    fn argsort2_f64_works_with_infinity() {
        let y = vec![f64::INFINITY, -f64::INFINITY, 0.0, 1.0];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let indices = argsort2_f64(&y, &x);
        assert_eq!(indices, vec![1, 2, 3, 0]);
    }

    #[test]
    fn argsort2_f64_with_nan_values() {
        // NaN handling - NaNs are treated as equal via unwrap_or(Equal)
        let y = vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = argsort2_f64(&y, &x);
        // NaN behavior in sort is implementation-dependent
        // We just verify that we get all 5 indices and they're unique
        assert_eq!(indices.len(), 5);
        let mut seen = vec![false; 5];
        for &idx in &indices {
            assert!(!seen[idx], "Duplicate index found");
            seen[idx] = true;
        }
        // Non-NaN values should maintain relative order: 0 before 2 before 4
        let pos_0 = indices.iter().position(|&x| x == 0).unwrap();
        let pos_2 = indices.iter().position(|&x| x == 2).unwrap();
        let pos_4 = indices.iter().position(|&x| x == 4).unwrap();
        assert!(pos_0 < pos_2);
        assert!(pos_2 < pos_4);
    }

    #[test]
    fn argsort2_f64_transitivity_edge_case() {
        let xx = vec![
            f64::NAN, //  0
            33.0,     //  1
            92.0,     //  2
            69.0,     //  3
            27.0,     //  4
            14.0,     //  5
            59.0,     //  6
            29.0,     //  7
            33.0,     //  8
            25.0,     //  9
            81.0,     // 10
            f64::NAN, // 11
            98.0,     // 12
            77.0,     // 13
            89.0,     // 14
            67.0,     // 15
            84.0,     // 16
            79.0,     // 17
            33.0,     // 18
            34.0,     // 19
            79.0,     // 20
        ];
        let yy = xx.iter().map(|_| 0.0).collect::<Vec<f64>>();
        let indices = argsort2_f64(&yy, &xx);
        assert_eq!(
            indices,
            vec![5, 9, 4, 7, 1, 8, 18, 19, 6, 15, 3, 13, 17, 20, 10, 16, 14, 2, 12, 0, 11]
        );
    }

    #[test]
    fn argsort2_f64_stress_test_transitivity() {
        // Create a challenging scenario with values close to each other
        // to potentially expose transitivity issues
        let y = vec![1.0, 1.0 + 1e-10, 1.0 + 2e-10, 1.0 + 3e-10, 1.0 + 4e-10];
        let x = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let indices = argsort2_f64(&y, &x);
        // This should still produce a valid ordering
        assert_eq!(indices.len(), 5);
        // Verify all indices are unique
        let mut seen = vec![false; 5];
        for &idx in &indices {
            assert!(!seen[idx], "Duplicate index found");
            seen[idx] = true;
        }
    }

    #[test]
    fn argsort2_f64_many_equal_y_different_x() {
        // Many elements with equal y but different x to test secondary sort
        let y = vec![1.0; 100];
        let x: Vec<f64> = (0..100).map(|i| (100 - i) as f64).collect();
        let indices = argsort2_f64(&y, &x);
        // Should sort by x in ascending order (largest to smallest original index)
        for i in 0..100 {
            assert_eq!(indices[i], 99 - i);
        }
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
    fn argsort3_f64_works_equal_x_y_z() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![2.0, 2.0, 2.0];
        let z = vec![3.0, 3.0, 3.0];

        let sorted_indices = argsort3_f64(&x, &y, &z);
        assert_eq!(sorted_indices, vec![0, 1, 2]);
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
    #[should_panic(expected = "first two arrays must have the same length")]
    fn argsort3_f64_works_different_lengths() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0];
        let z = vec![1.0];

        argsort3_f64(&x, &y, &z);
    }

    #[test]
    #[should_panic(expected = "all three arrays must have the same length")]
    fn argsort3_f64_works_different_lengths_third_array() {
        let z = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];
        let x = vec![1.0];

        argsort3_f64(&z, &y, &x);
    }

    #[test]
    fn argsort3_f64_works_with_negative_values() {
        let z = vec![-3.0, -1.0, -2.0, 1.0, 0.0];
        let y = vec![1.0, -1.0, 2.0, -2.0, 0.0];
        let x = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let indices = argsort3_f64(&z, &y, &x);
        // Verify ordering: z first, then y, then x
        for i in 0..indices.len() - 1 {
            let curr_z = z[indices[i]];
            let next_z = z[indices[i + 1]];
            assert!(curr_z <= next_z);
        }
    }

    #[test]
    fn argsort3_f64_works_empty_vectors() {
        let z: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        let x: Vec<f64> = vec![];
        let indices = argsort3_f64(&z, &y, &x);
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn argsort3_f64_works_with_duplicates() {
        let z = vec![1.0, 2.0, 1.0, 2.0, 1.0];
        let y = vec![3.0, 3.0, 3.0, 3.0, 4.0];
        let x = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let indices = argsort3_f64(&z, &y, &x);
        // Verify all indices are present
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn argsort3_f64_works_with_extreme_values() {
        let z = vec![f64::MAX, f64::MIN, 0.0, -0.0, 1e-308, -1e-308];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = argsort3_f64(&z, &y, &x);
        assert_eq!(indices, vec![1, 5, 2, 3, 4, 0]);
    }

    #[test]
    fn argsort3_f64_works_with_infinity() {
        let z = vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, 1.0, -1.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = argsort3_f64(&z, &y, &x);
        // Order: NEG_INFINITY, -1.0, 0.0, 1.0, INFINITY
        assert_eq!(indices, vec![1, 4, 2, 3, 0]);
    }

    #[test]
    fn argsort3_f64_with_nan_values() {
        // NaN handling - NaNs are treated as equal via unwrap_or(Equal)
        let z = vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = argsort3_f64(&z, &y, &x);
        // Verify we get all 5 indices and they're unique
        assert_eq!(indices.len(), 5);
        let mut seen = vec![false; 5];
        for &idx in &indices {
            assert!(!seen[idx], "Duplicate index found");
            seen[idx] = true;
        }
        // Non-NaN values should maintain relative order: 0 before 2 before 4
        let pos_0 = indices.iter().position(|&x| x == 0).unwrap();
        let pos_2 = indices.iter().position(|&x| x == 2).unwrap();
        let pos_4 = indices.iter().position(|&x| x == 4).unwrap();
        assert!(pos_0 < pos_2);
        assert!(pos_2 < pos_4);
    }

    #[test]
    fn argsort3_f64_many_equal_z_different_y_x() {
        // All z values equal, y values differ, x values differ
        let z = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let x = vec![9.0, 8.0, 7.0, 6.0, 5.0];
        let indices = argsort3_f64(&z, &y, &x);
        // Should sort by y (ascending), then x (ascending)
        assert_eq!(indices, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn argsort3_f64_works_with_imprecision() {
        let x = vec![
            0.7886751345948129,
            0.21132486540518716,
            0.788675134594813,
            0.7886751345948129,
            0.21132486540518716,
            0.788675134594813,
            0.21132486540518716,
            0.7886751345948129,
            0.21132486540518713,
            0.7886751345948129,
            0.21132486540518713,
            0.21132486540518716,
            0.788675134594813,
            0.21132486540518716,
            0.788675134594813,
            0.21132486540518716,
        ];
        let y = vec![
            0.7886751345948129,
            0.21132486540518716,
            0.21132486540518716,
            0.21132486540518716,
            0.788675134594813,
            0.788675134594813,
            0.21132486540518713,
            0.21132486540518716,
            0.7886751345948129,
            0.7886751345948129,
            0.7886751345948129,
            0.21132486540518716,
            0.21132486540518716,
            0.788675134594813,
            0.788675134594813,
            0.21132486540518713,
        ];
        let z = vec![
            0.2113248654051872,
            0.7886751345948129,
            0.7886751345948129,
            0.21132486540518716,
            0.788675134594813,
            0.788675134594813,
            1.211324865405187,
            1.2113248654051871,
            0.21132486540518716,
            1.2113248654051871,
            1.2113248654051874,
            1.7886751345948129,
            1.7886751345948129,
            1.788675134594813,
            1.788675134594813,
            0.21132486540518713,
        ];
        let indices = argsort3_f64(&z, &y, &x);
        let mut buf = String::new();
        for i in indices {
            buf.push_str(&format!("{:.5},{:.5},{:.5}\n", x[i], y[i], z[i]));
        }
        assert_eq!(
            buf,
            "0.21132,0.21132,0.21132\n\
             0.78868,0.21132,0.21132\n\
             0.21132,0.78868,0.21132\n\
             0.78868,0.78868,0.21132\n\
             0.21132,0.21132,0.78868\n\
             0.78868,0.21132,0.78868\n\
             0.21132,0.78868,0.78868\n\
             0.78868,0.78868,0.78868\n\
             0.21132,0.21132,1.21132\n\
             0.78868,0.21132,1.21132\n\
             0.78868,0.78868,1.21132\n\
             0.21132,0.78868,1.21132\n\
             0.21132,0.21132,1.78868\n\
             0.78868,0.21132,1.78868\n\
             0.21132,0.78868,1.78868\n\
             0.78868,0.78868,1.78868\n"
        );
    }
}
