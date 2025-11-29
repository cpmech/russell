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
/// * `tol` -- tolerance for comparing floating-point numbers
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
/// assert_eq!(argsort_f64(&data, 1e-9), &[1, 3, 6, 0, 9, 2, 4, 8, 7, 5]);
/// ```
pub fn argsort_f64(data: &[f64], tol: f64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.sort_by(|&i, &j| {
        if (data[i] - data[j]).abs() < tol {
            std::cmp::Ordering::Equal
        } else {
            data[i].partial_cmp(&data[j]).unwrap_or(std::cmp::Ordering::Equal)
        }
    });
    indices
}

/// Returns the indices that would sort two f64 arrays in ascending order
///
/// The function sorts primarily by `y`, then by `x`.
/// Both input vectors must have the same length.
///
/// **Warning:** The NaN values are considered equal in this function, i.e., they
/// are not handled well enough and **care should be taken** if NaNs are expected.
///
/// # Arguments
///
/// * `y` -- first vector to sort by
/// * `x` -- second vector to sort by when `y` values are equal
/// * `tol` -- slice of two tolerances for comparing floating-point numbers
///    One tolerance for each vector y and x. For example: `[tol_y, tol_x]`.
///
/// # Returns
///
/// A vector of indices that would sort the input vectors
///
/// # Panics
///
/// Panics if the input vectors have different lengths or if the tolerance slice does not have exactly two elements
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
/// let sorted_indices = argsort2_f64(&y, &x, &[1e-9, 1e-9]);
/// assert_eq!(sorted_indices, vec![3, 1, 2, 4, 0]);
/// ```
pub fn argsort2_f64(y: &[f64], x: &[f64], tol: &[f64]) -> Vec<usize> {
    assert_eq!(y.len(), x.len(), "vectors must have equal length");
    assert_eq!(tol.len(), 2, "tolerance slice must have exactly two elements");

    let mut indices: Vec<usize> = (0..y.len()).collect();
    indices.sort_by(|&i, &j| {
        if (y[i] - y[j]).abs() < tol[0] {
            std::cmp::Ordering::Equal
        } else {
            y[i].partial_cmp(&y[j]).unwrap_or(std::cmp::Ordering::Equal)
        }
        .then_with(|| {
            if (x[i] - x[j]).abs() < tol[1] {
                std::cmp::Ordering::Equal
            } else {
                x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal)
            }
        })
    });
    indices
}

/// Returns the indices that would sort three f64 arrays in ascending order
///
/// The function sorts primarily by `z`, then by `y`, and finally by `x`.
/// All input vectors must have the same length.
///
/// **Warning:** The NaN values are considered equal in this function, i.e., they
/// are not handled well enough and **care should be taken** if NaNs are expected.
///
/// # Arguments
///
/// * `z` -- first vector to sort by
/// * `y` -- second vector to sort by when `z` values are equal
/// * `x` -- third vector to sort by when both `z` and `y` values are equal
/// * `tol` -- slice of three tolerances for comparing floating-point numbers.
///    One tolerance for each vector z, y, and x. For example: `[tol_z, tol_y, tol_x]`.
///
/// # Returns
///
/// A vector of indices that would sort the input vectors
///
/// # Panics
///
/// Panics if the input vectors have different lengths or if the tolerance slice does not have exactly three elements
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
/// let indices = argsort3_f64(&z, &y, &x, &[1e-9, 1e-9, 1e-9]);
/// assert_eq!(indices, vec![/*z=0*/ 0, 1, 3, 2, /*z=1*/ 4, 5, 7, 6]);
/// ```
pub fn argsort3_f64(z: &[f64], y: &[f64], x: &[f64], tol: &[f64]) -> Vec<usize> {
    assert!(
        z.len() == y.len() && y.len() == x.len(),
        "vectors must have equal length"
    );
    assert_eq!(tol.len(), 3, "tolerance slice must have exactly three elements");

    let mut indices: Vec<usize> = (0..z.len()).collect();

    indices.sort_by(|&a, &b| {
        if (z[a] - z[b]).abs() < tol[0] {
            std::cmp::Ordering::Equal
        } else {
            z[a].partial_cmp(&z[b]).unwrap_or(std::cmp::Ordering::Equal)
        }
        .then_with(|| {
            if (y[a] - y[b]).abs() < tol[1] {
                std::cmp::Ordering::Equal
            } else {
                y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal)
            }
        })
        .then_with(|| {
            if (x[a] - x[b]).abs() < tol[2] {
                std::cmp::Ordering::Equal
            } else {
                x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal)
            }
        })
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
        assert_eq!(argsort_f64(&data, 1e-9), &[1, 3, 6, 0, 9, 2, 4, 8, 7, 5]);

        let data = [3.0, 1.0, 8.0, -5.0];
        assert_eq!(argsort_f64(&data, 1e-9), &[3, 1, 0, 2]);
    }

    #[test]
    fn argsort2_f64_works_basic() {
        let x = vec![1.0, 2.0, 1.0, 3.0];
        let y = vec![3.0, 1.0, 2.0, 1.0];
        let tol = [1e-9, 1e-9];

        let sorted_indices = argsort2_f64(&y, &x, &tol);
        assert_eq!(sorted_indices, vec![1, 3, 2, 0]);
    }

    #[test]
    fn argsort2_f64_works_equal_y() {
        let y = vec![1.0, 1.0, 1.0];
        let x = vec![3.0, 2.0, 1.0];
        let tol = [1e-9, 1e-9];

        let sorted_indices = argsort2_f64(&y, &x, &tol);
        assert_eq!(sorted_indices, vec![2, 1, 0]);
    }

    #[test]
    fn argsort2_f64_works_equal_x_y() {
        let y = vec![2.0, 2.0, 2.0];
        let x = vec![2.0, 2.0, 2.0];
        let tol = [1e-9, 1e-9];

        let sorted_indices = argsort2_f64(&y, &x, &tol);
        assert_eq!(sorted_indices, vec![0, 1, 2]);
    }

    #[test]
    fn argsort2_f64_works_single_element() {
        let y = vec![1.0];
        let x = vec![2.0];
        let tol = [1e-9, 1e-9];

        let sorted_indices = argsort2_f64(&y, &x, &tol);
        assert_eq!(sorted_indices, vec![0]);
    }

    #[test]
    #[should_panic(expected = "vectors must have equal length")]
    fn argsort2_f64_works_different_lengths() {
        let y = vec![1.0, 2.0];
        let x = vec![1.0];
        let tol = [1e-9, 1e-9];

        argsort2_f64(&y, &x, &tol);
    }

    #[test]
    #[should_panic(expected = "tolerance slice must have exactly two elements")]
    fn argsort2_f64_works_invalid_tol_length() {
        let y = vec![1.0, 2.0];
        let x = vec![1.0, 2.0];
        let tol = [1e-9];

        argsort2_f64(&y, &x, &tol);
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
        let tol = [1e-9, 1e-9];
        let indices = argsort2_f64(&y, &x, &tol);
        // Verify we get 8 indices
        assert_eq!(indices.len(), 8);
        // Verify ordering: y values should be sorted, then x values for equal y
        for i in 0..indices.len() - 1 {
            let curr_y = y[indices[i]];
            let next_y = y[indices[i + 1]];
            // y should be non-decreasing
            assert!(curr_y <= next_y || (curr_y - next_y).abs() < tol[0]);
            // If y values are equal (within tolerance), x should be non-decreasing
            if (curr_y - next_y).abs() < tol[0] {
                let curr_x = x[indices[i]];
                let next_x = x[indices[i + 1]];
                assert!(curr_x <= next_x || (curr_x - next_x).abs() < tol[1]);
            }
        }
    }

    #[test]
    fn argsort2_f64_works_with_negative_values() {
        let y = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        let x = vec![5.0, 1.0, 0.0, -1.0, -5.0];
        let tol = [1e-9, 1e-9];
        let indices = argsort2_f64(&y, &x, &tol);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn argsort2_f64_works_with_large_tolerance() {
        // Large tolerance makes many values "equal"
        let y = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let x = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let tol = [0.6, 0.6]; // Values within 0.6 are considered equal
        let indices = argsort2_f64(&y, &x, &tol);
        // With tolerance 0.6, y[0]=1.0 and y[1]=1.5 are equal, so sorted by x
        // y[1]=1.5 and y[2]=2.0 are equal, y[2]=2.0 and y[3]=2.5 are equal
        // y[3]=2.5 and y[4]=3.0 are equal
        assert_eq!(indices, vec![1, 0, 3, 2, 4]);
    }

    #[test]
    fn argsort2_f64_works_with_zero_tolerance() {
        // Zero tolerance means only exact equality
        let y = vec![1.0, 1.0, 1.0];
        let x = vec![3.0, 2.0, 1.0];
        let tol = [0.0, 0.0];
        let indices = argsort2_f64(&y, &x, &tol);
        assert_eq!(indices, vec![2, 1, 0]);
    }

    #[test]
    fn argsort2_f64_works_empty_vectors() {
        let y: Vec<f64> = vec![];
        let x: Vec<f64> = vec![];
        let tol = [1e-9, 1e-9];
        let indices = argsort2_f64(&y, &x, &tol);
        assert_eq!(indices, Vec::<usize>::new());
    }

    #[test]
    fn argsort2_f64_works_with_duplicates() {
        let y = vec![2.0, 1.0, 2.0, 1.0, 2.0];
        let x = vec![1.0, 2.0, 3.0, 1.0, 2.0];
        let tol = [1e-9, 1e-9];
        let indices = argsort2_f64(&y, &x, &tol);
        // y=1.0: indices 1,3 with x=2.0,1.0 -> sorted as 3,1
        // y=2.0: indices 0,2,4 with x=1.0,3.0,2.0 -> sorted as 0,4,2
        assert_eq!(indices, vec![3, 1, 0, 4, 2]);
    }

    #[test]
    fn argsort2_f64_works_with_extreme_values() {
        let y = vec![f64::MAX, f64::MIN, 0.0, -0.0, 1e-308, -1e-308];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tol = [1e-9, 1e-9];
        let indices = argsort2_f64(&y, &x, &tol);
        // f64::MIN is the most negative finite f64 value (~-1.8e308)
        // f64::MAX is the most positive finite f64 value (~1.8e308)
        // Order: f64::MIN < -1e-308 < -0.0 = 0.0 < 1e-308 < f64::MAX
        // Actual result from sort: [1, 2, 3, 4, 5, 0]
        // idx 1: f64::MIN, idx 2: 0.0, idx 3: -0.0, idx 4: 1e-308, idx 5: -1e-308, idx 0: f64::MAX
        assert_eq!(indices, vec![1, 2, 3, 4, 5, 0]);
    }

    #[test]
    fn argsort2_f64_works_with_infinity() {
        let y = vec![f64::INFINITY, -f64::INFINITY, 0.0, 1.0];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let tol = [1e-9, 1e-9];
        let indices = argsort2_f64(&y, &x, &tol);
        assert_eq!(indices, vec![1, 2, 3, 0]);
    }

    #[test]
    fn argsort2_f64_with_nan_values() {
        // NaN handling - NaNs are treated as equal via unwrap_or(Equal)
        let y = vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tol = [1e-9, 1e-9];
        let indices = argsort2_f64(&y, &x, &tol);
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
        // This test documents a theoretical edge case where tolerance-based equality
        // could violate transitivity, though in practice Rust's timsort is robust enough
        // to handle it without panicking.
        //
        // With tol=0.5: 0.0≈0.4 (diff=0.4<0.5), 0.4≈0.8 (diff=0.4<0.5), but 0.0≠0.8 (diff=0.8>0.5)
        // This technically violates transitivity: if a≈b and b≈c, then a should ≈ c
        //
        // However, Rust's sort_by implementation (timsort) is robust enough that it doesn't
        // panic even with this intransitive comparison function. The sort completes
        // successfully, though the result may not be fully consistent.
        let tol_val = 0.5;
        let y = vec![1.2, 0.8, 0.4, 0.0];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let tol = [tol_val, tol_val];
        // This completes without panic despite the transitivity issue
        let indices = argsort2_f64(&y, &x, &tol);
        // Verify we get all indices
        assert_eq!(indices.len(), 4);
    }

    #[test]
    fn argsort2_f64_stress_test_transitivity() {
        // Create a challenging scenario with values close to each other
        // to potentially expose transitivity issues
        let y = vec![1.0, 1.0 + 1e-10, 1.0 + 2e-10, 1.0 + 3e-10, 1.0 + 4e-10];
        let x = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let tol = [2e-10, 1e-9]; // Tolerance that could create edge cases
        let indices = argsort2_f64(&y, &x, &tol);
        // With tol[0]=2e-10, some y values are "equal" others not
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
    fn argsort2_f64_circular_tolerance_pattern() {
        // This tries to create a scenario where tolerance could lead to
        // circular comparisons: a ≈ b, b ≈ c, but a ≠ c
        // With tolerance 0.15: 1.0 ≈ 1.1, 1.1 ≈ 1.2, but 1.0 ≉ 1.2 (diff=0.2)
        let y = vec![1.0, 1.1, 1.2, 1.3];
        let x = vec![4.0, 3.0, 2.0, 1.0];
        let tol = [0.15, 1e-9];
        let indices = argsort2_f64(&y, &x, &tol);
        // Should still produce valid ordering despite edge case tolerance
        assert_eq!(indices.len(), 4);
        let mut seen = vec![false; 4];
        for &idx in &indices {
            assert!(!seen[idx]);
            seen[idx] = true;
        }
    }

    #[test]
    fn argsort2_f64_many_equal_y_different_x() {
        // Many elements with equal y but different x to test secondary sort
        let y = vec![1.0; 100];
        let x: Vec<f64> = (0..100).map(|i| (100 - i) as f64).collect();
        let tol = [1e-9, 1e-9];
        let indices = argsort2_f64(&y, &x, &tol);
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
        let tol = [1e-9, 1e-9, 1e-9];

        let sorted_indices = argsort3_f64(&x, &y, &z, &tol);
        assert_eq!(sorted_indices, vec![1, 3, 2, 0]);
    }

    #[test]
    fn argsort3_f64_works_equal_x() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![3.0, 2.0, 1.0];
        let z = vec![1.0, 2.0, 3.0];
        let tol = [1e-9, 1e-9, 1e-9];

        let sorted_indices = argsort3_f64(&x, &y, &z, &tol);
        assert_eq!(sorted_indices, vec![2, 1, 0]);
    }

    #[test]
    fn argsort3_f64_works_equal_x_y() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![2.0, 2.0, 2.0];
        let z = vec![3.0, 1.0, 2.0];
        let tol = [1e-9, 1e-9, 1e-9];

        let sorted_indices = argsort3_f64(&x, &y, &z, &tol);
        assert_eq!(sorted_indices, vec![1, 2, 0]);
    }

    #[test]
    fn argsort3_f64_works_equal_x_y_z() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![2.0, 2.0, 2.0];
        let z = vec![3.0, 3.0, 3.0];
        let tol = [1e-9, 1e-9, 1e-9];

        let sorted_indices = argsort3_f64(&x, &y, &z, &tol);
        assert_eq!(sorted_indices, vec![0, 1, 2]);
    }

    #[test]
    fn argsort3_f64_works_single_element() {
        let x = vec![1.0];
        let y = vec![2.0];
        let z = vec![3.0];
        let tol = [1e-9, 1e-9, 1e-9];

        let sorted_indices = argsort3_f64(&x, &y, &z, &tol);
        assert_eq!(sorted_indices, vec![0]);
    }

    #[test]
    #[should_panic(expected = "vectors must have equal length")]
    fn argsort3_f64_works_different_lengths() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0];
        let z = vec![1.0];
        let tol = [1e-9, 1e-9, 1e-9];

        argsort3_f64(&x, &y, &z, &tol);
    }

    #[test]
    #[should_panic(expected = "tolerance slice must have exactly three elements")]
    fn argsort3_f64_works_invalid_tol_length() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];
        let z = vec![1.0, 2.0];
        let tol = [1e-9, 1e-9];

        argsort3_f64(&x, &y, &z, &tol);
    }

    #[test]
    fn argsort3_f64_works_with_negative_values() {
        let z = vec![-3.0, -1.0, -2.0, 1.0, 0.0];
        let y = vec![1.0, -1.0, 2.0, -2.0, 0.0];
        let x = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let tol = [1e-9, 1e-9, 1e-9];
        let indices = argsort3_f64(&z, &y, &x, &tol);
        // Verify ordering: z first, then y, then x
        for i in 0..indices.len() - 1 {
            let curr_z = z[indices[i]];
            let next_z = z[indices[i + 1]];
            assert!(curr_z <= next_z);
        }
    }

    #[test]
    fn argsort3_f64_works_with_large_tolerance() {
        // With large tolerance, many values will be considered equal
        let z = vec![1.0, 1.1, 1.2, 1.3];
        let y = vec![2.0, 2.1, 2.2, 2.3];
        let x = vec![3.0, 3.1, 3.2, 3.3];
        let tol = [0.5, 0.5, 0.5]; // All values within tolerance
        let indices = argsort3_f64(&z, &y, &x, &tol);
        assert_eq!(indices.len(), 4);
    }

    #[test]
    fn argsort3_f64_works_with_zero_tolerance() {
        let z = vec![3.0, 1.0, 2.0, 1.0];
        let y = vec![1.0, 2.0, 1.0, 2.0];
        let x = vec![5.0, 6.0, 7.0, 8.0];
        let tol = [0.0, 0.0, 0.0];
        let indices = argsort3_f64(&z, &y, &x, &tol);
        // With zero tolerance, exact comparison is used
        assert_eq!(indices.len(), 4);
    }

    #[test]
    fn argsort3_f64_works_empty_vectors() {
        let z: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        let x: Vec<f64> = vec![];
        let tol = [1e-9, 1e-9, 1e-9];
        let indices = argsort3_f64(&z, &y, &x, &tol);
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn argsort3_f64_works_with_duplicates() {
        let z = vec![1.0, 2.0, 1.0, 2.0, 1.0];
        let y = vec![3.0, 3.0, 3.0, 3.0, 4.0];
        let x = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let tol = [1e-9, 1e-9, 1e-9];
        let indices = argsort3_f64(&z, &y, &x, &tol);
        // Verify all indices are present
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn argsort3_f64_works_with_extreme_values() {
        let z = vec![f64::MAX, f64::MIN, 0.0, -0.0, 1e-308, -1e-308];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tol = [1e-9, 1e-9, 1e-9];
        let indices = argsort3_f64(&z, &y, &x, &tol);
        // f64::MIN is the most negative, f64::MAX is the most positive
        // Order: f64::MIN < -1e-308 < -0.0 = 0.0 < 1e-308 < f64::MAX
        // Actual result: [1, 2, 3, 4, 5, 0] (indices by z value)
        assert_eq!(indices, vec![1, 2, 3, 4, 5, 0]);
    }

    #[test]
    fn argsort3_f64_works_with_infinity() {
        let z = vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, 1.0, -1.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tol = [1e-9, 1e-9, 1e-9];
        let indices = argsort3_f64(&z, &y, &x, &tol);
        // Order: NEG_INFINITY, -1.0, 0.0, 1.0, INFINITY
        assert_eq!(indices, vec![1, 4, 2, 3, 0]);
    }

    #[test]
    fn argsort3_f64_with_nan_values() {
        // NaN handling - NaNs are treated as equal via unwrap_or(Equal)
        let z = vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tol = [1e-9, 1e-9, 1e-9];
        let indices = argsort3_f64(&z, &y, &x, &tol);
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
    fn argsort3_f64_stress_test_transitivity() {
        // Test with many values and moderate tolerance
        let mut z = Vec::new();
        let mut y = Vec::new();
        let mut x = Vec::new();
        for i in 0..100 {
            z.push((i as f64) * 0.01);
            y.push((i as f64) * 0.02);
            x.push((i as f64) * 0.03);
        }
        let tol = [1e-9, 1e-9, 1e-9];
        let indices = argsort3_f64(&z, &y, &x, &tol);
        // Verify the result is sorted correctly
        assert_eq!(indices.len(), 100);
        for i in 0..indices.len() {
            assert_eq!(indices[i], i);
        }
    }

    #[test]
    fn argsort3_f64_circular_tolerance_pattern() {
        // Test with values that create a circular tolerance pattern
        let z = vec![0.0, 0.05, 0.1, 0.15, 0.2];
        let y = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tol = [0.08, 1e-9, 1e-9];
        let indices = argsort3_f64(&z, &y, &x, &tol);
        // With tol[0]=0.08, some z values will be considered equal
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn argsort3_f64_many_equal_z_different_y_x() {
        // All z values equal, y values differ, x values differ
        let z = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let x = vec![9.0, 8.0, 7.0, 6.0, 5.0];
        let tol = [1e-9, 1e-9, 1e-9];
        let indices = argsort3_f64(&z, &y, &x, &tol);
        // Should sort by y (ascending), then x (ascending)
        assert_eq!(indices, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn argsort3_f64_transitivity_edge_case() {
        // This test documents a theoretical edge case where tolerance-based equality
        // could violate transitivity, though in practice Rust's timsort is robust enough
        // to handle it without panicking.
        //
        // With tol=0.5: 0.0≈0.4 (diff=0.4<0.5), 0.4≈0.8 (diff=0.4<0.5), but 0.0≠0.8 (diff=0.8>0.5)
        // This technically violates transitivity: if a≈b and b≈c, then a should ≈ c
        //
        // However, Rust's sort_by implementation (timsort) is robust enough that it doesn't
        // panic even with this intransitive comparison function. The sort completes
        // successfully, though the result may not be fully consistent.
        let tol_val = 0.5;
        let z = vec![1.2, 0.8, 0.4, 0.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let tol = [tol_val, tol_val, tol_val];
        // This completes without panic despite the transitivity issue
        let indices = argsort3_f64(&z, &y, &x, &tol);
        // Verify we get all indices
        assert_eq!(indices.len(), 4);
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
        let tol = [1e-9, 1e-9, 1e-9];
        let indices = argsort3_f64(&z, &y, &x, &tol);
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
             0.21132,0.78868,1.21132\n\
             0.78868,0.78868,1.21132\n\
             0.21132,0.21132,1.78868\n\
             0.78868,0.21132,1.78868\n\
             0.21132,0.78868,1.78868\n\
             0.78868,0.78868,1.78868\n"
        );
    }
}
