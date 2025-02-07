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
