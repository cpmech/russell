/// Converts nested slice into vector representing a matrix in col-major format
///
/// Example of col-major data:
///        _      _
///       |  0  3  |
///   A = |  1  4  |            ⇒     a = [0, 1, 2, 3, 4, 5]
///       |_ 2  5 _|(m x n)
///
///   a[i+j*m] = A[i][j]
///
/// # Panics
///
/// This function panics if there are rows with different number of columns
///
pub fn slice_to_colmajor(a: &[&[f64]]) -> Vec<f64> {
    let nrow = a.len();
    if nrow == 0 {
        return Vec::new();
    }
    let ncol = a[0].len();
    let mut data = vec![0.0; nrow * ncol];
    for i in 0..nrow {
        if a[i].len() != ncol {
            panic!("all rows must have the same number of columns");
        }
        for j in 0..ncol {
            data[i + j * nrow] = a[i][j];
        }
    }
    data
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn slice_to_colmajor_works() {
        #[rustfmt::skip]
        let data = slice_to_colmajor(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 8.0],
        ]);
        let correct = &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 8.0];
        assert_vec_approx_eq!(data, correct, 1e-15);
    }

    #[test]
    #[should_panic(expected = "all rows must have the same number of columns")]
    fn slice_to_colmajor_panics_on_wrong_columns() {
        #[rustfmt::skip]
         slice_to_colmajor(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0],
            &[7.0, 8.0, 8.0],
        ]);
    }
}
