pub struct Matrix {
    pub(super) data: Vec<f64>, // col-major => Fortran
}

/// Holds matrix components
///
/// # Note
///
/// Data is stored in col-major format
///
/// Example of col-major data:
///        _      _
///       |  0  3  |
///   A = |  1  4  |            ⇒     a = [0, 1, 2, 3, 4, 5]
///       |_ 2  5 _|(m x n)
///
///   a[i+j*m] = A[i][j]
///
impl Matrix {
    pub fn new(nrow: usize, ncol: usize) -> Self {
        Matrix {
            data: vec![0.0; nrow * ncol],
        }
    }

    pub fn from(data: &[&[f64]]) -> Self {
        let nrow = data.len();
        if nrow == 0 {
            return Matrix { data: Vec::new() };
        }
        let ncol = data[0].len();
        let mut matrix = Matrix {
            data: vec![0.0; nrow * ncol],
        };
        for i in 0..nrow {
            for j in 0..ncol {
                matrix.data[i + j * nrow] = data[i][j];
            }
        }
        matrix
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn new_matrix_works() {
        let u = Matrix::new(3, 3);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }

    #[test]
    fn from_works() {
        #[rustfmt::skip]
        let aa = Matrix::from(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 8.0],
        ]);
        let correct = &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 8.0];
        assert_vec_approx_eq!(aa.data, correct, 1e-15);
    }
}
