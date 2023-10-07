use crate::StrError;
use crate::{Matrix, Vector};

/// Sorts (descending) the components of a vector and corresponding columns of a matrix
///
/// For example, this function is useful to sorts the eigenvalues in descending order
/// and, at the same time, rearrange the corresponding eigenvectors (columns).
///
/// # Input
///
/// * `l` -- e.g., vector of eigenvalues; dim = n
/// * `v` -- e.g., matrix of eigenvectors; square, dims = (n, n)
///
/// # Reference
///
/// This code is based on Section 11.1 Jacobi Transformations (page 576) of Numerical Recipes.
///
/// * Press WH, Teukolsky SA, Vetterling WT and Flannery BP (2007),
///   Numerical Recipes: The Art of Scientific Computing, 3rd Edition
pub fn sort_vec_mat(l: &mut Vector, v: &mut Matrix) -> Result<(), StrError> {
    let (m, n) = v.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if l.dim() != n {
        return Err("vector must have the same dimension as matrix");
    }
    for i in 0..(n - 1) {
        let mut p = l[i];
        let mut k = i;
        for j in i..n {
            if l[j] >= p {
                p = l[j];
                k = j;
            }
        }
        if k != i {
            l[k] = l[i];
            l[i] = p;
            for j in 0..n {
                p = v.get(j, i);
                v.set(j, i, v.get(j, k));
                v.set(j, k, p);
            }
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::sort_vec_mat;
    use crate::vec_approx_eq;
    use crate::{mat_approx_eq, Matrix, Vector};

    #[test]
    fn sort_vec_mat_handles_errors() {
        let mut l = Vector::new(2);
        let mut v = Matrix::new(1, 2);
        assert_eq!(sort_vec_mat(&mut l, &mut v).err(), Some("matrix must be square"));
        let mut v = Matrix::new(1, 1);
        assert_eq!(
            sort_vec_mat(&mut l, &mut v).err(),
            Some("vector must have the same dimension as matrix")
        );
    }

    #[test]
    fn sort_vec_mat_works() {
        // already sorted
        let mut l = Vector::from(&[2.0, 1.0]);
        let mut v = Matrix::from(&[[102.0, 101.0], [202.0, 201.0]]);
        sort_vec_mat(&mut l, &mut v).unwrap();
        let v_correct = &[[102.0, 101.0], [202.0, 201.0]];
        vec_approx_eq(l.as_data(), &[2.0, 1.0], 1e-15);
        mat_approx_eq(&v, v_correct, 1e-15);

        // need to sort
        let mut l = Vector::from(&[3.0, 7.0, 1.0, 4.0]);
        let mut v = Matrix::from(&[
            [103.0, 107.0, 101.0, 104.0],
            [203.0, 207.0, 201.0, 204.0],
            [303.0, 307.0, 301.0, 304.0],
            [403.0, 407.0, 401.0, 404.0],
        ]);
        sort_vec_mat(&mut l, &mut v).unwrap();
        let v_correct = &[
            [107.0, 104.0, 103.0, 101.0],
            [207.0, 204.0, 203.0, 201.0],
            [307.0, 304.0, 303.0, 301.0],
            [407.0, 404.0, 403.0, 401.0],
        ];
        vec_approx_eq(l.as_data(), &[7.0, 4.0, 3.0, 1.0], 1e-15);
        mat_approx_eq(&v, v_correct, 1e-15);
    }
}
