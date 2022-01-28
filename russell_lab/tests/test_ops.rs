use russell_lab::{mat_mat_mul, mat_vec_mul, matrix_norm, vector_norm, Matrix, NormMat, NormVec, StrError, Vector};

#[test]
fn test_mat_vec_mul() -> Result<(), StrError> {
    // v  :=  a  ⋅ u
    // (m)  (m,n) (n)
    for m in [0, 7, 15_usize] {
        for n in [0, 4, 8_usize] {
            let a = Matrix::filled(m, n, 1.0);
            let u = Vector::filled(n, 1.0);
            let mut v = Vector::new(m);
            mat_vec_mul(&mut v, 1.0, &a, &u)?;
            if m == 0 {
                assert_eq!(vector_norm(&v, NormVec::Max), 0.0);
            } else {
                assert_eq!(vector_norm(&v, NormVec::Max), n as f64);
            }
        }
    }
    Ok(())
}

#[test]
fn test_mat_mat_mul() -> Result<(), StrError> {
    //   c  :=  a  ⋅  b
    // (m,n)  (m,k) (k,n)
    for m in [0, 5, 7_usize] {
        for n in [0, 6, 12_usize] {
            let mut c = Matrix::new(m, n);
            for k in [0, 5, 10, 15_usize] {
                let a = Matrix::filled(m, k, 1.0);
                let b = Matrix::filled(k, n, 1.0);
                mat_mat_mul(&mut c, 1.0, &a, &b)?;
                if m == 0 || n == 0 {
                    assert_eq!(matrix_norm(&c, NormMat::Max), 0.0);
                } else {
                    assert_eq!(matrix_norm(&c, NormMat::Max), k as f64);
                }
            }
        }
    }
    Ok(())
}
