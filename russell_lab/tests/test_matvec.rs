use russell_lab::{mat_vec_mul, EnumVectorNorm, Matrix, Vector};

#[test]
fn test_matvec() -> Result<(), &'static str> {
    // v  :=  a  ⋅ u
    // (m)  (m,n) (n)
    for m in [0, 7, 15_usize] {
        for n in [0, 4, 8_usize] {
            let a = Matrix::filled(m, n, 1.0);
            let u = Vector::filled(n, 1.0);
            let mut v = Vector::new(m);
            mat_vec_mul(&mut v, 1.0, &a, &u)?;
            if m == 0 {
                assert_eq!(v.norm(EnumVectorNorm::Max), 0.0);
            } else {
                assert_eq!(v.norm(EnumVectorNorm::Max), n as f64);
            }
        }
    }
    Ok(())
}
