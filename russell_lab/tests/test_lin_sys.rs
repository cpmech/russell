use russell_lab::{mat_vec_mul, solve_lin_sys, vec_add, vec_norm, Matrix, Norm, StrError, Vector};

#[test]
fn test_lin_sys() -> Result<(), StrError> {
    const TARGET: f64 = 1234.0;
    for m in [0, 5, 7, 12_usize] {
        // prepare matrix and rhs
        let mut a = Matrix::filled(m, m, 1.0);
        let mut b = Vector::filled(m, TARGET);
        for i in 0..m {
            for j in (i + 1)..m {
                a[i][j] *= -1.0;
            }
        }

        // take copies
        let a_copy = a.clone();
        let b_copy = b.clone();

        // solve linear system: b := a⁻¹⋅b == x
        solve_lin_sys(&mut b, &mut a)?;

        // compare solution
        if m == 0 {
            assert_eq!(b.as_data(), &[]);
        } else {
            let mut x_correct = Vector::new(m);
            x_correct[0] = TARGET;
            assert_eq!(b.as_data(), x_correct.as_data());
        }

        // check solution a ⋅ x == rhs (with x == b)
        let mut rhs = Vector::new(m);
        let mut diff = Vector::new(m);
        mat_vec_mul(&mut rhs, 1.0, &a_copy, &b)?;
        vec_add(&mut diff, 1.0, &rhs, -1.0, &b_copy)?;
        assert_eq!(vec_norm(&diff, Norm::Max), 0.0);
    }
    Ok(())
}
