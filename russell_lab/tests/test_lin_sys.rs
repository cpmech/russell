use russell_lab::*;

#[test]
fn test_lin_sys() -> Result<(), &'static str> {
    const TARGET: f64 = 1234.0;
    for m in [0, 5, 7, 12_usize] {
        let mut a = Matrix::filled(m, m, 1.0);
        let mut b = Vector::filled(m, TARGET);
        for i in 0..m {
            for j in (i + 1)..m {
                a[i][j] *= -1.0;
            }
        }
        solve_lin_sys(&mut b, &mut a)?;
        if m == 0 {
            assert_eq!(b.as_data(), &[]);
        } else {
            let mut x_correct = Vector::new(m);
            x_correct[0] = TARGET;
            assert_eq!(b.as_data(), x_correct.as_data());
        }
    }
    Ok(())
}
