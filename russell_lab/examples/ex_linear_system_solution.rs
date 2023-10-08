use russell_lab::*;

fn main() -> Result<(), StrError> {
    // matrix
    #[rustfmt::skip]
    let mut a = Matrix::from(&[
        [2.0,  3.0,  0.0, 0.0, 0.0],
        [3.0,  0.0,  4.0, 0.0, 6.0],
        [0.0, -1.0, -3.0, 2.0, 0.0],
        [0.0,  0.0,  1.0, 0.0, 0.0],
        [0.0,  4.0,  2.0, 0.0, 1.0],
    ]);

    // right-hand-side
    let mut b = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    // solve b := x := A⁻¹ b
    solve_lin_sys(&mut b, &mut a)?;

    // alias for convenience
    let x = &b;

    // check
    let correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
    vec_approx_eq(x.as_data(), correct, 1e-14);
    Ok(())
}
