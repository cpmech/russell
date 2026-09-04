use russell_tensor::{SQRT_2, StrError, Tensor2};

fn main() -> Result<(), StrError> {
    // Allocate a general second-order tensor given the standard components
    let a = Tensor2::<9>::from_std_matrix(&[
        [1.0, SQRT_2 * 2.0, SQRT_2 * 3.0],
        [SQRT_2 * 4.0, 5.0, SQRT_2 * 6.0],
        [SQRT_2 * 7.0, SQRT_2 * 8.0, 9.0],
    ])?;
    assert_eq!(
        format!("{:.1}", a),
        "┌      ┐\n\
         │  1.0 │\n\
         │  5.0 │\n\
         │  9.0 │\n\
         │  6.0 │\n\
         │ 14.0 │\n\
         │ 10.0 │\n\
         │ -2.0 │\n\
         │ -2.0 │\n\
         │ -4.0 │\n\
         └      ┘"
    );

    // Allocate a symmetric second-order tensor given the standard components
    let b = Tensor2::<6>::from_std_matrix(&[
        [1.0, 4.0 / SQRT_2, 6.0 / SQRT_2],
        [4.0 / SQRT_2, 2.0, 5.0 / SQRT_2],
        [6.0 / SQRT_2, 5.0 / SQRT_2, 3.0],
    ])?;
    assert_eq!(
        format!("{:.1}", b),
        "┌     ┐\n\
         │ 1.0 │\n\
         │ 2.0 │\n\
         │ 3.0 │\n\
         │ 4.0 │\n\
         │ 5.0 │\n\
         │ 6.0 │\n\
         └     ┘"
    );

    // Allocate a symmetric second-order tensor given the standard components for 2D problems
    let c = Tensor2::<4>::from_std_matrix(&[[1.0, 4.0 / SQRT_2, 0.0], [4.0 / SQRT_2, 2.0, 0.0], [0.0, 0.0, 3.0]])?;
    assert_eq!(
        format!("{:.1}", c),
        "┌     ┐\n\
         │ 1.0 │\n\
         │ 2.0 │\n\
         │ 3.0 │\n\
         │ 4.0 │\n\
         └     ┘"
    );
    Ok(())
}
