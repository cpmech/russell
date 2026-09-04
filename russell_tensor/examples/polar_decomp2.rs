use russell_lab::mat_approx_eq;
use russell_tensor::{PolarAlgo, SQRT_2, StrError, Tensor2, polar_decomp};

fn main() -> Result<(), StrError> {
    #[rustfmt::skip]
    let ff = Tensor2::<9>::from_std_matrix(&[
        [(1.0 + 4.0 * SQRT_2) / 9.0, (8.0 + SQRT_2) / 18.0, (32.0 + SQRT_2) / 36.0],
        [(8.0 + 5.0 * SQRT_2) / 18.0, (7.0 + 2.0 * SQRT_2) / 9.0, (-4.0 + SQRT_2) / 9.0],
        [(-32.0 + 7.0 * SQRT_2) / 36.0, -4.0 * (-1.0 + SQRT_2) / 9.0, (-1.0 - 2.0 * SQRT_2) / 9.0],
    ])?;
    println!("F =\n{:.6}", ff.as_std_matrix());

    // Allocate the rotation tensor R (orthogonal) and the right stretch U (symmetric)
    let mut rr = Tensor2::<9>::new();
    let mut uu = Tensor2::<6>::new();

    // Compute the polar decomposition F = R · U (using the Higham algorithm)
    polar_decomp(&mut rr, &mut uu, None, PolarAlgo::Higham, &ff)?;

    // Print the factors
    println!("R =\n{:.6}", rr.as_std_matrix());
    println!("U =\n{:.6}", uu.as_std_matrix());

    // Check
    let expected_rr = [
        [1.0 / 9.0, 4.0 / 9.0, 8.0 / 9.0],
        [4.0 / 9.0, 7.0 / 9.0, -4.0 / 9.0],
        [-8.0 / 9.0, 4.0 / 9.0, -1.0 / 9.0],
    ];
    let expected_uu = [
        [1.0, 1.0 / SQRT_2, 1.0 / (2.0 * SQRT_2)],
        [1.0 / SQRT_2, 1.0, 0.0],
        [1.0 / (2.0 * SQRT_2), 0.0, 1.0],
    ];
    mat_approx_eq(&rr.as_std_matrix(), &expected_rr, 1e-15);
    mat_approx_eq(&uu.as_std_matrix(), &expected_uu, 1e-15);

    Ok(())
}
