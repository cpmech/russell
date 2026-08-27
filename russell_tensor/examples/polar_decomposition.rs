//! Example: polar decomposition of the deformation gradient
//!
//! This example computes the polar decomposition `F = R · U` of the
//! deformation gradient `F`, where `R` is a proper orthogonal tensor (a
//! rotation) and `U` is a symmetric positive-definite tensor (the right
//! stretch). The unified `polar_decomp` dispatcher is used, selecting the
//! quaternion-based algorithm of Higham & Noferini (2016).
//!
//! Reference: N. J. Higham and V. Noferini, "An algorithm to compute the polar
//! decomposition of a 3×3 matrix", Numer. Algorithms, 73(2):349–369, 2016.

use russell_lab::{Matrix, mat_approx_eq, mat_mat_mul, mat_t_mat_mul};
use russell_tensor::{PolarAlgo, Rep, StrError, Tensor2, polar_decomp};

fn main() -> Result<(), StrError> {
    // Deformation gradient (Higham & Noferini, test 5.1)
    let ff = Tensor2::from_std_matrix(
        &[
            [0.1, 0.2, 0.3], // row 1
            [0.1, 0.1, 0.0], // row 2
            [0.3, 0.2, 0.1], // row 3
        ],
        Rep::General,
    )?;
    println!("F =\n{:.6}", ff.as_std_matrix());

    // Allocate the rotation tensor R (orthogonal) and the right stretch U (symmetric)
    let mut rr = Tensor2::new(Rep::General);
    let mut uu = Tensor2::new(Rep::Symmetric);

    // Compute the polar decomposition F = R · U (using the Higham algorithm)
    polar_decomp(&mut rr, &mut uu, None, PolarAlgo::Higham, &ff).unwrap();

    // Print the factors
    println!("R =\n{:.6}", rr.as_std_matrix());
    println!("U =\n{:.6}", uu.as_std_matrix());

    // Verify that F = R · U ...
    let ff_mat = ff.as_std_matrix();
    let rr_mat = rr.as_std_matrix();
    let uu_mat = uu.as_std_matrix();
    let mut ru = Matrix::new(3, 3);
    mat_mat_mul(&mut ru, 1.0, &rr_mat, &uu_mat, 0.0)?;
    mat_approx_eq(&ru, &ff_mat, 1e-13);

    // ... and that R is orthogonal (Rᵀ · R = I)
    let mut rtr = Matrix::new(3, 3);
    mat_t_mat_mul(&mut rtr, 1.0, &rr_mat, &rr_mat, 0.0)?;
    mat_approx_eq(&rtr, &Matrix::diagonal(&[1.0, 1.0, 1.0]), 1e-13);

    Ok(())
}
