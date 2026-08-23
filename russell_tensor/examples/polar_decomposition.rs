//! Example: polar decomposition of a 3×3 matrix
//!
//! This example computes the polar decomposition `A = Q · H` of a 3×3 matrix,
//! where `Q` is orthogonal (a rotation) and `H` is symmetric positive
//! semidefinite (a stretch), using the quaternion-based algorithm of
//! Higham & Noferini (2016).
//!
//! Reference: N. J. Higham and V. Noferini, "An algorithm to compute the polar
//! decomposition of a 3×3 matrix", Num. Algorithms, 73(2):349–369, 2016.

use russell_lab::{Matrix, mat_approx_eq, mat_mat_mul, mat_t_mat_mul};
use russell_tensor::{Rep, StrError, Tensor2, polar_decomp_higham};

fn main() -> Result<(), StrError> {
    // Input matrix (Higham & Noferini, test 5.1)
    let a = Tensor2::from_std_matrix(
        &[
            [0.1, 0.2, 0.3], // row 1
            [0.1, 0.1, 0.0], // row 2
            [0.3, 0.2, 0.1], // row 3
        ],
        Rep::General,
    )?;
    println!("A =\n{:.6}", a.as_std_matrix());

    // Allocate the polar factor Q (orthogonal) and the stretch H (symmetric)
    let mut q = Tensor2::new(Rep::General);
    let mut h = Tensor2::new(Rep::Symmetric);

    // Compute the polar decomposition: A = Q · H
    polar_decomp_higham(&mut q, &mut h, &a);

    // Print the factors
    println!("Q =\n{:.6}", q.as_std_matrix());
    println!("H =\n{:.6}", h.as_std_matrix());

    // Verify that A = Q · H ...
    let am = a.as_std_matrix();
    let qm = q.as_std_matrix();
    let hm = h.as_std_matrix();
    let mut qh = Matrix::new(3, 3);
    mat_mat_mul(&mut qh, 1.0, &qm, &hm, 0.0)?;
    mat_approx_eq(&qh, &am, 1e-13);

    // ... and that Q is orthogonal (Qᵀ · Q = I)
    let mut qtq = Matrix::new(3, 3);
    mat_t_mat_mul(&mut qtq, 1.0, &qm, &qm, 0.0)?;
    mat_approx_eq(&qtq, &Matrix::diagonal(&[1.0, 1.0, 1.0]), 1e-13);

    Ok(())
}
