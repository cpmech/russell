//! Test cases and check helpers shared by the two polar-decomposition
//! implementations: Brannon's `polar_rotation_brannon` (iterative) and
//! Higham & Noferini's `polar_quaternion_higham` (quaternion).
//!
//! Each implementation is validated against all of these test cases and
//! cross-checked against the other implementation.

use crate::polar_decomp::{polar_decomp, PolarAlgo};
use crate::{Rep, Tensor2};
use russell_lab::{Matrix, mat_approx_eq, mat_mat_mul, mat_t_mat_mul};

// -----------------------------------------------------------------------------------
// Test matrices
// -----------------------------------------------------------------------------------

/// Example 01 (Brannon, Eq. 12.39): in-plane deformation gradient;
/// the polar rotation is a 60° rotation about the E3 axis.
pub fn example01() -> Tensor2 {
    #[rustfmt::skip]
    let a = Tensor2::from_std_matrix(&[
        [ 0.61784609690826542, -0.70889727457341833, 0.0],
        [ 0.59014083110323967,  0.13215390309173483, 0.0],
        [ 0.0,                  0.0,                 3.0],
    ], Rep::General).unwrap();
    a
}

/// Example 03 (McGinty, continuummechanics.org): fully 3-D deformation gradient.
pub fn example03() -> Tensor2 {
    #[rustfmt::skip]
    let a = Tensor2::from_std_matrix(&[
        [ 1.000,  0.495,  0.500],
        [-0.333,  1.000, -0.247],
        [ 0.959,  0.000,  1.500],
    ], Rep::General).unwrap();
    a
}

/// Higham & Noferini test (5.1).
pub fn case51() -> Tensor2 {
    #[rustfmt::skip]
    let a = Tensor2::from_std_matrix(&[
        [0.1, 0.2, 0.3],
        [0.1, 0.1, 0.0],
        [0.3, 0.2, 0.1],
    ], Rep::General).unwrap();
    a
}

/// Higham & Noferini test (5.2), for a given scale factor `y`.
pub fn case52(y: f64) -> Tensor2 {
    #[rustfmt::skip]
    let a = Tensor2::from_std_matrix(&[
        [(720.0 * y - 25.0) / 1275.0, (-650.0 * y + 300.0) / 1275.0, (710.0 * y + 300.0) / 1275.0],
        [(396.0 * y + 70.0) / 1275.0, (-145.0 * y - 840.0) / 1275.0, (178.0 * y - 840.0) / 1275.0],
        [(972.0 * y - 10.0) / 1275.0, (610.0 * y + 120.0) / 1275.0, (-529.0 * y + 120.0) / 1275.0],
    ], Rep::General).unwrap();
    a
}

// -----------------------------------------------------------------------------------
// Reference results
// -----------------------------------------------------------------------------------

/// Reference rotation for example 01 (60° about E3).
pub fn example01_rotation() -> [[f64; 3]; 3] {
    [
        [0.5, -0.8660254037844386, 0.0],
        [0.8660254037844386, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

/// Reference right stretch for example 01.
pub fn example01_stretch() -> [[f64; 3]; 3] {
    [[0.82, -0.24, 0.0], [-0.24, 0.68, 0.0], [0.0, 0.0, 3.0]]
}

/// Reference rotation for example 03 (3-decimal published values).
pub fn example03_rotation() -> [[f64; 3]; 3] {
    [[0.914, 0.377, -0.148], [-0.374, 0.926, 0.049], [0.156, 0.011, 0.988]]
}

/// Reference right stretch for example 03 (3-decimal published values).
pub fn example03_stretch() -> [[f64; 3]; 3] {
    [[1.188, 0.079, 0.783], [0.079, 1.113, -0.024], [0.783, -0.024, 1.396]]
}

/// Exact polar factor for test 5.2 (well-conditioned case).
pub fn case52_rotation() -> [[f64; 3]; 3] {
    [
        [139.0 / 255.0, -14.0 / 51.0, 202.0 / 255.0],
        [466.0 / 1275.0, -197.0 / 255.0, -662.0 / 1275.0],
        [962.0 / 1275.0, 146.0 / 255.0, -409.0 / 1275.0],
    ]
}

// -----------------------------------------------------------------------------------
// Check helpers
// -----------------------------------------------------------------------------------

/// Checks that `A = Q · H` with `Q` orthogonal, within the given tolerance.
pub fn check_polar(a: &Tensor2, q: &Tensor2, h: &Tensor2, tol: f64) {
    let am = a.as_std_matrix();
    let qm = q.as_std_matrix();
    let hm = h.as_std_matrix();
    let mut qh = Matrix::new(3, 3);
    mat_mat_mul(&mut qh, 1.0, &qm, &hm, 0.0).unwrap();
    mat_approx_eq(&qh, &am, tol);
    let mut qtq = Matrix::new(3, 3);
    mat_t_mat_mul(&mut qtq, 1.0, &qm, &qm, 0.0).unwrap();
    mat_approx_eq(&qtq, &Matrix::diagonal(&[1.0, 1.0, 1.0]), tol);
}

/// Runs both algorithms on `a` and checks that each satisfies `A = Q · H`
/// (with `Q` orthogonal) and that the two agree (the polar decomposition is
/// unique when `det(A) > 0`).
pub fn check_agree(a: &Tensor2) {
    // Brannon (iterative)
    let mut rb = Tensor2::new(Rep::General);
    let mut ub = Tensor2::new(Rep::Symmetric);
    let mut vb = Tensor2::new(Rep::Symmetric);
    polar_decomp(&mut rb, &mut ub, Some(&mut vb), PolarAlgo::Brannon, a).unwrap();
    check_polar(a, &rb, &ub, 1e-13);

    // Higham & Noferini (quaternion)
    let mut qh = Tensor2::new(Rep::General);
    let mut hh = Tensor2::new(Rep::Symmetric);
    polar_decomp(&mut qh, &mut hh, None, PolarAlgo::Higham, a).unwrap();
    check_polar(a, &qh, &hh, 1e-13);

    // The two implementations must agree
    mat_approx_eq(&rb.as_std_matrix(), &qh.as_std_matrix(), 1e-13);
    mat_approx_eq(&ub.as_std_matrix(), &hh.as_std_matrix(), 1e-13);
}
