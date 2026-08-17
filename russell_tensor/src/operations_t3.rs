use super::{Tensor2, Tensor3};
use russell_lab::Vector;

#[allow(unused)]
use crate::Rep; // for documentation

/// Adds two third-order tensors
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// # Panics
///
/// A panic will occur if the tensors have different [Rep] or different case flag
pub fn t3_add(c: &mut Tensor3, alpha: f64, a: &Tensor3, beta: f64, b: &Tensor3) {
    assert_eq!(b.rep, a.rep);
    assert_eq!(c.rep, a.rep);
    assert_eq!(b.case_a, a.case_a);
    assert_eq!(c.case_a, a.case_a);
    for i in 0..a.nrow {
        for j in 0..a.ncol {
            c.mat[i][j] = alpha * a.mat[i][j] + beta * b.mat[i][j];
        }
    }
}

/// Performs the single-dot operation between a Tensor3 and a vector resulting in a Tensor2 (Case A)
///
/// Computes:
///
/// ```text
/// T = α H · u
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Tᵢⱼ = α Σ Hᵢⱼₖ uₖ
///        k
/// ```
///
/// Or, in Kelvin basis:
///
/// ```text
/// Tₘ = α Σ Hₘₖ uₖ
///        k
/// ```
///
/// # Output
///
/// * `T` -- the resulting second-order tensor; with the same [Rep] as `H`
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `hh` -- the third-order tensor
/// * `u` -- the 3D vector; must have 3 components (w.r.t. standard Cartesian basis)
///
/// # Panics
///
/// 1. If `H` was not allocated for Case A
/// 2. If `T` and `H` have different [Rep]
/// 3. If `u` does not have 3 components
pub fn t3_dot_vec(tt: &mut Tensor2, alpha: f64, hh: &Tensor3, u: &Vector) {
    assert!(hh.case_a);
    assert_eq!(tt.rep, hh.rep);
    assert_eq!(u.dim(), 3);
    for m in 0..hh.nrow {
        tt.vec[m] = alpha * (hh.mat[m][0] * u[0] + hh.mat[m][1] * u[1] + hh.mat[m][2] * u[2]);
    }
}

/// Performs the double-dot operation between a Tensor3 and a Tensor2 resulting in a vector (Case B)
///
/// Computes:
///
/// ```text
/// u = α H : T
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// uᵢ = α Σ Σ Hᵢⱼₖ Tⱼₖ
///       j k
/// ```
///
/// Or, in Kelvin basis:
///
/// ```text
/// uᵢ = α Σ Hᵢₙ Tₙ
///       n
/// ```
///
/// # Output
///
/// * `u` -- the resulting vector (with 3 standard components)
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `hh` -- the third-order tensor
/// * `T` -- the second-order tensor with the same [Rep] as `H`
///
/// # Panics
///
/// 1. If `H` was not allocated for Case B
/// 2. If `T` and `H` have different [Rep]
/// 3. If `u` does not have 3 components
pub fn t3_dot_t2(u: &mut Vector, alpha: f64, hh: &Tensor3, tt: &Tensor2) {
    assert!(!hh.case_a);
    assert_eq!(tt.rep, hh.rep);
    assert_eq!(u.dim(), 3);
    u[0] = 0.0;
    u[1] = 0.0;
    u[2] = 0.0;
    for n in 0..hh.ncol {
        u[0] += alpha * hh.mat[0][n] * tt.vec[n];
        u[1] += alpha * hh.mat[1][n] * tt.vec[n];
        u[2] += alpha * hh.mat[2][n] * tt.vec[n];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{t3_add, t3_dot_t2, t3_dot_vec};
    use crate::{Rep, SamplesTensor3, Tensor2, Tensor3};
    use russell_lab::{Matrix, Vector, mat_approx_eq, vec_approx_eq};

    #[test]
    fn t3_dot_vec_works() {
        // General
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1, Rep::General, true).unwrap();
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let mut tt = Tensor2::new(Rep::General);
        t3_dot_vec(&mut tt, 0.5, &hh, &u);
        let mat_expected = Matrix::from(&[
            [7.0, 34.0, 52.0],  // 0
            [61.0, 16.0, 43.0], // 1
            [79.0, 70.0, 25.0], // 2
        ]);
        mat_approx_eq(&tt.as_std_matrix(), &mat_expected, 1e-13);

        // Symmetric
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1, Rep::Symmetric, true).unwrap();
        let mut tt = Tensor2::new(Rep::Symmetric);
        t3_dot_vec(&mut tt, 0.5, &hh, &u);
        let mat_expected = Matrix::from(&[
            [7.0, 34.0, 52.0],  // 0
            [34.0, 16.0, 43.0], // 1
            [52.0, 43.0, 25.0], // 2
        ]);
        mat_approx_eq(&tt.as_std_matrix(), &mat_expected, 1e-13);

        // Symmetric2D
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1, Rep::Symmetric2D, true).unwrap();
        let mut tt = Tensor2::new(Rep::Symmetric2D);
        t3_dot_vec(&mut tt, 0.5, &hh, &u);
        let mat_expected = Matrix::from(&[
            [7.0, 34.0, 0.0],  // 0
            [34.0, 16.0, 0.0], // 1
            [0.0, 0.0, 25.0],  // 2
        ]);
        mat_approx_eq(&tt.as_std_matrix(), &mat_expected, 1e-13);
    }

    #[test]
    fn t3_dot_t2_works() {
        // General
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1, Rep::General, false).unwrap();
        let tt = Tensor2::from_std_matrix(
            &[
                [1.0, 2.0, 3.0], // 0
                [4.0, 5.0, 6.0], // 1
                [7.0, 8.0, 9.0], // 2
            ],
            Rep::General,
        )
        .unwrap();
        let mut u = Vector::new(3);
        t3_dot_t2(&mut u, 0.5, &hh, &tt);
        let vec_expected = Vector::from(&[328.5, 351.0, 373.5]);
        vec_approx_eq(&u, &vec_expected, 1e-15);

        // Symmetric
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1, Rep::Symmetric, false).unwrap();
        let tt = Tensor2::from_std_matrix(
            &[
                [1.0, 2.0, 3.0], // 0
                [2.0, 5.0, 6.0], // 1
                [3.0, 6.0, 9.0], // 2
            ],
            Rep::Symmetric,
        )
        .unwrap();
        let mut u = Vector::new(3);
        t3_dot_t2(&mut u, 0.5, &hh, &tt);
        let vec_expected = Vector::from(&[188.0, 206.5, 225.0]);
        vec_approx_eq(&u, &vec_expected, 1e-15);

        // Symmetric2D
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1, Rep::Symmetric2D, false).unwrap();
        let tt = Tensor2::from_std_matrix(
            &[
                [1.0, 2.0, 0.0], // 0
                [2.0, 5.0, 0.0], // 1
                [0.0, 0.0, 9.0], // 2
            ],
            Rep::Symmetric2D,
        )
        .unwrap();
        let mut u = Vector::new(3);
        t3_dot_t2(&mut u, 0.5, &hh, &tt);
        let vec_expected = Vector::from(&[62.0, 71.5, 81.0]);
        vec_approx_eq(&u, &vec_expected, 1e-15);
    }
}
