use super::{Tensor1, Tensor2, Tensor3};

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
    assert_eq!(b.rep(), a.rep());
    assert_eq!(c.rep(), a.rep());
    assert_eq!(b.is_case_a(), a.is_case_a());
    assert_eq!(c.is_case_a(), a.is_case_a());
    for i in 0..a.dims().0 {
        for j in 0..a.dims().1 {
            c.set(i, j, alpha * a.get(i, j) + beta * b.get(i, j));
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
/// Or, in Kelvin-Mandel basis:
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
/// * `u` -- the 3D vector (first-order tensor)
///
/// # Panics
///
/// 1. If `H` was not allocated for Case A
/// 2. If `T` and `H` have different [Rep]
/// 3. If `u` does not have 3 components
pub fn t3_dot_t1(tt: &mut Tensor2, alpha: f64, hh: &Tensor3, u: &Tensor1) {
    assert!(hh.is_case_a());
    assert_eq!(tt.rep(), hh.rep());
    for m in 0..hh.dims().0 {
        tt.vec[m] = alpha * (hh.get(m, 0) * u.get(0) + hh.get(m, 1) * u.get(1) + hh.get(m, 2) * u.get(2));
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
/// Or, in Kelvin-Mandel basis:
///
/// ```text
/// uᵢ = α Σ Hᵢₙ Tₙ
///       n
/// ```
///
/// # Output
///
/// * `u` -- the 3D vector (first-order tensor)
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
pub fn t3_ddot_t2(u: &mut Tensor1, alpha: f64, hh: &Tensor3, tt: &Tensor2) {
    assert!(!hh.is_case_a());
    assert_eq!(tt.rep(), hh.rep());
    u.set(0, 0.0);
    u.set(1, 0.0);
    u.set(2, 0.0);
    for n in 0..hh.dims().1 {
        u.set(0, u.get(0) + alpha * hh.get(0, n) * tt.vec[n]);
        u.set(1, u.get(1) + alpha * hh.get(1, n) * tt.vec[n]);
        u.set(2, u.get(2) + alpha * hh.get(2, n) * tt.vec[n]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{t3_add, t3_ddot_t2, t3_dot_t1};
    use crate::{Rep, SamplesTensor3, Tensor1, Tensor2, Tensor3};
    use russell_lab::{Matrix, approx_eq, mat_approx_eq};

    #[test]
    fn t3_add_works_case_a() {
        // General
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1, Rep::General, true).unwrap();
        let mut mm = Tensor3::new(Rep::General, true);
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::from_std_array(
            &[
                [[2.5, 5.0, 7.5], [25.0, 27.5, 30.0], [40.0, 42.5, 45.0]],
                [[47.5, 50.0, 52.5], [10.0, 12.5, 15.0], [32.5, 35.0, 37.5]],
                [[62.5, 65.0, 67.5], [55.0, 57.5, 60.0], [17.5, 20.0, 22.5]],
            ],
            Rep::General,
            true,
        )
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);

        // Symmetric
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1, Rep::Symmetric, true).unwrap();
        let mut mm = Tensor3::new(Rep::Symmetric, true);
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::from_std_array(
            &[
                [[2.5, 5.0, 7.5], [25.0, 27.5, 30.0], [40.0, 42.5, 45.0]],
                [[25.0, 27.5, 30.0], [10.0, 12.5, 15.0], [32.5, 35.0, 37.5]],
                [[40.0, 42.5, 45.0], [32.5, 35.0, 37.5], [17.5, 20.0, 22.5]],
            ],
            Rep::Symmetric,
            true,
        )
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);

        // Symmetric2D
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1, Rep::Symmetric2D, true).unwrap();
        let mut mm = Tensor3::new(Rep::Symmetric2D, true);
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::from_std_array(
            &[
                [[2.5, 5.0, 7.5], [25.0, 27.5, 30.0], [0.0, 0.0, 0.0]],
                [[25.0, 27.5, 30.0], [10.0, 12.5, 15.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [17.5, 20.0, 22.5]],
            ],
            Rep::Symmetric2D,
            true,
        )
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);
    }

    #[test]
    fn t3_add_works_case_b() {
        // General
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1, Rep::General, false).unwrap();
        let mut mm = Tensor3::new(Rep::General, false);
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::from_std_array(
            &[
                [[2.5, 25.0, 40.0], [47.5, 10.0, 32.5], [62.5, 55.0, 17.5]],
                [[5.0, 27.5, 42.5], [50.0, 12.5, 35.0], [65.0, 57.5, 20.0]],
                [[7.5, 30.0, 45.0], [52.5, 15.0, 37.5], [67.5, 60.0, 22.5]],
            ],
            Rep::General,
            false,
        )
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);

        // Symmetric
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1, Rep::Symmetric, false).unwrap();
        let mut mm = Tensor3::new(Rep::Symmetric, false);
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::from_std_array(
            &[
                [[2.5, 25.0, 40.0], [25.0, 10.0, 32.5], [40.0, 32.5, 17.5]],
                [[5.0, 27.5, 42.5], [27.5, 12.5, 35.0], [42.5, 35.0, 20.0]],
                [[7.5, 30.0, 45.0], [30.0, 15.0, 37.5], [45.0, 37.5, 22.5]],
            ],
            Rep::Symmetric,
            false,
        )
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);

        // Symmetric2D
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1, Rep::Symmetric2D, false).unwrap();
        let mut mm = Tensor3::new(Rep::Symmetric2D, false);
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::from_std_array(
            &[
                [[2.5, 25.0, 0.0], [25.0, 10.0, 0.0], [0.0, 0.0, 17.5]],
                [[5.0, 27.5, 0.0], [27.5, 12.5, 0.0], [0.0, 0.0, 20.0]],
                [[7.5, 30.0, 0.0], [30.0, 15.0, 0.0], [0.0, 0.0, 22.5]],
            ],
            Rep::Symmetric2D,
            false,
        )
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);
    }

    #[test]
    fn t3_dot_t1_works() {
        // General
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1, Rep::General, true).unwrap();
        let u = Tensor1::from(&[1.0, 2.0, 3.0]);
        let mut tt = Tensor2::new(Rep::General);
        t3_dot_t1(&mut tt, 0.5, &hh, &u);
        let mat_expected = Matrix::from(&[
            [7.0, 34.0, 52.0],  // 0
            [61.0, 16.0, 43.0], // 1
            [79.0, 70.0, 25.0], // 2
        ]);
        mat_approx_eq(&tt.as_std_matrix(), &mat_expected, 1e-13);

        // Symmetric
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1, Rep::Symmetric, true).unwrap();
        let mut tt = Tensor2::new(Rep::Symmetric);
        t3_dot_t1(&mut tt, 0.5, &hh, &u);
        let mat_expected = Matrix::from(&[
            [7.0, 34.0, 52.0],  // 0
            [34.0, 16.0, 43.0], // 1
            [52.0, 43.0, 25.0], // 2
        ]);
        mat_approx_eq(&tt.as_std_matrix(), &mat_expected, 1e-13);

        // Symmetric2D
        let hh = Tensor3::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1, Rep::Symmetric2D, true).unwrap();
        let mut tt = Tensor2::new(Rep::Symmetric2D);
        t3_dot_t1(&mut tt, 0.5, &hh, &u);
        let mat_expected = Matrix::from(&[
            [7.0, 34.0, 0.0],  // 0
            [34.0, 16.0, 0.0], // 1
            [0.0, 0.0, 25.0],  // 2
        ]);
        mat_approx_eq(&tt.as_std_matrix(), &mat_expected, 1e-13);
    }

    #[test]
    fn t3_ddot_t2_works() {
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
        let mut u = Tensor1::new();
        t3_ddot_t2(&mut u, 0.5, &hh, &tt);
        approx_eq(u.get(0), 328.5, 1e-15);
        approx_eq(u.get(1), 351.0, 1e-15);
        approx_eq(u.get(2), 373.5, 1e-15);

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
        let mut u = Tensor1::new();
        t3_ddot_t2(&mut u, 0.5, &hh, &tt);
        approx_eq(u.get(0), 188.0, 1e-15);
        approx_eq(u.get(1), 206.5, 1e-15);
        approx_eq(u.get(2), 225.0, 1e-15);

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
        let mut u = Tensor1::new();
        t3_ddot_t2(&mut u, 0.5, &hh, &tt);
        approx_eq(u.get(0), 62.0, 1e-15);
        approx_eq(u.get(1), 71.5, 1e-15);
        approx_eq(u.get(2), 81.0, 1e-15);
    }
}
