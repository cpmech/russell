use super::{Tensor1, Tensor2, Tensor3};
use crate::{ADD, SET};

/// Adds two third-order tensors
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
pub fn t3_add<const M: usize, const N: usize>(
    c: &mut Tensor3<M, N>,
    alpha: f64,
    a: &Tensor3<M, N>,
    beta: f64,
    b: &Tensor3<M, N>,
) {
    for i in 0..M {
        for j in 0..N {
            c.set(i, j, alpha * a.get(i, j) + beta * b.get(i, j));
        }
    }
}

/// Performs the single-dot operation between a Tensor3 and a Tensor1 resulting in a Tensor2 (Case A)
///
/// Note: Case A (ij-pairwise) is (M, 3) with M = 4,6,9
///
/// Computes:
///
/// ```text
/// ADD: T += α H · u  or  SET: T = α H · u
/// ```
///
/// With Cartesian components (example with SET):
///
/// ```text
/// Tᵢⱼ = α Σ Hᵢⱼₖ uₖ
///        k
/// ```
///
/// Or, in Kelvin-Mandel basis (example with SET):
///
/// ```text
/// Tₘ = α Σ Hₘₖ uₖ
///        k
/// ```
///
/// In matrix notation (KM basis), this operation corresponds to:
///
/// ```text
/// [T] = α [H] [u]
/// M×1     M×3 3×1
/// ```
///
/// # Output
///
/// * `tt` -- the resulting second-order tensor (T)
///
/// # Input
///
/// * `op` -- operation: ADD or SET
/// * `alpha` -- the `α` multiplier
/// * `hh` -- the third-order tensor (H)
/// * `u` -- the 3D vector (first-order tensor)
pub fn t3_dot_t1<const M: usize, const N: usize>(
    tt: &mut Tensor2<M>,
    op: u8,
    alpha: f64,
    hh: &Tensor3<M, N>,
    u: &Tensor1,
) {
    if op == ADD {
        for m in 0..M {
            tt.vec[m] += alpha * (hh.get(m, 0) * u.get(0) + hh.get(m, 1) * u.get(1) + hh.get(m, 2) * u.get(2));
        }
    } else {
        for m in 0..M {
            tt.vec[m] = alpha * (hh.get(m, 0) * u.get(0) + hh.get(m, 1) * u.get(1) + hh.get(m, 2) * u.get(2));
        }
    }
}

/// Performs the double-dot operation between a Tensor2 and a Tensor3 resulting in a Tensor1 (Case A)
///
/// Note: Case A (ij-pairwise) is (M, 3) with M = 4,6,9
///
/// Computes:
///
/// ```text
/// ADD: u += α T : H   or   SET: u = α T : H
/// ```
///
/// With Cartesian components (example with SET):
///
/// ```text
/// uₖ = α Σ Σ Tᵢⱼ Hᵢⱼₖ
///        i j
/// ```
///
/// Or, in Kelvin-Mandel basis (example with SET):
///
/// ```text
/// uₖ = α Σ Tₘ Hₘₖ
///        m
/// ```
///
/// In matrix notation (KM basis), this operation corresponds to:
///
/// ```text
/// [u] = α [H]ᵀ [T]
/// 3×1     3×M  M×1
/// ```
///
/// # Output
///
/// * `u` -- the 3D vector (first-order tensor)
///
/// # Input
///
/// * `op` -- operation: ADD or SET
/// * `alpha` -- the `α` multiplier
/// * `tt` -- the second-order tensor (T)
/// * `hh` -- the third-order tensor (H)
pub fn t2_ddot_t3<const M: usize, const N: usize>(
    u: &mut Tensor1,
    op: u8,
    alpha: f64,
    tt: &Tensor2<M>,
    hh: &Tensor3<M, N>,
) {
    if op == SET {
        u.set(0, 0.0);
        u.set(1, 0.0);
        u.set(2, 0.0);
    }
    for m in 0..M {
        u.set(0, u.get(0) + alpha * tt.get(m) * hh.get(m, 0));
        u.set(1, u.get(1) + alpha * tt.get(m) * hh.get(m, 1));
        u.set(2, u.get(2) + alpha * tt.get(m) * hh.get(m, 2));
    }
}

/// Performs the double-dot operation between a Tensor3 and a Tensor2 resulting in a vector (Case B)
///
/// Note: Case B (jk-pairwise) is (3, N) with N = 4,6,9
///
/// Computes:
///
/// ```text
/// ADD: u += α H : T  or  SET: u = α H : T
/// ```
///
/// With Cartesian components (example with SET):
///
/// ```text
/// uᵢ = α Σ Σ Hᵢⱼₖ Tⱼₖ
///       j k
/// ```
///
/// Or, in Kelvin-Mandel basis (example with SET):
///
/// ```text
/// uᵢ = α Σ Hᵢₙ Tₙ
///       n
/// ```
///
/// In matrix notation (KM basis), this operation corresponds to:
///
/// ```text
/// [u] = α [H]  [T]
/// 3×1     3×N  N×1
/// ```
///
/// # Output
///
/// * `u` -- the 3D vector (first-order tensor)
///
/// # Input
///
/// * `op` -- operation: ADD or SET
/// * `alpha` -- the `α` multiplier
/// * `hh` -- the third-order tensor (H)
/// * `tt` -- the second-order tensor (T)
pub fn t3_ddot_t2<const M: usize, const N: usize>(
    u: &mut Tensor1,
    op: u8,
    alpha: f64,
    hh: &Tensor3<M, N>,
    tt: &Tensor2<N>,
) {
    if op == SET {
        u.set(0, 0.0);
        u.set(1, 0.0);
        u.set(2, 0.0);
    }
    for n in 0..N {
        u.set(0, u.get(0) + alpha * hh.get(0, n) * tt.vec[n]);
        u.set(1, u.get(1) + alpha * hh.get(1, n) * tt.vec[n]);
        u.set(2, u.get(2) + alpha * hh.get(2, n) * tt.vec[n]);
    }
}

/// Performs the single-dot operation between a Tensor3 and a vector resulting in a Tensor2 (Case B)
///
/// Note: Case B (jk-pairwise) is (3, N) with N = 4,6,9
///
/// Computes:
///
/// ```text
/// ADD: T += α u · H  or  SET: T = α u · H
/// ```
///
/// With Cartesian components (example with SET):
///
/// ```text
/// Tⱼₖ = α Σ uᵢ Hᵢⱼₖ
///         i
/// ```
///
/// Or, in Kelvin-Mandel basis (example with SET):
///
/// ```text
/// Tₙ = α Σ uᵢ Hᵢₙ
///        i
/// ```
///
/// In matrix notation (KM basis), this operation corresponds to:
///
/// ```text
/// [T] = α [H]ᵀ [u]
/// N×1     N×3  3×1
/// ```
///
/// # Output
///
/// * `tt` -- the resulting second-order tensor (T)
///
/// # Input
///
/// * `op` -- operation: ADD or SET
/// * `alpha` -- the `α` multiplier
/// * `u` -- the 3D vector (first-order tensor)
/// * `hh` -- the third-order tensor (H)
pub fn t1_dot_t3<const M: usize, const N: usize>(
    tt: &mut Tensor2<N>,
    op: u8,
    alpha: f64,
    u: &Tensor1,
    hh: &Tensor3<M, N>,
) {
    if op == ADD {
        for n in 0..N {
            tt.vec[n] += alpha * (u.get(0) * hh.get(0, n) + u.get(1) * hh.get(1, n) + u.get(2) * hh.get(2, n));
        }
    } else {
        for n in 0..N {
            tt.vec[n] = alpha * (u.get(0) * hh.get(0, n) + u.get(1) * hh.get(1, n) + u.get(2) * hh.get(2, n));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{t1_dot_t3, t2_ddot_t3, t3_add, t3_ddot_t2, t3_dot_t1};
    use crate::{ADD, SET, SamplesTensor3, Tensor1, Tensor2, Tensor3};
    use russell_lab::{Matrix, approx_eq, mat_approx_eq, mat_mat_mul, mat_t_mat_mul};

    #[test]
    fn t3_add_works_case_a() {
        // General
        let hh = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        let mut mm = Tensor3::<9, 3>::new();
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::<9, 3>::from_std_array(&[
            [[2.5, 5.0, 7.5], [25.0, 27.5, 30.0], [40.0, 42.5, 45.0]],
            [[47.5, 50.0, 52.5], [10.0, 12.5, 15.0], [32.5, 35.0, 37.5]],
            [[62.5, 65.0, 67.5], [55.0, 57.5, 60.0], [17.5, 20.0, 22.5]],
        ])
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);

        // Symmetric
        let hh = Tensor3::<6, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1).unwrap();
        let mut mm = Tensor3::<6, 3>::new();
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::<6, 3>::from_std_array(&[
            [[2.5, 5.0, 7.5], [25.0, 27.5, 30.0], [40.0, 42.5, 45.0]],
            [[25.0, 27.5, 30.0], [10.0, 12.5, 15.0], [32.5, 35.0, 37.5]],
            [[40.0, 42.5, 45.0], [32.5, 35.0, 37.5], [17.5, 20.0, 22.5]],
        ])
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);

        // Symmetric2D
        let hh = Tensor3::<4, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1).unwrap();
        let mut mm = Tensor3::<4, 3>::new();
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::<4, 3>::from_std_array(&[
            [[2.5, 5.0, 7.5], [25.0, 27.5, 30.0], [0.0, 0.0, 0.0]],
            [[25.0, 27.5, 30.0], [10.0, 12.5, 15.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [17.5, 20.0, 22.5]],
        ])
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);
    }

    #[test]
    fn t3_add_works_case_b() {
        // General
        let hh = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        let mut mm = Tensor3::<3, 9>::new();
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::<3, 9>::from_std_array(&[
            [[2.5, 25.0, 40.0], [47.5, 10.0, 32.5], [62.5, 55.0, 17.5]],
            [[5.0, 27.5, 42.5], [50.0, 12.5, 35.0], [65.0, 57.5, 20.0]],
            [[7.5, 30.0, 45.0], [52.5, 15.0, 37.5], [67.5, 60.0, 22.5]],
        ])
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);

        // Symmetric
        let hh = Tensor3::<3, 6>::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1).unwrap();
        let mut mm = Tensor3::<3, 6>::new();
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::<3, 6>::from_std_array(&[
            [[2.5, 25.0, 40.0], [25.0, 10.0, 32.5], [40.0, 32.5, 17.5]],
            [[5.0, 27.5, 42.5], [27.5, 12.5, 35.0], [42.5, 35.0, 20.0]],
            [[7.5, 30.0, 45.0], [30.0, 15.0, 37.5], [45.0, 37.5, 22.5]],
        ])
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);

        // Symmetric2D
        let hh = Tensor3::<3, 4>::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1).unwrap();
        let mut mm = Tensor3::<3, 4>::new();
        t3_add(&mut mm, 0.5, &hh, 2.0, &hh);
        let mm_expected = Tensor3::<3, 4>::from_std_array(&[
            [[2.5, 25.0, 0.0], [25.0, 10.0, 0.0], [0.0, 0.0, 17.5]],
            [[5.0, 27.5, 0.0], [27.5, 12.5, 0.0], [0.0, 0.0, 20.0]],
            [[7.5, 30.0, 0.0], [30.0, 15.0, 0.0], [0.0, 0.0, 22.5]],
        ])
        .unwrap();
        mat_approx_eq(&mm.as_std_matrix(), &mm_expected.as_std_matrix(), 1e-13);
    }

    #[test]
    fn t3_dot_t1_works() {
        // General
        let hh = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        let u = Tensor1::from(&[1.0, 2.0, 3.0]);
        let mut tt = Tensor2::<9>::new();
        t3_dot_t1(&mut tt, SET, 0.5, &hh, &u);
        let mat_expected = Matrix::from(&[
            [7.0, 34.0, 52.0],  // 0
            [61.0, 16.0, 43.0], // 1
            [79.0, 70.0, 25.0], // 2
        ]);
        mat_approx_eq(&tt.as_std_matrix(), &mat_expected, 1e-13);

        // Symmetric
        let hh = Tensor3::<6, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_SAMPLE1).unwrap();
        let mut tt = Tensor2::<6>::new();
        t3_dot_t1(&mut tt, SET, 0.5, &hh, &u);
        let mat_expected = Matrix::from(&[
            [7.0, 34.0, 52.0],  // 0
            [34.0, 16.0, 43.0], // 1
            [52.0, 43.0, 25.0], // 2
        ]);
        mat_approx_eq(&tt.as_std_matrix(), &mat_expected, 1e-13);

        // Symmetric2D
        let hh = Tensor3::<4, 3>::from_std_array(&SamplesTensor3::CASE_A_SYM_2D_SAMPLE1).unwrap();
        let mut tt = Tensor2::<4>::new();
        t3_dot_t1(&mut tt, SET, 0.5, &hh, &u);
        let mat_expected = Matrix::from(&[
            [7.0, 34.0, 0.0],  // 0
            [34.0, 16.0, 0.0], // 1
            [0.0, 0.0, 25.0],  // 2
        ]);
        mat_approx_eq(&tt.as_std_matrix(), &mat_expected, 1e-13);
    }

    #[test]
    fn t3_dot_t1_add_works() {
        // General
        let hh = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        let u = Tensor1::from(&[1.0, 2.0, 3.0]);
        let mut tt = Tensor2::<9>::from_std_matrix(&[
            [100.0, 0.0, 0.0], // 0
            [0.0, 200.0, 0.0], // 1
            [0.0, 0.0, 300.0], // 2
        ])
        .unwrap();
        t3_dot_t1(&mut tt, ADD, 0.5, &hh, &u);
        let mat_expected = Matrix::from(&[
            [107.0, 34.0, 52.0], // 0
            [61.0, 216.0, 43.0], // 1
            [79.0, 70.0, 325.0], // 2
        ]);
        mat_approx_eq(&tt.as_std_matrix(), &mat_expected, 1e-13);
    }

    #[test]
    fn t3_ddot_t2_works() {
        // General
        let hh = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        let tt = Tensor2::<9>::from_std_matrix(&[
            [1.0, 2.0, 3.0], // 0
            [4.0, 5.0, 6.0], // 1
            [7.0, 8.0, 9.0], // 2
        ])
        .unwrap();
        let mut u = Tensor1::new();
        t3_ddot_t2(&mut u, SET, 0.5, &hh, &tt);
        approx_eq(u.get(0), 328.5, 1e-15);
        approx_eq(u.get(1), 351.0, 1e-15);
        approx_eq(u.get(2), 373.5, 1e-15);

        // Symmetric
        let hh = Tensor3::<3, 6>::from_std_array(&SamplesTensor3::CASE_B_SYM_SAMPLE1).unwrap();
        let tt = Tensor2::<6>::from_std_matrix(&[
            [1.0, 2.0, 3.0], // 0
            [2.0, 5.0, 6.0], // 1
            [3.0, 6.0, 9.0], // 2
        ])
        .unwrap();
        let mut u = Tensor1::new();
        t3_ddot_t2(&mut u, SET, 0.5, &hh, &tt);
        approx_eq(u.get(0), 188.0, 1e-15);
        approx_eq(u.get(1), 206.5, 1e-15);
        approx_eq(u.get(2), 225.0, 1e-15);

        // Symmetric2D
        let hh = Tensor3::<3, 4>::from_std_array(&SamplesTensor3::CASE_B_SYM_2D_SAMPLE1).unwrap();
        let tt = Tensor2::<4>::from_std_matrix(&[
            [1.0, 2.0, 0.0], // 0
            [2.0, 5.0, 0.0], // 1
            [0.0, 0.0, 9.0], // 2
        ])
        .unwrap();
        let mut u = Tensor1::new();
        t3_ddot_t2(&mut u, SET, 0.5, &hh, &tt);
        approx_eq(u.get(0), 62.0, 1e-15);
        approx_eq(u.get(1), 71.5, 1e-15);
        approx_eq(u.get(2), 81.0, 1e-15);
    }

    #[test]
    fn t3_ddot_t2_add_works() {
        // General
        let hh = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        let tt = Tensor2::<9>::from_std_matrix(&[
            [1.0, 2.0, 3.0], // 0
            [4.0, 5.0, 6.0], // 1
            [7.0, 8.0, 9.0], // 2
        ])
        .unwrap();
        let mut u = Tensor1::from(&[100.0, 200.0, 300.0]);
        t3_ddot_t2(&mut u, ADD, 0.5, &hh, &tt);
        approx_eq(u.get(0), 428.5, 1e-13);
        approx_eq(u.get(1), 551.0, 1e-13);
        approx_eq(u.get(2), 673.5, 1e-13);
    }

    #[test]
    fn t2_ddot_t3_works() {
        // Case A general
        let hh = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        let tt = Tensor2::<9>::from_std_matrix(&[
            [1.0, 2.0, 3.0], // 0
            [4.0, 5.0, 6.0], // 1
            [7.0, 8.0, 9.0], // 2
        ])
        .unwrap();
        // reference: u_k = 0.5 Σᵢ Σⱼ Tᵢⱼ Hᵢⱼₖ
        let hh_std = hh.as_std_array();
        let tt_std = tt.as_std_matrix();
        let mut correct = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    correct[k] += 0.5 * tt_std.get(i, j) * hh_std[i][j][k];
                }
            }
        }
        // SET
        let mut u = Tensor1::new();
        t2_ddot_t3(&mut u, SET, 0.5, &tt, &hh);
        approx_eq(u.get(0), correct[0], 1e-12);
        approx_eq(u.get(1), correct[1], 1e-12);
        approx_eq(u.get(2), correct[2], 1e-12);
        // ADD
        let mut u = Tensor1::from(&[100.0, 200.0, 300.0]);
        t2_ddot_t3(&mut u, ADD, 0.5, &tt, &hh);
        approx_eq(u.get(0), 100.0 + correct[0], 1e-12);
        approx_eq(u.get(1), 200.0 + correct[1], 1e-12);
        approx_eq(u.get(2), 300.0 + correct[2], 1e-12);
    }

    #[test]
    fn t1_dot_t3_works() {
        // Case B general
        let hh = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        let u = Tensor1::from(&[1.0, 2.0, 3.0]);
        // reference: Tⱼₖ = 0.5 Σᵢ uᵢ Hᵢⱼₖ
        let hh_std = hh.as_std_array();
        let mut correct = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    correct[j][k] += 0.5 * u.get(i) * hh_std[i][j][k];
                }
            }
        }
        // SET
        let mut tt = Tensor2::<9>::new();
        t1_dot_t3(&mut tt, SET, 0.5, &u, &hh);
        mat_approx_eq(&tt.as_std_matrix(), &correct, 1e-12);
        // ADD
        let mut tt = Tensor2::<9>::from_std_matrix(&[
            [100.0, 0.0, 0.0], // 0
            [0.0, 200.0, 0.0], // 1
            [0.0, 0.0, 300.0], // 2
        ])
        .unwrap();
        t1_dot_t3(&mut tt, ADD, 0.5, &u, &hh);
        let correct_add = [
            [100.0 + correct[0][0], correct[0][1], correct[0][2]],
            [correct[1][0], 200.0 + correct[1][1], correct[1][2]],
            [correct[2][0], correct[2][1], 300.0 + correct[2][2]],
        ];
        mat_approx_eq(&tt.as_std_matrix(), &correct_add, 1e-12);
    }

    //
    // --- using matmul ---
    //

    // Returns the M x 1 Kelvin-Mandel matrix representing a Tensor2
    fn kelvin_matrix_t2<const M: usize>(tt: &Tensor2<M>) -> Matrix {
        let mut mat = Matrix::new(M, 1);
        for m in 0..M {
            mat.set(m, 0, tt.get(m));
        }
        mat
    }

    // Returns the M x N Kelvin-Mandel matrix representing a Tensor3
    fn kelvin_matrix_t3<const M: usize, const N: usize>(hh: &Tensor3<M, N>) -> Matrix {
        let mut mat = Matrix::new(M, N);
        for m in 0..M {
            for n in 0..N {
                mat.set(m, n, hh.get(m, n));
            }
        }
        mat
    }

    #[test]
    fn check_t3_dot_t1_using_matrix_notation() {
        // Case A
        // [T] = α [H] [u]
        // M×1     M×3 3×1
        // tensor form
        let hh_ten = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        let mut tt_ten = Tensor2::<9>::new();
        let u_ten = Tensor1::from(&[1.0, 2.0, 3.0]);
        t3_dot_t1(&mut tt_ten, SET, 1.0, &hh_ten, &u_ten);
        // matrix form
        let hh = kelvin_matrix_t3(&hh_ten);
        let u = Matrix::from(&[[1.0], [2.0], [3.0]]);
        let mut tt = Matrix::new(9, 1);
        mat_mat_mul(&mut tt, 1.0, &hh, &u, 0.0).unwrap();
        // check
        for m in 0..9 {
            approx_eq(tt[(m, 0)], tt_ten.get(m), 1e-13);
        }
    }

    #[test]
    fn check_t2_ddot_t3_using_matrix_notation() {
        // Case A
        // [u] = α [H]ᵀ [T]
        // 3×1     3×M  M×1
        // tensor form
        let hh_ten = Tensor3::<9, 3>::from_std_array(&SamplesTensor3::CASE_A_SAMPLE1).unwrap();
        let tt_ten = Tensor2::<9>::from_std_matrix(&[
            [1.0, 0.5, 0.1], // 0
            [0.4, 2.0, 0.3], // 1
            [0.2, 0.1, 3.0], // 2
        ])
        .unwrap();
        let mut u_ten = Tensor1::new();
        t2_ddot_t3(&mut u_ten, SET, 1.0, &tt_ten, &hh_ten);
        // matrix form
        let hh = kelvin_matrix_t3(&hh_ten);
        let tt = kelvin_matrix_t2(&tt_ten);
        let mut u = Matrix::new(3, 1);
        mat_t_mat_mul(&mut u, 1.0, &hh, &tt, 0.0).unwrap();
        // check
        for i in 0..3 {
            approx_eq(u[(i, 0)], u_ten.get(i), 1e-13);
        }
    }

    #[test]
    fn check_t3_ddot_t2_using_matrix_notation() {
        // Case B
        // [u] = α [H]  [T]
        // 3×1     3×N  N×1
        // tensor form
        let hh_ten = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        let tt_ten = Tensor2::<9>::from_std_matrix(&[
            [1.0, 0.5, 0.1], // 0
            [0.4, 2.0, 0.3], // 1
            [0.2, 0.1, 3.0], // 2
        ])
        .unwrap();
        let mut u_ten = Tensor1::new();
        t3_ddot_t2(&mut u_ten, SET, 1.0, &hh_ten, &tt_ten);
        // matrix form
        let hh = kelvin_matrix_t3(&hh_ten);
        let tt = kelvin_matrix_t2(&tt_ten);
        let mut u = Matrix::new(3, 1);
        mat_mat_mul(&mut u, 1.0, &hh, &tt, 0.0).unwrap();
        // check
        for i in 0..3 {
            approx_eq(u[(i, 0)], u_ten.get(i), 1e-14);
        }
    }

    #[test]
    fn check_t1_dot_t3_using_matrix_notation() {
        // Case B
        // [T] = α [H]ᵀ [u]
        // N×1     N×3  3×1
        // tensor form
        let hh_ten = Tensor3::<3, 9>::from_std_array(&SamplesTensor3::CASE_B_SAMPLE1).unwrap();
        let u_ten = Tensor1::from(&[1.0, 2.0, 3.0]);
        let mut tt_ten = Tensor2::<9>::new();
        t1_dot_t3(&mut tt_ten, SET, 1.0, &u_ten, &hh_ten);
        // matrix form
        let hh = kelvin_matrix_t3(&hh_ten);
        let u = Matrix::from(&[[1.0], [2.0], [3.0]]);
        let mut tt = Matrix::new(9, 1);
        mat_t_mat_mul(&mut tt, 1.0, &hh, &u, 0.0).unwrap();
        // check
        for m in 0..9 {
            approx_eq(tt[(m, 0)], tt_ten.get(m), 1e-13);
        }
    }
}
