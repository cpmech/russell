use russell_lab::{approx_eq, mat_approx_eq, vec_approx_eq};
use russell_tensor::{SET, StrError, Tensor1, Tensor2, Tensor3};
use russell_tensor::{t2_add, t2_matmul, t3_ddot_t2, t3_dot_t1};

fn main() -> Result<(), StrError> {
    // Allocate a Tensor2
    let ten = Tensor2::<9>::from_std_matrix(&[
        [4.0, 2.0, 2.0], // 0
        [6.0, 2.0, 4.0], // 1
        [8.0, 4.0, 2.0], // 2
    ])
    .unwrap();
    let mat_ten = ten.as_std_matrix();
    println!("ten =\n{:.2}", mat_ten);

    // Determinant, transpose, and inverse
    let det = ten.determinant();
    let mut tra = Tensor2::<9>::new();
    let mut inv = Tensor2::<9>::new();
    ten.transpose(&mut tra);
    ten.inverse(&mut inv, 1e-15);
    let mat_tra = tra.as_std_matrix();
    let mat_inv = inv.as_std_matrix();
    println!("det = {:.2}", det);
    println!("tra =\n{:.2}", mat_tra);
    println!("inv =\n{:.2}", mat_inv);
    let correct_inv = [[-1.5, 0.5, 0.5], [2.5, -1.0, -0.5], [1.0, 0.0, -0.5]];
    let correct_tra = [[4.0, 6.0, 8.0], [2.0, 2.0, 4.0], [2.0, 4.0, 2.0]];
    approx_eq(det, 8.0, 1e-13);
    mat_approx_eq(&mat_inv, &correct_inv, 1e-14);
    mat_approx_eq(&mat_tra, &correct_tra, 1e-14);

    // Check the inverse
    let mut inv_dot_ten = Tensor2::<9>::new();
    t2_matmul(&mut inv_dot_ten, 1.0, &inv, false, &ten, false)?;
    let mat_inv_dot_ten = inv_dot_ten.as_std_matrix();
    println!("inv_dot_ten =\n{:.2}", mat_inv_dot_ten);
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    mat_approx_eq(&mat_inv_dot_ten, &identity, 1e-14);

    // Squared tensor
    let mut ten2 = Tensor2::<9>::new();
    ten.squared(&mut ten2);
    let mat_ten2 = ten2.as_std_matrix();
    println!("ten2 =\n{:.2}", mat_ten2);
    let correct_ten2 = [[44.0, 20.0, 20.0], [68.0, 32.0, 28.0], [72.0, 32.0, 36.0]];
    mat_approx_eq(&mat_ten2, &correct_ten2, 1e-12);

    // Tensor to the cubic power
    let mut ten3 = Tensor2::<9>::new();
    t2_matmul(&mut ten3, 1.0, &ten, false, &ten2, false)?;
    let mat_ten3 = ten3.as_std_matrix();
    println!("ten3 =\n{:.2}", mat_ten3);
    let correct_ten3 = [[456.0, 208.0, 208.0], [688.0, 312.0, 320.0], [768.0, 352.0, 344.0]];
    mat_approx_eq(&mat_ten3, &correct_ten3, 1e-12);

    // Check the determinant
    let t = ten.trace();
    let tt = ten2.trace();
    let ttt = ten3.trace();
    let expected_det = (2.0 * ttt - 3.0 * t * tt + t * t * t) / 6.0;
    println!("expected_det = {}", expected_det);
    approx_eq(det, expected_det, 1e-13);

    // Symmetric and skew-symmetric parts
    let mut sym = Tensor2::<9>::new();
    let mut skw = Tensor2::<9>::new();
    ten.decompose(&mut sym, &mut skw);
    let mat_sym = sym.as_std_matrix();
    let mat_skw = skw.as_std_matrix();
    println!("sym =\n{:.2}", mat_sym);
    println!("skw =\n{:.2}", mat_skw);
    let correct_sym = [[4.0, 4.0, 5.0], [4.0, 2.0, 4.0], [5.0, 4.0, 2.0]];
    let correct_skw = [[0.0, -2.0, -3.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    mat_approx_eq(&mat_sym, &correct_sym, 1e-15);
    mat_approx_eq(&mat_skw, &correct_skw, 1e-15);

    // Calculate the axial vector
    let mut omega = Tensor1::new();
    ten.axial_vector(&mut omega);
    println!("omega = \n{:.2}", omega);
    let expected_omega = [0.0, -3.0, 2.0];
    vec_approx_eq(&omega.as_vector(), &expected_omega, 1e-15);

    // Verify: det(I + W) = 1 + om . om
    let ii = Tensor2::<9>::identity();
    let mut ii_plus_skw = Tensor2::<9>::new();
    t2_add(&mut ii_plus_skw, 1.0, &ii, 1.0, &skw);
    let det_ii_plus_skw = ii_plus_skw.determinant();
    let om_dot_om_plus_1 = omega.dot(&omega) + 1.0;
    println!("det(I + W) = {} ({})", det_ii_plus_skw, om_dot_om_plus_1);
    approx_eq(det_ii_plus_skw, om_dot_om_plus_1, 1e-15);

    // Levi-Civita (permutation) tensor (Case A)
    let perm_a = Tensor3::<9, 3>::constant_permutation();
    let mat_perm_a = perm_a.as_std_matrix();
    println!("perm_a =\n{:.2}", mat_perm_a);

    // Calculate: skw = -perm . om
    let mut skw_again = Tensor2::<9>::new();
    t3_dot_t1(&mut skw_again, SET, -1.0, &perm_a, &omega);
    let mat_skw_again = skw_again.as_std_matrix();
    println!("skw_again =\n{:.2}", mat_skw_again);
    mat_approx_eq(&mat_skw_again, &correct_skw, 1e-15);

    // Levi-Civita (permutation) tensor (Case B)
    let perm_b = Tensor3::<3, 9>::constant_permutation();
    let mat_perm_b = perm_b.as_std_matrix();
    println!("perm_b =\n{:.2}", mat_perm_b);

    // Calculate: om = - (1/2) perm : skw
    let mut om_again = Tensor1::new();
    t3_ddot_t2(&mut om_again, SET, -0.5, &perm_b, &skw);
    println!("omega_again = \n{:.2}", om_again);
    vec_approx_eq(&om_again.as_vector(), &expected_omega, 1e-15);

    // Verify that 0 = perm : sym
    let mut zero = Tensor1::new();
    t3_ddot_t2(&mut zero, SET, 1.0, &perm_b, &sym);
    vec_approx_eq(&zero.as_vector(), &[0.0, 0.0, 0.0], 1e-15);

    // Verify that omega = -(1/2) perm : ten
    let mut om_again2 = Tensor1::new();
    t3_ddot_t2(&mut om_again2, SET, -0.5, &perm_b, &ten);
    println!("omega_again2 = \n{:.2}", om_again2);
    vec_approx_eq(&om_again2.as_vector(), &expected_omega, 1e-15);

    Ok(())
}
