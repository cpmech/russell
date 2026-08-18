use russell_lab::{approx_eq, mat_approx_eq};
use russell_tensor::{Rep, StrError, Tensor1, Tensor2, Tensor3, t2_ddot_t2, t2_dot_t2};

fn main() -> Result<(), StrError> {
    // Allocate a Tensor2
    let ten = Tensor2::from_std_matrix(
        &[
            [4.0, 2.0, 2.0], // 0
            [6.0, 2.0, 4.0], // 1
            [8.0, 4.0, 2.0], // 2
        ],
        Rep::General,
    )
    .unwrap();
    let mat_ten = ten.as_std_matrix();
    println!("ten =\n{:.2}", mat_ten);

    // Determinant, transpose, and inverse
    let det = ten.determinant();
    let mut tra = Tensor2::new(ten.rep());
    let mut inv = Tensor2::new(ten.rep());
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

    // Squared tensor
    let mut ten2 = Tensor2::new(ten.rep());
    ten.squared(&mut ten2);
    let mat_ten2 = ten2.as_std_matrix();
    println!("ten2 =\n{:.2}", mat_ten2);
    let correct_ten2 = [[44.0, 20.0, 20.0], [68.0, 32.0, 28.0], [72.0, 32.0, 36.0]];
    mat_approx_eq(&mat_ten2, &correct_ten2, 1e-12);

    // Tensor to the power 3
    let mut ten3 = Tensor2::new(ten.rep());
    t2_dot_t2(&mut ten3, &ten, &ten2);
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
    let mut sym = Tensor2::new(Rep::General);
    let mut skw = Tensor2::new(Rep::General);
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
    let omega = Tensor1::from(&[-skw.get_std(1, 2), skw.get_std(0, 2), -skw.get_std(0, 1)]);
    println!("omega = \n{:.2}", omega);

    // Levi-Civita (permutation) tensor
    let perm = Tensor3::constant_permutation(Rep::General, true)?;
    println!("{}", perm);

    Ok(())
}
