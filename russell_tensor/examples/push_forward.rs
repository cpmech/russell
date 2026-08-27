use russell_tensor::{Rep, StrError, Tensor2, t2_matmulx};

fn main() -> Result<(), StrError> {
    // Deformation gradient and Jacobian
    #[rustfmt::skip]
    let ff = Tensor2::from_std_matrix(&[
        [1.2, 0.1, 0.0],
        [0.2, 0.9, 0.0],
        [0.0, 0.0, 1.0],
    ], Rep::General)?;
    let jj = ff.determinant();

    // Second Piola-Kirchhoff stress
    #[rustfmt::skip]
    let ss = Tensor2::from_std_matrix(&[
        [10.0,  2.0, 0.0],
        [ 2.0, 20.0, 0.0],
        [ 0.0,  0.0, 5.0],
    ], Rep::Symmetric)?;

    // 1. Push-Forward: Material to Spatial
    let mut sigma = Tensor2::new(Rep::Symmetric);
    t2_matmulx(&mut sigma, 1.0 / jj, &ff, true, &ss)?;

    // 2. Pull-Back: Spatial to Material
    let mut ff_inv = Tensor2::new(Rep::General);
    ff.inverse(&mut ff_inv, 1e-10).unwrap();

    // Check
    let mut ss_back = Tensor2::new(Rep::Symmetric);
    t2_matmulx(&mut ss_back, jj, &ff_inv, false, &sigma)?;

    Ok(())
}
