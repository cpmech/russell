use russell_lab::Vector;
use russell_tensor::{StrError, Tensor2, Tensor4};
use russell_tensor::{analysis, t4_add};

// Calculate the eigenvalues and stability properties
// for the results presented in the following paper:
//
// 1. M. Maździarz (2025) Mechanical stability conditions for 3D and 2D crystals under arbitrary
//    load, Archives of Mechanics, 77(4), 379–399, 2025, <https://doi.org/10.24423/aom.4679>
//
// The output should be:
//
// H =
// ┌                                                 ┐
// │  27.060 -27.060 -23.822   0.000  -0.000   0.000 │
// │ -27.060  27.060 -23.822   0.000   0.000  -0.000 │
// │ -23.822 -23.822  20.585  -0.000   0.000   0.000 │
// │   0.000   0.000  -0.000  54.120   0.000   0.000 │
// │  -0.000   0.000   0.000   0.000  47.645   0.000 │
// │   0.000  -0.000   0.000   0.000   0.000  47.645 │
// └                                                 ┘
// L =
// ┌                                                 ┐
// │  28.239 -17.481 -20.041   0.000   0.000   0.000 │
// │ -17.481  28.239 -20.041   0.000   0.000   0.000 │
// │ -20.041 -20.041  74.400   0.000   0.000   0.000 │
// │   0.000   0.000   0.000 133.428   0.000   0.000 │
// │   0.000   0.000   0.000   0.000 106.656   0.000 │
// │   0.000   0.000   0.000   0.000   0.000 106.656 │
// └                                                 ┘
// lambda(C): 79.308, 59.011, 59.011, 54.469, 10.104, -8.400
// lambda(L): 133.428, 106.656, 106.656, 85.192, 45.720, -0.034

fn main() -> Result<(), StrError> {
    // Eq (B.3) NiAl oriented X=[100] Y=[010] Z=[001]
    #[rustfmt::skip]
    let cc = Tensor4::<6>::from_matrix(&[
        [1.179, 9.579,  3.781,    0.0,    0.0,    0.0],
        [9.579, 1.179,  3.781,    0.0,    0.0,    0.0],
        [3.781, 3.781, 53.815,    0.0,    0.0,    0.0],
        [  0.0,   0.0,    0.0, 79.308,    0.0,    0.0],
        [  0.0,   0.0,    0.0,    0.0, 59.011,    0.0],
        [  0.0,   0.0,    0.0,    0.0,    0.0, 59.011],
    ])?;

    // Internal stability tensor
    let mut hh = Tensor4::<6>::new();
    #[rustfmt::skip]
    let sigma = Tensor2::<6>::from_std_matrix(&[
        [27.06,   0.0,    0.0],
        [  0.0, 27.06,    0.0],
        [  0.0,   0.0, 20.585],
    ])?;
    analysis::internal_stability_tensor(&mut hh, &sigma)?;
    println!("H = \n{:.3}", hh);

    // Tangent modulus L
    let mut ll = Tensor4::<6>::new();
    t4_add(&mut ll, 1.0, &cc, 1.0, &hh);
    println!("L = \n{:.3}", ll);

    // Eigenvalues
    let mut lam_cc = Vector::new(6);
    let mut lam_ll = Vector::new(6);
    cc.eigenvalues_sym(&mut lam_cc)?;
    ll.eigenvalues_sym(&mut lam_ll)?;
    print("lambda(C)", &lam_cc);
    print("lambda(L)", &lam_ll);

    Ok(())
}

fn print(label: &str, lam: &Vector) {
    println!(
        "{}: {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}",
        label, lam[5], lam[4], lam[3], lam[2], lam[1], lam[0]
    );
}
