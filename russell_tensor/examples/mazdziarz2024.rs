use russell_tensor::analysis::{self, VoigtReussHill};
use russell_tensor::{StrError, Tensor4};

// Calculate the Voigt-Reuss-Hill averages and the universal anisotropic index
// for the results presented in the following paper:
//
// 1. M. Maździarz, S. Nosewicz, Atomistic investigation of deformation and fracture
//    of individual structural components of metal matrix composites, Engineering Fracture
//    Mechanics 298 (2024) 109953. <https://doi:10.1016/j.engfracmech.2024.109953>.
//
// The output should be:
//
// NiAl Eq(11): Kh = 158.90, Gh = 64.37, Au = 3.92
// Al2O3 Eq(16): Kh = 242.15, Gh = 131.11, Au = 2.03
// Al2O3-NiAl Eq(19): Kh = 145.19, Gh = 84.19, Au = 1.45

fn main() -> Result<(), StrError> {
    // Eq (11) NiAl oriented X=[100] Y=[010] Z=[001]
    #[rustfmt::skip]
    let cc = Tensor4::<6>::from_matrix(&[
        [190.87, 142.91, 142.91,    0.0,    0.0,    0.0],
        [142.91, 190.87, 142.91,    0.0,    0.0,    0.0],
        [142.91, 142.91, 190.87,    0.0,    0.0,    0.0],
        [   0.0,    0.0,    0.0, 242.98,    0.0,    0.0],
        [   0.0,    0.0,    0.0,    0.0, 242.98,    0.0],
        [   0.0,    0.0,    0.0,    0.0,    0.0, 242.98],
    ])?;
    let mut ss = Tensor4::<6>::new();
    print("NiAl Eq(11)", &analysis::voigt_reuss_hill(&mut ss, &cc)?);
    // Expected output:
    // NiAl Eq(11): Kh = 158.90, Gh = 64.37, Au = 3.92

    // Eq (16) Al2O3 oriented X=[100] Y=[-1 Sqrt[3] 0] Z=[001]
    #[rustfmt::skip]
    let cc = Tensor4::<6>::from_matrix(&[
        [540.69,  186.42,  77.72,  86.394,    0.0,    0.0],
        [186.42,  540.69,  77.72, -86.394,    0.0,    0.0],
        [ 77.72,   77.72, 445.92,     0.0,    0.0,    0.0],
        [86.394, -86.394,    0.0,  192.58,    0.0,    0.0],
        [   0.0,     0.0,    0.0,     0.0, 192.58, 122.18],
        [   0.0,     0.0,    0.0,     0.0, 122.18, 354.26],
    ])?;
    let mut ss = Tensor4::<6>::new();
    print("Al2O3 Eq(16)", &analysis::voigt_reuss_hill(&mut ss, &cc)?);
    // Expected output:
    // Al2O3 Eq(16): Kh = 242.15, Gh = 131.11, Au = 2.03

    // Eq (19) Al2O3 - NiAl oriented X=[100] Y=[010] Z=[001]
    #[rustfmt::skip]
    let cc = Tensor4::<6>::from_matrix(&[
        [296.57,  144.76,   125.5, -49.879,  -3.535,  4.879],
        [144.76,  273.54,   74.42,  25.399,  -6.972,  1.937],
        [125.5,    74.42,  169.18, -55.677, -26.601, 13.364],
        [-49.879, 25.399, -55.677,  221.12,    0.04,   0.34],
        [-3.535,  -6.972, -26.601,    0.04,  226.06,  -62.3],
        [4.879,    1.937,  13.364,    0.34,   -62.3, 224.82],
    ])?;
    let mut ss = Tensor4::<6>::new();
    print("Al2O3-NiAl Eq(19)", &analysis::voigt_reuss_hill(&mut ss, &cc)?);
    // Expected output:
    // Al2O3-NiAl Eq(19): Kh = 145.19, Gh = 84.19, Au = 1.45

    Ok(())
}

fn print(label: &str, vrh: &VoigtReussHill) {
    println!(
        "{}: Kh = {:.2}, Gh = {:.2}, Au = {:.2}",
        label, vrh.kk_h, vrh.gg_h, vrh.aa_u
    );
}
