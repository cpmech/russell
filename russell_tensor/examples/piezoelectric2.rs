use russell_tensor::analysis::PiezoDatabase;
use russell_tensor::{ADD, SET, StrError, Tensor1, Tensor2};
use russell_tensor::{t1_dot_t3, t2_dot_t1, t3_ddot_t2, t4_ddot_t2};
use std::env;
use std::path::PathBuf;

// Calculates:
// σ = C:ε − E·e [Pa = (Pa)(−) − (C/m²)(V/m) = N/m²]
// D = e:ε + p·E [C/m² = (C/m²)(−) + (F/m)(V/m) = C/m²]
//
// Expected output:
//
// Ti3TeO8 (mp-774922) : Crystal System Triclinic : Space Group P1
// Permittivity [F/m] p =
// ┌           ┐
// │  1.05E-10 │
// │  8.69E-11 │
// │  1.09E-10 │
// │  4.70E-11 │
// │  3.98E-11 │
// │  5.48E-11 │
// └           ┘
// Piezoelectric tensor [C/m^2] e =
// ┌                                                             ┐
// │ -3.94E-01 -1.31E-01 -2.00E-01  2.99E-01 -2.73E-01  1.57E-01 │
// │  1.39E-01  1.60E-01  2.52E-01  1.89E-02 -7.70E-02  2.27E-01 │
// │ -4.07E-02 -5.02E-03 -1.32E-02  2.14E-02  2.10E-01  3.00E-02 │
// └                                                             ┘
// Stiffness tensor [GPa] C =
// ┌                                                             ┐
// │  2.50E+01  3.00E+00 -3.00E+00  7.07E+00  1.41E+00  1.41E+00 │
// │  3.00E+00  2.30E+01 -1.00E+00  8.49E+00  4.24E+00 -0.00E+00 │
// │ -3.00E+00 -1.00E+00  1.70E+01  2.83E+00  4.24E+00  2.83E+00 │
// │  7.07E+00  8.49E+00  2.83E+00  2.60E+01  6.00E+00  6.00E+00 │
// │  1.41E+00  4.24E+00  4.24E+00  6.00E+00  4.00E+01  1.60E+01 │
// │  1.41E+00 -0.00E+00  2.83E+00  6.00E+00  1.60E+01  4.40E+01 │
// └                                                             ┘
// Small strain tensor [-] eps =
// ┌           ┐
// │  0.00E+00 │
// │  0.00E+00 │
// │  1.25E-04 │
// │  0.00E+00 │
// │  0.00E+00 │
// │  0.00E+00 │
// └           ┘
// Electric field [V/m] E =
// ┌           ┐
// │  0.00E+00 │
// │  0.00E+00 │
// │  1.00E+06 │
// └           ┘
// Stress [kPa] sig =
// ┌           ┐
// │ -3.34E+02 │
// │ -1.20E+02 │
// │  2.14E+03 │
// │  3.32E+02 │
// │  3.20E+02 │
// │  3.24E+02 │
// └           ┘
// Electric displacement [C/m^2] D =
// ┌           ┐
// │  1.38E-05 │
// │  5.97E-05 │
// │  1.07E-04 │
// └           ┘

fn main() -> Result<(), StrError> {
    // Get the asset's full path (the JSON file is in the crate's data/ directory)
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let full_path = root.join("data/piezo_data.json");

    // Read piezoelectric material from database
    let db = PiezoDatabase::from_file(&full_path).unwrap();
    let mat = db.get("mp-774922")?;
    let (p, e, cc) = mat.moduli()?;

    // Small strain tensor [-]
    let eps = Tensor2::<6>::from_std_matrix(&[
        [0.0, 0.0, 0.0],     // 1
        [0.0, 0.0, 0.0],     // 2
        [0.0, 0.0, 1.25e-4], // 3
    ])?;

    // Electric field [V/m]
    let ee = Tensor1::from(&[0.0, 0.0, 1e6]);

    // Stress [Pa] sig = C : eps - E . e
    let mut sig = Tensor2::<6>::new();
    t4_ddot_t2(&mut sig, SET, 1.0, &cc, &eps); // sig = C : eps
    t1_dot_t3(&mut sig, ADD, -1.0, &ee, &e); // sig += -E . e

    // Electric displacement [C/m^2] D = e . eps + p . E
    let mut dd = Tensor1::new();
    t3_ddot_t2(&mut dd, SET, 1.0, &e, &eps); // D = e . eps
    t2_dot_t1(&mut dd, ADD, 1.0, &p, &ee); // D += p . E

    // Print the results
    println!("{}", mat.info());
    print!("{}", p.scientific("Permittivity [F/m] p", 1.0, 10, 2));
    print!("{}", e.scientific("Piezoelectric tensor [C/m^2] e", 1.0, 10, 2));
    print!("{}", cc.scientific("Stiffness tensor [GPa] C", 1e-9, 10, 2));
    print!("{}", eps.scientific("Small strain tensor [-] eps", 1.0, 10, 2));
    print!("{}", ee.scientific("Electric field [V/m] E", 1.0, 10, 2));
    print!("{}", sig.scientific("Stress [kPa] sig", 1e-3, 10, 2));
    print!("{}", dd.scientific("Electric displacement [C/m^2] D", 1.0, 10, 2));

    Ok(())
}
