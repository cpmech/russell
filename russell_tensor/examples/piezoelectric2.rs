use russell_tensor::analysis::PiezoDatabase;
use russell_tensor::{ADD, SET, StrError, Tensor1, Tensor2};
use russell_tensor::{t1_dot_t3, t2_dot_t1, t3_ddot_t2, t4_ddot_t2};
use std::env;
use std::path::PathBuf;

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
