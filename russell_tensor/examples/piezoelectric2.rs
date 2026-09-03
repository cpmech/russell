use russell_tensor::StrError;
use russell_tensor::analysis::*;
use russell_tensor::{ADD, SET, Tensor1, Tensor2};
use russell_tensor::{t1_dot_t3, t2_dot_t1, t3_ddot_t2, t4_ddot_t2};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), StrError> {
    // get the asset's full path (the JSON file is in the crate's data/ directory)
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let full_path = root.join("data/piezo_data.json");

    let db = PiezoDatabase::from_file(&full_path).unwrap();
    let mat = db.get("mp-774922")?;
    println!("{}", mat.info());
    let (p, e, cc) = mat.moduli()?;
    println!("Dielectric permittivity =\n{:.4}", p);
    println!("Piezoelectric tensor (C/m^2) =\n{:.4}", e);
    println!("Stiffness tensor (GPa) =\n{:.4}", cc);

    // Small strain tensor. TODO: find correct value range
    let eps = Tensor2::<6>::from_std_matrix(&[
        [0.001, 0.0005, 0.0], // 1
        [0.0005, 0.002, 0.0], // 2
        [0.0, 0.0, 0.001],    // 3
    ])?;

    // Electric field. TODO: find correct value range
    let ee = Tensor1::from(&[0.1, 0.1, 0.1]);

    // Compute stress: sig = cc : eps - ee . e
    let mut sig = Tensor2::<6>::new();
    t4_ddot_t2(&mut sig, SET, 1.0, &cc, &eps); // sig = cc : eps
    t1_dot_t3(&mut sig, ADD, -1.0, &ee, &e); // sig += -ee . e

    // Compute electric displacement: dd = e . eps + p . ee
    let mut dd = Tensor1::new();
    t3_ddot_t2(&mut dd, SET, 1.0, &e, &eps); // dd = e . eps
    t2_dot_t1(&mut dd, ADD, 1.0, &p, &ee); // dd += p . ee

    println!("sig =\n{}", sig);
    println!("D =\n{}", dd);

    Ok(())
}
