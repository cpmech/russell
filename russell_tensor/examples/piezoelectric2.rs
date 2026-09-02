use russell_tensor::StrError;
use russell_tensor::analysis::*;
use russell_tensor::t1_dot_t3;
use russell_tensor::t2_dot_t1;
use russell_tensor::t3_ddot_t2;
use russell_tensor::t4_ddot_t2;
use russell_tensor::{Tensor1, Tensor2};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), StrError> {
    // get the asset's full path (the JSON file is in the crate's data/ directory)
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let full_path = root.join("data/piezo_data.json");

    let db = PiezoDatabase::from_file(&full_path).unwrap();
    println!("{}", db.info("mp-774922")?);
    let (per, e, d, cc, ss) = db.get_tensors("mp-774922")?;
    println!("Dielectric permittivity =\n{:.4}", per);
    println!("Piezoelectric tensor =\n{:.4}", e);
    println!("Piezoelectric charge tensor =\n{:.4}", d);
    println!("Stiffness tensor =\n{:.4}", cc);
    println!("Compliance tensor =\n{:.4}", ss);

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
    let mut res = Tensor2::<6>::new();
    t4_ddot_t2(&mut sig, 1.0, &cc, &eps);
    t1_dot_t3(&mut res, -1.0, &ee, &e); // TODO: create t1_dot_t3_update

    // Compute electric displacement: dd = e . eps + per . ee
    let mut dd = Tensor1::new();
    let mut tmp = Tensor1::new();
    t3_ddot_t2(&mut dd, 1.0, &e, &eps);
    t2_dot_t1(&mut tmp, 1.0, &per, &ee); // TODO: create t2_dot_t1_update

    Ok(())
}
