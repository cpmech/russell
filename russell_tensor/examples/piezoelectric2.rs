use russell_tensor::StrError;
use russell_tensor::analysis::*;
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), StrError> {
    // get the asset's full path (the JSON file is in the crate's data/ directory)
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let full_path = root.join("data/piezo_data.json");

    let db = PiezoDatabase::from_file(&full_path).unwrap();
    println!("{}", db.info("mp-774922")?);
    let (eps, e, d, cc, ss) = db.get_tensors("mp-774922")?;
    println!("Dielectric permittivity =\n{:.4}", eps);
    println!("Piezoelectric tensor =\n{:.4}", e);
    println!("Piezoelectric charge tensor =\n{:.4}", d);
    println!("Stiffness tensor =\n{:.4}", cc);
    println!("Compliance tensor =\n{:.4}", ss);

    Ok(())
}
