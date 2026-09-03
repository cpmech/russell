use russell_lab::format_scientific;
use russell_tensor::analysis::PiezoDatabase;
use russell_tensor::{ADD, SET, StrError, Tensor1, Tensor2};
use russell_tensor::{t1_dot_t3, t2_dot_t1, t3_ddot_t2, t4_ddot_t2};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

// Calculates:
// σ = C:ε − E·e   [Pa = (Pa)(−) − (C/m²)(V/m) = N/m²]
// D = e:ε + p·E   [C/m² = (C/m²)(−) + (F/m)(V/m) = C/m²]
//
// Outputs the norms of σ and D
//
// Expected results:
// Material    ||sig|| [MPa]  ||D|| [C/m^2]
// AlN             4.646E+01      2.689E-04
// ZnO             2.878E+01      2.309E-04
// LiNbO3          2.754E+01      3.559E-04
// LiTaO3          3.588E+01      2.121E-04
// BaTiO3          2.856E+01      2.370E-04
// SiO2            5.800E+00      3.438E-05
// TiPbO3          6.696E+00      5.005E-04
// Ti3TeO8         2.240E+00      1.234E-04

fn main() -> Result<(), StrError> {
    // Get the asset's full path (the JSON file is in the crate's data/ directory)
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let full_path = root.join("data/piezo_data.json");

    // Read piezoelectric material from database
    let db = PiezoDatabase::from_file(&full_path).unwrap();

    // Small strain tensor [-]
    let eps = Tensor2::<6>::from_std_matrix(&[
        [0.0, 0.0, 0.0],     // 1
        [0.0, 0.0, 0.0],     // 2
        [0.0, 0.0, 1.25e-4], // 3
    ])?;

    // Electric field [V/m]
    let ee = Tensor1::from(&[0.0, 0.0, 1e6]);

    // Allocate stress and electric displacement tensors
    let mut sig = Tensor2::<6>::new();
    let mut dd = Tensor1::new();

    // Record results for each material
    let mut norm_sig = HashMap::new();
    let mut norm_dd = HashMap::new();

    // Loop over materials
    for id in ["661", "2133", "3731", "3666", "5020", "6945", "20459", "774922"] {
        // Get the material
        let mat = db.get(&format!("mp-{}", id))?;
        let (p, e, cc) = mat.moduli()?;

        // Stress [Pa] sig = C : eps - E . e
        t4_ddot_t2(&mut sig, SET, 1.0, &cc, &eps); // sig = C : eps
        t1_dot_t3(&mut sig, ADD, -1.0, &ee, &e); // sig += -E . e

        // Electric displacement [C/m^2] D = e . eps + p . E
        t3_ddot_t2(&mut dd, SET, 1.0, &e, &eps); // D = e . eps
        t2_dot_t1(&mut dd, ADD, 1.0, &p, &ee); // D += p . E

        // Save results
        norm_sig.insert(id, sig.norm());
        norm_dd.insert(id, dd.norm());

        // Print the results
        println!("\n{}", "=".repeat(80));
        println!("{}", mat.info());
        print!("{}", p.scientific("Permittivity [F/m] p", 1.0, 10, 2));
        print!("{}", e.scientific("Piezoelectric tensor [C/m^2] e", 1.0, 10, 2));
        print!("{}", cc.scientific("Stiffness tensor [GPa] C", 1e-9, 10, 2));
        print!("{}", eps.scientific("Small strain tensor [-] eps", 1.0, 10, 2));
        print!("{}", ee.scientific("Electric field [V/m] E", 1.0, 10, 2));
        print!("{}", sig.scientific("Stress [kPa] sig", 1e-3, 10, 2));
        print!("{}", dd.scientific("Electric displacement [C/m^2] D", 1.0, 10, 2));
    }

    // Analysis
    println!("\n{}", "=".repeat(80));
    println!("{:<10}{:>15}{:>15}", "Material", "||sig|| [MPa]", "||D|| [C/m^2]");
    for id in ["661", "2133", "3731", "3666", "5020", "6945", "20459", "774922"] {
        let mat = db.get(&format!("mp-{}", id))?;
        println!(
            "{:<10}{}{}",
            mat.formula,
            format_scientific(*norm_sig.get(&id).unwrap() * 1e-6, 15, 3),
            format_scientific(*norm_dd.get(&id).unwrap(), 15, 3),
        );
    }

    Ok(())
}
