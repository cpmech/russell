use russell_lab::format_scientific;
use russell_tensor::analysis::PiezoDatabase;
use russell_tensor::{ADD, SET, StrError, Tensor1, Tensor2, Tensor3, Tensor4};
use russell_tensor::{t1_dot_t3, t2_dot_t1, t3_ddot_t2, t4_ddot_t2};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), StrError> {
    // get the asset's full path (the JSON file is in the crate's data/ directory)
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let full_path = root.join("data/piezo_data.json");

    let db = PiezoDatabase::from_file(&full_path).unwrap();
    let mat = db.get("mp-774922")?;
    let (p, e, cc) = mat.moduli()?;
    println!("{}", mat.info());
    print_t2("Permittivity p", &p);
    print_t3("Piezoelectric tensor [C/m^2] e", &e);
    print_t4("Stiffness tensor [GPa] C", &cc);

    // Small strain tensor (dimensionless)
    let eps = Tensor2::<6>::from_std_matrix(&[
        [0.0, 0.0, 0.0],     // 1
        [0.0, 0.0, 0.0],     // 2
        [0.0, 0.0, 1.25e-4], // 3
    ])?;

    // Electric field (V/m)
    let ee = Tensor1::from(&[0.0, 0.0, 1e6]);

    print_t2("Small strain tensor [-] eps", &eps);
    print_t1("Electric field [V/m] E", &ee);

    // Compute stress: sig = C : eps - E . e
    const TO_GPA: f64 = 1e-9;
    let mut sig = Tensor2::<6>::new();
    t4_ddot_t2(&mut sig, SET, 1.0, &cc, &eps); // sig = C : eps
    t1_dot_t3(&mut sig, ADD, -TO_GPA, &ee, &e); // sig += -E . e

    // Compute electric displacement: D = e . eps + p . E
    let mut dd = Tensor1::new();
    t3_ddot_t2(&mut dd, SET, 1.0, &e, &eps); // D = e . eps
    t2_dot_t1(&mut dd, ADD, 1.0, &p, &ee); // D += p . E

    print_t2("Stress [GPa] sig", &sig);
    print_t1("Electric displacement [C/m^2] D", &dd);

    Ok(())
}

fn print_t1(label: &str, v: &Tensor1) {
    let width = 11;
    println!("{} =", label);
    println!("┌{:1$}┐", " ", width + 1);
    for m in 0..3 {
        if m > 0 {
            println!(" │");
        }
        print!("│");
        let val = v.get(m);
        print!("{:>1$}", format_scientific(val, width, 3), width);
    }
    println!(" │");
    println!("└{:1$}┘", " ", width + 1);
}

fn print_t2(label: &str, p: &Tensor2<6>) {
    let width = 11;
    println!("{} =", label);
    println!("┌{:1$}┐", " ", width + 1);
    for m in 0..6 {
        if m > 0 {
            println!(" │");
        }
        print!("│");
        let val = p.get(m);
        print!("{:>1$}", format_scientific(val, width, 3), width);
    }
    println!(" │");
    println!("└{:1$}┘", " ", width + 1);
}

fn print_t3(label: &str, e: &Tensor3<3, 6>) {
    println!("{} =\n{:.3}", label, e);
}

fn print_t4(label: &str, c: &Tensor4<6>) {
    println!("{} =\n{:.3}", label, c);
}
