use russell_sparse::*;

fn main() -> Result<(), &'static str> {
    println!("\nRunning Mem Check\n");
    let trip = SparseTriplet::new(3, 3, 5)?;
    let solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
    println!("{}", trip);
    println!("\n{}", solver);
    println!("\nDone\n");
    Ok(())
}
