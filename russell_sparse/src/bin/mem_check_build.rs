use russell_lab::*;
use russell_sparse::*;

fn main() -> Result<(), &'static str> {
    println!("\nRunning Mem Check\n");
    let mut trip = SparseTriplet::new(5, 5, 13)?;
    trip.put(0, 0, 1.0)?; // << duplicated
    trip.put(0, 0, 1.0)?; // << duplicated
    trip.put(1, 0, 3.0)?;
    trip.put(0, 1, 3.0)?;
    trip.put(2, 1, -1.0)?;
    trip.put(4, 1, 4.0)?;
    trip.put(1, 2, 4.0)?;
    trip.put(2, 2, -3.0)?;
    trip.put(3, 2, 1.0)?;
    trip.put(4, 2, 2.0)?;
    trip.put(2, 3, 2.0)?;
    trip.put(1, 4, 6.0)?;
    trip.put(4, 4, 1.0)?;
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
    trip.set_rhs(&rhs)?;
    let mut solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
    solver.analyze(&trip, true)?;
    solver.factorize(true)?;
    solver.solve(&trip, true)?;
    let x = trip.get_rhs()?;
    println!("\nx =\n{}", x);
    println!("\n{}", trip);
    println!("\n{}", solver);
    println!("\nDone\n");
    Ok(())
}
