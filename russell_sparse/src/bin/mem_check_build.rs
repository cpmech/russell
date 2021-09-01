use russell_lab::*;
use russell_sparse::*;

fn main() {
    println!("\nRunning Mem Check\n");

    let mut trip = match SparseTriplet::new(5, 5, 13) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new triplet): {}", e);
            return;
        }
    };

    trip.put(0, 0, 1.0); // << duplicated
    trip.put(0, 0, 1.0); // << duplicated
    trip.put(1, 0, 3.0);
    trip.put(0, 1, 3.0);
    trip.put(2, 1, -1.0);
    trip.put(4, 1, 4.0);
    trip.put(1, 2, 4.0);
    trip.put(2, 2, -3.0);
    trip.put(3, 2, 1.0);
    trip.put(4, 2, 2.0);
    trip.put(2, 3, 2.0);
    trip.put(1, 4, 6.0);
    trip.put(4, 4, 1.0);

    let mut solver = match FrenchSolver::new(EnumSymmetry::No, true) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    match solver.initialize(&trip, true) {
        Err(e) => {
            println!("FAIL(initialize): {}", e);
            return;
        }
        _ => (),
    };

    match solver.factorize(true) {
        Err(e) => {
            println!("FAIL(factorize): {}", e);
            return;
        }
        _ => (),
    };

    let mut x = Vector::new(5);
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    match solver.solve(&mut x, &rhs, false) {
        Err(e) => {
            println!("FAIL(solve): {}", e);
            return;
        }
        _ => (),
    }

    println!("\nx =\n{}", x);
    println!("\n{}", trip);
    println!("\n{}", solver);
    println!("\nDone\n");
}
