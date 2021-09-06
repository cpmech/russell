use russell_lab::*;
use russell_sparse::*;

fn test_solver_mmp() {
    println!("Testing SolverMMP\n");

    let mut trip = match SparseTriplet::new(5, 5, 13, false) {
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

    let config = ConfigSolver::new();
    let mut solver_mmp = match SolverMMP::new(config) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    match solver_mmp.initialize(&trip) {
        Err(e) => {
            println!("FAIL(initialize): {}", e);
            return;
        }
        _ => (),
    };

    match solver_mmp.factorize() {
        Err(e) => {
            println!("FAIL(factorize): {}", e);
            return;
        }
        _ => (),
    };

    let mut x = Vector::new(5);
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    match solver_mmp.solve(&mut x, &rhs) {
        Err(e) => {
            println!("FAIL(solve): {}", e);
            return;
        }
        _ => (),
    }

    match solver_mmp.solve(&mut x, &rhs) {
        Err(e) => {
            println!("FAIL(solve again): {}", e);
            return;
        }
        _ => (),
    }

    println!("{}", trip);
    println!("\n{}", solver_mmp);
    println!("\nx =\n{}", x);

    let mut trip_singular = match SparseTriplet::new(5, 5, 2, false) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new singular matrix): {}", e);
            return;
        }
    };

    trip_singular.put(0, 0, 1.0);
    trip_singular.put(4, 4, 1.0);
    match solver_mmp.initialize(&trip_singular) {
        Err(e) => {
            println!("FAIL(initialize singular matrix): {}", e);
            return;
        }
        _ => (),
    };

    match solver_mmp.factorize() {
        Err(e) => println!("\nOk(factorize singular matrix): {}", e),
        _ => (),
    };
}

fn test_solver_umf() {
    println!("\nTesting SolverUMF\n");

    let mut trip = match SparseTriplet::new(5, 5, 13, false) {
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

    let config = ConfigSolver::new();
    let mut solver_umf = match SolverUMF::new(config) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    match solver_umf.initialize(&trip) {
        Err(e) => {
            println!("FAIL(initialize): {}", e);
            return;
        }
        _ => (),
    };

    match solver_umf.factorize() {
        Err(e) => {
            println!("FAIL(factorize): {}", e);
            return;
        }
        _ => (),
    };

    let mut x = Vector::new(5);
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    match solver_umf.solve(&mut x, &rhs) {
        Err(e) => {
            println!("FAIL(solve): {}", e);
            return;
        }
        _ => (),
    }

    match solver_umf.solve(&mut x, &rhs) {
        Err(e) => {
            println!("FAIL(solve again): {}", e);
            return;
        }
        _ => (),
    }

    println!("{}", trip);
    println!("\n{}", solver_umf);
    println!("\nx =\n{}", x);

    let mut trip_singular = match SparseTriplet::new(5, 5, 2, false) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new singular matrix): {}", e);
            return;
        }
    };

    trip_singular.put(0, 0, 1.0);
    trip_singular.put(4, 4, 1.0);
    match solver_umf.initialize(&trip_singular) {
        Err(e) => {
            println!("FAIL(initialize singular matrix): {}", e);
            return;
        }
        _ => (),
    };

    match solver_umf.factorize() {
        Err(e) => println!("\nOk(factorize singular matrix): {}", e),
        _ => (),
    };

    println!("\nDone\n");
}

fn main() {
    println!("\nRunning Mem Check\n");
    test_solver_mmp();
    test_solver_umf();
    println!("\nDone\n");
}
