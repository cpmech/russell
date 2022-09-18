use russell_lab::Vector;
use russell_sparse::{ConfigSolver, LinSolKind, Solver, SparseTriplet};

fn test_solver(name: LinSolKind) {
    match name {
        LinSolKind::Mmp => println!("Testing MMP solver\n"),
        LinSolKind::Umf => println!("Testing UMF solver\n"),
    }

    let (neq, nnz) = (5, 13);

    let mut trip = match SparseTriplet::new(neq, nnz) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new triplet): {}", e);
            return;
        }
    };

    trip.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
    trip.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
    trip.put(1, 0, 3.0).unwrap();
    trip.put(0, 1, 3.0).unwrap();
    trip.put(2, 1, -1.0).unwrap();
    trip.put(4, 1, 4.0).unwrap();
    trip.put(1, 2, 4.0).unwrap();
    trip.put(2, 2, -3.0).unwrap();
    trip.put(3, 2, 1.0).unwrap();
    trip.put(4, 2, 2.0).unwrap();
    trip.put(2, 3, 2.0).unwrap();
    trip.put(1, 4, 6.0).unwrap();
    trip.put(4, 4, 1.0).unwrap();

    let mut config = ConfigSolver::new();
    config.lin_sol_kind(name);
    let mut solver = match Solver::new(config, neq, nnz, None) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    match solver.factorize(&trip) {
        Err(e) => {
            println!("FAIL(factorize): {}", e);
            return;
        }
        _ => (),
    };

    let mut x = Vector::new(5);
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    match solver.solve(&mut x, &rhs) {
        Err(e) => {
            println!("FAIL(solve): {}", e);
            return;
        }
        _ => (),
    }

    match solver.solve(&mut x, &rhs) {
        Err(e) => {
            println!("FAIL(solve again): {}", e);
            return;
        }
        _ => (),
    }

    println!("{}", trip);
    println!("{}", solver);
    println!("x =\n{}", x);
}

fn test_solver_singular(name: LinSolKind) {
    match name {
        LinSolKind::Mmp => println!("Testing MMP solver\n"),
        LinSolKind::Umf => println!("Testing UMF solver\n"),
    }

    let (neq, nnz) = (2, 2);

    let trip_singular = match SparseTriplet::new(neq, nnz) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new triplet): {}", e);
            return;
        }
    };

    let mut config = ConfigSolver::new();
    config.lin_sol_kind(name);
    let mut solver = match Solver::new(config, neq, nnz, None) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    match solver.factorize(&trip_singular) {
        Err(e) => println!("\nOk(factorize singular matrix): {}\n", e),
        _ => (),
    };
}

fn main() {
    println!("Running Mem Check\n");
    test_solver(LinSolKind::Mmp);
    test_solver(LinSolKind::Umf);
    test_solver_singular(LinSolKind::Mmp);
    test_solver_singular(LinSolKind::Umf);
    println!("Done\n");
}
