use russell_lab::Vector;
use russell_sparse::prelude::*;

fn test_solver(name: LinSolKind) {
    match name {
        LinSolKind::Umfpack => println!("Testing UMFPACK solver\n"),
    }

    let (nrow, ncol, nnz) = (5, 5, 13);

    let mut coo = match CooMatrix::new(None, nrow, ncol, nnz) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new CooMatrix): {}", e);
            return;
        }
    };

    coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
    coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
    coo.put(1, 0, 3.0).unwrap();
    coo.put(0, 1, 3.0).unwrap();
    coo.put(2, 1, -1.0).unwrap();
    coo.put(4, 1, 4.0).unwrap();
    coo.put(1, 2, 4.0).unwrap();
    coo.put(2, 2, -3.0).unwrap();
    coo.put(3, 2, 1.0).unwrap();
    coo.put(4, 2, 2.0).unwrap();
    coo.put(2, 3, 2.0).unwrap();
    coo.put(1, 4, 6.0).unwrap();
    coo.put(4, 4, 1.0).unwrap();

    let mut config = ConfigSolver::new();
    config.lin_sol_kind(name);
    let mut solver = match Solver::new(config, nrow, nnz, None) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    match solver.factorize(&coo) {
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

    println!("x =\n{}", x);
}

fn test_solver_singular(name: LinSolKind) {
    match name {
        LinSolKind::Umfpack => println!("Testing UMFPACK solver\n"),
    }

    let (nrow, ncol, nnz) = (2, 2, 2);

    let coo_singular = match CooMatrix::new(None, nrow, ncol, nnz) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new CooMatrix): {}", e);
            return;
        }
    };

    let mut config = ConfigSolver::new();
    config.lin_sol_kind(name);
    let mut solver = match Solver::new(config, nrow, nnz, None) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    match solver.factorize(&coo_singular) {
        Err(e) => println!("\nOk(factorize singular matrix): {}\n", e),
        _ => (),
    };
}

fn main() {
    println!("Running Mem Check\n");
    test_solver(LinSolKind::Umfpack);
    test_solver_singular(LinSolKind::Umfpack);
    println!("Done\n");
}
