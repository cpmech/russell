use russell_lab::Vector;
use russell_sparse::prelude::*;

fn test_solver(genie: Genie) {
    match genie {
        Genie::Mumps => println!("Testing MUMPS solver\n"),
        Genie::Umfpack => println!("Testing UMFPACK solver\n"),
        Genie::IntelDss => println!("Testing Intel DSS solver\n"),
    }

    let (ndim, nnz) = (5, 13);

    let mut solver = match Solver::new(genie) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    let one_based = if genie == Genie::Mumps { true } else { false };
    let mut coo = match SparseMatrix::new_coo(ndim, ndim, nnz, None, one_based) {
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

    match solver.actual.factorize(&mut coo, None) {
        Err(e) => {
            println!("FAIL(factorize): {}", e);
            return;
        }
        _ => (),
    };

    let mut x = Vector::new(5);
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    match solver.actual.solve(&mut x, &mut coo, &rhs, false) {
        Err(e) => {
            println!("FAIL(solve): {}", e);
            return;
        }
        _ => (),
    }

    match solver.actual.solve(&mut x, &coo, &rhs, false) {
        Err(e) => {
            println!("FAIL(solve again): {}", e);
            return;
        }
        _ => (),
    }

    println!("x =\n{}", x);
}

fn test_solver_singular(genie: Genie) {
    match genie {
        Genie::Mumps => println!("Testing MUMPS solver\n"),
        Genie::Umfpack => println!("Testing UMFPACK solver\n"),
        Genie::IntelDss => println!("Testing Intel DSS solver\n"),
    }

    let (ndim, nnz) = (2, 2);

    let mut solver = match Solver::new(genie) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    let one_based = if genie == Genie::Mumps { true } else { false };
    let mut coo_singular = match SparseMatrix::new_coo(ndim, ndim, nnz, None, one_based) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new CooMatrix): {}", e);
            return;
        }
    };
    coo_singular.put(0, 0, 1.0).unwrap();
    coo_singular.put(1, 0, 1.0).unwrap();

    match solver.actual.factorize(&mut coo_singular, None) {
        Err(e) => println!("\nOk(factorize singular matrix): {}\n", e),
        _ => (),
    };
}

fn main() {
    println!("Running Mem Check\n");
    test_solver(Genie::Umfpack);
    test_solver_singular(Genie::Umfpack);
    if cfg!(with_intel_dss) {
        test_solver(Genie::IntelDss);
        // Intel DSS cannot handle singular matrices
        // test_solver_singular(Genie::IntelDss);
    }
    println!("Done\n");
}
