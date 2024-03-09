use num_complex::Complex64;
use russell_lab::vec_approx_eq;
use russell_lab::{complex_vec_approx_eq, cpx, ComplexVector, Vector};
use russell_sparse::prelude::*;
use russell_sparse::Samples;

fn test_solver(genie: Genie) {
    println!("----------------------------------------------------------------------\n");
    match genie {
        Genie::Mumps => println!("Testing MUMPS solver\n"),
        Genie::Umfpack => println!("Testing UMFPACK solver\n"),
    }

    let mut solver = match LinSolver::new(genie) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
    let mut mat = SparseMatrix::from_coo(coo);

    match solver.actual.factorize(&mut mat, None) {
        Err(e) => {
            println!("FAIL(factorize): {}", e);
            return;
        }
        _ => (),
    };

    let mut x = Vector::new(5);
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    match solver.actual.solve(&mut x, &mut mat, &rhs, false) {
        Err(e) => {
            println!("FAIL(solve): {}", e);
            return;
        }
        _ => (),
    }

    match solver.actual.solve(&mut x, &mat, &rhs, false) {
        Err(e) => {
            println!("FAIL(solve again): {}", e);
            return;
        }
        _ => (),
    }

    println!("x =\n{}", x);
    let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
    vec_approx_eq(x.as_data(), x_correct, 1e-14);
}

fn test_complex_solver(genie: Genie) {
    println!("----------------------------------------------------------------------\n");
    match genie {
        Genie::Mumps => println!("Testing Complex MUMPS solver\n"),
        Genie::Umfpack => println!("Testing Complex UMFPACK solver\n"),
    }

    let mut solver = match ComplexLinSolver::new(genie) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    let coo = match genie {
        Genie::Mumps => Samples::complex_symmetric_3x3_lower().0,
        Genie::Umfpack => Samples::complex_symmetric_3x3_full().0,
    };
    let mut mat = ComplexSparseMatrix::from_coo(coo);

    match solver.actual.factorize(&mut mat, None) {
        Err(e) => {
            println!("FAIL(factorize): {}", e);
            return;
        }
        _ => (),
    };

    let mut x = ComplexVector::new(3);
    let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);

    match solver.actual.solve(&mut x, &mut mat, &rhs, false) {
        Err(e) => {
            println!("FAIL(solve): {}", e);
            return;
        }
        _ => (),
    }

    match solver.actual.solve(&mut x, &mat, &rhs, false) {
        Err(e) => {
            println!("FAIL(solve again): {}", e);
            return;
        }
        _ => (),
    }

    println!("x =\n{}", x);
    let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
    complex_vec_approx_eq(x.as_data(), x_correct, 1e-14);
}

fn test_solver_singular(genie: Genie) {
    println!("----------------------------------------------------------------------\n");
    match genie {
        Genie::Mumps => println!("Testing MUMPS solver (singular matrix)\n"),
        Genie::Umfpack => println!("Testing UMFPACK solver (singular matrix)\n"),
    }

    let (ndim, nnz) = (2, 2);

    let mut solver = match LinSolver::new(genie) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    let mut coo_singular = match SparseMatrix::new_coo(ndim, ndim, nnz, Sym::No) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new CooMatrix): {}", e);
            return;
        }
    };
    coo_singular.put(0, 0, 1.0).unwrap();
    coo_singular.put(1, 0, 1.0).unwrap();

    match solver.actual.factorize(&mut coo_singular, None) {
        Err(e) => println!("Ok(factorize singular matrix): {}\n", e),
        _ => (),
    };
}

fn main() {
    // real
    test_solver(Genie::Mumps);
    test_solver(Genie::Umfpack);

    // complex
    test_complex_solver(Genie::Mumps);
    test_complex_solver(Genie::Umfpack);

    // singular real
    test_solver_singular(Genie::Mumps);
    test_solver_singular(Genie::Umfpack);

    println!("----------------------------------------------------------------------\n");
}
