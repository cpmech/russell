use russell_lab::{ComplexVector, Vector, complex_vec_approx_eq, cpx, vec_approx_eq};
use russell_sparse::Samples;
use russell_sparse::prelude::*;

fn test_solver(genie: Genie) {
    println!("----------------------------------------------------------------------\n");
    match genie {
        Genie::Mumps => println!("Testing MUMPS solver\n"),
        Genie::Direct => println!("Testing Direct solver\n"),
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

    let params = LinSolParams::new();
    if let Err(e) = solver.actual.preprocess(&coo, Some(params)) {
        println!("FAIL(preprocess): {}", e);
        return;
    };

    let mut x = Vector::new(5);
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    if let Err(e) = solver.actual.solve(&mut x, &rhs, false) {
        println!("FAIL(solve): {}", e);
        return;
    }

    if let Err(e) = solver.actual.solve(&mut x, &rhs, false) {
        println!("FAIL(solve again): {}", e);
        return;
    }

    println!("x =\n{}", x);
    let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
    vec_approx_eq(&x, x_correct, 1e-14);
}

fn test_complex_solver(genie: Genie) {
    println!("----------------------------------------------------------------------\n");
    match genie {
        Genie::Mumps => println!("Testing Complex MUMPS solver\n"),
        Genie::Direct => println!("Testing Complex Direct solver\n"),
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
        Genie::Direct => Samples::complex_symmetric_3x3_full().0,
        Genie::Umfpack => Samples::complex_symmetric_3x3_full().0,
    };

    if let Err(e) = solver.actual.preprocess(&coo, None) {
        println!("FAIL(preprocess): {}", e);
        return;
    };

    let mut x = ComplexVector::new(3);
    let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);

    if let Err(e) = solver.actual.solve(&mut x, &rhs, false) {
        println!("FAIL(solve): {}", e);
        return;
    }

    if let Err(e) = solver.actual.solve(&mut x, &rhs, false) {
        println!("FAIL(solve again): {}", e);
        return;
    }

    println!("x =\n{}", x);
    let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
    complex_vec_approx_eq(&x, x_correct, 1e-14);
}

fn test_solver_singular(genie: Genie) {
    println!("----------------------------------------------------------------------\n");
    match genie {
        Genie::Mumps => println!("Testing MUMPS solver (singular matrix)\n"),
        Genie::Direct => println!("Testing Direct solver (singular matrix)\n"),
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

    let mut coo_singular = match CooMatrix::new(ndim, ndim, nnz, Sym::No) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new CooMatrix): {}", e);
            return;
        }
    };
    coo_singular.put(0, 0, 1.0).unwrap();
    coo_singular.put(1, 0, 1.0).unwrap();

    if let Err(e) = solver.actual.preprocess(&coo_singular, None) {
        println!("Ok(preprocess singular matrix): {}\n", e)
    };
}

fn main() {
    // real
    test_solver(Genie::Mumps);
    test_solver(Genie::Direct);
    test_solver(Genie::Umfpack);

    // complex
    test_complex_solver(Genie::Mumps);
    test_complex_solver(Genie::Direct);
    test_complex_solver(Genie::Umfpack);

    // singular real
    test_solver_singular(Genie::Mumps);
    test_solver_singular(Genie::Direct);
    test_solver_singular(Genie::Umfpack);

    println!("----------------------------------------------------------------------\n");
}
