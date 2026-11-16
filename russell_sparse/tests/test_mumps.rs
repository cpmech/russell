#![cfg(feature = "mumps")]

use russell_lab::*;
use russell_sparse::StrError;
use russell_sparse::prelude::*;

#[test]
fn test_mumps() -> Result<(), StrError> {
    let n = 10;
    let d = (n as f64) / 10.0;
    let mut coo = CooMatrix::new(n, n, n, Sym::No)?;
    let mut x_correct = Vector::new(n);
    let mut rhs = Vector::new(n);
    for k in 0..n {
        // put diagonal entries
        let akk = 10.0 + (k as f64) * d;
        coo.put(k, k, akk)?;
        // let the exact solution be k
        x_correct[k] = k as f64;
        // generate RHS to match solution
        rhs[k] = akk * x_correct[k];
    }
    // println!("a =\n{}", coo.as_dense());
    // println!("x =\n{}", x_correct);
    // println!("b =\n{}", rhs);
    let mut x = Vector::new(n);
    let mut solver = SolverMUMPS::new()?;
    solver.preprocess(&coo, None)?;
    solver.solve(&mut x, &rhs, false)?;
    vec_approx_eq(&x, &x_correct, 1e-14);
    Ok(())
}

/// Builds a diagonal system whose exact solution is known
///
/// Everything is built inside the calling thread so that nothing is shared
/// between the concurrent tests.
fn make_system() -> Result<(CooMatrix, Vector, Vector), StrError> {
    let n = 10;
    let d = (n as f64) / 10.0;
    let mut coo = CooMatrix::new(n, n, n, Sym::No)?;
    let mut x_correct = Vector::new(n);
    let mut rhs = Vector::new(n);
    for k in 0..n {
        let akk = 10.0 + (k as f64) * d;
        coo.put(k, k, akk)?;
        x_correct[k] = k as f64;
        rhs[k] = akk * x_correct[k];
    }
    Ok((coo, x_correct, rhs))
}

/// Same diagonal system as [`make_system`], in complex arithmetic
///
/// Mirrors the fixture in `tests/test_complex_mumps.rs`.
fn make_complex_system() -> Result<(ComplexCooMatrix, ComplexVector, ComplexVector), StrError> {
    let n = 10;
    let d = (n as f64) / 10.0;
    let mut coo = ComplexCooMatrix::new(n, n, n, Sym::No)?;
    let mut x_correct = ComplexVector::new(n);
    let mut rhs = ComplexVector::new(n);
    for k in 0..n {
        let akk = cpx!(10.0 + (k as f64) * d, 10.0 - (k as f64) * d);
        coo.put(k, k, akk)?;
        x_correct[k] = cpx!(k as f64, 0.5);
        rhs[k] = akk * x_correct[k];
    }
    Ok((coo, x_correct, rhs))
}

/// Exercises `dmumps_c` and `zmumps_c` from several threads at once
///
/// Ten caller threads hammer `new`/`preprocess`/`solve`/`drop` while every FFI call is
/// dispatched to the single process-wide MUMPS worker thread. MUMPS therefore keeps one
/// OpenMP master thread and its full team (the default `mumps_num_threads = 0` lets MUMPS
/// choose), even though the callers run concurrently. This test pins that protocol.
#[test]
fn test_mumps_concurrent_stress() {
    const N: usize = 150;

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(10);

        // 4 steady-solve threads: one solver per thread, refactorized in place
        for _ in 0..4 {
            handles.push(scope.spawn(|| -> Result<(), StrError> {
                let (coo, x_correct, rhs) = make_system()?;
                let mut solver = SolverMUMPS::new()?;
                for _ in 0..N {
                    let mut x = Vector::new(rhs.dim());
                    solver.preprocess(&coo, None)?;
                    solver.solve(&mut x, &rhs, false)?;
                    vec_approx_eq(&x, &x_correct, 1e-14);
                }
                Ok(())
            }));
        }

        // 4 real churn threads: new -> preprocess -> solve -> drop, every iteration
        //
        // preprocess is mandatory: without it the C interface never sets
        // done_job_init and drop skips JOB = -2, so nothing would be exercised.
        for _ in 0..4 {
            handles.push(scope.spawn(|| -> Result<(), StrError> {
                let (coo, x_correct, rhs) = make_system()?;
                for _ in 0..N {
                    let mut x = Vector::new(rhs.dim());
                    let mut solver = SolverMUMPS::new()?;
                    solver.preprocess(&coo, None)?;
                    solver.solve(&mut x, &rhs, false)?;
                    vec_approx_eq(&x, &x_correct, 1e-14);
                    drop(solver);
                }
                Ok(())
            }));
        }

        // 2 complex churn+solve threads: same protocol through ComplexSolverMUMPS
        for _ in 0..2 {
            handles.push(scope.spawn(|| -> Result<(), StrError> {
                let (coo, x_correct, rhs) = make_complex_system()?;
                for _ in 0..N {
                    let mut x = ComplexVector::new(rhs.dim());
                    let mut solver = ComplexSolverMUMPS::new()?;
                    solver.preprocess(&coo, None)?;
                    solver.solve(&mut x, &rhs, false)?;
                    complex_vec_approx_eq(&x, &x_correct, 1e-14);
                    drop(solver);
                }
                Ok(())
            }));
        }

        // propagate every thread's error instead of swallowing it
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
    });
}

/// Holds several MUMPS instances alive at once, interleaving their lifecycles
///
/// A solver that held a non-reentrant lock across its lifetime would deadlock
/// on the second `new` of the same thread. This test must pass both before and
/// after the synchronization is added: it pins the protocol that fix preserves.
#[test]
fn test_mumps_multiple_instances_single_thread() -> Result<(), StrError> {
    let (coo, x_correct, rhs) = make_system()?;
    let (cmatrix, cx_correct, crhs) = make_complex_system()?;

    // first round: create a, b, c; preprocess; solve; drop in creation order
    let mut a = SolverMUMPS::new()?;
    let mut b = SolverMUMPS::new()?;
    let mut c = ComplexSolverMUMPS::new()?;
    a.preprocess(&coo, None)?;
    b.preprocess(&coo, None)?;
    c.preprocess(&cmatrix, None)?;
    let mut xa = Vector::new(rhs.dim());
    let mut xb = Vector::new(rhs.dim());
    let mut xc = ComplexVector::new(crhs.dim());
    a.solve(&mut xa, &rhs, false)?;
    b.solve(&mut xb, &rhs, false)?;
    c.solve(&mut xc, &crhs, false)?;
    vec_approx_eq(&xa, &x_correct, 1e-14);
    vec_approx_eq(&xb, &x_correct, 1e-14);
    complex_vec_approx_eq(&xc, &cx_correct, 1e-14);
    drop(a);
    drop(b);
    drop(c);

    // second round: the reverse create/drop order of the first round
    let mut c2 = ComplexSolverMUMPS::new()?;
    let mut b2 = SolverMUMPS::new()?;
    let mut a2 = SolverMUMPS::new()?;
    c2.preprocess(&cmatrix, None)?;
    b2.preprocess(&coo, None)?;
    a2.preprocess(&coo, None)?;
    let mut xa2 = Vector::new(rhs.dim());
    let mut xb2 = Vector::new(rhs.dim());
    let mut xc2 = ComplexVector::new(crhs.dim());
    c2.solve(&mut xc2, &crhs, false)?;
    b2.solve(&mut xb2, &rhs, false)?;
    a2.solve(&mut xa2, &rhs, false)?;
    complex_vec_approx_eq(&xc2, &cx_correct, 1e-14);
    vec_approx_eq(&xb2, &x_correct, 1e-14);
    vec_approx_eq(&xa2, &x_correct, 1e-14);
    drop(a2);
    drop(b2);
    drop(c2);

    Ok(())
}
