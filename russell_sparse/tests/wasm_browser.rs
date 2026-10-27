//! Browser-target smoke test for the Direct solver.
//!
//! The browser target has neither a clock nor a filesystem: the symbolic
//! analysis and the numeric factorization must read the time through the
//! browser clock, and the helpers that take a file path can only return an
//! error. That is the one code path the WASI suite cannot reach, because WASI
//! does provide both. This test preprocesses and solves a small system there.
#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

use russell_lab::Vector;
use russell_sparse::prelude::*;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn direct_solver_runs_in_the_browser_target() {
    // A = [[4, 1], [1, 3]]
    let mut solver = SolverDirect::new().unwrap();
    let mut coo = CooMatrix::new(2, 2, 4, Sym::No).unwrap();
    coo.put(0, 0, 4.0).unwrap();
    coo.put(0, 1, 1.0).unwrap();
    coo.put(1, 0, 1.0).unwrap();
    coo.put(1, 1, 3.0).unwrap();
    solver.preprocess(&coo, None).unwrap();
    let rhs = Vector::from(&[1.0, 2.0]);
    let mut x = Vector::new(2);
    solver.solve(&mut x, &rhs, false).unwrap();
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-14);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-14);
}
