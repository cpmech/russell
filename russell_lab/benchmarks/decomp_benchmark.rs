//! Wall-clock for the decompositions.
//!
//! The native kernels are unblocked (dsytd2, dgebd2, dlahqr), so they make no
//! BLAS-3 calls and cannot benefit from the blocked dgemm. This exists to put a
//! number on what that costs, before deciding whether to port the blocked
//! variants -- dlaqr0/zlaqr0, dsytrd, dgebrd.
//!
//! No criterion: these are milliseconds to seconds and the point is the order of
//! magnitude.
use russell_lab::Matrix;
use russell_lab::native::lapack::{svd, sym};
use std::time::Instant;

/// Well-conditioned and full rank. A modulo pattern repeats rows once n exceeds
/// the modulus, which makes the matrix rank deficient -- dgesvd then legitimately
/// reports Singular/does-not-converge, so it is useless as a benchmark input.
fn example(n: usize) -> Matrix {
    let mut a = Matrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, ((i as f64) * 0.7 + (j as f64) * 1.3).sin() + 0.1);
        }
    }
    a
}

fn sym_data(n: usize) -> Vec<f64> {
    let a = example(n);
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            s[i + j * n] = a.get(i, j) + a.get(j, i);
        }
    }
    s
}

fn main() {
    // dsyev, called directly so the timing is the routine and nothing else.
    println!("dsyev (dsytrd/dsytd2 by size + dsteqr):");
    for n in [100usize, 200, 400, 600] {
        let mut a = sym_data(n);
        let mut w = vec![0.0; n];
        // warm
        let mut a_warm = a.clone();
        sym::dsyev(&mut a_warm, &mut w, n, false).unwrap();
        let t = Instant::now();
        sym::dsyev(&mut a, &mut w, n, false).unwrap();
        let el = t.elapsed();
        println!(
            "  n={n:<4} {:>10.1?}   ({:.1} ns/flop)",
            el,
            el.as_secs_f64() * 1e9 / (4.0 / 3.0 * (n as f64).powi(3))
        );
    }

    // dgesvd, same story.
    println!("dgesvd (dgebd2 + dbdsqr, no BLAS-3):");
    for n in [100usize, 200, 400] {
        let mut a = example(n).as_mut_data().to_vec();
        let mut s = vec![0.0; n];
        let mut u = vec![0.0; n * n];
        let mut vt = vec![0.0; n * n];
        let mut a_warm = a.clone();
        svd::dgesvd(&mut a_warm, n, n, &mut s, &mut u, &mut vt).unwrap();
        let t = Instant::now();
        svd::dgesvd(&mut a, n, n, &mut s, &mut u, &mut vt).unwrap();
        let el = t.elapsed();
        // the direct bidiagonalization path is ~8n^3 because U and VT are formed
        println!(
            "  n={n:<4} {:>10.1?}   ({:.1} ns/flop)",
            el,
            el.as_secs_f64() * 1e9 / (8.0 * (n as f64).powi(3))
        );
    }
}
