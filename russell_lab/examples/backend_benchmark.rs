//! Times the two backends on the operations where they differ most.
//!
//! Run it twice, once per backend, and compare:
//!
//! ```text
//! cargo run --release --example backend_benchmark --features open_blas
//! OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
//!     cargo run --release --example backend_benchmark --features open_blas
//! cargo run --release --example backend_benchmark
//! ```
//!
//! The second form pins OpenBLAS to one thread: the native routines are
//! single-threaded, so that is the only like-for-like comparison of the kernels.
//! Both variables are needed because OpenMP builds of OpenBLAS take the count
//! from `OMP_NUM_THREADS` and ignore `OPENBLAS_NUM_THREADS`.
//! Without it OpenBLAS uses every core, and the numbers say more about the
//! machine than about the code.
//!
//! The figures in `README.md` were produced with the second and third forms on
//! an Intel Core i9-13900KF, best of 5 to 20 runs per size.

use russell_lab::{Matrix, Vector, mat_mat_mul, solve_lin_sys};
use std::time::Instant;

/// A matrix that is not symmetric and has a well-conditioned diagonal.
fn filled(n: usize, seed: f64) -> Matrix {
    let mut a = Matrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let off = ((i * 31 + j * 17) as f64 * 0.001 + seed).sin();
            a.set(i, j, off + if i == j { 2.0 } else { 0.0 });
        }
    }
    a
}

/// Best of `reps` runs, to cut the scheduler noise.
fn best_of<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let backend = if cfg!(feature = "open_blas") {
        "open_blas"
    } else if cfg!(feature = "intel_mkl") {
        "intel_mkl"
    } else {
        "native"
    };
    println!("# backend: {backend}");
    println!("#{:<8} {:>10} {:>12} {:>12}", "operation", "size", "time", "rate");
    for n in [256, 512, 1024] {
        let a = filled(n, 0.0);
        let b = filled(n, 0.1);
        let mut c = Matrix::new(n, n);
        let reps = if n <= 256 {
            20
        } else if n == 512 {
            5
        } else {
            2
        };
        let t = best_of(reps, || mat_mat_mul(&mut c, 1.0, &a, &b, 0.0).unwrap());
        let gflops = 2.0 * (n * n * n) as f64 / t / 1e9;
        println!(
            "{:<10} {n:>4}x{n:<4} {:>9.2} ms {:>7.1} GFLOP/s",
            "dgemm",
            t * 1e3,
            gflops
        );
    }
    for n in [128, 512] {
        let a = filled(n, 0.0);
        let b = Vector::from(&vec![1.0; n]);
        let reps = if n <= 128 { 20 } else { 5 };
        let t = best_of(reps, || {
            let mut a = a.clone();
            let mut b = b.clone();
            solve_lin_sys(&mut b, &mut a).unwrap();
        });
        println!("{:<10} {n:>4}      {:>9.2} ms", "dgesv", t * 1e3);
    }
}
