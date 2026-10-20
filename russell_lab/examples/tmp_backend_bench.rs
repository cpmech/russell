use russell_lab::{Matrix, Vector, mat_mat_mul, solve_lin_sys};
use std::time::Instant;
fn filled(n: usize, seed: f64) -> Matrix {
    let mut a = Matrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            a.set(
                i,
                j,
                ((i * 31 + j * 17) as f64 * 0.001 + seed).sin() + 2.0 * if i == j { 1.0 } else { 0.0 },
            );
        }
    }
    a
}
fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}
fn main() {
    let backend = if cfg!(feature = "open_blas") { "open_blas" } else { "native" };
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
        let t = time(reps, || mat_mat_mul(&mut c, 1.0, &a, &b, 0.0).unwrap());
        println!(
            "{backend:<10} dgemm {n:>4}: {:>8.2} ms  {:>7.1} GFLOP/s",
            t * 1e3,
            2.0 * (n * n * n) as f64 / t / 1e9
        );
    }
    for n in [128, 512] {
        let a0 = filled(n, 0.0);
        let b0 = Vector::from(&vec![1.0; n]);
        let reps = if n <= 128 { 20 } else { 5 };
        let t = time(reps, || {
            let mut a = a0.clone();
            let mut b = b0.clone();
            solve_lin_sys(&mut b, &mut a).unwrap();
        });
        println!("{backend:<10} dgesv {n:>4}: {:>8.2} ms", t * 1e3);
    }
}
