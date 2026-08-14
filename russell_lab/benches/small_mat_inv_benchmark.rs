// Structure:
// - Three Criterion groups, each benchmarking sizes 3×3 … 9×9:
// - mat_inverse — LAPACK dgetrf/dgetri (analytic formulas for ≤3).
// - small_mat_inv_partial — small_mat_inv(&mut a, false).
// - small_mat_inv_full — small_mat_inv(&mut a, true) (the C full-pivoting path).
// - All three invert the same diagonally-dominant matrix (diag = n+1, off-diagonal 1), which is guaranteed non-singular.
// - A bench_small! macro generates one benchmark per N for the const-generic small_mat_inv (since N is compile-time), while mat_inverse uses a plain runtime loop over n.
// - Each small_mat_inv iteration copies the input array (let mut aa = a;) so the in-place inverse is recomputed fresh; mat_inverse doesn't need that (it reads a immutably and overwrites ai).

// Benchmarks for small-matrix inversion, comparing three approaches for
// (n×n) matrices with n = 3..9:
//
//   1. mat_inverse             -- LAPACK dgetrf/dgetri (analytic formulas for n <= 3)
//   2. small_mat_inv (partial) -- Gauss-Jordan with partial (row) pivoting (pure Rust)
//   3. small_mat_inv (full)    -- Gauss-Jordan with full pivoting (Numerical Recipes, compiled C)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use russell_lab::{Matrix, mat_inverse, small_mat_inv};

/// Returns the (i,j) element of a well-conditioned, diagonally dominant (n×n) matrix
fn element(i: usize, j: usize, n: usize) -> f64 {
    if i == j { n as f64 + 1.0 } else { 1.0 }
}

/// Benchmarks `mat_inverse` for (n×n) matrices with n = 3..9
fn bench_mat_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat_inverse");
    for n in 3..=9 {
        let mut a = Matrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                a.set(i, j, element(i, j, n));
            }
        }
        let mut ai = Matrix::new(n, n);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| mat_inverse(&mut ai, &a).unwrap());
        });
    }
    group.finish();
}

/// Generates one benchmark per N for the const-generic `small_mat_inv`
macro_rules! bench_small {
    ($group:expr, $full_pivot:expr, $($n:literal),+) => {
        $(
            {
                const N: usize = $n;
                let mut a = [[0.0; N]; N];
                for i in 0..N {
                    for j in 0..N {
                        a[i][j] = element(i, j, N);
                    }
                }
                $group.throughput(Throughput::Elements((N * N) as u64));
                $group.bench_with_input(BenchmarkId::from_parameter(N), &N, |b, _| {
                    b.iter(|| {
                        let mut aa = a;
                        small_mat_inv(&mut aa, $full_pivot).unwrap();
                    });
                });
            }
        )+
    };
}

/// Benchmarks `small_mat_inv` with partial pivoting for (n×n) matrices with n = 3..9
fn bench_small_mat_inv_partial(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_mat_inv_partial");
    bench_small!(group, false, 3, 4, 5, 6, 7, 8, 9);
    group.finish();
}

/// Benchmarks `small_mat_inv` with full pivoting for (n×n) matrices with n = 3..9
fn bench_small_mat_inv_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_mat_inv_full");
    bench_small!(group, true, 3, 4, 5, 6, 7, 8, 9);
    group.finish();
}

criterion_group!(
    benches,
    bench_mat_inverse,
    bench_small_mat_inv_partial,
    bench_small_mat_inv_full
);
criterion_main!(benches);
