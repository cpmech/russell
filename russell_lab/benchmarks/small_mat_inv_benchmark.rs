// Benchmarks for small-matrix inversion, comparing three approaches for
// (n×n) matrices with n = 3..9:
//
//   1. mat_inverse            -- LAPACK dgetrf/dgetri (analytic formulas for n <= 3)
//   2. small_mat_inv          -- Gauss-Jordan with partial (row) pivoting (pure Rust)
//   3. num_recipes_gaussj_inv -- Gauss-Jordan with full pivoting (Numerical Recipes, pure Rust)
//
// All three invert the same diagonally dominant matrix (diag = n + 1, off-diagonal = 1),
// which is guaranteed to be non-singular. A macro generates one benchmark per N for the
// const-generic functions, while `mat_inverse` uses a runtime loop over n.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use russell_lab::{Matrix, mat_inverse, num_recipes_gaussj_inv, small_mat_inv};

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
    ($group:expr, $($n:literal),+) => {
        $(
            {
                const N: usize = $n;
                let mut a = [[0.0; N]; N];
                for i in 0..N {
                    for j in 0..N {
                        a[i][j] = element(i, j, N);
                    }
                }
                // Prevent the compiler from const-folding the (deterministic) input
                let a = std::hint::black_box(a);
                $group.throughput(Throughput::Elements((N * N) as u64));
                $group.bench_with_input(BenchmarkId::from_parameter(N), &N, |b, _| {
                    let mut ai = [[0.0; N]; N];
                    b.iter(|| {
                        small_mat_inv(&mut ai, &a, N).unwrap();
                        // Prevent dead-store elimination of the output
                        std::hint::black_box(ai);
                    });
                });
            }
        )+
    };
}

/// Generates one benchmark per N for the const-generic `num_recipes_gaussj_inv`
macro_rules! bench_num_recipes {
    ($group:expr, $($n:literal),+) => {
        $(
            {
                const N: usize = $n;
                let mut a = [[0.0; N]; N];
                for i in 0..N {
                    for j in 0..N {
                        a[i][j] = element(i, j, N);
                    }
                }
                // Prevent the compiler from const-folding the (deterministic) input
                let a = std::hint::black_box(a);
                $group.throughput(Throughput::Elements((N * N) as u64));
                $group.bench_with_input(BenchmarkId::from_parameter(N), &N, |b, _| {
                    b.iter(|| {
                        let mut aa = a;
                        num_recipes_gaussj_inv(&mut aa).unwrap();
                        std::hint::black_box(aa);
                    });
                });
            }
        )+
    };
}

/// Benchmarks `small_mat_inv` (partial pivoting) for (n×n) matrices with n = 3..9
fn bench_small_mat_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_mat_inv");
    bench_small!(group, 3, 4, 5, 6, 7, 8, 9);
    group.finish();
}

/// Benchmarks `num_recipes_gaussj_inv` (full pivoting) for (n×n) matrices with n = 3..9
fn bench_num_recipes_gaussj_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("num_recipes_gaussj_inv");
    bench_num_recipes!(group, 3, 4, 5, 6, 7, 8, 9);
    group.finish();
}

criterion_group!(benches, bench_mat_inverse, bench_small_mat_inv, bench_num_recipes_gaussj_inv);
criterion_main!(benches);
