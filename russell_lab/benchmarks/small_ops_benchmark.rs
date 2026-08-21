// Benchmarks for the small (stack-allocated) operations in the `small` module,
// comparing each one against its heap-allocated counterpart (BLAS/LAPACK-backed).
//
// The small versions operate on stack arrays `[[f64; N]; N]` / `[f64; N]` with
// `N` known at compile time, while the heap versions allocate `Matrix`/`Vector`
// on the heap. For in-place operations (update, solve), the mutable input is
// reset on every iteration: a heap `clone()` vs a stack copy.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use russell_lab::{
    Matrix, Vector, mat_add, mat_mat_mul, mat_update, small_mat_add, small_mat_mat_mul,
    small_mat_update, small_solve_lin_sys, small_vec_add, small_vec_update, solve_lin_sys, vec_add,
    vec_update,
};

/// Matrix sizes for the matrix operations
const MAT_SIZES: [usize; 7] = [3, 4, 5, 6, 7, 8, 9];

/// Vector sizes for the vector operations
const VEC_SIZES: [usize; 6] = [4, 8, 16, 32, 64, 128];

/// Returns the (i,j) element of a well-conditioned (diagonally dominant) matrix
fn element(i: usize, j: usize) -> f64 {
    if i == j {
        2.0
    } else {
        0.1
    }
}

/// Generates the small (const-generic) benchmark for `small_mat_add`
macro_rules! bench_small_mat_add {
    ($group:expr, $($n:literal),+) => { $(
        {
            const N: usize = $n;
            let mut a = [[0.0; N]; N];
            let mut b = [[0.0; N]; N];
            for i in 0..N {
                for j in 0..N {
                    a[i][j] = element(i, j);
                    b[i][j] = element(j, i);
                }
            }
            let a = std::hint::black_box(a);
            let b = std::hint::black_box(b);
            let mut c = [[0.0; N]; N];
            $group.throughput(Throughput::Elements((N * N) as u64));
            $group.bench_with_input(BenchmarkId::new("small", N), &N, |bb, _| {
                bb.iter(|| {
                    small_mat_add(&mut c, 1.0, &a, 1.0, &b, N);
                    std::hint::black_box(c);
                });
            });
        }
    )+ };
}

/// Benchmarks `mat_add` (heap) vs `small_mat_add` (stack)
fn bench_mat_add(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("mat_add");
    for n in MAT_SIZES {
        let mut a = Matrix::new(n, n);
        let mut b = Matrix::new(n, n);
        let mut c = Matrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                a.set(i, j, element(i, j));
                b.set(i, j, element(j, i));
            }
        }
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("heap", n), &n, |bb, _| {
            bb.iter(|| {
                mat_add(&mut c, 1.0, &a, 1.0, &b).unwrap();
                std::hint::black_box(&c);
            });
        });
    }
    bench_small_mat_add!(group, 3, 4, 5, 6, 7, 8, 9);
    group.finish();
}

/// Generates the small (const-generic) benchmark for `small_mat_update`
macro_rules! bench_small_mat_update {
    ($group:expr, $($n:literal),+) => { $(
        {
            const N: usize = $n;
            let mut a = [[0.0; N]; N];
            for i in 0..N {
                for j in 0..N {
                    a[i][j] = element(i, j);
                }
            }
            let a = std::hint::black_box(a);
            let b0 = [[0.0; N]; N];
            $group.throughput(Throughput::Elements((N * N) as u64));
            $group.bench_with_input(BenchmarkId::new("small", N), &N, |bb, _| {
                bb.iter(|| {
                    let mut b = b0;
                    small_mat_update(&mut b, 1.0, &a, N);
                    std::hint::black_box(b);
                });
            });
        }
    )+ };
}

/// Benchmarks `mat_update` (heap) vs `small_mat_update` (stack)
fn bench_mat_update(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("mat_update");
    for n in MAT_SIZES {
        let mut a = Matrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                a.set(i, j, element(i, j));
            }
        }
        let b0 = Matrix::new(n, n);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("heap", n), &n, |bb, _| {
            bb.iter(|| {
                let mut b = b0.clone();
                mat_update(&mut b, 1.0, &a).unwrap();
                std::hint::black_box(&b);
            });
        });
    }
    bench_small_mat_update!(group, 3, 4, 5, 6, 7, 8, 9);
    group.finish();
}

/// Generates the small (const-generic) benchmark for `small_mat_mat_mul`
macro_rules! bench_small_mat_mat_mul {
    ($group:expr, $($n:literal),+) => { $(
        {
            const N: usize = $n;
            let mut a = [[0.0; N]; N];
            let mut b = [[0.0; N]; N];
            for i in 0..N {
                for j in 0..N {
                    a[i][j] = element(i, j);
                    b[i][j] = element(j, i);
                }
            }
            let a = std::hint::black_box(a);
            let b = std::hint::black_box(b);
            let mut c = [[0.0; N]; N];
            $group.throughput(Throughput::Elements((N * N * N) as u64));
            $group.bench_with_input(BenchmarkId::new("small", N), &N, |bb, _| {
                bb.iter(|| {
                    small_mat_mat_mul(&mut c, 1.0, &a, &b, 0.0, N);
                    std::hint::black_box(c);
                });
            });
        }
    )+ };
}

/// Benchmarks `mat_mat_mul` (heap) vs `small_mat_mat_mul` (stack)
fn bench_mat_mat_mul(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("mat_mat_mul");
    for n in MAT_SIZES {
        let mut a = Matrix::new(n, n);
        let mut b = Matrix::new(n, n);
        let mut c = Matrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                a.set(i, j, element(i, j));
                b.set(i, j, element(j, i));
            }
        }
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("heap", n), &n, |bb, _| {
            bb.iter(|| {
                mat_mat_mul(&mut c, 1.0, &a, &b, 0.0).unwrap();
                std::hint::black_box(&c);
            });
        });
    }
    bench_small_mat_mat_mul!(group, 3, 4, 5, 6, 7, 8, 9);
    group.finish();
}

/// Generates the small (const-generic) benchmark for `small_vec_add`
macro_rules! bench_small_vec_add {
    ($group:expr, $($n:literal),+) => { $(
        {
            const N: usize = $n;
            let mut u = [0.0; N];
            let mut v = [0.0; N];
            for i in 0..N {
                u[i] = (i as f64) * 0.1;
                v[i] = (i as f64) * 0.2;
            }
            let u = std::hint::black_box(u);
            let v = std::hint::black_box(v);
            let mut w = [0.0; N];
            $group.throughput(Throughput::Elements(N as u64));
            $group.bench_with_input(BenchmarkId::new("small", N), &N, |bb, _| {
                bb.iter(|| {
                    small_vec_add(&mut w, 1.0, &u, 1.0, &v, N);
                    std::hint::black_box(w);
                });
            });
        }
    )+ };
}

/// Benchmarks `vec_add` (heap) vs `small_vec_add` (stack)
fn bench_vec_add(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("vec_add");
    for n in VEC_SIZES {
        let mut u = Vector::new(n);
        let mut v = Vector::new(n);
        let mut w = Vector::new(n);
        for i in 0..n {
            u[i] = (i as f64) * 0.1;
            v[i] = (i as f64) * 0.2;
        }
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("heap", n), &n, |bb, _| {
            bb.iter(|| {
                vec_add(&mut w, 1.0, &u, 1.0, &v).unwrap();
                std::hint::black_box(&w);
            });
        });
    }
    bench_small_vec_add!(group, 4, 8, 16, 32, 64, 128);
    group.finish();
}

/// Generates the small (const-generic) benchmark for `small_vec_update`
macro_rules! bench_small_vec_update {
    ($group:expr, $($n:literal),+) => { $(
        {
            const N: usize = $n;
            let mut u = [0.0; N];
            for i in 0..N {
                u[i] = (i as f64) * 0.1;
            }
            let u = std::hint::black_box(u);
            let v0 = [0.0; N];
            $group.throughput(Throughput::Elements(N as u64));
            $group.bench_with_input(BenchmarkId::new("small", N), &N, |bb, _| {
                bb.iter(|| {
                    let mut v = v0;
                    small_vec_update(&mut v, 1.0, &u, N);
                    std::hint::black_box(v);
                });
            });
        }
    )+ };
}

/// Benchmarks `vec_update` (heap) vs `small_vec_update` (stack)
fn bench_vec_update(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("vec_update");
    for n in VEC_SIZES {
        let mut u = Vector::new(n);
        for i in 0..n {
            u[i] = (i as f64) * 0.1;
        }
        let v0 = Vector::new(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("heap", n), &n, |bb, _| {
            bb.iter(|| {
                let mut v = v0.clone();
                vec_update(&mut v, 1.0, &u).unwrap();
                std::hint::black_box(&v);
            });
        });
    }
    bench_small_vec_update!(group, 4, 8, 16, 32, 64, 128);
    group.finish();
}

/// Generates the small (const-generic) benchmark for `small_solve_lin_sys`
macro_rules! bench_small_solve {
    ($group:expr, $($n:literal),+) => { $(
        {
            const N: usize = $n;
            let mut a = [[0.0; N]; N];
            let mut b = [0.0; N];
            for i in 0..N {
                for j in 0..N {
                    a[i][j] = element(i, j);
                }
                b[i] = (i as f64) * 0.1;
            }
            let a0 = std::hint::black_box(a);
            let b0 = std::hint::black_box(b);
            $group.throughput(Throughput::Elements((N * N * N) as u64));
            $group.bench_with_input(BenchmarkId::new("small", N), &N, |bb, _| {
                bb.iter(|| {
                    let mut a = a0;
                    let mut b = b0;
                    small_solve_lin_sys(&mut b, &mut a).unwrap();
                    std::hint::black_box(b);
                });
            });
        }
    )+ };
}

/// Benchmarks `solve_lin_sys` (heap) vs `small_solve_lin_sys` (stack)
fn bench_solve_lin_sys(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("solve_lin_sys");
    for n in MAT_SIZES {
        let mut a = Matrix::new(n, n);
        let mut b = Vector::new(n);
        for i in 0..n {
            for j in 0..n {
                a.set(i, j, element(i, j));
            }
            b[i] = (i as f64) * 0.1;
        }
        let a0 = a;
        let b0 = b;
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("heap", n), &n, |bb, _| {
            bb.iter(|| {
                let mut a = a0.clone();
                let mut b = b0.clone();
                solve_lin_sys(&mut b, &mut a).unwrap();
                std::hint::black_box(&b);
            });
        });
    }
    bench_small_solve!(group, 3, 4, 5, 6, 7, 8, 9);
    group.finish();
}

criterion_group!(
    benches,
    bench_mat_add,
    bench_mat_update,
    bench_mat_mat_mul,
    bench_vec_add,
    bench_vec_update,
    bench_solve_lin_sys
);
criterion_main!(benches);
