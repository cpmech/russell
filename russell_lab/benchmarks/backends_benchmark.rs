//! Native Rust kernels against the linked BLAS library: what routing would buy.
//!
//! The `open_blas` and `intel_mkl` features link a library and expose its thread
//! controls, but they do **not** route the crate's math: `mat_mat_mul` calls
//! `native::blas::dgemm` unconditionally, and the C shim only wraps thread
//! control and `idamax`. All three builds therefore run the same Rust kernels.
//!
//! What the features do make possible is the comparison this bench is for: the
//! native kernel against the linked library's own CBLAS entry point, same call
//! shape, same data, same thread count. That is the number a decision to route
//! matmul through BLAS would be made on.
//!
//! Run it three ways and read the columns:
//!
//! ```text
//! cargo bench --bench backends_benchmark
//! cargo bench --features open_blas --bench backends_benchmark
//! cargo bench --features intel_mkl --bench backends_benchmark
//! ```
//!
//! The `native` group is in every run, so it doubles as the control: if its
//! numbers move between runs, the machine did, and the `lib` column means
//! nothing.
//!
//! Threads are pinned to one before the library is measured, because the native
//! kernels are single-threaded and a multi-threaded comparison would answer a
//! different question (with a different answer at 512²).

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use russell_lab::native;

/// The external library's own entry points, through CBLAS: both OpenBLAS and MKL
/// ship that interface (the crate's shim already includes `cblas.h` for both), so
/// one set of declarations serves either feature.
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
mod lib {
    pub const COL_MAJOR: i32 = 102;
    pub const NO_TRANS: i32 = 111;

    unsafe extern "C" {
        pub fn cblas_dgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *const f64,
            ldb: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32,
        );
        pub fn cblas_dgemv(
            order: i32,
            trans: i32,
            m: i32,
            n: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            x: *const f64,
            incx: i32,
            beta: f64,
            y: *mut f64,
            incy: i32,
        );
        pub fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
        pub fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64;
    }
}

/// Deterministic, non-trivial data.
fn data(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 17) as f64) - 8.0).collect()
}

/// `c := α a b + β c` for square `n x n` matrices: `n³` inner-loop steps.
fn bench_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");
    group.sample_size(10);
    for n in [64usize, 128, 256, 512] {
        group.throughput(Throughput::Elements((n * n * n) as u64));
        let (a, b) = (data(n * n), data(n * n));
        let mut cc = vec![0.0f64; n * n];
        group.bench_with_input(BenchmarkId::new("native", n), &n, |bench, &n| {
            bench.iter(|| native::blas::dgemm(1.0, &a, &b, 0.0, &mut cc, n, n, n));
        });
        #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
        {
            let (a, b) = (a.clone(), b.clone());
            group.bench_with_input(BenchmarkId::new("lib", n), &n, |bench, &n| {
                bench.iter(|| unsafe {
                    lib::cblas_dgemm(
                        lib::COL_MAJOR,
                        lib::NO_TRANS,
                        lib::NO_TRANS,
                        n as i32,
                        n as i32,
                        n as i32,
                        1.0,
                        a.as_ptr(),
                        n as i32,
                        b.as_ptr(),
                        n as i32,
                        0.0,
                        cc.as_mut_ptr(),
                        n as i32,
                    )
                });
            });
        }
    }
    group.finish();
}

/// `y := α a x + β y`: `n²` steps, the operation every triangular solve and
/// eigenvector update leans on.
fn bench_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv");
    for n in [128usize, 512, 2048] {
        group.throughput(Throughput::Elements((n * n) as u64));
        let (a, x) = (data(n * n), data(n));
        let mut y = vec![0.0f64; n];
        group.bench_with_input(BenchmarkId::new("native", n), &n, |bench, &n| {
            bench.iter(|| native::blas::dgemv(1.0, &a, &x, 0.0, &mut y, n, n));
        });
        #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
        {
            let (a, x) = (a.clone(), x.clone());
            group.bench_with_input(BenchmarkId::new("lib", n), &n, |bench, &n| {
                bench.iter(|| unsafe {
                    lib::cblas_dgemv(
                        lib::COL_MAJOR,
                        lib::NO_TRANS,
                        n as i32,
                        n as i32,
                        1.0,
                        a.as_ptr(),
                        n as i32,
                        x.as_ptr(),
                        1,
                        0.0,
                        y.as_mut_ptr(),
                        1,
                    )
                });
            });
        }
    }
    group.finish();
}

/// The two reductions, at sizes where the loop overhead is the question.
fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot");
    for n in [128usize, 1024, 8192] {
        group.throughput(Throughput::Elements(n as u64));
        let (x, y) = (data(n), data(n));
        group.bench_with_input(BenchmarkId::new("native", n), &n, |bench, _| {
            bench.iter(|| native::blas::ddot(&x, &y));
        });
        #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
        {
            group.bench_with_input(BenchmarkId::new("lib", n), &n, |bench, &n| {
                bench.iter(|| unsafe { lib::cblas_ddot(n as i32, x.as_ptr(), 1, y.as_ptr(), 1) });
            });
        }
    }
    group.finish();

    let mut group = c.benchmark_group("nrm2");
    for n in [128usize, 1024, 8192] {
        group.throughput(Throughput::Elements(n as u64));
        let x = data(n);
        group.bench_with_input(BenchmarkId::new("native", n), &n, |bench, _| {
            bench.iter(|| native::blas::dnrm2(&x));
        });
        #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
        {
            group.bench_with_input(BenchmarkId::new("lib", n), &n, |bench, &n| {
                bench.iter(|| unsafe { lib::cblas_dnrm2(n as i32, x.as_ptr(), 1) });
            });
        }
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    // One thread, to match the native kernels: both libraries default to one per
    // core, which would make the comparison meaningless at these sizes. The
    // crate's own call goes through its shim for either backend, so the bench
    // declares no library symbols of its own.
    russell_lab::set_num_threads(1);
    bench_gemm(c);
    bench_gemv(c);
    bench_reductions(c);
}

criterion_group!(group, benches);
criterion_main!(group);
