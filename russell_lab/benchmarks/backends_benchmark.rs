//! Native Rust kernels against the linked BLAS library: what routing buys, and
//! what it does not.
//!
//! The `open_blas` and `intel_mkl` features link a library, and since the BLAS
//! dispatch landed, a call site that goes through
//! `russell_lab::base::blas_dispatch` reaches that library instead of the Rust
//! kernel. The dispatch is only worth having where the library is actually
//! faster, which is a measurement per routine and not an assumption: the level-1
//! routines are memory-bound and short, so the call into C can cost more than the
//! loop it replaces.
//!
//! The `native` column of every group is the control: it does not change between
//! the two runs, so if it moves, the machine did and the `lib` column means
//! nothing.
//!
//! Run it two ways and read the columns:
//!
//! ```text
//! cargo bench --bench backends_benchmark
//! cargo bench --features open_blas --bench backends_benchmark
//! cargo bench --features intel_mkl --bench backends_benchmark
//! ```
//!
//! The third configuration (`--all-features`, which is MKL) is the one the
//! level-1 decisions recorded in `src/base/blas_dispatch.rs` were made on.
//!
//! Threads are pinned to one before anything is measured, because the native
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
    use russell_lab::Complex64;

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
        pub fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64;
        pub fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32;
        pub fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32);
        pub fn cblas_dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy: i32);
        pub fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32);

        pub fn cblas_zgemv(
            order: i32,
            trans: i32,
            m: i32,
            n: i32,
            alpha: *const Complex64,
            a: *const Complex64,
            lda: i32,
            x: *const Complex64,
            incx: i32,
            beta: *const Complex64,
            y: *mut Complex64,
            incy: i32,
        );
        pub fn cblas_zaxpy(
            n: i32,
            alpha: *const Complex64,
            x: *const Complex64,
            incx: i32,
            y: *mut Complex64,
            incy: i32,
        );
        pub fn cblas_zcopy(n: i32, x: *const Complex64, incx: i32, y: *mut Complex64, incy: i32);
        pub fn cblas_zscal(n: i32, alpha: *const Complex64, x: *mut Complex64, incx: i32);
        pub fn cblas_dznrm2(n: i32, x: *const Complex64, incx: i32) -> f64;
    }
}

/// Deterministic, non-trivial data.
fn data(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 17) as f64) - 8.0).collect()
}

/// Deterministic, non-trivial complex data.
fn cdata(n: usize) -> Vec<russell_lab::Complex64> {
    (0..n)
        .map(|i| russell_lab::cpx!(((i % 17) as f64) - 8.0, ((i % 7) as f64) - 3.0))
        .collect()
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

/// The level-1 routines, at the three sizes the dispatch decision was made on.
///
/// Each pair is the same operation on the same buffers: the crate's own kernel in
/// `native`, and the linked library's CBLAS entry point in `lib`. These are the
/// numbers `src/base/blas_dispatch.rs` cites when it says which level-1 routines
/// it does *not* send to the library.
fn bench_level1(c: &mut Criterion) {
    let mut group = c.benchmark_group("level1");
    group.sample_size(10);
    for n in [128usize, 1024, 8192] {
        group.throughput(Throughput::Elements(n as u64));
        let (x, y0) = (data(n), data(n));
        let (cx, cy0) = (cdata(n), cdata(n));
        let alpha = russell_lab::cpx!(0.5, -0.25);

        {
            let mut y = y0.clone();
            group.bench_with_input(BenchmarkId::new("daxpy/native", n), &n, |bench, _| {
                bench.iter(|| native::blas::daxpy(0.5, &x, &mut y));
            });
        }
        {
            let mut y = y0.clone();
            group.bench_with_input(BenchmarkId::new("dcopy/native", n), &n, |bench, _| {
                bench.iter(|| native::blas::dcopy(&x, &mut y));
            });
        }
        {
            let mut y = y0.clone();
            group.bench_with_input(BenchmarkId::new("dscal/native", n), &n, |bench, _| {
                bench.iter(|| native::blas::dscal(0.5, &mut y));
            });
        }
        group.bench_with_input(BenchmarkId::new("ddot/native", n), &n, |bench, _| {
            bench.iter(|| native::blas::ddot(&x, &y0));
        });
        group.bench_with_input(BenchmarkId::new("dnrm2/native", n), &n, |bench, _| {
            bench.iter(|| native::blas::dnrm2(&x));
        });
        group.bench_with_input(BenchmarkId::new("dasum/native", n), &n, |bench, _| {
            bench.iter(|| native::blas::dasum(&x));
        });
        group.bench_with_input(BenchmarkId::new("idamax/native", n), &n, |bench, _| {
            bench.iter(|| native::blas::idamax(&x));
        });
        {
            let mut y = cy0.clone();
            group.bench_with_input(BenchmarkId::new("zaxpy/native", n), &n, |bench, _| {
                bench.iter(|| native::blas::zaxpy(alpha, &cx, &mut y));
            });
        }
        {
            let mut y = cy0.clone();
            group.bench_with_input(BenchmarkId::new("zcopy/native", n), &n, |bench, _| {
                bench.iter(|| native::blas::zcopy(&cx, &mut y));
            });
        }
        {
            let mut y = cy0.clone();
            group.bench_with_input(BenchmarkId::new("zscal/native", n), &n, |bench, _| {
                bench.iter(|| native::blas::zscal(alpha, &mut y));
            });
        }
        group.bench_with_input(BenchmarkId::new("dznrm2/native", n), &n, |bench, _| {
            bench.iter(|| native::blas::dznrm2(&cx));
        });

        #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
        {
            let ni = n as i32;
            {
                let mut y = y0.clone();
                group.bench_with_input(BenchmarkId::new("daxpy/lib", n), &n, |bench, _| {
                    bench.iter(|| unsafe { lib::cblas_daxpy(ni, 0.5, x.as_ptr(), 1, y.as_mut_ptr(), 1) });
                });
            }
            {
                let mut y = y0.clone();
                group.bench_with_input(BenchmarkId::new("dcopy/lib", n), &n, |bench, _| {
                    bench.iter(|| unsafe { lib::cblas_dcopy(ni, x.as_ptr(), 1, y.as_mut_ptr(), 1) });
                });
            }
            {
                let mut y = y0.clone();
                group.bench_with_input(BenchmarkId::new("dscal/lib", n), &n, |bench, _| {
                    bench.iter(|| unsafe { lib::cblas_dscal(ni, 0.5, y.as_mut_ptr(), 1) });
                });
            }
            group.bench_with_input(BenchmarkId::new("ddot/lib", n), &n, |bench, _| {
                bench.iter(|| unsafe { lib::cblas_ddot(ni, x.as_ptr(), 1, y0.as_ptr(), 1) });
            });
            group.bench_with_input(BenchmarkId::new("dnrm2/lib", n), &n, |bench, _| {
                bench.iter(|| unsafe { lib::cblas_dnrm2(ni, x.as_ptr(), 1) });
            });
            group.bench_with_input(BenchmarkId::new("dasum/lib", n), &n, |bench, _| {
                bench.iter(|| unsafe { lib::cblas_dasum(ni, x.as_ptr(), 1) });
            });
            group.bench_with_input(BenchmarkId::new("idamax/lib", n), &n, |bench, _| {
                bench.iter(|| unsafe { lib::cblas_idamax(ni, x.as_ptr(), 1) });
            });
            {
                let mut y = cy0.clone();
                group.bench_with_input(BenchmarkId::new("zaxpy/lib", n), &n, |bench, _| {
                    bench.iter(|| unsafe { lib::cblas_zaxpy(ni, &alpha, cx.as_ptr(), 1, y.as_mut_ptr(), 1) });
                });
            }
            {
                let mut y = cy0.clone();
                group.bench_with_input(BenchmarkId::new("zcopy/lib", n), &n, |bench, _| {
                    bench.iter(|| unsafe { lib::cblas_zcopy(ni, cx.as_ptr(), 1, y.as_mut_ptr(), 1) });
                });
            }
            {
                let mut y = cy0.clone();
                group.bench_with_input(BenchmarkId::new("zscal/lib", n), &n, |bench, _| {
                    bench.iter(|| unsafe { lib::cblas_zscal(ni, &alpha, y.as_mut_ptr(), 1) });
                });
            }
            group.bench_with_input(BenchmarkId::new("dznrm2/lib", n), &n, |bench, _| {
                bench.iter(|| unsafe { lib::cblas_dznrm2(ni, cx.as_ptr(), 1) });
            });
        }
    }
    group.finish();
}

/// The complex matrix-vector product: `n²` steps, the operation the complex
/// matrix-vector call sites lean on.
fn bench_zgemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("zgemv");
    group.sample_size(10);
    for n in [128usize, 512, 1024] {
        group.throughput(Throughput::Elements((n * n) as u64));
        let (a, x) = (cdata(n * n), cdata(n));
        let alpha = russell_lab::cpx!(1.0, 0.0);
        let zero = russell_lab::cpx!(0.0, 0.0);
        let mut y = vec![zero; n];
        group.bench_with_input(BenchmarkId::new("native", n), &n, |bench, &n| {
            bench.iter(|| native::blas::zgemv(alpha, &a, &x, zero, &mut y, n, n));
        });
        #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
        {
            group.bench_with_input(BenchmarkId::new("lib", n), &n, |bench, &n| {
                bench.iter(|| unsafe {
                    lib::cblas_zgemv(
                        lib::COL_MAJOR,
                        lib::NO_TRANS,
                        n as i32,
                        n as i32,
                        &alpha,
                        a.as_ptr(),
                        n as i32,
                        x.as_ptr(),
                        1,
                        &zero,
                        y.as_mut_ptr(),
                        1,
                    )
                });
            });
        }
    }
    group.finish();
}

/// The public entry points, run in both configurations: this is what proves a
/// dispatch is live.
///
/// A bench that times `native::blas::daxpy` returns the same number under every
/// feature, so it cannot show whether a call site reached the library; only the
/// function a user calls can. `vec_copy` and `vec_scale` are here beside
/// `vec_update` because those two deliberately do *not* dispatch -- their columns
/// have to stay put while `vec_update`'s nearly halves.
fn bench_public(c: &mut Criterion) {
    use russell_lab::{ComplexMatrix, ComplexVector, Norm, Vector};

    let mut group = c.benchmark_group("public");
    group.sample_size(10);
    for n in [128usize, 1024, 8192] {
        group.throughput(Throughput::Elements(n as u64));
        let mut u = Vector::new(n);
        u.as_mut_data().copy_from_slice(&data(n));
        let mut v = Vector::new(n);
        v.as_mut_data().copy_from_slice(&data(n));

        group.bench_with_input(BenchmarkId::new("vec_copy", n), &n, |bench, _| {
            bench.iter(|| russell_lab::vec_copy(&mut v, &u).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("vec_scale", n), &n, |bench, _| {
            bench.iter(|| russell_lab::vec_scale(&mut v, 0.5));
        });
        group.bench_with_input(BenchmarkId::new("vec_update", n), &n, |bench, _| {
            bench.iter(|| russell_lab::vec_update(&mut v, 0.5, &u).unwrap());
        });
    }
    group.finish();

    // The complex matrix-vector product, the two matrix norms and the symmetric
    // rank-k update, at the sizes the call sites use them. All of them are O(n²)
    // and all of them were on a per-file dispatch before.
    let mut group = c.benchmark_group("public_complex");
    group.sample_size(10);
    for n in [128usize, 512, 1024] {
        group.throughput(Throughput::Elements((n * n) as u64));
        let mut a = ComplexMatrix::new(n, n);
        a.as_mut_data().copy_from_slice(&cdata(n * n));
        let mut u = ComplexVector::new(n);
        u.as_mut_data().copy_from_slice(&cdata(n));
        let mut v = ComplexVector::new(n);
        v.as_mut_data().copy_from_slice(&cdata(n));
        let mut c_mat = ComplexMatrix::new(n, n);

        group.bench_with_input(BenchmarkId::new("complex_mat_vec_mul", n), &n, |bench, _| {
            bench.iter(|| russell_lab::complex_mat_vec_mul(&mut v, russell_lab::cpx!(1.0, 0.0), &a, &u).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("complex_mat_norm_fro", n), &n, |bench, _| {
            bench.iter(|| russell_lab::complex_mat_norm(&a, Norm::Fro));
        });
        group.bench_with_input(BenchmarkId::new("complex_mat_norm_one", n), &n, |bench, _| {
            bench.iter(|| russell_lab::complex_mat_norm(&a, Norm::One));
        });
        group.bench_with_input(BenchmarkId::new("complex_mat_norm_max", n), &n, |bench, _| {
            bench.iter(|| russell_lab::complex_mat_norm(&a, Norm::Max));
        });
        let mut r = russell_lab::Matrix::new(n, n);
        r.as_mut_data().copy_from_slice(&data(n * n));
        group.bench_with_input(BenchmarkId::new("mat_norm_fro", n), &n, |bench, _| {
            bench.iter(|| russell_lab::mat_norm(&r, Norm::Fro));
        });
        group.bench_with_input(BenchmarkId::new("mat_norm_one", n), &n, |bench, _| {
            bench.iter(|| russell_lab::mat_norm(&r, Norm::One));
        });
        group.bench_with_input(BenchmarkId::new("mat_norm_max", n), &n, |bench, _| {
            bench.iter(|| russell_lab::mat_norm(&r, Norm::Max));
        });
        group.bench_with_input(BenchmarkId::new("complex_mat_sym_rank_op", n), &n, |bench, _| {
            bench.iter(|| {
                russell_lab::complex_mat_sym_rank_op(
                    &mut c_mat,
                    &a,
                    russell_lab::cpx!(1.0, 0.0),
                    russell_lab::cpx!(0.0, 0.0),
                    false,
                    false,
                )
                .unwrap()
            });
        });
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
    bench_level1(c);
    bench_zgemv(c);
    bench_public(c);
}

criterion_group!(group, benches);
criterion_main!(group);
