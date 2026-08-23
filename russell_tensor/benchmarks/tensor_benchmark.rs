//! Benchmarks for the `russell_tensor` crate.
//!
//! The `heap` cargo feature selects between the stack-allocated and
//! heap-allocated internal storage at compile time:
//!
//! * without `--features heap` — `Tensor2.vec: [f64; 9]`, `Tensor4.mat: [[f64; 9]; 9]` (stack)
//! * with `--features heap` — `Tensor2.vec: Vector`, `Tensor4.mat: Matrix` (heap)
//!
//! Run the benchmark twice (with and without `--features heap`) to compare the
//! two storage layouts.
//!
//! Each function is also benchmarked in two modes, controlled by the `use_loops` flag:
//!
//! * `unrolled` — `use_loops = false` (the default, production path)
//! * `loops` — `use_loops = true` (loop-based, uses the `get`/`set` accessors)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use russell_tensor::{AuxDeriv2InvariantJ3, AuxDeriv2InvariantLode, Rep, Tensor2, Tensor4};
use russell_tensor::{deriv2_invariant_jj3, deriv2_invariant_lode, t2_qsd_t2, t2_ssd};

/// Fixed symmetric 3×3 matrix used to build the input tensors
const SYMMETRIC: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];

/// Benchmarks `t2_ssd` (self-sum-dyadic)
fn bench_t2_ssd(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("t2_ssd");

    group.bench_with_input(BenchmarkId::new("unrolled", ""), &(), |b, _| {
        let aa = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        dd.use_loops = false;
        b.iter(|| {
            t2_ssd(&mut dd, 1.0, &aa);
            std::hint::black_box(dd.get(0, 0));
        });
    });

    group.bench_with_input(BenchmarkId::new("loops", ""), &(), |b, _| {
        let aa = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        dd.use_loops = true;
        b.iter(|| {
            t2_ssd(&mut dd, 1.0, &aa);
            std::hint::black_box(dd.get(0, 0));
        });
    });

    group.finish();
}

/// Benchmarks `t2_qsd_t2` (quartic-sum-dyadic)
fn bench_t2_qsd_t2(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("t2_qsd_t2");

    group.bench_with_input(BenchmarkId::new("unrolled", ""), &(), |b, _| {
        let aa = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let bb = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        dd.use_loops = false;
        b.iter(|| {
            t2_qsd_t2(&mut dd, 1.0, &aa, &bb);
            std::hint::black_box(dd.get(0, 0));
        });
    });

    group.bench_with_input(BenchmarkId::new("loops", ""), &(), |b, _| {
        let aa = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let bb = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        dd.use_loops = true;
        b.iter(|| {
            t2_qsd_t2(&mut dd, 1.0, &aa, &bb);
            std::hint::black_box(dd.get(0, 0));
        });
    });

    group.finish();
}

/// Benchmarks `deriv2_invariant_jj3` (second derivative of J3)
fn bench_deriv2_invariant_jj3(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("deriv2_invariant_jj3");

    group.bench_with_input(BenchmarkId::new("unrolled", ""), &(), |b, _| {
        let sigma = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut d2 = Tensor4::new(Rep::Symmetric);
        let mut aux = AuxDeriv2InvariantJ3::new();
        d2.use_loops = false;
        b.iter(|| {
            deriv2_invariant_jj3(&mut d2, &mut aux, &sigma);
            std::hint::black_box(d2.get(0, 0));
        });
    });

    group.bench_with_input(BenchmarkId::new("loops", ""), &(), |b, _| {
        let sigma = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut d2 = Tensor4::new(Rep::Symmetric);
        let mut aux = AuxDeriv2InvariantJ3::new();
        d2.use_loops = true;
        b.iter(|| {
            deriv2_invariant_jj3(&mut d2, &mut aux, &sigma);
            std::hint::black_box(d2.get(0, 0));
        });
    });

    group.finish();
}

/// Benchmarks `deriv2_invariant_lode` (second derivative of the Lode invariant)
fn bench_deriv2_invariant_lode(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("deriv2_invariant_lode");

    group.bench_with_input(BenchmarkId::new("unrolled", ""), &(), |b, _| {
        let sigma = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut d2 = Tensor4::new(Rep::Symmetric);
        let mut aux = AuxDeriv2InvariantLode::new();
        d2.use_loops = false;
        b.iter(|| {
            deriv2_invariant_lode(&mut d2, &mut aux, &sigma);
            std::hint::black_box(d2.get(0, 0));
        });
    });

    group.bench_with_input(BenchmarkId::new("loops", ""), &(), |b, _| {
        let sigma = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut d2 = Tensor4::new(Rep::Symmetric);
        let mut aux = AuxDeriv2InvariantLode::new();
        d2.use_loops = true;
        b.iter(|| {
            deriv2_invariant_lode(&mut d2, &mut aux, &sigma);
            std::hint::black_box(d2.get(0, 0));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_t2_ssd,
    bench_t2_qsd_t2,
    bench_deriv2_invariant_jj3,
    bench_deriv2_invariant_lode
);
criterion_main!(benches);
