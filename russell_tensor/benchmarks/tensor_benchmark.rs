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
//! Each function is also benchmarked in two variants:
//!
//! * `unrolled` — the production (manually-unrolled) implementation
//! * `loops` — the loop-based reference implementation from `russell_tensor::z_reference_loop_fns`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use russell_tensor::z_reference_loop_fns::{
    deriv_squared_tensor_loops, deriv2_invariant_jj3_loops, deriv2_invariant_lode_loops, qsd_fn_loops, ssd_fn_loops,
};
use russell_tensor::{AuxDeriv2InvariantLode, Rep, Tensor2, Tensor4};
use russell_tensor::{deriv_squared_tensor, deriv2_invariant_jj3, deriv2_invariant_lode, qsd_fn, ssd_fn};

/// Fixed symmetric 3×3 matrix used to build the input tensors
const SYMMETRIC: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];

/// Fixed general (non-symmetric) 3×3 matrix used to build the input tensors
const GENERAL: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

/// Benchmarks `ssd_fn` (self-sum-dyadic)
fn bench_ssd_fn(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("ssd_fn");

    group.bench_with_input(BenchmarkId::new("unrolled", ""), &(), |b, _| {
        let aa = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        b.iter(|| {
            ssd_fn(&mut dd, 1.0, &aa);
            std::hint::black_box(dd.get(0, 0));
        });
    });

    group.bench_with_input(BenchmarkId::new("loops", ""), &(), |b, _| {
        let aa = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        b.iter(|| {
            ssd_fn_loops(&mut dd, 1.0, &aa);
            std::hint::black_box(dd.get(0, 0));
        });
    });

    group.finish();
}

/// Benchmarks `qsd_fn` (quad-sum-dyadic)
fn bench_qsd_fn(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("qsd_fn");

    group.bench_with_input(BenchmarkId::new("unrolled", ""), &(), |b, _| {
        let aa = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let bb = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        b.iter(|| {
            qsd_fn(&mut dd, 1.0, &aa, &bb);
            std::hint::black_box(dd.get(0, 0));
        });
    });

    group.bench_with_input(BenchmarkId::new("loops", ""), &(), |b, _| {
        let aa = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let bb = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut dd = Tensor4::new(Rep::Symmetric);
        b.iter(|| {
            qsd_fn_loops(&mut dd, 1.0, &aa, &bb);
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
        b.iter(|| {
            deriv2_invariant_jj3(&mut d2, &sigma);
            std::hint::black_box(d2.get(0, 0));
        });
    });

    group.bench_with_input(BenchmarkId::new("loops", ""), &(), |b, _| {
        let sigma = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut d2 = Tensor4::new(Rep::Symmetric);
        b.iter(|| {
            deriv2_invariant_jj3_loops(&mut d2, &sigma);
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
        b.iter(|| {
            deriv2_invariant_lode(&mut d2, &mut aux, &sigma);
            std::hint::black_box(d2.get(0, 0));
        });
    });

    group.bench_with_input(BenchmarkId::new("loops", ""), &(), |b, _| {
        let sigma = Tensor2::from_std_matrix(&SYMMETRIC, Rep::Symmetric).unwrap();
        let mut d2 = Tensor4::new(Rep::Symmetric);
        b.iter(|| {
            deriv2_invariant_lode_loops(&mut d2, &sigma);
            std::hint::black_box(d2.get(0, 0));
        });
    });

    group.finish();
}

/// Benchmarks `deriv_squared_tensor` (derivative of the squared tensor, general)
fn bench_deriv_squared_tensor(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("deriv_squared_tensor");

    group.bench_with_input(BenchmarkId::new("unrolled", ""), &(), |b, _| {
        let aa = Tensor2::from_std_matrix(&GENERAL, Rep::General).unwrap();
        let mut da2_da = Tensor4::new(Rep::General);
        b.iter(|| {
            deriv_squared_tensor(&mut da2_da, &aa);
            std::hint::black_box(da2_da.get(0, 0));
        });
    });

    group.bench_with_input(BenchmarkId::new("loops", ""), &(), |b, _| {
        let aa = Tensor2::from_std_matrix(&GENERAL, Rep::General).unwrap();
        let mut da2_da = Tensor4::new(Rep::General);
        b.iter(|| {
            deriv_squared_tensor_loops(&mut da2_da, &aa);
            std::hint::black_box(da2_da.get(0, 0));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ssd_fn,
    bench_qsd_fn,
    bench_deriv2_invariant_jj3,
    bench_deriv2_invariant_lode,
    bench_deriv_squared_tensor
);
criterion_main!(benches);
