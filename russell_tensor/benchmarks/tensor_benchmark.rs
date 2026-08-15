//! Benchmarks comparing the stack-allocated (`russell_tensor`) and heap-allocated
//! (`russell_tensor_heap`) implementations of selected tensor functions.
//!
//! The two crates expose the same function names with the same signatures, but
//! differ in their internal storage:
//!
//! * `russell_tensor` — `Tensor2.vec: [f64; 9]`, `Tensor4.mat: [[f64; 9]; 9]` (stack)
//! * `russell_tensor_heap` — `Tensor2.vec: Vector`, `Tensor4.mat: Matrix` (heap)
//!
//! Each function is benchmarked in two modes, controlled by the `use_loops` flag:
//!
//! * `unrolled` — `use_loops = false` (the default, production path)
//! * `loops` — `use_loops = true` (loop-based, uses `get`/`set` accessors)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

// stack-allocated (russell_tensor)
use russell_tensor::{
    AuxDeriv2InvariantJ3 as StackAux, Rep as StackRep, Tensor2 as StackTensor2, Tensor4 as StackTensor4,
};
use russell_tensor::{
    deriv2_invariant_jj3 as stack_deriv2_invariant_jj3, t2_qsd_t2 as stack_t2_qsd_t2, t2_ssd as stack_t2_ssd,
};

// heap-allocated (russell_tensor_heap)
use russell_tensor_heap::{
    AuxDeriv2InvariantJ3 as HeapAux, Rep as HeapRep, Tensor2 as HeapTensor2, Tensor4 as HeapTensor4,
};
use russell_tensor_heap::{
    deriv2_invariant_jj3 as heap_deriv2_invariant_jj3, t2_qsd_t2 as heap_t2_qsd_t2, t2_ssd as heap_t2_ssd,
};

/// Fixed symmetric 3×3 matrix used to build the input tensors
const SYMMETRIC: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];

/// Benchmarks `t2_ssd` (self-sum-dyadic) for the stack and heap versions
fn bench_t2_ssd(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("t2_ssd");

    group.bench_with_input(BenchmarkId::new("stack", "unrolled"), &(), |b, _| {
        let aa = StackTensor2::from_std_matrix(&SYMMETRIC, StackRep::Symmetric).unwrap();
        let mut dd = StackTensor4::new(StackRep::Symmetric);
        dd.use_loops = false;
        b.iter(|| {
            stack_t2_ssd(&mut dd, 1.0, &aa);
            std::hint::black_box(dd.matrix());
        });
    });

    group.bench_with_input(BenchmarkId::new("stack", "loops"), &(), |b, _| {
        let aa = StackTensor2::from_std_matrix(&SYMMETRIC, StackRep::Symmetric).unwrap();
        let mut dd = StackTensor4::new(StackRep::Symmetric);
        dd.use_loops = true;
        b.iter(|| {
            stack_t2_ssd(&mut dd, 1.0, &aa);
            std::hint::black_box(dd.matrix());
        });
    });

    group.bench_with_input(BenchmarkId::new("heap", "unrolled"), &(), |b, _| {
        let aa = HeapTensor2::from_matrix(&SYMMETRIC, HeapRep::Symmetric).unwrap();
        let mut dd = HeapTensor4::new(HeapRep::Symmetric);
        dd.use_loops = false;
        b.iter(|| {
            heap_t2_ssd(&mut dd, 1.0, &aa);
            std::hint::black_box(dd.matrix());
        });
    });

    group.bench_with_input(BenchmarkId::new("heap", "loops"), &(), |b, _| {
        let aa = HeapTensor2::from_matrix(&SYMMETRIC, HeapRep::Symmetric).unwrap();
        let mut dd = HeapTensor4::new(HeapRep::Symmetric);
        dd.use_loops = true;
        b.iter(|| {
            heap_t2_ssd(&mut dd, 1.0, &aa);
            std::hint::black_box(dd.matrix());
        });
    });

    group.finish();
}

/// Benchmarks `t2_qsd_t2` (quartic-sum-dyadic) for the stack and heap versions
fn bench_t2_qsd_t2(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("t2_qsd_t2");

    group.bench_with_input(BenchmarkId::new("stack", "unrolled"), &(), |b, _| {
        let aa = StackTensor2::from_std_matrix(&SYMMETRIC, StackRep::Symmetric).unwrap();
        let bb = StackTensor2::from_std_matrix(&SYMMETRIC, StackRep::Symmetric).unwrap();
        let mut dd = StackTensor4::new(StackRep::Symmetric);
        dd.use_loops = false;
        b.iter(|| {
            stack_t2_qsd_t2(&mut dd, 1.0, &aa, &bb);
            std::hint::black_box(dd.matrix());
        });
    });

    group.bench_with_input(BenchmarkId::new("stack", "loops"), &(), |b, _| {
        let aa = StackTensor2::from_std_matrix(&SYMMETRIC, StackRep::Symmetric).unwrap();
        let bb = StackTensor2::from_std_matrix(&SYMMETRIC, StackRep::Symmetric).unwrap();
        let mut dd = StackTensor4::new(StackRep::Symmetric);
        dd.use_loops = true;
        b.iter(|| {
            stack_t2_qsd_t2(&mut dd, 1.0, &aa, &bb);
            std::hint::black_box(dd.matrix());
        });
    });

    group.bench_with_input(BenchmarkId::new("heap", "unrolled"), &(), |b, _| {
        let aa = HeapTensor2::from_matrix(&SYMMETRIC, HeapRep::Symmetric).unwrap();
        let bb = HeapTensor2::from_matrix(&SYMMETRIC, HeapRep::Symmetric).unwrap();
        let mut dd = HeapTensor4::new(HeapRep::Symmetric);
        dd.use_loops = false;
        b.iter(|| {
            heap_t2_qsd_t2(&mut dd, 1.0, &aa, &bb);
            std::hint::black_box(dd.matrix());
        });
    });

    group.bench_with_input(BenchmarkId::new("heap", "loops"), &(), |b, _| {
        let aa = HeapTensor2::from_matrix(&SYMMETRIC, HeapRep::Symmetric).unwrap();
        let bb = HeapTensor2::from_matrix(&SYMMETRIC, HeapRep::Symmetric).unwrap();
        let mut dd = HeapTensor4::new(HeapRep::Symmetric);
        dd.use_loops = true;
        b.iter(|| {
            heap_t2_qsd_t2(&mut dd, 1.0, &aa, &bb);
            std::hint::black_box(dd.matrix());
        });
    });

    group.finish();
}

/// Benchmarks `deriv2_invariant_jj3` (second derivative of J3) for the stack and heap versions
fn bench_deriv2_invariant_jj3(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("deriv2_invariant_jj3");

    group.bench_with_input(BenchmarkId::new("stack", "unrolled"), &(), |b, _| {
        let sigma = StackTensor2::from_std_matrix(&SYMMETRIC, StackRep::Symmetric).unwrap();
        let mut d2 = StackTensor4::new(StackRep::Symmetric);
        let mut aux = StackAux::new();
        d2.use_loops = false;
        b.iter(|| {
            stack_deriv2_invariant_jj3(&mut d2, &mut aux, &sigma);
            std::hint::black_box(d2.matrix());
        });
    });

    group.bench_with_input(BenchmarkId::new("stack", "loops"), &(), |b, _| {
        let sigma = StackTensor2::from_std_matrix(&SYMMETRIC, StackRep::Symmetric).unwrap();
        let mut d2 = StackTensor4::new(StackRep::Symmetric);
        let mut aux = StackAux::new();
        d2.use_loops = true;
        b.iter(|| {
            stack_deriv2_invariant_jj3(&mut d2, &mut aux, &sigma);
            std::hint::black_box(d2.matrix());
        });
    });

    group.bench_with_input(BenchmarkId::new("heap", "unrolled"), &(), |b, _| {
        let sigma = HeapTensor2::from_matrix(&SYMMETRIC, HeapRep::Symmetric).unwrap();
        let mut d2 = HeapTensor4::new(HeapRep::Symmetric);
        let mut aux = HeapAux::new();
        d2.use_loops = false;
        b.iter(|| {
            heap_deriv2_invariant_jj3(&mut d2, &mut aux, &sigma);
            std::hint::black_box(d2.matrix());
        });
    });

    group.bench_with_input(BenchmarkId::new("heap", "loops"), &(), |b, _| {
        let sigma = HeapTensor2::from_matrix(&SYMMETRIC, HeapRep::Symmetric).unwrap();
        let mut d2 = HeapTensor4::new(HeapRep::Symmetric);
        let mut aux = HeapAux::new();
        d2.use_loops = true;
        b.iter(|| {
            heap_deriv2_invariant_jj3(&mut d2, &mut aux, &sigma);
            std::hint::black_box(d2.matrix());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_t2_ssd, bench_t2_qsd_t2, bench_deriv2_invariant_jj3);
criterion_main!(benches);
